import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from typing import Union
from typing_extensions import Literal

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.wan_video_dit import WanModel, normalize_action_injection_method
from ..models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from .wan_video import model_fn_wan_video


def normalize_action_normalization_mode(mode: str = "standard"):
    mode = "standard" if mode is None else str(mode).strip().lower().replace("-", "_")
    supported_modes = ("standard", "scale_only")
    if mode not in supported_modes:
        raise ValueError(f"`action_normalization_mode` must be one of {supported_modes}, got {mode!r}.")
    return mode


class WanWorldModelActionEmbedder(torch.nn.Module):
    """
    Embed per-frame robot actions into the conditioning space required by the selected injection method.
    """

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, embed_dim, bias=True),
        )
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, action: torch.Tensor):
        return self.mlp(action)


class WanWorldModelPipeline(BasePipeline):
    model_id = "Wan-AI/Wan2.2-TI2V-5B"

    def __init__(
        self,
        device=get_device_type(),
        torch_dtype=torch.bfloat16,
        action_dim: int = None,
        action_embedder_hidden_dim: int = None,
        action_injection_method: str = "context",
        action_metadata_path: str = None,
        action_metadata_key: str = "robot_statistics",
        action_normalization_eps: float = 1e-6,
        action_normalization_mode: str = "standard",
        use_text_condition: bool = True,
        text_context_length: int = 512,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        if int(text_context_length) <= 0:
            raise ValueError("text_context_length must be a positive integer.")
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.use_text_condition = bool(use_text_condition)
        self.text_context_length = int(text_context_length)
        self.action_dim = action_dim
        self.action_embedder_hidden_dim = action_embedder_hidden_dim
        self.action_injection_method = normalize_action_injection_method(action_injection_method)
        if not self.use_text_condition and action_dim is not None and self.action_injection_method == "context":
            raise ValueError("`action_injection_method=context` requires language/context attention. Use additive, cross_attention, adaln, or film when language conditioning is disabled.")
        self.action_metadata_path = action_metadata_path
        self.action_metadata_key = action_metadata_key
        self.action_normalization_eps = action_normalization_eps
        self.action_normalization_mode = normalize_action_normalization_mode(action_normalization_mode)
        self.action_normalization_stats = self._load_action_normalization_stats(
            action_metadata_path,
            metadata_key=action_metadata_key,
        )
        self.action_embedder: WanWorldModelActionEmbedder = None
        self.in_iteration_models = ("dit",)
        # 这里只保留 Wan2.2-TI2V-5B 推理必需的前处理单元，不接入音频或其它控制分支。
        self.units = [
            WanWorldModelUnit_ShapeChecker(),
            WanWorldModelUnit_NoiseInitializer(),
            WanWorldModelUnit_InputVideoEmbedder(),
            WanWorldModelUnit_PromptEmbedder(),
            WanWorldModelUnit_ActionEmbedder(),
            WanWorldModelUnit_InputImageEmbedderFused(),
        ]
        self.model_fn = model_fn_wan_video
        self.compilable_models = ["dit"]

    @staticmethod
    def _load_action_normalization_stats(metadata_path: str = None, metadata_key: str = "robot_statistics"):
        if metadata_path is None:
            return None
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Action metadata file does not exist: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError(f"Action metadata must be a JSON object: {metadata_path}")
        if metadata_key in metadata:
            metadata = metadata[metadata_key]
        elif "arms" not in metadata:
            raise KeyError(f"Action metadata `{metadata_path}` does not contain `{metadata_key}` or `arms`.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Action metadata `{metadata_key}` must be a JSON object: {metadata_path}")

        arms = metadata.get("arms", {})
        mean = []
        std = []
        for arm in ("left", "right"):
            try:
                action_stats = arms[arm]["action"]
            except KeyError as error:
                raise KeyError(f"Action metadata is missing `{arm}.action` statistics.") from error
            mean.extend(action_stats["mean"])
            std.extend(action_stats["std"])

        if len(mean) != len(std):
            raise ValueError(f"Action metadata mean/std dimensions do not match: {len(mean)} vs {len(std)}.")
        return {
            "path": metadata_path,
            "order": ("left.action", "right.action"),
            "mean": tuple(float(value) for value in mean),
            "std": tuple(float(value) for value in std),
        }

    def build_action_embedder(self, action_dim: int = None, hidden_dim: int = None, action_injection_method: str = None):
        self.action_dim = action_dim
        if action_injection_method is not None:
            self.action_injection_method = normalize_action_injection_method(action_injection_method)
        if action_dim is None:
            self.action_embedder_hidden_dim = hidden_dim
            self.action_embedder = None
            return
        if self.action_normalization_stats is not None and len(self.action_normalization_stats["std"]) != action_dim:
            raise ValueError(
                f"Action metadata dimension is {len(self.action_normalization_stats['std'])}, "
                f"but pipeline action_dim is {action_dim}."
            )
        if self.dit is None:
            raise ValueError("DiT must be loaded before building the action embedder.")
        self.dit.configure_action_injection(self.action_injection_method)
        embed_dim = self.dit.action_embedding_dim(self.action_injection_method)
        if embed_dim is None:
            self.action_embedder_hidden_dim = hidden_dim
            self.action_embedder = None
            return
        hidden_dim = hidden_dim or self.dit.dim
        self.action_embedder_hidden_dim = hidden_dim
        self.action_embedder = WanWorldModelActionEmbedder(
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

    @classmethod
    def default_model_configs(cls, **kwargs) -> list[ModelConfig]:
        return [
            ModelConfig(model_id=cls.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **kwargs),
            ModelConfig(model_id=cls.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **kwargs),
            ModelConfig(model_id=cls.model_id, origin_file_pattern="Wan2.2_VAE.pth", **kwargs),
        ]

    @staticmethod
    def _is_text_encoder_config(model_config: ModelConfig):
        values = []
        for value in (model_config.path, model_config.model_id, model_config.origin_file_pattern):
            if isinstance(value, (list, tuple)):
                values.extend(value)
            elif value is not None:
                values.append(value)
        text_encoder_markers = ("umt5", "models_t5", "text_encoder")
        return any(any(marker in str(value).lower() for marker in text_encoder_markers) for value in values)

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = None,
        tokenizer_config: ModelConfig = None,
        redirect_common_files: bool = True,
        vram_limit: float = None,
        action_dim: int = None,
        action_embedder_hidden_dim: int = None,
        action_injection_method: str = "context",
        action_metadata_path: str = None,
        action_metadata_key: str = "robot_statistics",
        action_normalization_eps: float = 1e-6,
        action_normalization_mode: str = "standard",
        use_text_condition: bool = True,
        text_context_length: int = 512,
    ):
        if model_configs is None:
            model_configs = cls.default_model_configs()
        if use_text_condition and tokenizer_config is None:
            tokenizer_config = ModelConfig(model_id=cls.model_id, origin_file_pattern="google/umt5-xxl/")

        # 公共权重重定向到 DiffSynth 已转换 safetensors，避免重复下载原始 pth 文件。
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        if not use_text_condition:
            model_configs = [model_config for model_config in model_configs if not cls._is_text_encoder_config(model_config)]

        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            action_dim=action_dim,
            action_embedder_hidden_dim=action_embedder_hidden_dim,
            action_injection_method=action_injection_method,
            action_metadata_path=action_metadata_path,
            action_metadata_key=action_metadata_key,
            action_normalization_eps=action_normalization_eps,
            action_normalization_mode=action_normalization_mode,
            use_text_condition=use_text_condition,
            text_context_length=text_context_length,
        )
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Wan World Model 只需要文本编码器、DiT 和 VAE；audio 相关模型与 processor 不加载。
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder") if use_text_condition else None
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.build_action_embedder(
            action_dim=action_dim,
            hidden_dim=action_embedder_hidden_dim,
            action_injection_method=action_injection_method,
        )

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if use_text_condition:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=text_context_length, clean="whitespace")

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        input_image: Image.Image = None,
        seed: int = None,
        rand_device: str = "cpu",
        height: int = 704,
        width: int = 1248,
        num_frames: int = 121,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        action: Union[torch.Tensor, list, tuple, dict] = None,
        progress_bar_cmd=tqdm,
        output_type: Literal["quantized", "floatpoint"] = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "action": action,
            "disable_context_attention": not self.use_text_condition,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 扩散迭代阶段只调用 DiT；负向提示词仅在 CFG 生效时计算。
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        self.load_models_to_device(["vae"])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanWorldModelUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanWorldModelPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames, verbose=False)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanWorldModelUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: WanWorldModelPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanWorldModelUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanWorldModelPipeline, input_video, noise, height, width, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = [image.resize((width, height)) for image in input_video]
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents}


class WanWorldModelUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanWorldModelPipeline, prompt) -> dict:
        if not pipe.use_text_condition:
            if pipe.dit is None:
                raise ValueError("DiT must be loaded before creating the empty text context placeholder.")
            batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
            prompt_emb = torch.empty(
                (batch_size, 0, pipe.dit.text_dim),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            return {"context": prompt_emb}

        pipe.load_models_to_device(self.onload_model_names)
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for seq_len in seq_lens:
            prompt_emb[:, seq_len:] = 0
        return {"context": prompt_emb}


class WanWorldModelUnit_ActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("action", "num_frames"),
            input_params_posi={"context": "context"},
            input_params_nega={"context": "context"},
            output_params=("context", "action_emb"),
            onload_model_names=("action_embedder",),
        )

    @staticmethod
    def _flatten_robot_action(robot):
        pieces = []
        for arm in ("left", "right"):
            arm_action = robot.get(arm, {}).get("action", {})
            for key in ("arm_joint", "gripper"):
                value = arm_action.get(key)
                if value is None:
                    continue
                value = torch.as_tensor(value)
                if value.ndim == 1:
                    value = value.unsqueeze(-1)
                pieces.append(value)
        if len(pieces) == 0:
            raise ValueError("Robot action dict does not contain any action tensors.")
        return torch.cat(pieces, dim=-1)

    @classmethod
    def _prepare_action(cls, action):
        if isinstance(action, dict):
            action = cls._flatten_robot_action(action)
        elif not torch.is_tensor(action):
            action = torch.as_tensor(action)

        if action.ndim == 1:
            action = action.reshape(1, -1, 1)
        elif action.ndim == 2:
            action = action.unsqueeze(0)
        elif action.ndim != 3:
            raise ValueError(f"`action` must have shape [T, D] or [B, T, D], got {tuple(action.shape)}.")
        return action

    @staticmethod
    def _normalize_action(pipe: WanWorldModelPipeline, action: torch.Tensor):
        stats = getattr(pipe, "action_normalization_stats", None)
        if stats is None:
            return action
        if action.shape[-1] != len(stats["std"]):
            raise ValueError(
                f"Action metadata dimension is {len(stats['std'])}, but `action` last dimension is {action.shape[-1]}."
            )

        normalize_dtype = (
            action.dtype
            if torch.is_floating_point(action) and action.dtype not in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        action = action.to(dtype=normalize_dtype)
        mean = torch.as_tensor(stats["mean"], dtype=normalize_dtype, device=action.device).view(1, 1, -1)
        std = torch.as_tensor(stats["std"], dtype=normalize_dtype, device=action.device).view(1, 1, -1)
        eps = float(getattr(pipe, "action_normalization_eps", 1e-6))
        std = torch.clamp(std, min=eps)
        mode = normalize_action_normalization_mode(getattr(pipe, "action_normalization_mode", "standard"))
        if mode == "scale_only":
            return action / std
        return (action - mean) / std

    @staticmethod
    def _sample_action_at_positions(action: torch.Tensor, positions: torch.Tensor):
        if positions.numel() == 0:
            return action[:, :0]
        original_dtype = action.dtype
        action = action.float()
        positions = positions.to(device=action.device, dtype=torch.float32)
        positions = positions.clamp(0, max(action.shape[1] - 1, 0))
        low_indices = torch.floor(positions).long()
        high_indices = torch.clamp(low_indices + 1, max=action.shape[1] - 1)
        weight_high = (positions - low_indices.to(dtype=positions.dtype)).view(1, -1, 1)
        action_low = action.index_select(1, low_indices)
        action_high = action.index_select(1, high_indices)
        action = action_low * (1.0 - weight_high) + action_high * weight_high
        return action.to(dtype=original_dtype)

    @classmethod
    def _resample_previous_action_for_latents(cls, action: torch.Tensor, frame_count: int):
        if action.shape[1] <= 0:
            raise ValueError("`action` must contain at least one frame.")
        if frame_count <= 1 or action.shape[1] <= 1:
            return action[:, :0]

        # Latent frames align to full video-frame positions; each generated frame uses the previous raw action.
        video_positions = torch.linspace(
            0,
            action.shape[1] - 1,
            steps=frame_count,
            device=action.device,
            dtype=torch.float32,
        )
        previous_action_positions = video_positions[1:] - 1.0
        return cls._sample_action_at_positions(action[:, :-1], previous_action_positions)

    @staticmethod
    def _latent_frame_count(num_frames: int):
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
        return (num_frames - 1) // 4 + 1

    @classmethod
    def _embed_latent_aligned_action(cls, pipe: WanWorldModelPipeline, action: torch.Tensor, frame_count: int):
        if frame_count <= 0:
            raise ValueError(f"`frame_count` must be positive, got {frame_count}.")

        if frame_count == 1 or action.shape[1] <= 1:
            reference_action = action[:, :1].to(dtype=pipe.torch_dtype)
            reference_emb = pipe.action_embedder(reference_action)
            return reference_emb.new_zeros(action.shape[0], frame_count, reference_emb.shape[-1])

        with torch.no_grad():
            action_tail = cls._resample_previous_action_for_latents(action, frame_count)
            action_tail = action_tail.to(dtype=pipe.torch_dtype)
        action_tail_emb = pipe.action_embedder(action_tail)
        return torch.cat(
            [
                action_tail_emb.new_zeros(action_tail_emb.shape[0], 1, action_tail_emb.shape[-1]),
                action_tail_emb,
            ],
            dim=1,
        )

    def process(self, pipe: WanWorldModelPipeline, action, num_frames, context) -> dict:
        if action is None:
            return {}
        injection_method = normalize_action_injection_method(pipe.action_injection_method)
        if injection_method == "none":
            return {}
        if pipe.action_embedder is None:
            raise ValueError("Action conditioning requires `action_dim` when building WanWorldModelPipeline.")

        action = self._prepare_action(action).to(device=pipe.device)
        if action.shape[-1] != pipe.action_dim:
            raise ValueError(f"`action` last dimension is {action.shape[-1]}, but pipeline action_dim is {pipe.action_dim}.")
        action = self._normalize_action(pipe, action)

        if action.shape[0] != context.shape[0]:
            if action.shape[0] == 1:
                action = action.expand(context.shape[0], -1, -1)
            elif context.shape[0] == 1:
                context = context.expand(action.shape[0], -1, -1)
            else:
                raise ValueError(
                    f"Action batch size {action.shape[0]} does not match context batch size {context.shape[0]}."
                )

        pipe.action_embedder.to(dtype=pipe.torch_dtype, device=pipe.device)
        if injection_method != "context":
            frame_count = self._latent_frame_count(num_frames)
            action_emb = self._embed_latent_aligned_action(pipe, action, frame_count)
            return {
                "context": context,
                "action_emb": action_emb,
            }

        action = action.to(dtype=pipe.torch_dtype)
        action_emb = pipe.action_embedder(action)
        action_emb = torch.cat(
            [
                torch.zeros_like(action_emb[:, :1]),
                action_emb[:, :-1],
            ],
            dim=1,
        ) # 第一帧无动作；后续帧使用上一帧原始 action 作为条件。

        context = context.clone()
        action_emb = action_emb.to(dtype=context.dtype, device=context.device)
        context_length = context.shape[1]
        for batch_id in range(context.shape[0]):
            prompt_length = int(torch.any(context[batch_id] != 0, dim=-1).sum().item())
            start = prompt_length if prompt_length < context_length else max(0, context_length - action_emb.shape[1])
            token_count = min(action_emb.shape[1], context_length - start)
            if token_count > 0:
                context[batch_id, start:start + token_count] = (
                    context[batch_id, start:start + token_count] + action_emb[batch_id, :token_count]
                )
        return {"context": context}


class WanWorldModelUnit_InputImageEmbedderFused(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanWorldModelPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        first_frame_latents = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0:1] = first_frame_latents
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first_frame_latents,
        }
