import argparse
import json
import os
import warnings

import accelerate
import torch

from diffsynth.core import UnifiedDataset, WorldModelDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import *
from diffsynth.models.wan_video_dit import normalize_action_injection_method
from diffsynth.pipelines.wan_world_model import ModelConfig, WanWorldModelPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def split_csv(value):
    if value is None:
        return None
    values = [item.strip() for item in value.split(",")]
    return tuple(item for item in values if item)


def contains_cached_data_files(path):
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        subpath = os.path.join(path, name)
        if os.path.isfile(subpath) and name.endswith(".pth"):
            return True
        if os.path.isdir(subpath):
            try:
                if any(file_name.endswith(".pth") for file_name in os.listdir(subpath)):
                    return True
            except OSError:
                continue
    return False


def default_action_metadata_path(dataset_base_path, metadata_key="robot_statistics"):
    if dataset_base_path is None:
        return None
    path = os.path.join(dataset_base_path, "metadata.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(metadata, dict) and (metadata_key in metadata or "arms" in metadata):
        return path
    return None


class WorldModelTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_camera="head_camera", frame_processor=None):
        self.dataset = dataset
        self.video_camera = video_camera
        self.frame_processor = frame_processor
        self.load_from_cache = False

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def robot_action_to_tensor(robot):
        pieces = []
        for arm in ("left", "right"):
            arm_action = robot.get(arm, {}).get("action", {})
            for key in ("arm_joint", "gripper"):
                value = arm_action.get(key)
                if value is None:
                    continue
                if value.ndim == 1:
                    value = value.unsqueeze(-1)
                pieces.append(value)
        if len(pieces) == 0:
            return None
        return torch.cat(pieces, dim=-1)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.video_camera not in data["cameras"]:
            raise KeyError(
                f"Camera `{self.video_camera}` is not available in sample "
                f"{data['task']}/{data['episode']}. Available cameras: {list(data['cameras'])}"
            )
        video = data["cameras"][self.video_camera]["rgb"]
        if self.frame_processor is not None:
            video = [self.frame_processor(frame) for frame in video]
        data["video"] = video
        data["input_image"] = video[0]
        action = self.robot_action_to_tensor(data["robot"])
        if action is not None:
            data["action"] = action
        return data


def build_unified_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            random_start=args.video_random_start,
        ),
    )


def build_world_model_dataset(args):
    cameras = split_csv(args.world_model_cameras) or (args.world_model_video_camera,)
    if args.world_model_video_camera not in cameras:
        cameras = (args.world_model_video_camera,) + cameras
    dataset = WorldModelDataset(
        root=args.dataset_base_path,
        tasks=split_csv(args.world_model_tasks),
        cameras=cameras,
        num_frames=args.num_frames,
        stride=args.world_model_stride,
        include_depth=args.world_model_include_depth,
        include_camera_params=args.world_model_include_camera_params,
        include_failed=args.world_model_include_failed,
        repeat=args.dataset_repeat,
    )
    frame_processor = ImageCropAndResize(
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    return WorldModelTrainingDataset(
        dataset,
        video_camera=args.world_model_video_camera,
        frame_processor=frame_processor,
    )


def build_dataset(args):
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        has_unified_data = args.dataset_metadata_path is not None or contains_cached_data_files(args.dataset_base_path)
        dataset_type = "unified" if has_unified_data else "world_model"

    if dataset_type == "world_model":
        return build_world_model_dataset(args)
    if dataset_type == "unified":
        return build_unified_dataset(args)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


class WanWorldModelTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        action_dim=None,
        action_embedder_hidden_dim=None,
        action_injection_method="context",
        action_metadata_path=None,
        action_metadata_key="robot_statistics",
        action_normalization_eps=1e-6,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is disabled. The training framework will enable it to reduce OOM risk.")
            use_gradient_checkpointing = True

        action_injection_method = normalize_action_injection_method(action_injection_method)
        action_enabled = action_dim is not None and action_injection_method != "none"
        if action_enabled:
            if trainable_models is None:
                trainable_models = "dit,action_embedder"
            else:
                trainable_model_names = [name.strip() for name in trainable_models.split(",") if name.strip()]
                if "action_embedder" not in trainable_model_names:
                    trainable_model_names.append("action_embedder")
                    trainable_models = ",".join(trainable_model_names)

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        if len(model_configs) == 0:
            model_configs = None
        tokenizer_config = (
            ModelConfig(model_id=WanWorldModelPipeline.model_id, origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(path=tokenizer_path)
        )
        self.pipe = WanWorldModelPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            action_dim=action_dim,
            action_embedder_hidden_dim=action_embedder_hidden_dim,
            action_injection_method=action_injection_method,
            action_metadata_path=action_metadata_path,
            action_metadata_key=action_metadata_key,
            action_normalization_eps=action_normalization_eps,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = [item.strip() for item in extra_inputs.split(",") if item.strip()] if extra_inputs is not None else []
        if action_enabled and "action" not in self.extra_inputs:
            self.extra_inputs.append("action")
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                # Wan2.2-TI2V-5B 的 I2V 条件直接取训练视频首帧，不要求 metadata 单独提供 image。
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "action" and "action" not in data and "robot" in data:
                inputs_shared["action"] = WorldModelTrainingDataset.robot_action_to_tensor(data["robot"])
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_world_model_parser():
    parser = argparse.ArgumentParser(description="WanWorldModelPipeline training script.")
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_gradient_config(parser)
    parser = add_template_model_config(parser)
    parser = add_offload_training_config(parser)
    parser = add_logger_config(parser)
    parser = add_video_size_config(parser)
    parser.set_defaults(data_file_keys="video", extra_inputs="input_image")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="world_model",
        choices=("auto", "unified", "world_model"),
        help="Dataset backend. Defaults to WorldModelDataset. Use `unified` for metadata/cached data, or `auto` to detect it.",
    )
    parser.add_argument("--world_model_tasks", type=str, default=None, help="Comma-separated task folders to load from WorldModelDataset.")
    parser.add_argument("--world_model_cameras", type=str, default=None, help="Comma-separated cameras to load. Defaults to --world_model_video_camera.")
    parser.add_argument("--world_model_video_camera", type=str, default="head_camera", help="Camera RGB stream used as training video.")
    parser.add_argument("--world_model_stride", type=int, default=None, help="Stride between fixed-length world-model windows. Defaults to num_frames.")
    parser.add_argument("--world_model_include_depth", default=False, action="store_true", help="Load depth arrays from WorldModelDataset.")
    parser.add_argument("--world_model_include_camera_params", default=False, action="store_true", help="Load camera intrinsics/extrinsics from WorldModelDataset.")
    parser.add_argument("--world_model_include_failed", default=False, action="store_true", help="Include failed episodes from WorldModelDataset.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--action_dim", type=int, default=14, help="Robot action vector dimension. Enable action conditioning when set.")
    parser.add_argument(
        "--action_metadata_path",
        type=str,
        default=None,
        help="Path to robot action normalization metadata. Defaults to <dataset_base_path>/metadata.json when it contains robot_statistics.",
    )
    parser.add_argument("--action_metadata_key", type=str, default="robot_statistics", help="Top-level metadata key for robot statistics.")
    parser.add_argument("--action_normalization_eps", type=float, default=1e-6, help="Minimum std value used for action normalization.")
    parser.add_argument("--action_embedder_hidden_dim", type=int, default=None, help="Hidden dimension of the action embedder MLP. Defaults to Wan DiT hidden dimension.")
    parser.add_argument(
        "--action_injection_method",
        type=str,
        default="additive",
        choices=("none", "context", "additive", "cross_attention", "cross-attention", "adaln"),
        help="Action conditioning method: `context` keeps the previous context-token scheme; the others inject action in Wan DiT blocks.",
    )
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Maximum timestep boundary ratio.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Minimum timestep boundary ratio.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = wan_world_model_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args)
    action_enabled = args.action_dim is not None and normalize_action_injection_method(args.action_injection_method) != "none"
    action_metadata_path = None
    if action_enabled:
        action_metadata_path = args.action_metadata_path or default_action_metadata_path(
            args.dataset_base_path,
            metadata_key=args.action_metadata_key,
        )
    model = WanWorldModelTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_dim=args.action_dim,
        action_embedder_hidden_dim=args.action_embedder_hidden_dim,
        action_injection_method=args.action_injection_method,
        action_metadata_path=action_metadata_path,
        action_metadata_key=args.action_metadata_key,
        action_normalization_eps=args.action_normalization_eps,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
