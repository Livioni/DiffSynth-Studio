import argparse
import json
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import accelerate
import torch

from diffsynth.core import ModelConfig, WorldModelDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import *
from diffsynth.pipelines.wan_world_action_model import (
    FlowMatchWanWorldActionLoss,
    WanWorldActionModelPipeline,
)

try:
    from examples.wanworldactionmodel.action_space_utils import (
        relative_eef6d_action_from_state_sequence,
        robot_state_to_eef6d,
    )
except ModuleNotFoundError:
    from action_space_utils import relative_eef6d_action_from_state_sequence, robot_state_to_eef6d


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def split_csv(value):
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",")]
    return tuple(item for item in values if item)


def validate_probability(value, name):
    value = float(value or 0.0)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def load_tau_statistics(path, eps=1e-6):
    if path is None or path == "":
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Statistics file does not exist: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    stats = {}
    for key in ("action", "state"):
        if key not in data:
            raise KeyError(f"Statistics file is missing key `{key}`.")
        mean = torch.as_tensor(data[key]["mean"], dtype=torch.float32).view(1, -1)
        std = torch.as_tensor(data[key]["std"], dtype=torch.float32).view(1, -1).clamp_min(float(eps))
        if mean.shape[-1] != 20 or std.shape[-1] != 20:
            raise ValueError(f"`{key}` statistics must have dim 20, got mean={tuple(mean.shape)}, std={tuple(std.shape)}.")
        stats[key] = {"mean": mean, "std": std}
    return stats


def normalize_with_stats(value, stats, key):
    if stats is None:
        return value
    return (value - stats[key]["mean"]) / stats[key]["std"]


class RobotwinWorldActionModelTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        video_camera="head_camera",
        multiview_cameras=None,
        frame_processor=None,
        num_video_frames=9,
        action_horizon=33,
        statistics_path=None,
        statistics_eps=1e-6,
        quat_order="xyzw",
    ):
        self.dataset = dataset
        video_cameras = split_csv(video_camera) or (video_camera,)
        self.video_camera = video_cameras[0]
        self.multiview_cameras = tuple(multiview_cameras or (video_cameras if len(video_cameras) > 1 else ()))
        self.frame_processor = frame_processor
        self.num_video_frames = int(num_video_frames)
        self.action_horizon = int(action_horizon)
        self.statistics_path = statistics_path
        self.statistics = load_tau_statistics(statistics_path, eps=statistics_eps)
        self.quat_order = quat_order
        self._sample_weights = getattr(dataset, "sample_weights", None)
        self.load_from_cache = False
        if self.num_video_frames <= 0:
            raise ValueError("num_video_frames must be positive.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

    def __len__(self):
        return len(self.dataset)

    @property
    def sample_weights(self):
        return self._sample_weights

    def _process_video(self, video):
        video = list(video)[: self.num_video_frames]
        if self.frame_processor is None:
            return video
        return [self.frame_processor(frame) for frame in video]

    def _sample_weight(self, index):
        if self._sample_weights is None or len(self._sample_weights) == 0:
            return None
        return float(self._sample_weights[index % len(self._sample_weights)])

    def sample_statistics(self):
        episodes = getattr(self.dataset, "episodes", None)
        windows = getattr(self.dataset, "windows", None)
        if episodes is None or windows is None:
            return None
        task_rows = defaultdict(lambda: {"episodes": 0, "windows": 0})
        for episode in episodes:
            task_rows[episode.task]["episodes"] += 1
        for window in windows:
            task_rows[episodes[window.episode_id].task]["windows"] += 1
        return {
            "roots": getattr(self.dataset, "roots", ()),
            "cameras": getattr(self.dataset, "cameras", ()),
            "video_camera": self.video_camera,
            "multiview_cameras": self.multiview_cameras,
            "num_video_frames": self.num_video_frames,
            "action_horizon": self.action_horizon,
            "statistics_path": self.statistics_path,
            "episode_count": len(episodes),
            "window_count": len(windows),
            "task_rows": dict(sorted(task_rows.items())),
        }

    def __getitem__(self, index):
        data = self.dataset[index]
        sample_weight = self._sample_weight(index)
        if sample_weight is not None:
            data["sample_weight"] = sample_weight

        if self.multiview_cameras:
            missing = [camera for camera in self.multiview_cameras if camera not in data["cameras"]]
            if missing:
                raise KeyError(f"Missing cameras {missing} in sample {data['task']}/{data['episode']}.")
            video_by_camera = {
                camera: self._process_video(data["cameras"][camera]["rgb"])
                for camera in self.multiview_cameras
            }
            data["video_views"] = [video_by_camera[camera] for camera in self.multiview_cameras]
            data["input_image_views"] = [video[0] for video in data["video_views"]]
            data["video"] = video_by_camera[self.video_camera] if self.video_camera in video_by_camera else data["video_views"][0]
        else:
            if self.video_camera not in data["cameras"]:
                raise KeyError(f"Camera `{self.video_camera}` is not available in sample {data['task']}/{data['episode']}.")
            data["video"] = self._process_video(data["cameras"][self.video_camera]["rgb"])
            data["input_image"] = data["video"][0]

        eef6d_state = robot_state_to_eef6d(data["robot"], quat_order=self.quat_order)
        current_state, action_target = relative_eef6d_action_from_state_sequence(eef6d_state, self.action_horizon)
        data["current_state"] = normalize_with_stats(current_state, self.statistics, "state")
        data["action_target"] = normalize_with_stats(action_target, self.statistics, "action")
        data["video_supervision_mask"] = torch.ones((), dtype=torch.float32)
        data["action_supervision_mask"] = torch.ones((), dtype=torch.float32)
        return data


def print_dataset_summary(dataset, accelerator, label):
    if not accelerator.is_main_process:
        return
    stats_fn = getattr(dataset, "sample_statistics", None)
    stats = stats_fn() if stats_fn is not None else None
    if stats is None:
        print(f"[RobotwinWorldActionModelTrainingDataset:{label}] samples={len(dataset)}")
        return
    roots = ", ".join(stats["roots"])
    cameras = ", ".join(stats["cameras"])
    multiview = ", ".join(stats["multiview_cameras"]) if stats["multiview_cameras"] else "None"
    print(
        f"[RobotwinWorldActionModelTrainingDataset:{label}] roots={roots}; cameras={cameras}; "
        f"video_camera={stats['video_camera']}; multiview_cameras={multiview}; "
        f"num_video_frames={stats['num_video_frames']}; action_horizon={stats['action_horizon']}; "
        f"statistics_path={stats['statistics_path']}"
    )
    print(
        f"[RobotwinWorldActionModelTrainingDataset:{label}] episodes={stats['episode_count']}; "
        f"windows={stats['window_count']}; samples_per_epoch={len(dataset)}"
    )
    for task_name, row in stats["task_rows"].items():
        print(f"  {task_name}: episodes={row['episodes']}, windows={row['windows']}")


def resolve_cameras(args):
    video_cameras = split_csv(args.world_model_video_camera) or (args.world_model_video_camera,)
    multiview_cameras = video_cameras if len(video_cameras) > 1 else ()
    cameras = list(split_csv(args.world_model_cameras) or ())
    if len(cameras) == 0:
        cameras = list(video_cameras)
    for camera in video_cameras:
        if camera not in cameras:
            cameras.append(camera)
    return tuple(cameras), tuple(multiview_cameras)


def build_dataset(args):
    cameras, multiview_cameras = resolve_cameras(args)
    dataset_window_frames = max(int(args.num_frames), int(args.action_horizon))
    dataset = WorldModelDataset(
        root=args.dataset_base_path,
        tasks=split_csv(args.world_model_tasks),
        cameras=cameras,
        num_frames=dataset_window_frames,
        stride=args.world_model_stride,
        include_depth=False,
        include_camera_params=False,
        include_failed=args.world_model_include_failed,
        repeat=args.dataset_repeat,
        max_data_items=args.max_data_items,
    )
    frame_processor = ImageCropAndResize(
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    return RobotwinWorldActionModelTrainingDataset(
        dataset,
        video_camera=args.world_model_video_camera,
        multiview_cameras=multiview_cameras,
        frame_processor=frame_processor,
        num_video_frames=args.num_frames,
        action_horizon=args.action_horizon,
        statistics_path=args.statistics_path,
        statistics_eps=args.statistics_eps,
        quat_order=args.quat_order,
    )


@contextmanager
def main_process_output(is_main_process):
    if is_main_process:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class WanWorldActionModelTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models="dit",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        action_dim=20,
        action_horizon=33,
        action_max_seq_len=60,
        use_text_condition=True,
        text_context_length=512,
        world_action_checkpoint_path=None,
        lambda_video=1.0,
        lambda_action=1.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is disabled. This model is large and may OOM.")
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        if len(model_configs) == 0:
            model_configs = None
        tokenizer_config = None if tokenizer_path is None else ModelConfig(path=tokenizer_path)
        dit_kwargs = {
            "action_in_dim": action_dim,
            "action_max_seq_len": max(int(action_max_seq_len), int(action_horizon) + 1),
        }
        self.pipe = WanWorldActionModelPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            action_dim=action_dim,
            use_text_condition=use_text_condition,
            text_context_length=text_context_length,
            world_action_checkpoint_path=world_action_checkpoint_path,
            dit_kwargs=dit_kwargs,
        )
        loss_required_params = (
            "input_latents",
            "first_frame_latents",
            "clean_prefix_latent_count",
            "current_state",
            "action_target",
            "video_supervision_mask",
            "action_supervision_mask",
            "max_timestep_boundary",
            "min_timestep_boundary",
            "num_video_views",
            "cfg_scale",
            "lambda_video",
            "lambda_action",
        )
        self.pipe = self.split_pipeline_units(
            task,
            self.pipe,
            trainable_models,
            remove_unnecessary_params=task.endswith(":data_process"),
            loss_required_params=loss_required_params,
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.switch_pipe_to_training_mode(self.pipe, trainable_models, task=task)
        if not use_text_condition:
            self.pipe.remove_dit_language_condition_modules(self.pipe.dit)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.use_text_condition = bool(use_text_condition)
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.lambda_video = float(lambda_video)
        self.lambda_action = float(lambda_action)
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchWanWorldActionLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchWanWorldActionLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        video_views = data.get("video_views")
        view_count = len(video_views) if video_views is not None else 1
        video = data["video"]
        frame_height = video[0].size[1]
        frame_width = video[0].size[0]
        inputs_shared = {
            "input_video": video,
            "height": frame_height,
            "width": frame_width * view_count,
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": False,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "seed": None,
            "current_state": data["current_state"],
            "action_target": data["action_target"],
            "video_supervision_mask": data.get("video_supervision_mask"),
            "action_supervision_mask": data.get("action_supervision_mask"),
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "lambda_video": self.lambda_video,
            "lambda_action": self.lambda_action,
            "disable_context_attention": not self.use_text_condition,
        }
        if video_views is not None:
            inputs_shared["input_video_views"] = video_views
            inputs_shared["input_image_views"] = data.get("input_image_views") or [view[0] for view in video_views]
            inputs_shared["num_video_views"] = view_count
        else:
            inputs_shared["input_image"] = data.get("input_image") or video[0]
        inputs_posi = {"prompt": data.get("prompt", "") if self.use_text_condition else ""}
        inputs_nega = {}
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        inputs = self.get_pipeline_inputs(data) if inputs is None else inputs
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def wan_world_action_model_parser():
    parser = argparse.ArgumentParser(description="WanWorldActionModel training script.")
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_gradient_config(parser)
    parser = add_template_model_config(parser)
    parser = add_offload_training_config(parser)
    parser = add_logger_config(parser)
    parser = add_video_size_config(parser)

    parser.set_defaults(data_file_keys="video", enable_tensorboard_log=True, num_frames=9)
    parser.add_argument("--disable_tensorboard_log", dest="enable_tensorboard_log", default=True, action="store_false")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--world_model_tasks", type=str, default=None)
    parser.add_argument("--world_model_cameras", type=str, default=None)
    parser.add_argument("--world_model_video_camera", type=str, default="head_camera,left_camera,right_camera")
    parser.add_argument("--world_model_stride", type=int, default=None)
    parser.add_argument("--world_model_include_failed", default=False, action="store_true")
    parser.add_argument("--max_data_items", type=int, default=None)
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--action_horizon", type=int, default=33)
    parser.add_argument("--action_max_seq_len", type=int, default=60)
    parser.add_argument("--statistics_path", type=str, default=None)
    parser.add_argument("--statistics_eps", type=float, default=1e-6)
    parser.add_argument("--quat_order", type=str, default="xyzw", choices=("xyzw", "wxyz"))
    parser.add_argument("--world_action_checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--disable_language_condition", "--no_language_condition", dest="use_text_condition", default=True, action="store_false")
    parser.add_argument("--text_context_length", type=int, default=512)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--lambda_video", type=float, default=1.0)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="warmup_cosine", choices=("constant", "warmup_cosine"))
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_cosine_min_ratio", type=float, default=0.1)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--disable_training_checkpoint", dest="save_training_checkpoint", default=True, action="store_false")
    parser.add_argument("--resume_training_checkpoint", type=str, default=None)
    parser.add_argument("--training_checkpoint_dir", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = wan_world_action_model_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args)
    print_dataset_summary(dataset, accelerator, label="train")
    accelerator.wait_for_everyone()

    resume_model_checkpoint = args.resume_from_checkpoint
    if args.resume_training_checkpoint is not None:
        if resume_model_checkpoint is not None:
            warnings.warn("--resume_training_checkpoint restores model weights; ignoring --resume_from_checkpoint for initial load.")
        resume_model_checkpoint = None

    with main_process_output(accelerator.is_main_process):
        model = WanWorldActionModelTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            tokenizer_path=args.tokenizer_path,
            trainable_models=args.trainable_models or "dit",
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            resume_from_checkpoint=resume_model_checkpoint,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
            task=args.task,
            device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            action_max_seq_len=args.action_max_seq_len,
            use_text_condition=args.use_text_condition,
            text_context_length=args.text_context_length,
            world_action_checkpoint_path=args.world_action_checkpoint_path,
            lambda_video=args.lambda_video,
            lambda_action=args.lambda_action,
        )
    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        keep_latest_checkpoint_only=args.keep_latest_checkpoint_only,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
