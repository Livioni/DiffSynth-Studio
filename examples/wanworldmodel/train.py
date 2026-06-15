import argparse
import json
import math
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth.core import UnifiedDataset, WorldModelDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import *
from diffsynth.models.wan_video_dit import normalize_action_injection_method
from diffsynth.pipelines.wan_world_model import ModelConfig, WanWorldModelPipeline
try:
    from examples.wanworldmodel.action_utils import robot_action_to_tensor
except ModuleNotFoundError:
    from action_utils import robot_action_to_tensor


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
    roots = split_csv(dataset_base_path) or (dataset_base_path,)
    candidate_paths = []
    for root in roots:
        candidate_paths.append(os.path.join(root, "metadata.json"))
        candidate_paths.append(os.path.join(os.path.dirname(root), "metadata.json"))
    for path in candidate_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict) and (metadata_key in metadata or "arms" in metadata):
            return path
    return None


def validate_world_model_history_frames(history_frames):
    history_frames = int(history_frames or 0)
    if history_frames < 0:
        raise ValueError(f"--world_model_history_frames must be non-negative, got {history_frames}.")
    if history_frames % 4 != 0:
        raise ValueError(
            "--world_model_history_frames must be divisible by 4 for Wan causal VAE history encoding, "
            f"got {history_frames}."
        )
    return history_frames


def validate_world_model_history_stride(history_stride):
    history_stride = int(history_stride or 4)
    if history_stride < 4:
        raise ValueError(
            "--world_model_history_stride must be at least 4, "
            f"got {history_stride}."
        )
    return history_stride


def validate_probability(value, name):
    value = float(value or 0.0)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def build_history_frame_segments(history_end, history_latent_count, history_stride):
    history_end = int(history_end)
    history_latent_count = int(history_latent_count)
    history_stride = validate_world_model_history_stride(history_stride)
    segments = []
    for offset in reversed(range(history_latent_count)):
        segment_start = history_end - 4 - offset * history_stride
        if segment_start < 0:
            continue
        segments.append(torch.arange(segment_start, segment_start + 4, dtype=torch.long))
    return segments


def flatten_history_frame_segments(segments):
    if len(segments) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(segments, dim=0)


class WorldModelTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        video_camera="head_camera",
        frame_processor=None,
        history_frames=0,
        history_stride=4,
        history_dropout_prob=0.0,
    ):
        self.dataset = dataset
        self.video_camera = video_camera
        self.frame_processor = frame_processor
        self.history_frames = validate_world_model_history_frames(history_frames)
        self.history_stride = validate_world_model_history_stride(history_stride)
        self.history_dropout_prob = validate_probability(history_dropout_prob, "history_dropout_prob")
        self.load_from_cache = False

    def __len__(self):
        return len(self.dataset)

    @property
    def sample_weights(self):
        return getattr(self.dataset, "sample_weights", None)

    def sample_statistics(self):
        episodes = getattr(self.dataset, "episodes", None)
        windows = getattr(self.dataset, "windows", None)
        if episodes is None or windows is None:
            return None

        raw_counts = [0 for _ in episodes]
        for window in windows:
            raw_counts[window.episode_id] += 1

        effective_counts = [0 for _ in episodes]
        effective_sample_count = len(self)
        unique_sample_count = len(windows)
        if unique_sample_count > 0 and effective_sample_count > 0:
            full_cycles, remainder = divmod(effective_sample_count, unique_sample_count)
            effective_counts = [count * full_cycles for count in raw_counts]
            for window in windows[:remainder]:
                effective_counts[window.episode_id] += 1

        task_rows = defaultdict(lambda: {"episodes": 0, "unique_samples": 0, "effective_samples": 0})
        episode_rows = []
        for episode_id, episode in enumerate(episodes):
            root = self._episode_root(episode.path)
            root_label = os.path.basename(root.rstrip(os.sep)) if root else ""
            sample_name = os.path.relpath(episode.path, root) if root else os.path.join(episode.task, episode.episode)
            if root_label:
                sample_name = os.path.join(root_label, sample_name)
            task_key = os.path.join(root_label, episode.task) if root_label else episode.task
            task_rows[task_key]["episodes"] += 1
            task_rows[task_key]["unique_samples"] += raw_counts[episode_id]
            task_rows[task_key]["effective_samples"] += effective_counts[episode_id]
            episode_rows.append(
                {
                    "name": sample_name,
                    "frames": episode.length,
                    "unique_samples": raw_counts[episode_id],
                    "effective_samples": effective_counts[episode_id],
                }
            )

        return {
            "roots": getattr(self.dataset, "roots", ()),
            "cameras": getattr(self.dataset, "cameras", ()),
            "video_camera": self.video_camera,
            "num_frames": getattr(self.dataset, "num_frames", None),
            "stride": getattr(self.dataset, "stride", None),
            "history_frames": self.history_frames,
            "history_stride": self.history_stride,
            "history_dropout_prob": self.history_dropout_prob,
            "repeat": getattr(self.dataset, "repeat", None),
            "max_data_items": getattr(self.dataset, "max_data_items", None),
            "episode_count": len(episodes),
            "unique_sample_count": unique_sample_count,
            "effective_sample_count": effective_sample_count,
            "action_window_quality_stats": getattr(self.dataset, "action_window_quality_stats", None),
            "task_rows": dict(sorted(task_rows.items())),
            "episode_rows": episode_rows,
        }

    def _episode_root(self, episode_path):
        episode_path = os.path.abspath(episode_path)
        for root in getattr(self.dataset, "roots", ()):
            root_path = os.path.abspath(root)
            try:
                if os.path.commonpath((root_path, episode_path)) == root_path:
                    return root
            except ValueError:
                continue
        return None

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
        action = robot_action_to_tensor(data["robot"])
        if action is not None:
            data["action"] = action
        if self.history_frames > 0:
            history_end = int(data["frame_indices"][0])
            history_latent_count = self.history_frames // 4
            use_sparse_history = self.history_stride != 4
            if use_sparse_history:
                history_segments = build_history_frame_segments(history_end, history_latent_count, self.history_stride)
                history_indices = flatten_history_frame_segments(history_segments)
            else:
                history_start = max(0, history_end - self.history_frames)
                history_indices = torch.arange(history_start, history_end, dtype=torch.long)
            data["history_latent_count"] = history_latent_count
            data["history_video"] = []
            drop_history = bool((torch.rand(()) < self.history_dropout_prob).item())
            data["history_dropped"] = drop_history
            data["history_frame_indices"] = torch.empty(0, dtype=torch.long) if drop_history else history_indices
            if len(history_indices) > 0 and not drop_history:
                if use_sparse_history:
                    history_video_segments = []
                    history_action_segments = []
                    for segment_indices in history_segments:
                        history_video_segment = self.dataset._load_rgb_window(
                            data["episode_path"],
                            self.video_camera,
                            segment_indices,
                        )
                        if self.frame_processor is not None:
                            history_video_segment = [self.frame_processor(frame) for frame in history_video_segment]
                        history_video_segments.append(history_video_segment)
                        history_action = robot_action_to_tensor(
                            self.dataset._load_robot_window(data["episode_path"], segment_indices)
                        )
                        if history_action is not None:
                            history_action_segments.append(history_action)
                    data["history_video_segments"] = history_video_segments
                    data["history_video"] = [frame for segment in history_video_segments for frame in segment]
                    if len(history_action_segments) == len(history_video_segments):
                        data["history_action"] = torch.cat(history_action_segments, dim=0)
                else:
                    history_video = self.dataset._load_rgb_window(
                        data["episode_path"],
                        self.video_camera,
                        history_indices,
                    )
                    if self.frame_processor is not None:
                        history_video = [self.frame_processor(frame) for frame in history_video]
                    data["history_video"] = history_video
                    history_action = robot_action_to_tensor(
                        self.dataset._load_robot_window(data["episode_path"], history_indices)
                    )
                    if history_action is not None:
                        data["history_action"] = history_action
        return data


class IndexedEvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_id = self.indices[index]
        return sample_id, self.dataset[sample_id]


@contextmanager
def main_process_output(is_main_process):
    if is_main_process:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def print_world_model_training_dataset_summary(dataset, accelerator, label):
    if not accelerator.is_main_process:
        return
    sample_statistics = getattr(dataset, "sample_statistics", None)
    stats = sample_statistics() if sample_statistics is not None else None
    if stats is None:
        print(f"[Dataset:{label}] samples_per_epoch={len(dataset)}")
        return

    roots = ", ".join(stats["roots"])
    cameras = ", ".join(stats["cameras"])
    max_data_items = stats["max_data_items"] if stats["max_data_items"] is not None else "None"
    print(
        f"[WorldModelTrainingDataset:{label}] roots={roots}; cameras={cameras}; "
        f"video_camera={stats['video_camera']}; num_frames={stats['num_frames']}; "
        f"stride={stats['stride']}; history_frames={stats['history_frames']}; "
        f"history_stride={stats['history_stride']}; "
        f"history_dropout_prob={stats['history_dropout_prob']}; "
        f"repeat={stats['repeat']}; max_data_items={max_data_items}"
    )
    print(
        f"[WorldModelTrainingDataset:{label}] episodes={stats['episode_count']}; "
        f"unique_samples/windows={stats['unique_sample_count']}; "
        f"samples_per_epoch_after_repeat={stats['effective_sample_count']}"
    )
    action_quality = stats.get("action_window_quality_stats")
    if action_quality is not None and action_quality.get("enabled"):
        print(
            f"[WorldModelTrainingDataset:{label}] action_quality="
            f"total_before_filter={action_quality['total_windows_before_filter']}; "
            f"filtered_static={action_quality['filtered_static_windows']}; "
            f"retained={action_quality['retained_windows']}; "
            f"low_delta_weighted={action_quality['low_delta_weighted_windows']}; "
            f"low_threshold={action_quality['action_delta_low_threshold']}; "
            f"low_weight={action_quality['action_delta_low_weight']}"
        )
    print(f"[WorldModelTrainingDataset:{label}] task sample counts:")
    for task_name, row in stats["task_rows"].items():
        print(
            f"  {task_name}: episodes={row['episodes']}, "
            f"unique_samples/windows={row['unique_samples']}, "
            f"samples_per_epoch_after_repeat={row['effective_samples']}"
        )
    print(f"[WorldModelTrainingDataset:{label}] episode sample counts:")
    # for row in stats["episode_rows"]:
    #     print(
    #         f"  {row['name']}: frames={row['frames']}, "
    #         f"unique_samples/windows={row['unique_samples']}, "
    #         f"samples_per_epoch_after_repeat={row['effective_samples']}"
    #     )


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
    history_frames = validate_world_model_history_frames(args.world_model_history_frames)
    history_stride = validate_world_model_history_stride(args.world_model_history_stride)
    history_dropout_prob = validate_probability(args.world_model_history_dropout_prob, "--world_model_history_dropout_prob")
    if history_dropout_prob > 0.0 and history_frames == 0:
        raise ValueError("--world_model_history_dropout_prob requires --world_model_history_frames > 0.")
    cameras = split_csv(args.world_model_cameras) or (args.world_model_video_camera,)
    if args.world_model_video_camera not in cameras:
        cameras = (args.world_model_video_camera,) + cameras
    action_metadata_path = args.action_metadata_path or default_action_metadata_path(
        args.dataset_base_path,
        metadata_key=args.action_metadata_key,
    )
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
        action_metadata_path=action_metadata_path,
        action_metadata_key=args.action_metadata_key,
        action_normalization_eps=args.action_normalization_eps,
        action_normalization_mode=args.action_normalization_mode,
        filter_static_action_windows=args.world_model_filter_static_action_windows,
        static_action_eps=args.world_model_static_action_eps,
        action_delta_low_threshold=args.world_model_action_delta_low_threshold,
        action_delta_low_weight=args.world_model_action_delta_low_weight,
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
        history_frames=history_frames,
        history_stride=history_stride,
        history_dropout_prob=history_dropout_prob,
    )


def build_eval_dataset(args):
    if args.eval_dataset_base_path is None or args.eval_dataset_base_path == "":
        return None
    if not os.path.isdir(args.eval_dataset_base_path):
        warnings.warn(f"Eval dataset root does not exist, skip periodic eval: {args.eval_dataset_base_path}")
        return None

    history_frames = validate_world_model_history_frames(args.world_model_history_frames)
    history_stride = validate_world_model_history_stride(args.world_model_history_stride)
    cameras = split_csv(args.world_model_cameras) or (args.world_model_video_camera,)
    if args.world_model_video_camera not in cameras:
        cameras = (args.world_model_video_camera,) + cameras
    dataset = WorldModelDataset(
        root=args.eval_dataset_base_path,
        tasks=split_csv(args.world_model_tasks),
        cameras=cameras,
        num_frames=args.num_frames,
        stride=args.world_model_stride,
        include_depth=args.world_model_include_depth,
        include_camera_params=args.world_model_include_camera_params,
        include_failed=args.world_model_include_failed,
        repeat=1,
        max_data_items=args.eval_max_samples,
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
        history_frames=history_frames,
        history_stride=history_stride,
        history_dropout_prob=0.0,
    )


def build_dataset(args):
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        has_unified_data = args.dataset_metadata_path is not None or contains_cached_data_files(args.dataset_base_path)
        dataset_type = "unified" if has_unified_data else "world_model"
    history_frames = validate_world_model_history_frames(getattr(args, "world_model_history_frames", 0))
    history_stride = validate_world_model_history_stride(getattr(args, "world_model_history_stride", 4))
    history_dropout_prob = validate_probability(getattr(args, "world_model_history_dropout_prob", 0.0), "--world_model_history_dropout_prob")
    if (history_frames > 0 or history_dropout_prob > 0.0) and dataset_type != "world_model":
        raise ValueError("--world_model_history_frames is only supported with --dataset_type=world_model.")
    if history_dropout_prob > 0.0 and history_frames == 0:
        raise ValueError("--world_model_history_dropout_prob requires --world_model_history_frames > 0.")

    if dataset_type == "world_model":
        return build_world_model_dataset(args)
    if dataset_type == "unified":
        return build_unified_dataset(args)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def pil_video_to_tensor(video):
    frames = [np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0 for frame in video]
    tensor = torch.from_numpy(np.stack(frames, axis=0))
    return tensor.permute(0, 3, 1, 2).contiguous()


def save_pil_video(video, path, fps=4):
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video]
    imageio.mimsave(path, frames, fps=fps)


def resize_pil_video(video, width, height):
    if len(video) == 0:
        return video
    target_size = (int(width), int(height))
    if all(frame.size == target_size for frame in video):
        return video
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return [frame.resize(target_size, resample=resample) for frame in video]


def FlowMatchWanWorldModelHistoryLoss(pipe, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    history_latent_count = int(inputs.get("history_latent_count") or 0)
    model_inputs = inputs
    if history_latent_count > 0:
        history_latents = inputs.get("history_latents")
        if history_latents is None:
            raise ValueError("history_latents is required when history_latent_count > 0.")
        model_inputs = dict(inputs)
        model_inputs["latents"] = torch.cat([history_latents, inputs["latents"]], dim=2)
        model_inputs["clean_prefix_latent_count"] = history_latent_count + (
            1 if "first_frame_latents" in inputs else 0
        )

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

    if history_latent_count > 0:
        noise_pred = noise_pred[:, :, history_latent_count:]
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def ssim_torch(x, y, window_size=11, sigma=1.5):
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    channels = x.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean()


class WanWorldModelEvalCallback:
    def __init__(
        self,
        dataset,
        eval_steps=1000,
        num_inference_steps=50,
        num_workers=0,
        num_videos_to_log=4,
        upload_video_steps=None,
        video_fps=4,
        metric_batch_size=4,
        output_path="./models",
    ):
        self.dataset = dataset
        self.eval_steps = int(eval_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.num_videos_to_log = int(num_videos_to_log)
        self.upload_video_steps = self.eval_steps if upload_video_steps is None else int(upload_video_steps)
        self.video_fps = int(video_fps)
        self.metric_batch_size = int(metric_batch_size)
        self.output_path = output_path
        self.num_workers = int(num_workers)
        self.lpips_metric = None
        self.fid_metric = None

    def should_run(self, step):
        return self.eval_steps > 0 and step > 0 and step % self.eval_steps == 0

    def should_upload_videos(self, step):
        return self.upload_video_steps > 0 and step > 0 and step % self.upload_video_steps == 0

    def build_rank_dataloader(self, accelerator):
        indices = range(accelerator.process_index, len(self.dataset), accelerator.num_processes)
        dataset = IndexedEvalDataset(self.dataset, indices)
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=self.num_workers,
        )

    def _ensure_lpips_metric(self, device):
        if self.lpips_metric is None:
            from diffsynth.metrics import LPIPSMetric
            self.lpips_metric = LPIPSMetric.from_pretrained(
                net="vgg",
                device=device,
                batch_size=self.metric_batch_size,
            )

    def _ensure_fid_metric(self, device):
        if self.fid_metric is None:
            from diffsynth.metrics import FIDMetric
            self.fid_metric = FIDMetric.from_pretrained(
                device=device,
                batch_size=self.metric_batch_size,
            )

    def _ensure_metrics(self, device):
        self._ensure_lpips_metric(device)
        self._ensure_fid_metric(device)

    def _reconstruct_video(self, pipe, video):
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(video)
        latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        decoded = pipe.vae.decode(
            latents,
            device=pipe.device,
            tiled=False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        return pipe.vae_output_to_video(decoded)

    def _evaluate_sample(self, pipe, data, sample_id, step, video_paths, upload_videos):
        x_gt = data["video"]
        target_height, target_width, target_num_frames = pipe.check_resize_height_width(
            x_gt[0].size[1],
            x_gt[0].size[0],
            len(x_gt),
            verbose=False,
        )
        x_gt = resize_pil_video(x_gt, target_width, target_height)
        x_pred = pipe(
            prompt=data.get("prompt", ""),
            negative_prompt="",
            input_image=x_gt[0],
            seed=sample_id,
            rand_device=pipe.device,
            height=target_height,
            width=target_width,
            num_frames=target_num_frames,
            cfg_scale=1,
            num_inference_steps=self.num_inference_steps,
            tiled=False,
            action=data.get("action"),
            history_video=data.get("history_video"),
            history_action=data.get("history_action"),
            history_latent_count=data.get("history_latent_count", 0),
            progress_bar_cmd=lambda x: x,
            output_type="quantized",
        )
        if len(x_pred) > 0 and x_pred[0].size != x_gt[0].size:
            x_gt = resize_pil_video(x_gt, x_pred[0].size[0], x_pred[0].size[1])
        if upload_videos and sample_id < self.num_videos_to_log:
            x_reconst = self._reconstruct_video(pipe, x_gt)
            sample_dir = os.path.join(self.output_path, "eval_videos", f"step-{step}", f"sample_{sample_id}")
            paths = {
                f"val_vis/sample_{sample_id}/x_gt": os.path.join(sample_dir, "x_gt.mp4"),
                f"val_vis/sample_{sample_id}/x_pred": os.path.join(sample_dir, "x_pred.mp4"),
                f"val_vis/sample_{sample_id}/x_reconst": os.path.join(sample_dir, "x_reconst.mp4"),
            }
            save_pil_video(x_gt, paths[f"val_vis/sample_{sample_id}/x_gt"], fps=self.video_fps)
            save_pil_video(x_pred, paths[f"val_vis/sample_{sample_id}/x_pred"], fps=self.video_fps)
            save_pil_video(x_reconst, paths[f"val_vis/sample_{sample_id}/x_reconst"], fps=self.video_fps)
            video_paths.update(paths)

        metric_frame_count = min(len(x_pred), len(x_gt))
        if metric_frame_count <= 1:
            raise ValueError(
                f"Eval sample {sample_id} has too few aligned frames: "
                f"len(x_pred)={len(x_pred)}, len(x_gt)={len(x_gt)}."
            )
        pred_metrics = x_pred[1:metric_frame_count]
        gt_metrics = x_gt[1:metric_frame_count]
        pred_tensor = pil_video_to_tensor(pred_metrics)
        gt_tensor = pil_video_to_tensor(gt_metrics)
        return pred_metrics, gt_metrics, pred_tensor, gt_tensor

    def _sum_across_processes(self, accelerator, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=accelerator.device)
        if accelerator.num_processes == 1:
            return tensor.detach().cpu()
        gathered = accelerator.gather(tensor)
        return gathered.view(accelerator.num_processes, tensor.numel()).sum(dim=0).detach().cpu()

    def _gather_variable_length_tensor(self, accelerator, tensor):
        if tensor.device != accelerator.device:
            tensor = tensor.to(device=accelerator.device)
        length = torch.tensor([tensor.shape[0]], dtype=torch.long, device=accelerator.device)
        if accelerator.num_processes == 1:
            return tensor.detach().cpu()

        lengths = accelerator.gather(length).to(device=accelerator.device)
        max_length = int(lengths.max().item())
        feature_shape = tensor.shape[1:]
        if tensor.shape[0] < max_length:
            padding = tensor.new_zeros((max_length - tensor.shape[0], *feature_shape))
            tensor = torch.cat([tensor, padding], dim=0)
        if max_length == 0:
            gathered = tensor.new_empty((accelerator.num_processes, 0, *feature_shape))
        else:
            gathered = accelerator.gather(tensor.contiguous())
            gathered = gathered.view(accelerator.num_processes, max_length, *feature_shape)

        if not accelerator.is_main_process:
            return None
        chunks = []
        for rank, rank_length in enumerate(lengths.detach().cpu().tolist()):
            if rank_length > 0:
                chunks.append(gathered[rank, :rank_length].detach().cpu())
        if len(chunks) == 0:
            return tensor.new_empty((0, *feature_shape)).detach().cpu()
        return torch.cat(chunks, dim=0)

    def _gather_objects(self, accelerator, value):
        if accelerator.num_processes == 1:
            return [value]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return [value]
        gathered = [None for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _compute_local_metric_sums(self, pred_frames, gt_frames, pred_tensors, gt_tensors, device):
        if len(pred_tensors) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        pred = torch.cat(pred_tensors, dim=0).to(device=device)
        gt = torch.cat(gt_tensors, dim=0).to(device=device)
        frame_count = pred.shape[0]

        squared_error_sum = F.mse_loss(pred, gt, reduction="sum").detach().cpu().item()
        numel = float(pred.numel())
        ssim_sum = float(ssim_torch(pred, gt).detach().cpu().item()) * frame_count

        self._ensure_lpips_metric(device)
        lpips_sum = 0.0
        for pred_frame, gt_frame in zip(pred_frames, gt_frames):
            lpips_sum += float(self.lpips_metric.compute(pred_frame, gt_frame))

        return [
            squared_error_sum,
            numel,
            ssim_sum,
            float(frame_count),
            lpips_sum,
            float(len(pred_frames)),
        ]

    def _compute_local_fid_activations(self, gt_frames, pred_frames, device):
        feature_dim = 2048
        if len(pred_frames) == 0:
            empty = torch.empty((0, feature_dim), dtype=torch.float64, device=device)
            return empty, empty
        self._ensure_fid_metric(device)
        gt_activations = self.fid_metric.model.get_activations(
            gt_frames,
            batch_size=self.metric_batch_size,
        ).to(device=device)
        pred_activations = self.fid_metric.model.get_activations(
            pred_frames,
            batch_size=self.metric_batch_size,
        ).to(device=device)
        return gt_activations, pred_activations

    def _compute_distributed_metrics(self, accelerator, pred_frames, gt_frames, pred_tensors, gt_tensors, device):
        local_sums = self._compute_local_metric_sums(
            pred_frames,
            gt_frames,
            pred_tensors,
            gt_tensors,
            device=device,
        )
        global_sums = self._sum_across_processes(accelerator, local_sums)

        gt_activations, pred_activations = self._compute_local_fid_activations(gt_frames, pred_frames, device)
        gt_activations = self._gather_variable_length_tensor(accelerator, gt_activations)
        pred_activations = self._gather_variable_length_tensor(accelerator, pred_activations)

        if not accelerator.is_main_process:
            return None

        squared_error_sum, numel, ssim_sum, ssim_count, lpips_sum, lpips_count = global_sums.tolist()
        mse = float(squared_error_sum / max(numel, 1.0))
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim = float(ssim_sum / max(ssim_count, 1.0))
        lpips = float(lpips_sum / max(lpips_count, 1.0))

        self._ensure_fid_metric(device)
        gt_mean, gt_covariance = self.fid_metric.model.activation_statistics(gt_activations)
        pred_mean, pred_covariance = self.fid_metric.model.activation_statistics(pred_activations)
        fid = self.fid_metric.model.frechet_distance(
            gt_mean,
            gt_covariance,
            pred_mean,
            pred_covariance,
        )
        fid = fid.detach().cpu().item() if torch.is_tensor(fid) else float(fid)

        return {
            "val/mse": mse,
            "val/psnr": psnr,
            "val/ssim": ssim,
            "val/lpips": lpips,
            "val/fid": fid,
        }

    def __call__(self, accelerator, model, model_logger):
        training_module = accelerator.unwrap_model(model)
        module_modes = [(module, module.training) for module in training_module.modules()]
        pipe = training_module.pipe
        training_units = pipe.units
        step = model_logger.num_steps

        pred_frames = []
        gt_frames = []
        pred_tensors = []
        gt_tensors = []
        video_paths = {}
        upload_videos = self.should_upload_videos(step)

        try:
            pipe.units = getattr(training_module, "inference_units", pipe.units)
            training_module.eval()
            with torch.no_grad():
                for sample_id, data in self.build_rank_dataloader(accelerator):
                    pred_metric_frames, gt_metric_frames, pred_tensor, gt_tensor = self._evaluate_sample(
                        pipe,
                        data,
                        sample_id,
                        step,
                        video_paths,
                        upload_videos,
                    )
                    pred_frames.extend(pred_metric_frames)
                    gt_frames.extend(gt_metric_frames)
                    pred_tensors.append(pred_tensor)
                    gt_tensors.append(gt_tensor)

                metrics = self._compute_distributed_metrics(
                    accelerator,
                    pred_frames,
                    gt_frames,
                    pred_tensors,
                    gt_tensors,
                    device=pipe.device,
                )
                gathered_video_paths = self._gather_objects(accelerator, video_paths)
                if accelerator.is_main_process:
                    merged_video_paths = {}
                    for rank_video_paths in gathered_video_paths:
                        if rank_video_paths:
                            merged_video_paths.update(rank_video_paths)
                    print(f"[Eval step {step}] " + ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
                    model_logger.log_metrics(metrics, step=step)
                    if merged_video_paths:
                        model_logger.log_videos(merged_video_paths, step=step, fps=self.video_fps)
        finally:
            pipe.units = training_units
            for module, training in module_modes:
                module.train(training)
            pipe.scheduler.set_timesteps(1000, training=True)
            pipe.load_models_to_device([])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
        action_normalization_mode="standard",
        use_text_condition=True,
        text_context_length=512,
        world_model_history_frames=0,
    ):
        super().__init__()
        world_model_history_frames = validate_world_model_history_frames(world_model_history_frames)
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

        model_load_device = device
        if not use_text_condition and not fp8_models and not offload_models:
            model_load_device = "cpu"
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=model_load_device,
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
            device=model_load_device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            action_dim=action_dim,
            action_embedder_hidden_dim=action_embedder_hidden_dim,
            action_injection_method=action_injection_method,
            action_metadata_path=action_metadata_path,
            action_metadata_key=action_metadata_key,
            action_normalization_eps=action_normalization_eps,
            action_normalization_mode=action_normalization_mode,
            use_text_condition=use_text_condition,
            text_context_length=text_context_length,
            redirect_common_files=True,
        )
        self.inference_units = list(self.pipe.units)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.use_text_condition = bool(use_text_condition)

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            task=task,
        )
        if not self.use_text_condition:
            self.freeze_language_condition_modules()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.world_model_history_frames = world_model_history_frames
        self.world_model_history_latent_count = world_model_history_frames // 4
        self.extra_inputs = [item.strip() for item in extra_inputs.split(",") if item.strip()] if extra_inputs is not None else []
        if action_enabled and "action" not in self.extra_inputs:
            self.extra_inputs.append("action")
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchWanWorldModelHistoryLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchWanWorldModelHistoryLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @staticmethod
    def _is_language_condition_state_key(name):
        return (
            ".text_embedding." in name
            or ".cross_attn." in name
            or ".norm3." in name
        )

    def _language_condition_modules_removed(self):
        dit = getattr(self.pipe, "dit", None)
        return dit is not None and getattr(dit, "text_embedding", None) is None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self._language_condition_modules_removed():
            state_dict = {
                name: value
                for name, value in state_dict.items()
                if not self._is_language_condition_state_key(name)
            }
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def freeze_language_condition_modules(self):
        dit = self.pipe.dit
        if dit is None:
            return
        if getattr(dit, "text_embedding", None) is not None:
            dit.text_embedding.eval()
            dit.text_embedding.requires_grad_(False)
        for block in getattr(dit, "blocks", []):
            for name in ("cross_attn", "norm3"):
                module = getattr(block, name, None)
                if module is not None:
                    module.eval()
                    module.requires_grad_(False)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                # Wan2.2-TI2V-5B 的 I2V 条件直接取训练视频首帧，不要求 metadata 单独提供 image。
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "action" and "action" not in data and "robot" in data:
                inputs_shared["action"] = robot_action_to_tensor(data["robot"])
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"] if self.use_text_condition else data.get("prompt", "")}
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
            "disable_context_attention": not self.use_text_condition,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        if self.world_model_history_latent_count > 0:
            inputs_shared["history_video"] = data.get("history_video", [])
            inputs_shared["history_video_segments"] = data.get("history_video_segments")
            inputs_shared["history_latent_count"] = data.get("history_latent_count", self.world_model_history_latent_count)
            if "history_action" in data:
                inputs_shared["history_action"] = data["history_action"]
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

    # Shared DiffSynth training/parser options.
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_gradient_config(parser)
    parser = add_template_model_config(parser)
    parser = add_offload_training_config(parser)
    parser = add_logger_config(parser)
    parser = add_video_size_config(parser)

    # Wan world-model defaults.
    parser.set_defaults(data_file_keys="video", extra_inputs="input_image", enable_tensorboard_log=True)

    # Logging.
    parser.add_argument("--disable_tensorboard_log", dest="enable_tensorboard_log", default=True, action="store_false", help="Disable tensorboard logging.")
    parser.add_argument("--log_steps", type=int, default=10, help="Log train scalar metrics every N training steps. Set <= 0 to log every step.")

    # Dataset backend.
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="world_model",
        choices=("auto", "unified", "world_model"),
        help="Dataset backend. Defaults to WorldModelDataset. Use `unified` for metadata/cached data, or `auto` to detect it.",
    )

    # WorldModelDataset indexing and camera streams.
    parser.add_argument("--world_model_tasks", type=str, default=None, help="Comma-separated task folders to load from WorldModelDataset.")
    parser.add_argument("--world_model_cameras", type=str, default=None, help="Comma-separated cameras to load. Defaults to --world_model_video_camera.")
    parser.add_argument("--world_model_video_camera", type=str, default="head_camera", help="Camera RGB stream used as training video.")
    parser.add_argument("--world_model_stride", type=int, default=None, help="Stride between fixed-length world-model windows. Defaults to num_frames.")
    parser.add_argument(
        "--world_model_history_frames",
        type=int,
        default=0,
        help="Number of RGB frames before each window used as causal VAE history. Must be divisible by 4; 16 gives 4 history latents. Missing prefix history is zero-padded in latent space.",
    )
    parser.add_argument(
        "--world_model_history_stride",
        type=int,
        default=4,
        help="Distance in raw frames between consecutive 4-frame history blocks. The default 4 keeps contiguous history; values >4 sparsify history while each VAE history latent still sees 4 continuous frames.",
    )
    parser.add_argument(
        "--world_model_history_dropout_prob",
        type=float,
        default=0.0,
        help="Training-only probability of dropping all history RGB/action for a sample while keeping zero-padded history latent slots.",
    )
    parser.add_argument("--world_model_include_depth", default=False, action="store_true", help="Load depth arrays from WorldModelDataset.")
    parser.add_argument("--world_model_include_camera_params", default=False, action="store_true", help="Load camera intrinsics/extrinsics from WorldModelDataset.")
    parser.add_argument("--world_model_include_failed", default=False, action="store_true", help="Include failed episodes from WorldModelDataset.")
    parser.add_argument(
        "--world_model_filter_static_action_windows",
        default=False,
        action="store_true",
        help="Filter windows whose adjacent-frame action delta is zero within --world_model_static_action_eps.",
    )
    parser.add_argument(
        "--world_model_static_action_eps",
        type=float,
        default=1e-8,
        help="Maximum normalized action delta max used to classify a window as static.",
    )
    parser.add_argument(
        "--world_model_action_delta_low_threshold",
        type=float,
        default=None,
        help="Normalized action delta mean threshold for down-weighting low-motion windows. Leave unset to disable.",
    )
    parser.add_argument(
        "--world_model_action_delta_low_weight",
        type=float,
        default=1.0,
        help="Sampling weight assigned to windows below --world_model_action_delta_low_threshold.",
    )

    # Periodic evaluation.
    parser.add_argument(
        "--eval_dataset_base_path",
        type=str,
        default="world_model_data/robotwin_aloha_testset/custom_aloha_clean",
        help="Eval dataset root. Uses the same WorldModelDataset config as training and train-set action metadata.",
    )
    parser.add_argument("--eval_steps", type=int, default=2000, help="Run periodic eval every N training steps. Set <= 0 to disable.")
    parser.add_argument("--eval_num_inference_steps", type=int, default=50, help="Diffusion inference steps used during eval.")
    parser.add_argument("--eval_max_samples", type=int, default=2, help="Maximum eval windows to run. Defaults to all eval windows.")
    parser.add_argument("--eval_dataset_num_workers", type=int, default=0, help="Number of workers for eval data loading.")
    parser.add_argument("--eval_num_videos_to_log", type=int, default=4, help="Number of eval samples whose videos are logged.")
    parser.add_argument("--upload_video_steps", type=int, default=None, help="Upload eval videos every N training steps. Defaults to eval_steps. Set <= 0 to disable video uploads.")
    parser.add_argument("--eval_video_fps", type=int, default=4, help="FPS for logged eval videos.")
    parser.add_argument("--eval_metric_batch_size", type=int, default=4, help="Batch size for LPIPS/FID metric models.")

    # Language / text conditioning.
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--disable_language_condition",
        "--no_language_condition",
        dest="use_text_condition",
        default=True,
        action="store_false",
        help="Disable prompt/language conditioning. The pipeline skips context attention and does not load the text encoder/tokenizer.",
    )
    parser.add_argument(
        "--text_context_length",
        type=int,
        default=512,
        help="Text context token length used when language conditioning is enabled.",
    )

    # Robot action conditioning.
    parser.add_argument("--action_dim", type=int, default=14, help="Robot action vector dimension. Enable action conditioning when set.")
    parser.add_argument(
        "--action_metadata_path",
        type=str,
        default=None,
        help="Path to robot action normalization metadata. Defaults to <dataset_base_path>/metadata.json when it contains robot_statistics.",
    )
    parser.add_argument("--action_metadata_key", type=str, default="robot_statistics", help="Top-level metadata key for robot statistics.")
    parser.add_argument("--action_normalization_eps", type=float, default=1e-6, help="Minimum std value used for action normalization.")
    parser.add_argument(
        "--action_normalization_mode",
        type=str,
        default="standard",
        choices=("standard", "scale_only", "scale-only"),
        help="Action normalization mode. `standard` uses (action - mean) / std; `scale_only` uses action / std.",
    )
    parser.add_argument("--action_embedder_hidden_dim", type=int, default=None, help="Hidden dimension of the action embedder MLP. Defaults to Wan DiT hidden dimension.")
    parser.add_argument(
        "--action_injection_method",
        type=str,
        default="additive",
        choices=("none", "context", "additive", "cross_attention", "cross-attention", "adaln", "film"),
        help="Action conditioning method: `context` keeps the previous context-token scheme; the others inject action in Wan DiT blocks.",
    )

    # Diffusion timestep sampling.
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Maximum timestep boundary ratio.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Minimum timestep boundary ratio.")

    # Optimizer schedule.
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="warmup_cosine",
        choices=("constant", "warmup_cosine"),
        help="Learning-rate scheduler. Defaults to linear warmup followed by cosine decay.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Linear learning-rate warmup steps.")
    parser.add_argument(
        "--lr_cosine_min_ratio",
        type=float,
        default=0.1,
        help="Final learning-rate ratio for warmup_cosine decay.",
    )

    # Initialization and full training-state checkpoints.
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument(
        "--disable_training_checkpoint",
        dest="save_training_checkpoint",
        default=True,
        action="store_false",
        help="Disable full training-state checkpoints saved at --save_steps.",
    )
    parser.add_argument(
        "--resume_training_checkpoint",
        type=str,
        default=None,
        help="Resume optimizer, scheduler, RNG, model state, and global step from a training checkpoint directory. Use `latest` for the latest checkpoint under --training_checkpoint_dir.",
    )
    parser.add_argument(
        "--training_checkpoint_dir",
        type=str,
        default=None,
        help="Directory for full training-state checkpoints. Defaults to <output_path>/training_checkpoints.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_world_model_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args)
    print_world_model_training_dataset_summary(dataset, accelerator, label="train")
    eval_dataset = None
    if args.eval_steps > 0 and not args.task.endswith(":data_process"):
        eval_dataset = build_eval_dataset(args)
    if eval_dataset is not None and len(eval_dataset) == 0:
        warnings.warn("Eval dataset is empty, skip periodic eval.")
        eval_dataset = None
    if eval_dataset is not None:
        print_world_model_training_dataset_summary(eval_dataset, accelerator, label="eval")
    if eval_dataset is not None and args.enable_model_cpu_offload:
        raise ValueError("Periodic eval is not supported with --enable_model_cpu_offload. Disable eval or model CPU offload.")
    accelerator.wait_for_everyone()

    action_enabled = args.action_dim is not None and normalize_action_injection_method(args.action_injection_method) != "none"
    action_metadata_path = None
    if action_enabled:
        action_metadata_path = args.action_metadata_path or default_action_metadata_path(
            args.dataset_base_path,
            metadata_key=args.action_metadata_key,
        )
    resume_model_checkpoint = args.resume_from_checkpoint
    if args.resume_training_checkpoint is not None:
        if resume_model_checkpoint is not None:
            warnings.warn(
                "--resume_training_checkpoint restores model weights itself; "
                "ignore --resume_from_checkpoint for the initial model-only load."
            )
        resume_model_checkpoint = None

    with main_process_output(accelerator.is_main_process):
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
            resume_from_checkpoint=resume_model_checkpoint,
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
            action_normalization_mode=args.action_normalization_mode,
            use_text_condition=args.use_text_condition,
            text_context_length=args.text_context_length,
            world_model_history_frames=args.world_model_history_frames,
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
    eval_callback = None
    if eval_dataset is not None:
        eval_callback = WanWorldModelEvalCallback(
            dataset=eval_dataset,
            eval_steps=args.eval_steps,
            num_inference_steps=args.eval_num_inference_steps,
            num_workers=args.eval_dataset_num_workers,
            num_videos_to_log=args.eval_num_videos_to_log,
            upload_video_steps=args.upload_video_steps,
            video_fps=args.eval_video_fps,
            metric_batch_size=args.eval_metric_batch_size,
            output_path=args.output_path,
        )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args, eval_callback=eval_callback)
