import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core.data.world_model_dataset import WorldModelDataset


IMAGE_METRICS = (
    "image_frame_l1_mean",
    "image_frame_l1_max",
    "image_first_last_l1",
)
ACTION_METRICS = (
    "action_l2_mean",
    "action_l2_max",
    "action_delta_l2_mean",
    "action_delta_l2_max",
    "action_first_last_l2",
)
METRICS = IMAGE_METRICS + ACTION_METRICS
PERCENTILES = (10, 25, 50, 75, 90, 95)


def split_csv(value):
    if value is None:
        return None
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(item.strip() for item in values if str(item).strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-task RobotWin window motion/action statistics. "
            "Image metrics are mean absolute RGB differences in [0, 1]."
        )
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/clean_set,world_model_data/robotwin_aloha/fail_set",
        help="Comma-separated RobotWin roots, matching training --dataset_base_path.",
    )
    parser.add_argument("--tasks", type=str, default=None, help="Optional comma-separated task filter.")
    parser.add_argument("--camera", type=str, default="head_camera", help="Camera used for image motion statistics.")
    parser.add_argument("--num_frames", type=int, default=25, help="Window length, matching training --num_frames.")
    parser.add_argument("--stride", type=int, default=6, help="Window stride, matching training --world_model_stride.")
    parser.add_argument(
        "--include_failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include episodes marked as failed by metadata.",
    )
    parser.add_argument(
        "--group_by",
        choices=("task", "root_task"),
        default="task",
        help="Use task to merge train/fail roots, or root_task to split them.",
    )
    parser.add_argument(
        "--metrics",
        choices=("all", "action", "image"),
        default="all",
        help="Choose which metric family to compute. `action` skips image loading entirely.",
    )
    parser.add_argument(
        "--action_only",
        action="store_true",
        help="Alias for --metrics action.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize frames to image_size x image_size before computing pixel differences. Use 0 for original size.",
    )
    parser.add_argument(
        "--max_windows_per_group",
        type=int,
        default=0,
        help="Deterministically sample at most this many windows per group. 0 means all windows.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for --max_windows_per_group.")
    parser.add_argument(
        "--action_metadata_path",
        type=str,
        default=None,
        help="Path to metadata.json for action normalization. By default, auto-detects next to dataset roots.",
    )
    parser.add_argument("--action_metadata_key", type=str, default="robot_statistics")
    parser.add_argument(
        "--normalize_action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize actions with metadata mean/std when metadata is available.",
    )
    parser.add_argument("--action_normalization_eps", type=float, default=1e-6)
    parser.add_argument("--image_frame_low_threshold", type=float, default=0.01)
    parser.add_argument("--image_first_last_low_threshold", type=float, default=0.03)
    parser.add_argument("--action_l2_low_threshold", type=float, default=0.5)
    parser.add_argument("--action_delta_low_threshold", type=float, default=0.25)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/robotwin_motion_stats_by_task.csv",
        help="Per-group summary CSV path.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/robotwin_motion_stats_by_task.json",
        help="Per-group summary JSON path. Use empty string to disable.",
    )
    parser.add_argument(
        "--window_csv",
        type=str,
        default=None,
        help="Optional per-window metrics CSV. This can be large when scanning all windows.",
    )
    parser.add_argument("--progress_every", type=int, default=50, help="Print progress every N loaded episodes.")
    args = parser.parse_args()
    if args.action_only:
        args.metrics = "action"
    return args


def selected_metric_names(args):
    if args.metrics == "action":
        return ACTION_METRICS
    if args.metrics == "image":
        return IMAGE_METRICS
    return METRICS


def root_label_for_episode(dataset, episode_path):
    episode_path = os.path.abspath(episode_path)
    for root in dataset.roots:
        root_path = os.path.abspath(root)
        try:
            if os.path.commonpath((root_path, episode_path)) == root_path:
                return os.path.basename(root_path.rstrip(os.sep))
        except ValueError:
            continue
    return ""


def group_key_for_episode(dataset, episode, group_by):
    root_label = root_label_for_episode(dataset, episode.path)
    if group_by == "root_task":
        return f"{root_label}/{episode.task}" if root_label else episode.task
    return episode.task


def auto_action_metadata_path(dataset_base_path):
    for root in split_csv(dataset_base_path) or ():
        candidates = [
            os.path.join(root, "metadata.json"),
            os.path.join(os.path.dirname(root), "metadata.json"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    return None


def load_action_stats(path, metadata_key, action_dim, eps):
    if not path:
        return None
    with open(path, "r") as f:
        metadata = json.load(f)
    if metadata_key in metadata:
        metadata = metadata[metadata_key]

    mean = []
    std = []
    for arm in ("left", "right"):
        action_stats = metadata["arms"][arm]["action"]
        mean.extend(float(value) for value in action_stats["mean"])
        std.extend(max(float(value), eps) for value in action_stats["std"])

    if len(mean) != action_dim:
        raise ValueError(f"Action metadata dimension {len(mean)} does not match action_dim {action_dim}.")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def action_value_to_numpy(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    return value


def load_episode_actions(dataset, episode, action_stats, eps):
    paths = dataset._robot_paths(episode.path)
    if paths is None:
        raise FileNotFoundError(f"Missing robot_data files: {episode.path}")

    pieces = []
    for arm in ("left", "right"):
        for key in ("arm_joint", "gripper"):
            values = np.load(paths[arm]["action"][key], mmap_mode="r")[: episode.length]
            pieces.append(action_value_to_numpy(values))
    actions = np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)

    if action_stats is not None:
        mean, std = action_stats
        actions = (actions - mean.reshape(1, -1)) / (std.reshape(1, -1) + eps)
    return actions


def load_rgb_files(dataset, episode, camera):
    folder = os.path.join(episode.path, "camera_data", "images", camera)
    files = dataset._list_frame_files(folder, dataset.rgb_extensions)
    if len(files) < episode.length:
        raise FileNotFoundError(f"Not enough RGB frames in {folder}: {len(files)} < {episode.length}")
    return files[: episode.length]


def load_image_array(path, image_size):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size > 0:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.resize((image_size, image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_episode_frames(dataset, episode, camera, image_size):
    files = load_rgb_files(dataset, episode, camera)
    return np.stack([load_image_array(path, image_size) for path in files], axis=0)


def episode_window_starts(dataset):
    starts_by_episode = defaultdict(list)
    for window in dataset.windows:
        starts_by_episode[window.episode_id].append(window.start)
    return starts_by_episode


def sample_windows_by_group(dataset, starts_by_episode, group_by, max_windows_per_group, seed):
    if max_windows_per_group <= 0:
        return starts_by_episode

    refs_by_group = defaultdict(list)
    for episode_id, starts in starts_by_episode.items():
        episode = dataset.episodes[episode_id]
        group = group_key_for_episode(dataset, episode, group_by)
        for start in starts:
            refs_by_group[group].append((episode_id, start))

    rng = random.Random(seed)
    sampled_starts = defaultdict(list)
    for group, refs in refs_by_group.items():
        if len(refs) > max_windows_per_group:
            refs = rng.sample(refs, max_windows_per_group)
        for episode_id, start in refs:
            sampled_starts[episode_id].append(start)
    for starts in sampled_starts.values():
        starts.sort()
    return sampled_starts


def empty_group(metric_names):
    return {
        "episodes": set(),
        "root_windows": defaultdict(int),
        "success_windows": 0,
        "failed_windows": 0,
        "metrics": {name: [] for name in metric_names},
        "low": defaultdict(int),
    }


def safe_mean(values):
    return float(np.mean(values)) if len(values) else math.nan


def safe_max(values):
    return float(np.max(values)) if len(values) else math.nan


def compute_window_metrics(frames, actions, start, num_frames, metric_names):
    end = start + num_frames
    metric_names = set(metric_names)
    metrics = {}

    if any(name in metric_names for name in IMAGE_METRICS):
        window_frames = frames[start:end]
        image_pair_l1 = np.mean(np.abs(window_frames[1:] - window_frames[:-1]), axis=(1, 2, 3))
        metrics.update(
            {
                "image_frame_l1_mean": safe_mean(image_pair_l1),
                "image_frame_l1_max": safe_max(image_pair_l1),
                "image_first_last_l1": float(np.mean(np.abs(window_frames[-1] - window_frames[0]))),
            }
        )

    if any(name in metric_names for name in ACTION_METRICS):
        window_actions = actions[start:end]
        action_l2 = np.linalg.norm(window_actions, axis=-1)
        action_delta_l2 = np.linalg.norm(window_actions[1:] - window_actions[:-1], axis=-1)
        metrics.update(
            {
                "action_l2_mean": safe_mean(action_l2),
                "action_l2_max": safe_max(action_l2),
                "action_delta_l2_mean": safe_mean(action_delta_l2),
                "action_delta_l2_max": safe_max(action_delta_l2),
                "action_first_last_l2": float(np.linalg.norm(window_actions[-1] - window_actions[0])),
            }
        )

    return metrics


def update_low_motion_counts(group, metrics, args):
    image_frame_low = None
    image_first_last_low = None
    action_l2_low = None
    action_delta_low = None

    if "image_frame_l1_mean" in metrics:
        image_frame_low = metrics["image_frame_l1_mean"] < args.image_frame_low_threshold
        group["low"]["image_frame_low"] += int(image_frame_low)
    if "image_first_last_l1" in metrics:
        image_first_last_low = metrics["image_first_last_l1"] < args.image_first_last_low_threshold
        group["low"]["image_first_last_low"] += int(image_first_last_low)
    if "action_l2_mean" in metrics:
        action_l2_low = metrics["action_l2_mean"] < args.action_l2_low_threshold
        group["low"]["action_l2_low"] += int(action_l2_low)
    if "action_delta_l2_mean" in metrics:
        action_delta_low = metrics["action_delta_l2_mean"] < args.action_delta_low_threshold
        group["low"]["action_delta_low"] += int(action_delta_low)
    if image_frame_low is not None and action_delta_low is not None:
        group["low"]["image_frame_and_action_delta_low"] += int(image_frame_low and action_delta_low)
    if image_first_last_low is not None and action_delta_low is not None:
        group["low"]["image_first_last_and_action_delta_low"] += int(image_first_last_low and action_delta_low)


def summarize_metric(values):
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for percentile in PERCENTILES:
        summary[f"p{percentile}"] = float(np.percentile(array, percentile))
    return summary


def flatten_summary(group_name, group_summary):
    row = {
        "group": group_name,
        "windows": group_summary["windows"],
        "episodes": group_summary["episodes"],
        "roots": group_summary["roots"],
        "success_windows": group_summary["success_windows"],
        "failed_windows": group_summary["failed_windows"],
    }
    windows = max(group_summary["windows"], 1)
    for low_name, count in group_summary["low"].items():
        row[f"{low_name}_ratio"] = count / windows
        row[f"{low_name}_count"] = count
    for metric_name, metric_summary in group_summary["metrics"].items():
        for key, value in metric_summary.items():
            row[f"{metric_name}_{key}"] = value
    return row


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["group", "windows", "episodes", "roots", "success_windows", "failed_windows"]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    metric_names = selected_metric_names(args)
    dataset = WorldModelDataset(
        root=args.dataset_base_path,
        tasks=split_csv(args.tasks),
        cameras=(args.camera,),
        num_frames=args.num_frames,
        stride=args.stride,
        include_depth=False,
        include_camera_params=False,
        include_failed=args.include_failed,
        repeat=1,
    )
    print(
        f"Indexed {len(dataset.episodes)} episodes and {len(dataset.windows)} windows "
        f"from {len(dataset.roots)} roots."
    )

    starts_by_episode = episode_window_starts(dataset)
    starts_by_episode = sample_windows_by_group(
        dataset,
        starts_by_episode,
        args.group_by,
        args.max_windows_per_group,
        args.seed,
    )
    selected_window_count = sum(len(starts) for starts in starts_by_episode.values())
    print(f"Selected {selected_window_count} windows for statistics.")

    action_metadata_path = args.action_metadata_path or auto_action_metadata_path(args.dataset_base_path)
    action_stats = None
    if args.normalize_action and action_metadata_path:
        action_stats = load_action_stats(action_metadata_path, args.action_metadata_key, action_dim=14, eps=args.action_normalization_eps)
        print(f"Using action normalization stats from {action_metadata_path}.")
    elif args.normalize_action:
        print("Action metadata was not found; action metrics will use raw action values.")

    groups = defaultdict(lambda: empty_group(metric_names))
    window_rows = []
    processed_episodes = 0

    for episode_id in sorted(starts_by_episode):
        starts = starts_by_episode[episode_id]
        if len(starts) == 0:
            continue
        episode = dataset.episodes[episode_id]
        group_name = group_key_for_episode(dataset, episode, args.group_by)
        root_label = root_label_for_episode(dataset, episode.path)
        is_success = dataset._is_successful(episode.meta)

        frames = None
        actions = None
        if any(name in metric_names for name in IMAGE_METRICS):
            frames = load_episode_frames(dataset, episode, args.camera, args.image_size)
        if any(name in metric_names for name in ACTION_METRICS):
            actions = load_episode_actions(dataset, episode, action_stats, args.action_normalization_eps)

        group = groups[group_name]
        group["episodes"].add(episode.path)

        for start in starts:
            metrics = compute_window_metrics(frames, actions, start, args.num_frames, metric_names)
            group["root_windows"][root_label] += 1
            if is_success:
                group["success_windows"] += 1
            else:
                group["failed_windows"] += 1
            for name, value in metrics.items():
                group["metrics"][name].append(value)
            update_low_motion_counts(group, metrics, args)

            if args.window_csv:
                row = {
                    "group": group_name,
                    "task": episode.task,
                    "root": root_label,
                    "episode": episode.episode,
                    "start": start,
                    "success": int(is_success),
                }
                row.update(metrics)
                window_rows.append(row)

        processed_episodes += 1
        if args.progress_every > 0 and processed_episodes % args.progress_every == 0:
            print(f"Processed {processed_episodes}/{len(starts_by_episode)} selected episodes...")

    summaries = {}
    rows = []
    for group_name in sorted(groups):
        group = groups[group_name]
        windows = len(group["metrics"][metric_names[0]])
        group_summary = {
            "windows": windows,
            "episodes": len(group["episodes"]),
            "roots": ",".join(f"{root}:{count}" for root, count in sorted(group["root_windows"].items())),
            "success_windows": group["success_windows"],
            "failed_windows": group["failed_windows"],
            "low": dict(group["low"]),
            "metrics": {
                metric_name: summarize_metric(values)
                for metric_name, values in group["metrics"].items()
                if len(values) > 0
            },
        }
        summaries[group_name] = group_summary
        rows.append(flatten_summary(group_name, group_summary))

    write_csv(args.output_csv, rows)
    write_json(
        args.output_json,
        {
            "settings": {
                "dataset_base_path": args.dataset_base_path,
                "camera": args.camera,
                "num_frames": args.num_frames,
                "stride": args.stride,
                "group_by": args.group_by,
                "metrics": args.metrics,
                "image_size": args.image_size,
                "max_windows_per_group": args.max_windows_per_group,
                "normalize_action": args.normalize_action,
                "action_metadata_path": action_metadata_path,
                "thresholds": {
                    "image_frame_low_threshold": args.image_frame_low_threshold,
                    "image_first_last_low_threshold": args.image_first_last_low_threshold,
                    "action_l2_low_threshold": args.action_l2_low_threshold,
                    "action_delta_low_threshold": args.action_delta_low_threshold,
                },
            },
            "groups": summaries,
        },
    )
    if args.window_csv:
        write_csv(args.window_csv, window_rows)

    print(f"Wrote {len(rows)} group rows to {args.output_csv}.")
    if args.output_json:
        print(f"Wrote JSON summary to {args.output_json}.")
    if args.window_csv:
        print(f"Wrote {len(window_rows)} window rows to {args.window_csv}.")


if __name__ == "__main__":
    main()
