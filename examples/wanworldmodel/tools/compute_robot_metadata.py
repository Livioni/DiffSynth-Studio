#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROBOT_FILES = {
    "left": {
        "action": (
            ("arm_joint", "left_arm_joint_action.npy"),
            ("gripper", "left_gripper_action.npy"),
        ),
        "state": (
            ("endpose", "left_endpose.npy"),
            ("gripper", "left_endpose_gripper.npy"),
        ),
    },
    "right": {
        "action": (
            ("arm_joint", "right_arm_joint_action.npy"),
            ("gripper", "right_gripper_action.npy"),
        ),
        "state": (
            ("endpose", "right_endpose.npy"),
            ("gripper", "right_endpose_gripper.npy"),
        ),
    },
}


@dataclass
class RunningStats:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        if values.shape[0] == 0:
            return

        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        if self.mean.shape != batch_mean.shape:
            raise ValueError(f"Feature dimension changed from {self.mean.shape} to {batch_mean.shape}.")

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total_count
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * batch_count / total_count
        self.count = total_count

    @property
    def feature_dim(self):
        return 0 if self.mean is None else int(self.mean.shape[0])

    def to_metadata(self):
        if self.count == 0:
            return {
                "count": 0,
                "feature_dim": 0,
                "mean": [],
                "std": [],
            }
        variance = self.m2 / self.count
        return {
            "count": self.count,
            "feature_dim": self.feature_dim,
            "mean": self.mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        }


def episode_sort_key(path):
    name = path.name
    prefix = "episode"
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return 0, int(name[len(prefix):])
    return 1, name


def parse_csv(value):
    if value is None:
        return None
    values = [item.strip() for item in value.split(",")]
    return tuple(item for item in values if item)


def iter_episode_dirs(root, tasks=None):
    task_names = tasks
    if task_names is None:
        task_names = tuple(
            path.name
            for path in sorted(root.iterdir())
            if path.is_dir() and not path.name.startswith("_")
        )

    for task in task_names:
        task_path = root / task
        if not task_path.is_dir():
            raise FileNotFoundError(f"Task folder does not exist: {task_path}")
        episode_paths = sorted(
            (path for path in task_path.iterdir() if path.is_dir() and path.name.startswith("episode")),
            key=episode_sort_key,
        )
        for episode_path in episode_paths:
            yield task, episode_path.name, episode_path


def load_robot_array(path):
    array = np.load(path, mmap_mode="r")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"Robot array must include a frame dimension: {path}")
    if array.ndim == 1:
        return array[:, None]
    return array.reshape(array.shape[0], -1)


def init_stats():
    group_stats = {}
    component_stats = {}
    field_dims = {}
    for arm, arm_config in ROBOT_FILES.items():
        group_stats[arm] = {}
        component_stats[arm] = {}
        field_dims[arm] = {}
        for group, group_config in arm_config.items():
            group_stats[arm][group] = RunningStats()
            component_stats[arm][group] = {name: RunningStats() for name, _ in group_config}
            field_dims[arm][group] = {}
    return group_stats, component_stats, field_dims


def update_field_dim(field_dims, arm, group, name, dim, path):
    previous_dim = field_dims[arm][group].get(name)
    if previous_dim is None:
        field_dims[arm][group][name] = dim
    elif previous_dim != dim:
        raise ValueError(f"Feature dimension for {arm}.{group}.{name} changed from {previous_dim} to {dim}: {path}")


def compute_statistics(root, tasks=None, strict=False):
    group_stats, component_stats, field_dims = init_stats()
    summary = {
        "episode_count": 0,
        "group_vector_episode_count": {
            arm: {group: 0 for group in arm_config}
            for arm, arm_config in ROBOT_FILES.items()
        },
        "missing_files": [],
        "length_mismatches": [],
    }

    for task, episode, episode_path in iter_episode_dirs(root, tasks=tasks):
        summary["episode_count"] += 1
        robot_root = episode_path / "robot_data"
        for arm, arm_config in ROBOT_FILES.items():
            for group, group_config in arm_config.items():
                pieces = []
                missing_group_file = False
                for name, file_name in group_config:
                    path = robot_root / file_name
                    if not path.is_file():
                        record = {"task": task, "episode": episode, "path": str(path)}
                        summary["missing_files"].append(record)
                        missing_group_file = True
                        if strict:
                            raise FileNotFoundError(f"Missing robot data file: {path}")
                        continue

                    values = load_robot_array(path)
                    update_field_dim(field_dims, arm, group, name, int(values.shape[1]), path)
                    component_stats[arm][group][name].update(values)
                    pieces.append((name, values))

                if missing_group_file:
                    continue

                lengths = [values.shape[0] for _, values in pieces]
                min_length = min(lengths)
                if any(length != min_length for length in lengths):
                    summary["length_mismatches"].append(
                        {
                            "task": task,
                            "episode": episode,
                            "arm": arm,
                            "group": group,
                            "lengths": {name: int(values.shape[0]) for name, values in pieces},
                            "used_length": int(min_length),
                        }
                    )
                    if strict:
                        raise ValueError(f"Length mismatch in {task}/{episode} {arm}.{group}: {lengths}")

                vector = np.concatenate([values[:min_length] for _, values in pieces], axis=1)
                group_stats[arm][group].update(vector)
                summary["group_vector_episode_count"][arm][group] += 1

    return group_stats, component_stats, field_dims, summary


def fields_metadata(arm_config, field_dims):
    fields = []
    start = 0
    for name, _ in arm_config:
        dim = int(field_dims.get(name, 0))
        end = start + dim
        fields.append({"name": name, "start": start, "end": end, "dim": dim})
        start = end
    return fields


def build_metadata(root, group_stats, component_stats, field_dims, summary):
    arms = {}
    for arm, arm_config in ROBOT_FILES.items():
        arms[arm] = {}
        for group, group_config in arm_config.items():
            metadata = group_stats[arm][group].to_metadata()
            metadata["fields"] = fields_metadata(group_config, field_dims[arm][group])
            metadata["components"] = {
                name: component_stats[arm][group][name].to_metadata()
                for name, _ in group_config
            }
            arms[arm][group] = metadata

    return {
        "dataset_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "std_definition": "population",
        **summary,
        "arms": arms,
    }


def load_existing_metadata(path, overwrite_non_object=False):
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        metadata = json.load(f)
    if isinstance(metadata, dict):
        return metadata
    if overwrite_non_object:
        return {}
    raise ValueError(
        f"Existing metadata is not a JSON object: {path}. "
        "Pass --overwrite_non_object to replace it."
    )


def write_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std metadata for RoboTwin ALOHA robot state and action arrays."
    )
    parser.add_argument("--root", type=Path, default=Path("world_model_data/robotwin_aloha"))
    parser.add_argument("--tasks", type=parse_csv, default=None, help="Comma-separated task folders. Defaults to all tasks.")
    parser.add_argument("--output_path", type=Path, default=None, help="Defaults to <root>/metadata.json.")
    parser.add_argument("--metadata_key", type=str, default="robot_statistics")
    parser.add_argument("--strict", action="store_true", help="Fail on missing files or per-group frame length mismatches.")
    parser.add_argument(
        "--overwrite_non_object",
        action="store_true",
        help="Replace an existing metadata file if it is not a JSON object.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    output_path = args.output_path or root / "metadata.json"
    group_stats, component_stats, field_dims, summary = compute_statistics(root, tasks=args.tasks, strict=args.strict)
    robot_metadata = build_metadata(root, group_stats, component_stats, field_dims, summary)

    metadata = load_existing_metadata(output_path, overwrite_non_object=args.overwrite_non_object)
    metadata[args.metadata_key] = robot_metadata
    write_metadata(output_path, metadata)

    print(f"Wrote {args.metadata_key} for {summary['episode_count']} episodes to {output_path}")
    if summary["missing_files"]:
        print(f"Missing robot files: {len(summary['missing_files'])}")
    if summary["length_mismatches"]:
        print(f"Length mismatches: {len(summary['length_mismatches'])}")


if __name__ == "__main__":
    main()
