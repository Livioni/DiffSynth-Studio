import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


_FRAME_RE = re.compile(r"frame_(\d+)")


@dataclass(frozen=True)
class _EpisodeInfo:
    task: str
    episode: str
    path: str
    meta: dict
    text_conditions: tuple[str, ...]
    length: int


@dataclass(frozen=True)
class _WindowInfo:
    episode_id: int
    start: int


class WorldModelDataset(torch.utils.data.Dataset):
    """
    Dataset for RoboTwin ALOHA world-model episodes.

    Each item is a fixed-length window from one episode. RGB/depth/camera
    parameters and robot action/state values are aligned by frame index.
    """

    robot_files = {
        "left": {
            "action": {
                "arm_joint": "left_arm_joint_action.npy",
                "gripper": "left_gripper_action.npy",
            },
            "state": {
                "endpose": "left_endpose.npy",
                "gripper": "left_endpose_gripper.npy",
            },
        },
        "right": {
            "action": {
                "arm_joint": "right_arm_joint_action.npy",
                "gripper": "right_gripper_action.npy",
            },
            "state": {
                "endpose": "right_endpose.npy",
                "gripper": "right_endpose_gripper.npy",
            },
        },
    }

    rgb_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    depth_extensions = (".npy", ".png")

    def __init__(
        self,
        root="world_model_data/robotwin_aloha",
        tasks=None,
        cameras=("head_camera", "left_camera", "right_camera", "third_view"),
        num_frames=81,
        stride=None,
        include_depth=True,
        include_camera_params=True,
        include_failed=True,
        repeat=1,
        max_data_items=None,
        action_metadata_path=None,
        action_metadata_key="robot_statistics",
        action_normalization_eps=1e-6,
        action_normalization_mode="standard",
        filter_static_action_windows=False,
        static_action_eps=1e-8,
        action_delta_low_threshold=None,
        action_delta_low_weight=1.0,
    ):
        self.root = root
        self.roots = self._normalize_roots(root)
        self.tasks = self._normalize_names(tasks)
        self.cameras = tuple(self._normalize_names(cameras))
        self.num_frames = int(num_frames)
        self.stride = int(num_frames if stride is None else stride)
        self.include_depth = include_depth
        self.include_camera_params = include_camera_params
        self.include_failed = include_failed
        self.repeat = int(repeat)
        self.max_data_items = max_data_items
        self.action_metadata_path = action_metadata_path
        self.action_metadata_key = action_metadata_key
        self.action_normalization_eps = float(action_normalization_eps)
        self.action_normalization_mode = self._normalize_action_normalization_mode(action_normalization_mode)
        self.filter_static_action_windows = bool(filter_static_action_windows)
        self.static_action_eps = float(static_action_eps)
        self.action_delta_low_threshold = None if action_delta_low_threshold is None else float(action_delta_low_threshold)
        self.action_delta_low_weight = float(action_delta_low_weight)
        self.action_normalization_stats = self._load_action_normalization_stats(
            action_metadata_path,
            action_metadata_key,
        ) if self._uses_action_quality_control() and action_metadata_path is not None else None

        if self.num_frames <= 0:
            raise ValueError("num_frames must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if self.action_normalization_eps <= 0:
            raise ValueError("action_normalization_eps must be positive.")
        if self.static_action_eps < 0:
            raise ValueError("static_action_eps must be non-negative.")
        if self.action_delta_low_threshold is not None and self.action_delta_low_threshold < 0:
            raise ValueError("action_delta_low_threshold must be non-negative.")
        if not 0 < self.action_delta_low_weight <= 1:
            raise ValueError("action_delta_low_weight must be in (0, 1].")
        if self._uses_action_delta_weighting() and self.action_normalization_stats is None:
            raise ValueError("Action delta weighting requires action_metadata_path for normalized thresholds.")
        if len(self.cameras) == 0:
            raise ValueError("At least one camera must be selected.")
        missing_roots = [root for root in self.roots if not os.path.isdir(root)]
        if missing_roots:
            raise FileNotFoundError(f"Dataset root does not exist: {missing_roots}")

        self.episodes: list[_EpisodeInfo] = []
        self.windows: list[_WindowInfo] = []
        self._window_sample_weights = [] if self._uses_action_delta_weighting() else None
        self.action_window_quality_stats = {
            "enabled": self._uses_action_quality_control(),
            "filter_static_action_windows": self.filter_static_action_windows,
            "static_action_eps": self.static_action_eps,
            "action_delta_low_threshold": self.action_delta_low_threshold,
            "action_delta_low_weight": self.action_delta_low_weight,
            "total_windows_before_filter": 0,
            "filtered_static_windows": 0,
            "retained_windows": 0,
            "low_delta_weighted_windows": 0,
        }
        self._frame_file_cache = {}
        self._build_index()

    @staticmethod
    def _normalize_roots(root):
        if isinstance(root, str):
            roots = [name.strip() for name in root.split(",")]
        else:
            roots = [str(name).strip() for name in root]
        roots = tuple(name for name in roots if name)
        if len(roots) == 0:
            raise ValueError("At least one dataset root must be provided.")
        return roots

    @staticmethod
    def _normalize_names(names):
        if names is None:
            return None
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        return tuple(name for name in names if name)

    @staticmethod
    def _normalize_action_normalization_mode(mode):
        mode = "standard" if mode is None else str(mode).strip().lower().replace("-", "_")
        if mode not in ("standard", "scale_only"):
            raise ValueError(f"action_normalization_mode must be `standard` or `scale_only`, got {mode!r}.")
        return mode

    def _uses_action_delta_weighting(self):
        return self.action_delta_low_threshold is not None and self.action_delta_low_weight != 1.0

    def _uses_action_quality_control(self):
        return self.filter_static_action_windows or self._uses_action_delta_weighting()

    @staticmethod
    def _load_action_normalization_stats(metadata_path, metadata_key):
        if metadata_path is None:
            return None
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Action metadata file does not exist: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if metadata_key in metadata:
            metadata = metadata[metadata_key]
        elif "arms" not in metadata:
            raise KeyError(f"Action metadata `{metadata_path}` does not contain `{metadata_key}` or `arms`.")

        mean = []
        std = []
        for arm in ("left", "right"):
            try:
                action_stats = metadata["arms"][arm]["action"]
            except KeyError as error:
                raise KeyError(f"Action metadata is missing `{arm}.action` statistics.") from error
            mean.extend(float(value) for value in action_stats["mean"])
            std.extend(float(value) for value in action_stats["std"])

        if len(mean) != len(std):
            raise ValueError(f"Action metadata mean/std dimensions do not match: {len(mean)} vs {len(std)}.")
        return (
            np.asarray(mean, dtype=np.float32),
            np.asarray(std, dtype=np.float32),
        )

    @staticmethod
    def _action_value_to_array(value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        elif value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        return value

    def _load_action_array(self, episode_path, length):
        paths = self._robot_paths(episode_path)
        if paths is None:
            raise FileNotFoundError(f"Missing robot_data files: {episode_path}")

        pieces = []
        for arm in ("left", "right"):
            for key in ("arm_joint", "gripper"):
                array = np.load(paths[arm]["action"][key], mmap_mode="r")
                pieces.append(self._action_value_to_array(array[:length]))
        action = np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)

        if self.action_normalization_stats is None:
            return action
        mean, std = self.action_normalization_stats
        if action.shape[-1] != mean.shape[0]:
            raise ValueError(f"Action metadata dimension {mean.shape[0]} does not match action dimension {action.shape[-1]}.")
        std = np.maximum(std, self.action_normalization_eps)
        if self.action_normalization_mode == "standard":
            return (action - mean.reshape(1, -1)) / std.reshape(1, -1)
        return action / std.reshape(1, -1)

    @staticmethod
    def _window_action_delta_stats(action, start, num_frames):
        window = action[start:start + num_frames]
        delta = np.linalg.norm(window[1:] - window[:-1], axis=-1)
        return float(delta.mean()), float(delta.max())

    @property
    def sample_weights(self):
        if self._window_sample_weights is None:
            return None
        if len(self._window_sample_weights) == 0:
            return []
        weights = self._window_sample_weights * max(self.repeat, 0)
        if self.max_data_items is not None:
            weights = weights[:int(self.max_data_items)]
        return weights

    @staticmethod
    def _episode_sort_key(name):
        if name.startswith("episode") and name[len("episode"):].isdigit():
            return 0, int(name[len("episode"):])
        return 1, name

    @staticmethod
    def _frame_sort_key(path):
        match = _FRAME_RE.search(os.path.basename(path))
        if match is not None:
            return 0, int(match.group(1))
        return 1, os.path.basename(path)

    @staticmethod
    def _load_json(path):
        if not os.path.isfile(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _is_successful(meta):
        if "success" in meta and not bool(meta["success"]):
            return False
        if "result" in meta and str(meta["result"]).lower() != "success":
            return False
        return True

    @staticmethod
    def _prompt_from_task(task):
        return task.replace("_", " ")

    @staticmethod
    def _text_from_instruction_item(item):
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("instruction", "text", "prompt", "caption"):
                value = item.get(key)
                if isinstance(value, str):
                    return value.strip()
        return None

    def _instruction_json_path(self, episode_path, episode):
        candidate_paths = [
            os.path.join(episode_path, f"{episode}.json"),
            os.path.join(episode_path, "instruction.json"),
            os.path.join(episode_path, "instructions.json"),
        ]
        for path in candidate_paths:
            if os.path.isfile(path):
                return path

        for name in sorted(os.listdir(episode_path)):
            if name.endswith(".json") and name != "meta.json":
                return os.path.join(episode_path, name)
        return None

    def _load_text_conditions(self, episode_path, episode, task):
        instruction_path = self._instruction_json_path(episode_path, episode)
        if instruction_path is None:
            return (self._prompt_from_task(task),)

        instruction = self._load_json(instruction_path)
        seen = instruction.get("seen", []) if isinstance(instruction, dict) else []
        if isinstance(seen, str):
            seen = seen.splitlines()

        text_conditions = []
        for item in seen:
            text = self._text_from_instruction_item(item)
            if text:
                text_conditions.append(text)
        if len(text_conditions) == 0:
            text_conditions.append(self._prompt_from_task(task))
        return tuple(text_conditions)

    @staticmethod
    def _sample_text_condition(text_conditions):
        if len(text_conditions) == 1:
            return text_conditions[0]
        index = int(torch.randint(len(text_conditions), (1,)).item())
        return text_conditions[index]

    def _list_task_names(self, root):
        if self.tasks is not None:
            missing = [task for task in self.tasks if not os.path.isdir(os.path.join(root, task))]
            if missing:
                raise FileNotFoundError(f"Task folders do not exist under {root}: {missing}")
            return list(self.tasks)
        tasks = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path) and not name.startswith("_"):
                tasks.append(name)
        return tasks

    def _list_episode_names(self, task_path):
        names = []
        for name in os.listdir(task_path):
            path = os.path.join(task_path, name)
            if os.path.isdir(path) and name.startswith("episode"):
                names.append(name)
        return sorted(names, key=self._episode_sort_key)

    def _list_frame_files(self, folder, extensions):
        if not os.path.isdir(folder):
            return []
        files_by_frame = {}
        extension_rank = {ext: rank for rank, ext in enumerate(extensions)}
        for name in os.listdir(folder):
            ext = os.path.splitext(name)[1].lower()
            if ext not in extension_rank:
                continue
            path = os.path.join(folder, name)
            match = _FRAME_RE.search(name)
            frame_key = int(match.group(1)) if match is not None else name
            current = files_by_frame.get(frame_key)
            if current is None or extension_rank[ext] < extension_rank[os.path.splitext(current)[1].lower()]:
                files_by_frame[frame_key] = path
        return sorted(files_by_frame.values(), key=self._frame_sort_key)

    def _get_frame_files(self, folder, extensions):
        key = (folder, tuple(extensions))
        files = self._frame_file_cache.get(key)
        if files is None:
            files = self._list_frame_files(folder, extensions)
            self._frame_file_cache[key] = files
        return files

    def _robot_paths(self, episode_path):
        robot_root = os.path.join(episode_path, "robot_data")
        paths = {}
        for arm, arm_config in self.robot_files.items():
            paths[arm] = {}
            for group, group_config in arm_config.items():
                paths[arm][group] = {}
                for key, file_name in group_config.items():
                    path = os.path.join(robot_root, file_name)
                    if not os.path.isfile(path):
                        return None
                    paths[arm][group][key] = path
        return paths

    def _robot_length(self, episode_path):
        paths = self._robot_paths(episode_path)
        if paths is None:
            return None
        lengths = []
        for arm in paths.values():
            for group in arm.values():
                for path in group.values():
                    array = np.load(path, mmap_mode="r")
                    lengths.append(int(array.shape[0]))
        return min(lengths) if lengths else None

    def _camera_length(self, episode_path, camera):
        camera_root = os.path.join(episode_path, "camera_data")
        lengths = []

        rgb_folder = os.path.join(camera_root, "images", camera)
        rgb_files = self._get_frame_files(rgb_folder, self.rgb_extensions)
        if not rgb_files:
            return None
        lengths.append(len(rgb_files))

        if self.include_depth:
            depth_folder = os.path.join(camera_root, "depths", camera)
            depth_files = self._get_frame_files(depth_folder, self.depth_extensions)
            if not depth_files:
                return None
            lengths.append(len(depth_files))

        if self.include_camera_params:
            intrinsic_path = os.path.join(camera_root, "intrinsics", camera, "intrinsic.npy")
            if not os.path.isfile(intrinsic_path):
                return None
            extrinsic_folder = os.path.join(camera_root, "extrinsics", camera)
            extrinsic_files = self._get_frame_files(extrinsic_folder, (".npy",))
            if not extrinsic_files:
                return None
            lengths.append(len(extrinsic_files))

        return min(lengths)

    def _episode_length(self, episode_path):
        lengths = []
        for camera in self.cameras:
            length = self._camera_length(episode_path, camera)
            if length is None:
                return None
            lengths.append(length)
        robot_length = self._robot_length(episode_path)
        if robot_length is None:
            return None
        lengths.append(robot_length)
        return min(lengths)

    def _build_index(self):
        for root in self.roots:
            for task in self._list_task_names(root):
                task_path = os.path.join(root, task)
                for episode in self._list_episode_names(task_path):
                    episode_path = os.path.join(task_path, episode)
                    meta = self._load_json(os.path.join(episode_path, "meta.json"))
                    if not self.include_failed and not self._is_successful(meta):
                        continue
                    length = self._episode_length(episode_path)
                    if length is None or length < self.num_frames:
                        continue
                    text_conditions = self._load_text_conditions(episode_path, episode, task)
                    episode_id = len(self.episodes)
                    self.episodes.append(_EpisodeInfo(task, episode, episode_path, meta, text_conditions, length))
                    action = self._load_action_array(episode_path, length) if self._uses_action_quality_control() else None
                    for start in range(0, length - self.num_frames + 1, self.stride):
                        weight = 1.0
                        if action is not None:
                            delta_mean, delta_max = self._window_action_delta_stats(action, start, self.num_frames)
                            self.action_window_quality_stats["total_windows_before_filter"] += 1
                            if self.filter_static_action_windows and delta_max <= self.static_action_eps:
                                self.action_window_quality_stats["filtered_static_windows"] += 1
                                continue
                            if self._uses_action_delta_weighting() and delta_mean <= self.action_delta_low_threshold:
                                weight = self.action_delta_low_weight
                                self.action_window_quality_stats["low_delta_weighted_windows"] += 1
                        self.windows.append(_WindowInfo(episode_id, start))
                        if self._window_sample_weights is not None:
                            self._window_sample_weights.append(weight)
                        self.action_window_quality_stats["retained_windows"] += 1

    def _load_rgb_window(self, episode_path, camera, frame_indices):
        folder = os.path.join(episode_path, "camera_data", "images", camera)
        files = self._get_frame_files(folder, self.rgb_extensions)
        frames = []
        for frame_index in frame_indices:
            with Image.open(files[int(frame_index)]) as image:
                frames.append(image.convert("RGB").copy())
        return frames

    def _load_depth_window(self, episode_path, camera, frame_indices):
        folder = os.path.join(episode_path, "camera_data", "depths", camera)
        files = self._get_frame_files(folder, self.depth_extensions)
        depth_frames = []
        for frame_index in frame_indices:
            path = files[int(frame_index)]
            if path.lower().endswith(".npy"):
                depth = np.load(path)
            else:
                with Image.open(path) as image:
                    depth = np.array(image)
            depth_frames.append(torch.from_numpy(np.asarray(depth)).to(dtype=torch.float32))
        return torch.stack(depth_frames, dim=0)

    def _load_camera_params_window(self, episode_path, camera, frame_indices):
        camera_root = os.path.join(episode_path, "camera_data")
        intrinsic_path = os.path.join(camera_root, "intrinsics", camera, "intrinsic.npy")
        intrinsic = torch.from_numpy(np.load(intrinsic_path)).to(dtype=torch.float32)

        extrinsic_folder = os.path.join(camera_root, "extrinsics", camera)
        extrinsic_files = self._get_frame_files(extrinsic_folder, (".npy",))
        extrinsics = []
        for frame_index in frame_indices:
            extrinsic = np.load(extrinsic_files[int(frame_index)])
            extrinsics.append(torch.from_numpy(extrinsic).to(dtype=torch.float32))
        return intrinsic, torch.stack(extrinsics, dim=0)

    def _load_robot_window(self, episode_path, frame_indices):
        paths = self._robot_paths(episode_path)
        robot = {}
        start = int(frame_indices[0])
        end = int(frame_indices[-1]) + 1
        for arm, arm_paths in paths.items():
            robot[arm] = {}
            for group, group_paths in arm_paths.items():
                robot[arm][group] = {}
                for key, path in group_paths.items():
                    array = np.load(path, mmap_mode="r")
                    values = np.asarray(array[start:end])
                    robot[arm][group][key] = torch.from_numpy(values.copy()).to(dtype=torch.float32)
        return robot

    def __getitem__(self, index):
        if len(self.windows) == 0:
            raise IndexError("WorldModelDataset contains no valid windows.")
        window = self.windows[index % len(self.windows)]
        episode = self.episodes[window.episode_id]
        frame_indices = torch.arange(window.start, window.start + self.num_frames, dtype=torch.long)
        prompt = self._sample_text_condition(episode.text_conditions)

        cameras = {}
        for camera in self.cameras:
            camera_data = {
                "rgb": self._load_rgb_window(episode.path, camera, frame_indices),
            }
            if self.include_depth:
                camera_data["depth"] = self._load_depth_window(episode.path, camera, frame_indices)
            if self.include_camera_params:
                intrinsics, extrinsics = self._load_camera_params_window(episode.path, camera, frame_indices)
                camera_data["intrinsics"] = intrinsics
                camera_data["extrinsics"] = extrinsics
            cameras[camera] = camera_data

        return {
            "task": episode.task,
            "episode": episode.episode,
            "episode_path": episode.path,
            "prompt": prompt,
            "frame_indices": frame_indices,
            "cameras": cameras,
            "robot": self._load_robot_window(episode.path, frame_indices),
            "meta": episode.meta,
        }

    def __len__(self):
        length = len(self.windows) * self.repeat
        if self.max_data_items is not None:
            length = min(length, int(self.max_data_items))
        return length
