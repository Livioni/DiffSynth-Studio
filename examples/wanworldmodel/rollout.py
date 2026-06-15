import json
import os

import torch

try:
    from examples.wanworldmodel.action_utils import robot_action_to_tensor
    from examples.wanworldmodel.infer import (
        build_pipeline,
        none_if_empty,
        resize_pil_video,
        sanitize_name,
        save_action,
        save_pil_video,
        split_csv,
    )
except ModuleNotFoundError:
    from action_utils import robot_action_to_tensor
    from infer import (
        build_pipeline,
        none_if_empty,
        resize_pil_video,
        sanitize_name,
        save_action,
        save_pil_video,
        split_csv,
    )


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.abspath("models"))
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")


def build_rollout_dataset(args):
    from diffsynth.core import WorldModelDataset
    from diffsynth.core.data.operators import ImageCropAndResize

    height = None if args.height is None or args.height <= 0 else args.height
    width = None if args.width is None or args.width <= 0 else args.width
    frame_processor = ImageCropAndResize(
        height=height,
        width=width,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )
    dataset = WorldModelDataset(
        root=args.eval_dataset_base_path,
        tasks=split_csv(args.task),
        cameras=(args.camera,),
        num_frames=1,
        stride=1,
        include_depth=False,
        include_camera_params=False,
        include_failed=args.include_failed,
        repeat=1,
    )
    return dataset, frame_processor


def first_csv_value(value):
    values = split_csv(value)
    return None if not values else values[0]


def list_episodes(dataset):
    print("Available eval episodes:")
    for episode in dataset.episodes:
        print(f"  task={episode.task} episode={episode.episode} length={episode.length}")


def select_episode(dataset, task=None, episode_name=None):
    if len(dataset.episodes) == 0:
        raise ValueError("Eval dataset contains no valid episodes.")

    if task is None:
        task = dataset.episodes[0].task

    matches = [
        episode
        for episode in dataset.episodes
        if episode.task == task and (episode_name is None or episode.episode == episode_name)
    ]
    if len(matches) == 0:
        available = sorted({episode.task for episode in dataset.episodes})
        raise ValueError(f"No episode matches task={task!r}, episode={episode_name!r}. Available tasks: {available}")
    return matches[0]


def load_episode_video_and_action(dataset, episode, camera, frame_processor, start_frame=0):
    if start_frame < 0 or start_frame >= episode.length:
        raise ValueError(
            f"Invalid start_frame={start_frame} for {episode.task}/{episode.episode}. "
            f"Valid range is 0..{episode.length - 1}."
        )

    frame_indices = torch.arange(start_frame, episode.length, dtype=torch.long)
    video = dataset._load_rgb_window(episode.path, camera, frame_indices)
    video = [frame_processor(frame) for frame in video]
    robot = dataset._load_robot_window(episode.path, frame_indices)
    action = robot_action_to_tensor(robot)
    if action is None:
        raise ValueError(f"No robot action found in episode {episode.task}/{episode.episode}.")
    return video, action, frame_indices


def pad_action_window(action, start, length):
    end = min(start + length, action.shape[0])
    window = action[start:end]
    if window.shape[0] == length:
        return window
    if window.shape[0] == 0:
        window = action[-1:]
    pad = window[-1:].expand(length - window.shape[0], -1)
    return torch.cat([window, pad], dim=0)


def rollout_episode(args):
    dataset, frame_processor = build_rollout_dataset(args)
    if args.list_samples:
        list_episodes(dataset)
        return

    episode = select_episode(
        dataset,
        task=first_csv_value(none_if_empty(args.task)),
        episode_name=none_if_empty(args.episode),
    )
    gt_video, action, frame_indices = load_episode_video_and_action(
        dataset,
        episode,
        args.camera,
        frame_processor,
        start_frame=args.start_frame,
    )
    prompt = args.prompt if args.prompt is not None else episode.text_conditions[0]

    pipe, device = build_pipeline(args)
    target_height, target_width, rollout_window_frames = pipe.check_resize_height_width(
        gt_video[0].size[1],
        gt_video[0].size[0],
        args.num_frames,
        verbose=False,
    )
    if rollout_window_frames <= 1 and len(gt_video) > 1:
        raise ValueError(f"Rollout requires at least 2 frames per window, got {rollout_window_frames}.")

    gt_video = resize_pil_video(gt_video, target_width, target_height)
    rand_device = args.rand_device if args.rand_device is not None else device

    pred_video = []
    window_infos = []
    task_name = sanitize_name(episode.task)
    episode_name = sanitize_name(episode.episode)
    sample_output_dir = os.path.join(args.output_dir, f"{task_name}_{episode_name}")
    os.makedirs(sample_output_dir, exist_ok=True)
    stem = "_".join(
        [
            task_name,
            episode_name,
            f"rollout_start{int(args.start_frame)}",
            f"step_{int(args.num_inference_steps):04d}",
            f"seed{int(args.seed) if args.seed is not None else 'none'}",
        ]
    )
    window_dir = os.path.join(sample_output_dir, f"{stem}_windows")

    with torch.no_grad():
        while len(pred_video) < len(gt_video):
            local_start = 0 if len(pred_video) == 0 else len(pred_video) - 1
            remaining = len(gt_video) - local_start
            requested_frames = min(rollout_window_frames, remaining)
            _, _, target_window_frames = pipe.check_resize_height_width(
                target_height,
                target_width,
                requested_frames,
                verbose=False,
            )

            input_image = gt_video[0] if len(pred_video) == 0 else pred_video[-1]
            action_window = pad_action_window(action, local_start, target_window_frames)
            window_index = len(window_infos)
            window_seed = None if args.seed is None else int(args.seed) + window_index
            chunk_video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                seed=window_seed,
                rand_device=rand_device,
                height=target_height,
                width=target_width,
                num_frames=target_window_frames,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                denoising_strength=args.denoising_strength,
                tiled=args.tiled,
                tile_size=tuple(args.tile_size),
                tile_stride=tuple(args.tile_stride),
                action=action_window,
                output_type="quantized",
            )
            if len(chunk_video) == 0:
                raise ValueError(f"Rollout window {window_index} generated no frames.")

            if len(pred_video) == 0:
                if len(gt_video) > 1 and len(chunk_video) <= 1:
                    raise ValueError(
                        f"Rollout window {window_index} generated too few frames: {len(chunk_video)}."
                    )
                keep_count = min(len(chunk_video), len(gt_video))
                pred_video.extend(chunk_video[:keep_count])
            else:
                new_frame_budget = len(gt_video) - len(pred_video)
                if len(chunk_video) <= 1 and new_frame_budget > 0:
                    raise ValueError(
                        f"Rollout window {window_index} generated too few frames: {len(chunk_video)}."
                    )
                pred_video.extend(chunk_video[1:1 + new_frame_budget])

            kept_until = len(pred_video)
            window_info = {
                "window_index": int(window_index),
                "episode_start_frame": int(args.start_frame + local_start),
                "local_start_frame": int(local_start),
                "remaining_frames_before_window": int(remaining),
                "requested_frames": int(requested_frames),
                "generated_frames": int(len(chunk_video)),
                "kept_rollout_frames_after_window": int(kept_until),
                "seed": window_seed,
                "action_shape": [int(item) for item in action_window.shape],
            }
            window_infos.append(window_info)

            if args.save_windows:
                os.makedirs(window_dir, exist_ok=True)
                window_path = os.path.join(window_dir, f"window_{window_index:04d}.mp4")
                save_pil_video(chunk_video, window_path, fps=args.fps)
                window_info["path"] = window_path

    pred_video = pred_video[:len(gt_video)]
    if len(pred_video) != len(gt_video):
        raise ValueError(f"Rollout frame mismatch: pred={len(pred_video)}, gt={len(gt_video)}.")

    pred_path = os.path.join(sample_output_dir, f"{stem}_pred.mp4")
    gt_path = os.path.join(sample_output_dir, f"{stem}_gt.mp4")
    input_path = os.path.join(sample_output_dir, f"{stem}_input.png")
    action_path = os.path.join(sample_output_dir, f"{stem}_action.npy")
    metadata_path = os.path.join(sample_output_dir, f"{stem}.json")

    save_pil_video(pred_video, pred_path, fps=args.fps)
    if args.save_gt:
        save_pil_video(gt_video, gt_path, fps=args.fps)
    if args.save_input:
        gt_video[0].save(input_path)
    if args.save_action:
        save_action(action, action_path)

    metadata = {
        "task": episode.task,
        "episode": episode.episode,
        "episode_path": episode.path,
        "camera": args.camera,
        "prompt": prompt,
        "use_text_condition": bool(args.use_text_condition),
        "source_start_frame": int(args.start_frame),
        "episode_length": int(episode.length),
        "rollout_frame_indices": [int(item) for item in frame_indices.tolist()],
        "rollout_frame_count": int(len(pred_video)),
        "ground_truth_frame_count": int(len(gt_video)),
        "num_frames_requested_per_window": int(args.num_frames),
        "num_frames_generated_per_full_window": int(rollout_window_frames),
        "height": int(target_height),
        "width": int(target_width),
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "action_shape": [int(item) for item in action.shape],
        "checkpoint_path": args.checkpoint_path,
        "output_dir": sample_output_dir,
        "prediction_path": pred_path,
        "ground_truth_path": gt_path if args.save_gt else None,
        "input_image_path": input_path if args.save_input else None,
        "action_path": action_path if args.save_action else None,
        "windows": window_infos,
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved rollout video: {pred_path}")
    if args.save_gt:
        print(f"Saved ground-truth video: {gt_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Aligned frames: pred={len(pred_video)} gt={len(gt_video)}")


def build_parser():
    try:
        from examples.wanworldmodel.infer import build_parser as build_infer_parser
    except ModuleNotFoundError:
        from infer import build_parser as build_infer_parser

    parser = build_infer_parser()
    parser.description = "Roll out a complete WanWorldModel episode with its action trajectory."
    parser.set_defaults(output_dir="outputs/rollout")
    parser.add_argument(
        "--save_windows",
        default=False,
        action="store_true",
        help="Also save every generated rollout window before overlap trimming.",
    )

    for action in parser._actions:
        if action.dest == "start_frame":
            action.help = "Episode frame used as the rollout anchor. Defaults to 0 for the full episode."
        elif action.dest == "num_frames":
            action.help = "Frames generated per rollout window before overlapping the last frame into the next window."
        elif action.dest == "list_samples":
            action.help = "List task/episode lengths and exit."
    return parser


if __name__ == "__main__":
    rollout_episode(build_parser().parse_args())
