import argparse
import glob
import json
import os
import re

try:
    from examples.wanworldmodel.action_utils import robot_action_to_tensor
except ModuleNotFoundError:
    from action_utils import robot_action_to_tensor


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.abspath("models"))
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")


def split_csv(value):
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",")]
    return tuple(item for item in values if item)


def none_if_empty(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def resolve_path_or_glob(path_or_pattern, required=True):
    path_or_pattern = os.path.expanduser(path_or_pattern)
    has_glob = any(char in path_or_pattern for char in "*?[")
    if has_glob:
        paths = sorted(glob.glob(path_or_pattern))
        if required and len(paths) == 0:
            raise FileNotFoundError(f"No files match pattern: {path_or_pattern}")
        return paths
    if required and not os.path.exists(path_or_pattern):
        raise FileNotFoundError(f"Path does not exist: {path_or_pattern}")
    return path_or_pattern


def sanitize_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def parse_dtype(name):
    import torch

    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def infer_action_embedder_config(checkpoint_path):
    from safetensors.torch import safe_open

    candidate_keys = (
        "pipe.action_embedder.mlp.0.weight",
        "action_embedder.mlp.0.weight",
    )
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for key in candidate_keys:
            if key in handle.keys():
                hidden_dim, action_dim = handle.get_slice(key).get_shape()
                return int(action_dim), int(hidden_dim)
    return None, None


def save_pil_video(video, path, fps=4):
    import imageio.v2 as imageio
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video]
    imageio.mimsave(path, frames, fps=fps)


def save_action(action, path):
    import numpy as np
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    np.save(path, np.asarray(action, dtype=np.float32))


def resize_pil_video(video, width, height):
    from PIL import Image

    if len(video) == 0:
        return video
    target_size = (int(width), int(height))
    if all(frame.size == target_size for frame in video):
        return video
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return [frame.resize(target_size, resample=resample) for frame in video]


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


def build_history_frame_segments(history_end, history_latent_count, history_stride):
    import torch

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
    import torch

    if len(segments) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(segments, dim=0)


def load_history_inputs(dataset, frame_processor, data, camera, history_frames, history_stride=4):
    history_frames = validate_world_model_history_frames(history_frames)
    history_stride = validate_world_model_history_stride(history_stride)
    if history_frames == 0:
        return [], None, None, 0, []

    import torch

    history_end = int(data["frame_indices"][0])
    history_latent_count = history_frames // 4
    if history_stride == 4:
        history_start = max(0, history_end - history_frames)
        history_indices = torch.arange(history_start, history_end, dtype=torch.long)
    else:
        history_segments = build_history_frame_segments(history_end, history_latent_count, history_stride)
        history_indices = flatten_history_frame_segments(history_segments)
    if len(history_indices) == 0:
        return [], None, None, history_latent_count, []

    if history_stride == 4:
        history_video = dataset._load_rgb_window(data["episode_path"], camera, history_indices)
        history_video = [frame_processor(frame) for frame in history_video]
        history_action = robot_action_to_tensor(dataset._load_robot_window(data["episode_path"], history_indices))
        history_video_segments = None
    else:
        history_video_segments = []
        history_action_segments = []
        for segment_indices in history_segments:
            history_video_segment = dataset._load_rgb_window(data["episode_path"], camera, segment_indices)
            history_video_segment = [frame_processor(frame) for frame in history_video_segment]
            history_video_segments.append(history_video_segment)
            history_action = robot_action_to_tensor(dataset._load_robot_window(data["episode_path"], segment_indices))
            if history_action is not None:
                history_action_segments.append(history_action)
        history_video = [frame for segment in history_video_segments for frame in segment]
        history_action = torch.cat(history_action_segments, dim=0) if len(history_action_segments) == len(history_video_segments) else None
    return history_video, history_video_segments, history_action, history_latent_count, [int(item) for item in history_indices.tolist()]


def build_dataset(args):
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
        num_frames=args.num_frames,
        stride=1,
        include_depth=False,
        include_camera_params=False,
        include_failed=args.include_failed,
        repeat=1,
    )
    return dataset, frame_processor


def list_samples(dataset):
    print("Available eval samples:")
    for episode_id, episode in enumerate(dataset.episodes):
        starts = [window.start for window in dataset.windows if window.episode_id == episode_id]
        if len(starts) == 0:
            continue
        print(
            f"  task={episode.task} episode={episode.episode} "
            f"length={episode.length} start_range={min(starts)}..{max(starts)}"
        )


def select_window(dataset, task=None, episode_name=None, start_frame=0):
    if len(dataset.episodes) == 0:
        raise ValueError("Eval dataset contains no valid episodes.")

    if task is None:
        task = dataset.episodes[0].task

    matching_episode_ids = [
        episode_id
        for episode_id, episode in enumerate(dataset.episodes)
        if episode.task == task and (episode_name is None or episode.episode == episode_name)
    ]
    if len(matching_episode_ids) == 0:
        available = sorted({episode.task for episode in dataset.episodes})
        raise ValueError(f"No episode matches task={task!r}, episode={episode_name!r}. Available tasks: {available}")

    episode_id = matching_episode_ids[0]
    episode = dataset.episodes[episode_id]
    if episode_name is None:
        episode_name = episode.episode

    if start_frame < 0 or start_frame + dataset.num_frames > episode.length:
        max_start = episode.length - dataset.num_frames
        raise ValueError(
            f"Invalid start_frame={start_frame} for {task}/{episode_name}. "
            f"Valid range is 0..{max_start} with num_frames={dataset.num_frames}."
        )

    for index, window in enumerate(dataset.windows):
        if window.episode_id == episode_id and window.start == start_frame:
            return index, episode
    raise ValueError(f"No indexed window found for {task}/{episode_name} start_frame={start_frame}.")


def make_model_configs(args):
    from diffsynth.pipelines.wan_world_model import ModelConfig

    model_configs = [
        ModelConfig(path=resolve_path_or_glob(args.dit_path)),
        ModelConfig(path=resolve_path_or_glob(args.vae_path)),
    ]
    if args.use_text_condition:
        model_configs.insert(0, ModelConfig(path=resolve_path_or_glob(args.text_encoder_path)))
    return model_configs


def load_finetuned_checkpoint(pipe, checkpoint_path, strict_checkpoint=False):
    from diffsynth.core import load_state_dict

    state_dict = load_state_dict(checkpoint_path, torch_dtype=pipe.torch_dtype, device="cpu")
    dit_state_dict = {}
    action_state_dict = {}
    other_keys = []

    for key, value in state_dict.items():
        if key.startswith("pipe.dit."):
            dit_state_dict[key[len("pipe.dit."):]] = value
        elif key.startswith("dit."):
            dit_state_dict[key[len("dit."):]] = value
        elif key.startswith("pipe.action_embedder."):
            action_state_dict[key[len("pipe.action_embedder."):]] = value
        elif key.startswith("action_embedder."):
            action_state_dict[key[len("action_embedder."):]] = value
        elif key.startswith("pipe."):
            other_keys.append(key)
        else:
            dit_state_dict[key] = value

    missing_dit, unexpected_dit = pipe.dit.load_state_dict(dit_state_dict, strict=False)
    print(
        f"Loaded DiT checkpoint tensors={len(dit_state_dict)} "
        f"missing={len(missing_dit)} unexpected={len(unexpected_dit)}"
    )
    if len(missing_dit) > 0:
        print("  DiT missing examples:", list(missing_dit)[:8])
    if len(unexpected_dit) > 0:
        print("  DiT unexpected examples:", list(unexpected_dit)[:8])

    missing_action = []
    unexpected_action = []
    if len(action_state_dict) > 0:
        if pipe.action_embedder is None:
            raise ValueError("Checkpoint contains action_embedder tensors, but pipeline action conditioning is disabled.")
        missing_action, unexpected_action = pipe.action_embedder.load_state_dict(action_state_dict, strict=False)
        print(
            f"Loaded action checkpoint tensors={len(action_state_dict)} "
            f"missing={len(missing_action)} unexpected={len(unexpected_action)}"
        )
        if len(missing_action) > 0:
            print("  Action missing examples:", list(missing_action)[:8])
        if len(unexpected_action) > 0:
            print("  Action unexpected examples:", list(unexpected_action)[:8])

    if len(other_keys) > 0:
        print("Ignored non-DiT/non-action checkpoint keys:", other_keys[:8])

    has_errors = len(unexpected_dit) > 0 or len(unexpected_action) > 0
    if strict_checkpoint:
        has_errors = has_errors or len(missing_dit) > 0 or len(missing_action) > 0
    if has_errors:
        raise ValueError(f"Checkpoint did not load cleanly: {checkpoint_path}")


def build_pipeline(args):
    import torch
    from diffsynth.pipelines.wan_world_model import ModelConfig, WanWorldModelPipeline

    checkpoint_action_dim, checkpoint_hidden_dim = infer_action_embedder_config(args.checkpoint_path)
    action_dim = args.action_dim if args.action_dim is not None else checkpoint_action_dim
    action_hidden_dim = (
        args.action_embedder_hidden_dim
        if args.action_embedder_hidden_dim is not None
        else checkpoint_hidden_dim
    )

    if args.action_injection_method != "none" and action_dim is None:
        raise ValueError("Could not infer action_dim from checkpoint. Pass --action_dim explicitly.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = parse_dtype(args.torch_dtype)

    action_metadata_path = none_if_empty(args.action_metadata_path)
    if action_metadata_path is not None and not os.path.isfile(action_metadata_path):
        raise FileNotFoundError(f"Action metadata file does not exist: {action_metadata_path}")

    pipe = WanWorldModelPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=make_model_configs(args),
        tokenizer_config=ModelConfig(path=args.tokenizer_path),
        redirect_common_files=False,
        action_dim=action_dim,
        action_embedder_hidden_dim=action_hidden_dim,
        action_injection_method=args.action_injection_method,
        action_metadata_path=action_metadata_path,
        action_metadata_key=args.action_metadata_key,
        action_normalization_eps=args.action_normalization_eps,
        action_normalization_mode=args.action_normalization_mode,
        use_text_condition=args.use_text_condition,
        text_context_length=args.text_context_length,
    )
    load_finetuned_checkpoint(pipe, args.checkpoint_path, strict_checkpoint=args.strict_checkpoint)
    pipe.eval()
    return pipe, device


def run(args):
    import torch

    dataset, frame_processor = build_dataset(args)
    if args.list_samples:
        list_samples(dataset)
        return

    selected_index, selected_episode = select_window(
        dataset,
        task=none_if_empty(args.task),
        episode_name=none_if_empty(args.episode),
        start_frame=args.start_frame,
    )
    data = dataset[selected_index]
    video = [frame_processor(frame) for frame in data["cameras"][args.camera]["rgb"]]
    action = robot_action_to_tensor(data["robot"])
    if action is None:
        raise ValueError(f"No robot action found in sample {data['task']}/{data['episode']}.")
    history_frames = validate_world_model_history_frames(args.world_model_history_frames if args.history else 0)
    history_stride = validate_world_model_history_stride(args.world_model_history_stride)
    history_video, history_video_segments, history_action, history_latent_count, history_frame_indices = load_history_inputs(
        dataset,
        frame_processor,
        data,
        args.camera,
        history_frames,
        history_stride,
    )

    prompt = args.prompt if args.prompt is not None else selected_episode.text_conditions[0]
    pipe, device = build_pipeline(args)
    target_height, target_width, target_num_frames = pipe.check_resize_height_width(
        video[0].size[1],
        video[0].size[0],
        len(video),
        verbose=False,
    )
    gt_video = resize_pil_video(video, target_width, target_height)
    rand_device = args.rand_device if args.rand_device is not None else device

    with torch.no_grad():
        pred_video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            input_image=gt_video[0],
            seed=args.seed,
            rand_device=rand_device,
            height=target_height,
            width=target_width,
            num_frames=target_num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            denoising_strength=args.denoising_strength,
            tiled=args.tiled,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
            action=action,
            history_video=history_video,
            history_video_segments=history_video_segments,
            history_action=history_action,
            history_latent_count=history_latent_count,
            output_type="quantized",
        )

    task_name = sanitize_name(data["task"])
    episode_name = sanitize_name(data["episode"])
    sample_output_dir = os.path.join(args.output_dir, f"{task_name}_{episode_name}")
    os.makedirs(sample_output_dir, exist_ok=True)
    stem = "_".join(
        [
            task_name,
            episode_name,
            f"start{int(args.start_frame)}",
            f"history{int(history_frames)}",
            f"step_{int(args.num_inference_steps):04d}",
            f"seed{int(args.seed) if args.seed is not None else 'none'}",
        ]
    )
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
        "task": data["task"],
        "episode": data["episode"],
        "episode_path": data["episode_path"],
        "camera": args.camera,
        "prompt": prompt,
        "use_text_condition": bool(args.use_text_condition),
        "start_frame": int(args.start_frame),
        "frame_indices": [int(item) for item in data["frame_indices"].tolist()],
        "history_enabled": bool(args.history),
        "history_frame_indices": history_frame_indices,
        "history_latent_count": int(history_latent_count),
        "history_frames_requested": int(history_frames),
        "history_stride": int(history_stride),
        "num_frames_requested": int(args.num_frames),
        "num_frames_generated": int(len(pred_video)),
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
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved prediction video: {pred_path}")
    if args.save_gt:
        print(f"Saved ground-truth video: {gt_path}")
    print(f"Saved metadata: {metadata_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run WanWorldModel inference on a selected eval dataset window.")

    parser.add_argument("--eval_dataset_base_path", type=str, default="world_model_data/robotwin_aloha/val_set")
    parser.add_argument("--task", type=str, default="adjust_bottle", help="Eval task folder. Defaults to the first indexed task.")
    parser.add_argument("--episode", type=str, default="episode57", help="Episode name, e.g. episode0. Defaults to the first episode for the task.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame of the eval window.")
    parser.add_argument("--camera", type=str, default="head_camera")
    parser.add_argument("--include_failed", default=False, action="store_true")
    parser.add_argument("--list_samples", default=False, action="store_true", help="List task/episode/start ranges and exit.")

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.set_defaults(history=False)
    parser.add_argument(
        "--history",
        dest="history",
        action="store_true",
        help="Enable causal VAE history conditioning for this one-chunk rollout.",
    )
    parser.add_argument(
        "--no_history",
        dest="history",
        action="store_false",
        help="Disable causal VAE history conditioning.",
    )
    parser.add_argument(
        "--world_model_history_frames",
        type=int,
        default=16,
        help="Number of RGB frames before the selected window used as causal VAE history. Must be divisible by 4; 16 gives 4 history latents.",
    )
    parser.add_argument(
        "--world_model_history_stride",
        type=int,
        default=8,
        help="Distance in raw frames between consecutive 4-frame history blocks. The default 8 matches the history training script; use 4 for contiguous history.",
    )
    parser.add_argument("--fps", type=int, default=25)

    parser.add_argument("--checkpoint_path", type=str, default="outputs/WanWorldModel_film_w_history/step-40000.safetensors")
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors")
    parser.add_argument("--text_encoder_path", type=str, default="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str, default="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--tokenizer_path", type=str, default="models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl")
    parser.add_argument("--strict_checkpoint", default=False, action="store_true")
    parser.add_argument(
        "--disable_language_condition",
        "--no_language_condition",
        dest="use_text_condition",
        default=False,
        action="store_false",
        help="Disable prompt/language conditioning. The pipeline skips context attention and does not load the text encoder/tokenizer.",
    )
    parser.add_argument(
        "--enable_language_condition",
        dest="use_text_condition",
        action="store_true",
        help="Enable prompt/language conditioning for checkpoints trained with text context.",
    )
    parser.add_argument(
        "--text_context_length",
        type=int,
        default=512,
        help="Text context token length used when language conditioning is enabled.",
    )

    parser.add_argument("--action_dim", type=int, default=None, help="Defaults to the checkpoint action embedder input dim.")
    parser.add_argument("--action_embedder_hidden_dim", type=int, default=None, help="Defaults to the checkpoint action embedder hidden dim.")
    parser.add_argument(
        "--action_injection_method",
        type=str,
        default="film",
        choices=("none", "context", "additive", "cross_attention", "cross-attention", "adaln", "film"),
    )
    parser.add_argument("--action_metadata_path", type=str, default="world_model_data/robotwin_aloha/metadata.json")
    parser.add_argument("--action_metadata_key", type=str, default="robot_statistics")
    parser.add_argument("--action_normalization_eps", type=float, default=1e-6)
    parser.add_argument(
        "--action_normalization_mode",
        type=str,
        default="standard",
        choices=("standard", "scale_only", "scale-only"),
    )

    parser.add_argument("--prompt", type=str, default=None, help="Override the episode instruction prompt.")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rand_device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"))
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--tiled", default=False, action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))

    parser.set_defaults(save_gt=True, save_input=True, save_action=True)
    parser.add_argument("--no_save_gt", dest="save_gt", action="store_false")
    parser.add_argument("--no_save_input", dest="save_input", action="store_false")
    parser.add_argument("--no_save_action", dest="save_action", action="store_false")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
