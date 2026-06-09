import argparse
from pathlib import Path

import torch
from PIL import Image

'''
  python wan_image_text_infer.py \
    --input_image world_model_data/robotwin_aloha_testset/custom_aloha_2ep_clean/adjust_bottle/episode0/camera_data/images/head_camera/frame_000000.png \
    --prompt "A fixed overhead camera view of a tabletop robot manipulation scene. A black robotic gripper enters from the left,
    reaches for the green plastic bottle with an indented base, grasps it gently, and lifts it upright. The white tabletop and
    camera remain steady, realistic robotics dataset video, smooth motion. " \
    --output outputs/wan_image_text.mp4
'''

'''
A fixed overhead camera view of a tabletop robot manipulation scene. A black robotic gripper enters from the left,
    reaches for the green plastic bottle with an indented base, grasps it gently, and lifts it upright. The white tabletop and
    camera remain steady, realistic robotics dataset video, smooth motion. 
'''

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
NEGATIVE_PROMPT = ()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B image + text to video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_image", "-i", required=True, help="Input image path.")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt.")
    parser.add_argument("--negative_prompt", default=NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--output", "-o", default="outputs/wan_image_text.mp4", help="Output video path.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    parser.add_argument("--width", type=int, default=640, help="Output width.")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames. Wan rounds this to 4n+1.")
    parser.add_argument("--fps", type=int, default=15, help="Saved video FPS.")
    parser.add_argument("--quality", type=int, default=8, help="imageio video quality, 0-10.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Scheduler sigma shift.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="DiT switch boundary.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument("--no_tiled", action="store_true", help="Disable tiled VAE encode/decode.")
    parser.add_argument("--vram_limit_gb", type=float, default=None, help="Override VRAM limit in GB.")
    parser.add_argument("--vram_reserve_gb", type=float, default=2.0, help="Reserve this much CUDA VRAM.")
    parser.add_argument("--disable_vram_management", action="store_true", help="Load models without offload config.")
    return parser.parse_args()


def get_torch_dtype():
    return torch.bfloat16


def get_vram_config(args, torch_dtype):
    if args.disable_vram_management:
        return {}
    return {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch_dtype,
        "onload_device": "cpu",
        "preparing_dtype": torch_dtype,
        "preparing_device": args.device,
        "computation_dtype": torch_dtype,
        "computation_device": args.device,
    }


def get_vram_limit(args):
    if args.disable_vram_management:
        return None
    if args.vram_limit_gb is not None:
        return args.vram_limit_gb
    device = torch.device(args.device)
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or run on a CUDA machine.")
    total_gb = torch.cuda.mem_get_info(device)[1] / (1024**3)
    return max(total_gb - args.vram_reserve_gb, 1.0)


def load_input_image(image_path, width, height):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return Image.open(image_path).convert("RGB").resize((width, height), resampling)


def build_pipeline(args):
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    torch_dtype = get_torch_dtype()
    vram_config = get_vram_config(args, torch_dtype)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="google/umt5-xxl/"),
        vram_limit=get_vram_limit(args),
    )


def main():
    args = parse_args()
    from diffsynth.utils.data import save_video

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = load_input_image(args.input_image, args.width, args.height)
    pipe = build_pipeline(args)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        switch_DiT_boundary=args.switch_DiT_boundary,
        tiled=not args.no_tiled,
    )
    save_video(video, str(output_path), fps=args.fps, quality=args.quality)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
