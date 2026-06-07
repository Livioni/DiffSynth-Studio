#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_ID = "DiffSynth-Studio/ImageMetrics"
METRIC_FILES = {
    "fid": {
        "pattern": "FID/model.safetensors",
        "expected_hash": "d4e9549be726259b444d1f62db4ce413",
    },
    "lpips_alex": {
        "pattern": "LPIPS/alexnet.safetensors",
        "expected_hash": "08a75c660c9b2e775c530a0955857f1f",
    },
    "lpips_vgg": {
        "pattern": "LPIPS/vgg.safetensors",
        "expected_hash": "5740953aaa8aba2ecd9b9c23da813591",
    },
    "lpips_squeeze": {
        "pattern": "LPIPS/squeezenet.safetensors",
        "expected_hash": "ff994b70a30599287a332105396d5004",
    },
}


def default_model_base_path():
    return Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", REPO_ROOT / "models"))


def metric_keys(args):
    keys = []
    if args.metric in ("all", "lpips"):
        keys.append(f"lpips_{args.lpips_net}")
    if args.metric in ("all", "fid"):
        keys.append("fid")
    return keys


def local_path(model_base_path, pattern):
    return model_base_path / MODEL_ID / pattern


def hash_file(path):
    from diffsynth.core.loader.file import hash_model_file

    return hash_model_file(str(path))


def verify(path, expected_hash):
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded file does not exist: {path}")
    actual_hash = hash_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Metric weight hash mismatch: {path}\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual_hash}\n"
            "Delete the file or rerun this script with --force to download again."
        )
    return actual_hash


def download_metric(key, args):
    from diffsynth.core import ModelConfig

    info = METRIC_FILES[key]
    model_base_path = args.model_base_path.resolve()
    path = local_path(model_base_path, info["pattern"])

    if args.force and path.exists():
        path.unlink()

    if path.is_file():
        actual_hash = verify(path, info["expected_hash"])
        print(f"[OK] {key}: {path} ({actual_hash})")
        return path

    print(f"[Download] {key}: {MODEL_ID}:{info['pattern']}")
    config = ModelConfig(
        model_id=MODEL_ID,
        origin_file_pattern=info["pattern"],
        download_source=args.download_source,
        local_model_path=str(model_base_path),
    )
    config.download_if_necessary()

    actual_hash = verify(path, info["expected_hash"])
    print(f"[OK] {key}: {path} ({actual_hash})")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LPIPS/FID weights required by WanWorldModel periodic eval."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=("all", "lpips", "fid"),
        help="Which eval metric weights to download.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone. train.py currently uses vgg by default.",
    )
    parser.add_argument(
        "--model_base_path",
        type=Path,
        default=default_model_base_path(),
        help="Base model cache path. Defaults to $DIFFSYNTH_MODEL_BASE_PATH or repo ./models.",
    )
    parser.add_argument(
        "--download_source",
        type=str,
        default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
        choices=("modelscope", "huggingface"),
        help="DiffSynth download source.",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Delete existing selected metric files before downloading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_base_path = args.model_base_path.expanduser()

    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source

    print(f"Model base path: {args.model_base_path.resolve()}")
    print(f"Download source: {args.download_source}")
    for key in metric_keys(args):
        download_metric(key, args)
    print("Done. train.py can load these metric weights from the same model cache.")


if __name__ == "__main__":
    main()
