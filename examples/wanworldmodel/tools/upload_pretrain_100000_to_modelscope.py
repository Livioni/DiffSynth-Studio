#!/usr/bin/env python3
"""Upload the WanWorldModel no-language 100000-step checkpoint to ModelScope."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_MODEL_PATH = "outputs/WanWorldModel_film_no_language_abs_action_fix_resume2/step-100000.safetensors"
DEFAULT_REPO_ID = "livion/WanWorldModel_film_no_language_abs_action_pretrain_100000"
DEFAULT_PATH_IN_REPO = "step-100000.safetensors"
DEFAULT_REPO_TYPE = "model"


def env_token() -> str | None:
    return (
        os.environ.get("MODELSCOPE_SDK_TOKEN")
        or os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
    )


def file_size_gib(path: Path) -> float:
    return path.stat().st_size / (1024**3)


def create_repo(repo_id: str, repo_type: str, visibility: str, token: str | None, endpoint: str | None) -> None:
    try:
        from modelscope_hub import HubApi
    except ModuleNotFoundError:
        try:
            from modelscope.hub.api import HubApi
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Creating repos requires `modelscope-hub` or `modelscope`. "
                "Install one with `python -m pip install -U modelscope-hub`."
            ) from exc

    kwargs: dict[str, str] = {}
    if token:
        kwargs["token"] = token
    if endpoint:
        kwargs["endpoint"] = endpoint
    api = HubApi(**kwargs)

    try:
        if api.repo_exists(repo_id, repo_type):
            print(f"[repo] exists: {repo_id}")
            return
    except TypeError:
        if api.repo_exists(repo_id, repo_type=repo_type):
            print(f"[repo] exists: {repo_id}")
            return
    except Exception:
        pass

    print(f"[repo] create {repo_type} repo: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, visibility=visibility)
    except TypeError:
        try:
            api.create_repo(repo_id, repo_type=repo_type, visibility=visibility)
        except TypeError:
            api.create_repo(repo_id, repo_type, visibility=visibility)


def upload_with_sdk(
    local_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    token: str | None,
    endpoint: str | None,
    revision: str | None,
    commit_message: str,
) -> bool:
    try:
        from modelscope_hub import HubApi
    except ModuleNotFoundError:
        try:
            from modelscope.hub.api import HubApi
        except ModuleNotFoundError:
            return False

    kwargs: dict[str, str] = {}
    if token:
        kwargs["token"] = token
    if endpoint:
        kwargs["endpoint"] = endpoint
    api = HubApi(**kwargs)
    if not hasattr(api, "upload_file"):
        return False

    print(f"[upload:sdk] {local_path} -> {repo_id}:{path_in_repo}")
    try:
        api.upload_file(
            repo_id,
            repo_type,
            str(local_path),
            path_in_repo,
            revision=revision,
            commit_message=commit_message,
        )
    except TypeError:
        upload_kwargs = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "path_or_fileobj": str(local_path),
            "path_in_repo": path_in_repo,
            "commit_message": commit_message,
        }
        if revision:
            upload_kwargs["revision"] = revision
        api.upload_file(**upload_kwargs)
    return True


def upload_with_cli(
    local_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    token: str | None,
    endpoint: str | None,
    revision: str | None,
    commit_message: str,
) -> None:
    if shutil.which("modelscope") is None:
        raise RuntimeError(
            "ModelScope CLI is not available, and SDK upload was unavailable. "
            "Install one with `python -m pip install -U modelscope-hub modelscope`."
        )

    cmd = [
        "modelscope",
        "upload",
        repo_id,
        str(local_path),
        path_in_repo,
        "--repo-type",
        repo_type,
        "--commit-message",
        commit_message,
    ]
    if token:
        cmd.extend(["--token", token])
    if endpoint:
        cmd.extend(["--endpoint", endpoint])
    if revision:
        cmd.extend(["--revision", revision])

    print(f"[upload:cli] {local_path} -> {repo_id}:{path_in_repo}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a WanWorldModel checkpoint file to a ModelScope model repo."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Local checkpoint path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"ModelScope model repo id. Default: {DEFAULT_REPO_ID}")
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO, help=f"Remote file path. Default: {DEFAULT_PATH_IN_REPO}")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("model", "dataset"), help="ModelScope repo type. Default: model.")
    parser.add_argument("--revision", default=None, help="Target branch/revision. Default: ModelScope backend default.")
    parser.add_argument("--token", default=env_token(), help="ModelScope token. Defaults to MODELSCOPE_SDK_TOKEN, MODELSCOPE_API_TOKEN, or MODELSCOPE_TOKEN.")
    parser.add_argument("--endpoint", default=os.environ.get("MODELSCOPE_ENDPOINT"), help="ModelScope endpoint. Default: MODELSCOPE_ENDPOINT or backend default.")
    parser.add_argument("--commit-message", default="Upload WanWorldModel no-language pretrain checkpoint step 100000", help="Upload commit message.")
    parser.add_argument("--create-repo", action="store_true", help="Create the target repo first if it does not exist.")
    parser.add_argument("--visibility", choices=("public", "private", "internal"), default="public", help="Visibility used with --create-repo. Default: public.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned upload and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {model_path}")

    print(f"[file] {model_path} ({file_size_gib(model_path):.2f} GiB)")
    print(f"[target] {args.repo_id}:{args.path_in_repo} ({args.repo_type})")
    if args.dry_run:
        return

    if args.create_repo:
        create_repo(args.repo_id, args.repo_type, args.visibility, args.token, args.endpoint)

    uploaded = upload_with_sdk(
        model_path,
        args.repo_id,
        args.repo_type,
        args.path_in_repo,
        args.token,
        args.endpoint,
        args.revision,
        args.commit_message,
    )
    if not uploaded:
        upload_with_cli(
            model_path,
            args.repo_id,
            args.repo_type,
            args.path_in_repo,
            args.token,
            args.endpoint,
            args.revision,
            args.commit_message,
        )
    print("[done] upload finished")


if __name__ == "__main__":
    main()
