#!/usr/bin/env python3
"""Pack Robotwin ALOHA task folders one by one and upload them to ModelScope.

Default behavior:
  - package each direct child directory under:
      world_model_data/robotwin_aloha/train_set
      world_model_data/robotwin_aloha/val_set
  - upload each archive to:
      livion/world_action_model_datatset
  - list existing remote files first and skip files that were already uploaded
  - delete the archive only after its upload succeeds
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


DEFAULT_ROOTS = (
    "world_model_data/robotwin_aloha/train_set",
    "world_model_data/robotwin_aloha/val_set",
)
DEFAULT_REPO_ID = "livion/world_action_model_datatset"
DEFAULT_REPO_TYPE = "dataset"


@dataclass(frozen=True)
class UploadItem:
    task_dir: Path
    archive_path: Path
    path_in_repo: str


class ModelScopeUploader:
    """Upload files to ModelScope via SDK, with CLI fallback."""

    def __init__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        token: str | None,
        revision: str | None,
        endpoint: str | None,
    ) -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token
        self.revision = revision
        self.endpoint = endpoint
        self.api = None
        self.backend = "modelscope_cli"
        self.cli_available = shutil.which("modelscope") is not None

        try:
            from modelscope_hub import HubApi
            backend = "modelscope_hub"
        except ModuleNotFoundError:
            try:
                from modelscope.hub.api import HubApi
                backend = "modelscope"
            except ModuleNotFoundError:
                if not self.cli_available:
                    raise RuntimeError(
                        "ModelScope upload dependency not found. Install one of:\n"
                        "  python -m pip install -U modelscope-hub\n"
                        "  python -m pip install -U modelscope"
                    ) from None
                return

        kwargs: dict[str, str] = {}
        if token:
            kwargs["token"] = token
        if endpoint:
            kwargs["endpoint"] = endpoint
        self.api = HubApi(**kwargs)
        self.backend = backend
        if not hasattr(self.api, "upload_file") and self.cli_available:
            self.backend = f"{backend}+modelscope_cli"

    def create_repo(self, *, visibility: str) -> None:
        if self.api is None:
            print(
                "[repo] CLI backend selected; the upload command will create the "
                "repo if supported by your ModelScope CLI.",
                flush=True,
            )
            return

        print(f"[repo] ensure {self.repo_type} repo exists: {self.repo_id}", flush=True)
        try:
            if self._repo_exists():
                return
        except Exception:
            pass
        try:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                visibility=visibility,
            )
        except TypeError:
            try:
                self.api.create_repo(
                    self.repo_id,
                    repo_type=self.repo_type,
                    visibility=visibility,
                )
            except TypeError:
                self.api.create_repo(
                    self.repo_id,
                    self.repo_type,
                    visibility=visibility,
                )

    def _repo_exists(self) -> bool:
        try:
            return bool(self.api.repo_exists(self.repo_id, self.repo_type))
        except TypeError:
            return bool(
                self.api.repo_exists(self.repo_id, repo_type=self.repo_type)
            )

    def list_repo_files(self) -> set[str] | None:
        if self.api is None:
            return None

        if not hasattr(self.api, "list_repo_files"):
            return self._list_repo_files_legacy()

        try:
            files = self.api.list_repo_files(
                self.repo_id,
                self.repo_type,
                revision=self.revision,
            )
        except TypeError:
            try:
                kwargs = self._repo_kwargs()
                if self.revision:
                    kwargs["revision"] = self.revision
                files = self.api.list_repo_files(**kwargs)
            except Exception as exc:
                if self._is_missing_repo_tree_error(exc):
                    return self._empty_remote_listing()
                raise
        except Exception as exc:
            if self._is_missing_repo_tree_error(exc):
                return self._empty_remote_listing()
            raise
        return self._normalize_remote_files(files)

    def _list_repo_files_legacy(self) -> set[str] | None:
        try:
            if self.repo_type == "dataset" and hasattr(self.api, "get_dataset_files"):
                files = self.api.get_dataset_files(
                    self.repo_id,
                    revision=self.revision or "master",
                    recursive=True,
                )
                return self._normalize_remote_files(files)
            if self.repo_type == "model" and hasattr(self.api, "get_model_files"):
                files = self.api.get_model_files(self.repo_id, recursive=True)
                return self._normalize_remote_files(files)
        except TypeError:
            return None
        except Exception as exc:
            if self._is_missing_repo_tree_error(exc):
                return self._empty_remote_listing()
            raise
        return None

    def _repo_kwargs(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "repo_type": self.repo_type}

    @staticmethod
    def _normalize_remote_files(files: object) -> set[str]:
        remote_files: set[str] = set()
        if files is None:
            return remote_files
        for item in files:
            if isinstance(item, str):
                ModelScopeUploader._add_remote_path(remote_files, item)
                continue
            if isinstance(item, dict):
                path = (
                    item.get("Path")
                    or item.get("path")
                    or item.get("Name")
                    or item.get("name")
                )
                if path:
                    ModelScopeUploader._add_remote_path(remote_files, path)
                continue
            path = (
                getattr(item, "path", None)
                or getattr(item, "Path", None)
                or getattr(item, "name", None)
                or getattr(item, "Name", None)
            )
            if path:
                ModelScopeUploader._add_remote_path(remote_files, path)
        return remote_files

    @staticmethod
    def _add_remote_path(remote_files: set[str], path: object) -> None:
        normalized = str(path).lstrip("/")
        if normalized:
            remote_files.add(normalized)

    @staticmethod
    def _is_missing_repo_tree_error(exc: Exception) -> bool:
        try:
            from modelscope_hub.errors import NotExistError
        except Exception:
            return "record not found" in str(exc).lower()
        return isinstance(exc, NotExistError)

    def _empty_remote_listing(self) -> set[str]:
        print(
            "[remote] repo file tree not found; treating remote file list as empty. "
            "If the repo does not exist or is inaccessible, upload will still fail. "
            "Use --create-repo to create it first when needed.",
            flush=True,
        )
        return set()

    def upload_file(
        self, local_path: Path, path_in_repo: str, commit_message: str
    ) -> None:
        if self.api is not None and hasattr(self.api, "upload_file"):
            try:
                self.api.upload_file(
                    self.repo_id,
                    self.repo_type,
                    str(local_path),
                    path_in_repo,
                    revision=self.revision,
                    commit_message=commit_message,
                )
            except TypeError:
                try:
                    kwargs = self._repo_kwargs()
                    if self.revision:
                        kwargs["revision"] = self.revision
                    kwargs["commit_message"] = commit_message
                    self.api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=path_in_repo,
                        **kwargs,
                    )
                except TypeError:
                    self.api.upload_file(
                        repo_id=self.repo_id,
                        path_or_fileobj=str(local_path),
                        path_in_repo=path_in_repo,
                    )
            return

        if not self.cli_available:
            raise RuntimeError(
                "ModelScope CLI is required for upload because this SDK backend "
                "does not provide upload_file()."
            )

        cmd = [
            "modelscope",
            "upload",
            self.repo_id,
            str(local_path),
            path_in_repo,
            "--repo-type",
            self.repo_type,
            "--commit-message",
            commit_message,
        ]
        if self.token:
            cmd.extend(["--token", self.token])
        if self.revision:
            cmd.extend(["--revision", self.revision])
        if self.endpoint:
            cmd.extend(["--endpoint", self.endpoint])
        subprocess.run(cmd, check=True)


def env_token() -> str | None:
    return (
        os.environ.get("MODELSCOPE_SDK_TOKEN")
        or os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive each Robotwin ALOHA subtask, upload it to a ModelScope "
            "repo, then delete the local archive after upload succeeds."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"ModelScope repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=("model", "dataset"),
        help=f"ModelScope repo type. Default: {DEFAULT_REPO_TYPE}",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help=(
            "Dataset root to scan. Can be used multiple times. "
            f"Default roots: {', '.join(DEFAULT_ROOTS)}"
        ),
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help=(
            "Base directory used to compute remote paths. "
            "Default: current working directory."
        ),
    )
    parser.add_argument(
        "--archive-dir",
        default=".modelscope_upload_archives",
        help=(
            "Temporary directory for archives. Make sure this filesystem has "
            "enough free space. Default: .modelscope_upload_archives"
        ),
    )
    parser.add_argument(
        "--token",
        default="ms-4e1b885b-2e2c-498d-8661-a6df12072473",
        help=(
            "ModelScope access token. If omitted, the script reads "
            "MODELSCOPE_SDK_TOKEN, MODELSCOPE_API_TOKEN, or MODELSCOPE_TOKEN."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT"),
        help=(
            "ModelScope endpoint, e.g. https://www.modelscope.cn. "
            "Default: MODELSCOPE_ENDPOINT or SDK/CLI default."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Target branch/revision in the ModelScope repo. Default: backend default.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Only process these task names or remote archive paths, e.g. "
            "adjust_bottle robotwin_aloha/adjust_bottle.tar.gz"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Upload even if the remote archive already exists. Remote existence "
            "checks require an SDK backend."
        ),
    )
    parser.add_argument(
        "--force-repack",
        action="store_true",
        help="Recreate local archives even if an archive already exists.",
    )
    parser.add_argument(
        "--include-root-files",
        action="store_true",
        help=(
            "Also upload direct files under each root, such as "
            "robotwin_aloha/metadata.json. These files are not archived."
        ),
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the target ModelScope repo first if the SDK backend is available.",
    )
    parser.add_argument(
        "--visibility",
        choices=("public", "private", "internal"),
        default="public",
        help="Visibility used with --create-repo. Default: public.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without creating archives or uploading.",
    )
    parser.add_argument(
        "--no-pigz",
        action="store_true",
        help="Do not use pigz even if it is available.",
    )
    parser.add_argument(
        "--pigz-threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of pigz threads when pigz is available. Default: up to 8.",
    )
    return parser.parse_args()


def rel_posix(path: Path, base_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def archive_name_for(path_in_repo: str) -> str:
    return path_in_repo.replace("/", "__")


def human_size(path: Path) -> str:
    size = path.stat().st_size
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def discover_items(
    roots: Iterable[Path],
    base_dir: Path,
    archive_dir: Path,
    only: set[str] | None,
) -> list[UploadItem]:
    items: list[UploadItem] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Root is not a directory: {root}")

        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            path_in_repo = f"{rel_posix(task_dir, base_dir)}.tar.gz"
            if only and task_dir.name not in only and path_in_repo not in only:
                continue
            archive_path = archive_dir / archive_name_for(path_in_repo)
            items.append(
                UploadItem(
                    task_dir=task_dir,
                    archive_path=archive_path,
                    path_in_repo=path_in_repo,
                )
            )
    return items


def tar_command(
    task_dir: Path, archive_path: Path, use_pigz: bool, pigz_threads: int
) -> list[str]:
    if use_pigz:
        return [
            "tar",
            "-C",
            str(task_dir.parent),
            "-I",
            f"pigz -p {pigz_threads}",
            "-cf",
            str(archive_path),
            task_dir.name,
        ]
    return [
        "tar",
        "-C",
        str(task_dir.parent),
        "-czf",
        str(archive_path),
        task_dir.name,
    ]


def make_archive(
    item: UploadItem,
    *,
    force_repack: bool,
    use_pigz: bool,
    pigz_threads: int,
) -> None:
    item.archive_path.parent.mkdir(parents=True, exist_ok=True)

    if item.archive_path.exists():
        if not force_repack:
            print(
                f"[reuse] {item.archive_path} ({human_size(item.archive_path)})",
                flush=True,
            )
            return
        item.archive_path.unlink()

    print(f"[pack] {item.task_dir} -> {item.archive_path}", flush=True)
    cmd = tar_command(item.task_dir, item.archive_path, use_pigz, pigz_threads)
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        if item.archive_path.exists():
            item.archive_path.unlink()
        raise
    print(f"[packed] {item.archive_path} ({human_size(item.archive_path)})", flush=True)


def upload_archive(
    uploader: ModelScopeUploader,
    item: UploadItem,
) -> None:
    print(
        f"[upload] {item.archive_path} -> {uploader.repo_id}:{item.path_in_repo}",
        flush=True,
    )
    uploader.upload_file(
        item.archive_path,
        item.path_in_repo,
        commit_message=f"Upload {item.path_in_repo}",
    )
    item.archive_path.unlink()
    print(f"[done] uploaded and deleted {item.archive_path}", flush=True)


def upload_root_files(
    uploader: ModelScopeUploader,
    roots: Iterable[Path],
    *,
    base_dir: Path,
    remote_files: set[str] | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    for root in roots:
        for file_path in sorted(p for p in root.iterdir() if p.is_file()):
            path_in_repo = rel_posix(file_path, base_dir)
            if (
                remote_files is not None
                and not overwrite
                and path_in_repo in remote_files
            ):
                print(f"[skip] remote exists: {path_in_repo}", flush=True)
                continue
            if dry_run:
                print(
                    f"[dry-run] upload file {file_path} -> {path_in_repo}", flush=True
                )
                continue
            print(
                f"[upload] {file_path} -> {uploader.repo_id}:{path_in_repo}",
                flush=True,
            )
            uploader.upload_file(
                file_path,
                path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )
            if remote_files is not None:
                remote_files.add(path_in_repo)
            print(f"[done] uploaded {path_in_repo}", flush=True)


def main() -> int:
    args = parse_args()
    token = args.token or env_token()
    base_dir = Path(args.base_dir).resolve()
    roots = [Path(p) for p in (args.roots or DEFAULT_ROOTS)]
    archive_dir = Path(args.archive_dir)
    only = set(args.only) if args.only else None
    use_pigz = (not args.no_pigz) and shutil.which("pigz") is not None

    items = discover_items(roots, base_dir, archive_dir, only)
    if not items:
        print("No matching task directories found.", file=sys.stderr)
        return 1

    print(f"ModelScope repo: {args.repo_id}", flush=True)
    print(f"Repo type: {args.repo_type}", flush=True)
    print(f"Tasks: {len(items)}", flush=True)
    print(f"Compression: {'pigz' if use_pigz else 'gzip'}", flush=True)

    uploader: ModelScopeUploader | None = None
    remote_files: set[str] | None = set()
    if not args.dry_run:
        uploader = ModelScopeUploader(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            revision=args.revision,
            endpoint=args.endpoint,
        )
        print(f"Uploader backend: {uploader.backend}", flush=True)
        if args.create_repo:
            uploader.create_repo(visibility=args.visibility)
        if not args.overwrite or args.include_root_files:
            print("[remote] listing existing files", flush=True)
            remote_files = uploader.list_repo_files()
            if remote_files is None:
                print(
                    "[remote] existing file list is unavailable with the CLI "
                    "backend; uploads will not be skipped.",
                    flush=True,
                )
            else:
                planned_paths = {item.path_in_repo for item in items}
                if args.include_root_files:
                    for root in roots:
                        planned_paths.update(
                            rel_posix(path, base_dir)
                            for path in root.iterdir()
                            if path.is_file()
                        )
                existing_planned = len(planned_paths & remote_files)
                print(
                    f"[remote] found {len(remote_files)} files; "
                    f"{existing_planned} planned uploads already exist.",
                    flush=True,
                )

    if args.include_root_files:
        if args.dry_run:
            dry_run_uploader = ModelScopeUploader.__new__(ModelScopeUploader)
            dry_run_uploader.repo_id = args.repo_id
            uploader_for_root_files = dry_run_uploader
        else:
            assert uploader is not None
            uploader_for_root_files = uploader
        upload_root_files(
            uploader_for_root_files,
            roots,
            base_dir=base_dir,
            remote_files=remote_files,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.path_in_repo}", flush=True)
        if args.dry_run:
            print(f"[dry-run] pack {item.task_dir} -> {item.archive_path}", flush=True)
            print(f"[dry-run] upload -> {args.repo_id}:{item.path_in_repo}", flush=True)
            continue
        if (
            remote_files is not None
            and not args.overwrite
            and item.path_in_repo in remote_files
        ):
            print(f"[skip] remote exists: {item.path_in_repo}", flush=True)
            continue

        make_archive(
            item,
            force_repack=args.force_repack,
            use_pigz=use_pigz,
            pigz_threads=args.pigz_threads,
        )
        assert uploader is not None
        upload_archive(uploader, item)
        if remote_files is not None:
            remote_files.add(item.path_in_repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
