#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from diffsynth.core.data.world_model_dataset import WorldModelDataset


DEFAULT_ROOTS = (
    "world_model_data/robotwin_aloha_fail",
    "world_model_data/robotwin_aloha/train_set",
    "world_model_data/robotwin_aloha/val_set",
)
DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera", "third_view")
TASK_EXCLUDE_NAMES = {".agents", ".codex", ".git", "metadata", "wm_stats_cache"}
EPISODE_RE = re.compile(r"episode(\d+)$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def episode_sort_key(name: str) -> tuple[int, int | str]:
    match = EPISODE_RE.match(name)
    if match is not None:
        return 0, int(match.group(1))
    return 1, name


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def list_task_names(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    tasks = []
    for path in sorted(root_path.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if path.name.startswith("_") or path.name.startswith(".") or path.name in TASK_EXCLUDE_NAMES:
            continue
        tasks.append(path.name)
    return tasks


def list_episode_names(root: str, task: str) -> list[str]:
    task_path = Path(root) / task
    if not task_path.is_dir():
        return []
    episodes = [
        path.name
        for path in task_path.iterdir()
        if path.is_dir() and path.name.startswith("episode")
    ]
    return sorted(episodes, key=episode_sort_key)


def count_episodes(root: str, task: str) -> int:
    return len(list_episode_names(root, task))


def public_host(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def safe_name(value: str) -> str:
    value = SAFE_NAME_RE.sub("_", value.strip())
    value = value.strip("._")
    return value or "recording"


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "success"}:
        return True
    if text in {"0", "false", "no", "fail", "failure"}:
        return False
    return None


def scalar_series_names(field: str, dim: int) -> list[str]:
    if field == "endpose" and dim == 7:
        return ["x", "y", "z", "qx", "qy", "qz", "qw"]
    if field == "arm_joint":
        return [f"joint_{index}" for index in range(dim)]
    if dim == 1:
        return [field]
    return [f"{field}_{index}" for index in range(dim)]


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


@dataclass
class ActiveRecording:
    recording_path: Path
    recording_url: str
    viewer_url: str
    root: str
    task: str
    episode: str
    frame_count: int
    created_at: float


class AppState:
    def __init__(
        self,
        *,
        roots: tuple[str, ...],
        cameras: tuple[str, ...],
        app_host: str,
        app_port: int,
        rerun_web_port: int,
        recording_dir: Path,
        depth_meter: float,
        server_open_browser: bool,
    ) -> None:
        self.roots = tuple(str(Path(root)) for root in roots)
        self.cameras = cameras
        self.app_host = app_host
        self.app_port = int(app_port)
        self.rerun_web_port = int(rerun_web_port)
        self.recording_dir = recording_dir
        self.depth_meter = float(depth_meter)
        self.server_open_browser = server_open_browser
        self.lock = threading.Lock()
        self.web_viewer_started = False
        self.viewer_process: subprocess.Popen[bytes] | None = None
        self.active: ActiveRecording | None = None

    def resolve_root(self, root: str) -> str:
        root = unquote(root)
        if root not in self.roots:
            raise ValueError(f"Unknown dataset root: {root}")
        return root

    def roots_payload(self) -> dict[str, Any]:
        roots = []
        for root in self.roots:
            task_names = list_task_names(root)
            roots.append(
                {
                    "path": root,
                    "exists": Path(root).is_dir(),
                    "task_count": len(task_names),
                    "episode_count": sum(count_episodes(root, task) for task in task_names),
                }
            )
        return {"roots": roots}

    def tasks_payload(self, root: str) -> dict[str, Any]:
        root = self.resolve_root(root)
        tasks = [
            {"name": task, "episode_count": count_episodes(root, task)}
            for task in list_task_names(root)
        ]
        return {"root": root, "tasks": tasks}

    def episodes_payload(self, root: str, task: str) -> dict[str, Any]:
        root = self.resolve_root(root)
        episodes = []
        manual_episode_names = set(list_episode_names(root, task))
        if not manual_episode_names:
            return {"root": root, "task": task, "episodes": [], "warning": "No episode folders found."}

        try:
            dataset = WorldModelDataset(
                root=root,
                tasks=(task,),
                cameras=self.cameras,
                num_frames=1,
                stride=1,
                include_depth=True,
                include_camera_params=True,
                include_failed=True,
            )
        except Exception as exc:
            return {
                "root": root,
                "task": task,
                "episodes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        valid_names = {episode.episode for episode in dataset.episodes}
        for name in sorted(manual_episode_names, key=episode_sort_key):
            episode_path = Path(root) / task / name
            meta = read_json(episode_path / "meta.json")
            valid_info = next((item for item in dataset.episodes if item.episode == name), None)
            episodes.append(
                {
                    "name": name,
                    "valid": name in valid_names,
                    "length": int(valid_info.length) if valid_info is not None else None,
                    "result": meta.get("result"),
                    "success": to_bool(meta.get("success")),
                    "failure_mode": meta.get("failure_mode"),
                    "video_fps": meta.get("video_fps"),
                }
            )
        return {"root": root, "task": task, "episodes": episodes}

    def ensure_web_viewer(self) -> None:
        if self.web_viewer_started and self.viewer_responds():
            return
        if self.viewer_process is not None and self.viewer_process.poll() is None and self.viewer_responds():
            self.web_viewer_started = True
            return
        if self.viewer_responds():
            self.web_viewer_started = True
            return
        if self.rerun_web_port <= 0:
            raise ValueError("--rerun-web-port must be a positive fixed port.")

        executable = shutil.which("rerun")
        if executable is None:
            candidate = Path(sys.executable).with_name("rerun")
            if candidate.is_file():
                executable = str(candidate)
        if executable is None:
            raise FileNotFoundError("Could not find rerun executable in PATH or next to sys.executable.")

        log_path = Path("/tmp/worldmodel_rerun_viewer.log")
        log_file = log_path.open("ab")
        try:
            self.viewer_process = subprocess.Popen(
                [
                    executable,
                    "--serve-web",
                    "--web-viewer-port",
                    str(self.rerun_web_port),
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    "auto",
                ],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        finally:
            log_file.close()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if self.viewer_responds():
                self.web_viewer_started = True
                return
            if self.viewer_process.poll() is not None:
                raise RuntimeError(f"Rerun web viewer exited early. See {log_path}.")
            time.sleep(0.2)
        raise TimeoutError(f"Rerun web viewer did not respond on port {self.rerun_web_port}. See {log_path}.")

    def viewer_responds(self) -> bool:
        try:
            with urlopen(f"http://127.0.0.1:{self.rerun_web_port}/", timeout=0.5) as response:
                return 200 <= int(response.status) < 500
        except Exception:
            return False

    def open_episode(
        self,
        *,
        root: str,
        task: str,
        episode: str,
        depth_meter: float | None = None,
    ) -> dict[str, Any]:
        root = self.resolve_root(root)
        depth_meter = self.depth_meter if depth_meter is None else float(depth_meter)
        with self.lock:
            self.ensure_web_viewer()
            self.recording_dir.mkdir(parents=True, exist_ok=True)
            recording_path = self.recording_path(task, episode)
            summary = render_with_worker(
                root=root,
                task=task,
                episode=episode,
                cameras=self.cameras,
                output_path=recording_path,
                depth_meter=depth_meter,
            )
            frame_count = int(summary["frame_count"])

            recording_url = self.make_recording_url(recording_path.name)
            viewer_url = self.make_viewer_url(recording_url)
            if self.server_open_browser:
                import webbrowser

                webbrowser.open(viewer_url)

            self.active = ActiveRecording(
                recording_path=recording_path,
                recording_url=recording_url,
                viewer_url=viewer_url,
                root=root,
                task=task,
                episode=episode,
                frame_count=frame_count,
                created_at=time.time(),
            )
            return {
                "root": root,
                "task": task,
                "episode": episode,
                "frame_count": frame_count,
                "recording_url": recording_url,
                "recording_path_url": recording_url,
                "recording_path": str(recording_path),
                "viewer_url": viewer_url,
                "viewer_path": viewer_url,
            }

    def recording_path(self, task: str, episode: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        name = f"{safe_name(task)}_{safe_name(episode)}_{stamp}_{suffix}.rrd"
        return self.recording_dir / name

    def make_recording_url(self, file_name: str) -> str:
        return f"/recordings/{quote(file_name)}"

    def make_viewer_url(self, recording_url: str) -> str:
        return f"/rerun/?url={quote(recording_url, safe='')}&persist=0"

    def status_payload(self) -> dict[str, Any]:
        active = None
        if self.active is not None:
            active = {
                "root": self.active.root,
                "task": self.active.task,
                "episode": self.active.episode,
                "frame_count": self.active.frame_count,
                "recording_url": self.active.recording_url,
                "recording_path_url": self.active.recording_url,
                "recording_path": str(self.active.recording_path),
                "viewer_url": self.active.viewer_url,
                "viewer_path": self.active.viewer_url,
                "created_at": self.active.created_at,
            }
        return {
            "web_viewer_started": self.web_viewer_started,
            "rerun_web_port": self.rerun_web_port,
            "viewer_pid": self.viewer_process.pid if self.viewer_process is not None else None,
            "active": active,
        }

    def shutdown(self) -> None:
        if self.viewer_process is None or self.viewer_process.poll() is not None:
            return
        try:
            os.killpg(self.viewer_process.pid, signal.SIGTERM)
            self.viewer_process.wait(timeout=5)
        except Exception:
            self.viewer_process.kill()


def load_episode(*, root: str, task: str, episode: str, cameras: tuple[str, ...]) -> dict[str, Any]:
    dataset = WorldModelDataset(
        root=root,
        tasks=(task,),
        cameras=cameras,
        num_frames=1,
        stride=1,
        include_depth=True,
        include_camera_params=True,
        include_failed=True,
    )
    info = next((item for item in dataset.episodes if item.episode == episode), None)
    if info is None:
        raise FileNotFoundError(f"No valid episode found for {root}/{task}/{episode}.")

    frame_indices = np.arange(info.length, dtype=np.int64)
    camera_data = {}
    for camera in cameras:
        camera_data[camera] = {
            "rgb": dataset._load_rgb_window(info.path, camera, frame_indices),
            "depth": dataset._load_depth_window(info.path, camera, frame_indices),
        }
    return {
        "root": root,
        "task": info.task,
        "episode": info.episode,
        "episode_path": info.path,
        "meta": info.meta,
        "prompt": info.text_conditions[0] if info.text_conditions else "",
        "length": int(info.length),
        "frame_indices": frame_indices,
        "cameras": camera_data,
        "robot": dataset._load_robot_window(info.path, frame_indices),
    }


def render_with_worker(
    *,
    root: str,
    task: str,
    episode: str,
    cameras: tuple[str, ...],
    output_path: Path,
    depth_meter: float,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "worldmodel_rerun.app",
        "--render-rrd",
        "--root",
        root,
        "--task",
        task,
        "--episode",
        episode,
        "--output",
        str(output_path),
        "--depth-meter",
        str(depth_meter),
        "--cameras",
        ",".join(cameras),
    ]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"RRD render worker failed: {detail[-4000:]}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("RRD render worker produced no summary.")
    return json.loads(lines[-1])


def render_rrd(
    *,
    root: str,
    task: str,
    episode: str,
    cameras: tuple[str, ...],
    output_path: Path,
    depth_meter: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_episode(root=root, task=task, episode=episode, cameras=cameras)
    recording = rr.RecordingStream(
        application_id="worldmodeldataset_rerun",
        recording_id=str(uuid.uuid4()),
    )
    recording.save(output_path, default_blueprint=build_blueprint(cameras))
    log_episode(recording=recording, loaded=loaded, depth_meter=depth_meter)
    recording.disconnect()
    return {
        "frame_count": int(loaded["length"]),
        "recording_path": str(output_path),
        "bytes": output_path.stat().st_size,
    }


def build_blueprint(cameras: tuple[str, ...]) -> rrb.Blueprint:
    rgb_views = [
        rrb.Spatial2DView(origin=f"cameras/{camera}/rgb", name=f"{camera} rgb")
        for camera in cameras
    ]
    depth_views = [
        rrb.Spatial2DView(origin=f"cameras/{camera}/depth", name=f"{camera} depth")
        for camera in cameras
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(origin="info", name="Episode"),
            rrb.Grid(
                *(rgb_views + depth_views),
                grid_columns=len(cameras),
                name="RGB and Depth",
            ),
            rrb.Tabs(
                rrb.TimeSeriesView(origin="robot/left/action", name="left action"),
                rrb.TimeSeriesView(origin="robot/right/action", name="right action"),
                rrb.TimeSeriesView(origin="robot/left/state", name="left state"),
                rrb.TimeSeriesView(origin="robot/right/state", name="right state"),
                name="State and Action",
            ),
            row_shares=[1.0, 8.0, 3.0],
        ),
        rrb.TimePanel(timeline="frame", fps=30.0),
        collapse_panels=True,
    )


def log_episode(*, recording: rr.RecordingStream, loaded: dict[str, Any], depth_meter: float) -> None:
    meta = loaded["meta"]
    fps = float(meta.get("video_fps") or 30.0)
    recording.log("info", rr.TextDocument(info_markdown(loaded, depth_meter), media_type="text/markdown"), static=True)
    log_robot_series_metadata(recording, loaded["robot"])

    frame_indices = loaded["frame_indices"]
    for local_index, frame_index in enumerate(frame_indices):
        frame = int(frame_index)
        recording.set_time("frame", sequence=frame)
        recording.set_time("seconds", duration=frame / fps)

        for camera, camera_data in loaded["cameras"].items():
            rgb = np.asarray(camera_data["rgb"][local_index].convert("RGB"))
            recording.log(f"cameras/{camera}/rgb", rr.Image(rgb))

            depth = as_numpy(camera_data["depth"][local_index]).astype(np.float32, copy=False)
            recording.log(f"cameras/{camera}/depth", rr.DepthImage(depth, meter=depth_meter))

        for arm, arm_data in loaded["robot"].items():
            for group, group_data in arm_data.items():
                for field, values in group_data.items():
                    vector = as_numpy(values[local_index]).reshape(-1).astype(np.float64, copy=False)
                    recording.log(f"robot/{arm}/{group}/{field}", rr.Scalars(vector))


def log_robot_series_metadata(recording: rr.RecordingStream, robot: dict[str, Any]) -> None:
    for arm, arm_data in robot.items():
        for group, group_data in arm_data.items():
            for field, values in group_data.items():
                values_np = as_numpy(values)
                dim = 1 if values_np.ndim == 1 else int(values_np.shape[-1])
                recording.log(
                    f"robot/{arm}/{group}/{field}",
                    rr.SeriesLines(names=scalar_series_names(field, dim)),
                    static=True,
                )


def info_markdown(loaded: dict[str, Any], depth_meter: float) -> str:
    meta = loaded["meta"]
    lines = [
        "# WorldModelDataset Episode",
        "",
        f"- root: `{loaded['root']}`",
        f"- task: `{loaded['task']}`",
        f"- episode: `{loaded['episode']}`",
        f"- path: `{loaded['episode_path']}`",
        f"- frames: `{loaded['length']}`",
        f"- video_fps: `{meta.get('video_fps', 30.0)}`",
        f"- result: `{meta.get('result', 'unknown')}`",
        f"- success: `{meta.get('success', 'unknown')}`",
        f"- depth_meter: `{depth_meter}`",
    ]
    for key in ("failure_mode", "failure_variant", "failure_seed"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    prompt = loaded.get("prompt") or ""
    if prompt:
        lines.extend(["", "## Prompt", "", prompt])
    return "\n".join(lines)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "WorldModelRerun/1.0"

    def __init__(self, *args: Any, state: AppState, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path.startswith("/rerun"):
            self.proxy_rerun(parsed.path, parsed.query, head_only=False)
            return
        if parsed.path.startswith("/recordings/"):
            self.send_recording(parsed.path, head_only=False)
            return
        if parsed.path == "/api/roots":
            self.send_json(self.state.roots_payload())
            return
        if parsed.path == "/api/tasks":
            query = parse_qs(parsed.query)
            self.handle_json(lambda: self.state.tasks_payload(required_query(query, "root")))
            return
        if parsed.path == "/api/episodes":
            query = parse_qs(parsed.query)
            self.handle_json(
                lambda: self.state.episodes_payload(
                    required_query(query, "root"),
                    required_query(query, "task"),
                )
            )
            return
        if parsed.path == "/api/status":
            self.send_json(self.state.status_payload())
            return
        if self.state.web_viewer_started:
            self.proxy_rerun(parsed.path, parsed.query, head_only=False)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/rerun"):
            self.proxy_rerun(parsed.path, parsed.query, head_only=True)
            return
        if parsed.path.startswith("/recordings/"):
            self.send_recording(parsed.path, head_only=True)
            return
        if self.state.web_viewer_started:
            self.proxy_rerun(parsed.path, parsed.query, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/open":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        payload = self.read_json_body()

        def open_episode() -> dict[str, Any]:
            return self.state.open_episode(
                root=str(payload.get("root", "")),
                task=str(payload.get("task", "")),
                episode=str(payload.get("episode", "")),
                depth_meter=payload.get("depth_meter"),
            )

        self.handle_json(open_episode)

    def handle_json(self, func: Any) -> None:
        try:
            self.send_json(func())
        except Exception as exc:
            self.send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_recording(self, request_path: str, *, head_only: bool) -> None:
        file_name = unquote(request_path[len("/recordings/"):])
        if "/" in file_name or "\\" in file_name or not file_name.endswith(".rrd"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid recording path")
            return
        path = self.state.recording_dir / file_name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Recording not found")
            return
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if not head_only:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def proxy_rerun(self, request_path: str, query: str, *, head_only: bool) -> None:
        self.state.ensure_web_viewer()
        upstream_path = request_path
        if upstream_path == "/rerun":
            upstream_path = "/"
        elif upstream_path.startswith("/rerun/"):
            upstream_path = upstream_path[len("/rerun"):]
        if not upstream_path:
            upstream_path = "/"
        upstream_url = f"http://127.0.0.1:{self.state.rerun_web_port}{upstream_path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"

        request = Request(upstream_url, method="HEAD" if head_only else "GET")
        try:
            with urlopen(request, timeout=10.0) as response:
                body = b"" if head_only else response.read()
                self.send_response(response.status)
                blocked_headers = {"connection", "transfer-encoding", "content-encoding"}
                for key, value in response.headers.items():
                    if key.lower() in blocked_headers:
                        continue
                    if key.lower() == "content-length" and not head_only:
                        value = str(len(body))
                    self.send_header(key, value)
                self.end_headers()
                if not head_only:
                    self.wfile.write(body)
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"Rerun viewer proxy failed: {exc}")


def required_query(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"Missing query parameter: {key}")
    return values[0]


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldModelDataset Rerun</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #68707c;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --warn: #9a3412;
      --bad: #b42318;
      --good: #027a48;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 24px 28px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(260px, 1.7fr) minmax(220px, 1fr) minmax(170px, 0.7fr) 140px 128px;
      gap: 12px;
      align-items: end;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
    }
    select, input, button {
      height: 38px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      font: inherit;
      min-width: 0;
    }
    select, input {
      padding: 0 10px;
    }
    button {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: not-allowed;
      background: #b7c3cc;
      border-color: #b7c3cc;
    }
    .status {
      margin-top: 14px;
      min-height: 24px;
      color: var(--muted);
      font-size: 14px;
    }
    .status.error { color: var(--bad); }
    .status.good { color: var(--good); }
    .layout {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
      margin-top: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 13px 16px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }
    th {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      background: #fafbfc;
    }
    tbody tr {
      cursor: pointer;
    }
    tbody tr:hover {
      background: #eef7f5;
    }
    tbody tr.selected {
      background: #dff3ef;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #eef1f5;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .pill.good { background: #dcfae6; color: var(--good); }
    .pill.bad { background: #fee4e2; color: var(--bad); }
    .pill.warn { background: #ffead5; color: var(--warn); }
    .details {
      padding: 14px 16px;
      display: grid;
      gap: 12px;
      font-size: 14px;
    }
    .kv {
      display: grid;
      gap: 3px;
    }
    .kv span:first-child {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 650;
    }
    .kv span:last-child {
      overflow-wrap: anywhere;
    }
    @media (max-width: 920px) {
      .controls {
        grid-template-columns: 1fr 1fr;
      }
      .layout {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 560px) {
      header { padding: 0 16px; }
      main { padding: 16px; }
      .controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>WorldModelDataset Rerun</h1>
    <span id="activeBadge" class="pill">Idle</span>
  </header>
  <main>
    <section class="controls">
      <label>Root<select id="rootSelect"></select></label>
      <label>Task<select id="taskSelect"></select></label>
      <label>Episode<select id="episodeSelect"></select></label>
      <label>Depth Meter<input id="depthMeter" type="number" min="0.001" step="1" value="1000"></label>
      <button id="openButton" type="button">Open Rerun</button>
    </section>
    <div id="status" class="status">Loading roots...</div>
    <section class="layout">
      <div class="panel">
        <h2>Episodes</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Frames</th>
              <th>Result</th>
              <th>Failure</th>
            </tr>
          </thead>
          <tbody id="episodeRows"></tbody>
        </table>
      </div>
      <aside class="panel">
        <h2>Selection</h2>
        <div id="details" class="details"></div>
      </aside>
    </section>
  </main>
  <script>
    const rootSelect = document.getElementById("rootSelect");
    const taskSelect = document.getElementById("taskSelect");
    const episodeSelect = document.getElementById("episodeSelect");
    const depthMeter = document.getElementById("depthMeter");
    const openButton = document.getElementById("openButton");
    const statusEl = document.getElementById("status");
    const episodeRows = document.getElementById("episodeRows");
    const details = document.getElementById("details");
    const activeBadge = document.getElementById("activeBadge");

    let roots = [];
    let tasks = [];
    let episodes = [];

    function setStatus(text, kind = "") {
      statusEl.textContent = text;
      statusEl.className = "status" + (kind ? " " + kind : "");
    }

    function setStatusHtml(html, kind = "") {
      statusEl.innerHTML = html;
      statusEl.className = "status" + (kind ? " " + kind : "");
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function option(select, value, text, disabled = false) {
      const item = document.createElement("option");
      item.value = value;
      item.textContent = text;
      item.disabled = disabled;
      select.appendChild(item);
    }

    function selectedEpisode() {
      return episodes.find((item) => item.name === episodeSelect.value) || null;
    }

    function renderDetails() {
      const root = rootSelect.value || "-";
      const task = taskSelect.value || "-";
      const episode = selectedEpisode();
      const rows = [
        ["Root", root],
        ["Task", task],
        ["Episode", episode ? episode.name : "-"],
        ["Frames", episode && episode.length ? String(episode.length) : "-"],
        ["Result", episode && episode.result ? episode.result : "-"],
      ];
      details.innerHTML = rows.map(([key, value]) => `
        <div class="kv"><span>${key}</span><span>${value}</span></div>
      `).join("");
    }

    function renderEpisodes() {
      episodeRows.innerHTML = "";
      for (const episode of episodes) {
        const tr = document.createElement("tr");
        if (episode.name === episodeSelect.value) tr.classList.add("selected");
        tr.innerHTML = `
          <td>${episode.name}</td>
          <td>${episode.length || "-"}</td>
          <td>${resultPill(episode)}</td>
          <td>${episode.failure_mode || "-"}</td>
        `;
        tr.addEventListener("click", () => {
          episodeSelect.value = episode.name;
          renderEpisodes();
          renderDetails();
        });
        episodeRows.appendChild(tr);
      }
      if (!episodes.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="4">No episodes</td>';
        episodeRows.appendChild(tr);
      }
    }

    function resultPill(episode) {
      if (!episode.valid) return '<span class="pill warn">invalid</span>';
      if (episode.success === true || episode.result === "success") return '<span class="pill good">success</span>';
      if (episode.success === false || episode.result === "fail") return '<span class="pill bad">fail</span>';
      return '<span class="pill">unknown</span>';
    }

    async function loadRoots() {
      const data = await api("/api/roots");
      roots = data.roots;
      rootSelect.innerHTML = "";
      for (const root of roots) {
        const label = `${root.path} (${root.task_count} tasks, ${root.episode_count} episodes)`;
        option(rootSelect, root.path, label, !root.exists);
      }
      await loadTasks();
    }

    async function loadTasks() {
      taskSelect.innerHTML = "";
      episodeSelect.innerHTML = "";
      episodes = [];
      renderEpisodes();
      renderDetails();
      if (!rootSelect.value) return;
      const data = await api(`/api/tasks?root=${encodeURIComponent(rootSelect.value)}`);
      tasks = data.tasks;
      for (const task of tasks) {
        option(taskSelect, task.name, `${task.name} (${task.episode_count})`, task.episode_count === 0);
      }
      if (!tasks.length) {
        setStatus("No tasks found.", "error");
        return;
      }
      await loadEpisodes();
    }

    async function loadEpisodes() {
      episodeSelect.innerHTML = "";
      episodes = [];
      renderEpisodes();
      renderDetails();
      if (!rootSelect.value || !taskSelect.value) return;
      const data = await api(`/api/episodes?root=${encodeURIComponent(rootSelect.value)}&task=${encodeURIComponent(taskSelect.value)}`);
      if (data.error) throw new Error(data.error);
      episodes = data.episodes.filter((episode) => episode.valid);
      for (const episode of episodes) {
        option(episodeSelect, episode.name, `${episode.name} (${episode.length || "-"} frames)`);
      }
      if (data.warning) setStatus(data.warning, "error");
      else if (!episodes.length) setStatus("No valid episodes for this task.", "error");
      else setStatus(`${episodes.length} valid episodes loaded.`, "good");
      renderEpisodes();
      renderDetails();
    }

    async function openRerun() {
      if (!rootSelect.value || !taskSelect.value || !episodeSelect.value) {
        setStatus("Select a valid episode.", "error");
        return;
      }
      const popup = window.open("about:blank", "_blank");
      openButton.disabled = true;
      activeBadge.textContent = "Loading";
      activeBadge.className = "pill warn";
      setStatus("Preparing Rerun recording...");
      try {
        const data = await api("/api/open", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            root: rootSelect.value,
            task: taskSelect.value,
            episode: episodeSelect.value,
            depth_meter: Number(depthMeter.value || 1000),
          }),
        });
        const recordingPath = data.recording_url || data.recording_path_url;
        const recordingUrl = new URL(recordingPath, window.location.origin).href;
        const viewer = new URL("/rerun/", window.location.origin);
        viewer.searchParams.set("url", recordingUrl);
        viewer.searchParams.set("persist", "0");
        const viewerUrl = viewer.href;
        if (popup) popup.location = viewerUrl;
        activeBadge.textContent = "Active";
        activeBadge.className = "pill good";
        setStatusHtml(`Opened ${data.task}/${data.episode} (${data.frame_count} frames). <a href="${viewerUrl}" target="_blank" rel="noreferrer">Open Rerun</a>`, "good");
      } catch (error) {
        if (popup) popup.close();
        activeBadge.textContent = "Error";
        activeBadge.className = "pill bad";
        setStatus(error.message, "error");
      } finally {
        openButton.disabled = false;
      }
    }

    rootSelect.addEventListener("change", () => loadTasks().catch((error) => setStatus(error.message, "error")));
    taskSelect.addEventListener("change", () => loadEpisodes().catch((error) => setStatus(error.message, "error")));
    episodeSelect.addEventListener("change", () => { renderEpisodes(); renderDetails(); });
    openButton.addEventListener("click", openRerun);

    loadRoots().catch((error) => setStatus(error.message, "error"));
  </script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldModelDataset Rerun web visualizer.")
    parser.add_argument("--render-rrd", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--episode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1", help="Web UI host.")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port.")
    parser.add_argument(
        "--roots",
        type=parse_csv,
        default=DEFAULT_ROOTS,
        help="Comma-separated dataset roots.",
    )
    parser.add_argument(
        "--cameras",
        type=parse_csv,
        default=DEFAULT_CAMERAS,
        help="Comma-separated cameras to visualize.",
    )
    parser.add_argument("--rerun-web-port", type=int, default=9090)
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=Path("/tmp/worldmodel_rerun_recordings"),
        help="Directory for generated .rrd files.",
    )
    parser.add_argument("--depth-meter", type=float, default=1000.0)
    parser.add_argument(
        "--server-open-browser",
        action="store_true",
        help="Also let rerun open the viewer from the server process.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = args.roots or DEFAULT_ROOTS
    cameras = args.cameras or DEFAULT_CAMERAS
    if args.render_rrd:
        if not args.root or not args.task or not args.episode or args.output is None:
            raise ValueError("--render-rrd requires --root, --task, --episode, and --output.")
        summary = render_rrd(
            root=args.root,
            task=args.task,
            episode=args.episode,
            cameras=tuple(cameras),
            output_path=args.output,
            depth_meter=args.depth_meter,
        )
        print(json.dumps(summary))
        return

    state = AppState(
        roots=tuple(roots),
        cameras=tuple(cameras),
        app_host=args.host,
        app_port=args.port,
        rerun_web_port=args.rerun_web_port,
        recording_dir=args.recording_dir,
        depth_meter=args.depth_meter,
        server_open_browser=args.server_open_browser,
    )
    handler = partial(RequestHandler, state=state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{public_host(args.host)}:{args.port}/"
    print(f"WorldModelDataset Rerun UI: {url}")
    print(f"Rerun web viewer port: {args.rerun_web_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
