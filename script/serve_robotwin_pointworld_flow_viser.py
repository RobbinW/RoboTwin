from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import viser
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POINTWORLD_REPO = Path("/data/dex/PointWorld")
if POINTWORLD_REPO.exists() and str(POINTWORLD_REPO) not in sys.path:
    sys.path.insert(0, str(POINTWORLD_REPO))

from visualization.viser_flow.timeline import (  # noqa: E402
    FlowTimeline,
    PointTimeline,
    build_point_timeline,
    build_rainbow_flow_timeline,
)

from robotwin_pointworld.behavior_online import decode_behavior_online_camera_group  # noqa: E402


@dataclass(slots=True)
class ClipView:
    clip_key: str
    camera_key: str
    flows: np.ndarray
    colors: np.ndarray
    visibility: np.ndarray
    robot_mask: np.ndarray
    point_timeline: PointTimeline
    flow_timeline: FlowTimeline
    non_robot_point_timeline: PointTimeline
    non_robot_flow_timeline: FlowTimeline
    initial_rgb: np.ndarray | None
    mean_final_displacement: float
    max_final_displacement: float

    @property
    def num_frames(self) -> int:
        return int(self.flows.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.flows.shape[1])

    @property
    def num_robot_points(self) -> int:
        return int(np.count_nonzero(self.robot_mask))


def _decode_jpeg_dataset(dataset: h5py.Dataset) -> np.ndarray | None:
    raw = dataset[()]
    val = raw.flat[0] if isinstance(raw, np.ndarray) and raw.dtype.kind in {"O", "S"} else raw
    if isinstance(val, np.ndarray):
        buf = val.astype(np.uint8, copy=False)
    elif isinstance(val, (bytes, bytearray, np.bytes_)):
        buf = np.frombuffer(val, dtype=np.uint8)
    else:
        return None
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return img_bgr[..., ::-1].copy()


def _clip_keys(h5_path: Path, camera_key: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        keys = [key for key in f.keys() if isinstance(f[key], h5py.Group) and camera_key in f[key]]
    return sorted(keys)


def _choose_motion_clip(h5_path: Path, camera_key: str, keys: Sequence[str]) -> str:
    best_key = keys[0]
    best_motion = -1.0
    with h5py.File(h5_path, "r") as f:
        for key in keys:
            camera = f[key][camera_key]
            if "scene_flows" in camera:
                flows = np.asarray(camera["scene_flows"], dtype=np.float32)
            else:
                flows = decode_behavior_online_camera_group(camera).scene_flows
            if flows.shape[0] < 2 or flows.shape[1] == 0:
                motion = 0.0
            else:
                motion = float(np.linalg.norm(flows[-1] - flows[0], axis=-1).max())
            if motion > best_motion:
                best_key = key
                best_motion = motion
    return best_key


def _select_points(
    flows: np.ndarray,
    visibility: np.ndarray,
    max_points: int,
    robot_mask: np.ndarray | None = None,
) -> np.ndarray:
    def _sample_even(values: np.ndarray, budget: int) -> np.ndarray:
        if budget <= 0 or values.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if values.size <= budget:
            return values.astype(np.int64, copy=False)
        sample = np.linspace(0, values.shape[0] - 1, budget, dtype=np.int64)
        return values[sample].astype(np.int64, copy=False)

    finite = np.isfinite(flows).all(axis=(0, 2))
    visible = np.asarray(visibility, dtype=bool).any(axis=0)
    valid = finite & visible
    indices = np.nonzero(valid)[0]
    if max_points <= 0 or indices.shape[0] <= max_points:
        return indices.astype(np.int64, copy=False)
    if robot_mask is None or not np.any(robot_mask[indices]):
        return _sample_even(indices, max_points)

    robot_indices = indices[robot_mask[indices]]
    non_robot_indices = indices[~robot_mask[indices]]
    robot_budget = min(robot_indices.shape[0], max(1, max_points // 4))
    non_robot_budget = max_points - robot_budget
    if non_robot_indices.shape[0] < non_robot_budget:
        robot_budget = min(robot_indices.shape[0], robot_budget + non_robot_budget - non_robot_indices.shape[0])
        non_robot_budget = non_robot_indices.shape[0]
    indices = np.concatenate([
        _sample_even(robot_indices, robot_budget),
        _sample_even(non_robot_indices, non_robot_budget),
    ])
    indices.sort()
    return indices.astype(np.int64, copy=False)


def _normalize_colors(colors: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.ndim == 2:
        colors = np.broadcast_to(colors[None, ...], target_shape)
    if colors.shape != target_shape:
        raise ValueError(f"scene_colors must have shape {target_shape} or {target_shape[1:]}, got {colors.shape}")
    if colors.dtype != np.uint8:
        if np.nanmax(colors) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def _load_clip_view(
    h5_path: Path,
    *,
    clip_key: str,
    camera_key: str,
    max_points: int,
    colormap: str,
) -> ClipView:
    with h5py.File(h5_path, "r") as f:
        if clip_key not in f:
            raise KeyError(f"clip '{clip_key}' not found in {h5_path}")
        clip = f[clip_key]
        if camera_key not in clip:
            raise KeyError(f"camera '{camera_key}' not found in clip '{clip_key}'")
        camera = clip[camera_key]
        if "scene_flows" in camera:
            flows = np.asarray(camera["scene_flows"][:], dtype=np.float32)
            visibility = np.asarray(camera["scene_visibility"][:], dtype=bool)
            colors = _normalize_colors(np.asarray(camera["scene_colors"][:]), flows.shape)
            if "scene_robot_mask" in camera:
                robot_mask = np.asarray(camera["scene_robot_mask"][:], dtype=bool)
            else:
                robot_mask = np.zeros((flows.shape[1],), dtype=bool)
        else:
            decoded = decode_behavior_online_camera_group(camera)
            flows = decoded.scene_flows
            visibility = decoded.scene_visibility
            colors = decoded.scene_colors
            robot_mask = decoded.scene_robot_mask
        initial_rgb = _decode_jpeg_dataset(camera["initial_rgb"]) if "initial_rgb" in camera else None

    if flows.ndim != 3 or flows.shape[-1] != 3:
        raise ValueError(f"scene_flows must be (T,N,3), got {flows.shape}")
    if visibility.shape != flows.shape[:2]:
        raise ValueError(f"scene_visibility shape mismatch: {visibility.shape} vs {flows.shape[:2]}")
    if robot_mask.shape != (flows.shape[1],):
        raise ValueError(f"scene_robot_mask must have shape ({flows.shape[1]},), got {robot_mask.shape}")

    indices = _select_points(flows, visibility, max_points=max_points, robot_mask=robot_mask)
    flows = flows[:, indices, :]
    visibility = visibility[:, indices]
    colors = colors[:, indices, :]
    robot_mask = robot_mask[indices]

    if flows.shape[1] == 0:
        displacement = np.zeros((0,), dtype=np.float32)
    else:
        displacement = np.linalg.norm(flows[-1] - flows[0], axis=-1)

    point_timeline = build_point_timeline(flows, colors, visibility)
    flow_timeline = build_rainbow_flow_timeline(
        flows,
        visibility,
        colormap=lambda u: colormaps[colormap](u),
        min_brightness=0.35,
    )
    non_robot = ~robot_mask
    non_robot_point_timeline = build_point_timeline(flows[:, non_robot, :], colors[:, non_robot, :], visibility[:, non_robot])
    non_robot_flow_timeline = build_rainbow_flow_timeline(
        flows[:, non_robot, :],
        visibility[:, non_robot],
        colormap=lambda u: colormaps[colormap](u),
        min_brightness=0.35,
    )
    return ClipView(
        clip_key=clip_key,
        camera_key=camera_key,
        flows=flows,
        colors=colors,
        visibility=visibility,
        robot_mask=robot_mask,
        point_timeline=point_timeline,
        flow_timeline=flow_timeline,
        non_robot_point_timeline=non_robot_point_timeline,
        non_robot_flow_timeline=non_robot_flow_timeline,
        initial_rgb=initial_rgb,
        mean_final_displacement=float(displacement.mean()) if displacement.size else 0.0,
        max_final_displacement=float(displacement.max()) if displacement.size else 0.0,
    )


def _summary_markdown(h5_path: Path, clip: ClipView, frame: int) -> str:
    return (
        "### RobotWin GT Point Flow\n"
        f"- H5: `{h5_path.name}`\n"
        f"- Clip: `{clip.clip_key}`\n"
        f"- Camera: `{clip.camera_key}`\n"
        f"- Frame: `{frame}/{clip.num_frames - 1}`\n"
        f"- Points: `{clip.num_points}`\n"
        f"- Robot points: `{clip.num_robot_points}`\n"
        f"- Mean final displacement: `{clip.mean_final_displacement:.4f} m`\n"
        f"- Max final displacement: `{clip.max_final_displacement:.4f} m`\n"
    )


def _empty_segments() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((0, 2, 3), dtype=np.uint8),
    )


def _add_world_axes(server: viser.ViserServer) -> None:
    points = np.array(
        [
            [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [[255, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 255, 0]],
            [[0, 0, 255], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    server.scene.add_line_segments("world/axes", points=points, colors=colors, line_width=3.0)


def serve(args: argparse.Namespace) -> None:
    h5_path = Path(args.h5).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    keys = _clip_keys(h5_path, args.camera)
    if not keys:
        raise RuntimeError(f"No clips with camera '{args.camera}' found in {h5_path}")

    initial_clip = _choose_motion_clip(h5_path, args.camera, keys) if args.clip == "auto" else args.clip
    if initial_clip not in keys:
        raise KeyError(f"clip '{initial_clip}' not found. Available examples: {keys[:5]}")

    current = _load_clip_view(
        h5_path,
        clip_key=initial_clip,
        camera_key=args.camera,
        max_points=args.max_points,
        colormap=args.colormap,
    )

    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.enable_default_lights()
    try:
        server.scene.set_up_direction((0.0, 0.0, 1.0))
        server.scene.world_axes.visible = False
    except Exception:
        pass
    _add_world_axes(server)

    state = {"frame": 0, "show_trail": True, "show_final": False, "show_robot": False}

    def _active_point_timeline(clip: ClipView) -> PointTimeline:
        return clip.point_timeline if state["show_robot"] else clip.non_robot_point_timeline

    def _active_flow_timeline(clip: ClipView) -> FlowTimeline:
        return clip.flow_timeline if state["show_robot"] else clip.non_robot_flow_timeline

    def _active_final_points_and_colors(clip: ClipView) -> tuple[np.ndarray, np.ndarray]:
        if state["show_robot"]:
            return clip.flows[-1], np.full_like(clip.colors[-1], 180, dtype=np.uint8)
        mask = ~clip.robot_mask
        return clip.flows[-1, mask], np.full_like(clip.colors[-1, mask], 180, dtype=np.uint8)

    points0, colors0 = _active_point_timeline(current).frame(0)
    seg0, seg_colors0 = _empty_segments()
    point_handle = server.scene.add_point_cloud(
        "scene/current_points",
        points=points0,
        colors=colors0,
        point_size=float(args.point_size),
        point_shape="rounded",
        precision="float32",
    )
    flow_handle = server.scene.add_line_segments(
        "scene/flow_trail",
        points=seg0,
        colors=seg_colors0,
        line_width=float(args.line_width),
    )
    final_points, final_colors = _active_final_points_and_colors(current)
    final_handle = server.scene.add_point_cloud(
        "scene/final_ghost",
        points=final_points,
        colors=final_colors,
        point_size=float(args.point_size) * 0.8,
        point_shape="rounded",
        precision="float32",
    )
    final_handle.visible = False

    summary = server.gui.add_markdown(_summary_markdown(h5_path, current, 0))
    if current.initial_rgb is None:
        image_handle = None
        server.gui.add_markdown("Initial RGB: unavailable")
    else:
        image_handle = server.gui.add_image(current.initial_rgb, label="initial RGB")

    with server.gui.add_folder("flow controls", expand_by_default=True):
        clip_dropdown = server.gui.add_dropdown("Clip", options=keys, initial_value=current.clip_key)
        frame_slider = server.gui.add_slider("Frame", min=0, max=current.num_frames - 1, step=1, initial_value=0)
        play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
        fps_slider = server.gui.add_slider("Play FPS", min=1.0, max=15.0, step=1.0, initial_value=float(args.fps))
        trail_checkbox = server.gui.add_checkbox("Accumulated flow trail", initial_value=True)
        robot_checkbox = server.gui.add_checkbox("Show robot points", initial_value=False)
        final_checkbox = server.gui.add_checkbox("Show final ghost points", initial_value=False)
        point_size_slider = server.gui.add_slider(
            "Point size",
            min=0.0005,
            max=0.02,
            step=0.0005,
            initial_value=float(args.point_size),
        )
        line_width_slider = server.gui.add_slider(
            "Flow line width",
            min=0.5,
            max=12.0,
            step=0.5,
            initial_value=float(args.line_width),
        )

    def _apply_frame(frame: int) -> None:
        nonlocal current
        frame = int(np.clip(frame, 0, current.num_frames - 1))
        state["frame"] = frame
        pts, cols = _active_point_timeline(current).frame(frame)
        point_handle.points = pts.astype(np.float32, copy=False)
        point_handle.colors = cols.astype(np.uint8, copy=False)
        if state["show_trail"]:
            segs, seg_cols = _active_flow_timeline(current).slice_for_frame(frame)
        else:
            segs, seg_cols = _empty_segments()
        flow_handle.points = segs.astype(np.float32, copy=False)
        flow_handle.colors = seg_cols.astype(np.uint8, copy=False)
        final_points, final_colors = _active_final_points_and_colors(current)
        final_handle.points = final_points.astype(np.float32, copy=False)
        final_handle.colors = final_colors.astype(np.uint8, copy=False)
        final_handle.visible = bool(state["show_final"])
        summary.content = _summary_markdown(h5_path, current, frame)

    def _set_current_clip(clip_key: str) -> None:
        nonlocal current
        current = _load_clip_view(
            h5_path,
            clip_key=clip_key,
            camera_key=args.camera,
            max_points=args.max_points,
            colormap=args.colormap,
        )
        final_points, final_colors = _active_final_points_and_colors(current)
        final_handle.points = final_points.astype(np.float32, copy=False)
        final_handle.colors = final_colors.astype(np.uint8, copy=False)
        if image_handle is not None and current.initial_rgb is not None:
            image_handle.image = current.initial_rgb
        frame_slider.value = 0
        _apply_frame(0)

    def _clip_cb(event) -> None:
        _set_current_clip(str(event.target.value))

    def _frame_cb(event) -> None:
        _apply_frame(int(event.target.value))

    def _trail_cb(event) -> None:
        state["show_trail"] = bool(event.target.value)
        _apply_frame(int(state["frame"]))

    def _robot_cb(event) -> None:
        state["show_robot"] = bool(event.target.value)
        _apply_frame(int(state["frame"]))

    def _final_cb(event) -> None:
        state["show_final"] = bool(event.target.value)
        final_handle.visible = bool(state["show_final"])

    def _point_size_cb(event) -> None:
        size = float(event.target.value)
        point_handle.point_size = size
        final_handle.point_size = size * 0.8

    def _line_width_cb(event) -> None:
        flow_handle.line_width = float(event.target.value)

    clip_dropdown.on_update(_clip_cb)
    frame_slider.on_update(_frame_cb)
    trail_checkbox.on_update(_trail_cb)
    robot_checkbox.on_update(_robot_cb)
    final_checkbox.on_update(_final_cb)
    point_size_slider.on_update(_point_size_cb)
    line_width_slider.on_update(_line_width_cb)
    _apply_frame(0)

    print(f"VISER_URL=http://127.0.0.1:{args.port}", flush=True)
    print(f"H5={h5_path}", flush=True)
    print(f"DEFAULT_CLIP={current.clip_key}", flush=True)
    print(f"CAMERA={args.camera}", flush=True)
    print(f"FLOW_SHAPE={tuple(current.flows.shape)}", flush=True)

    stop = {"value": False}

    def _handle_signal(_signum, _frame) -> None:
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.smoke_test:
        frame_slider.value = current.num_frames - 1
        _apply_frame(current.num_frames - 1)
        time.sleep(float(args.smoke_seconds))
        return

    while not stop["value"]:
        if bool(play_checkbox.value):
            next_frame = (int(state["frame"]) + 1) % current.num_frames
            frame_slider.value = next_frame
            _apply_frame(next_frame)
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))
        else:
            time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a viser viewer for RobotWin PointWorld-style GT point flow.")
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("/data/dex/RoboTwin/data/adjust_bottle/pointworld_behavior_compact_head/data/episode0.hdf5"),
    )
    parser.add_argument("--clip", type=str, default="auto", help="'auto' chooses the clip with largest final displacement.")
    parser.add_argument("--camera", type=str, default="camera_head")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max_points", type=int, default=2048)
    parser.add_argument("--point_size", type=float, default=0.006)
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--colormap", type=str, default="turbo")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
