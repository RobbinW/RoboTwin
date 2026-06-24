from __future__ import annotations

import argparse
import signal
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import trimesh
import viser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.serve_robotwin_pointworld_flow_viser import (  # noqa: E402
    _add_world_axes,
    _choose_motion_clip,
    _clip_keys,
    _load_clip_view,
    _summary_markdown,
)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _origin_to_matrix(origin: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if origin is None:
        return matrix
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float32)
    matrix[:3, 3] = xyz
    matrix[:3, :3] = _rpy_to_matrix(rpy)
    return matrix


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    return matrix


def _translation_to_matrix(axis: np.ndarray, distance: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(axis, dtype=np.float32) * float(distance)
    return matrix


@dataclass(slots=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfFk:
    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path
        self.urdf_dir = urdf_path.parent
        root = ET.parse(urdf_path).getroot()
        self.child_to_joint: dict[str, UrdfJoint] = {}
        self.link_meshes: dict[str, Path] = {}
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            mesh_path = self._find_link_mesh(link)
            if mesh_path is not None:
                self.link_meshes[link_name] = mesh_path
        for joint in root.findall("joint"):
            axis_el = joint.find("axis")
            axis = (
                np.fromstring(axis_el.attrib.get("xyz", "1 0 0"), sep=" ", dtype=np.float32)
                if axis_el is not None
                else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            child = joint.find("child").attrib["link"]
            self.child_to_joint[child] = UrdfJoint(
                name=joint.attrib["name"],
                joint_type=joint.attrib.get("type", "fixed"),
                parent=joint.find("parent").attrib["link"],
                origin=_origin_to_matrix(joint.find("origin")),
                axis=axis,
            )

    def _find_link_mesh(self, link: ET.Element) -> Path | None:
        for group_name in ("collision", "visual"):
            group = link.find(group_name)
            if group is None:
                continue
            mesh = group.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            return (self.urdf_dir / filename).resolve()
        return None

    def link_matrix(self, link_name: str, joint_values: dict[str, float]) -> np.ndarray:
        if link_name not in self.child_to_joint:
            return np.eye(4, dtype=np.float32)
        joint = self.child_to_joint[link_name]
        matrix = self.link_matrix(joint.parent, joint_values) @ joint.origin
        value = float(joint_values.get(joint.name, 0.0))
        if joint.joint_type in {"revolute", "continuous"}:
            matrix = matrix @ _axis_angle_to_matrix(joint.axis, value)
        elif joint.joint_type == "prismatic":
            matrix = matrix @ _translation_to_matrix(joint.axis, value)
        return matrix.astype(np.float32)


def _sample_link_points(fk: UrdfFk, link_names: list[str], total_points: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    base = total_points // len(link_names)
    remainder = total_points % len(link_names)
    samples: dict[str, np.ndarray] = {}
    for idx, link_name in enumerate(link_names):
        count = base + (1 if idx < remainder else 0)
        mesh_path = fk.link_meshes.get(link_name)
        if mesh_path is None:
            raise KeyError(f"No mesh found in URDF for link '{link_name}'.")
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
        samples[link_name] = np.asarray(points, dtype=np.float32)
    return samples


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (matrix @ hom.T).T[:, :3].astype(np.float32)


def _load_joint_series(h5_path: Path, clip_key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        clip = h5[clip_key]
        names = [name.decode("utf-8").rstrip("\x00") for name in clip["joint_names"][:]]
        qpos = np.asarray(clip["joint_positions"][:], dtype=np.float32)
        base_pose = np.asarray(clip["base_pose"][:], dtype=np.float32)
    return names, qpos, base_pose


def _gripper_points_for_frame(
    fk: UrdfFk,
    local_samples: dict[str, np.ndarray],
    joint_names: list[str],
    qpos: np.ndarray,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    joint_values = {name: float(value) for name, value in zip(joint_names, qpos[frame])}
    points = []
    colors = []
    palette = {
        "fl_link7": np.array([255, 40, 220], dtype=np.uint8),
        "fl_link8": np.array([255, 170, 30], dtype=np.uint8),
        "fr_link7": np.array([40, 210, 255], dtype=np.uint8),
        "fr_link8": np.array([80, 255, 80], dtype=np.uint8),
    }
    for link_name, local_points in local_samples.items():
        world_points = _transform_points(local_points, fk.link_matrix(link_name, joint_values))
        points.append(world_points)
        colors.append(np.broadcast_to(palette.get(link_name, np.array([255, 255, 255], dtype=np.uint8)), world_points.shape))
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def serve(args: argparse.Namespace) -> None:
    h5_path = Path(args.h5).expanduser().resolve()
    urdf_path = Path(args.urdf).expanduser().resolve()
    camera_key = args.camera
    keys = _clip_keys(h5_path, camera_key)
    if not keys:
        raise RuntimeError(f"No clips with camera '{camera_key}' found in {h5_path}")
    clip_key = _choose_motion_clip(h5_path, camera_key, keys) if args.clip == "auto" else args.clip
    current = _load_clip_view(
        h5_path,
        clip_key=clip_key,
        camera_key=camera_key,
        max_points=args.max_points,
        colormap=args.colormap,
    )
    fk = UrdfFk(urdf_path)
    gripper_links = [item.strip() for item in args.gripper_links.split(",") if item.strip()]
    local_samples = _sample_link_points(fk, gripper_links, int(args.gripper_points))
    joint_names, qpos, _base_pose = _load_joint_series(h5_path, current.clip_key)

    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.enable_default_lights()
    try:
        server.scene.set_up_direction((0.0, 0.0, 1.0))
        server.scene.world_axes.visible = False
    except Exception:
        pass
    _add_world_axes(server)

    state = {"frame": 0, "show_robot": False}
    scene_mask = ~current.robot_mask
    scene_points = current.flows[0, scene_mask]
    scene_colors = current.colors[0, scene_mask]
    scene_handle = server.scene.add_point_cloud(
        "scene/current_points",
        points=scene_points,
        colors=scene_colors,
        point_size=float(args.point_size),
        point_shape="rounded",
        precision="float32",
    )
    gripper_points, gripper_colors = _gripper_points_for_frame(fk, local_samples, joint_names, qpos, 0)
    gripper_handle = server.scene.add_point_cloud(
        "fk_gripper/points",
        points=gripper_points,
        colors=gripper_colors,
        point_size=float(args.gripper_point_size),
        point_shape="rounded",
        precision="float32",
    )
    summary = server.gui.add_markdown(_summary_markdown(h5_path, current, 0) + f"- FK gripper points: `{gripper_points.shape[0]}`\n")
    with server.gui.add_folder("controls", expand_by_default=True):
        frame_slider = server.gui.add_slider("Frame", min=0, max=current.num_frames - 1, step=1, initial_value=0)
        play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
        fps_slider = server.gui.add_slider("Play FPS", min=1.0, max=15.0, step=1.0, initial_value=float(args.fps))
        robot_checkbox = server.gui.add_checkbox("Show scene robot points", initial_value=False)
        point_size_slider = server.gui.add_slider("Scene point size", min=0.0005, max=0.02, step=0.0005, initial_value=float(args.point_size))
        gripper_size_slider = server.gui.add_slider("FK gripper point size", min=0.002, max=0.04, step=0.001, initial_value=float(args.gripper_point_size))

    def _apply_frame(frame: int) -> None:
        frame = int(np.clip(frame, 0, current.num_frames - 1))
        state["frame"] = frame
        mask = np.ones((current.num_points,), dtype=bool) if state["show_robot"] else ~current.robot_mask
        scene_handle.points = current.flows[frame, mask].astype(np.float32, copy=False)
        scene_handle.colors = current.colors[frame, mask].astype(np.uint8, copy=False)
        points, colors = _gripper_points_for_frame(fk, local_samples, joint_names, qpos, frame)
        gripper_handle.points = points
        gripper_handle.colors = colors
        summary.content = _summary_markdown(h5_path, current, frame) + f"- FK gripper points: `{points.shape[0]}`\n"

    def _frame_cb(event) -> None:
        _apply_frame(int(event.target.value))

    def _robot_cb(event) -> None:
        state["show_robot"] = bool(event.target.value)
        _apply_frame(state["frame"])

    def _point_size_cb(event) -> None:
        scene_handle.point_size = float(event.target.value)

    def _gripper_size_cb(event) -> None:
        gripper_handle.point_size = float(event.target.value)

    frame_slider.on_update(_frame_cb)
    robot_checkbox.on_update(_robot_cb)
    point_size_slider.on_update(_point_size_cb)
    gripper_size_slider.on_update(_gripper_size_cb)
    _apply_frame(0)

    print(f"VISER_URL=http://127.0.0.1:{args.port}", flush=True)
    print(f"H5={h5_path}", flush=True)
    print(f"CLIP={current.clip_key}", flush=True)
    print(f"GRIPPER_LINKS={','.join(gripper_links)}", flush=True)
    print(f"FK_GRIPPER_POINTS={gripper_points.shape}", flush=True)

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
    parser = argparse.ArgumentParser(description="Serve RobotWin point flow with FK-sampled gripper points.")
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("/tx-NFS/public_datasets/processed/pointflow_robotwin/adjust_bottle/pointworld_behavior_compact_head_sparsewm_clean_full_replay/data/episode0.hdf5"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/data/dex/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"),
    )
    parser.add_argument("--clip", type=str, default="auto")
    parser.add_argument("--camera", type=str, default="camera_head")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max_points", type=int, default=10000)
    parser.add_argument("--gripper-points", type=int, default=100)
    parser.add_argument("--gripper-links", type=str, default="fl_link7,fl_link8,fr_link7,fr_link8")
    parser.add_argument("--point_size", type=float, default=0.006)
    parser.add_argument("--gripper-point-size", type=float, default=0.02)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--colormap", type=str, default="turbo")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
