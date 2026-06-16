"""BEHAVIOR-style online decoding for RoboTwin PointWorld clips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import local_normals_to_scene_normals, local_points_to_scene_flows


@dataclass
class BehaviorOnlineCameraFlow:
    scene_flows: np.ndarray
    scene_colors: np.ndarray
    scene_normals: np.ndarray
    scene_visibility: np.ndarray
    scene_robot_mask: np.ndarray
    part_names: list[str]
    part_slices: dict[str, slice]


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    norm = np.linalg.norm(quat_xyzw)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive.")
    return quat_xyzw / norm


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = _normalize_quat_xyzw(quat_xyzw)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def pointworld_pose_to_matrix(pose_xyzw: np.ndarray) -> np.ndarray:
    """Convert PointWorld pose [x, y, z, qx, qy, qz, qw] to a 4x4 matrix."""

    pose_xyzw = np.asarray(pose_xyzw, dtype=np.float32)
    if pose_xyzw.shape != (7,):
        raise ValueError(f"Expected PointWorld pose shape (7,), got {pose_xyzw.shape}.")

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quat_xyzw_to_matrix(pose_xyzw[3:])
    matrix[:3, 3] = pose_xyzw[:3]
    return matrix


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    if hasattr(value, "decode"):
        return value.decode("utf-8").rstrip("\x00")
    return str(value).rstrip("\x00")


def _decode_string_dataset(dataset) -> list[str]:
    return [_decode_bytes(value) for value in np.asarray(dataset[:]).tolist()]


def _decode_part_names(camera_group, local_points_group) -> list[str]:
    if "scene_part_names" in camera_group:
        part_names = _decode_string_dataset(camera_group["scene_part_names"])
        missing = [name for name in part_names if name not in local_points_group]
        if missing:
            raise KeyError(f"scene_part_names references missing local_scene_points part(s): {missing}")
        return part_names
    return [str(name) for name in local_points_group.keys()]


def _decode_part_is_robot(camera_group, part_names: list[str]) -> list[bool]:
    if "scene_part_is_robot" in camera_group:
        values = np.asarray(camera_group["scene_part_is_robot"][:], dtype=bool)
        if values.shape != (len(part_names),):
            raise ValueError(
                f"scene_part_is_robot must have shape ({len(part_names)},), got {values.shape}."
            )
        return [bool(value) for value in values.tolist()]
    return [_is_robot_part_name(name) for name in part_names]


def _decode_part_point_counts(camera_group, part_names: list[str]) -> list[int] | None:
    if "scene_part_point_count" not in camera_group:
        return None
    values = np.asarray(camera_group["scene_part_point_count"][:], dtype=np.int64)
    if values.shape != (len(part_names),):
        raise ValueError(
            f"scene_part_point_count must have shape ({len(part_names)},), got {values.shape}."
        )
    return [int(value) for value in values.tolist()]


def _read_part_array(group, part_name: str, fallback_shape: tuple[int, ...], dtype) -> np.ndarray:
    if group is None or part_name not in group:
        return np.zeros(fallback_shape, dtype=dtype)
    return np.asarray(group[part_name][:], dtype=dtype)


def _decode_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals)
    if normals.dtype == np.int8:
        return normals.astype(np.float32) / 127.0
    return normals.astype(np.float32)


def _is_robot_part_name(part_name: str) -> bool:
    return part_name.startswith(("robot__", "robot0_", "gripper0_", "mount0_", "base0_"))


def decode_behavior_online_camera_group(camera_group) -> BehaviorOnlineCameraFlow:
    """Reconstruct scene flow online from local scene points and part trajectories."""

    required = ("local_scene_points", "scene_mesh_trajectories")
    missing = [name for name in required if name not in camera_group]
    if missing:
        raise KeyError(f"Camera group is missing BEHAVIOR-style field(s): {missing}")

    local_points_group = camera_group["local_scene_points"]
    local_colors_group = camera_group.get("local_scene_colors")
    local_normals_group = camera_group.get("local_scene_normals")
    trajectories_group = camera_group["scene_mesh_trajectories"]

    flow_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    robot_mask_chunks: list[np.ndarray] = []
    part_slices: dict[str, slice] = {}
    part_names = _decode_part_names(camera_group, local_points_group)
    part_is_robot = _decode_part_is_robot(camera_group, part_names)
    part_point_counts = _decode_part_point_counts(camera_group, part_names)
    offset = 0

    for part_idx, part_name in enumerate(part_names):
        if part_name not in trajectories_group:
            raise KeyError(f"Missing trajectory for local scene part '{part_name}'.")
        local_points = np.asarray(local_points_group[part_name][:], dtype=np.float32)
        if part_point_counts is not None and local_points.shape[0] != part_point_counts[part_idx]:
            raise ValueError(
                f"scene_part_point_count for '{part_name}' is {part_point_counts[part_idx]}, "
                f"but local_scene_points has {local_points.shape[0]} points."
            )
        trajectory_poses = np.asarray(trajectories_group[part_name][:], dtype=np.float32)
        if trajectory_poses.ndim != 2 or trajectory_poses.shape[1] != 7:
            raise ValueError(
                f"Expected scene_mesh_trajectories/{part_name} shape (T, 7), got {trajectory_poses.shape}."
            )

        trajectory_mats = np.asarray([pointworld_pose_to_matrix(pose) for pose in trajectory_poses], dtype=np.float32)
        flows = local_points_to_scene_flows(local_points, trajectory_mats)
        colors_part = _read_part_array(local_colors_group, part_name, local_points.shape, np.uint8)
        normals_part = _decode_normals(
            _read_part_array(local_normals_group, part_name, local_points.shape, np.float32)
        )
        colors = np.repeat(colors_part[None], flows.shape[0], axis=0)
        normals = local_normals_to_scene_normals(normals_part, trajectory_mats)

        flow_chunks.append(flows)
        color_chunks.append(colors)
        normal_chunks.append(normals)
        robot_mask_chunks.append(np.full((local_points.shape[0],), part_is_robot[part_idx], dtype=bool))
        part_slices[part_name] = slice(offset, offset + local_points.shape[0])
        offset += local_points.shape[0]

    if flow_chunks:
        scene_flows = np.concatenate(flow_chunks, axis=1).astype(np.float32)
        scene_colors = np.concatenate(color_chunks, axis=1).astype(np.uint8)
        scene_normals = np.concatenate(normal_chunks, axis=1).astype(np.float32)
        scene_robot_mask = np.concatenate(robot_mask_chunks, axis=0).astype(bool)
    else:
        scene_flows = np.zeros((0, 0, 3), dtype=np.float32)
        scene_colors = np.zeros((0, 0, 3), dtype=np.uint8)
        scene_normals = np.zeros((0, 0, 3), dtype=np.float32)
        scene_robot_mask = np.zeros((0,), dtype=bool)

    if "scene_visibility" in camera_group and camera_group["scene_visibility"].shape[:2] == scene_flows.shape[:2]:
        scene_visibility = np.asarray(camera_group["scene_visibility"][:], dtype=bool)
    else:
        scene_visibility = np.ones(scene_flows.shape[:2], dtype=bool)

    return BehaviorOnlineCameraFlow(
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        scene_normals=scene_normals,
        scene_visibility=scene_visibility,
        scene_robot_mask=scene_robot_mask,
        part_names=part_names,
        part_slices=part_slices,
    )
