"""Geometry helpers for RoboTwin to PointWorld GT-flow conversion."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass
class Frame0ActorObservations:
    points_world: np.ndarray
    colors: np.ndarray
    actor_ids: np.ndarray
    normals_world: np.ndarray


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive.")
    return quat / norm


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = _normalize_quat(quat_wxyz)
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


def sapien_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a SAPIEN pose vector [x, y, z, qw, qx, qy, qz] to a 4x4 matrix."""

    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected SAPIEN pose shape (7,), got {pose.shape}.")

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _quat_wxyz_to_matrix(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat


def _matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32)
    if rot.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rot.shape}.")

    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    return _normalize_quat(np.array([qx, qy, qz, qw], dtype=np.float32))


def matrix_to_pointworld_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix to PointWorld pose [x, y, z, qx, qy, qz, qw]."""

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4), got {matrix.shape}.")

    out = np.empty(7, dtype=np.float32)
    out[:3] = matrix[:3, 3]
    out[3:] = _matrix_to_quat_xyzw(matrix[:3, :3])
    return out


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}.")
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4), got {matrix.shape}.")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=np.float32), where=norms > 1e-9)


def transform_vectors(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Expected vectors shape (N, 3), got {vectors.shape}.")
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4), got {matrix.shape}.")
    return _normalize_vectors(vectors @ matrix[:3, :3].T)


def world_points_to_local(world_points: np.ndarray, object_pose_world: np.ndarray) -> np.ndarray:
    inv_pose = np.linalg.inv(np.asarray(object_pose_world, dtype=np.float32)).astype(np.float32)
    return transform_points(world_points, inv_pose)


def world_normals_to_local(world_normals: np.ndarray, object_pose_world: np.ndarray) -> np.ndarray:
    inv_pose = np.linalg.inv(np.asarray(object_pose_world, dtype=np.float32)).astype(np.float32)
    return transform_vectors(world_normals, inv_pose)


def local_points_to_scene_flows(local_points: np.ndarray, trajectory_matrices: np.ndarray) -> np.ndarray:
    """Apply a T-long object/link trajectory to fixed local points."""

    local_points = np.asarray(local_points, dtype=np.float32)
    trajectory_matrices = np.asarray(trajectory_matrices, dtype=np.float32)
    if trajectory_matrices.ndim != 3 or trajectory_matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected trajectory shape (T, 4, 4), got {trajectory_matrices.shape}.")

    flows = np.empty((trajectory_matrices.shape[0], local_points.shape[0], 3), dtype=np.float32)
    for t, matrix in enumerate(trajectory_matrices):
        flows[t] = transform_points(local_points, matrix)
    return flows


def local_normals_to_scene_normals(local_normals: np.ndarray, trajectory_matrices: np.ndarray) -> np.ndarray:
    """Apply only trajectory rotations to fixed local normals."""

    local_normals = np.asarray(local_normals, dtype=np.float32)
    trajectory_matrices = np.asarray(trajectory_matrices, dtype=np.float32)
    if trajectory_matrices.ndim != 3 or trajectory_matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected trajectory shape (T, 4, 4), got {trajectory_matrices.shape}.")

    normals = np.empty((trajectory_matrices.shape[0], local_normals.shape[0], 3), dtype=np.float32)
    for t, matrix in enumerate(trajectory_matrices):
        normals[t] = transform_vectors(local_normals, matrix)
    return normals


def estimate_camera_normals_from_position(
    position: np.ndarray,
    raw_actor_segmentation: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate camera-space normals from a SAPIEN Position buffer."""

    position = np.asarray(position, dtype=np.float32)
    if position.ndim != 3 or position.shape[-1] != 4:
        raise ValueError(f"Expected position shape (H, W, 4), got {position.shape}.")
    if raw_actor_segmentation is not None and raw_actor_segmentation.shape != position.shape[:2]:
        raise ValueError("raw_actor_segmentation must match position image shape.")

    points = position[..., :3]
    valid = np.isfinite(points).all(axis=-1) & (position[..., 3] < 1.0)
    normals = np.zeros_like(points, dtype=np.float32)
    if position.shape[0] < 3 or position.shape[1] < 3:
        return normals

    center_valid = valid[1:-1, 1:-1]
    neighbor_valid = valid[1:-1, :-2] & valid[1:-1, 2:] & valid[:-2, 1:-1] & valid[2:, 1:-1]
    good = center_valid & neighbor_valid
    if raw_actor_segmentation is not None:
        actors = np.asarray(raw_actor_segmentation)
        center_actor = actors[1:-1, 1:-1]
        same_actor = (
            (actors[1:-1, :-2] == center_actor)
            & (actors[1:-1, 2:] == center_actor)
            & (actors[:-2, 1:-1] == center_actor)
            & (actors[2:, 1:-1] == center_actor)
        )
        good &= same_actor

    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normal_inner = _normalize_vectors(np.cross(dx, dy))

    to_camera = -points[1:-1, 1:-1]
    flip = np.sum(normal_inner * to_camera, axis=-1) < 0.0
    normal_inner[flip] *= -1.0
    normal_inner[~good] = 0.0
    normals[1:-1, 1:-1] = normal_inner
    return normals


def _prepare_camera_normals(
    normal: np.ndarray | None,
    position: np.ndarray,
    raw_actor_segmentation: np.ndarray,
) -> np.ndarray:
    if normal is None:
        normals = estimate_camera_normals_from_position(position, raw_actor_segmentation)
    else:
        normal = np.asarray(normal, dtype=np.float32)
        if normal.shape[:2] != position.shape[:2] or normal.shape[-1] not in (3, 4):
            raise ValueError(f"Expected normal shape (H, W, 3|4) matching position, got {normal.shape}.")
        normals = normal[..., :3].astype(np.float32, copy=False)

    normals = _normalize_vectors(normals)
    to_camera = -position[..., :3]
    flip = np.sum(normals * to_camera, axis=-1) < 0.0
    normals = normals.copy()
    normals[flip] *= -1.0
    return normals


def select_frame0_actor_observations(
    *,
    position: np.ndarray,
    raw_actor_segmentation: np.ndarray,
    rgb: np.ndarray,
    cam2world_gl: np.ndarray,
    normal: np.ndarray | None = None,
    keep_actor_ids: Iterable[int] | None = None,
    drop_actor_ids: Iterable[int] | None = None,
) -> Frame0ActorObservations:
    """Select valid frame-0 pixels and transform Position/Normal buffers to world."""

    position = np.asarray(position, dtype=np.float32)
    raw_actor_segmentation = np.asarray(raw_actor_segmentation)
    rgb = np.asarray(rgb)
    cam2world_gl = np.asarray(cam2world_gl, dtype=np.float32)

    if position.ndim != 3 or position.shape[-1] != 4:
        raise ValueError(f"Expected position shape (H, W, 4), got {position.shape}.")
    if raw_actor_segmentation.shape != position.shape[:2]:
        raise ValueError("raw_actor_segmentation must match position image shape.")
    if rgb.shape[:2] != position.shape[:2] or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (H, W, 3) matching position.")
    if cam2world_gl.shape != (4, 4):
        raise ValueError(f"Expected cam2world_gl shape (4, 4), got {cam2world_gl.shape}.")

    valid = (
        np.isfinite(position[..., :3]).all(axis=-1)
        & (position[..., 3] < 1.0)
    )
    if keep_actor_ids is not None:
        keep_actor_ids_arr = np.asarray(sorted(set(int(v) for v in keep_actor_ids)), dtype=raw_actor_segmentation.dtype)
        valid &= np.isin(raw_actor_segmentation, keep_actor_ids_arr)
    if drop_actor_ids is not None:
        drop_actor_ids_arr = np.asarray(sorted(set(int(v) for v in drop_actor_ids)), dtype=raw_actor_segmentation.dtype)
        if drop_actor_ids_arr.size:
            valid &= ~np.isin(raw_actor_segmentation, drop_actor_ids_arr)

    normals_camera = _prepare_camera_normals(normal, position, raw_actor_segmentation)
    points_camera = position[..., :3][valid].astype(np.float32)
    normals_camera_selected = normals_camera[valid].astype(np.float32)
    points_world = transform_points(points_camera, cam2world_gl)
    normals_world = transform_vectors(normals_camera_selected, cam2world_gl)
    colors = rgb[valid, :3].astype(np.uint8, copy=False)
    actor_ids = raw_actor_segmentation[valid].astype(np.int32, copy=False)
    return Frame0ActorObservations(
        points_world=points_world.astype(np.float32),
        colors=colors,
        actor_ids=actor_ids,
        normals_world=normals_world.astype(np.float32),
    )


def select_frame0_actor_points(
    *,
    position: np.ndarray,
    raw_actor_segmentation: np.ndarray,
    rgb: np.ndarray,
    cam2world_gl: np.ndarray,
    keep_actor_ids: Iterable[int] | None = None,
    drop_actor_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select valid frame-0 pixels by raw actor id and transform Position points to world."""
    obs = select_frame0_actor_observations(
        position=position,
        raw_actor_segmentation=raw_actor_segmentation,
        rgb=rgb,
        cam2world_gl=cam2world_gl,
        keep_actor_ids=keep_actor_ids,
        drop_actor_ids=drop_actor_ids,
    )
    return obs.points_world, obs.colors, obs.actor_ids
