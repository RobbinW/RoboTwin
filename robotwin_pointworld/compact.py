"""Compact PointWorld/BEHAVIOR-style flow helpers for RoboTwin replay."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import h5py
import numpy as np

from .geometry import (
    local_normals_to_scene_normals,
    local_points_to_scene_flows,
    matrix_to_pointworld_pose,
    select_frame0_actor_observations,
    world_points_to_local,
    world_normals_to_local,
)


TARGET_IMAGE_SHAPE = (180, 320)

BEHAVIOR_CLIP_ATTRIBUTE_KEYS = (
    "clip_key",
    "num_frames",
    "num_scene_points",
    "has_transition",
    "any_object_moving",
    "gripper_moving",
    "has_gripper_state_change",
    "robot_nonbase_moving",
    "has_trunk_arm_collision",
    "has_left_gripper_finger_collision",
    "has_right_gripper_finger_collision",
    "max_object_pos_movement",
    "max_object_rot_movement",
    "max_gripper_pos_movement",
    "max_gripper_rot_movement",
    "max_joint_movement",
    "left_min_distance_to_moving_objects",
    "left_min_distance_to_all_objects",
    "right_min_distance_to_moving_objects",
    "right_min_distance_to_all_objects",
    "clip_complete",
)


@dataclass
class PointWorldCameraClip:
    local_scene_points: dict[str, np.ndarray]
    local_scene_colors: dict[str, np.ndarray]
    local_scene_normals: dict[str, np.ndarray]
    scene_mesh_trajectories: dict[str, np.ndarray]
    scene_part_names: list[str]
    scene_part_is_robot: np.ndarray
    scene_part_category: list[str]
    scene_part_actor_id: np.ndarray
    scene_part_point_count: np.ndarray
    scene_flows: np.ndarray
    scene_colors: np.ndarray
    scene_normals: np.ndarray
    scene_visibility: np.ndarray
    object_names: list[str]


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    if hasattr(value, "decode"):
        return value.decode("utf-8").rstrip("\x00")
    return str(value).rstrip("\x00")


def _take_point_subset(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points is None or points.shape[0] <= max_points:
        return points, colors, normals
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices], colors[indices], normals[indices]


def _quantize_normals_to_int8(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(normals).all():
        raise ValueError("Normals contain non-finite values.")
    return np.rint(np.clip(normals, -1.0, 1.0) * 127.0).astype(np.int8)


def _camera_group_name(robotwin_camera_name: str) -> str:
    if robotwin_camera_name.endswith("_camera"):
        return f"camera_{robotwin_camera_name[:-7]}"
    return f"camera_{robotwin_camera_name}"


def _as_homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = extrinsic
        return out
    raise ValueError(f"Expected extrinsic shape (3, 4) or (4, 4), got {extrinsic.shape}.")


def _normalize_camera_payload(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_h, target_w = TARGET_IMAGE_SHAPE
    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (H, W, 3), got {rgb.shape}.")
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth_m.shape} does not match rgb shape {rgb.shape[:2]}.")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {intrinsic.shape}.")

    rgb = rgb.astype(np.uint8, copy=False)
    if rgb.shape[:2] == TARGET_IMAGE_SHAPE:
        return rgb, depth_m, intrinsic

    src_h, src_w = rgb.shape[:2]
    scale_x = float(target_w) / float(src_w)
    scale_y = float(target_h) / float(src_h)
    intr_scaled = intrinsic.copy()
    intr_scaled[0, 0] *= scale_x
    intr_scaled[0, 2] *= scale_x
    intr_scaled[1, 1] *= scale_y
    intr_scaled[1, 2] *= scale_y
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return rgb_resized, depth_resized, intr_scaled


def _save_rgb_as_jpeg(group: h5py.Group, name: str, rgb: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"Failed to encode {name} as JPEG.")
    payload = encoded.tobytes()
    dset = group.create_dataset(name, data=np.asarray([payload], dtype=f"S{len(payload)}"))
    dset.attrs["write_complete"] = True


def _save_depth_uint16_mm(group: h5py.Group, name: str, depth_m: np.ndarray) -> None:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float32) * 1000.0, 0, np.iinfo(np.uint16).max)
    dset = group.create_dataset(name, data=depth_mm.astype(np.uint16))
    dset.attrs["write_complete"] = True


def _create_group_arrays(parent: h5py.Group, name: str, arrays: dict[str, np.ndarray], dtype=None) -> None:
    group = parent.create_group(name, track_order=True)
    for key, value in arrays.items():
        data = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
        dset = group.create_dataset(key, data=data)
        dset.attrs["write_complete"] = True


def _matrices_to_pointworld_poses(matrices: np.ndarray) -> np.ndarray:
    return np.asarray([matrix_to_pointworld_pose(mat) for mat in np.asarray(matrices, dtype=np.float32)], dtype=np.float32)


def _default_gripper_pose(num_frames: int) -> np.ndarray:
    out = np.zeros((num_frames, 7), dtype=np.float32)
    out[:, 6] = 1.0
    return out


def _write_default_clip_attrs(
    clip_group: h5py.Group,
    *,
    clip_key: str,
    num_frames: int,
    num_scene_points: int,
    pose_robot0_traj: np.ndarray,
    joint_positions: np.ndarray,
    left_gripper_open: np.ndarray,
    right_gripper_open: np.ndarray,
) -> None:
    translations = pose_robot0_traj[..., :3, 3]
    object_motion = 0.0
    if translations.size:
        object_motion = float(np.max(np.linalg.norm(translations - translations[:1], axis=-1)))
    max_joint_movement = 0.0
    if joint_positions.size:
        max_joint_movement = float(np.max(np.abs(joint_positions - joint_positions[:1])))
    left_grip_delta = float(np.max(np.abs(left_gripper_open - left_gripper_open[0]))) if left_gripper_open.size else 0.0
    right_grip_delta = float(np.max(np.abs(right_gripper_open - right_gripper_open[0]))) if right_gripper_open.size else 0.0
    max_grip_delta = max(left_grip_delta, right_grip_delta)

    values = {
        "clip_key": clip_key,
        "num_frames": int(num_frames),
        "num_scene_points": int(num_scene_points),
        "has_transition": bool(object_motion > 1e-5 or max_joint_movement > 1e-5),
        "any_object_moving": bool(object_motion > 1e-5),
        "gripper_moving": bool(max_grip_delta > 1e-5),
        "has_gripper_state_change": bool(max_grip_delta > 0.05),
        "robot_nonbase_moving": bool(max_joint_movement > 1e-5),
        "has_trunk_arm_collision": False,
        "has_left_gripper_finger_collision": False,
        "has_right_gripper_finger_collision": False,
        "max_object_pos_movement": float(object_motion),
        "max_object_rot_movement": 0.0,
        "max_gripper_pos_movement": 0.0,
        "max_gripper_rot_movement": 0.0,
        "max_joint_movement": float(max_joint_movement),
        "left_min_distance_to_moving_objects": -1.0,
        "left_min_distance_to_all_objects": -1.0,
        "right_min_distance_to_moving_objects": -1.0,
        "right_min_distance_to_all_objects": -1.0,
        "clip_complete": True,
    }
    for attr_name in BEHAVIOR_CLIP_ATTRIBUTE_KEYS:
        clip_group.attrs[attr_name] = values[attr_name]


def build_pointworld_camera_clip(
    *,
    position0: np.ndarray,
    raw_actor0: np.ndarray,
    rgb0: np.ndarray,
    cam2world0: np.ndarray,
    normal0: np.ndarray | None = None,
    actor_ids: np.ndarray,
    part_names: np.ndarray,
    object_names: np.ndarray,
    pose_world_traj: np.ndarray,
    robot_actor_ids: np.ndarray | None = None,
    robot_part_names: np.ndarray | None = None,
    robot_pose_world_traj: np.ndarray | None = None,
    max_points_per_part: int | None = None,
    drop_actor_ids: tuple[int, ...] = (),
    include_untracked_static: bool = True,
) -> PointWorldCameraClip:
    """Build one camera clip from frame-0 visible points and part pose trajectories.

    `pose_world_traj` must be `(T, P, 4, 4)`, where `P` matches `actor_ids`.
    Every output point is anchored to frame 0 and then reconstructed through the
    corresponding object/link trajectory.
    """

    actor_ids = np.asarray(actor_ids, dtype=np.int32)
    pose_world_traj = np.asarray(pose_world_traj, dtype=np.float32)
    if pose_world_traj.ndim != 4 or pose_world_traj.shape[1] != actor_ids.shape[0] or pose_world_traj.shape[2:] != (4, 4):
        raise ValueError(
            "pose_world_traj must have shape (T, P, 4, 4), with P matching actor_ids; "
            f"got {pose_world_traj.shape} and actor_ids {actor_ids.shape}."
        )
    num_frames = int(pose_world_traj.shape[0])
    if robot_actor_ids is None:
        robot_actor_ids = np.zeros((0,), dtype=np.int32)
    robot_actor_ids = np.asarray(robot_actor_ids, dtype=np.int32)
    if robot_part_names is None:
        robot_part_names = np.asarray([f"actor_{int(actor_id)}".encode("utf-8") for actor_id in robot_actor_ids], dtype="S128")
    robot_part_names = np.asarray(robot_part_names)
    if robot_pose_world_traj is None:
        robot_pose_world_traj = np.zeros((num_frames, 0, 4, 4), dtype=np.float32)
    robot_pose_world_traj = np.asarray(robot_pose_world_traj, dtype=np.float32)
    if robot_pose_world_traj.ndim != 4 or robot_pose_world_traj.shape[0] != num_frames or robot_pose_world_traj.shape[2:] != (4, 4):
        raise ValueError(f"robot_pose_world_traj must have shape (T, R, 4, 4), got {robot_pose_world_traj.shape}.")
    if robot_actor_ids.shape[0] != robot_pose_world_traj.shape[1]:
        if robot_pose_world_traj.shape[1] == 0:
            robot_actor_ids = np.zeros((0,), dtype=np.int32)
            robot_part_names = np.zeros((0,), dtype="S128")
        else:
            raise ValueError(
                "robot_pose_world_traj part dimension must match robot_actor_ids; "
                f"got {robot_pose_world_traj.shape[1]} and {robot_actor_ids.shape[0]}."
            )
    if robot_part_names.shape[0] != robot_actor_ids.shape[0]:
        robot_part_names = np.asarray([f"actor_{int(actor_id)}".encode("utf-8") for actor_id in robot_actor_ids], dtype="S128")
    if robot_actor_ids.size:
        unique_indices: list[int] = []
        seen_robot_actor_ids: set[int] = set()
        for idx, actor_id in enumerate(robot_actor_ids.tolist()):
            actor_id_int = int(actor_id)
            if actor_id_int < 0 or actor_id_int in seen_robot_actor_ids:
                continue
            seen_robot_actor_ids.add(actor_id_int)
            unique_indices.append(idx)
        unique_indices_np = np.asarray(unique_indices, dtype=np.int64)
        robot_actor_ids = robot_actor_ids[unique_indices_np]
        robot_part_names = robot_part_names[unique_indices_np]
        robot_pose_world_traj = robot_pose_world_traj[:, unique_indices_np]

    drop_actor_id_set = {int(v) for v in drop_actor_ids if int(v) >= 0}
    tracked_actor_ids = {int(v) for v in actor_ids.tolist() if int(v) >= 0 and int(v) not in drop_actor_id_set}
    keep_actor_ids = None if include_untracked_static else tracked_actor_ids
    frame0 = select_frame0_actor_observations(
        position=position0,
        raw_actor_segmentation=raw_actor0,
        rgb=rgb0,
        cam2world_gl=cam2world0,
        normal=normal0,
        keep_actor_ids=keep_actor_ids,
        drop_actor_ids=drop_actor_id_set,
    )
    points_world0 = frame0.points_world
    colors0 = frame0.colors
    normals_world0 = frame0.normals_world
    point_actor_ids = frame0.actor_ids

    local_scene_points: dict[str, np.ndarray] = {}
    local_scene_colors: dict[str, np.ndarray] = {}
    local_scene_normals: dict[str, np.ndarray] = {}
    scene_mesh_trajectories: dict[str, np.ndarray] = {}
    scene_part_names: list[str] = []
    scene_part_is_robot: list[bool] = []
    scene_part_category: list[str] = []
    scene_part_actor_id: list[int] = []
    scene_part_point_count: list[int] = []
    object_name_list: list[str] = []
    flow_chunks, color_chunks, normal_chunks = [], [], []

    used_actor_ids: set[int] = set()
    for part_idx, actor_id in enumerate(actor_ids.tolist()):
        if int(actor_id) < 0:
            continue
        if int(actor_id) in drop_actor_id_set:
            continue
        mask = point_actor_ids == int(actor_id)
        if not np.any(mask):
            continue

        part_name = _decode_bytes(part_names[part_idx])
        object_name = _decode_bytes(object_names[part_idx])
        points_part_world0, colors_part, normals_part_world0 = _take_point_subset(
            points_world0[mask].astype(np.float32),
            colors0[mask].astype(np.uint8),
            normals_world0[mask].astype(np.float32),
            max_points_per_part,
        )
        local_points = world_points_to_local(points_part_world0, pose_world_traj[0, part_idx]).astype(np.float32)
        local_normals = world_normals_to_local(normals_part_world0, pose_world_traj[0, part_idx]).astype(np.float32)
        traj_mats = pose_world_traj[:, part_idx].astype(np.float32)
        traj_poses = np.asarray([matrix_to_pointworld_pose(mat) for mat in traj_mats], dtype=np.float32)
        flows = local_points_to_scene_flows(local_points, traj_mats)
        normals = local_normals_to_scene_normals(local_normals, traj_mats)
        colors = np.repeat(colors_part[None], flows.shape[0], axis=0)

        local_scene_points[part_name] = local_points
        local_scene_colors[part_name] = colors_part
        local_scene_normals[part_name] = local_normals
        scene_mesh_trajectories[part_name] = traj_poses
        scene_part_names.append(part_name)
        scene_part_is_robot.append(False)
        scene_part_category.append("task_object")
        scene_part_actor_id.append(int(actor_id))
        scene_part_point_count.append(int(local_points.shape[0]))
        object_name_list.append(object_name)
        flow_chunks.append(flows)
        color_chunks.append(colors)
        normal_chunks.append(normals)
        used_actor_ids.add(int(actor_id))

    robot_actor_id_set = {int(v) for v in robot_actor_ids.tolist() if int(v) >= 0}
    for robot_idx, actor_id in enumerate(robot_actor_ids.tolist()):
        if int(actor_id) < 0:
            continue
        if int(actor_id) in drop_actor_id_set:
            continue
        mask = point_actor_ids == int(actor_id)
        if not np.any(mask):
            continue

        robot_name = _decode_bytes(robot_part_names[robot_idx])
        part_name = f"robot__{robot_name}"
        points_part_world0, colors_part, normals_part_world0 = _take_point_subset(
            points_world0[mask].astype(np.float32),
            colors0[mask].astype(np.uint8),
            normals_world0[mask].astype(np.float32),
            max_points_per_part,
        )
        local_points = world_points_to_local(points_part_world0, robot_pose_world_traj[0, robot_idx]).astype(np.float32)
        local_normals = world_normals_to_local(normals_part_world0, robot_pose_world_traj[0, robot_idx]).astype(np.float32)
        traj_mats = robot_pose_world_traj[:, robot_idx].astype(np.float32)
        traj_poses = np.asarray([matrix_to_pointworld_pose(mat) for mat in traj_mats], dtype=np.float32)
        flows = local_points_to_scene_flows(local_points, traj_mats)
        normals = local_normals_to_scene_normals(local_normals, traj_mats)
        colors = np.repeat(colors_part[None], flows.shape[0], axis=0)

        local_scene_points[part_name] = local_points
        local_scene_colors[part_name] = colors_part
        local_scene_normals[part_name] = local_normals
        scene_mesh_trajectories[part_name] = traj_poses
        scene_part_names.append(part_name)
        scene_part_is_robot.append(True)
        scene_part_category.append("robot")
        scene_part_actor_id.append(int(actor_id))
        scene_part_point_count.append(int(local_points.shape[0]))
        object_name_list.append(part_name)
        flow_chunks.append(flows)
        color_chunks.append(colors)
        normal_chunks.append(normals)
        used_actor_ids.add(int(actor_id))

    if include_untracked_static:
        static_pose_traj = np.repeat(np.eye(4, dtype=np.float32)[None], pose_world_traj.shape[0], axis=0)
        static_pose_pw = np.asarray([matrix_to_pointworld_pose(mat) for mat in static_pose_traj], dtype=np.float32)
        for actor_id in sorted(int(v) for v in np.unique(point_actor_ids).tolist()):
            if actor_id in used_actor_ids or actor_id in drop_actor_id_set:
                continue
            if actor_id in robot_actor_id_set:
                continue
            mask = point_actor_ids == actor_id
            if not np.any(mask):
                continue
            part_name = f"static_actor_{actor_id}"
            points_part_world0, colors_part, normals_part_world0 = _take_point_subset(
                points_world0[mask].astype(np.float32),
                colors0[mask].astype(np.uint8),
                normals_world0[mask].astype(np.float32),
                max_points_per_part,
            )
            local_points = points_part_world0.astype(np.float32, copy=False)
            local_normals = normals_part_world0.astype(np.float32, copy=False)
            flows = np.repeat(local_points[None], pose_world_traj.shape[0], axis=0).astype(np.float32)
            normals = np.repeat(local_normals[None], pose_world_traj.shape[0], axis=0).astype(np.float32)
            colors = np.repeat(colors_part[None], flows.shape[0], axis=0)

            local_scene_points[part_name] = local_points
            local_scene_colors[part_name] = colors_part
            local_scene_normals[part_name] = local_normals
            scene_mesh_trajectories[part_name] = static_pose_pw
            scene_part_names.append(part_name)
            scene_part_is_robot.append(False)
            scene_part_category.append("static")
            scene_part_actor_id.append(int(actor_id))
            scene_part_point_count.append(int(local_points.shape[0]))
            object_name_list.append(part_name)
            flow_chunks.append(flows)
            color_chunks.append(colors)
            normal_chunks.append(normals)

    if flow_chunks:
        scene_flows = np.concatenate(flow_chunks, axis=1).astype(np.float32)
        scene_colors = np.concatenate(color_chunks, axis=1).astype(np.uint8)
        scene_normals = np.concatenate(normal_chunks, axis=1).astype(np.float32)
    else:
        scene_flows = np.zeros((pose_world_traj.shape[0], 0, 3), dtype=np.float32)
        scene_colors = np.zeros((pose_world_traj.shape[0], 0, 3), dtype=np.uint8)
        scene_normals = np.zeros((pose_world_traj.shape[0], 0, 3), dtype=np.float32)

    scene_visibility = np.ones(scene_flows.shape[:2], dtype=bool)

    return PointWorldCameraClip(
        local_scene_points=local_scene_points,
        local_scene_colors=local_scene_colors,
        local_scene_normals=local_scene_normals,
        scene_mesh_trajectories=scene_mesh_trajectories,
        scene_part_names=scene_part_names,
        scene_part_is_robot=np.asarray(scene_part_is_robot, dtype=bool),
        scene_part_category=scene_part_category,
        scene_part_actor_id=np.asarray(scene_part_actor_id, dtype=np.int32),
        scene_part_point_count=np.asarray(scene_part_point_count, dtype=np.int32),
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        scene_normals=scene_normals,
        scene_visibility=scene_visibility,
        object_names=sorted(set(object_name_list)),
    )


def _write_camera_clip_group(
    clip_group: h5py.Group,
    *,
    camera_group_name: str,
    camera_clip: PointWorldCameraClip,
    initial_rgb: np.ndarray,
    initial_depth_m: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    extrinsic_trajectory: np.ndarray,
) -> None:
    camera_group = clip_group.create_group(camera_group_name, track_order=True)
    _create_group_arrays(camera_group, "local_scene_points", camera_clip.local_scene_points, dtype=np.float16)
    _create_group_arrays(camera_group, "local_scene_colors", camera_clip.local_scene_colors, dtype=np.uint8)
    local_normals_i8 = {name: _quantize_normals_to_int8(normals) for name, normals in camera_clip.local_scene_normals.items()}
    _create_group_arrays(camera_group, "local_scene_normals", local_normals_i8, dtype=np.int8)
    _create_group_arrays(camera_group, "scene_mesh_trajectories", camera_clip.scene_mesh_trajectories, dtype=np.float32)
    camera_group.create_dataset("scene_part_names", data=np.asarray(camera_clip.scene_part_names, dtype="S128"))
    camera_group.create_dataset("scene_part_is_robot", data=camera_clip.scene_part_is_robot, dtype=bool)
    camera_group.create_dataset("scene_part_category", data=np.asarray(camera_clip.scene_part_category, dtype="S32"))
    camera_group.create_dataset("scene_part_actor_id", data=camera_clip.scene_part_actor_id, dtype=np.int32)
    camera_group.create_dataset("scene_part_point_count", data=camera_clip.scene_part_point_count, dtype=np.int32)

    rgb_norm, depth_norm, intrinsic_norm = _normalize_camera_payload(initial_rgb, initial_depth_m, intrinsic)
    camera_group.create_dataset("intrinsic", data=intrinsic_norm, dtype=np.float32)
    camera_group.create_dataset("extrinsic", data=np.asarray(extrinsic, dtype=np.float32), dtype=np.float32)
    camera_group.create_dataset("extrinsic_trajectory", data=np.asarray(extrinsic_trajectory, dtype=np.float32), dtype=np.float32)
    _save_rgb_as_jpeg(camera_group, "initial_rgb", rgb_norm)
    _save_depth_uint16_mm(camera_group, "initial_depth", depth_norm)
    camera_group.attrs["num_scene_points"] = int(camera_clip.scene_flows.shape[1])
