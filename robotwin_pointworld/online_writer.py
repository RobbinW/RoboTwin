"""Streaming BEHAVIOR-style compact writer for RoboTwin replay observations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .compact import (
    _as_homogeneous_extrinsic,
    _camera_group_name,
    _default_gripper_pose,
    _matrices_to_pointworld_poses,
    _write_camera_clip_group,
    _write_default_clip_attrs,
    build_pointworld_camera_clip,
)
from .geometry import matrix_to_pointworld_pose, sapien_pose_to_matrix


class BehaviorCompactEpisodeWriter:
    """Write compact PointWorld/BEHAVIOR-style clips directly from replay frames."""

    def __init__(
        self,
        *,
        output_h5_path: str | Path,
        clip_len: int = 15,
        stride: int = 15,
        frame_interval: int = 1,
        camera_name: str = "head_camera",
        min_object_motion: float = 0.0,
        max_points_per_part: int | None = None,
        drop_actor_ids: tuple[int, ...] = (),
    ):
        self.output_h5_path = Path(output_h5_path)
        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        if self.clip_len <= 0:
            raise ValueError("clip_len must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        self.frame_interval = int(frame_interval)
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be positive.")
        self.required_input_frames = (self.clip_len - 1) * self.frame_interval + 1
        self.camera_name = str(camera_name)
        self.min_object_motion = float(min_object_motion)
        self.max_points_per_part = max_points_per_part
        self.drop_actor_ids = tuple(int(v) for v in drop_actor_ids)
        self._frames: list[dict] = []
        self._appended_frames = 0
        self._written = 0
        self._closed = False
        self._h5 = h5py.File(self.output_h5_path, "w")
        self._h5.attrs["domain"] = "robotwin"
        self._h5.attrs["format"] = "behavior_compact"
        self._h5.attrs["camera_name"] = self.camera_name
        self._h5.attrs["clip_len"] = self.clip_len
        self._h5.attrs["stride"] = self.stride
        self._h5.attrs["frame_interval"] = self.frame_interval
        self._h5.attrs["episode_complete"] = False

    @property
    def written_clips(self) -> int:
        return self._written

    def append(self, frame: dict) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed BehaviorCompactEpisodeWriter.")
        frame = dict(frame)
        metadata = dict(frame.get("frame_metadata", {}))
        metadata.setdefault("demo_clean_frame_index", self._appended_frames)
        metadata.setdefault("traj_frame_index", metadata["demo_clean_frame_index"])
        metadata.setdefault("save_freq", -1)
        frame["frame_metadata"] = metadata
        self._appended_frames += 1
        self._frames.append(frame)
        while len(self._frames) >= self.required_input_frames:
            sampled_frames = self._frames[: self.required_input_frames : self.frame_interval]
            self._write_clip(sampled_frames)
            del self._frames[: self.stride]

    def close(self) -> None:
        if self._closed:
            return
        if self._frames:
            padded_frames = self._pad_residual_frames(self._frames)
            sampled_frames = padded_frames[: self.required_input_frames : self.frame_interval]
            self._write_clip(sampled_frames)
            self._frames.clear()
        self._h5.attrs["num_clips"] = int(self._written)
        self._h5.attrs["episode_complete"] = True
        self._h5.flush()
        self._h5.close()
        self._closed = True

    def _pad_residual_frames(self, frames: list[dict]) -> list[dict]:
        if len(frames) >= self.required_input_frames:
            return frames[: self.required_input_frames]
        padded = list(frames)
        pad_count = self.required_input_frames - len(padded)
        last_frame = padded[-1]
        for _ in range(pad_count):
            frame = dict(last_frame)
            metadata = dict(frame.get("frame_metadata", {}))
            metadata["pointworld_is_padding"] = True
            frame["frame_metadata"] = metadata
            padded.append(frame)
        return padded

    def _write_clip(self, frames: list[dict]) -> None:
        if len(frames) != self.clip_len:
            raise ValueError(f"Expected {self.clip_len} frames, got {len(frames)}.")
        for frame_idx, frame in enumerate(frames):
            if "observation" not in frame or self.camera_name not in frame["observation"]:
                raise KeyError(f"Frame {frame_idx} is missing observation/{self.camera_name}.")
            if "flow_parts" not in frame:
                raise KeyError(f"Frame {frame_idx} is missing flow_parts.")

        first = frames[0]
        raw_camera = first["observation"][self.camera_name]
        required = ("position", "raw_actor_segmentation", "rgb", "cam2world_gl", "intrinsic_cv", "extrinsic_cv")
        missing = [name for name in required if name not in raw_camera]
        if missing:
            raise KeyError(f"observation/{self.camera_name} is missing required field(s): {missing}")

        actor_ids = np.asarray(first["flow_parts"]["actor_ids"], dtype=np.int32)
        part_names = np.asarray(first["flow_parts"]["part_names"])
        object_names = np.asarray(first["flow_parts"]["object_names"])
        pose_world_traj = np.stack(
            [np.asarray(frame["flow_parts"]["pose_world"], dtype=np.float32) for frame in frames],
            axis=0,
        )

        base_world_traj = self._stack_robot_matrix(frames, "base_pose_world", fallback=np.eye(4, dtype=np.float32))
        world_to_robot0 = np.linalg.inv(base_world_traj[0]).astype(np.float32)
        pose_robot0_traj = np.matmul(world_to_robot0[None, None], pose_world_traj).astype(np.float32)

        translations = pose_robot0_traj[..., :3, 3]
        object_motion = float(np.max(np.linalg.norm(translations - translations[:1], axis=-1))) if translations.size else 0.0
        if object_motion < self.min_object_motion:
            self._advance_after_skip()
            return

        robot_actor_ids, robot_part_names, robot_pose_robot0_traj = self._robot_flow_parts(frames, world_to_robot0)
        base_robot0_traj = np.matmul(world_to_robot0[None], base_world_traj).astype(np.float32)

        rgb0 = np.asarray(raw_camera["rgb"], dtype=np.uint8)
        depth0 = self._depth_meters(raw_camera)
        cam2robot0 = (world_to_robot0 @ np.asarray(raw_camera["cam2world_gl"], dtype=np.float32)).astype(np.float32)
        camera_clip = build_pointworld_camera_clip(
            position0=np.asarray(raw_camera["position"], dtype=np.float32),
            raw_actor0=np.asarray(raw_camera["raw_actor_segmentation"], dtype=np.int32),
            rgb0=rgb0,
            cam2world0=cam2robot0,
            normal0=np.asarray(raw_camera["normal"], dtype=np.float32) if "normal" in raw_camera else None,
            actor_ids=actor_ids,
            part_names=part_names,
            object_names=object_names,
            pose_world_traj=pose_robot0_traj,
            robot_actor_ids=robot_actor_ids,
            robot_part_names=robot_part_names,
            robot_pose_world_traj=robot_pose_robot0_traj,
            max_points_per_part=self.max_points_per_part,
            drop_actor_ids=self.drop_actor_ids,
        )

        clip_key = f"{self.output_h5_path.stem}:clip{self._written:06d}"
        clip_group = self._h5.create_group(clip_key, track_order=True)
        self._write_source_frame_metadata(clip_group, frames)
        extrinsic = _as_homogeneous_extrinsic(raw_camera["extrinsic_cv"]) @ base_world_traj[0]
        extrinsic_traj = np.asarray(
            [
                _as_homogeneous_extrinsic(frame["observation"][self.camera_name]["extrinsic_cv"]) @ base_world_traj[0]
                for frame in frames
            ],
            dtype=np.float32,
        )
        _write_camera_clip_group(
            clip_group,
            camera_group_name=_camera_group_name(self.camera_name),
            camera_clip=camera_clip,
            initial_rgb=rgb0,
            initial_depth_m=depth0,
            intrinsic=np.asarray(raw_camera["intrinsic_cv"], dtype=np.float32),
            extrinsic=extrinsic,
            extrinsic_trajectory=extrinsic_traj,
        )

        joint_positions, joint_names, left_gripper_open, right_gripper_open = self._robot_series(frames)
        clip_group.create_dataset("joint_positions", data=joint_positions, dtype=np.float32)
        clip_group.create_dataset("joint_names", data=joint_names)
        clip_group.create_dataset("base_pose", data=_matrices_to_pointworld_poses(base_robot0_traj), dtype=np.float32)
        clip_group.create_dataset("left_gripper_open", data=left_gripper_open, dtype=np.float32)
        clip_group.create_dataset("right_gripper_open", data=right_gripper_open, dtype=np.float32)
        clip_group.create_dataset(
            "left_gripper_pose",
            data=self._gripper_pose(frames, "left_endpose", world_to_robot0),
            dtype=np.float32,
        )
        clip_group.create_dataset(
            "right_gripper_pose",
            data=self._gripper_pose(frames, "right_endpose", world_to_robot0),
            dtype=np.float32,
        )
        clip_group.create_dataset("object_names", data=np.asarray(sorted(set(camera_clip.object_names)), dtype="S128"))
        clip_group.attrs["domain"] = "robotwin"

        _write_default_clip_attrs(
            clip_group,
            clip_key=clip_key,
            num_frames=self.clip_len,
            num_scene_points=int(camera_clip.scene_flows.shape[1]),
            pose_robot0_traj=pose_robot0_traj,
            joint_positions=joint_positions,
            left_gripper_open=left_gripper_open,
            right_gripper_open=right_gripper_open,
        )
        self._written += 1
        self._h5.flush()

    def _advance_after_skip(self) -> None:
        return

    def _write_source_frame_metadata(self, clip_group: h5py.Group, frames: list[dict]) -> None:
        demo_indices = np.asarray(
            [int(frame.get("frame_metadata", {}).get("demo_clean_frame_index", idx)) for idx, frame in enumerate(frames)],
            dtype=np.int64,
        )
        traj_indices = np.asarray(
            [int(frame.get("frame_metadata", {}).get("traj_frame_index", demo_indices[idx])) for idx, frame in enumerate(frames)],
            dtype=np.int64,
        )
        save_freqs = np.asarray(
            [int(frame.get("frame_metadata", {}).get("save_freq", -1)) for frame in frames],
            dtype=np.int64,
        )
        is_padding = np.asarray(
            [bool(frame.get("frame_metadata", {}).get("pointworld_is_padding", False)) for frame in frames],
            dtype=bool,
        )
        clip_group.create_dataset("source_demo_clean_frame_indices", data=demo_indices, dtype=np.int64)
        clip_group.create_dataset("source_traj_frame_indices", data=traj_indices, dtype=np.int64)
        clip_group.create_dataset("source_save_freq", data=save_freqs, dtype=np.int64)
        clip_group.create_dataset("source_frame_is_padding", data=is_padding, dtype=bool)
        clip_group.attrs["source_demo_clean_start_frame"] = int(demo_indices[0]) if demo_indices.size else -1
        clip_group.attrs["source_demo_clean_end_frame"] = int(demo_indices[-1]) if demo_indices.size else -1
        clip_group.attrs["source_traj_start_frame"] = int(traj_indices[0]) if traj_indices.size else -1
        clip_group.attrs["source_traj_end_frame"] = int(traj_indices[-1]) if traj_indices.size else -1
        clip_group.attrs["source_frame_interval"] = int(self.frame_interval)
        clip_group.attrs["source_clip_stride"] = int(self.stride)
        clip_group.attrs["source_is_padded_clip"] = bool(np.any(is_padding))
        clip_group.attrs["source_num_padding_frames"] = int(np.count_nonzero(is_padding))

    def _stack_robot_matrix(self, frames: list[dict], key: str, fallback: np.ndarray) -> np.ndarray:
        values = []
        for frame in frames:
            robot_state = frame.get("robot_state", {})
            values.append(np.asarray(robot_state.get(key, fallback), dtype=np.float32))
        return np.stack(values, axis=0).astype(np.float32)

    def _robot_flow_parts(self, frames: list[dict], world_to_robot0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first_robot = frames[0].get("robot_state", {})
        if "robot_actor_ids" not in first_robot or "robot_pose_world" not in first_robot:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype="S128"),
                np.zeros((self.clip_len, 0, 4, 4), dtype=np.float32),
            )
        actor_ids = np.asarray(first_robot["robot_actor_ids"], dtype=np.int32)
        part_names = np.asarray(
            first_robot.get(
                "robot_part_names",
                np.asarray([f"actor_{int(actor_id)}".encode("utf-8") for actor_id in actor_ids], dtype="S128"),
            )
        )
        pose_world_traj = np.stack(
            [np.asarray(frame.get("robot_state", {})["robot_pose_world"], dtype=np.float32) for frame in frames],
            axis=0,
        )
        pose_robot0_traj = np.matmul(world_to_robot0[None, None], pose_world_traj).astype(np.float32)
        return actor_ids, part_names, pose_robot0_traj

    def _robot_series(self, frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        first_robot = frames[0].get("robot_state", {})
        if "joint_positions" in first_robot:
            joint_positions = np.stack(
                [np.asarray(frame.get("robot_state", {})["joint_positions"], dtype=np.float32) for frame in frames],
                axis=0,
            )
            joint_names = np.asarray(first_robot.get("joint_names", []))
            left_gripper_open = np.asarray(
                [frame.get("robot_state", {}).get("left_gripper_open", 0.0) for frame in frames],
                dtype=np.float32,
            )
            right_gripper_open = np.asarray(
                [frame.get("robot_state", {}).get("right_gripper_open", 0.0) for frame in frames],
                dtype=np.float32,
            )
        else:
            joint_positions = np.stack(
                [np.asarray(frame.get("joint_action", {}).get("vector", []), dtype=np.float32) for frame in frames],
                axis=0,
            )
            joint_names = np.asarray([f"joint_{idx}".encode("utf-8") for idx in range(joint_positions.shape[1])])
            left_gripper_open = np.zeros((self.clip_len,), dtype=np.float32)
            right_gripper_open = np.zeros((self.clip_len,), dtype=np.float32)
        return joint_positions, joint_names, left_gripper_open.reshape(-1), right_gripper_open.reshape(-1)

    def _gripper_pose(self, frames: list[dict], key: str, world_to_robot0: np.ndarray) -> np.ndarray:
        if "endpose" not in frames[0] or key not in frames[0]["endpose"]:
            return _default_gripper_pose(self.clip_len)
        poses = []
        for frame in frames:
            pose = np.asarray(frame["endpose"][key], dtype=np.float32)
            poses.append(matrix_to_pointworld_pose(world_to_robot0 @ sapien_pose_to_matrix(pose)))
        return np.asarray(poses, dtype=np.float32)

    @staticmethod
    def _depth_meters(camera_frame: dict) -> np.ndarray:
        if "depth" in camera_frame:
            depth = np.asarray(camera_frame["depth"], dtype=np.float32)
            if depth.size and np.nanmax(depth) > 20.0:
                depth = depth / 1000.0
            return depth
        position = np.asarray(camera_frame["position"], dtype=np.float32)
        return np.maximum(-position[..., 2], 0.0).astype(np.float32)
