#!/usr/bin/env python3
"""Convert RoboTwin compact PointWorld H5 episodes to WebDataset shards.

The output keeps the compact representation:
local_scene_points + scene_mesh_trajectories are stored instead of dense
scene_flows. 3dwam/PointWorld decoders can reconstruct scene_flows online.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import webdataset as wds
from tqdm import tqdm

DEFAULT_TEST_PERCENTAGE = 0.1
DEFAULT_MAX_HISTORY_HORIZON = 15

COMPACT_DATA_KEYS = [
    "local_scene_points",
    "local_scene_colors",
    "local_scene_normals",
    "scene_mesh_trajectories",
    "left_gripper_open",
    "left_gripper_pose",
    "right_gripper_open",
    "right_gripper_pose",
    "joint_positions",
    "joint_names",
    "base_pose",
    "initial_rgb",
    "initial_depth",
    "intrinsic",
    "extrinsic",
    "source_demo_clean_frame_indices",
    "source_traj_frame_indices",
    "source_frame_is_padding",
    "source_save_freq",
    "history_joint_positions",
    "history_valid_mask",
    "history_raw_indices",
    "robotwin_source_metadata",
]

CLIP_ATTRIBUTE_KEYS = [
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
]


def _parse_csv_arg(value: str) -> list[str]:
    if value.strip() == "":
        return []
    if value.lower() in {"all", "none"}:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def _episode_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode"):
        return 10**12
    suffix = stem[len("episode") :]
    return int(suffix) if suffix.isdigit() else 10**12


def _task_from_robotwin_h5_path(h5_path: str | Path, input_dir: str | Path) -> str:
    rel_path = os.path.relpath(os.path.abspath(h5_path), os.path.abspath(input_dir)).replace("\\", "/")
    parts = rel_path.split("/")
    if len(parts) < 4:
        raise AssertionError(
            "RoboTwin path does not match '<task>/<config>/data/episode<N>.hdf5': "
            f"{h5_path}"
        )
    return parts[0]


def _robotwin_episode_id(h5_path: str | Path, input_dir: str | Path) -> str:
    task = _task_from_robotwin_h5_path(h5_path, input_dir)
    return f"{task}_{Path(h5_path).stem}"


def _clip_id(h5_path: str | Path, clip_key: str, input_dir: str | Path) -> str:
    return f"{_robotwin_episode_id(h5_path, input_dir)}-{clip_key}"


def _iter_robotwin_h5_files(
    input_dir: Path,
    *,
    tasks: list[str],
    max_episodes_per_task: int,
) -> list[Path]:
    task_filter = set(tasks)
    selected: list[Path] = []
    for task_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if task_filter and task_dir.name not in task_filter:
            continue
        files = sorted(
            task_dir.glob("*/data/episode*.hdf5"),
            key=lambda path: (_episode_number(path), str(path)),
        )
        if max_episodes_per_task > 0:
            files = files[:max_episodes_per_task]
        selected.extend(files)
    return selected


def _serialize_numpy(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value)
    return buffer.getvalue()


def _serialize_pickle(value: Any) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(value, buffer)
    return buffer.getvalue()


def _dataset_value(dataset: h5py.Dataset) -> np.ndarray:
    return dataset[()] if dataset.shape == () else dataset[:]


@lru_cache(maxsize=64)
def _load_episode_history_frame_index(h5_path: str) -> dict[str, Any]:
    joint_by_frame: dict[int, np.ndarray] = {}
    padding_by_frame: dict[int, bool] = {}
    with h5py.File(h5_path, "r") as h5:
        for clip_key, clip in h5.items():
            if not isinstance(clip, h5py.Group) or ":" not in clip_key:
                continue
            if "source_demo_clean_frame_indices" not in clip or "joint_positions" not in clip:
                continue
            frame_indices = np.asarray(clip["source_demo_clean_frame_indices"][:], dtype=np.int64)
            padding = np.asarray(
                clip["source_frame_is_padding"][:] if "source_frame_is_padding" in clip else np.zeros_like(frame_indices, dtype=bool),
                dtype=np.bool_,
            )
            joint_positions = np.asarray(clip["joint_positions"][:], dtype=np.float32)
            for i, frame in enumerate(frame_indices):
                frame_int = int(frame)
                if frame_int not in joint_by_frame or bool(padding_by_frame.get(frame_int, True)):
                    joint_by_frame[frame_int] = joint_positions[i].copy()
                    padding_by_frame[frame_int] = bool(padding[i])

    if not joint_by_frame:
        raise ValueError(f"No RobotWin frame index could be built from {h5_path}")
    return {
        "joint_by_frame": joint_by_frame,
        "padding_by_frame": padding_by_frame,
        "sorted_frames": np.asarray(sorted(joint_by_frame.keys()), dtype=np.int64),
    }


def _nearest_available_history_frame(target: int, sorted_frames: np.ndarray) -> int:
    pos = int(np.searchsorted(sorted_frames, target, side="right") - 1)
    pos = max(0, min(pos, len(sorted_frames) - 1))
    return int(sorted_frames[pos])


def _history_robot_state_payload(
    h5_path: str,
    clip_key: str,
    *,
    max_history_horizon: int,
) -> dict[str, np.ndarray]:
    max_history_horizon = int(max_history_horizon)
    if max_history_horizon <= 0:
        return {}

    with h5py.File(h5_path, "r") as h5:
        clip = h5[clip_key]
        current_indices = np.asarray(clip["source_demo_clean_frame_indices"][:], dtype=np.int64)
        if current_indices.size == 0:
            raise ValueError(f"{h5_path}:{clip_key} source_demo_clean_frame_indices is empty")
        if current_indices.size >= 2:
            frame_interval = int(current_indices[1] - current_indices[0])
        else:
            frame_interval = int(h5.attrs.get("frame_interval", 1))
        if frame_interval <= 0:
            frame_interval = 1
        t0 = int(current_indices[0])
        current_joint = np.asarray(clip["joint_positions"][0], dtype=np.float32)

    raw_indices = np.arange(
        t0 - max_history_horizon * frame_interval,
        t0 + frame_interval,
        frame_interval,
        dtype=np.int64,
    )
    index = _load_episode_history_frame_index(str(h5_path))
    sorted_frames = index["sorted_frames"]
    joint_by_frame = index["joint_by_frame"]
    padding_by_frame = index["padding_by_frame"]

    joint_rows = []
    valid_mask = []
    for frame in raw_indices:
        frame_int = int(frame)
        exact = frame_int in joint_by_frame and not bool(padding_by_frame.get(frame_int, False))
        source_frame = frame_int if exact else _nearest_available_history_frame(frame_int, sorted_frames)
        joint_rows.append(np.asarray(joint_by_frame[source_frame], dtype=np.float32))
        valid_mask.append(bool(exact))

    history_joint_positions = np.stack(joint_rows, axis=0)
    history_joint_positions[-1] = current_joint
    valid_mask[-1] = True
    return {
        "history_joint_positions": history_joint_positions,
        "history_valid_mask": np.asarray(valid_mask, dtype=np.bool_),
        "history_raw_indices": raw_indices,
    }


def _validate_camera_group(camera: h5py.Group) -> list[str]:
    issues: list[str] = []
    required = [
        "local_scene_points",
        "local_scene_colors",
        "local_scene_normals",
        "scene_mesh_trajectories",
        "initial_rgb",
        "initial_depth",
        "intrinsic",
        "extrinsic",
    ]
    for key in required:
        if key not in camera:
            issues.append(f"missing camera_head/{key}")
    if "initial_rgb" in camera and tuple(camera["initial_rgb"].shape) != (1,):
        issues.append(f"initial_rgb expected shape (1,), got {camera['initial_rgb'].shape}")
    if "initial_depth" in camera and tuple(camera["initial_depth"].shape) != (180, 320):
        issues.append(f"initial_depth expected shape (180, 320), got {camera['initial_depth'].shape}")
    if "intrinsic" in camera and tuple(camera["intrinsic"].shape) != (3, 3):
        issues.append(f"intrinsic expected shape (3, 3), got {camera['intrinsic'].shape}")
    if "extrinsic" in camera and tuple(camera["extrinsic"].shape) != (4, 4):
        issues.append(f"extrinsic expected shape (4, 4), got {camera['extrinsic'].shape}")
    if "local_scene_normals" in camera:
        for part_name, normals in camera["local_scene_normals"].items():
            if normals.dtype != np.dtype(np.int8):
                issues.append(f"local_scene_normals/{part_name} expected int8, got {normals.dtype}")
                break
    return issues


def _validate_clip(clip: h5py.Group) -> list[str]:
    issues: list[str] = []
    required_clip_fields = [
        "joint_positions",
        "joint_names",
        "base_pose",
        "left_gripper_open",
        "left_gripper_pose",
        "right_gripper_open",
        "right_gripper_pose",
        "camera_head",
        "source_demo_clean_frame_indices",
        "source_traj_frame_indices",
        "source_frame_is_padding",
        "source_save_freq",
    ]
    for key in required_clip_fields:
        if key not in clip:
            issues.append(f"missing clip/{key}")
    for attr_name in CLIP_ATTRIBUTE_KEYS:
        if attr_name not in clip.attrs:
            issues.append(f"missing attr {attr_name}")
    if "camera_head" in clip and isinstance(clip["camera_head"], h5py.Group):
        issues.extend(_validate_camera_group(clip["camera_head"]))
    return issues


def _collect_valid_clips(h5_files: Iterable[Path]) -> tuple[list[list[str]], dict[str, Any]]:
    h5_files = list(h5_files)
    valid_clips: list[list[str]] = []
    invalid: list[dict[str, Any]] = []
    total_clips = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            for clip_key, clip in h5.items():
                if not isinstance(clip, h5py.Group) or ":" not in clip_key:
                    continue
                total_clips += 1
                issues = _validate_clip(clip)
                if issues:
                    invalid.append({"h5_path": str(h5_path), "clip_key": clip_key, "issues": issues})
                    continue
                valid_clips.append([str(h5_path), clip_key])
    stats: dict[str, Any] = {
        "total_files": len(h5_files),
        "total_clips": total_clips,
        "valid_clips": len(valid_clips),
        "invalid_clips": len(invalid),
        "timestamp": time.time(),
    }
    if invalid:
        stats["invalid_examples"] = invalid[:20]
    return valid_clips, stats


def _assign_episode_level_splits(
    valid_clips: list[list[str]],
    *,
    seed: int,
    test_percentage: float,
) -> tuple[list[list[str]], list[list[str]]]:
    if not (0 <= test_percentage <= 1):
        raise ValueError(f"test_percentage must be in [0, 1], got {test_percentage}")

    by_episode: dict[str, list[list[str]]] = {}
    for h5_path, clip_key in valid_clips:
        by_episode.setdefault(h5_path, []).append([h5_path, clip_key])

    episodes = sorted(by_episode.keys())
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    num_test = int(math.ceil(len(shuffled) * test_percentage)) if test_percentage > 0 else 0
    num_test = min(num_test, len(shuffled))
    test_episodes = set(shuffled[-num_test:]) if num_test > 0 else set()

    train: list[list[str]] = []
    test: list[list[str]] = []
    for episode in episodes:
        target = test if episode in test_episodes else train
        target.extend(sorted(by_episode[episode]))
    return train, test


def _clip_attrs_payload(clip: h5py.Group) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr_name in CLIP_ATTRIBUTE_KEYS:
        value = clip.attrs[attr_name]
        payload[attr_name] = value.item() if hasattr(value, "item") else value
    return payload


def _source_metadata_payload(h5: h5py.File, h5_path: str | Path, clip_key: str, input_dir: Path) -> dict[str, Any]:
    h5_path = Path(h5_path)
    return {
        "source_h5_path": str(h5_path),
        "source_h5_relpath": os.path.relpath(os.path.abspath(h5_path), os.path.abspath(input_dir)).replace("\\", "/"),
        "input_dir": str(input_dir),
        "episode_id": _robotwin_episode_id(h5_path, input_dir),
        "task": _task_from_robotwin_h5_path(h5_path, input_dir),
        "clip_key": clip_key,
        "clip_len": int(h5.attrs.get("clip_len", 0)),
        "frame_interval": int(h5.attrs.get("frame_interval", 0)),
        "stride": int(h5.attrs.get("stride", 0)),
    }


def _read_group_payload(group: h5py.Group) -> dict[str, np.ndarray]:
    return {key: dataset[:] for key, dataset in group.items()}


def _read_initial_rgb_bytes(dataset: h5py.Dataset) -> bytes:
    value = dataset[0]
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value)
    return value.tobytes()


def _write_clip_sample(
    writer: wds.ShardWriter,
    h5_path: str,
    clip_key: str,
    input_dir: Path,
    *,
    max_history_horizon: int,
) -> None:
    with h5py.File(h5_path, "r") as h5:
        clip = h5[clip_key]
        camera = clip["camera_head"]
        sample: dict[str, Any] = {"__key__": _clip_id(h5_path, clip_key, input_dir)}

        for key in ["joint_positions", "base_pose", "left_gripper_open", "left_gripper_pose", "right_gripper_open", "right_gripper_pose"]:
            sample[f"{key}.npy"] = _serialize_numpy(_dataset_value(clip[key]))
        sample["joint_names.pyd"] = _serialize_pickle(_dataset_value(clip["joint_names"]))
        sample["clip_attributes.pyd"] = _serialize_pickle(_clip_attrs_payload(clip))
        sample["robotwin_source_metadata.pyd"] = _serialize_pickle(_source_metadata_payload(h5, h5_path, clip_key, input_dir))
        for key in ["source_demo_clean_frame_indices", "source_traj_frame_indices", "source_frame_is_padding", "source_save_freq"]:
            sample[f"{key}.npy"] = _serialize_numpy(_dataset_value(clip[key]))
        for key, value in _history_robot_state_payload(
            h5_path,
            clip_key,
            max_history_horizon=max_history_horizon,
        ).items():
            sample[f"{key}.npy"] = _serialize_numpy(value)

        for key in ["local_scene_points", "local_scene_colors", "local_scene_normals", "scene_mesh_trajectories"]:
            sample[f"camera_head_{key}.pyd"] = _serialize_pickle(_read_group_payload(camera[key]))
        sample["camera_head_initial_rgb.jpg"] = _read_initial_rgb_bytes(camera["initial_rgb"])
        for key in ["initial_depth", "intrinsic", "extrinsic"]:
            sample[f"camera_head_{key}.npy"] = _serialize_numpy(_dataset_value(camera[key]))

        writer.write(sample)


def _write_split(
    split_name: str,
    clips: list[list[str]],
    *,
    output_dir: Path,
    input_dir: Path,
    maxsize: float,
    max_history_horizon: int,
) -> set[str]:
    source_paths: set[str] = set()
    if not clips:
        return source_paths

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(split_dir / f"{split_name}-rank00-%06d.tar")
    writer = wds.ShardWriter(shard_pattern, maxsize=maxsize, encoder=False)
    try:
        for h5_path, clip_key in tqdm(clips, desc=f"{split_name}"):
            _write_clip_sample(
                writer,
                h5_path,
                clip_key,
                input_dir,
                max_history_horizon=max_history_horizon,
            )
            source_paths.add(h5_path)
    finally:
        writer.close()
    return source_paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_source_paths(path: Path, source_paths: set[str]) -> None:
    if not source_paths:
        return
    with path.open("w", encoding="utf-8") as f:
        for source_path in sorted(source_paths):
            f.write(f"{source_path}\n")


def _build_manifest(
    valid_clips: list[list[str]],
    test_clips: list[list[str]],
    *,
    input_dir: Path,
    seed: int,
    test_percentage: float,
    filters: dict[str, Any],
) -> dict[str, Any]:
    include_clip_keys = sorted(_clip_id(h5_path, clip_key, input_dir) for h5_path, clip_key in valid_clips)
    test_clip_keys = sorted(_clip_id(h5_path, clip_key, input_dir) for h5_path, clip_key in test_clips)
    return {
        "schema_version": "wds_manifest.v1",
        "domain": "robotwin",
        "split_level": "episode",
        "seed": seed,
        "test_percentage": test_percentage,
        "filters": filters,
        "stats": {
            "num_selected_total": len(include_clip_keys),
            "num_selected_test": len(test_clip_keys),
            "num_selected_train": len(include_clip_keys) - len(test_clip_keys),
            "num_test_episodes": len({h5_path for h5_path, _ in test_clips}),
            "num_train_episodes": len({h5_path for h5_path, _ in valid_clips}) - len({h5_path for h5_path, _ in test_clips}),
        },
        "test_clip_keys": test_clip_keys,
        "include_clip_keys": include_clip_keys,
    }


def convert_robotwin_h5_to_wds(
    *,
    input_dir: Path,
    output_dir: Path,
    tasks: list[str],
    max_episodes_per_task: int,
    test_percentage: float,
    seed: int,
    maxsize: float,
    max_history_horizon: int,
) -> dict[str, Any]:
    h5_files = _iter_robotwin_h5_files(
        input_dir,
        tasks=tasks,
        max_episodes_per_task=max_episodes_per_task,
    )
    if not h5_files:
        raise FileNotFoundError(f"No RobotWin episode H5 files selected under {input_dir}")

    valid_clips, integrity_stats = _collect_valid_clips(h5_files)
    if not valid_clips:
        raise AssertionError("No valid clips found in selected RobotWin H5 files.")

    train_clips, test_clips = _assign_episode_level_splits(
        valid_clips,
        seed=seed,
        test_percentage=test_percentage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filters = {
        "tasks": tasks,
        "max_episodes_per_task": max_episodes_per_task,
        "max_history_horizon": max_history_horizon,
    }
    integrity_payload = {
        "valid_clips": valid_clips,
        "stats": integrity_stats,
        "config": {
            "domain": "robotwin",
            "data_keys": COMPACT_DATA_KEYS,
            "split_level": "episode",
            "max_history_horizon": max_history_horizon,
        },
    }
    manifest = _build_manifest(
        valid_clips,
        test_clips,
        input_dir=input_dir,
        seed=seed,
        test_percentage=test_percentage,
        filters=filters,
    )
    metadata = {
        "train": {
            "processed_count": len(train_clips),
            "global_selected_count": len(train_clips),
            "worker_assigned_count": len(train_clips),
        },
        "test": {
            "processed_count": len(test_clips),
            "global_selected_count": len(test_clips),
            "worker_assigned_count": len(test_clips),
        },
        "config": {
            "seed": seed,
            "test_percentage": test_percentage,
            "split_level": "episode",
            "tasks": tasks,
            "max_episodes_per_task": max_episodes_per_task,
            "max_history_horizon": max_history_horizon,
        },
    }

    _write_json(output_dir / "integrity_check.json", integrity_payload)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metadata_rank0.json", metadata)

    train_sources = _write_split(
        "train",
        train_clips,
        output_dir=output_dir,
        input_dir=input_dir,
        maxsize=maxsize,
        max_history_horizon=max_history_horizon,
    )
    test_sources = _write_split(
        "test",
        test_clips,
        output_dir=output_dir,
        input_dir=input_dir,
        maxsize=maxsize,
        max_history_horizon=max_history_horizon,
    )
    _write_source_paths(output_dir / "train_source_paths_rank0.txt", train_sources)
    _write_source_paths(output_dir / "test_source_paths_rank0.txt", test_sources)

    return {
        "selected_files": len(h5_files),
        "valid_clips": len(valid_clips),
        "train_clips": len(train_clips),
        "test_clips": len(test_clips),
        "train_episodes": len({h5_path for h5_path, _ in train_clips}),
        "test_episodes": len({h5_path for h5_path, _ in test_clips}),
        "max_history_horizon": max_history_horizon,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RoboTwin compact PointWorld H5 episodes to WDS shards.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task names; empty/all means all tasks.")
    parser.add_argument("--max_episodes_per_task", type=int, default=-1)
    parser.add_argument("--test_percentage", type=float, default=DEFAULT_TEST_PERCENTAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxsize", type=float, default=1e9)
    parser.add_argument(
        "--max_history_horizon",
        type=int,
        default=DEFAULT_MAX_HISTORY_HORIZON,
        help="Maximum RobotWin history horizon to cache in each WDS sample; use <=0 to disable cached history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_robotwin_h5_to_wds(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        tasks=_parse_csv_arg(args.tasks),
        max_episodes_per_task=int(args.max_episodes_per_task),
        test_percentage=float(args.test_percentage),
        seed=int(args.seed),
        maxsize=float(args.maxsize),
        max_history_horizon=int(args.max_history_horizon),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
