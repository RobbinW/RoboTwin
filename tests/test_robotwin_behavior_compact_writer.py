import h5py
import numpy as np

from robotwin_pointworld.behavior_online import decode_behavior_online_camera_group
from robotwin_pointworld.online_writer import BehaviorCompactEpisodeWriter


def _frame(t: int) -> dict:
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[0, 1] = [0, 255, 0]
    rgb[0, 2] = [0, 0, 255]

    position = np.zeros((1, 3, 4), dtype=np.float32)
    position[..., 3] = 0.0
    position[0, 0] = [0.0, 0.0, -1.0, 0.0]
    position[0, 1] = [1.0, 0.0, -1.0, 0.0]
    position[0, 2] = [2.0, 0.0, -1.0, 0.0]

    raw_actor = np.array([[5, 6, 99]], dtype=np.int32)

    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(t)

    robot_pose = np.eye(4, dtype=np.float32)
    robot_pose[1, 3] = float(t) * 0.25

    return {
        "observation": {
            "head_camera": {
                "rgb": rgb,
                "depth": np.ones((1, 3), dtype=np.float32),
                "position": position,
                "raw_actor_segmentation": raw_actor,
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "extrinsic_cv": np.eye(4, dtype=np.float32)[:3],
                "cam2world_gl": np.eye(4, dtype=np.float32),
            }
        },
        "flow_parts": {
            "actor_ids": np.array([5], dtype=np.int32),
            "part_names": np.array([b"bottle"]),
            "object_names": np.array([b"bottle"]),
            "pose_world": pose[None],
        },
        "robot_state": {
            "joint_positions": np.zeros((1,), dtype=np.float32),
            "joint_names": np.array([b"j0"]),
            "base_pose_world": np.eye(4, dtype=np.float32),
            "left_gripper_open": np.array(0.0, dtype=np.float32),
            "right_gripper_open": np.array(0.0, dtype=np.float32),
            "robot_actor_ids": np.array([99], dtype=np.int32),
            "robot_part_names": np.array([b"right_link"]),
            "robot_pose_world": robot_pose[None],
        },
        "endpose": {
            "left_endpose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_endpose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
        "frame_metadata": {
            "demo_clean_frame_index": t,
            "traj_frame_index": t * 15,
            "save_freq": 15,
        },
    }


def test_behavior_compact_episode_writer_streams_frames_to_compact_h5(tmp_path):
    out_path = tmp_path / "episode0.hdf5"
    writer = BehaviorCompactEpisodeWriter(
        output_h5_path=out_path,
        clip_len=15,
        stride=15,
        camera_name="head_camera",
        min_object_motion=0.01,
    )

    for t in range(15):
        writer.append(_frame(t))
    writer.close()

    with h5py.File(out_path, "r") as f:
        assert f.attrs["episode_complete"]
        assert list(f.keys()) == ["episode0:clip000000"]
        cam = f["episode0:clip000000"]["camera_head"]
        assert "scene_flows" not in cam
        assert cam["local_scene_points"]["bottle"].dtype == np.float16
        decoded = decode_behavior_online_camera_group(cam)
        assert decoded.scene_flows.shape == (15, 3, 3)
        np.testing.assert_array_equal(decoded.scene_robot_mask, [False, True, False])
        np.testing.assert_allclose(decoded.scene_flows[-1, 0], [14.0, 0.0, -1.0])
        np.testing.assert_allclose(decoded.scene_flows[-1, 1], [2.0, 3.5, -1.0])
        np.testing.assert_allclose(decoded.scene_flows[-1, 2], [1.0, 0.0, -1.0])


def test_behavior_compact_episode_writer_uses_frame_interval_and_source_metadata(tmp_path):
    out_path = tmp_path / "episode0.hdf5"
    writer = BehaviorCompactEpisodeWriter(
        output_h5_path=out_path,
        clip_len=4,
        stride=2,
        frame_interval=2,
        camera_name="head_camera",
        min_object_motion=0.0,
    )

    for t in range(9):
        writer.append(_frame(t))
    writer.close()

    with h5py.File(out_path, "r") as f:
        assert list(f.keys()) == ["episode0:clip000000", "episode0:clip000001"]

        clip0 = f["episode0:clip000000"]
        np.testing.assert_array_equal(clip0["source_demo_clean_frame_indices"][:], [0, 2, 4, 6])
        np.testing.assert_array_equal(clip0["source_traj_frame_indices"][:], [0, 30, 60, 90])
        assert clip0.attrs["source_demo_clean_start_frame"] == 0
        assert clip0.attrs["source_demo_clean_end_frame"] == 6
        assert clip0.attrs["source_traj_start_frame"] == 0
        assert clip0.attrs["source_traj_end_frame"] == 90
        assert clip0.attrs["source_frame_interval"] == 2
        assert clip0.attrs["source_clip_stride"] == 2

        clip1 = f["episode0:clip000001"]
        np.testing.assert_array_equal(clip1["source_demo_clean_frame_indices"][:], [2, 4, 6, 8])
        np.testing.assert_array_equal(clip1["source_traj_frame_indices"][:], [30, 60, 90, 120])
        decoded = decode_behavior_online_camera_group(clip1["camera_head"])
        assert decoded.scene_flows.shape == (4, 3, 3)
