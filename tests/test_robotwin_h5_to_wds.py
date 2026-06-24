import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np

from script.robotwin_h5_to_wds import (
    CLIP_ATTRIBUTE_KEYS,
    DEFAULT_TEST_PERCENTAGE,
    _assign_episode_level_splits,
    _write_clip_sample,
)


class RobotWinH5ToWdsTest(unittest.TestCase):
    def test_default_test_percentage_is_ten_percent(self):
        self.assertEqual(DEFAULT_TEST_PERCENTAGE, 0.1)

    def test_episode_level_split_keeps_clips_from_same_episode_together(self):
        valid_clips = [
            ["/data/root/task_a/config/data/episode0.hdf5", "episode0:clip000000"],
            ["/data/root/task_a/config/data/episode0.hdf5", "episode0:clip000001"],
            ["/data/root/task_a/config/data/episode1.hdf5", "episode1:clip000000"],
            ["/data/root/task_b/config/data/episode0.hdf5", "episode0:clip000000"],
            ["/data/root/task_b/config/data/episode1.hdf5", "episode1:clip000000"],
            ["/data/root/task_b/config/data/episode1.hdf5", "episode1:clip000001"],
            ["/data/root/task_b/config/data/episode2.hdf5", "episode2:clip000000"],
            ["/data/root/task_b/config/data/episode3.hdf5", "episode3:clip000000"],
            ["/data/root/task_b/config/data/episode4.hdf5", "episode4:clip000000"],
            ["/data/root/task_b/config/data/episode5.hdf5", "episode5:clip000000"],
        ]

        train, test = _assign_episode_level_splits(valid_clips, seed=7, test_percentage=0.2)

        train_episodes = {path for path, _clip_key in train}
        test_episodes = {path for path, _clip_key in test}
        self.assertTrue(train_episodes.isdisjoint(test_episodes))
        self.assertEqual(len(test_episodes), 2)
        self.assertEqual(len(train) + len(test), len(valid_clips))

    def test_write_clip_sample_includes_dynamic_history_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            h5_path = input_dir / "adjust_bottle" / "cfg" / "data" / "episode0.hdf5"
            h5_path.parent.mkdir(parents=True)
            with h5py.File(h5_path, "w") as h5:
                h5.attrs["clip_len"] = 16
                h5.attrs["stride"] = 10
                h5.attrs["frame_interval"] = 2
                clip = h5.create_group("episode0:clip000000")
                for attr_name in CLIP_ATTRIBUTE_KEYS:
                    clip.attrs[attr_name] = "episode0:clip000000" if attr_name == "clip_key" else 0
                clip.create_dataset("joint_positions", data=np.zeros((16, 38), dtype=np.float32))
                clip.create_dataset("base_pose", data=np.zeros((16, 7), dtype=np.float32))
                clip.create_dataset("left_gripper_open", data=np.zeros((16, 1), dtype=np.float32))
                clip.create_dataset("right_gripper_open", data=np.zeros((16, 1), dtype=np.float32))
                clip.create_dataset("left_gripper_pose", data=np.zeros((16, 7), dtype=np.float32))
                clip.create_dataset("right_gripper_pose", data=np.zeros((16, 7), dtype=np.float32))
                clip.create_dataset("joint_names", data=np.array([b"j0"]))
                clip.create_dataset("source_demo_clean_frame_indices", data=np.arange(0, 32, 2, dtype=np.int64))
                clip.create_dataset("source_traj_frame_indices", data=np.arange(16, dtype=np.int64))
                clip.create_dataset("source_frame_is_padding", data=np.zeros((16,), dtype=bool))
                clip.create_dataset("source_save_freq", data=np.full((16,), 15, dtype=np.int64))
                camera = clip.create_group("camera_head")
                for name in ["local_scene_points", "local_scene_colors", "local_scene_normals", "scene_mesh_trajectories"]:
                    group = camera.create_group(name)
                    shape = (1, 3) if name != "scene_mesh_trajectories" else (16, 7)
                    dtype = np.int8 if name == "local_scene_normals" else np.float32
                    group.create_dataset("object", data=np.zeros(shape, dtype=dtype))
                camera.create_dataset("initial_rgb", data=np.array([b"\xff\xd8\xff\xd9"], dtype=h5py.string_dtype("ascii")))
                camera.create_dataset("initial_depth", data=np.zeros((180, 320), dtype=np.uint16))
                camera.create_dataset("intrinsic", data=np.eye(3, dtype=np.float32))
                camera.create_dataset("extrinsic", data=np.eye(4, dtype=np.float32))

            class FakeWriter:
                def __init__(self):
                    self.sample = None

                def write(self, sample):
                    self.sample = sample

            writer = FakeWriter()
            _write_clip_sample(writer, str(h5_path), "episode0:clip000000", input_dir)

        self.assertIn("source_demo_clean_frame_indices.npy", writer.sample)
        self.assertIn("source_traj_frame_indices.npy", writer.sample)
        self.assertIn("source_frame_is_padding.npy", writer.sample)
        self.assertIn("source_save_freq.npy", writer.sample)
        self.assertIn("robotwin_source_metadata.pyd", writer.sample)


if __name__ == "__main__":
    unittest.main()
