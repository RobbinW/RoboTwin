import h5py
import numpy as np

from robotwin_pointworld.behavior_online import decode_behavior_online_camera_group, pointworld_pose_to_matrix


def test_pointworld_pose_to_matrix_uses_xyzw_quaternion_layout():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    matrix = pointworld_pose_to_matrix(pose)

    np.testing.assert_allclose(matrix, np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32))


def test_decode_behavior_online_camera_group_reconstructs_flows_from_local_points(tmp_path):
    h5_path = tmp_path / "behavior_online.h5"
    with h5py.File(h5_path, "w") as f:
        camera = f.create_group("camera_head")
        local_points = camera.create_group("local_scene_points")
        local_points.create_dataset("moving", data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        local_points.create_dataset("static_actor_5", data=np.array([[3.0, 0.0, 0.0]], dtype=np.float32))

        local_colors = camera.create_group("local_scene_colors")
        local_colors.create_dataset("moving", data=np.array([[255, 0, 0], [128, 0, 0]], dtype=np.uint8))
        local_colors.create_dataset("static_actor_5", data=np.array([[0, 255, 0]], dtype=np.uint8))

        local_normals = camera.create_group("local_scene_normals")
        local_normals.create_dataset("moving", data=np.zeros((2, 3), dtype=np.int8))
        local_normals.create_dataset("static_actor_5", data=np.zeros((1, 3), dtype=np.int8))

        trajectories = camera.create_group("scene_mesh_trajectories")
        moving_traj = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        static_traj = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        trajectories.create_dataset("moving", data=moving_traj)
        trajectories.create_dataset("static_actor_5", data=static_traj)

    with h5py.File(h5_path, "r") as f:
        decoded = decode_behavior_online_camera_group(f["camera_head"])

    assert decoded.scene_flows.shape == (3, 3, 3)
    np.testing.assert_allclose(decoded.scene_flows[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(decoded.scene_flows[2], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_array_equal(decoded.scene_colors[0], [[255, 0, 0], [128, 0, 0], [0, 255, 0]])
    assert decoded.scene_visibility.shape == (3, 3)
    assert decoded.scene_visibility.all()


def test_decode_behavior_online_camera_group_uses_part_metadata_for_order_and_robot_flags(tmp_path):
    h5_path = tmp_path / "behavior_online_part_metadata.h5"
    identity_traj = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    with h5py.File(h5_path, "w") as f:
        camera = f.create_group("camera_head")
        local_points = camera.create_group("local_scene_points")
        local_points.create_dataset("arm_link", data=np.ones((2, 3), dtype=np.float32))
        local_points.create_dataset("object", data=np.zeros((1, 3), dtype=np.float32))
        trajectories = camera.create_group("scene_mesh_trajectories")
        trajectories.create_dataset("object", data=identity_traj)
        trajectories.create_dataset("arm_link", data=identity_traj)
        camera.create_dataset("scene_part_names", data=np.asarray([b"object", b"arm_link"]))
        camera.create_dataset("scene_part_is_robot", data=np.asarray([False, True], dtype=bool))
        camera.create_dataset("scene_part_category", data=np.asarray([b"task_object", b"robot"]))
        camera.create_dataset("scene_part_actor_id", data=np.asarray([5, 99], dtype=np.int32))
        camera.create_dataset("scene_part_point_count", data=np.asarray([1, 2], dtype=np.int32))

    with h5py.File(h5_path, "r") as f:
        decoded = decode_behavior_online_camera_group(f["camera_head"])

    np.testing.assert_array_equal(decoded.scene_robot_mask, [False, True, True])
    assert decoded.part_names == ["object", "arm_link"]
