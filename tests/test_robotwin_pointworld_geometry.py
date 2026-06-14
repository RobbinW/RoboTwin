import numpy as np

from robotwin_pointworld.geometry import (
    local_points_to_scene_flows,
    matrix_to_pointworld_pose,
    sapien_pose_to_matrix,
    select_frame0_actor_observations,
    select_frame0_actor_points,
)


def test_sapien_pose_to_pointworld_pose_converts_wxyz_to_xyzw():
    angle = np.pi / 2.0
    sapien_pose = np.array(
        [1.0, 2.0, 3.0, np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float32,
    )

    mat = sapien_pose_to_matrix(sapien_pose)
    pointworld_pose = matrix_to_pointworld_pose(mat)

    np.testing.assert_allclose(pointworld_pose[:3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        np.abs(pointworld_pose[3:]),
        [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)],
        atol=1e-6,
    )
    rotated = mat @ np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(rotated[:3], [1.0, 3.0, 3.0], atol=1e-6)


def test_local_points_to_scene_flows_preserves_point_identity():
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    traj = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    traj[1, :3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    traj[2] = sapien_pose_to_matrix(
        np.array(
            [0.0, 0.0, 0.0, np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)],
            dtype=np.float32,
        )
    )

    flows = local_points_to_scene_flows(local_points, traj)

    assert flows.shape == (3, 2, 3)
    np.testing.assert_allclose(flows[0], local_points, atol=1e-6)
    np.testing.assert_allclose(flows[1], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(flows[2], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-6)


def test_select_frame0_actor_points_uses_raw_actor_ids_and_camera_pose():
    position = np.array(
        [
            [[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]],
            [[0.0, 1.0, -1.0, 1.0], [2.0, 0.0, -1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    raw_actor = np.array([[7, 9], [7, 7]], dtype=np.int32)
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, 3] = np.array([10.0, 0.0, 0.0], dtype=np.float32)

    points, colors, actor_ids = select_frame0_actor_points(
        position=position,
        raw_actor_segmentation=raw_actor,
        rgb=rgb,
        cam2world_gl=cam2world,
        keep_actor_ids={7},
    )

    assert actor_ids.tolist() == [7, 7]
    np.testing.assert_allclose(points, [[10.0, 0.0, -1.0], [12.0, 0.0, -1.0]], atol=1e-6)
    np.testing.assert_array_equal(colors, [[10, 20, 30], [100, 110, 120]])


def test_select_frame0_actor_observations_transforms_camera_normals_to_world():
    position = np.array(
        [
            [[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    raw_actor = np.array([[7, 9]], dtype=np.int32)
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    normal = np.array([[[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    obs = select_frame0_actor_observations(
        position=position,
        raw_actor_segmentation=raw_actor,
        rgb=rgb,
        cam2world_gl=cam2world,
        normal=normal,
        keep_actor_ids={7},
    )

    assert obs.actor_ids.tolist() == [7]
    np.testing.assert_allclose(obs.points_world, [[-1.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_array_equal(obs.colors, [[10, 20, 30]])
    np.testing.assert_allclose(obs.normals_world, [[1.0, 0.0, 0.0]], atol=1e-6)
