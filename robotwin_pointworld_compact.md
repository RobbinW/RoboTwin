# RoboTwin PointWorld Compact Flow

This path writes PointWorld/BEHAVIOR-style compact GT flow clips directly during RoboTwin replay.

## Collect Compact H5

```bash
cd /data/dex/RoboTwin
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
CUDA_VISIBLE_DEVICES=0 /data/dex/conda-envs/RoboTwin/bin/python \
script/collect_data.py adjust_bottle pointworld_behavior_compact_head --episode_num 1
```

The config uses `save_freq: 15`, `pointworld_clip_len: 15`, `pointworld_stride: 15`, head camera only, and writes:

```text
data/<task>/pointworld_behavior_compact_head/data/episode<N>.hdf5
```

Each clip stores `camera_head/local_scene_points`, `local_scene_colors`, `local_scene_normals`, `scene_mesh_trajectories`, `scene_robot_mask`, camera initial RGB/depth/intrinsic/extrinsic, joint state, base pose, and left/right gripper pose/open.

## Export WDS

```bash
cd /data/dex/RoboTwin
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
/data/dex/conda-envs/RoboTwin/bin/python script/export_robotwin_pointworld_wds.py \
  --input_dir data/adjust_bottle/pointworld_behavior_compact_head/data \
  --output_dir data/adjust_bottle/pointworld_behavior_compact_head/wds \
  --pointworld_data_repo /data/dex/PointWorld-data \
  --pointworld_python /root/miniconda3/bin/python
```

The wrapper runs PointWorld-data integrity check, manifest generation, then `convert_wds.py --domain robotwin`.

## Visualize

```bash
cd /data/dex/RoboTwin
/data/dex/conda-envs/RoboTwin/bin/python script/serve_robotwin_pointworld_flow_viser.py \
  --h5 data/adjust_bottle/pointworld_behavior_compact_head/data/episode0.hdf5 \
  --port 8099
```

The viewer decodes dense `scene_flows` online from `local_scene_points` and `scene_mesh_trajectories`; the dense flow is not stored in the compact H5.
