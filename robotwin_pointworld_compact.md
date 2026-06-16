# RoboTwin PointWorld Compact Flow

This pipeline writes PointWorld/BEHAVIOR-style compact GT flow clips directly while replaying RoboTwin demonstrations.

## Replay And Planning Logic

The current `pointworld_behavior_compact_head` config is a replay config:

```yaml
use_seed: true
replay_source_config: demo_clean
pointworld_behavior_online: true
```

For a task such as `adjust_bottle`, the command below reads existing demo data from:

```text
data/adjust_bottle/demo_clean/seed.txt
data/adjust_bottle/demo_clean/_traj_data/episode<N>.pkl
```

Each `_traj_data/episode<N>.pkl` stores the planned dense robot action paths:

```text
left_joint_path   # list of planned left-arm dense joint paths
right_joint_path  # list of planned right-arm dense joint paths
```

During compact collection, RoboTwin does not plan again. It calls `setup_demo()` with the saved seed, loads the saved joint paths, replays them in simulation, and renders the extra information needed for GT flow: RGB, depth, Position buffer, raw actor segmentation, normals, object poses, robot poses, qpos, and gripper state.

Those raw render buffers are consumed online by `BehaviorCompactEpisodeWriter`. The compact H5 intentionally stores the smaller PointWorld/BEHAVIOR representation instead of dumping every per-frame raw segmentation or Position image.

Important distinction:

- With the current compact config, missing `demo_clean/_traj_data/episode<N>.pkl` or `demo_clean/seed.txt` is an error. It does not automatically fall back to planning.
- RoboTwin's original `collect_data.py` can plan first when `use_seed: false`. In that mode it first runs task planning, writes a new `seed.txt` and `_traj_data`, then enters the normal data-collection stage.
- Even with `use_seed: false`, the current implementation still performs two phases inside one command: planning phase -> replay collection phase. It does not write compact H5 directly during the planning rollout, because failed plans should not become training samples and the collection stage expects saved dense paths.
- To generate compact flow from scratch, either first collect `demo_clean`, then run this compact replay config, or make a separate compact config with `use_seed: false` and no `replay_source_config: demo_clean`. The latter removes the dependency on an existing `demo_clean`, but it still plans first and then replays the newly saved `_traj_data`.

## Collect Compact H5

```bash
cd /data/dex/RoboTwin
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
CUDA_VISIBLE_DEVICES=0 /data/dex/conda-envs/RoboTwin/bin/python \
script/collect_data.py adjust_bottle pointworld_behavior_compact_head --episode_num 1
```

The config uses head camera only and builds clips from the saved observation stream:

- `save_freq: 15`: RoboTwin first saves replay observations every 15 low-level control steps.
- `pointworld_clip_len: 16`: each compact clip contains 16 training frames.
- `pointworld_frame_interval: 2`: clip frames are sampled as `start, start+2, ..., start+30` from saved observations.
- `pointworld_stride: 10`: consecutive clip starts are 10 saved-observation frames apart, so clips overlap.
- Final residual windows are kept: if the episode ends before a full window is available,
  the writer pads by repeating the last saved observation. Padded frames are marked by
  `source_frame_is_padding`.
- `pointworld_min_object_motion: 0.0`: no motion-based clip filtering is applied.

It writes:

```text
data/<task>/pointworld_behavior_compact_head/data/episode<N>.hdf5
```

## HDF5 Layout

Notation:

```text
C  = number of clips in this episode H5
T  = 16 clip frames
H  = 180
W  = 320
N  = number of frame-0 visible valid points in one clip
D  = robot joint dimension, usually 38 for aloha-agilex
K  = number of visible parts in one clip
Ni = number of points for part i
```

Root attributes:

```text
domain          str   "robotwin"
format          str   "behavior_compact"
camera_name     str   "head_camera"
clip_len        int   16
stride          int   10, in saved-observation frame units
frame_interval  int   2, in saved-observation frame units
num_clips       int   C
episode_complete bool true after writer close
```

Each clip is stored as a root group:

```text
episode<N>:clip000000
episode<N>:clip000001
...
```

Clip attributes:

```text
clip_key                         str
num_frames                       int, T
num_scene_points                 int, N
clip_complete                    bool
has_transition                   bool
any_object_moving                bool
robot_nonbase_moving             bool
gripper_moving                   bool
has_gripper_state_change         bool
max_object_pos_movement          float32, meters
max_object_rot_movement          float32, currently 0
max_joint_movement               float32
max_gripper_pos_movement         float32, currently 0
max_gripper_rot_movement         float32, currently 0
left/right_min_distance_*        float32, placeholder -1
has_*_collision                  bool, placeholder false
source_demo_clean_start_frame    int
source_demo_clean_end_frame      int
source_traj_start_frame          int
source_traj_end_frame            int
source_frame_interval            int, 2
source_clip_stride               int, 10
source_is_padded_clip            bool
source_num_padding_frames        int, number of sampled clip frames that were padded
```

Clip datasets:

```text
source_demo_clean_frame_indices  (T,) int64
  Indices into data/<task>/demo_clean/data/episode<N>.hdf5 after save_freq sampling.
  Example: [0, 2, 4, ..., 30].

source_traj_frame_indices        (T,) int64
  Replay low-level control-step indices at which those observations were saved.
  This is the unified control-step timeline after expanding arm and gripper actions.

source_save_freq                 (T,) int64
  The save_freq value for each source frame, usually 15.

source_frame_is_padding          (T,) bool
  True for clip frames created by repeating the final saved observation. Example for
  the final padded clip: source_demo_clean_frame_indices =
  [120, 122, ..., 142, 143, 143, 143, 143] and source_frame_is_padding =
  [false, false, ..., false, true, true, true, true].

joint_positions                  (T, D) float32
  Robot qpos / joint state per clip frame.

joint_names                      (D,) bytes/string
  Names matching joint_positions columns.

base_pose                        (T, 7) float32
  Robot base pose in PointWorld pose layout: x, y, z, qx, qy, qz, qw.

left_gripper_pose                (T, 7) float32
right_gripper_pose               (T, 7) float32
  Left/right end-effector pose in PointWorld pose layout.

left_gripper_open                (T,) float32
right_gripper_open               (T,) float32
  Normalized gripper open values.

object_names                     (K,) bytes/string
  Names of visible tracked, robot, and static parts included in this clip.
```

Camera group:

```text
camera_head/intrinsic            (3, 3) float32
  Camera intrinsic matrix for the saved initial RGB/depth.

camera_head/extrinsic            (4, 4) float32
  Camera extrinsic for the initial frame.

camera_head/extrinsic_trajectory (T, 4, 4) float32
  Camera extrinsic for all clip frames.

camera_head/initial_rgb          (1,) bytes
  JPEG-encoded RGB image for the clip's first frame.

camera_head/initial_depth        (H, W) uint16
  First-frame depth in millimeters.

camera_head/scene_part_names        (P,) bytes/string
  Ordered part names. This order is the source of truth for decoding local scene
  groups and trajectories.

camera_head/scene_part_is_robot     (P,) bool
  True if the corresponding part is a robot link. This is the source of truth
  for optionally including or excluding robot scene points during decoding.

camera_head/scene_part_category     (P,) bytes/string
  Part category: task_object, robot, or static.

camera_head/scene_part_actor_id     (P,) int32
  SAPIEN actor id for each part.

camera_head/scene_part_point_count  (P,) int32
  Number of frame-0 visible points saved for each part.
```

Compact scene groups:

```text
camera_head/local_scene_points/<part_name>        (Ni, 3) float16
  Frame-0 visible points for this part, transformed into that part's local frame.
  For static_actor_* parts, this is already in the fixed scene frame.

camera_head/local_scene_colors/<part_name>        (Ni, 3) uint8
  RGB color from frame 0, one color per point.

camera_head/local_scene_normals/<part_name>       (Ni, 3) int8
  Quantized local normals. Decode by dividing by 127. These come from SAPIEN normal
  buffer when available, otherwise from Position-buffer finite differences.

camera_head/scene_mesh_trajectories/<part_name>   (T, 7) float32
  Per-frame part pose in PointWorld pose layout. Applying this trajectory to
  local_scene_points reconstructs dense scene_flows.
```

Dense fields are not stored in the compact H5:

```text
scene_flows       (T, N, 3) float32
scene_colors      (T, N, 3) uint8
scene_normals     (T, N, 3) float32
scene_visibility  (T, N) bool
scene_robot_mask  (N,) bool
```

They are decoded online by applying each part's `scene_mesh_trajectories` to its fixed `local_scene_points`. `scene_robot_mask` is derived from `scene_part_is_robot` and `scene_part_point_count`; it is not saved as a redundant HDF5 dataset. This is the same storage idea as BEHAVIOR/PointWorld: save local points plus pose trajectories, not the large dense flow tensor.

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
/data/dex/conda-envs/pointworld-env/bin/python script/serve_robotwin_pointworld_flow_viser.py \
  --h5 data/adjust_bottle/pointworld_behavior_compact_head/data/episode0.hdf5 \
  --port 8099
```

/data/dex/conda-envs/pointworld-env/bin/python script/serve_robotwin_pointworld_flow_viser.py \
  --h5 /data/dex/RoboTwin/data/episode1.hdf5 \
  --port 8099

/data/dex/conda-envs/pointworld-env/bin/python script/serve_robotwin_pointworld_flow_viser.py \
  --h5 /data/dex/RoboTwin/data/stack_bowls_two/pointworld_behavior_compact_head/data/episode0.hdf5 \
  --port 8099



The viewer decodes dense `scene_flows` online from `local_scene_points` and `scene_mesh_trajectories`; the dense flow is not stored in the compact H5.
