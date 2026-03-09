# aruco_sam_ailab

A production-ready **Vision-IMU SLAM** (Simultaneous Localization and Mapping) package for ROS 2 Humble. It fuses ArUco marker detection from an RGB-D camera, factor graph optimization (GTSAM ISAM2), IMU preintegration, and an EKF smoother to provide robust global pose estimation and map building for mobile robots.

Designed for the **Hunter 2 UGV** platform but compatible with any ROS 2 robot equipped with an RGB-D camera and IMU.

---

## Demo Videos

### Simulation — Full Pipeline (Mapping + Localization + Planning)

https://github.com/user-attachments/assets/1bf4b570-bf2f-4c00-a3c6-1c8070da9a6d

End-to-end demonstration in Gazebo: the robot maps the environment using ArUco landmarks (ID 11–20), saves the landmark map, then switches to localization mode and autonomously navigates within the mapped boundary.

### Real Robot — Standalone SLAM

https://github.com/user-attachments/assets/eed05143-c812-43bb-a7ad-7fb459602618

Real-world SLAM on the Hunter 2 UGV with an Intel RealSense D455. Shows ArUco detection, factor graph optimization, and EKF-smoothed odometry output in a physical environment.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Node Details](#node-details)
- [Operating Modes](#operating-modes)
- [Usage](#usage)
- [Configuration](#configuration)
- [Topics, Services & TF](#topics-services--tf)
- [Message Definitions](#message-definitions)
- [Package Structure](#package-structure)
- [Dependencies](#dependencies)

---

## System Architecture

```
                    ┌─────────────────────┐
                    │  aruco_detector_node │
                    │  (RGB-D Camera)     │
                    └──────────┬──────────┘
                               │ /aruco_poses
                               ▼
┌──────────────────┐    ┌─────────────────────┐
│  Wheel Odometry  │───▶│   graph_optimizer   │───▶ /aruco_slam/odom
│  (/w_odom)       │    │   (GTSAM ISAM2)     │    /aruco_slam/landmarks
└──────────────────┘    └──────────┬──────────┘    /aruco_slam/path
                                   │
┌──────────────────┐               │ /optimized_keyframe_state
│ imu_preintegration│              ▼
│ (Raw IMU → Odom) │───▶┌──────────────────┐
└──────────────────┘    │   ekf_smoother   │───▶ /ekf/odom
                        │ (IMU + SLAM EKF) │    odom→base_footprint TF
                        └──────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │ ego_state_publisher│───▶ /ego_state
                        └──────────────────┘

         [Localization Mode Only]
                        ┌─────────────────────────────┐
                        │ landmark_boundary_occupancy  │
                        │ _grid_node                   │
                        │ → /map (OccupancyGrid)       │
                        └─────────────────────────────┘
```

### Data Flow Summary

1. **aruco_detector_node** detects ArUco markers (ID 11–20) from the RGB-D camera and publishes 6-DOF marker poses.
2. **graph_optimizer** constructs a factor graph with wheel odometry factors, IMU preintegration factors, and ArUco observation factors, then runs ISAM2 incremental optimization to produce globally consistent poses and landmark positions.
3. **imu_preintegration** provides high-frequency (~200 Hz) dead-reckoning odometry from raw IMU data.
4. **ekf_smoother** fuses the high-frequency IMU odometry with low-frequency (~10 Hz) SLAM corrections using an SE(3) Extended Kalman Filter, producing smooth 200 Hz odometry output.
5. **ego_state_publisher** converts EKF odometry into a compact kinematic state for the planning module.
6. **landmark_boundary_occupancy_grid_node** (localization mode only) generates a 2D occupancy grid by connecting landmark positions into a boundary polygon.

---

## Node Details

### aruco_detector_node (C++)

Real-time ArUco marker detection and pose estimation from an RGB-D camera.

**Key Features:**
- OpenCV ArUco detection (supports `DICT_4X4_50`, `DICT_5X5_*`, `DICT_6X6_*`)
- **Depth correction**: Median sampling of depth values in a 9×9 pixel region around the marker center, with a PnP-to-sensor depth ratio consensus check (0.7–1.3 tolerance). Falls back to PnP distance if depth is inconsistent.
- Marker ID whitelist filtering (ID 11–20) to prevent false positives
- Publishes detection debug images with overlaid marker axes

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `marker_size` | `0.30` | ArUco marker edge length (m) |
| `aruco_dict_type` | `DICT_4X4_50` | ArUco dictionary type |
| `use_depth_correction` | `true` | Enable depth-based Z refinement |
| `depth_max_range` | `5.0` | Reject depth readings beyond this range (m) |
| `allowed_marker_ids` | `[11..20]` | Only process these marker IDs |

---

### graph_optimizer (C++)

Factor graph SLAM backend using GTSAM ISAM2 (Incremental Smoothing and Mapping).

**Factor Graph Components:**

| Factor | Source | Noise Strategy |
|--------|--------|----------------|
| **Prior Factor** | Initial pose | 0.01 m position, 0.01 rad orientation |
| **Wheel Odometry (BetweenFactor)** | `/w_odom` | X/Y: 0.05 m (tight), Yaw: 1.0 rad (loose — heading from IMU) |
| **IMU Preintegration** | Raw IMU | Accumulates acc/gyro between keyframes |
| **ArUco Observation** | `/aruco_poses` | Dynamic: `σ = base_σ × (1 + dist/10)`, rotation always loose (0.50 rad) |

**Keyframe Selection Policy:**
- Time interval ≥ 0.5 s, **OR**
- Translation ≥ 0.15 m, **OR**
- Rotation ≥ 0.1 rad (~5.7°)

**Observation Filtering:**
- **Viewing angle filter**: Rejects markers observed nearly face-on (≤15°) to avoid ill-conditioned Jacobians
- **Range filter**: Rejects observations beyond 4.0 m
- **Dynamic noise scaling**: Uncertainty grows linearly with observation distance

**Concurrency:** Uses ROS 2 `MultiThreadedExecutor` with separate callback groups for IMU (high-frequency), ArUco (lightweight), and SLAM timer (heavy optimization ~100 ms).

---

### imu_preintegration (C++)

Dead-reckoning odometry from raw IMU data at ~200 Hz.

**Features:**
- GTSAM `PreintegratedImuMeasurements` for numerical integration of acceleration and angular velocity
- **Stationary detection**: Monitors accelerometer standard deviation in a sliding window; if σ < 0.15 m/s², resets velocity to zero to prevent drift accumulation
- **Initial bias estimation**: Collects 1 second of stationary IMU samples at startup to compute gyro/accel biases
- Publishes `/odometry/imu_incremental` (does not broadcast TF)

---

### ekf_smoother (C++)

SE(3) Extended Kalman Filter that fuses IMU odometry with SLAM corrections.

**EKF Cycle:**

| Step | Rate | Description |
|------|------|-------------|
| **Prediction** | ~200 Hz | Computes relative delta from IMU odom: `state = state ⊕ Δpose` (SE(3) composition) |
| **Update** | ~10 Hz | Innovation: `e = log(state⁻¹ · measurement)` with Kalman gain `K = P(P+R)⁻¹` |

**Safety Mechanisms:**
- **Jump detection**: Rejects single-step deltas exceeding 0.5 m or 1.0 rad (handles sim teleportation, sensor dropout)
- **Innovation gating**: After an initial convergence phase (10 updates), rejects SLAM corrections > 5.0 m or > 2.0 rad
- **Joseph-form covariance update**: `P = (I-K)P(I-K)ᵀ + KRKᵀ` for numerical stability

**Velocity Smoothing:** Low-pass filter (α = 0.1) combined with a sliding window moving average to suppress IMU vibration noise.

**Noise Parameters:**
| Parameter | Value | Role |
|-----------|-------|------|
| `ekf_process_noise_pos` | 0.15 | IMU position drift allowance (m/√s) |
| `ekf_process_noise_rot` | 0.02 | IMU heading drift allowance (rad/√s) |
| `ekf_measurement_noise_pos` | 0.05 | SLAM position correction trust (m) |
| `ekf_measurement_noise_rot` | 0.15 | SLAM rotation correction trust (rad) |

---

### ego_state_publisher (Python)

Converts EKF odometry into a compact `EgoState` message for downstream planners.

- Extracts position (x, y), yaw (from quaternion), velocity, and yaw rate
- Computes acceleration via numerical differentiation with LPF (α = 0.1) and 20-sample moving average
- Clamps acceleration to ±3.0 m/s² to suppress numerical noise

---

### landmark_boundary_occupancy_grid_node (Python)

Generates a 2D occupancy grid by connecting ArUco landmarks into a boundary polygon (localization mode only).

- Loads landmark positions from `landmarks_map.json`
- Connects landmarks in `boundary_id_order: [11,12,13,14,15,16,17,18,20,19]` using Bresenham line drawing
- Grid resolution: 0.1 m/cell
- Publishes both `/global_map` (full boundary) and `/local_map` (10×10 m robot-centric window)

---

## Operating Modes

| Mode | Description |
|------|-------------|
| **mapping** | Explores the environment, registers ArUco markers as landmarks, builds and saves the landmark map |
| **localization** | Loads a previously saved map, fixes landmark positions, and performs pure pose estimation + occupancy grid publishing |

---

## Usage

### Build

```bash
cd /ros_ws
colcon build --packages-select aruco_sam_ailab --symlink-install
source install/setup.bash
```

### Run — Simulation (Gazebo)

```bash
# Standalone
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=true

# Full bringup (Gazebo + SLAM + Planning + RViz)
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=true run_mode:=mapping
```

`use_sim_time:=true` automatically loads `config/slam_params_sim.yaml`.

### Run — Real Robot

```bash
# Standalone
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false

# Full bringup (Hardware drivers + SLAM + Planning + RViz)
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=false run_mode:=mapping
```

`use_sim_time:=false` automatically loads `config/slam_params_real.yaml`.

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim_time` | `false` | `true`: load sim config, `false`: load real config |
| `run_mode` | `mapping` | `"mapping"` or `"localization"` |
| `map_path` | (from config) | Path to landmark map JSON file for localization |
| `marker_size` | `0.30` | ArUco marker edge length (m) |
| `enable_topic_debug_log` | `false` | Enable ArUco topic reception debug logging |

### Save Map (Mapping Mode)

After driving through the environment in mapping mode:

```bash
# Full command
ros2 service call /save_landmarks std_srvs/srv/Trigger "{}"

# Docker alias
savem
```

Map is saved to the path specified by `landmarks_save_path` (default: `/ros_ws/src/aruco_sam_ailab/map/landmarks_map.json`).

### Switch to Localization Mode

```bash
# Using saved map
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=false run_mode:=localization

# Custom map file
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false \
  run_mode:=localization map_path:=/path/to/my_map.json
```

### Debug

```bash
ros2 topic echo /aruco_slam/odom      # Optimized global pose (map frame)
ros2 topic echo /ekf/odom             # EKF-smoothed odometry (~200 Hz)
ros2 topic echo /aruco_poses          # Raw ArUco detections
ros2 topic echo /ego_state            # Planner input state
ros2 param get /imu_preintegration imu_topic
ros2 param get /graph_optimizer imu_frame
```

---

## Configuration

### Config File Selection

The launch file automatically selects the config based on `use_sim_time`:

| File | Condition | Target |
|------|-----------|--------|
| `config/slam_params_sim.yaml` | `use_sim_time:=true` | Gazebo simulation |
| `config/slam_params_real.yaml` | `use_sim_time:=false` | Real hardware (RealSense D455) |

### Environment-Specific Differences

| Parameter | Simulation | Real Hardware |
|-----------|-----------|---------------|
| `imu_topic` | `/camera/imu` | `/camera/camera/imu` |
| `imu_frame` | `base_footprint` | `camera_imu_optical_frame` |
| `ext_trans_base_imu` | `[0, 0, 0]` (identity) | `[0.395, 0.017, 0.405]` |
| `ext_rot_base_imu` | identity matrix | `[0,0,1, -1,0,0, 0,-1,0]` |
| `imu_acc_noise` | `0.005` | `0.15` |
| `imu_gyr_noise` | `0.0005` | `0.03` |
| `aruco_min_viewing_angle` | `0.0` | `0.17 rad (10°)` |

### Common Parameters (Shared)

```yaml
# Mode
run_mode: "mapping"              # "mapping" | "localization"
map_path: "/ros_ws/src/aruco_sam_ailab/map/landmarks_map.json"

# ArUco Observation Noise
aruco_trans_noise: 0.08          # Position correction trust
aruco_rot_noise: 0.50            # Rotation kept loose (heading from IMU)

# Wheel Odometry Noise
wheel_odom_noise_x: 0.05         # Tight (wheel odom position trusted)
wheel_odom_noise_yaw: 1.0        # Loose (heading from IMU)

# Keyframe Policy
keyframe_time_interval: 0.5      # 0.5 s minimum interval
keyframe_distance_threshold: 0.15  # 15 cm translation
keyframe_angle_threshold: 0.1    # ~5.7° rotation

# EKF Smoother
ekf_process_noise_pos: 0.15      # IMU position drift allowance
ekf_process_noise_rot: 0.02      # IMU heading trust
ekf_measurement_noise_pos: 0.05  # SLAM position correction (fast)
ekf_measurement_noise_rot: 0.15  # Heading correction (smooth)
```

### Landmark Map Format (`landmarks_map.json`)

```json
{
  "frame_id": "map",
  "timestamp": "...",
  "landmarks": [
    {
      "id": 11,
      "position": {"x": 0.0, "y": 3.0, "z": 0.4},
      "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    }
  ]
}
```

The occupancy grid connects landmarks **ID 11–20** in `boundary_id_order: [11,12,13,14,15,16,17,18,20,19]` as a closed polygon to form walls.

---

## Topics, Services & TF

### Subscribed Topics

| Topic | Message Type | Consumer Node |
|-------|-------------|---------------|
| `/aruco_poses` | `aruco_sam_ailab/msg/MarkerArray` | graph_optimizer |
| `/w_odom` | `nav_msgs/Odometry` | graph_optimizer |
| Raw IMU (sim: `/camera/imu`, real: `/camera/camera/imu`) | `sensor_msgs/Imu` | imu_preintegration |

### Published Topics

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/aruco_slam/odom` | `nav_msgs/Odometry` | Optimized global pose (map frame) |
| `/aruco_slam/path` | `nav_msgs/Path` | Robot trajectory |
| `/aruco_slam/landmarks` | `visualization_msgs/MarkerArray` | Landmark visualization (3D positions + covariance) |
| `/imu/odom` | `nav_msgs/Odometry` | IMU dead-reckoning odometry (~200 Hz) |
| `/ekf/odom` | `nav_msgs/Odometry` | EKF-smoothed odometry (~200 Hz) |
| `/ego_state` | `hunter_msgs2/EgoState` | Robot kinematic state (x, y, yaw, v, a, yaw_rate) |
| `/map` | `nav_msgs/OccupancyGrid` | Boundary occupancy grid (localization mode) |
| `/aruco_debug_image` | `sensor_msgs/Image` | Detection visualization with marker axes |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/save_landmarks` | `std_srvs/Trigger` | Save current landmarks to JSON (alias: `savem`) |

### TF Tree

| Transform | Published By |
|-----------|-------------|
| `map` → `odom` | static_transform_publisher (identity) |
| `odom` → `base_footprint` | ekf_smoother (EKF-fused pose) |

---

## Message Definitions

### MarkerObservation.msg
```
int32 id                        # ArUco marker ID
geometry_msgs/Pose pose         # Marker pose relative to camera frame
```

### MarkerArray.msg
```
std_msgs/Header header          # Timestamp + frame_id
MarkerObservation[] markers     # Array of marker observations
```

### OptimizedKeyframeState.msg
```
std_msgs/Header header
geometry_msgs/Pose pose         # Robot pose (map frame)
geometry_msgs/Vector3 velocity  # Robot velocity (map frame)
float64[] bias                  # IMU bias [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
```

---

## Package Structure

```
aruco_sam_ailab/
├── assets/
│   ├── Final sim.mp4                  # Simulation demo (mapping + localization + planning)
│   ├── real_solo_results.mp4          # Real robot standalone SLAM
│   └── real_object_avoid_results.mp4  # Real robot SLAM + obstacle avoidance
├── config/
│   ├── slam_params_sim.yaml           # Gazebo simulation parameters
│   └── slam_params_real.yaml          # Real hardware parameters (RealSense D455)
├── launch/
│   └── aruco_slam.launch.py           # Main launch (OpaqueFunction for config auto-selection)
├── map/
│   └── landmarks_map.json             # Landmark map (mapping output / localization input)
├── models/
│   └── aruco_box/                     # Gazebo ArUco box models (ID 11–20)
├── msg/
│   ├── MarkerArray.msg
│   ├── MarkerObservation.msg
│   └── OptimizedKeyframeState.msg
├── scripts/
│   ├── ego_state_publisher.py
│   └── landmark_boundary_occupancy_grid_node.py
├── src/
│   ├── aruco_detector_node.cpp        # ArUco detection (RGB-D + depth correction)
│   ├── graph_optimizer.cpp            # Factor graph SLAM backend (GTSAM ISAM2)
│   ├── imu_preintegration.cpp         # IMU dead-reckoning odometry
│   └── ekf_smoother.cpp              # IMU + SLAM EKF fusion
├── include/aruco_sam_ailab/
│   └── utility.hpp
├── rviz/
│   └── aruco_viz.rviz
├── CMakeLists.txt
├── package.xml
└── README.md
```

---

## Design Rationale

### Why IMU-dominant heading + Wheel-dominant position?

For a 2D UGV on a planar surface:
- **Heading (yaw)**: IMU gyroscope provides high-frequency, smooth heading — wheel odometry yaw is noisy from slip. ArUco rotation noise is set loose (0.50 rad) so it doesn't fight the gyro.
- **Position (x, y)**: Wheel odometry gives reliable short-range translation. IMU position integrates acceleration twice, accumulating drift rapidly.
- **Vertical (z, roll, pitch)**: Tightly constrained to zero — the robot stays on the ground plane.

### Why EKF on top of GTSAM?

GTSAM ISAM2 produces globally optimal poses but only updates at keyframe rate (~10 Hz, only when markers are visible). The planner needs smooth, continuous odometry at ~200 Hz. The EKF bridges this gap:
- **Between SLAM updates**: IMU odometry provides high-frequency prediction
- **On SLAM update**: Kalman correction smoothly incorporates the optimization result
- **Result**: 200 Hz jitter-free odometry with global consistency

### Why depth correction for ArUco?

- PnP-only depth estimation is sensitive to camera calibration errors and marker pose ambiguity
- Raw RGB-D depth is noisy at edges and has holes
- The median-sampling + consensus check approach robustly fuses both sources, rejecting outliers from either

---

## Dependencies

- **ROS 2** Humble
- **GTSAM** (Georgia Tech Smoothing and Mapping) ≥ 4.0
- **OpenCV** ≥ 4.5 + **cv_bridge**
- **Eigen3**
- **tf2**, **tf2_ros**, **tf2_geometry_msgs**
- **hunter_msgs2** (custom messages: `EgoState`)
