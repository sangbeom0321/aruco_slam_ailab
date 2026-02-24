# aruco_sam_ailab

ArUco 마커 기반 Vision-IMU SLAM 패키지. Factor Graph 최적화(GTSAM ISAM2), IMU Preintegration, EKF Smoother를 결합하여 로봇의 글로벌 위치 추정 및 맵 생성을 수행합니다.

---

## 목차

- [시스템 아키텍처](#시스템-아키텍처)
- [사용법](#사용법)
- [설정 파일](#설정-파일)
- [토픽 및 서비스](#토픽-및-서비스)
- [패키지 구조](#패키지-구조)

---

## 시스템 아키텍처

```
                    ┌─────────────────────┐
                    │  aruco_detector_node │
                    │  (RGB-D 카메라)      │
                    └──────────┬──────────┘
                               │ /aruco_poses
                               ▼
┌──────────────────┐    ┌─────────────────────┐
│  Wheel Odometry  │───▶│   graph_optimizer   │───▶ /aruco_slam/odom
│  (/w_odom)       │    │   (GTSAM ISAM2)    │    /aruco_slam/landmarks
└──────────────────┘    └──────────┬──────────┘    /aruco_slam/path
                                   │
┌──────────────────┐               │ /optimized_keyframe_state
│ imu_preintegration│              ▼
│ (raw IMU → odom) │───▶┌──────────────────┐
└──────────────────┘    │   ekf_smoother   │───▶ /ekf/odom
                        │ (IMU + SLAM 융합) │    odom→base_footprint TF
                        └──────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │ ego_state_publisher│───▶ /ego_state
                        └──────────────────┘

         [localization 모드 시]
                        ┌─────────────────────────────┐
                        │ landmark_boundary_occupancy  │
                        │ _grid_node                  │
                        │ → /map (OccupancyGrid)      │
                        └─────────────────────────────┘
```

### 노드 구성

| 노드 | 역할 |
|------|------|
| `aruco_detector_node` | RGB-D 카메라에서 ArUco 마커(ID 11~20) 검출, PnP + Depth 보정 |
| `graph_optimizer` | Factor Graph SLAM 백엔드 (GTSAM ISAM2), Wheel Odom + ArUco Factor |
| `imu_preintegration` | Raw IMU → dead reckoning odom (~200Hz) |
| `ekf_smoother` | IMU odom + SLAM correction 융합 → smoothed odom, TF 발행 |
| `ego_state_publisher` | /ekf/odom → /ego_state 변환 |
| `landmark_boundary_occupancy_grid_node` | Localization 시 ID 11~20 랜드마크를 벽 테두리 occupancy grid로 발행 |

### 동작 모드

| 모드 | 설명 |
|------|------|
| **mapping** | 환경 내 ArUco 마커를 랜드마크로 등록하며 맵을 생성 |
| **localization** | 저장된 맵을 로드하여 고정된 랜드마크 기반으로 위치 추정 + occupancy grid 발행 |

---

## 사용법

### 빌드

```bash
cd /ros_ws
colcon build --packages-select aruco_sam_ailab --symlink-install
source install/setup.bash
```

### 실행 — 시뮬레이션 (Gazebo)

```bash
# 단독 실행
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=true

# bringup 통합 실행 (Gazebo + SLAM + Planning + RViz)
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=true run_mode:=mapping
```

`use_sim_time:=true` → `config/slam_params_sim.yaml` 자동 로드

### 실행 — 실제 로봇

```bash
# 단독 실행
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false

# bringup 통합 실행 (하드웨어 드라이버 + SLAM + Planning + RViz)
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=false run_mode:=mapping
```

`use_sim_time:=false` → `config/slam_params_real.yaml` 자동 로드

### Launch 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `use_sim_time` | `false` | `true`: sim config 로드, `false`: real config 로드 |
| `run_mode` | `mapping` | `"mapping"` (맵 생성) 또는 `"localization"` (위치 추정) |
| `map_path` | (config 기본값) | Localization 모드에서 로드할 맵 파일 경로 |
| `marker_size` | `0.30` | ArUco 마커 크기 (m) |
| `enable_topic_debug_log` | `false` | ArUco 토픽 수신 디버그 로그 |

### 맵 저장 (Mapping 모드)

Mapping 모드에서 환경을 주행한 뒤, 다음 명령으로 랜드마크 맵을 저장합니다.

```bash
# 전체 명령
ros2 service call /save_landmarks std_srvs/srv/Trigger "{}"

# Docker 컨테이너 안에서 alias 사용
savem
```

저장 경로: `landmarks_save_path` 파라미터 (기본 `/ros_ws/src/aruco_sam_ailab/map/landmarks_map.json`)

### Localization 모드 전환

```bash
# 저장된 맵으로 localization 실행
ros2 launch hunter2_bringup bringup.launch.py use_sim_time:=false run_mode:=localization

# 또는 커스텀 맵 파일 지정
ros2 launch aruco_sam_ailab aruco_slam.launch.py use_sim_time:=false \
  run_mode:=localization map_path:=/path/to/my_map.json
```

### 디버그

```bash
# SLAM 최적화 결과 확인
ros2 topic echo /aruco_slam/odom

# EKF 융합 odom 확인
ros2 topic echo /ekf/odom

# ArUco 검출 결과 확인
ros2 topic echo /aruco_poses

# 파라미터 확인 (환경별 config 적용 여부)
ros2 param get /imu_preintegration imu_topic
ros2 param get /graph_optimizer imu_frame
```

---

## 설정 파일

### Config 분리 구조

`use_sim_time` 값에 따라 자동 선택됩니다.

| 파일 | 조건 | 용도 |
|------|------|------|
| `config/slam_params_sim.yaml` | `use_sim_time:=true` | Gazebo 시뮬레이션 |
| `config/slam_params_real.yaml` | `use_sim_time:=false` | 실제 하드웨어 |

### 환경별 주요 차이점

| 파라미터 | Sim (`slam_params_sim.yaml`) | Real (`slam_params_real.yaml`) |
|---------|-----|------|
| `imu_topic` | `/camera/imu` | `/camera/camera/imu` |
| `imu_frame` | `base_footprint` | `camera_imu_optical_frame` |
| `ext_trans_base_imu` | `[0, 0, 0]` (identity) | `[0.395, 0.017, 0.405]` |
| `ext_rot_base_imu` | identity matrix | `[0,0,1, -1,0,0, 0,-1,0]` |

### 공통 파라미터 (양쪽 동일)

```yaml
# 모드 설정
run_mode: "mapping"       # "mapping" | "localization"
map_path: "/ros_ws/src/aruco_sam_ailab/map/landmarks_map.json"

# ArUco 관측 노이즈
aruco_trans_noise: 0.08   # position 보정 신뢰도
aruco_rot_noise: 0.50     # rotation은 loose (heading은 IMU 주도)

# Wheel Odometry 노이즈
wheel_odom_noise_x: 0.05  # tight (wheel odom position 신뢰)
wheel_odom_noise_yaw: 1.0 # loose (heading은 IMU가 담당)

# 키프레임 정책
keyframe_time_interval: 0.5     # 0.5초 간격
keyframe_distance_threshold: 0.15  # 15cm 이동
keyframe_angle_threshold: 0.1     # ~5.7° 회전

# EKF Smoother
ekf_process_noise_pos: 0.15   # IMU position 드리프트 허용
ekf_process_noise_rot: 0.02   # IMU heading 신뢰
ekf_measurement_noise_pos: 0.05  # SLAM position 보정 빠르게
ekf_measurement_noise_rot: 0.15  # heading 보정은 부드럽게
```

### landmarks_map.json 형식

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

Occupancy grid는 **ID 11~20**을 `boundary_id_order: [11,12,13,14,15,16,17,18,20,19]` 순서로 직선 연결한 폴리곤을 벽으로 사용합니다.

---

## 토픽 및 서비스

### 구독 토픽

| 토픽 | 메시지 타입 | 노드 |
|------|-------------|------|
| `/aruco_poses` | `aruco_sam_ailab/msg/MarkerArray` | graph_optimizer |
| `/w_odom` | `nav_msgs/Odometry` | graph_optimizer |
| Raw IMU (sim: `/camera/imu`, real: `/camera/camera/imu`) | `sensor_msgs/Imu` | imu_preintegration |

### 발행 토픽

| 토픽 | 메시지 타입 | 설명 |
|------|-------------|------|
| `/aruco_slam/odom` | `nav_msgs/Odometry` | 최적화된 글로벌 pose (map frame) |
| `/aruco_slam/path` | `nav_msgs/Path` | 로봇 경로 |
| `/aruco_slam/landmarks` | `visualization_msgs/MarkerArray` | 랜드마크 시각화 |
| `/imu/odom` | `nav_msgs/Odometry` | IMU dead reckoning odom (~200Hz) |
| `/ekf/odom` | `nav_msgs/Odometry` | EKF 융합 odom (~200Hz) |
| `/ego_state` | `hunter_msgs2/EgoState` | 로봇 상태 (x, y, yaw, v, a, yaw_rate) |
| `/map` | `nav_msgs/OccupancyGrid` | 벽 테두리 맵 (localization 시) |

### 서비스

| 서비스 | 타입 | 설명 |
|--------|------|------|
| `/save_landmarks` | `std_srvs/Trigger` | 현재 랜드마크를 JSON으로 저장 (alias: `savem`) |

### TF

| 변환 | 발행 노드 |
|------|----------|
| `map` → `odom` | static_transform_publisher (identity) |
| `odom` → `base_footprint` | ekf_smoother (EKF 융합 pose) |

---

## 패키지 구조

```
aruco_sam_ailab/
├── config/
│   ├── slam_params_sim.yaml   # 시뮬레이션 파라미터
│   └── slam_params_real.yaml  # 실제 하드웨어 파라미터
├── launch/
│   └── aruco_slam.launch.py   # 메인 launch (OpaqueFunction으로 config 자동 선택)
├── map/
│   └── landmarks_map.json     # 랜드마크 맵 (mapping 결과 / localization 입력)
├── models/
│   └── aruco_box/             # Gazebo용 ArUco 박스 모델 (ID 11~20)
├── msg/
│   ├── MarkerArray.msg
│   ├── MarkerObservation.msg
│   └── OptimizedKeyframeState.msg
├── scripts/
│   ├── ego_state_publisher.py
│   └── landmark_boundary_occupancy_grid_node.py
├── src/
│   ├── aruco_detector_node.cpp    # ArUco 검출 (RGB-D + Depth 보정)
│   ├── graph_optimizer.cpp        # Factor Graph SLAM 백엔드
│   ├── imu_preintegration.cpp     # IMU dead reckoning
│   └── ekf_smoother.cpp          # IMU + SLAM EKF 융합
├── include/aruco_sam_ailab/
│   └── utility.hpp
├── rviz/
│   └── aruco_viz.rviz
├── CMakeLists.txt
├── package.xml
└── README.md
```

---

## 의존성

- **ROS 2** Humble
- **GTSAM** (Georgia Tech Smoothing and Mapping)
- **OpenCV** + **cv_bridge**
- **Eigen3**
