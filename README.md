# aruco_sam_ailab

ArUco 마커 기반 Vision SLAM 패키지. Factor Graph 최적화(GTSAM ISAM2)와 휠 오도메트리를 결합하여 로봇의 글로벌 위치 추정 및 맵 생성을 수행합니다.

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [이론적 배경](#이론적-배경)
- [사용법](#사용법)
- [설정](#설정)
- [토픽 및 서비스](#토픽-및-서비스)
- [패키지 구조](#패키지-구조)

---

## 프로젝트 개요

### 주요 기능

| 기능 | 설명 |
|------|------|
| **Mapping** | 환경 내 ArUco 마커를 랜드마크로 등록하며 맵을 생성 |
| **Localization** | 저장된 맵을 로드하여 고정된 랜드마크 기반으로 위치 추정 |
| **Occupancy Grid** | Localization 모드에서 랜드마크 0~9를 직선 연결한 벽 테두리 맵 발행 |

### 동작 모드

- **mapping**: 맵 생성 모드. 새 ArUco 마커를 발견하면 랜드마크로 추가하고 그래프에 반영
- **localization**: 위치 추정 모드. `landmarks_map.json`에서 랜드마크를 로드하고, 로봇 pose만 최적화 (랜드마크는 고정)

모드는 `config/slam_params.yaml`의 `run_mode`에서만 설정합니다.

---

## 시스템 아키텍처

```
                    ┌─────────────────────┐
                    │  aruco_detector_node │
                    │  (RGB-D 카메라)      │
                    └──────────┬──────────┘
                               │ /aruco_poses
                               ▼
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  wheel_odom_node │───▶│   graph_optimizer   │───▶│ /odometry/global     │
│  (joint_states)  │    │   (GTSAM ISAM2)    │    │ /aruco_slam/landmarks│
└──────────────────┘    └──────────┬──────────┘    │ /aruco_slam/path     │
         │                         │              └──────────────────────┘
         │ /w_odom                 │
         └─────────────────────────┘
                               │
         [localization 모드 시]
                               ▼
                    ┌─────────────────────────────┐
                    │ landmark_boundary_occupancy  │
                    │ _grid_node                  │
                    │ → /map (OccupancyGrid)      │
                    │ → /aruco_slam/landmarks     │
                    └─────────────────────────────┘
```

### 노드 구성

| 노드 | 역할 |
|------|------|
| `aruco_detector_node` | RGB-D 카메라에서 ArUco 마커 검출, `camera_color_optical_frame` 기준 pose 발행 |
| `wheel_odom_node` | `joint_states` → 휠 오도메트리 변환, SLAM 결과와 동기화하여 TF 발행 |
| `graph_optimizer` | Factor Graph SLAM 백엔드 (GTSAM ISAM2), Odom + ArUco 랜드마크 Factor |
| `landmark_boundary_occupancy_grid_node` | Localization 시 랜드마크 0~9를 벽 테두리로 한 2D occupancy grid 발행 |

---

## 이론적 배경

### Factor Graph SLAM

로봇 pose와 랜드마크를 노드로, 관측·오도메트리를 Factor로 표현하여 비선형 최적화로 추정합니다.

$$
\min_{x} \sum_i \| h_i(x) - z_i \|_{\Sigma_i}^2
$$

- **X**: 로봇 pose (SE(3))
- **L**: 랜드마크 pose (map frame)
- **PriorFactor**: 첫 pose 고정
- **OdomBetweenFactor**: 휠 오도메트리 기반 pose 간 제약
- **ArUco Factor**: 로봇–랜드마크 상대 pose 관측

### ISAM2 (Incremental Smoothing and Mapping)

GTSAM의 ISAM2는 **incremental** 최적화를 수행합니다. 새 Factor/노드가 추가될 때마다 전체를 재계산하지 않고, 영향받는 부분만 업데이트하여 실시간 SLAM에 적합합니다.

### 키프레임 정책

다음 조건 중 하나라도 만족하면 키프레임을 추가하고 최적화를 수행합니다.

- 첫 프레임
- 시간 간격 ≥ `keyframe_time_interval` (기본 2초)
- 이동 거리 ≥ `keyframe_distance_threshold` (기본 0.5m)
- 회전 각도 ≥ `keyframe_angle_threshold` (기본 약 3°)
- 새로운 ArUco 마커 등장/퇴장

### ZUPT (Zero Velocity Update)

로봇이 거의 정지한 상태(이동 < 1cm, 회전 < 0.5°)일 때는 ArUco 관측을 무시합니다. 카메라 노이즈로 인한 pose 흔들림을 줄이기 위한 기법입니다.

### 랜드마크 기반 Loop Closure

동일 ArUco ID를 재관측하면 **같은 랜드마크 변수**에 추가 Factor가 연결됩니다. ID 기반 loop closure로 drift를 보정합니다.

---

## 사용법

### 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select aruco_sam_ailab
source install/setup.bash
```

### 실행

```bash
# SLAM 전체 실행 (Gazebo 시뮬레이션과 함께 사용 시 use_sim_time=true)
ros2 launch aruco_sam_ailab aruco_slam.launch.py
```

### Launch 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `use_sim_time` | `true` | 시뮬레이션 시간 사용 (Gazebo 연동 시) |
| `odom_topic` | `/w_odom` | 휠 오도메트리 토픽 |

### 맵 저장 (Mapping 모드)

Mapping 모드에서 환경을 주행한 뒤, 다음 서비스로 랜드마크를 저장합니다.

```bash
ros2 service call /save_landmarks std_srvs/srv/Trigger "{}"
```

저장 경로는 `slam_params.yaml`의 `landmarks_save_path`에 지정합니다.

---

## 설정

### slam_params.yaml

모드 및 핵심 파라미터는 `config/slam_params.yaml`에서 설정합니다.

```yaml
# [모드 설정]
run_mode: "mapping"   # 또는 "localization"
map_path: "/path/to/landmarks_map.json"  # localization 시 맵 파일

# ArUco 관측 노이즈 (작을수록 제약 강함)
aruco_trans_noise: 0.02   # m
aruco_rot_noise: 0.05     # rad

# ZUPT (정지 판단)
zupt_trans_thresh: 0.01   # 1cm 미만 = 정지
zupt_rot_thresh: 0.008    # 약 0.5도 미만 = 정지

# 키프레임 정책
keyframe_time_interval: 2.0
keyframe_distance_threshold: 0.5
keyframe_angle_threshold: 0.0524
```

### landmarks_map.json 형식

```json
{
  "frame_id": "map",
  "timestamp": "...",
  "landmarks": [
    {
      "id": 0,
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    }
  ]
}
```

Occupancy grid는 **id 0~9**를 순서 `0→1→2→3→4→5→6→7→9→8→0`으로 직선 연결한 폴리곤을 벽으로 사용합니다.

---

## 토픽 및 서비스

### 구독 토픽

| 토픽 | 메시지 타입 | 노드 |
|------|-------------|------|
| `/aruco_poses` | `aruco_sam_ailab/msg/MarkerArray` | graph_optimizer |
| `/w_odom` | `nav_msgs/Odometry` | graph_optimizer |
| `/joint_states` | `sensor_msgs/JointState` | wheel_odom_node |

### 발행 토픽

| 토픽 | 메시지 타입 | 설명 |
|------|-------------|------|
| `/aruco_slam/odom` | `nav_msgs/Odometry` | 최적화된 글로벌 pose |
| `/aruco_slam/path` | `nav_msgs/Path` | 로봇 경로 |
| `/aruco_slam/landmarks` | `visualization_msgs/MarkerArray` | 랜드마크 시각화 |
| `/map` | `nav_msgs/OccupancyGrid` | 벽 테두리 맵 (localization 시) |
| `/odometry/wheel_odom_correction` | `nav_msgs/Odometry` | 휠 오도메트리 보정 신호 |

### 서비스

| 서비스 | 타입 | 설명 |
|--------|------|------|
| `/save_landmarks` | `std_srvs/Trigger` | 현재 랜드마크를 JSON으로 저장 |

### TF

- `map` → `odom`: graph_optimizer가 간접적으로 제공 (wheel_odom_node가 사용)
- `odom` → `base_link`: wheel_odom_node가 발행

---

## 패키지 구조

```
aruco_sam_ailab/
├── config/
│   └── slam_params.yaml      # SLAM 파라미터
├── launch/
│   └── aruco_slam.launch.py # 메인 launch
├── map/
│   └── landmarks_map.json   # 랜드마크 맵 (localization용)
├── msg/
│   ├── MarkerArray.msg
│   ├── MarkerObservation.msg
│   └── OptimizedKeyframeState.msg
├── scripts/
│   └── landmark_boundary_occupancy_grid_node.py
├── src/
│   ├── aruco_detector_node.cpp
│   ├── graph_optimizer.cpp
│   ├── wheel_odom_node.cpp
│   └── imu_preintegration.cpp
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

- **ROS 2** (Humble)
- **GTSAM** (Georgia Tech Smoothing and Mapping)
- **OpenCV**
- **cv_bridge**

---

## 라이선스

Apache License, Version 2.0
