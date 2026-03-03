#!/bin/bash
# SLAM 평가용 최소 토픽 rosbag 녹화 스크립트
# 이미지/포인트클라우드 제외 → 용량 절약 & 렉 방지
#
# 사용법:
#   ./record_slam_bag.sh                     # 기본 이름 (mapping_YYYYMMDD_HHMMSS)
#   ./record_slam_bag.sh my_experiment       # 커스텀 이름
#   ./record_slam_bag.sh my_exp /bags        # 커스텀 이름 + 저장 경로

BAG_NAME="${1:-mapping_$(date +%Y%m%d_%H%M%S)}"
BAG_DIR="${2:-/bags}"
BAG_PATH="${BAG_DIR}/${BAG_NAME}"

echo "================================================"
echo "  SLAM 평가용 Rosbag 녹화"
echo "  저장 경로: ${BAG_PATH}"
echo "  Ctrl+C 로 녹화 종료"
echo "================================================"

ros2 bag record -o "${BAG_PATH}" \
    /odom_gt \
    /ekf/odom \
    /ekf/path \
    /aruco_slam/odom \
    /aruco_slam/path \
    /aruco_slam/landmarks \
    /aruco_slam/wheel_odom \
    /aruco_slam/wheel_odom_path \
    /aruco_poses \
    /camera/imu \
    /w_odom \
    /w_odom_path \
    /cmd_vel \
    /ego_state \
    /tf \
    /tf_static \
    /odometry/imu_incremental \
    /distance \
    /optimized_keyframe_state
