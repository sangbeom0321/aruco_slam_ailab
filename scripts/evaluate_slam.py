#!/usr/bin/env python3
"""
SLAM 평가 스크립트 (Rosbag 후처리)
===================================
Gazebo GT 포즈와 SLAM 추정 포즈를 비교하여 정량 평가 지표를 산출합니다.

사용법:
    python3 evaluate_slam.py <rosbag_path> [옵션]

평가 지표:
    - APE (Absolute Pose Error): 전역 위치/회전 정확도
    - RPE (Relative Pose Error): 로컬 일관성 (delta=1m, 5m, 10m)
    - Drift Rate: 주행 거리 대비 누적 오차 비율
    - Landmark Accuracy: 마커 위치 추정 정밀도 (옵션)

출력:
    - 콘솔 요약 테이블
    - CSV 파일 (metrics.csv)
    - matplotlib 그래프 (6장)
    - TUM 포맷 궤적 파일 (evo CLI 호환)

의존성:
    pip install rosbags evo matplotlib numpy scipy
"""

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

# evo 라이브러리 임포트
try:
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D
except ImportError:
    print("[오류] evo 라이브러리가 필요합니다: pip install evo --upgrade")
    sys.exit(1)

# rosbags 임포트
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("[오류] rosbags 라이브러리가 필요합니다: pip install rosbags")
    sys.exit(1)

# matplotlib 임포트 (headless 환경 대응)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────
# Gazebo 마커 GT 위치 (gazebo_aruco.launch.py)
# Gazebo 좌표계 (X, Y) — Z는 0.40m
# ─────────────────────────────────────────────
GAZEBO_MARKER_POSITIONS = {
    11: (0.0, 3.0),
    12: (1.9, 3.0),
    13: (1.9, -1.5),
    14: (4.6, -1.5),
    15: (4.6, -3.75),
    16: (4.6, -6.0),
    17: (1.45, -6.0),
    18: (-1.7, -6.0),
    19: (-1.9, 3.0),
    20: (-1.7, -1.5),
}

# GT 토픽 후보 (우선순위 순)
GT_TOPIC_CANDIDATES = ["/odom_gt", "/w_odom", "/odom"]
# 추정 토픽 후보 (우선순위 순)
EST_TOPIC_CANDIDATES = ["/ekf/odom", "/aruco_slam/odom"]


# ═══════════════════════════════════════════════
# [1] Rosbag 추출
# ═══════════════════════════════════════════════

def detect_available_topics(bag_path: str) -> dict:
    """rosbag에서 nav_msgs/Odometry 토픽 목록과 메시지 수를 반환합니다."""
    topics = {}
    with Reader(bag_path) as reader:
        for conn in reader.connections:
            if conn.msgtype == "nav_msgs/msg/Odometry":
                topics[conn.topic] = conn.msgcount
    return topics


def extract_odometry(bag_path: str, topic: str) -> list:
    """
    rosbag2에서 nav_msgs/Odometry 메시지를 추출합니다.

    Returns:
        list of tuples: (timestamp_sec, x, y, z, qx, qy, qz, qw)
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    poses = []

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"[경고] 토픽 '{topic}'을 rosbag에서 찾을 수 없습니다.")
            return poses

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            poses.append((t_sec, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    print(f"  [{topic}] {len(poses)}개 메시지 추출 완료")
    return poses


def auto_select_topics(available: dict) -> tuple:
    """사용 가능한 토픽에서 GT/추정 토픽을 자동 선택합니다."""
    gt_topic = None
    for cand in GT_TOPIC_CANDIDATES:
        if cand in available:
            gt_topic = cand
            break

    est_topics = []
    for cand in EST_TOPIC_CANDIDATES:
        if cand in available:
            est_topics.append(cand)

    return gt_topic, est_topics


# ═══════════════════════════════════════════════
# [2] 궤적 변환
# ═══════════════════════════════════════════════

def odom_list_to_arrays(odom_list: list) -> tuple:
    """odom list → (timestamps, positions Nx3, quaternions Nx4) 분리."""
    arr = np.array(odom_list, dtype=np.float64)
    timestamps = arr[:, 0]
    positions = arr[:, 1:4]  # x, y, z
    quats = arr[:, 4:8]      # qx, qy, qz, qw
    return timestamps, positions, quats


def quats_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    """쿼터니언 배열(Nx4, [qx,qy,qz,qw]) → 회전 행렬 배열(Nx3x3)."""
    n = quats.shape[0]
    R = np.zeros((n, 3, 3))
    for i in range(n):
        qx, qy, qz, qw = quats[i]
        R[i] = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
        ])
    return R


def reorthogonalize(R: np.ndarray) -> np.ndarray:
    """SVD로 3x3 회전 행렬을 SO(3)로 재투영."""
    U, _, Vt = np.linalg.svd(R)
    d = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def reorthogonalize_poses(poses_se3: np.ndarray) -> np.ndarray:
    """SE(3) 배열의 회전 부분을 SO(3)로 재투영."""
    out = poses_se3.copy()
    for i in range(len(out)):
        out[i, :3, :3] = reorthogonalize(out[i, :3, :3])
    return out


def to_evo_trajectory(odom_list: list) -> PoseTrajectory3D:
    """odom list → evo PoseTrajectory3D 변환."""
    timestamps, positions, quats = odom_list_to_arrays(odom_list)
    # evo는 SE(3) 행렬 배열 (Nx4x4)을 받음
    rot_mats = quats_to_rotation_matrices(quats)
    n = len(timestamps)
    poses_se3 = np.zeros((n, 4, 4))
    # 쿼터니언 정규화 오차로 인한 SO(3) 위반 방지
    for i in range(n):
        rot_mats[i] = reorthogonalize(rot_mats[i])
    poses_se3[:, :3, :3] = rot_mats
    poses_se3[:, :3, 3] = positions
    poses_se3[:, 3, 3] = 1.0
    return PoseTrajectory3D(poses_se3=poses_se3, timestamps=timestamps)


def save_tum_file(odom_list: list, filepath: str):
    """TUM 포맷으로 저장: timestamp x y z qx qy qz qw"""
    with open(filepath, "w") as f:
        for row in odom_list:
            t, x, y, z, qx, qy, qz, qw = row
            f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    print(f"  TUM 파일 저장: {filepath}")


# ═══════════════════════════════════════════════
# [3] 좌표 정렬
# ═══════════════════════════════════════════════

def align_trajectories(traj_gt: PoseTrajectory3D,
                       traj_est: PoseTrajectory3D,
                       method: str = "origin",
                       max_diff: float = 0.05):
    """
    GT/추정 궤적 시간 동기화 및 정렬.

    Args:
        method: 'origin' (첫 포즈 기준), 'umeyama' (SE3 Umeyama), 'none'

    Returns:
        (traj_gt_sync, traj_est_aligned, alignment_info)
    """
    # 시간 동기화
    traj_gt_sync, traj_est_sync = sync.associate_trajectories(
        traj_gt, traj_est, max_diff=max_diff)

    n_sync = traj_gt_sync.num_poses
    print(f"  시간 동기화: {n_sync}개 포즈 매칭 (max_diff={max_diff}s)")

    if n_sync < 10:
        print("[경고] 동기화된 포즈가 10개 미만입니다. max_diff를 늘려보세요.")

    info = {"method": method, "n_synced": n_sync}

    if method == "none":
        return traj_gt_sync, traj_est_sync, info

    if method == "origin":
        # 첫 포즈 기준 SE(3) 정렬
        T_gt0 = traj_gt_sync.poses_se3[0]
        T_est0 = traj_est_sync.poses_se3[0]
        T_align = T_gt0 @ np.linalg.inv(T_est0)

        aligned_poses = reorthogonalize_poses(
            np.array([T_align @ p for p in traj_est_sync.poses_se3]))
        traj_est_aligned = PoseTrajectory3D(
            poses_se3=aligned_poses, timestamps=traj_est_sync.timestamps)

        info["T_align"] = T_align
        return traj_gt_sync, traj_est_aligned, info

    if method == "umeyama":
        # SE(3) Umeyama 정렬
        from evo.core.geometry import umeyama_alignment

        gt_xyz = traj_gt_sync.positions_xyz.T   # 3xN
        est_xyz = traj_est_sync.positions_xyz.T  # 3xN
        rot, trans, _ = umeyama_alignment(est_xyz, gt_xyz, with_scale=False)

        # SE(3) 변환 행렬 구성
        T_align = np.eye(4)
        T_align[:3, :3] = rot
        T_align[:3, 3] = trans

        aligned_poses = reorthogonalize_poses(
            np.array([T_align @ p for p in traj_est_sync.poses_se3]))
        traj_est_aligned = PoseTrajectory3D(
            poses_se3=aligned_poses, timestamps=traj_est_sync.timestamps)

        info["T_align"] = T_align
        info["note"] = "SE(3) Umeyama alignment applied"
        return traj_gt_sync, traj_est_aligned, info

    raise ValueError(f"알 수 없는 정렬 방법: {method}")


# ═══════════════════════════════════════════════
# [4] 지표 산출
# ═══════════════════════════════════════════════

def compute_ape(traj_gt: PoseTrajectory3D,
                traj_est: PoseTrajectory3D) -> dict:
    """APE (Absolute Pose Error) 계산."""
    # 병진 APE
    ape_trans = metrics.APE(PoseRelation.translation_part)
    ape_trans.process_data((traj_gt, traj_est))
    stats_t = ape_trans.get_all_statistics()

    # 회전 APE (도 단위)
    ape_rot = metrics.APE(PoseRelation.rotation_angle_deg)
    ape_rot.process_data((traj_gt, traj_est))
    stats_r = ape_rot.get_all_statistics()

    return {
        "trans_rmse": stats_t["rmse"],
        "trans_mean": stats_t["mean"],
        "trans_median": stats_t["median"],
        "trans_std": stats_t["std"],
        "trans_max": stats_t["max"],
        "rot_rmse": stats_r["rmse"],
        "rot_mean": stats_r["mean"],
        "rot_max": stats_r["max"],
        "trans_errors": ape_trans.error,
        "rot_errors": ape_rot.error,
        "timestamps": traj_gt.timestamps,
    }


def compute_rpe(traj_gt: PoseTrajectory3D,
                traj_est: PoseTrajectory3D,
                delta_m: float) -> dict:
    """RPE (Relative Pose Error) 계산."""
    try:
        rpe_trans = metrics.RPE(
            PoseRelation.translation_part,
            delta=delta_m, delta_unit=Unit.meters, all_pairs=False)
        rpe_trans.process_data((traj_gt, traj_est))
        stats_t = rpe_trans.get_all_statistics()

        rpe_rot = metrics.RPE(
            PoseRelation.rotation_angle_deg,
            delta=delta_m, delta_unit=Unit.meters, all_pairs=False)
        rpe_rot.process_data((traj_gt, traj_est))
        stats_r = rpe_rot.get_all_statistics()

        return {
            "delta_m": delta_m,
            "trans_rmse": stats_t["rmse"],
            "trans_mean": stats_t["mean"],
            "trans_std": stats_t["std"],
            "rot_rmse": stats_r["rmse"],
            "rot_mean": stats_r["mean"],
            "trans_errors": rpe_trans.error,
        }
    except Exception as e:
        print(f"  [경고] RPE(delta={delta_m}m) 계산 실패: {e}")
        return {
            "delta_m": delta_m,
            "trans_rmse": float("nan"),
            "trans_mean": float("nan"),
            "trans_std": float("nan"),
            "rot_rmse": float("nan"),
            "rot_mean": float("nan"),
            "trans_errors": np.array([]),
        }


def compute_drift_rate(traj_gt: PoseTrajectory3D,
                       traj_est: PoseTrajectory3D) -> dict:
    """Drift Rate 계산: endpoint error / total travel distance × 100%."""
    gt_pos = traj_gt.positions_xyz
    est_pos = traj_est.positions_xyz

    # 총 주행 거리 (GT 기준)
    diffs = np.diff(gt_pos, axis=0)
    travel_dist = np.sum(np.linalg.norm(diffs, axis=1))

    # 끝점 오차
    endpoint_err = np.linalg.norm(gt_pos[-1] - est_pos[-1])

    # Yaw 끝점 오차
    gt_yaw = rotation_matrix_to_yaw(traj_gt.poses_se3[-1][:3, :3])
    est_yaw = rotation_matrix_to_yaw(traj_est.poses_se3[-1][:3, :3])
    yaw_err = abs(normalize_angle(gt_yaw - est_yaw))

    drift_pct = (endpoint_err / travel_dist * 100.0) if travel_dist > 0 else float("nan")

    return {
        "endpoint_error_m": endpoint_err,
        "endpoint_yaw_error_deg": math.degrees(yaw_err),
        "total_travel_distance_m": travel_dist,
        "drift_rate_pct": drift_pct,
    }


def rotation_matrix_to_yaw(R: np.ndarray) -> float:
    """3x3 회전 행렬 → yaw (Z축 회전각)."""
    return math.atan2(R[1, 0], R[0, 0])


def normalize_angle(angle: float) -> float:
    """각도를 [-pi, pi] 범위로 정규화."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def compute_landmark_accuracy(landmarks_map_path: str) -> dict:
    """
    SLAM 추정 랜드마크와 Gazebo GT 마커 위치를 비교합니다.

    Gazebo 마커 위치를 ROS 좌표계로 변환 후, 로봇 초기 포즈 기준 원점 리셋을 적용합니다.
    로봇 Gazebo spawn: (0, 0, yaw=-90°)
    → ROS 변환 후: (0, 0, yaw=-180°)
    → 원점 리셋: 첫 포즈 R^T * (p - p0) → R(180°)^T * (p_ros)
    """
    # SLAM 추정 랜드마크 로드
    with open(landmarks_map_path, "r") as f:
        data = json.load(f)

    est_landmarks = {}
    for lm in data["landmarks"]:
        est_landmarks[lm["id"]] = np.array([
            lm["position"]["x"],
            lm["position"]["y"],
            lm["position"]["z"],
        ])

    # Gazebo → ROS 좌표 변환 + 원점 리셋
    # 로봇 spawn: Gazebo (0, 0, yaw=-90°)
    # ROS 변환: (Y_gz, -X_gz) = (0, 0), yaw_ros = -90° + (-90°) = -180°
    # 원점 리셋: R(-180°)^T * (p_ros - 0) = R(180°) * p_ros
    # R(180°) = [[-1, 0], [0, -1]]
    robot_yaw_ros = math.radians(-180.0)
    cos_y = math.cos(robot_yaw_ros)  # -1
    sin_y = math.sin(robot_yaw_ros)  # ~0

    gt_landmarks = {}
    for mid, (gx, gy) in GAZEBO_MARKER_POSITIONS.items():
        # Gazebo → ROS: X_ros = Y_gz, Y_ros = -X_gz
        rx, ry = gy, -gx
        # 원점 리셋: R(-yaw)^T * (p - p0)
        # R(-yaw)^T = R(yaw) → 여기서 yaw = -180°
        ox = cos_y * rx - sin_y * ry
        oy = sin_y * rx + cos_y * ry
        gt_landmarks[mid] = np.array([ox, oy])

    # 비교
    per_marker = {}
    for mid in sorted(set(est_landmarks.keys()) & set(gt_landmarks.keys())):
        est_xy = est_landmarks[mid][:2]
        gt_xy = gt_landmarks[mid]
        err = np.linalg.norm(est_xy - gt_xy)
        per_marker[mid] = err

    if not per_marker:
        return {"per_marker": {}, "mean_error": float("nan"), "max_error": float("nan")}

    errors = list(per_marker.values())
    max_id = max(per_marker, key=per_marker.get)

    return {
        "per_marker": per_marker,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "max_error_id": max_id,
        "est_landmarks": est_landmarks,
        "gt_landmarks": gt_landmarks,
    }


# ═══════════════════════════════════════════════
# [5] 시각화
# ═══════════════════════════════════════════════

def extract_yaws(traj: PoseTrajectory3D) -> np.ndarray:
    """궤적에서 yaw 배열 추출 (rad)."""
    yaws = np.zeros(traj.num_poses)
    for i, pose in enumerate(traj.poses_se3):
        yaws[i] = rotation_matrix_to_yaw(pose[:3, :3])
    return yaws


def plot_trajectories_2d(traj_gt, traj_est_dict, output_dir, landmarks=None):
    """2D XY 궤적 비교 그래프."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # GT 궤적
    gt_xy = traj_gt.positions_xyz[:, :2]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2.0, label="GT", zorder=3)
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], "go", markersize=10, label="시작점", zorder=5)
    ax.plot(gt_xy[-1, 0], gt_xy[-1, 1], "rx", markersize=10, mew=2, label="끝점", zorder=5)

    # 추정 궤적
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, traj) in enumerate(traj_est_dict.items()):
        xy = traj.positions_xyz[:, :2]
        color = colors[idx % len(colors)]
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, linewidth=1.5,
                label=name, alpha=0.8, zorder=2)

    # 랜드마크
    if landmarks:
        gt_lm = landmarks.get("gt_landmarks", {})
        est_lm = landmarks.get("est_landmarks", {})
        for mid, pos in gt_lm.items():
            ax.plot(pos[0], pos[1], "k^", markersize=8, zorder=4)
            ax.annotate(f"M{mid}", (pos[0], pos[1]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        for mid, pos in est_lm.items():
            ax.plot(pos[0], pos[1], "m^", markersize=6, alpha=0.7, zorder=4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("2D Trajectory Comparison")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_trajectory_2d.png"), dpi=150)
    plt.close(fig)


def plot_ape_over_time(ape_result, output_dir):
    """APE 시계열 그래프."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t = ape_result["timestamps"] - ape_result["timestamps"][0]
    ax1.plot(t, ape_result["trans_errors"], "b-", linewidth=0.8)
    ax1.axhline(y=ape_result["trans_rmse"], color="r", linestyle="--",
                label=f"RMSE = {ape_result['trans_rmse']:.3f} m")
    ax1.set_ylabel("APE Translation (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, ape_result["rot_errors"], "b-", linewidth=0.8)
    ax2.axhline(y=ape_result["rot_rmse"], color="r", linestyle="--",
                label=f"RMSE = {ape_result['rot_rmse']:.2f}\u00b0")
    ax2.set_ylabel("APE Rotation (\u00b0)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Absolute Pose Error over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_ape_time.png"), dpi=150)
    plt.close(fig)


def plot_ape_distribution(ape_result, output_dir):
    """APE 분포 히스토그램 + CDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    errs = ape_result["trans_errors"]
    ax1.hist(errs, bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax1.axvline(x=ape_result["trans_rmse"], color="r", linestyle="--",
                label=f"RMSE = {ape_result['trans_rmse']:.3f} m")
    ax1.axvline(x=ape_result["trans_median"], color="g", linestyle="--",
                label=f"Median = {ape_result['trans_median']:.3f} m")
    ax1.set_xlabel("APE Translation (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram")
    ax1.legend()

    sorted_errs = np.sort(errs)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
    ax2.plot(sorted_errs, cdf, "b-", linewidth=1.5)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("APE Translation (m)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative Distribution")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("APE Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_ape_distribution.png"), dpi=150)
    plt.close(fig)


def plot_rpe_by_delta(rpe_results, output_dir):
    """RPE delta별 박스 플롯."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = []
    trans_data = []
    rot_data = []

    for rpe in rpe_results:
        d = rpe["delta_m"]
        labels.append(f"\u0394={d}m")
        if len(rpe["trans_errors"]) > 0:
            trans_data.append(rpe["trans_errors"])
        else:
            trans_data.append([float("nan")])

    # 병진 RPE 박스 플롯
    if trans_data:
        bp1 = ax1.boxplot(trans_data, labels=labels, patch_artist=True)
        for patch in bp1["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.5)
    ax1.set_ylabel("RPE Translation (m)")
    ax1.set_title("Translation RPE by Distance Interval")
    ax1.grid(True, alpha=0.3, axis="y")

    # RMSE 막대 그래프
    deltas = [rpe["delta_m"] for rpe in rpe_results]
    trans_rmses = [rpe["trans_rmse"] for rpe in rpe_results]
    rot_rmses = [rpe["rot_rmse"] for rpe in rpe_results]

    x = np.arange(len(deltas))
    width = 0.35
    ax2.bar(x - width/2, trans_rmses, width, label="Trans RMSE (m)", color="#1f77b4")
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, rot_rmses, width, label="Rot RMSE (\u00b0)", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"\u0394={d}m" for d in deltas])
    ax2.set_ylabel("Trans RMSE (m)")
    ax2_twin.set_ylabel("Rot RMSE (\u00b0)")
    ax2.set_title("RPE RMSE by Delta")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    fig.suptitle("Relative Pose Error Analysis")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_rpe_analysis.png"), dpi=150)
    plt.close(fig)


def plot_landmark_comparison(lm_result, output_dir):
    """GT vs 추정 랜드마크 위치 비교."""
    gt_lm = lm_result["gt_landmarks"]
    est_lm = lm_result["est_landmarks"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2D 위치 비교
    for mid in sorted(gt_lm.keys()):
        gt_pos = gt_lm[mid]
        ax1.plot(gt_pos[0], gt_pos[1], "k^", markersize=10)
        ax1.annotate(f"GT {mid}", (gt_pos[0], gt_pos[1]),
                     fontsize=7, xytext=(-10, 8), textcoords="offset points")
        if mid in est_lm:
            est_pos = est_lm[mid][:2]
            ax1.plot(est_pos[0], est_pos[1], "r^", markersize=8, alpha=0.7)
            ax1.plot([gt_pos[0], est_pos[0]], [gt_pos[1], est_pos[1]],
                     "r--", alpha=0.4, linewidth=0.8)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Landmark Positions: GT (black) vs SLAM (red)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # 마커별 오차 막대 그래프
    per_marker = lm_result["per_marker"]
    ids = sorted(per_marker.keys())
    errors = [per_marker[mid] for mid in ids]

    ax2.bar([f"M{mid}" for mid in ids], errors, color="#1f77b4", alpha=0.7)
    ax2.axhline(y=lm_result["mean_error"], color="r", linestyle="--",
                label=f"Mean = {lm_result['mean_error']:.3f} m")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title("Per-Marker Estimation Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Landmark Estimation Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_landmark_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_yaw_comparison(traj_gt, traj_est, output_dir):
    """GT vs 추정 yaw 비교."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t_gt = traj_gt.timestamps - traj_gt.timestamps[0]
    t_est = traj_est.timestamps - traj_est.timestamps[0]

    yaw_gt = np.degrees(extract_yaws(traj_gt))
    yaw_est = np.degrees(extract_yaws(traj_est))

    ax1.plot(t_gt, yaw_gt, "k-", linewidth=1.5, label="GT")
    ax1.plot(t_est, yaw_est, "b-", linewidth=1.0, alpha=0.8, label="Estimated")
    ax1.set_ylabel("Yaw (\u00b0)")
    ax1.set_title("Heading Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 동기화된 yaw 차이 (evo sync 후 같은 길이)
    n = min(len(yaw_gt), len(yaw_est))
    yaw_diff = np.array([
        math.degrees(normalize_angle(
            math.radians(yaw_gt[i]) - math.radians(yaw_est[i])))
        for i in range(n)])
    t_diff = t_gt[:n]

    ax2.plot(t_diff, yaw_diff, "r-", linewidth=0.8)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Yaw Error (\u00b0)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_yaw_comparison.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════
# [6] 리포트
# ═══════════════════════════════════════════════

def print_summary(ape, rpe_list, drift, lm_result, alignment_info, gt_topic, est_topic):
    """콘솔 요약 테이블 출력."""
    print()
    print("=" * 62)
    print("       ArUco-SAM SLAM 정량 평가 결과")
    print("=" * 62)
    print(f"  GT 토픽:    {gt_topic}")
    print(f"  추정 토픽:  {est_topic}")
    print(f"  정렬 방법:  {alignment_info['method']}")
    print(f"  동기화:     {alignment_info['n_synced']}개 포즈")
    print("-" * 62)

    print("  [APE - Translation]")
    print(f"    RMSE:   {ape['trans_rmse']:.4f} m")
    print(f"    Mean:   {ape['trans_mean']:.4f} m")
    print(f"    Median: {ape['trans_median']:.4f} m")
    print(f"    Std:    {ape['trans_std']:.4f} m")
    print(f"    Max:    {ape['trans_max']:.4f} m")
    print(f"  [APE - Rotation]")
    print(f"    RMSE:   {ape['rot_rmse']:.2f}\u00b0")
    print(f"    Mean:   {ape['rot_mean']:.2f}\u00b0")
    print(f"    Max:    {ape['rot_max']:.2f}\u00b0")
    print("-" * 62)

    print("  [RPE]")
    for rpe in rpe_list:
        d = rpe["delta_m"]
        print(f"    \u0394={d}m  Trans RMSE: {rpe['trans_rmse']:.4f} m"
              f"  Rot RMSE: {rpe['rot_rmse']:.2f}\u00b0")
    print("-" * 62)

    print("  [Drift Rate]")
    print(f"    Endpoint Error:     {drift['endpoint_error_m']:.4f} m")
    print(f"    Endpoint Yaw Error: {drift['endpoint_yaw_error_deg']:.2f}\u00b0")
    print(f"    Travel Distance:    {drift['total_travel_distance_m']:.2f} m")
    print(f"    Drift Rate:         {drift['drift_rate_pct']:.3f} %")

    if lm_result and not math.isnan(lm_result.get("mean_error", float("nan"))):
        print("-" * 62)
        print("  [Landmark Accuracy]")
        print(f"    Mean Error: {lm_result['mean_error']:.4f} m")
        print(f"    Max Error:  {lm_result['max_error']:.4f} m (ID {lm_result['max_error_id']})")
        for mid, err in sorted(lm_result["per_marker"].items()):
            print(f"    Marker {mid}: {err:.4f} m")

    print("=" * 62)
    print()


def save_metrics_csv(ape, rpe_list, drift, lm_result, output_dir):
    """지표를 CSV로 저장."""
    filepath = os.path.join(output_dir, "metrics.csv")
    with open(filepath, "w") as f:
        f.write("metric,value\n")
        f.write(f"ape_trans_rmse,{ape['trans_rmse']:.6f}\n")
        f.write(f"ape_trans_mean,{ape['trans_mean']:.6f}\n")
        f.write(f"ape_trans_median,{ape['trans_median']:.6f}\n")
        f.write(f"ape_trans_std,{ape['trans_std']:.6f}\n")
        f.write(f"ape_trans_max,{ape['trans_max']:.6f}\n")
        f.write(f"ape_rot_rmse,{ape['rot_rmse']:.6f}\n")
        f.write(f"ape_rot_mean,{ape['rot_mean']:.6f}\n")
        f.write(f"ape_rot_max,{ape['rot_max']:.6f}\n")

        for rpe in rpe_list:
            d = rpe["delta_m"]
            f.write(f"rpe_trans_rmse_d{d},{ rpe['trans_rmse']:.6f}\n")
            f.write(f"rpe_rot_rmse_d{d},{rpe['rot_rmse']:.6f}\n")

        f.write(f"drift_endpoint_error_m,{drift['endpoint_error_m']:.6f}\n")
        f.write(f"drift_yaw_error_deg,{drift['endpoint_yaw_error_deg']:.6f}\n")
        f.write(f"drift_travel_distance_m,{drift['total_travel_distance_m']:.6f}\n")
        f.write(f"drift_rate_pct,{drift['drift_rate_pct']:.6f}\n")

        if lm_result and not math.isnan(lm_result.get("mean_error", float("nan"))):
            f.write(f"landmark_mean_error,{lm_result['mean_error']:.6f}\n")
            f.write(f"landmark_max_error,{lm_result['max_error']:.6f}\n")
            for mid, err in sorted(lm_result["per_marker"].items()):
                f.write(f"landmark_{mid}_error,{err:.6f}\n")

    print(f"  CSV 저장: {filepath}")


# ═══════════════════════════════════════════════
# [7] CLI & Main
# ═══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ArUco-SAM SLAM 정량 평가 스크립트 (rosbag 후처리)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        사용 예:
            python3 evaluate_slam.py ./rosbag_001/
            python3 evaluate_slam.py ./rosbag_001/ --gt-topic /odom_gt --est-topic /ekf/odom
            python3 evaluate_slam.py ./rosbag_001/ --est-topic /ekf/odom /aruco_slam/odom
            python3 evaluate_slam.py ./rosbag_001/ --landmarks-map ./map/landmarks_map.json
            python3 evaluate_slam.py ./rosbag_001/ --export-tum-only
        """),
    )
    parser.add_argument("bag_path", type=str, help="rosbag2 디렉토리 경로")
    parser.add_argument("--gt-topic", type=str, default=None,
                        help="GT 오도메트리 토픽 (기본: 자동 감지)")
    parser.add_argument("--est-topic", type=str, nargs="+", default=None,
                        help="추정 오도메트리 토픽 (복수 가능, 기본: 자동 감지)")
    parser.add_argument("--alignment", type=str, default="umeyama",
                        choices=["origin", "umeyama", "none"],
                        help="궤적 정렬 방법 (기본: umeyama)")
    parser.add_argument("--rpe-deltas", type=float, nargs="+", default=[1.0, 5.0, 10.0],
                        help="RPE 거리 간격 (m) (기본: 1 5 10)")
    parser.add_argument("--landmarks-map", type=str, default=None,
                        help="SLAM 추정 랜드마크 맵 JSON 경로")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="결과 저장 디렉토리 (기본: bag_path/eval_results/)")
    parser.add_argument("--no-plot", action="store_true",
                        help="그래프 생성 건너뛰기")
    parser.add_argument("--export-tum-only", action="store_true",
                        help="TUM 파일만 내보내고 종료")
    parser.add_argument("--max-time-diff", type=float, default=0.05,
                        help="시간 동기화 최대 허용 차이 (초, 기본: 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()
    bag_path = args.bag_path

    if not os.path.exists(bag_path):
        print(f"[오류] rosbag 경로를 찾을 수 없습니다: {bag_path}")
        sys.exit(1)

    # 출력 디렉토리 설정
    output_dir = args.output_dir or os.path.join(bag_path, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # ──── 토픽 감지 ────
    print("\n[1/6] 토픽 스캔 중...")
    available = detect_available_topics(bag_path)
    if not available:
        print("[오류] rosbag에 nav_msgs/Odometry 토픽이 없습니다.")
        sys.exit(1)

    print("  사용 가능 토픽:")
    for topic, count in available.items():
        print(f"    {topic}: {count}개 메시지")

    # GT 토픽 결정
    gt_topic = args.gt_topic
    if gt_topic is None:
        gt_topic, auto_est = auto_select_topics(available)
    if gt_topic is None or gt_topic not in available:
        print(f"[오류] GT 토픽을 찾을 수 없습니다. --gt-topic으로 지정하세요.")
        print(f"  사용 가능: {list(available.keys())}")
        sys.exit(1)

    # 추정 토픽 결정
    est_topics = args.est_topic
    if est_topics is None:
        _, est_topics = auto_select_topics(available)
    est_topics = [t for t in est_topics if t in available and t != gt_topic]
    if not est_topics:
        print(f"[오류] 추정 토픽을 찾을 수 없습니다. --est-topic으로 지정하세요.")
        sys.exit(1)

    print(f"\n  GT 토픽:   {gt_topic}")
    print(f"  추정 토픽: {est_topics}")

    # ──── Rosbag 추출 ────
    print("\n[2/6] Odometry 메시지 추출 중...")
    gt_data = extract_odometry(bag_path, gt_topic)
    if len(gt_data) < 10:
        print(f"[오류] GT 데이터가 부족합니다 ({len(gt_data)}개).")
        sys.exit(1)

    est_data_dict = {}
    for topic in est_topics:
        data = extract_odometry(bag_path, topic)
        if len(data) >= 10:
            est_data_dict[topic] = data
        else:
            print(f"  [경고] {topic}: 데이터 부족 ({len(data)}개), 건너뜀")

    if not est_data_dict:
        print("[오류] 유효한 추정 데이터가 없습니다.")
        sys.exit(1)

    # ──── TUM 파일 내보내기 ────
    print("\n[3/6] TUM 파일 내보내기...")
    save_tum_file(gt_data, os.path.join(output_dir, "gt.tum"))
    for topic, data in est_data_dict.items():
        safe_name = topic.replace("/", "_").strip("_")
        save_tum_file(data, os.path.join(output_dir, f"{safe_name}.tum"))

    if args.export_tum_only:
        print("\nTUM 파일 내보내기 완료. evo CLI로 분석하세요:")
        print(f"  evo_ape tum {output_dir}/gt.tum {output_dir}/<est>.tum -p --plot_mode xz")
        return

    # ──── 궤적 변환 & 정렬 ────
    print("\n[4/6] 궤적 정렬 및 지표 산출 중...")
    traj_gt = to_evo_trajectory(gt_data)

    for est_topic, est_data in est_data_dict.items():
        print(f"\n  ── {est_topic} ──")
        traj_est = to_evo_trajectory(est_data)

        traj_gt_sync, traj_est_aligned, align_info = align_trajectories(
            traj_gt, traj_est, method=args.alignment, max_diff=args.max_time_diff)

        # APE 계산
        ape = compute_ape(traj_gt_sync, traj_est_aligned)

        # origin 정렬 후 APE가 큰 경우 경고
        if args.alignment == "origin" and ape["trans_rmse"] > 1.0:
            print(f"  [경고] APE RMSE = {ape['trans_rmse']:.3f}m (>1.0m)")
            print("    좌표계 불일치 가능. --alignment umeyama 사용을 권장합니다.")

        # RPE 계산
        rpe_list = []
        for delta in args.rpe_deltas:
            rpe = compute_rpe(traj_gt_sync, traj_est_aligned, delta)
            rpe_list.append(rpe)

        # Drift Rate 계산
        drift = compute_drift_rate(traj_gt_sync, traj_est_aligned)

        # 랜드마크 정확도 (옵션)
        lm_result = None
        if args.landmarks_map and os.path.exists(args.landmarks_map):
            print("  랜드마크 정확도 계산 중...")
            lm_result = compute_landmark_accuracy(args.landmarks_map)

        # ──── 콘솔 출력 ────
        print_summary(ape, rpe_list, drift, lm_result, align_info, gt_topic, est_topic)

        # ──── CSV 저장 ────
        print("[5/6] 결과 저장 중...")
        safe_name = est_topic.replace("/", "_").strip("_")
        est_output_dir = os.path.join(output_dir, safe_name)
        os.makedirs(est_output_dir, exist_ok=True)
        save_metrics_csv(ape, rpe_list, drift, lm_result, est_output_dir)

        # ──── 시각화 ────
        if not args.no_plot and HAS_MATPLOTLIB:
            print("[6/6] 그래프 생성 중...")
            traj_est_dict = {est_topic: traj_est_aligned}
            lm_for_plot = lm_result if lm_result else None

            plot_trajectories_2d(traj_gt_sync, traj_est_dict, est_output_dir,
                                landmarks=lm_for_plot)
            plot_ape_over_time(ape, est_output_dir)
            plot_ape_distribution(ape, est_output_dir)
            plot_rpe_by_delta(rpe_list, est_output_dir)
            plot_yaw_comparison(traj_gt_sync, traj_est_aligned, est_output_dir)

            if lm_result and not math.isnan(lm_result.get("mean_error", float("nan"))):
                plot_landmark_comparison(lm_result, est_output_dir)

            print(f"  그래프 저장: {est_output_dir}/")
        elif not HAS_MATPLOTLIB:
            print("[6/6] matplotlib 없음, 그래프 생성 건너뜀")

    print("\n평가 완료.")
    print(f"결과 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
