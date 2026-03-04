#!/usr/bin/env python3
"""Visualize GT boxes, ArUco markers, GT/SLAM trajectories, SLAM landmarks,
and compute SLAM evaluation metrics (APE, RPE, Drift, Landmark accuracy).

Usage:
    python3 visualize_gt_markers.py                           # boxes only
    python3 visualize_gt_markers.py --bag /bags/mapping_xxx   # full evaluation
"""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Gazebo frame box center positions (x, y) ──
GAZEBO_BOX_CENTERS = {
    11: (0.0,   3.0),
    12: (1.9,   3.0),
    13: (1.9,  -1.5),
    14: (4.6,  -1.5),
    15: (4.6,  -3.75),
    16: (4.6,  -6.0),
    17: (1.45, -6.0),
    18: (-1.7, -6.0),
    19: (-1.9,  3.0),
    20: (-1.7, -1.5),
}

BOX_SIZE = 0.50
MARKER_FACE_OFFSET = 0.251
MARKER_SIZE = 0.30
BOX_CENTER_GZ = (1.495, -2.425)


def compute_box_yaw(bx, by):
    cx, cy = BOX_CENTER_GZ
    dx = cx - bx
    dy = cy - by
    return math.atan2(dy, dx) + math.pi


def compute_gt_marker_face(bx, by):
    yaw = compute_box_yaw(bx, by)
    return bx - MARKER_FACE_OFFSET * math.cos(yaw), \
           by - MARKER_FACE_OFFSET * math.sin(yaw)


def rotated_rect_corners(cx, cy, w, h, yaw):
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    hw, hh = w / 2, h / 2
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + cos_a * lx - sin_a * ly,
             cy + sin_a * lx + cos_a * ly) for lx, ly in local]


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


# ═══════════════════════════════════════
# Bag extraction
# ═══════════════════════════════════════

def _get_reader_and_typestore(bag_path):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    from pathlib import Path
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Register custom aruco_sam_ailab message types
    add_types = {}
    add_types.update(get_types_from_msg(
        'int32 id\ngeometry_msgs/Pose pose',
        'aruco_sam_ailab/msg/MarkerObservation'))
    add_types.update(get_types_from_msg(
        'std_msgs/Header header\n'
        'aruco_sam_ailab/msg/MarkerObservation[] markers',
        'aruco_sam_ailab/msg/MarkerArray'))
    typestore.register(add_types)

    reader = AnyReader([Path(bag_path)], default_typestore=typestore)
    return reader, typestore


def extract_odom_with_ts(bag_path: str, topic: str):
    """Extract odometry: (N,) ts, (N,) x, (N,) y, (N,) yaw."""
    reader, typestore = _get_reader_and_typestore(bag_path)
    ts_list, xs, ys, yaws = [], [], [], []

    with reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic '{topic}' not found")
            return None, None, None, None

        print(f"  Extracting '{topic}' ...")
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            q = msg.pose.pose.orientation
            ts_list.append(t)
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)
            yaws.append(quat_to_yaw(q.x, q.y, q.z, q.w))

    print(f"  Extracted {len(xs)} poses from '{topic}'")
    return np.array(ts_list), np.array(xs), np.array(ys), np.array(yaws)


def extract_aruco_detection_times(bag_path: str,
                                  topic: str = "/aruco_poses"):
    """Extract timestamps when ArUco markers were/weren't detected.

    Returns (det_ts, nodet_ts) — numpy arrays of timestamps.
    """
    reader, typestore = _get_reader_and_typestore(bag_path)
    det_ts, nodet_ts = [], []

    with reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic '{topic}' not found for detection times")
            return np.array([]), np.array([])

        print(f"  Extracting detection times from '{topic}' ...")
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if len(msg.markers) > 0:
                det_ts.append(t)
            else:
                nodet_ts.append(t)

    print(f"  Detection: {len(det_ts)} frames, No-detection: {len(nodet_ts)} frames")
    return np.array(det_ts), np.array(nodet_ts)


def extract_slam_landmarks(bag_path: str,
                           topic: str = "/aruco_slam/landmarks"):
    """Extract SLAM landmark positions and orientations from last message.

    Returns {id: (x, y, yaw)}.
    """
    reader, typestore = _get_reader_and_typestore(bag_path)
    last_msg = None

    with reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic '{topic}' not found")
            return {}
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            last_msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

    if last_msg is None:
        return {}

    # sphere(ns="landmarks")에서 위치, arrow(ns="landmark_normal")에서 방향 추출
    positions = {}  # id -> (x, y)
    normals = {}    # id -> yaw
    for m in last_msg.markers:
        if m.ns == "landmarks":
            positions[m.id] = (m.pose.position.x, m.pose.position.y)
        elif m.ns == "landmark_normal" and len(m.points) >= 2:
            dx = m.points[1].x - m.points[0].x
            dy = m.points[1].y - m.points[0].y
            normals[m.id] = math.atan2(dy, dx)

    landmarks = {}
    for mid, (px, py) in positions.items():
        yaw = normals.get(mid, 0.0)
        landmarks[mid] = (px, py, yaw)

    print(f"  Extracted {len(landmarks)} SLAM landmarks from '{topic}'")
    return landmarks


# ═══════════════════════════════════════
# Timestamp sync & Umeyama alignment
# ═══════════════════════════════════════

def sync_by_timestamp(ts_a, ts_b, max_diff=0.1):
    """Nearest-neighbor timestamp matching. Returns (idx_a, idx_b)."""
    idx_a, idx_b = [], []
    j = 0
    for i in range(len(ts_a)):
        while j < len(ts_b) - 1 and abs(ts_b[j + 1] - ts_a[i]) < abs(ts_b[j] - ts_a[i]):
            j += 1
        if abs(ts_b[j] - ts_a[i]) <= max_diff:
            idx_a.append(i)
            idx_b.append(j)
    return np.array(idx_a), np.array(idx_b)


def umeyama_2d(src, dst):
    """2D Umeyama (rigid, no scale). Returns R(2x2), t(2,)."""
    assert src.shape == dst.shape and src.shape[1] == 2
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    H = (src - mu_src).T @ (dst - mu_dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def apply_transform_2d(R, t, points):
    return (R @ points.T).T + t


def find_loop_split(x, y):
    n = len(x)
    dist = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    s, e = int(n * 0.3), int(n * 0.8)
    return s + np.argmin(dist[s:e])


def split_by_time(ts, split_time):
    return np.searchsorted(ts, split_time)


# ═══════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════

def compute_ape(gt_x, gt_y, gt_yaw, est_x, est_y, est_yaw):
    """APE (Absolute Pose Error) on synced pairs.

    Returns dict with translation and rotation stats.
    """
    trans_err = np.sqrt((gt_x - est_x)**2 + (gt_y - est_y)**2)
    rot_err = np.array([abs(normalize_angle(g - e))
                        for g, e in zip(gt_yaw, est_yaw)])
    rot_err_deg = np.degrees(rot_err)

    return {
        "trans_rmse": float(np.sqrt((trans_err**2).mean())),
        "trans_mean": float(trans_err.mean()),
        "trans_median": float(np.median(trans_err)),
        "trans_std": float(trans_err.std()),
        "trans_max": float(trans_err.max()),
        "rot_rmse": float(np.sqrt((rot_err_deg**2).mean())),
        "rot_mean": float(rot_err_deg.mean()),
        "rot_max": float(rot_err_deg.max()),
        "trans_errors": trans_err,
        "rot_errors": rot_err_deg,
    }


def compute_rpe(gt_x, gt_y, gt_yaw, est_x, est_y, est_yaw, delta_m):
    """RPE (Relative Pose Error) at given distance interval.

    Pairs are selected where GT travel distance ~ delta_m.
    """
    n = len(gt_x)
    gt_cumlen = np.zeros(n)
    for i in range(1, n):
        gt_cumlen[i] = gt_cumlen[i-1] + math.hypot(gt_x[i]-gt_x[i-1],
                                                     gt_y[i]-gt_y[i-1])

    trans_errs, rot_errs = [], []
    j = 0
    for i in range(n):
        while j < n and gt_cumlen[j] - gt_cumlen[i] < delta_m:
            j += 1
        if j >= n:
            break

        # GT relative
        dx_gt = gt_x[j] - gt_x[i]
        dy_gt = gt_y[j] - gt_y[i]
        dyaw_gt = normalize_angle(gt_yaw[j] - gt_yaw[i])

        # Est relative
        dx_est = est_x[j] - est_x[i]
        dy_est = est_y[j] - est_y[i]
        dyaw_est = normalize_angle(est_yaw[j] - est_yaw[i])

        # RPE
        t_err = math.hypot(dx_gt - dx_est, dy_gt - dy_est)
        r_err = abs(normalize_angle(dyaw_gt - dyaw_est))
        trans_errs.append(t_err)
        rot_errs.append(math.degrees(r_err))

    if not trans_errs:
        return {"delta_m": delta_m,
                "trans_rmse": float("nan"), "trans_mean": float("nan"),
                "trans_std": float("nan"),
                "rot_rmse": float("nan"), "rot_mean": float("nan")}

    te = np.array(trans_errs)
    re = np.array(rot_errs)
    return {
        "delta_m": delta_m,
        "trans_rmse": float(np.sqrt((te**2).mean())),
        "trans_mean": float(te.mean()),
        "trans_std": float(te.std()),
        "rot_rmse": float(np.sqrt((re**2).mean())),
        "rot_mean": float(re.mean()),
    }


def compute_drift_rate(gt_x, gt_y, gt_yaw, est_x, est_y, est_yaw):
    """Drift rate: endpoint error / total GT travel distance."""
    diffs = np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)
    travel = float(diffs.sum())
    endpoint_err = math.hypot(gt_x[-1] - est_x[-1], gt_y[-1] - est_y[-1])
    yaw_err = abs(normalize_angle(gt_yaw[-1] - est_yaw[-1]))
    drift_pct = (endpoint_err / travel * 100.0) if travel > 0 else float("nan")
    return {
        "endpoint_error_m": endpoint_err,
        "endpoint_yaw_error_deg": math.degrees(yaw_err),
        "total_travel_distance_m": travel,
        "drift_rate_pct": drift_pct,
    }


def compute_landmark_accuracy(slam_lm_aligned):
    """Per-marker position & yaw error vs GT marker face."""
    per_marker = {}
    per_marker_yaw = {}
    for mid, (sx, sy, syaw) in slam_lm_aligned.items():
        if mid in GAZEBO_BOX_CENTERS:
            bx, by = GAZEBO_BOX_CENTERS[mid]
            gt_mx, gt_my = compute_gt_marker_face(bx, by)
            per_marker[mid] = math.hypot(sx - gt_mx, sy - gt_my)
            # GT normal: 박스 중심을 향하는 방향
            gt_normal = math.atan2(BOX_CENTER_GZ[1] - by,
                                   BOX_CENTER_GZ[0] - bx)
            yaw_err = abs(normalize_angle(syaw - gt_normal))
            per_marker_yaw[mid] = math.degrees(yaw_err)

    if not per_marker:
        return {"per_marker": {}, "per_marker_yaw": {},
                "mean_error": float("nan"), "max_error": float("nan"),
                "mean_yaw_error": float("nan"), "max_error_id": -1}

    errors = list(per_marker.values())
    yaw_errors = list(per_marker_yaw.values())
    max_id = max(per_marker, key=per_marker.get)
    return {
        "per_marker": per_marker,
        "per_marker_yaw": per_marker_yaw,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "max_error_id": max_id,
        "mean_yaw_error": float(np.mean(yaw_errors)),
        "max_yaw_error": float(np.max(yaw_errors)),
    }


def print_summary(ape, rpe_list, drift, lm, slam_topic):
    """Print evaluation summary table."""
    print()
    print("=" * 62)
    print("       SLAM Evaluation Results (Gazebo Frame)")
    print("=" * 62)
    print(f"  GT topic:   /odom_gt")
    print(f"  SLAM topic: {slam_topic}")
    print("-" * 62)
    print("  [APE - Translation]")
    print(f"    RMSE:   {ape['trans_rmse']:.4f} m")
    print(f"    Mean:   {ape['trans_mean']:.4f} m")
    print(f"    Median: {ape['trans_median']:.4f} m")
    print(f"    Std:    {ape['trans_std']:.4f} m")
    print(f"    Max:    {ape['trans_max']:.4f} m")
    print("  [APE - Rotation]")
    print(f"    RMSE:   {ape['rot_rmse']:.2f}\u00b0")
    print(f"    Mean:   {ape['rot_mean']:.2f}\u00b0")
    print(f"    Max:    {ape['rot_max']:.2f}\u00b0")
    print("-" * 62)
    print("  [RPE]")
    for rpe in rpe_list:
        print(f"    d={rpe['delta_m']}m  Trans RMSE: {rpe['trans_rmse']:.4f} m"
              f"  Rot RMSE: {rpe['rot_rmse']:.2f}\u00b0")
    print("-" * 62)
    print("  [Drift Rate]")
    print(f"    Endpoint Error:     {drift['endpoint_error_m']:.4f} m")
    print(f"    Endpoint Yaw Error: {drift['endpoint_yaw_error_deg']:.2f}\u00b0")
    print(f"    Travel Distance:    {drift['total_travel_distance_m']:.2f} m")
    print(f"    Drift Rate:         {drift['drift_rate_pct']:.3f} %")
    if lm and not math.isnan(lm.get("mean_error", float("nan"))):
        print("-" * 62)
        print("  [Landmark Accuracy]")
        print(f"    Mean Error: {lm['mean_error']:.4f} m")
        print(f"    Max Error:  {lm['max_error']:.4f} m (ID {lm['max_error_id']})")
        if not math.isnan(lm.get("mean_yaw_error", float("nan"))):
            print(f"    Mean Yaw Error: {lm['mean_yaw_error']:.2f}\u00b0")
            print(f"    Max Yaw Error:  {lm['max_yaw_error']:.2f}\u00b0")
        per_yaw = lm.get("per_marker_yaw", {})
        for mid, err in sorted(lm["per_marker"].items()):
            yaw_str = f"  yaw: {per_yaw[mid]:.1f}\u00b0" if mid in per_yaw else ""
            print(f"      Marker {mid}: {err:.4f} m{yaw_str}")
    print("=" * 62)
    print()


def save_metrics_csv(filepath, metrics_full, metrics_loop1, metrics_loop2):
    """Save all metrics to CSV."""
    import csv
    rows = []

    def _add(prefix, m):
        if m is None:
            return
        a = m["ape"]
        rows.append((f"{prefix}/ape_trans_rmse", f"{a['trans_rmse']:.6f}"))
        rows.append((f"{prefix}/ape_trans_mean", f"{a['trans_mean']:.6f}"))
        rows.append((f"{prefix}/ape_trans_median", f"{a['trans_median']:.6f}"))
        rows.append((f"{prefix}/ape_trans_std", f"{a['trans_std']:.6f}"))
        rows.append((f"{prefix}/ape_trans_max", f"{a['trans_max']:.6f}"))
        rows.append((f"{prefix}/ape_rot_rmse", f"{a['rot_rmse']:.6f}"))
        rows.append((f"{prefix}/ape_rot_mean", f"{a['rot_mean']:.6f}"))
        rows.append((f"{prefix}/ape_rot_max", f"{a['rot_max']:.6f}"))
        for rpe in m["rpe"]:
            d = rpe["delta_m"]
            rows.append((f"{prefix}/rpe_d{d}_trans_rmse", f"{rpe['trans_rmse']:.6f}"))
            rows.append((f"{prefix}/rpe_d{d}_rot_rmse", f"{rpe['rot_rmse']:.6f}"))
        dr = m["drift"]
        rows.append((f"{prefix}/drift_endpoint_m", f"{dr['endpoint_error_m']:.6f}"))
        rows.append((f"{prefix}/drift_yaw_deg", f"{dr['endpoint_yaw_error_deg']:.6f}"))
        rows.append((f"{prefix}/drift_travel_m", f"{dr['total_travel_distance_m']:.6f}"))
        rows.append((f"{prefix}/drift_rate_pct", f"{dr['drift_rate_pct']:.6f}"))
        lm = m["lm"]
        if lm and not math.isnan(lm.get("mean_error", float("nan"))):
            rows.append((f"{prefix}/landmark_mean_error", f"{lm['mean_error']:.6f}"))
            rows.append((f"{prefix}/landmark_max_error", f"{lm['max_error']:.6f}"))
            if not math.isnan(lm.get("mean_yaw_error", float("nan"))):
                rows.append((f"{prefix}/landmark_mean_yaw_error_deg",
                             f"{lm['mean_yaw_error']:.6f}"))
                rows.append((f"{prefix}/landmark_max_yaw_error_deg",
                             f"{lm['max_yaw_error']:.6f}"))
            per_yaw = lm.get("per_marker_yaw", {})
            for mid, err in sorted(lm["per_marker"].items()):
                rows.append((f"{prefix}/landmark_{mid}_error", f"{err:.6f}"))
                if mid in per_yaw:
                    rows.append((f"{prefix}/landmark_{mid}_yaw_error_deg",
                                 f"{per_yaw[mid]:.6f}"))

    _add("full", metrics_full)
    _add("loop1", metrics_loop1)
    _add("loop2", metrics_loop2)

    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    print(f"  CSV saved: {filepath}")


# ═══════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════

def draw_boxes_and_markers(ax, slam_lm_aligned=None):
    for mid, (bx, by) in GAZEBO_BOX_CENTERS.items():
        yaw = compute_box_yaw(bx, by)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        corners = rotated_rect_corners(bx, by, BOX_SIZE, BOX_SIZE, yaw)
        poly = plt.Polygon(corners, closed=True,
                           facecolor="moccasin", edgecolor="darkorange",
                           linewidth=1.5, zorder=3)
        ax.add_patch(poly)
        ax.plot(bx, by, "o", color="darkorange", markersize=2, zorder=5)

        ax.annotate(f"{mid}", (bx, by),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, fontweight="bold", ha="center", color="darkblue",
                    zorder=6)

        mx = bx - MARKER_FACE_OFFSET * cos_y
        my = by - MARKER_FACE_OFFSET * sin_y
        half_m = MARKER_SIZE / 2
        pc, ps = math.cos(yaw + math.pi/2), math.sin(yaw + math.pi/2)
        ax.plot([mx + half_m*pc, mx - half_m*pc],
                [my + half_m*ps, my - half_m*ps],
                color="red", linewidth=3, solid_capstyle="butt", zorder=4)
        ax.plot(mx, my, "s", color="red", markersize=3, zorder=5)

        ny = math.atan2(BOX_CENTER_GZ[1] - by, BOX_CENTER_GZ[0] - bx)
        ax.annotate("", xy=(mx + 0.3*math.cos(ny), my + 0.3*math.sin(ny)),
                    xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->,head_width=0.06,head_length=0.05",
                                   color="blue", lw=1.2), zorder=5)

        if slam_lm_aligned and mid in slam_lm_aligned:
            sx, sy, syaw = slam_lm_aligned[mid]
            gt_mx, gt_my = compute_gt_marker_face(bx, by)
            ax.plot([gt_mx, sx], [gt_my, sy], "--", color="gray",
                    linewidth=0.8, alpha=0.6, zorder=3)
            ax.plot(sx, sy, "D", color="magenta", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=8)
            err = math.hypot(sx - gt_mx, sy - gt_my)
            gt_normal = math.atan2(BOX_CENTER_GZ[1] - by,
                                   BOX_CENTER_GZ[0] - bx)
            yaw_err = abs(normalize_angle(syaw - gt_normal))
            ax.annotate(f"{err:.2f}m / {math.degrees(yaw_err):.0f}\u00b0",
                        (sx, sy),
                        textcoords="offset points", xytext=(6, -7),
                        fontsize=6, color="magenta", zorder=8)
            # SLAM landmark normal direction
            ax.annotate("", xy=(sx + 0.3*math.cos(syaw), sy + 0.3*math.sin(syaw)),
                        xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->,head_width=0.06,head_length=0.05",
                                        color="magenta", lw=1.2), zorder=9)


def set_common_limits(ax, gt_x, gt_y, slam_x=None, slam_y=None):
    all_x = [v[0] for v in GAZEBO_BOX_CENTERS.values()]
    all_y = [v[1] for v in GAZEBO_BOX_CENTERS.values()]
    if gt_x is not None and len(gt_x) > 0:
        all_x.extend([gt_x.min(), gt_x.max()])
        all_y.extend([gt_y.min(), gt_y.max()])
    if slam_x is not None and len(slam_x) > 0:
        all_x.extend([slam_x.min(), slam_x.max()])
        all_y.extend([slam_y.min(), slam_y.max()])
    pad = 1.2
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)


def compute_blind_mask(gt_ts, det_ts, threshold=0.5):
    """True where nearest ArUco detection is > threshold seconds away."""
    if len(det_ts) == 0:
        return np.ones(len(gt_ts), dtype=bool)
    idx = np.searchsorted(det_ts, gt_ts)
    idx = np.clip(idx, 0, len(det_ts) - 1)
    dist = np.abs(det_ts[idx] - gt_ts)
    idx_prev = np.clip(idx - 1, 0, len(det_ts) - 1)
    dist_prev = np.abs(det_ts[idx_prev] - gt_ts)
    nearest_dist = np.minimum(dist, dist_prev)
    return nearest_dist > threshold


def draw_blind_zones(ax, x, y, blind_mask):
    """Overlay red segments on trajectory where no marker was detected."""
    if not np.any(blind_mask):
        return
    changes = np.diff(blind_mask.astype(int))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0] + 1)
    if blind_mask[0]:
        starts.insert(0, 0)
    if blind_mask[-1]:
        ends.append(len(blind_mask))
    for s, e in zip(starts, ends):
        # Extend by 1 on each side for visual continuity
        s2 = max(0, s - 1)
        e2 = min(len(x), e + 1)
        ax.plot(x[s2:e2], y[s2:e2], "-", color="red", linewidth=3.5,
                alpha=0.5, zorder=3, solid_capstyle="round")


def add_metrics_text(ax, ape, drift, lm, title=""):
    """Add metrics summary as text box on subplot."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"APE  RMSE: {ape['trans_rmse']:.3f} m / {ape['rot_rmse']:.1f}\u00b0")
    lines.append(f"APE  Mean: {ape['trans_mean']:.3f} m  Max: {ape['trans_max']:.3f} m")
    lines.append(f"Drift: {drift['drift_rate_pct']:.2f}% "
                 f"({drift['endpoint_error_m']:.3f}m / "
                 f"{drift['total_travel_distance_m']:.1f}m)")
    if lm and not math.isnan(lm.get("mean_error", float("nan"))):
        lines.append(f"Landmark: mean={lm['mean_error']:.3f}m  "
                     f"max={lm['max_error']:.3f}m")
        if not math.isnan(lm.get("mean_yaw_error", float("nan"))):
            lines.append(f"LM Yaw:   mean={lm['mean_yaw_error']:.1f}\u00b0  "
                         f"max={lm['max_yaw_error']:.1f}\u00b0")
    text = "\n".join(lines)
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            zorder=20)


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, default=None)
    parser.add_argument("--slam-topic", type=str, default="/aruco_slam/odom")
    parser.add_argument("--ekf-topic", type=str, default="/ekf/odom")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (plot + CSV)")
    args = parser.parse_args()

    # ── Extract ──
    gt_ts = gt_x = gt_y = gt_yaw = None
    slam_ts = slam_x = slam_y = slam_yaw = None
    slam_aligned_x = slam_aligned_y = slam_aligned_yaw = None
    ekf_ts = ekf_x = ekf_y = ekf_yaw = None
    ekf_aligned_x = ekf_aligned_y = ekf_aligned_yaw = None
    slam_lm_aligned = {}
    det_ts = np.array([])
    gt_blind_mask = None

    if args.bag:
        gt_ts, gt_x, gt_y, gt_yaw = extract_odom_with_ts(args.bag, "/odom_gt")
        slam_ts, slam_x, slam_y, slam_yaw = extract_odom_with_ts(
            args.bag, args.slam_topic)
        ekf_ts, ekf_x, ekf_y, ekf_yaw = extract_odom_with_ts(
            args.bag, args.ekf_topic)
        det_ts, _ = extract_aruco_detection_times(args.bag)
        if gt_ts is not None and len(det_ts) > 0:
            gt_blind_mask = compute_blind_mask(gt_ts, det_ts, threshold=0.5)

        def _align_to_gt(est_ts, est_x, est_y, est_yaw, label):
            """Umeyama alignment of estimated trajectory to GT.

            Returns (aligned_x, aligned_y, aligned_yaw, R, t, angle).
            """
            if (gt_ts is None or est_ts is None
                    or len(gt_ts) == 0 or len(est_ts) == 0):
                return None, None, None, None, None, 0.0
            idx_est, idx_gt = sync_by_timestamp(est_ts, gt_ts, max_diff=0.1)
            print(f"  [{label}] Synced {len(idx_est)} pose pairs")
            if len(idx_est) < 3:
                return None, None, None, None, None, 0.0

            # 시작 시간/위치 진단
            dt_start = est_ts[0] - gt_ts[0]
            print(f"  [{label}] Start time gap: {dt_start:.2f}s "
                  f"(GT t0={gt_ts[0]:.2f}, {label} t0={est_ts[0]:.2f})")
            print(f"  [{label}] Raw start: ({est_x[0]:.3f}, {est_y[0]:.3f})  "
                  f"GT start: ({gt_x[0]:.3f}, {gt_y[0]:.3f})")

            src = np.column_stack([est_x[idx_est], est_y[idx_est]])
            dst = np.column_stack([gt_x[idx_gt], gt_y[idx_gt]])
            R, t = umeyama_2d(src, dst)
            angle = math.atan2(R[1, 0], R[0, 0])
            aligned = apply_transform_2d(R, t,
                                         np.column_stack([est_x, est_y]))

            # 정렬 후 시작점 비교
            # GT에서 est 시작 시각과 가장 가까운 점 찾기
            gt_idx_at_est_start = np.argmin(np.abs(gt_ts - est_ts[0]))
            print(f"  [{label}] Aligned start: ({aligned[0, 0]:.3f}, {aligned[0, 1]:.3f})  "
                  f"GT@same_time: ({gt_x[gt_idx_at_est_start]:.3f}, "
                  f"{gt_y[gt_idx_at_est_start]:.3f})  "
                  f"offset: {math.hypot(aligned[0, 0] - gt_x[gt_idx_at_est_start], aligned[0, 1] - gt_y[gt_idx_at_est_start]):.3f}m")

            return aligned[:, 0], aligned[:, 1], est_yaw + angle, R, t, angle

        # ── Align SLAM → Gazebo ──
        (slam_aligned_x, slam_aligned_y, slam_aligned_yaw,
         R_slam, t_slam, align_angle_slam) = _align_to_gt(
            slam_ts, slam_x, slam_y, slam_yaw, "SLAM")

        # SLAM landmarks
        if R_slam is not None:
            slam_lm_raw = extract_slam_landmarks(args.bag)
            for mid, (lx, ly, lyaw) in slam_lm_raw.items():
                pt = apply_transform_2d(R_slam, t_slam, np.array([[lx, ly]]))[0]
                slam_lm_aligned[mid] = (pt[0], pt[1], lyaw + align_angle_slam)

        # ── Align EKF → Gazebo ──
        (ekf_aligned_x, ekf_aligned_y, ekf_aligned_yaw,
         _, _, _) = _align_to_gt(
            ekf_ts, ekf_x, ekf_y, ekf_yaw, "EKF")

    has_gt = gt_ts is not None and len(gt_ts) > 0

    # ── Compute metrics (full + per-loop) ──
    gt_split = slam_split = None
    metrics_full = metrics_loop1 = metrics_loop2 = None

    if (gt_x is not None and slam_aligned_x is not None
            and len(gt_x) > 100 and len(slam_aligned_x) > 10):

        gt_split = find_loop_split(gt_x, gt_y)
        split_time = gt_ts[gt_split]
        slam_split = split_by_time(slam_ts, split_time)
        print(f"  Loop split: GT idx={gt_split} ({split_time - gt_ts[0]:.1f}s), "
              f"SLAM idx={slam_split}")

        # Re-sync on aligned data for metrics
        idx_s, idx_g = sync_by_timestamp(slam_ts, gt_ts, max_diff=0.1)

        def _metrics_for_slice(name, s_mask, g_indices,
                               est_ax, est_ay, est_ayaw):
            """Compute APE, RPE, Drift for a subset of synced indices."""
            gx = gt_x[g_indices]
            gy = gt_y[g_indices]
            gyaw = gt_yaw[g_indices]
            sx = est_ax[s_mask]
            sy = est_ay[s_mask]
            syaw = est_ayaw[s_mask]

            ape = compute_ape(gx, gy, gyaw, sx, sy, syaw)
            rpe_list = [compute_rpe(gx, gy, gyaw, sx, sy, syaw, d)
                        for d in [1.0, 2.0, 5.0]]
            drift = compute_drift_rate(gx, gy, gyaw, sx, sy, syaw)
            lm = compute_landmark_accuracy(slam_lm_aligned)
            return {"ape": ape, "rpe": rpe_list, "drift": drift, "lm": lm,
                    "name": name}

        # Full (SLAM)
        metrics_full = _metrics_for_slice(
            "Full", idx_s, idx_g,
            slam_aligned_x, slam_aligned_y, slam_aligned_yaw)
        print_summary(metrics_full["ape"], metrics_full["rpe"],
                      metrics_full["drift"], metrics_full["lm"], args.slam_topic)

        # Loop 1: synced pairs where slam_ts < split_time
        mask1 = slam_ts[idx_s] < split_time
        if mask1.sum() > 10:
            metrics_loop1 = _metrics_for_slice(
                "Loop 1", idx_s[mask1], idx_g[mask1],
                slam_aligned_x, slam_aligned_y, slam_aligned_yaw)

        # Loop 2: synced pairs where slam_ts >= split_time
        mask2 = slam_ts[idx_s] >= split_time
        if mask2.sum() > 10:
            metrics_loop2 = _metrics_for_slice(
                "Loop 2", idx_s[mask2], idx_g[mask2],
                slam_aligned_x, slam_aligned_y, slam_aligned_yaw)

    # ── Helper: draw a single trajectory plot ──
    def _make_trajectory_plot(est_aligned_x, est_aligned_y, est_ts_arr,
                              est_label, est_color, show_landmarks):
        fig, ax = plt.subplots(1, 1, figsize=(12, 11))
        ax.set_aspect("equal")
        ax.set_title(f"GT vs {est_label}", fontsize=13)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # 1m 간격 정사각형 그리드
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.grid(True, alpha=0.3)

        # 공통 시간 구간으로 트림
        gt_s = slice(None)
        est_s = slice(None)
        if (gt_ts is not None and est_ts_arr is not None
                and len(gt_ts) > 0 and len(est_ts_arr) > 0):
            t_start = max(gt_ts[0], est_ts_arr[0])
            t_end = min(gt_ts[-1], est_ts_arr[-1])
            gt_s = slice(np.searchsorted(gt_ts, t_start),
                         np.searchsorted(gt_ts, t_end, side='right'))
            est_s = slice(np.searchsorted(est_ts_arr, t_start),
                          np.searchsorted(est_ts_arr, t_end, side='right'))

        if gt_x is not None:
            gx, gy = gt_x[gt_s], gt_y[gt_s]
            ax.plot(gx, gy, "-", color="green", linewidth=1.5, alpha=0.8,
                    zorder=2, label="GT (/odom_gt)")
            ax.plot(gx[0], gy[0], "o", color="green", markersize=8,
                    zorder=7)
            ax.plot(gx[-1], gy[-1], "^", color="green", markersize=8,
                    zorder=7)
            if gt_blind_mask is not None:
                draw_blind_zones(ax, gx, gy, gt_blind_mask[gt_s])

        if est_aligned_x is not None:
            ex, ey = est_aligned_x[est_s], est_aligned_y[est_s]
            ax.plot(ex, ey, "-", color=est_color,
                    linewidth=1.2, alpha=0.8, zorder=2, label=est_label)
            ax.plot(ex[0], ey[0], "o",
                    color=est_color, markersize=6, zorder=7)
            ax.plot(ex[-1], ey[-1], "^",
                    color=est_color, markersize=6, zorder=7)

        lm = slam_lm_aligned if show_landmarks else None
        draw_boxes_and_markers(ax, lm)
        set_common_limits(ax, gt_x, gt_y, est_aligned_x, est_aligned_y)

        # Legend
        handles, _ = ax.get_legend_handles_labels()
        handles.extend([
            mpatches.Patch(facecolor="moccasin", edgecolor="darkorange",
                           label="GT Box"),
            plt.Line2D([0], [0], color="red", linewidth=3,
                       label="GT Marker Face"),
            plt.Line2D([0], [0], color="blue", linewidth=1.2, marker=">",
                       markersize=5, label="GT Normal"),
        ])
        if show_landmarks and slam_lm_aligned:
            handles.extend([
                plt.Line2D([0], [0], color="magenta", marker="D",
                           linestyle="None", markersize=6,
                           markeredgecolor="black", markeredgewidth=0.5,
                           label="SLAM Landmark"),
                plt.Line2D([0], [0], color="magenta", linewidth=1.2,
                           marker=">", markersize=5, label="SLAM Normal"),
                plt.Line2D([0], [0], color="gray", linestyle="--",
                           linewidth=0.8, label="Landmark Error"),
            ])
        if gt_blind_mask is not None and np.any(gt_blind_mask):
            handles.append(
                plt.Line2D([0], [0], color="red", linewidth=3.5, alpha=0.5,
                           label="Blind Zone"))
        handles.extend([
            plt.Line2D([0], [0], color="gray", marker="o", linestyle="None",
                       markersize=6, label="Start"),
            plt.Line2D([0], [0], color="gray", marker="^", linestyle="None",
                       markersize=6, label="End"),
        ])
        ax.legend(handles=handles, loc="upper right", fontsize=8,
                  framealpha=0.9)
        plt.tight_layout()
        return fig

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    # SLAM plot
    fig_slam = _make_trajectory_plot(
        slam_aligned_x, slam_aligned_y, slam_ts,
        f"SLAM ({args.slam_topic})", "purple", show_landmarks=True)
    slam_path = os.path.join(out_dir, "trajectory_slam.png")
    fig_slam.savefig(slam_path, dpi=150)
    plt.close(fig_slam)
    print(f"  Plot saved: {slam_path}")

    # EKF plot
    fig_ekf = _make_trajectory_plot(
        ekf_aligned_x, ekf_aligned_y, ekf_ts,
        f"EKF ({args.ekf_topic})", "darkorange", show_landmarks=False)
    ekf_path = os.path.join(out_dir, "trajectory_ekf.png")
    fig_ekf.savefig(ekf_path, dpi=150)
    plt.close(fig_ekf)
    print(f"  Plot saved: {ekf_path}")

    # Save CSV
    if metrics_full and args.output_dir:
        csv_path = os.path.join(out_dir, "metrics.csv")
        save_metrics_csv(csv_path, metrics_full, metrics_loop1, metrics_loop2)


if __name__ == "__main__":
    main()
