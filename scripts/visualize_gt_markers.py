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
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path
    typestore = get_typestore(Stores.ROS2_HUMBLE)
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


def extract_slam_landmarks(bag_path: str,
                           topic: str = "/aruco_slam/landmarks"):
    """Extract SLAM landmark positions from last message. Returns {id: (x,y)}."""
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

    landmarks = {}
    for m in last_msg.markers:
        if m.ns == "landmarks":
            landmarks[m.id] = (m.pose.position.x, m.pose.position.y)

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
    """Per-marker error vs GT marker face position."""
    per_marker = {}
    for mid, (sx, sy) in slam_lm_aligned.items():
        if mid in GAZEBO_BOX_CENTERS:
            bx, by = GAZEBO_BOX_CENTERS[mid]
            gt_mx, gt_my = compute_gt_marker_face(bx, by)
            per_marker[mid] = math.hypot(sx - gt_mx, sy - gt_my)

    if not per_marker:
        return {"per_marker": {}, "mean_error": float("nan"),
                "max_error": float("nan"), "max_error_id": -1}

    errors = list(per_marker.values())
    max_id = max(per_marker, key=per_marker.get)
    return {
        "per_marker": per_marker,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "max_error_id": max_id,
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
        for mid, err in sorted(lm["per_marker"].items()):
            print(f"      Marker {mid}: {err:.4f} m")
    print("=" * 62)
    print()


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
            sx, sy = slam_lm_aligned[mid]
            gt_mx, gt_my = compute_gt_marker_face(bx, by)
            ax.plot([gt_mx, sx], [gt_my, sy], "--", color="gray",
                    linewidth=0.8, alpha=0.6, zorder=3)
            ax.plot(sx, sy, "D", color="magenta", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=8)
            err = math.hypot(sx - gt_mx, sy - gt_my)
            ax.annotate(f"{err:.2f}m", (sx, sy),
                        textcoords="offset points", xytext=(6, -7),
                        fontsize=6, color="magenta", zorder=8)


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
    args = parser.parse_args()

    # ── Extract ──
    gt_ts = gt_x = gt_y = gt_yaw = None
    slam_ts = slam_x = slam_y = slam_yaw = None
    slam_aligned_x = slam_aligned_y = slam_aligned_yaw = None
    slam_lm_aligned = {}
    R = t = None
    align_angle = 0.0

    if args.bag:
        gt_ts, gt_x, gt_y, gt_yaw = extract_odom_with_ts(args.bag, "/odom_gt")
        slam_ts, slam_x, slam_y, slam_yaw = extract_odom_with_ts(
            args.bag, args.slam_topic)

        # ── Align SLAM → Gazebo ──
        if (gt_ts is not None and slam_ts is not None
                and len(gt_ts) > 0 and len(slam_ts) > 0):
            idx_slam, idx_gt = sync_by_timestamp(slam_ts, gt_ts, max_diff=0.1)
            print(f"  Synced {len(idx_slam)} pose pairs")

            if len(idx_slam) >= 3:
                src = np.column_stack([slam_x[idx_slam], slam_y[idx_slam]])
                dst = np.column_stack([gt_x[idx_gt], gt_y[idx_gt]])
                R, t = umeyama_2d(src, dst)
                align_angle = math.atan2(R[1, 0], R[0, 0])

                # Transform full SLAM trajectory
                slam_all = np.column_stack([slam_x, slam_y])
                slam_aligned = apply_transform_2d(R, t, slam_all)
                slam_aligned_x = slam_aligned[:, 0]
                slam_aligned_y = slam_aligned[:, 1]
                slam_aligned_yaw = slam_yaw + align_angle

                # SLAM landmarks
                slam_lm_raw = extract_slam_landmarks(args.bag)
                for mid, (lx, ly) in slam_lm_raw.items():
                    pt = apply_transform_2d(R, t, np.array([[lx, ly]]))[0]
                    slam_lm_aligned[mid] = (pt[0], pt[1])

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

        def _metrics_for_slice(name, s_mask, g_indices):
            """Compute APE, RPE, Drift for a subset of synced indices."""
            gx = gt_x[g_indices]
            gy = gt_y[g_indices]
            gyaw = gt_yaw[g_indices]
            sx = slam_aligned_x[s_mask]
            sy = slam_aligned_y[s_mask]
            syaw = slam_aligned_yaw[s_mask]

            ape = compute_ape(gx, gy, gyaw, sx, sy, syaw)
            rpe_list = [compute_rpe(gx, gy, gyaw, sx, sy, syaw, d)
                        for d in [1.0, 2.0, 5.0]]
            drift = compute_drift_rate(gx, gy, gyaw, sx, sy, syaw)
            lm = compute_landmark_accuracy(slam_lm_aligned)
            return {"ape": ape, "rpe": rpe_list, "drift": drift, "lm": lm,
                    "name": name}

        # Full
        metrics_full = _metrics_for_slice("Full", idx_s, idx_g)
        print_summary(metrics_full["ape"], metrics_full["rpe"],
                      metrics_full["drift"], metrics_full["lm"], args.slam_topic)

        # Loop 1: synced pairs where slam_ts < split_time
        mask1 = slam_ts[idx_s] < split_time
        if mask1.sum() > 10:
            metrics_loop1 = _metrics_for_slice("Loop 1", idx_s[mask1], idx_g[mask1])

        # Loop 2: synced pairs where slam_ts >= split_time
        mask2 = slam_ts[idx_s] >= split_time
        if mask2.sum() > 10:
            metrics_loop2 = _metrics_for_slice("Loop 2", idx_s[mask2], idx_g[mask2])

    # ── Plot: 2 subplots (Loop 1 / Loop 2) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))

    for ax, loop_idx, title_suffix, m in [
        (ax1, 0, "Loop 1", metrics_loop1),
        (ax2, 1, "Loop 2", metrics_loop2),
    ]:
        ax.set_aspect("equal")
        ax.set_title(title_suffix, fontsize=13)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)

        if gt_split is not None:
            gt_sl = slice(0, gt_split + 1) if loop_idx == 0 else slice(gt_split, None)
        else:
            gt_sl = slice(None)

        if slam_split is not None:
            slam_sl = slice(0, slam_split + 1) if loop_idx == 0 else slice(slam_split, None)
        else:
            slam_sl = slice(None)

        if gt_x is not None:
            gx, gy = gt_x[gt_sl], gt_y[gt_sl]
            ax.plot(gx, gy, "-", color="green", linewidth=1.5, alpha=0.8, zorder=2)
            ax.plot(gx[0], gy[0], "go", markersize=8, zorder=7)
            ax.plot(gx[-1], gy[-1], "g^", markersize=8, zorder=7)

        if slam_aligned_x is not None:
            sx, sy = slam_aligned_x[slam_sl], slam_aligned_y[slam_sl]
            ax.plot(sx, sy, "-", color="purple", linewidth=1.2, alpha=0.8, zorder=2)
            ax.plot(sx[0], sy[0], "mo", markersize=6, zorder=7)
            ax.plot(sx[-1], sy[-1], "m^", markersize=6, zorder=7)

        draw_boxes_and_markers(ax, slam_lm_aligned)
        set_common_limits(ax, gt_x, gt_y,
                          slam_aligned_x, slam_aligned_y)

        if m:
            add_metrics_text(ax, m["ape"], m["drift"], m["lm"])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="moccasin", edgecolor="darkorange",
                       label="GT Box (0.5\u00d70.5 m)"),
        plt.Line2D([0], [0], color="red", linewidth=3,
                   label="ArUco Marker Face"),
        plt.Line2D([0], [0], color="blue", linewidth=1.2, marker=">",
                   markersize=5, label="Marker Normal"),
        plt.Line2D([0], [0], color="green", linewidth=1.5,
                   label="GT Trajectory (/odom_gt)"),
    ]
    if slam_aligned_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="purple", linewidth=1.2,
                       label=f"SLAM Trajectory ({args.slam_topic})"))
    if slam_lm_aligned:
        legend_elements.extend([
            plt.Line2D([0], [0], color="magenta", marker="D",
                       linestyle="None", markersize=6,
                       markeredgecolor="black", markeredgewidth=0.5,
                       label="SLAM Landmark (aligned)"),
            plt.Line2D([0], [0], color="gray", linestyle="--",
                       linewidth=0.8, label="Landmark Error"),
        ])
    legend_elements.extend([
        plt.Line2D([0], [0], color="green", marker="o", linestyle="None",
                   markersize=6, label="Start"),
        plt.Line2D([0], [0], color="green", marker="^", linestyle="None",
                   markersize=6, label="End"),
    ])
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "gt_box_marker_layout.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
