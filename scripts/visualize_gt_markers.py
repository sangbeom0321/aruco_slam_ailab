#!/usr/bin/env python3
"""Visualize GT boxes, ArUco markers, GT/SLAM trajectories, and SLAM landmarks.

SLAM trajectory & landmarks are aligned to Gazebo frame via 2D Umeyama on
all time-synchronized poses.

Usage:
    python3 visualize_gt_markers.py                           # boxes only
    python3 visualize_gt_markers.py --bag /bags/mapping_xxx   # boxes + GT + SLAM
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
    """Return marker face center (x, y) in Gazebo frame."""
    yaw = compute_box_yaw(bx, by)
    mx = bx - MARKER_FACE_OFFSET * math.cos(yaw)
    my = by - MARKER_FACE_OFFSET * math.sin(yaw)
    return mx, my


def rotated_rect_corners(cx, cy, w, h, yaw):
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    hw, hh = w / 2, h / 2
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + cos_a * lx - sin_a * ly,
             cy + sin_a * lx + cos_a * ly) for lx, ly in local]


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
    """Extract odometry as (N,) timestamps, (N,) x, (N,) y arrays."""
    reader, typestore = _get_reader_and_typestore(bag_path)
    ts_list, xs, ys = [], [], []

    with reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  Topic '{topic}' not found")
            return None, None, None

        print(f"  Extracting '{topic}' ...")
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts_list.append(t)
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)

    print(f"  Extracted {len(xs)} poses from '{topic}'")
    return np.array(ts_list), np.array(xs), np.array(ys)


def extract_slam_landmarks(bag_path: str,
                           topic: str = "/aruco_slam/landmarks"):
    """Extract SLAM landmark positions from last message.

    Returns: dict {marker_id: (x, y)}
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
    """Match timestamps from A to B (nearest neighbor).

    Returns: (idx_a, idx_b) index arrays of matched pairs.
    """
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
    """2D Umeyama alignment (rigid, no scale).

    Args:
        src: (N, 2) source points (SLAM)
        dst: (N, 2) target points (GT/Gazebo)

    Returns:
        R (2x2), t (2,) such that dst ~ R @ src + t
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    # cross-covariance
    H = src_c.T @ dst_c  # (2, 2)
    U, _, Vt = np.linalg.svd(H)

    # ensure proper rotation (det > 0)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])

    R = Vt.T @ S @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def apply_transform_2d(R, t, points):
    """Apply 2D rigid transform to (N,2) points."""
    return (R @ points.T).T + t


# ═══════════════════════════════════════
# Plot
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualize GT boxes & ArUco markers")
    parser.add_argument("--bag", type=str, default=None,
                        help="Path to rosbag for trajectory & SLAM overlay")
    parser.add_argument("--slam-topic", type=str, default="/aruco_slam/odom",
                        help="SLAM odometry topic (default: /aruco_slam/odom)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect("equal")
    ax.set_title("GT Boxes, Markers & SLAM Results (Gazebo Frame)", fontsize=13)
    ax.set_xlabel("Gazebo X (m)")
    ax.set_ylabel("Gazebo Y (m)")
    ax.grid(True, alpha=0.3)

    gt_x = gt_y = None
    slam_aligned_x = slam_aligned_y = None
    slam_lm_aligned = {}

    if args.bag:
        # ── Extract GT & SLAM odometry ──
        gt_ts, gt_x, gt_y = extract_odom_with_ts(args.bag, "/odom_gt")
        slam_ts, slam_x, slam_y = extract_odom_with_ts(args.bag, args.slam_topic)

        # ── GT trajectory ──
        if gt_x is not None and len(gt_x) > 0:
            ax.plot(gt_x, gt_y, "-", color="green", linewidth=1.2,
                    alpha=0.7, zorder=2)
            ax.plot(gt_x[0], gt_y[0], "go", markersize=8, zorder=7)
            ax.plot(gt_x[-1], gt_y[-1], "g^", markersize=8, zorder=7)

        # ── Align SLAM → Gazebo ──
        if (gt_ts is not None and slam_ts is not None
                and len(gt_ts) > 0 and len(slam_ts) > 0):
            idx_gt, idx_slam = sync_by_timestamp(slam_ts, gt_ts, max_diff=0.1)
            print(f"  Synced {len(idx_gt)} pose pairs "
                  f"(SLAM={len(slam_ts)}, GT={len(gt_ts)})")

            if len(idx_gt) >= 3:
                src = np.column_stack([slam_x[idx_gt], slam_y[idx_gt]])
                dst = np.column_stack([gt_x[idx_slam], gt_y[idx_slam]])
                R, t = umeyama_2d(src, dst)

                # Residual stats
                aligned_sync = apply_transform_2d(R, t, src)
                residuals = np.linalg.norm(aligned_sync - dst, axis=1)
                print(f"  Alignment residual: "
                      f"mean={residuals.mean():.4f}m, "
                      f"max={residuals.max():.4f}m, "
                      f"RMSE={np.sqrt((residuals**2).mean()):.4f}m")

                # Transform full SLAM trajectory
                slam_all = np.column_stack([slam_x, slam_y])
                slam_aligned = apply_transform_2d(R, t, slam_all)
                slam_aligned_x = slam_aligned[:, 0]
                slam_aligned_y = slam_aligned[:, 1]

                ax.plot(slam_aligned_x, slam_aligned_y, "-",
                        color="purple", linewidth=1.0, alpha=0.7, zorder=2)
                ax.plot(slam_aligned_x[0], slam_aligned_y[0],
                        "mo", markersize=6, zorder=7)
                ax.plot(slam_aligned_x[-1], slam_aligned_y[-1],
                        "m^", markersize=6, zorder=7)

                # ── SLAM landmarks (aligned) ──
                slam_lm_raw = extract_slam_landmarks(args.bag)
                for mid, (lx, ly) in slam_lm_raw.items():
                    pt = apply_transform_2d(R, t, np.array([[lx, ly]]))[0]
                    slam_lm_aligned[mid] = (pt[0], pt[1])

    # ── Draw GT boxes & markers ──
    ax.plot(*BOX_CENTER_GZ, "kx", markersize=10, markeredgewidth=2, zorder=10)
    ax.annotate("center (1.495, -2.425)", BOX_CENTER_GZ,
                textcoords="offset points", xytext=(10, 10), fontsize=8, color="gray")

    for mid, (bx, by) in GAZEBO_BOX_CENTERS.items():
        yaw = compute_box_yaw(bx, by)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # Box
        corners = rotated_rect_corners(bx, by, BOX_SIZE, BOX_SIZE, yaw)
        poly = plt.Polygon(corners, closed=True,
                           facecolor="moccasin", edgecolor="darkorange",
                           linewidth=1.5, zorder=3)
        ax.add_patch(poly)
        ax.plot(bx, by, "o", color="darkorange", markersize=3, zorder=5)

        # ID label
        ax.annotate(f"ID {mid}", (bx, by),
                    textcoords="offset points", xytext=(0, 14),
                    fontsize=9, fontweight="bold", ha="center", color="darkblue",
                    zorder=6)

        # Marker face
        mx = bx - MARKER_FACE_OFFSET * cos_y
        my = by - MARKER_FACE_OFFSET * sin_y

        half_m = MARKER_SIZE / 2
        perp_cos = math.cos(yaw + math.pi / 2)
        perp_sin = math.sin(yaw + math.pi / 2)
        m_x1 = mx + half_m * perp_cos
        m_y1 = my + half_m * perp_sin
        m_x2 = mx - half_m * perp_cos
        m_y2 = my - half_m * perp_sin

        ax.plot([m_x1, m_x2], [m_y1, m_y2],
                color="red", linewidth=3, solid_capstyle="butt", zorder=4)
        ax.plot(mx, my, "s", color="red", markersize=4, zorder=5)

        # Marker normal arrow
        normal_yaw = math.atan2(BOX_CENTER_GZ[1] - by, BOX_CENTER_GZ[0] - bx)
        arrow_len = 0.35
        ax.annotate("",
                    xy=(mx + arrow_len * math.cos(normal_yaw),
                        my + arrow_len * math.sin(normal_yaw)),
                    xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->,head_width=0.08,head_length=0.06",
                                   color="blue", lw=1.5),
                    zorder=5)

        # SLAM landmark + error line
        if mid in slam_lm_aligned:
            sx, sy = slam_lm_aligned[mid]
            gt_mx, gt_my = compute_gt_marker_face(bx, by)
            # Error line: GT marker face ↔ SLAM landmark
            ax.plot([gt_mx, sx], [gt_my, sy], "--", color="gray",
                    linewidth=0.8, alpha=0.6, zorder=3)
            ax.plot(sx, sy, "D", color="magenta", markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=8)
            err = math.hypot(sx - gt_mx, sy - gt_my)
            ax.annotate(f"{err:.2f}m", (sx, sy),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=7, color="magenta", zorder=8)

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(facecolor="moccasin", edgecolor="darkorange",
                       label="GT Box (0.5\u00d70.5 m)"),
        plt.Line2D([0], [0], color="red", linewidth=3,
                   label="ArUco Marker Face (0.3 m)"),
        plt.Line2D([0], [0], color="blue", linewidth=1.5, marker=">",
                   markersize=6, label="Marker Normal"),
        plt.Line2D([0], [0], color="black", marker="x", linestyle="None",
                   markersize=8, markeredgewidth=2, label="Box Look-At Center"),
    ]
    if args.bag:
        legend_elements.extend([
            plt.Line2D([0], [0], color="green", linewidth=1.2,
                       label="GT Trajectory (/odom_gt)"),
        ])
        if slam_aligned_x is not None:
            legend_elements.extend([
                plt.Line2D([0], [0], color="purple", linewidth=1.0,
                           label=f"SLAM Trajectory ({args.slam_topic})"),
            ])
        if slam_lm_aligned:
            legend_elements.extend([
                plt.Line2D([0], [0], color="magenta", marker="D",
                           linestyle="None", markersize=6,
                           markeredgecolor="black", markeredgewidth=0.5,
                           label="SLAM Landmark (aligned)"),
                plt.Line2D([0], [0], color="gray", linestyle="--",
                           linewidth=0.8, label="Landmark Error"),
            ])
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Auto-fit axis
    all_x = [v[0] for v in GAZEBO_BOX_CENTERS.values()]
    all_y = [v[1] for v in GAZEBO_BOX_CENTERS.values()]
    if gt_x is not None and len(gt_x) > 0:
        all_x.extend([gt_x.min(), gt_x.max()])
        all_y.extend([gt_y.min(), gt_y.max()])
    if slam_aligned_x is not None:
        all_x.extend([slam_aligned_x.min(), slam_aligned_x.max()])
        all_y.extend([slam_aligned_y.min(), slam_aligned_y.max()])
    pad = 1.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "gt_box_marker_layout.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
