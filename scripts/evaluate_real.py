#!/usr/bin/env python3
"""Real Bag SLAM Evaluation (no GT odom)
=========================================
Known GT waypoints에서 정지 정확도, loop closure, SLAM-EKF divergence를 평가합니다.

사용법:
    python3 evaluate_real.py --bag /bags/real --output-dir results/exp1
    python3 evaluate_real.py --bag /bags/real --waypoints "0,4;6,4"
    python3 evaluate_real.py --bag /bags/real --swap-xy   # user_x=slam_y, user_y=slam_x

출력:
    - trajectory_slam.png / trajectory_ekf.png (user 좌표계)
    - metrics_real.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── 공통 유틸은 visualize_gt_markers.py에서 import ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_gt_markers import (
    extract_odom_with_ts,
    extract_slam_landmarks,
    extract_aruco_detection_times,
    quat_to_yaw,
    normalize_angle,
)


# ═══════════════════════════════════════
# Stop detection
# ═══════════════════════════════════════

def detect_stops(ts, x, y, speed_threshold=0.03, min_duration=1.0,
                 smooth_window=10):
    """Detect stop segments where speed < threshold for > min_duration.

    Returns list of dicts: {t_start, t_end, duration, x, y, idx}.
    """
    dt = np.diff(ts)
    dt[dt == 0] = 1e-6
    speed = np.sqrt((np.diff(x) / dt)**2 + (np.diff(y) / dt)**2)
    kernel = np.ones(smooth_window) / smooth_window
    speed_smooth = np.convolve(speed, kernel, mode="same")
    is_stopped = speed_smooth < speed_threshold

    changes = np.diff(is_stopped.astype(int))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0] + 1)
    if is_stopped[0]:
        starts.insert(0, 0)
    if is_stopped[-1]:
        ends.append(len(is_stopped))

    stops = []
    for s, e in zip(starts, ends):
        if e >= len(ts):
            e = len(ts) - 1
        duration = ts[e] - ts[s]
        if duration >= min_duration:
            mid = (s + e) // 2
            stops.append({
                "t_start": float(ts[s] - ts[0]),
                "t_end": float(ts[e] - ts[0]),
                "duration": float(duration),
                "x": float(x[mid]),
                "y": float(y[mid]),
                "idx": mid,
            })
    return stops


def match_waypoints(waypoints, stops, traj_x, traj_y, traj_ts):
    """Match GT waypoints to stops and closest trajectory points.

    Returns list of dicts per waypoint.
    """
    results = []
    for gx, gy in waypoints:
        # Nearest stop
        best_err = float("inf")
        best_stop = None
        for st in stops:
            err = math.hypot(st["x"] - gx, st["y"] - gy)
            if err < best_err:
                best_err = err
                best_stop = st

        # Closest trajectory point
        dists = np.sqrt((traj_x - gx)**2 + (traj_y - gy)**2)
        closest_idx = int(np.argmin(dists))
        closest_dist = float(dists[closest_idx])

        results.append({
            "gt": (gx, gy),
            "stop": best_stop,
            "stop_error": best_err if best_stop else float("nan"),
            "closest_idx": closest_idx,
            "closest_pos": (float(traj_x[closest_idx]),
                            float(traj_y[closest_idx])),
            "closest_dist": closest_dist,
            "closest_time": float(traj_ts[closest_idx] - traj_ts[0]),
        })
    return results


# ═══════════════════════════════════════
# Plotting
# ═══════════════════════════════════════

def draw_slam_landmarks(ax, landmarks):
    """Draw SLAM estimated landmark positions + normals."""
    if not landmarks:
        return
    for mid, (sx, sy, syaw) in landmarks.items():
        ax.plot(sx, sy, "D", color="magenta", markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, zorder=8)
        ax.annotate(f"{mid}", (sx, sy),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, fontweight="bold", ha="center",
                    color="darkblue", zorder=9)
        ax.annotate("", xy=(sx + 0.3 * math.cos(syaw),
                            sy + 0.3 * math.sin(syaw)),
                    xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.06,head_length=0.05",
                        color="magenta", lw=1.2), zorder=9)


def make_trajectory_plot(traj_x, traj_y, traj_ts, label, color,
                         landmarks, waypoints, wp_results, stops):
    """Generate trajectory plot in user coordinate frame."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))
    ax.set_aspect("equal")
    ax.set_title(f"{label} (Real)", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # 1m 격자
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, alpha=0.3)

    # Trajectory
    if traj_x is not None and len(traj_x) > 0:
        ax.plot(traj_x, traj_y, "-", color=color, linewidth=1.5,
                alpha=0.8, zorder=2, label=label)
        ax.plot(traj_x[0], traj_y[0], "o", color=color, markersize=8,
                zorder=7)
        ax.plot(traj_x[-1], traj_y[-1], "^", color=color, markersize=8,
                zorder=7)

    # Landmarks
    draw_slam_landmarks(ax, landmarks)

    # GT waypoints
    for i, (gx, gy) in enumerate(waypoints):
        ax.plot(gx, gy, "x", color="blue", markersize=14,
                markeredgewidth=3, zorder=10)
        ax.annotate(f"GT{i+1} ({gx:.0f},{gy:.0f})", (gx, gy),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="blue", fontweight="bold", zorder=10)

    # Stop markers
    for i, st in enumerate(stops):
        ax.plot(st["x"], st["y"], "s", color="red", markersize=10,
                markeredgecolor="black", markeredgewidth=1, zorder=9,
                alpha=0.7)

    # Error lines from closest traj point to GT
    for wp in wp_results:
        gx, gy = wp["gt"]
        cx, cy = wp["closest_pos"]
        ax.plot([gx, cx], [gy, cy], "--", color="gray", linewidth=1,
                alpha=0.6, zorder=3)
        mid_x, mid_y = (gx + cx) / 2, (gy + cy) / 2
        ax.annotate(f"{wp['closest_dist']:.2f}m", (mid_x, mid_y),
                    fontsize=8, color="red", fontweight="bold",
                    ha="center", va="bottom", zorder=10)

    # Set limits
    all_x, all_y = [], []
    if traj_x is not None and len(traj_x) > 0:
        all_x.extend([traj_x.min(), traj_x.max()])
        all_y.extend([traj_y.min(), traj_y.max()])
    for gx, gy in waypoints:
        all_x.append(gx)
        all_y.append(gy)
    if landmarks:
        for _, (lx, ly, _) in landmarks.items():
            all_x.append(lx)
            all_y.append(ly)
    if all_x:
        pad = 1.5
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # Legend
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color="blue", marker="x", linestyle="None",
                   markersize=10, markeredgewidth=3, label="GT Waypoint"),
        plt.Line2D([0], [0], color="red", marker="s", linestyle="None",
                   markersize=8, markeredgecolor="black",
                   label="Detected Stop"),
    ])
    if landmarks:
        handles.extend([
            plt.Line2D([0], [0], color="magenta", marker="D",
                       linestyle="None", markersize=7,
                       markeredgecolor="black", markeredgewidth=0.5,
                       label="SLAM Landmark"),
            plt.Line2D([0], [0], color="magenta", linewidth=1.2,
                       marker=">", markersize=5, label="Landmark Normal"),
        ])
    handles.extend([
        plt.Line2D([0], [0], color="gray", marker="o", linestyle="None",
                   markersize=6, label="Start"),
        plt.Line2D([0], [0], color="gray", marker="^", linestyle="None",
                   markersize=6, label="End"),
    ])
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    # Metrics text box
    lines = []
    for i, wp in enumerate(wp_results):
        gx, gy = wp["gt"]
        lines.append(f"WP{i+1} ({gx:.0f},{gy:.0f}): "
                     f"closest={wp['closest_dist']:.3f}m  "
                     f"stop={wp['stop_error']:.3f}m")
    if traj_x is not None and len(traj_x) > 1:
        travel = float(np.sum(np.sqrt(np.diff(traj_x)**2 +
                                      np.diff(traj_y)**2)))
        loop_err = math.hypot(traj_x[-1] - traj_x[0],
                              traj_y[-1] - traj_y[0])
        lines.append(f"Travel: {travel:.1f}m  Loop err: {loop_err:.3f}m")
    text = "\n".join(lines)
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85),
            zorder=20)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════
# Metrics
# ═══════════════════════════════════════

def compute_all_metrics(slam_ux, slam_uy, slam_ts,
                        ekf_ux, ekf_uy, ekf_ts,
                        waypoints, slam_stops, ekf_stops,
                        slam_wp, ekf_wp):
    """Compute all metrics and return as dict."""
    m = {}

    # Travel distance
    if slam_ux is not None and len(slam_ux) > 1:
        m["slam/travel_distance_m"] = float(
            np.sum(np.sqrt(np.diff(slam_ux)**2 + np.diff(slam_uy)**2)))
    if ekf_ux is not None and len(ekf_ux) > 1:
        m["ekf/travel_distance_m"] = float(
            np.sum(np.sqrt(np.diff(ekf_ux)**2 + np.diff(ekf_uy)**2)))

    # Loop closure error
    if slam_ux is not None and len(slam_ux) > 1:
        m["slam/loop_closure_error_m"] = math.hypot(
            slam_ux[-1] - slam_ux[0], slam_uy[-1] - slam_uy[0])
    if ekf_ux is not None and len(ekf_ux) > 1:
        m["ekf/loop_closure_error_m"] = math.hypot(
            ekf_ux[-1] - ekf_ux[0], ekf_uy[-1] - ekf_uy[0])

    # Waypoint accuracy
    for prefix, wp_results in [("slam", slam_wp), ("ekf", ekf_wp)]:
        if wp_results is None:
            continue
        for i, wp in enumerate(wp_results):
            m[f"{prefix}/wp{i+1}_stop_error_m"] = wp["stop_error"]
            m[f"{prefix}/wp{i+1}_closest_error_m"] = wp["closest_dist"]

    # SLAM vs EKF divergence
    if (slam_ts is not None and ekf_ts is not None
            and len(slam_ts) > 0 and len(ekf_ts) > 0):
        idx_s, idx_e = [], []
        j = 0
        for i in range(len(slam_ts)):
            while (j < len(ekf_ts) - 1
                   and abs(ekf_ts[j + 1] - slam_ts[i])
                   < abs(ekf_ts[j] - slam_ts[i])):
                j += 1
            if abs(ekf_ts[j] - slam_ts[i]) <= 0.5:
                idx_s.append(i)
                idx_e.append(j)
        if len(idx_s) > 0:
            idx_s, idx_e = np.array(idx_s), np.array(idx_e)
            diff = np.sqrt((slam_ux[idx_s] - ekf_ux[idx_e])**2 +
                           (slam_uy[idx_s] - ekf_uy[idx_e])**2)
            m["slam_vs_ekf/synced_pairs"] = len(idx_s)
            m["slam_vs_ekf/rmse_m"] = float(np.sqrt((diff**2).mean()))
            m["slam_vs_ekf/mean_m"] = float(diff.mean())
            m["slam_vs_ekf/max_m"] = float(diff.max())
            m["slam_vs_ekf/std_m"] = float(diff.std())

    return m


def save_metrics_csv(filepath, metrics, landmarks_user):
    """Save metrics and landmark positions to CSV."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            w.writerow([k, f"{v:.4f}" if isinstance(v, float) else str(v)])
        for mid, (lx, ly, lyaw) in sorted(landmarks_user.items()):
            w.writerow([f"landmark/{mid}/x", f"{lx:.4f}"])
            w.writerow([f"landmark/{mid}/y", f"{ly:.4f}"])
            w.writerow([f"landmark/{mid}/yaw_deg",
                         f"{math.degrees(lyaw):.1f}"])
    print(f"  CSV saved: {filepath}")


def print_summary(metrics, slam_wp, ekf_wp, waypoints,
                  slam_stops, ekf_stops, landmarks_user):
    """Print summary table to console."""
    print()
    print("=" * 62)
    print("       Real Bag SLAM Evaluation")
    print("=" * 62)

    # Waypoint accuracy
    for prefix, wp_results, stops in [
            ("SLAM", slam_wp, slam_stops),
            ("EKF", ekf_wp, ekf_stops)]:
        if wp_results is None:
            continue
        print(f"\n  [{prefix} Waypoint Accuracy]")
        for i, wp in enumerate(wp_results):
            gx, gy = wp["gt"]
            print(f"    GT{i+1} ({gx:.0f}, {gy:.0f}):")
            if wp["stop"] is not None:
                st = wp["stop"]
                print(f"      Stop:    ({st['x']:.3f}, {st['y']:.3f})  "
                      f"t={st['t_start']:.1f}-{st['t_end']:.1f}s  "
                      f"err={wp['stop_error']:.4f} m")
            print(f"      Closest: ({wp['closest_pos'][0]:.3f}, "
                  f"{wp['closest_pos'][1]:.3f})  "
                  f"t={wp['closest_time']:.1f}s  "
                  f"err={wp['closest_dist']:.4f} m")

    # Travel & loop closure
    print(f"\n  [Travel & Loop Closure]")
    for prefix in ["slam", "ekf"]:
        travel = metrics.get(f"{prefix}/travel_distance_m")
        loop = metrics.get(f"{prefix}/loop_closure_error_m")
        if travel is not None:
            print(f"    {prefix.upper():4s} travel: {travel:.2f} m  "
                  f"loop err: {loop:.4f} m")

    # SLAM vs EKF
    n = metrics.get("slam_vs_ekf/synced_pairs")
    if n is not None:
        print(f"\n  [SLAM vs EKF Divergence]  ({n} synced)")
        print(f"    RMSE: {metrics['slam_vs_ekf/rmse_m']:.4f} m  "
              f"Mean: {metrics['slam_vs_ekf/mean_m']:.4f} m  "
              f"Max: {metrics['slam_vs_ekf/max_m']:.4f} m")

    # Landmarks
    if landmarks_user:
        print(f"\n  [SLAM Landmarks]")
        for mid, (lx, ly, lyaw) in sorted(landmarks_user.items()):
            print(f"    ID {mid}: ({lx:.3f}, {ly:.3f})  "
                  f"yaw={math.degrees(lyaw):.1f} deg")

    print("=" * 62)
    print()


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def parse_waypoints(s):
    """Parse waypoint string 'x1,y1;x2,y2;...' into list of (x, y)."""
    wps = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        wps.append((float(x.strip()), float(y.strip())))
    return wps


def main():
    parser = argparse.ArgumentParser(
        description="Real bag SLAM evaluation with GT waypoints")
    parser.add_argument("--bag", type=str, required=True,
                        help="Path to rosbag directory")
    parser.add_argument("--waypoints", type=str, default="0,4;6,4",
                        help="GT waypoints 'x1,y1;x2,y2' (default: 0,4;6,4)")
    parser.add_argument("--slam-topic", type=str,
                        default="/aruco_slam/odom")
    parser.add_argument("--ekf-topic", type=str, default="/ekf/odom")
    parser.add_argument("--swap-xy", action="store_true",
                        help="Coordinate transform: user_x=slam_y, "
                             "user_y=slam_x")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    waypoints = parse_waypoints(args.waypoints)
    print(f"  GT waypoints: {waypoints}")
    print(f"  Coordinate swap: {args.swap_xy}")

    # ── Extract ──
    slam_ts, slam_x, slam_y, slam_yaw = extract_odom_with_ts(
        args.bag, args.slam_topic)
    ekf_ts, ekf_x, ekf_y, ekf_yaw = extract_odom_with_ts(
        args.bag, args.ekf_topic)
    lm_raw = extract_slam_landmarks(args.bag)
    det_ts, _ = extract_aruco_detection_times(args.bag)

    # ── Coordinate transform ──
    if args.swap_xy:
        def to_user_xy(sx, sy):
            return sy.copy(), sx.copy()
        def to_user_yaw(syaw):
            # swap axes: rotate -90 then mirror → effectively swap
            return np.array([math.atan2(math.cos(y), math.sin(y))
                             for y in syaw])
    else:
        def to_user_xy(sx, sy):
            return sx.copy(), sy.copy()
        def to_user_yaw(syaw):
            return syaw.copy()

    # Transform trajectories
    slam_ux = slam_uy = slam_uyaw = None
    ekf_ux = ekf_uy = ekf_uyaw = None
    if slam_ts is not None:
        slam_ux, slam_uy = to_user_xy(slam_x, slam_y)
        slam_uyaw = to_user_yaw(slam_yaw)
    if ekf_ts is not None:
        ekf_ux, ekf_uy = to_user_xy(ekf_x, ekf_y)
        ekf_uyaw = to_user_yaw(ekf_yaw)

    # Transform landmarks
    landmarks_user = {}
    for mid, (lx, ly, lyaw) in lm_raw.items():
        ux, uy = to_user_xy(np.array([lx]), np.array([ly]))
        uyaw = to_user_yaw(np.array([lyaw]))
        landmarks_user[mid] = (float(ux[0]), float(uy[0]), float(uyaw[0]))

    # ── Detect stops ──
    slam_stops = []
    ekf_stops = []
    if slam_ux is not None:
        slam_stops = detect_stops(slam_ts, slam_ux, slam_uy)
    if ekf_ux is not None:
        ekf_stops = detect_stops(ekf_ts, ekf_ux, ekf_uy,
                                 smooth_window=20)

    # ── Match waypoints ──
    slam_wp = None
    ekf_wp = None
    if slam_ux is not None:
        slam_wp = match_waypoints(waypoints, slam_stops,
                                  slam_ux, slam_uy, slam_ts)
    if ekf_ux is not None:
        ekf_wp = match_waypoints(waypoints, ekf_stops,
                                 ekf_ux, ekf_uy, ekf_ts)

    # ── Compute metrics ──
    metrics = compute_all_metrics(
        slam_ux, slam_uy, slam_ts,
        ekf_ux, ekf_uy, ekf_ts,
        waypoints, slam_stops, ekf_stops,
        slam_wp, ekf_wp)

    # ── Print summary ──
    print_summary(metrics, slam_wp, ekf_wp, waypoints,
                  slam_stops, ekf_stops, landmarks_user)

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ── SLAM plot ──
    if slam_ux is not None:
        fig = make_trajectory_plot(
            slam_ux, slam_uy, slam_ts,
            f"SLAM ({args.slam_topic})", "purple",
            landmarks_user, waypoints, slam_wp, slam_stops)
        path = os.path.join(out_dir, "trajectory_slam.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {path}")

    # ── EKF plot ──
    if ekf_ux is not None:
        fig = make_trajectory_plot(
            ekf_ux, ekf_uy, ekf_ts,
            f"EKF ({args.ekf_topic})", "darkorange",
            {}, waypoints, ekf_wp, ekf_stops)
        path = os.path.join(out_dir, "trajectory_ekf.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {path}")

    # ── Save CSV ──
    csv_path = os.path.join(out_dir, "metrics_real.csv")
    save_metrics_csv(csv_path, metrics, landmarks_user)


if __name__ == "__main__":
    main()
