#!/usr/bin/env python3
"""Visualize GT boxes and ArUco marker attachment positions/orientations in Gazebo frame."""

import math
import os
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
MARKER_FACE_OFFSET = 0.251   # box center -> -X face (marker surface)
MARKER_SIZE = 0.30
BOX_CENTER_GZ = (1.495, -2.425)  # look-at center for all boxes


def compute_box_yaw(bx, by):
    """Box +X axis yaw (Gazebo frame)."""
    cx, cy = BOX_CENTER_GZ
    dx = cx - bx
    dy = cy - by
    return math.atan2(dy, dx) + math.pi


def rotated_rect_corners(cx, cy, w, h, yaw):
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    hw, hh = w / 2, h / 2
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + cos_a * lx - sin_a * ly,
             cy + sin_a * lx + cos_a * ly) for lx, ly in local]


def main():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_title("GT Boxes & ArUco Marker Positions (Gazebo Frame, Top-Down)", fontsize=13)
    ax.set_xlabel("Gazebo X (m)")
    ax.set_ylabel("Gazebo Y (m)")
    ax.grid(True, alpha=0.3)

    # Look-at center point
    ax.plot(*BOX_CENTER_GZ, "kx", markersize=10, markeredgewidth=2, zorder=10)
    ax.annotate("center (1.495, -2.425)", BOX_CENTER_GZ,
                textcoords="offset points", xytext=(10, 10), fontsize=8, color="gray")

    for mid, (bx, by) in GAZEBO_BOX_CENTERS.items():
        yaw = compute_box_yaw(bx, by)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # 1) Box (rotated square)
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

        # 2) Marker on -X face (red line segment)
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

        # 3) Marker normal arrow (facing direction toward center)
        normal_yaw = math.atan2(BOX_CENTER_GZ[1] - by, BOX_CENTER_GZ[0] - bx)
        arrow_len = 0.35
        ax.annotate("",
                    xy=(mx + arrow_len * math.cos(normal_yaw),
                        my + arrow_len * math.sin(normal_yaw)),
                    xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->,head_width=0.08,head_length=0.06",
                                   color="blue", lw=1.5),
                    zorder=5)

        # 4) Position & yaw text
        ax.annotate(f"({bx:.1f}, {by:.1f})\nyaw={math.degrees(yaw):.0f}\u00b0",
                    (bx, by),
                    textcoords="offset points", xytext=(0, -18),
                    fontsize=6, ha="center", color="gray", zorder=6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="moccasin", edgecolor="darkorange",
                       label="GT Box (0.5\u00d70.5 m)"),
        plt.Line2D([0], [0], color="red", linewidth=3,
                   label="ArUco Marker Face (0.3 m)"),
        plt.Line2D([0], [0], color="blue", linewidth=1.5, marker=">",
                   markersize=6, label="Marker Normal (facing dir)"),
        plt.Line2D([0], [0], color="black", marker="x", linestyle="None",
                   markersize=8, markeredgewidth=2,
                   label="Box Look-At Center"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    all_x = [v[0] for v in GAZEBO_BOX_CENTERS.values()]
    all_y = [v[1] for v in GAZEBO_BOX_CENTERS.values()]
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
