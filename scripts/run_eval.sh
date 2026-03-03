#!/bin/bash
# SLAM evaluation runner.
# Auto-creates output directory with incremental exp numbering.
#
# Usage (inside container):
#   bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval.sh /bags/mapping_20260303_075750
#   bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval.sh /bags/mapping_xxx --slam-topic /ekf/odom
#
# Usage (from host via docker exec):
#   docker exec hunter_humble bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval.sh /bags/mapping_20260303_075750

set -e

BAG_PATH="$1"
if [ -z "$BAG_PATH" ]; then
    echo "Usage: $0 <bag_path> [--slam-topic <topic>]"
    exit 1
fi
shift

RESULTS_BASE="/ros_ws/src/aruco_sam_ailab/results"
mkdir -p "$RESULTS_BASE"

# Find next exp number
LAST=$(ls -d "$RESULTS_BASE"/exp* 2>/dev/null | sort -V | tail -1 | grep -oP '\d+$' || echo 0)
NEXT=$((LAST + 1))
OUT_DIR="$RESULTS_BASE/exp${NEXT}"
mkdir -p "$OUT_DIR"

echo "================================================"
echo "  SLAM Evaluation — exp${NEXT}"
echo "  Bag:    $BAG_PATH"
echo "  Output: $OUT_DIR"
echo "================================================"

python3 /ros_ws/src/aruco_sam_ailab/scripts/visualize_gt_markers.py \
    --bag "$BAG_PATH" \
    --output-dir "$OUT_DIR" \
    "$@"

# Copy bag path info
echo "$BAG_PATH" > "$OUT_DIR/bag_path.txt"
echo "$@" >> "$OUT_DIR/bag_path.txt"

echo ""
echo "Done! Results in: $OUT_DIR"
ls -la "$OUT_DIR"
