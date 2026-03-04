#!/bin/bash
# Real bag SLAM evaluation runner.
# No GT odom — evaluates using known GT waypoints.
#
# Usage (inside container):
#   bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval_real.sh /bags/real
#   bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval_real.sh /bags/real --waypoints "0,4;6,4"
#
# Usage (from host via docker exec):
#   docker exec hunter_humble bash /ros_ws/src/aruco_sam_ailab/scripts/run_eval_real.sh /bags/real

set -e

BAG_PATH="$1"
if [ -z "$BAG_PATH" ]; then
    echo "Usage: $0 <bag_path> [--waypoints 'x1,y1;x2,y2'] [--swap-xy] [extra args...]"
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
echo "  Real Bag Evaluation — exp${NEXT}"
echo "  Bag:    $BAG_PATH"
echo "  Output: $OUT_DIR"
echo "================================================"

python3 /ros_ws/src/aruco_sam_ailab/scripts/evaluate_real.py \
    --bag "$BAG_PATH" \
    --swap-xy \
    --flip-x \
    --output-dir "$OUT_DIR" \
    "$@"

# Copy bag path info
echo "$BAG_PATH" > "$OUT_DIR/bag_path.txt"
echo "$@" >> "$OUT_DIR/bag_path.txt"

echo ""
echo "Done! Results in: $OUT_DIR"
ls -la "$OUT_DIR"
