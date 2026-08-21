#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
PROCESSED_ROOT=${RACFORMER_COMPANY_PROCESSED_ROOT:-/mnt/diskNvme1/hyh/datasets/company_20260818/processed_racformer}
OUTPUT_ROOT=${COMPANY_QUALITY_VIDEO_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_quality_video}

if [[ ! -f $PROCESSED_ROOT/custom_infos_train_sweep.pkl ||
      ! -f $PROCESSED_ROOT/custom_infos_val_sweep.pkl ]]; then
    echo "Converted train/val info files are missing under $PROCESSED_ROOT" >&2
    exit 2
fi
if pgrep -af 'tools/render_chengtech_quality_video.py' >/dev/null; then
    echo "A ChengTech quality-video renderer is already running:" >&2
    pgrep -af 'tools/render_chengtech_quality_video.py' >&2
    exit 3
fi

mkdir -p "$OUTPUT_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$OUTPUT_ROOT/company_20260818_quality_${STAMP}.mp4"
LOG="$OUTPUT_ROOT/company_20260818_quality_${STAMP}.log"
PID_FILE="$OUTPUT_ROOT/company_20260818_quality_${STAMP}.pid"

cd "$REPO_ROOT"
runner=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then
    runner+=(ionice -c 2 -n 7)
fi
nohup "${runner[@]}" python tools/render_chengtech_quality_video.py \
    --processed-root "$PROCESSED_ROOT" \
    --output "$OUTPUT" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
echo "$OUTPUT" > "$OUTPUT_ROOT/latest_video.txt"
echo "$LOG" > "$OUTPUT_ROOT/latest_log.txt"
echo "$PID_FILE" > "$OUTPUT_ROOT/latest_pid_file.txt"

echo "Quality-video rendering started at reduced CPU/I/O priority"
echo "PID=$PID"
echo "VIDEO=$OUTPUT"
echo "LOG=$LOG"
