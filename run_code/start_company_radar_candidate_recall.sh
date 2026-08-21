#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
PROCESSED_ROOT=${RACFORMER_COMPANY_PROCESSED_ROOT:-/mnt/diskNvme1/hyh/datasets/company_20260818/processed_racformer}
OUTPUT_ROOT=${COMPANY_GAUSSIAN_ANALYSIS_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_gaussian_analysis}

if [[ ! -f $PROCESSED_ROOT/custom_infos_train_sweep.pkl ||
      ! -f $PROCESSED_ROOT/custom_infos_val_sweep.pkl ]]; then
    echo "Converted train/val info files are missing under $PROCESSED_ROOT" >&2
    exit 2
fi
if pgrep -af 'tools/analysis/company_radar_candidate_recall.py' >/dev/null; then
    echo "Company radar candidate analysis is already running" >&2
    exit 3
fi

mkdir -p "$OUTPUT_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUTPUT_ROOT/$STAMP"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/analysis.log"

cd "$REPO_ROOT"
runner=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then
    runner+=(ionice -c 2 -n 7)
fi
nohup "${runner[@]}" python tools/analysis/company_radar_candidate_recall.py \
    --processed-root "$PROCESSED_ROOT" \
    --out-dir "$OUT_DIR" \
    --split all \
    --use-frames 4 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/analysis.pid"
echo "$OUT_DIR" > "$OUTPUT_ROOT/latest_run.txt"

echo "Radar candidate analysis started at reduced CPU/I/O priority"
echo "PID=$PID"
echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"
