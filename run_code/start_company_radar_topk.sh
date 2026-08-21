#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
PROCESSED_ROOT=${RACFORMER_COMPANY_PROCESSED_ROOT:-/mnt/diskNvme1/hyh/datasets/company_20260818/processed_racformer}
OUTPUT_ROOT=${COMPANY_RADAR_TOPK_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_radar_topk}
GPU_ID=${COMPANY_RADAR_TOPK_GPU:-0}

if [[ ! -f $PROCESSED_ROOT/custom_infos_train_sweep.pkl ||
      ! -f $PROCESSED_ROOT/custom_infos_val_sweep.pkl ]]; then
    echo "Converted train/val info files are missing under $PROCESSED_ROOT" >&2
    exit 2
fi
if pgrep -af 'tools/analysis/train_company_radar_topk.py' >/dev/null; then
    echo "Company radar Top-K experiment is already running" >&2
    exit 3
fi

mkdir -p "$OUTPUT_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUTPUT_ROOT/$STAMP"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"

cd "$REPO_ROOT"
runner=(nice -n 5)
if command -v ionice >/dev/null 2>&1; then
    runner+=(ionice -c 2 -n 5)
fi
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "${runner[@]}" \
    python tools/analysis/train_company_radar_topk.py \
    --processed-root "$PROCESSED_ROOT" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --epochs 12 \
    --use-frames 4 \
    --topk 64 128 256 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "$OUT_DIR" > "$OUTPUT_ROOT/latest_run.txt"

echo "200m learned radar Top-K experiment started"
echo "Physical GPU=$GPU_ID"
echo "PID=$PID"
echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"
