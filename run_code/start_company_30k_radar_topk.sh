#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
TRAINVAL_ROOT=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
TEST_ROOT=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3
OUTPUT_ROOT=${COMPANY_30K_RADAR_TOPK_ROOT:-/mnt/diskNvme1/hyh/results/3DH-Query/company_20260818_30k_radar_topk}
GPU_ID=${COMPANY_30K_RADAR_TOPK_GPU:-1}

for required in \
        "$TRAINVAL_ROOT/custom_infos_train_sweep.pkl" \
        "$TRAINVAL_ROOT/custom_infos_val_sweep.pkl" \
        "$TRAINVAL_ROOT/conversion_summary.json" \
        "$TEST_ROOT/custom_infos_test_sweep.pkl"; do
    [[ -s $required ]] || { echo "missing or empty: $required" >&2; exit 2; }
done
if pgrep -af 'train_company_radar_topk.py.*company_20260818_30k' >/dev/null; then
    echo "A 30k radar Top-K experiment is already running" >&2
    exit 3
fi

mkdir -p "$OUTPUT_ROOT"
OUT_DIR="$OUTPUT_ROOT/front50_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
echo "$OUT_DIR" > "$OUTPUT_ROOT/latest_front50.txt"

cd "$REPO_ROOT"
python tools/analysis/audit_company_radar_processed.py \
    --processed-root "$TRAINVAL_ROOT" --splits train val \
    --out "$OUT_DIR/trainval_audit.json" > "$OUT_DIR/trainval_audit.log"
python tools/analysis/audit_company_radar_processed.py \
    --processed-root "$TEST_ROOT" --splits test \
    --out "$OUT_DIR/test_audit.json" > "$OUT_DIR/test_audit.log"
nohup bash -c '
set -Eeuo pipefail
repo=$1
out=$2
gpu=$3
trainval=$4
testroot=$5
cd "$repo"
trap '\''touch "$out/FAILED"'\'' ERR
runner=(nice -n 5)
if command -v ionice >/dev/null 2>&1; then runner+=(ionice -c 2 -n 5); fi
env CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" \
    python tools/analysis/train_company_radar_topk.py \
    --processed-root "$trainval" --test-root "$testroot" \
    --out-dir "$out" \
    --device cuda --epochs 8 --batch-size 8 --use-frames 4 \
    --topk 64 128 256 \
    --point-cloud-range 0 -20 -3 50 20 3 \
    --range-bins 0 25 50 > "$out/train.log" 2>&1
touch "$out/ALL_DONE"
trap - ERR
' _ "$REPO_ROOT" "$OUT_DIR" "$GPU_ID" "$TRAINVAL_ROOT" "$TEST_ROOT" \
    > "$OUT_DIR/nohup.log" 2>&1 < /dev/null &
echo $! > "$OUT_DIR/train.pid"
echo "30k 50m radar Top-K experiment started on physical GPU $GPU_ID"
echo "OUT_DIR=$OUT_DIR"
echo "PID=$(cat "$OUT_DIR/train.pid")"
