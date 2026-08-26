#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WEIGHTS=${1:?Usage: $0 CHECKPOINT [GPU]}
GPU=${2:-1}
TEST_ROOT=${RACFORMER_COMPANY_TEST_ROOT:-/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3}
CONFIG=configs/racformer_company_front_50m_q200_f4_30k_train.py
RESULT_ROOT=${COMPANY_BASELINE_RESULT_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/50m_q200_f4_30k_baseline}
RUN_DIR="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$RESULT_ROOT/latest_run.txt"
cd "$REPO_ROOT"

[[ -s $WEIGHTS ]] || { echo "checkpoint not found: $WEIGHTS" >&2; exit 3; }
[[ -s $TEST_ROOT/custom_infos_test_sweep.pkl ]] || {
    echo "test infos not found under $TEST_ROOT" >&2; exit 3; }

export RACFORMER_COMPANY_TEST_ROOT="$TEST_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" python val.py \
    --config "$CONFIG" --weights "$WEIGHTS" --split test --batch_size 1 \
    --eval-profiles car_only main3 --out "$RUN_DIR/predictions.pkl" \
    2>&1 | tee "$RUN_DIR/eval.log"
touch "$RUN_DIR/EVALUATION_DONE"
grep -E 'company/(car_only|main3)/' "$RUN_DIR/eval.log" \
    > "$RUN_DIR/metrics_summary.txt"

CUDA_VISIBLE_DEVICES="$GPU" python tools/audit_company_prediction_geometry.py \
    --config "$CONFIG" --split test \
    --ann-file "$TEST_ROOT/custom_infos_test_sweep.pkl" \
    --predictions "$RUN_DIR/predictions.pkl" \
    --classes car truck bicycle --score-threshold 0.1 \
    --output "$RUN_DIR/geometry_audit.json" \
    2>&1 | tee "$RUN_DIR/geometry_audit.log"
touch "$RUN_DIR/GEOMETRY_AUDIT_DONE"

CUDA_VISIBLE_DEVICES="$GPU" python tools/visualize_company_predictions.py \
    --ann-file "$TEST_ROOT/custom_infos_test_sweep.pkl" \
    --predictions "$RUN_DIR/predictions.pkl" \
    --output-dir "$RUN_DIR/visualizations" --num-samples 100 \
    --score-threshold 0.1 --bev-forward-range 50 \
    2>&1 | tee "$RUN_DIR/visualize.log"
touch "$RUN_DIR/VISUALIZATION_DONE"
touch "$RUN_DIR/ALL_DONE"
echo "baseline evaluation complete: $RUN_DIR"
