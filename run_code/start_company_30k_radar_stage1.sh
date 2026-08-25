#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
FRONT50_TRAINVAL=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
FRONT50_TEST=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3
FRONT350_TRAINVAL=/mnt/diskNvme1/hyh/data/company_20260818_30k_q5_350m_f4/processed_trainval_v1
OUTPUT_ROOT=${COMPANY_30K_RADAR_STAGE1_ROOT:-/mnt/diskNvme1/hyh/results/3DH-Query/company_20260818_30k_radar_stage1}

usage() {
    echo "Usage: $0 front50 | front350"
}

MODE=${1:-}
[[ $MODE == front50 || $MODE == front350 ]] || { usage; exit 2; }
if pgrep -af 'company_radar_candidate_recall.py.*company_20260818_30k' >/dev/null; then
    echo "A 30k radar Stage-1 analysis is already running" >&2
    exit 3
fi

if [[ $MODE == front50 ]]; then
    for required in \
            "$FRONT50_TRAINVAL/custom_infos_train_sweep.pkl" \
            "$FRONT50_TRAINVAL/custom_infos_val_sweep.pkl" \
            "$FRONT50_TRAINVAL/conversion_summary.json" \
            "$FRONT50_TEST/custom_infos_test_sweep.pkl"; do
        [[ -s $required ]] || { echo "missing or empty: $required" >&2; exit 4; }
    done
else
    for required in \
            "$FRONT350_TRAINVAL/custom_infos_train_sweep.pkl" \
            "$FRONT350_TRAINVAL/custom_infos_val_sweep.pkl" \
            "$FRONT350_TRAINVAL/conversion_summary.json"; do
        [[ -s $required ]] || {
            echo "350m conversion is not complete: $required" >&2
            exit 4
        }
    done
fi

mkdir -p "$OUTPUT_ROOT"
OUT_DIR="$OUTPUT_ROOT/${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
echo "$OUT_DIR" > "$OUTPUT_ROOT/latest_${MODE}.txt"

cd "$REPO_ROOT"
if [[ $MODE == front50 ]]; then
    python tools/analysis/audit_company_radar_processed.py \
        --processed-root "$FRONT50_TRAINVAL" --splits train val \
        --out "$OUT_DIR/trainval_audit.json" > "$OUT_DIR/trainval_audit.log"
    python tools/analysis/audit_company_radar_processed.py \
        --processed-root "$FRONT50_TEST" --splits test \
        --out "$OUT_DIR/test_audit.json" > "$OUT_DIR/test_audit.log"
else
    python tools/analysis/audit_company_radar_processed.py \
        --processed-root "$FRONT350_TRAINVAL" --splits train val \
        --out "$OUT_DIR/trainval_audit.json" > "$OUT_DIR/trainval_audit.log"
fi
nohup bash -c '
set -Eeuo pipefail
repo=$1
mode=$2
out=$3
root50=$4
test50=$5
root350=$6
cd "$repo"
trap '\''touch "$out/FAILED"'\'' ERR
runner=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then runner+=(ionice -c 2 -n 7); fi
if [[ $mode == front50 ]]; then
    "${runner[@]}" python tools/analysis/company_radar_candidate_recall.py \
        --processed-root "$root50" --out-dir "$out/trainval" \
        --split all --use-frames 4 \
        --point-cloud-range 0 -20 -3 50 20 3 \
        --range-bins 0 25 50 > "$out/trainval.log" 2>&1
    "${runner[@]}" python tools/analysis/company_radar_candidate_recall.py \
        --processed-root "$test50" --out-dir "$out/test" \
        --split test --use-frames 4 \
        --point-cloud-range 0 -20 -3 50 20 3 \
        --range-bins 0 25 50 > "$out/test.log" 2>&1
else
    "${runner[@]}" python tools/analysis/company_radar_candidate_recall.py \
        --processed-root "$root350" --out-dir "$out/all_classes_0_200" \
        --split all --use-frames 4 \
        --point-cloud-range 0 -20 -3 200 20 3 \
        --range-bins 0 50 100 150 200 > "$out/all_classes_0_200.log" 2>&1
    "${runner[@]}" python tools/analysis/company_radar_candidate_recall.py \
        --processed-root "$root350" --out-dir "$out/car_0_350" \
        --split all --use-frames 4 --class-names car \
        --point-cloud-range 0 -20 -3 350 20 3 \
        --range-bins 0 50 100 150 200 250 300 350 \
        > "$out/car_0_350.log" 2>&1
fi
touch "$out/ALL_DONE"
trap - ERR
' _ "$REPO_ROOT" "$MODE" "$OUT_DIR" "$FRONT50_TRAINVAL" \
    "$FRONT50_TEST" "$FRONT350_TRAINVAL" \
    > "$OUT_DIR/nohup.log" 2>&1 < /dev/null &
echo $! > "$OUT_DIR/analysis.pid"
echo "30k radar Stage-1 started: mode=$MODE"
echo "OUT_DIR=$OUT_DIR"
echo "PID=$(cat "$OUT_DIR/analysis.pid")"
