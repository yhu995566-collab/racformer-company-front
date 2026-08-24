#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:-1}
DATASET_ROOT=${RACFORMER_COMPANY_PROCESSED_ROOT:-$HOME/hyh/company_20260818/processed_racformer}
RUN_TAG=$(date +%Y%m%d_%H%M%S)
RESULT_ROOT=${Q_PROFILE_RESULT_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_q1_q5_profiles/$RUN_TAG}
LATEST_FILE=/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_q1_q5_profiles/latest_run.txt

mkdir -p "$RESULT_ROOT"
echo "$RESULT_ROOT" > "$LATEST_FILE"
cd "$REPO_ROOT"

declare -A RUN_DIRS=(
    [q1]="outputs/3dh_query_company_20260818_q1/2026-08-21/10-12-05"
    [q2]="outputs/3dh_query_company_20260818_q2/2026-08-21/12-38-41"
    [q3]="outputs/3dh_query_company_20260818_q3/2026-08-21/15-06-03"
    [q4]="outputs/3dh_query_company_20260818_q4/2026-08-21/17-33-48"
    [q5]="outputs/3dh_query_company_20260818_q5/2026-08-21/20-03-00"
)
declare -A EPOCHS=(
    [q1]=34
    [q2]=32
    [q3]=32
    [q4]=36
    [q5]=34
)

test -f "$DATASET_ROOT/custom_infos_val_sweep.pkl"

for q in q1 q2 q3 q4 q5; do
    config="configs/3dh_query_company_20260818_${q}.py"
    checkpoint="${RUN_DIRS[$q]}/epoch_${EPOCHS[$q]}.pth"
    output_dir="$RESULT_ROOT/$q"
    mkdir -p "$output_dir"
    test -f "$config"
    test -f "$checkpoint"

    echo "===== ${q^^}: epoch ${EPOCHS[$q]} =====" | tee "$output_dir/header.txt"
    CUDA_VISIBLE_DEVICES="$GPU" \
    RACFORMER_COMPANY_PROCESSED_ROOT="$DATASET_ROOT" \
    python val.py \
        --config "$config" \
        --weights "$checkpoint" \
        --split val \
        --batch_size 1 \
        --eval-profiles car_only main3 \
        --out "$output_dir/predictions.pkl" \
        2>&1 | tee "$output_dir/eval.log"
done

grep -hE \
    'profile:|company/(car_only|main3)/(BEV_mAP|3D_mAP|overall_precision|overall_recall|total_gt)' \
    "$RESULT_ROOT"/q*/eval.log > "$RESULT_ROOT/profile_summary.txt"

echo "Q1-Q5 profile evaluation complete: $RESULT_ROOT"
