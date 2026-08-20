#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
RUN_ROOT=${COMPANY_Q_RUN_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_q1_q5}
PROCESSED_ROOT=${RACFORMER_COMPANY_PROCESSED_ROOT:-/mnt/diskNvme1/hyh/datasets/company_20260818/processed_racformer}
GPU_IDS=${COMPANY_Q_GPU_IDS:-4,5,6,7}

usage() {
    echo "Usage: $0 --background | --run RUN_DIR"
}

log_queue() {
    echo "$(date --iso-8601=seconds) $*" | tee -a "$RUN_DIR/queue.log"
}

if [[ ${1:-} == "--background" ]]; then
    mkdir -p "$RUN_ROOT"
    RUN_DIR="$RUN_ROOT/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    echo "$RUN_DIR" > "$RUN_ROOT/latest_run.txt"
    nohup bash "$SCRIPT_PATH" --run "$RUN_DIR" \
        > "$RUN_DIR/nohup.log" 2>&1 < /dev/null &
    QUEUE_PID=$!
    echo "$QUEUE_PID" > "$RUN_DIR/queue.pid"
    echo "Q1-Q5 queue started"
    echo "RUN_DIR=$RUN_DIR"
    echo "QUEUE_PID=$QUEUE_PID"
    echo "Status: cat \"$RUN_DIR/queue.log\""
    exit 0
fi

if [[ ${1:-} != "--run" || -z ${2:-} ]]; then
    usage
    exit 2
fi

RUN_DIR=$2
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

export RACFORMER_COMPANY_PROCESSED_ROOT="$PROCESSED_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

TRAIN_INFO="$PROCESSED_ROOT/custom_infos_train_sweep.pkl"
VAL_INFO="$PROCESSED_ROOT/custom_infos_val_sweep.pkl"
PRETRAIN="pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"

for required in "$TRAIN_INFO" "$VAL_INFO" "$PRETRAIN"; do
    if [[ ! -f $required ]]; then
        log_queue "missing required file: $required"
        touch "$RUN_DIR/PREFLIGHT_FAILED"
        exit 3
    fi
done

if pgrep -af 'train.py --config configs/3dh_query_company_20260818_q[1-5].py' \
        > "$RUN_DIR/existing_processes.txt"; then
    log_queue "another company Q1-Q5 training process is already running"
    touch "$RUN_DIR/PREFLIGHT_FAILED"
    exit 4
fi

log_queue "preflight passed; GPUs=$GPU_IDS; processed_root=$PROCESSED_ROOT"

for q in 1 2 3 4 5; do
    port=$((29800 + q))
    log_queue "starting Q${q} on physical GPUs $GPU_IDS"

    if torchrun \
        --nproc_per_node=4 \
        --master_port="$port" \
        train.py \
        --config "configs/3dh_query_company_20260818_q${q}.py" \
        --override batch_size=1 \
        > "$RUN_DIR/q${q}.log" 2>&1; then
        log_queue "Q${q} completed successfully"
        touch "$RUN_DIR/Q${q}_DONE"
    else
        status=$?
        log_queue "Q${q} failed with exit code $status; queue stopped"
        touch "$RUN_DIR/Q${q}_FAILED"
        exit "$status"
    fi
done

log_queue "ALL Q1-Q5 COMPLETED"
touch "$RUN_DIR/ALL_DONE"
