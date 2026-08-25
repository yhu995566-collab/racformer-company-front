#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
RUN_ROOT=${COMPANY_30K_Q5_RUN_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_30k_q5_350m}
PROCESSED_ROOT=${COMPANY_30K_Q5_PROCESSED_ROOT:-/mnt/diskNvme1/hyh/data/company_20260818_30k_q5_350m_f4/processed_trainval_v1}
SOURCE_ROOT=/mnt/diskNvme1/DataSet/radar_camera_GT
TRUTH_ROOT=/mnt/diskNvme2/TruthData/20260818
MANIFEST=data_splits/company_20260818_30k_v2.json
CONFIG=configs/3dh_query_company_20260818_30k_q5_f4.py
GPU_IDS=${COMPANY_30K_Q5_GPU_IDS:-4,5,6,7}
MASTER_PORT=${COMPANY_30K_Q5_MASTER_PORT:-29915}
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
GPU_COUNT=${#GPU_LIST[@]}
TARGET_GLOBAL_BATCH=${COMPANY_30K_Q5_GLOBAL_BATCH:-4}
TRAIN_LR=${COMPANY_30K_Q5_TRAIN_LR:-4e-4}
WAIT_FOR_50M=${COMPANY_30K_Q5_WAIT_FOR_50M:-1}
WORKERS_PER_GPU=${COMPANY_30K_Q5_WORKERS_PER_GPU:-1}

usage() {
    echo "Usage: $0 --background | --run RUN_DIR"
}

log() {
    echo "$(date --iso-8601=seconds) $*" | tee -a "$RUN_DIR/queue.log"
}

fail() {
    log "FAILED: $*"
    touch "$RUN_DIR/FAILED"
    exit 1
}

if [[ ${1:-} == "--background" ]]; then
    mkdir -p "$RUN_ROOT"
    RUN_DIR="$RUN_ROOT/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    echo "$RUN_DIR" > "$RUN_ROOT/latest_run.txt"
    nohup bash "$SCRIPT_PATH" --run "$RUN_DIR" \
        > "$RUN_DIR/nohup.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/queue.pid"
    echo "Q5 conversion/training queue started"
    echo "RUN_DIR=$RUN_DIR"
    echo "QUEUE_PID=$(cat "$RUN_DIR/queue.pid")"
    exit 0
fi

if [[ ${1:-} != "--run" || -z ${2:-} ]]; then
    usage
    exit 2
fi

if (( GPU_COUNT < 1 )); then
    echo "COMPANY_30K_Q5_GPU_IDS must contain at least one GPU" >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || {
        echo "invalid GPU ID in COMPANY_30K_Q5_GPU_IDS: $gpu" >&2
        exit 2
    }
    [[ -z ${SEEN_GPUS[$gpu]+x} ]] || {
        echo "duplicate GPU ID in COMPANY_30K_Q5_GPU_IDS: $gpu" >&2
        exit 2
    }
    SEEN_GPUS[$gpu]=1
done
if (( TARGET_GLOBAL_BATCH < GPU_COUNT ||
      TARGET_GLOBAL_BATCH % GPU_COUNT != 0 )); then
    echo "global batch $TARGET_GLOBAL_BATCH must be a positive multiple of GPU count $GPU_COUNT" >&2
    exit 2
fi
ACCUMULATIVE_ITERS=$((TARGET_GLOBAL_BATCH / GPU_COUNT))
if [[ $WAIT_FOR_50M != 0 && $WAIT_FOR_50M != 1 ]]; then
    echo "COMPANY_30K_Q5_WAIT_FOR_50M must be 0 or 1" >&2
    exit 2
fi
if [[ ! $WORKERS_PER_GPU =~ ^[0-9]+$ ]]; then
    echo "COMPANY_30K_Q5_WORKERS_PER_GPU must be a non-negative integer" >&2
    exit 2
fi

RUN_DIR=$2
mkdir -p "$RUN_DIR" "$PROCESSED_ROOT"
exec 9>"$RUN_ROOT/queue.lock"
flock -n 9 || fail "another 30k Q5 conversion/training queue holds $RUN_ROOT/queue.lock"
cd "$REPO_ROOT"
trap 'status=$?; if (( status != 0 )); then touch "$RUN_DIR/FAILED"; fi' EXIT

for required in "$MANIFEST" "$CONFIG" \
        pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth; do
    [[ -f $required ]] || fail "missing required file: $required"
done

log "queue configured: GPUs=$GPU_IDS gpu_count=$GPU_COUNT effective_global_batch=$TARGET_GLOBAL_BATCH accumulation=$ACCUMULATIVE_ITERS train_lr=$TRAIN_LR workers_per_gpu=$WORKERS_PER_GPU wait_for_50m=$WAIT_FOR_50M processed_root=$PROCESSED_ROOT"
if [[ $WAIT_FOR_50M == 1 ]]; then
    log "waiting for any active 50m train/val conversion to finish"
    while pgrep -f 'convert_chengtech_20260818_collection.py.*processed_trainval_v1.*--splits train val' >/dev/null; do
        sleep 30
    done
else
    log "starting concurrently with active 50m conversion; disk throughput may decrease"
fi

log "starting isolated 350m train/val conversion"
if python -u tools/convert_chengtech_20260818_collection.py \
        --data-root "$SOURCE_ROOT" \
        --truth-root "$TRUTH_ROOT" \
        --out-root "$PROCESSED_ROOT" \
        --split-manifest "$MANIFEST" \
        --splits train val \
        --num-sweeps 3 \
        --point-cloud-range 0 -20 -3 350 20 3 \
        --max-empty-lidar-frames 32 \
        --keep-empty-lidar \
        --reuse-existing-lidar \
        --reuse-existing-radar \
        --resume-sequence-cache \
        > "$RUN_DIR/convert_350m.log" 2>&1; then
    touch "$RUN_DIR/CONVERSION_DONE"
    log "350m conversion completed"
else
    fail "350m conversion failed; inspect $RUN_DIR/convert_350m.log"
fi

for required in \
        "$PROCESSED_ROOT/custom_infos_train_sweep.pkl" \
        "$PROCESSED_ROOT/custom_infos_val_sweep.pkl" \
        "$PROCESSED_ROOT/conversion_summary.json"; do
    [[ -s $required ]] || fail "conversion output missing or empty: $required"
done

export RACFORMER_COMPANY_PROCESSED_ROOT="$PROCESSED_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

log "starting $GPU_COUNT-GPU NCCL preflight with P2P_DISABLE=$NCCL_P2P_DISABLE IB_DISABLE=$NCCL_IB_DISABLE"
if torchrun --nproc_per_node="$GPU_COUNT" --master_port=$((MASTER_PORT - 1)) \
        tools/test_nccl_collectives.py > "$RUN_DIR/nccl_test.log" 2>&1; then
    touch "$RUN_DIR/NCCL_TEST_DONE"
    log "$GPU_COUNT-GPU NCCL preflight passed"
else
    fail "NCCL preflight failed; training was not started"
fi

if pgrep -af 'train.py.*3dh_query_company_20260818_30k_q5_f4.py' \
        > "$RUN_DIR/existing_training_processes.txt"; then
    fail "another 30k Q5 training process is already running"
fi

TRAIN_OVERRIDES=(
    batch_size=1
    data.workers_per_gpu="$WORKERS_PER_GPU"
    optimizer.lr="$TRAIN_LR")
if (( ACCUMULATIVE_ITERS > 1 )); then
    TRAIN_OVERRIDES+=(
        optimizer_config.type=GradientCumulativeFp16OptimizerHook
        optimizer_config.cumulative_iters="$ACCUMULATIVE_ITERS")
fi
log "starting 36-epoch Q5 training; physical GPUs=$GPU_IDS effective_global_batch=$TARGET_GLOBAL_BATCH accumulation=$ACCUMULATIVE_ITERS lr=$TRAIN_LR"
if torchrun --nproc_per_node="$GPU_COUNT" --master_port="$MASTER_PORT" \
        train.py --config "$CONFIG" --override "${TRAIN_OVERRIDES[@]}" \
        > "$RUN_DIR/train_q5.log" 2>&1; then
    touch "$RUN_DIR/TRAINING_DONE"
    log "Q5 training completed successfully"
else
    fail "Q5 training failed; inspect $RUN_DIR/train_q5.log"
fi

touch "$RUN_DIR/ALL_DONE"
trap - EXIT
