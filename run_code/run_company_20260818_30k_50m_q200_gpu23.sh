#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
RUN_ROOT=${COMPANY_30K_Q200_RUN_ROOT:-/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_30k_50m_q200_f4}
TRAINVAL_ROOT=${RACFORMER_COMPANY_TRAINVAL_ROOT:-/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1}
TEST_ROOT=${RACFORMER_COMPANY_TEST_ROOT:-/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3}
CONFIG=configs/racformer_company_front_50m_q200_f4_30k_train.py
GPU_IDS=${COMPANY_30K_Q200_GPU_IDS:-2,3}
MASTER_PORT=${COMPANY_30K_Q200_MASTER_PORT:-30033}
WORKERS_PER_GPU=${COMPANY_30K_Q200_WORKERS_PER_GPU:-1}
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
GPU_COUNT=${#GPU_LIST[@]}
TARGET_GLOBAL_BATCH=4

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
    echo "50m Q200 training queue started"
    echo "RUN_DIR=$RUN_DIR"
    echo "QUEUE_PID=$(cat "$RUN_DIR/queue.pid")"
    exit 0
fi

if [[ ${1:-} != "--run" || -z ${2:-} ]]; then
    usage
    exit 2
fi
if (( GPU_COUNT != 2 )); then
    echo "this launcher requires exactly two GPUs; got $GPU_IDS" >&2
    exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || {
        echo "invalid GPU ID: $gpu" >&2
        exit 2
    }
done

RUN_DIR=$2
mkdir -p "$RUN_DIR"
exec 9>"$RUN_ROOT/queue.lock"
flock -n 9 || fail "another 50m Q200 training queue holds $RUN_ROOT/queue.lock"
cd "$REPO_ROOT"
trap 'status=$?; if (( status != 0 )); then touch "$RUN_DIR/FAILED"; fi' EXIT

for required in \
        "$TRAINVAL_ROOT/custom_infos_train_sweep.pkl" \
        "$TRAINVAL_ROOT/custom_infos_val_sweep.pkl" \
        "$TRAINVAL_ROOT/conversion_summary.json" \
        "$TEST_ROOT/custom_infos_test_sweep.pkl" \
        "$CONFIG" \
        pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth; do
    [[ -s $required ]] || fail "missing or empty required file: $required"
done

export RACFORMER_COMPANY_TRAINVAL_ROOT="$TRAINVAL_ROOT"
export RACFORMER_COMPANY_TEST_ROOT="$TEST_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

log "configured GPUs=$GPU_IDS effective_global_batch=$TARGET_GLOBAL_BATCH workers_per_gpu=$WORKERS_PER_GPU"
python tools/smoke_company_training_data.py --config "$CONFIG" \
    > "$RUN_DIR/data_smoke.log" 2>&1 || \
    fail "data smoke failed; inspect $RUN_DIR/data_smoke.log"
touch "$RUN_DIR/DATA_SMOKE_DONE"

torchrun --nproc_per_node=2 --master_port=$((MASTER_PORT - 1)) \
    tools/test_nccl_collectives.py > "$RUN_DIR/nccl_test.log" 2>&1 || \
    fail "two-GPU NCCL preflight failed; inspect $RUN_DIR/nccl_test.log"
touch "$RUN_DIR/NCCL_TEST_DONE"

if pgrep -af 'train.py.*racformer_company_front_50m_q200_f4_30k_train.py' \
        > "$RUN_DIR/existing_training_processes.txt"; then
    fail "another 50m Q200 training process is already running"
fi

log "starting 36-epoch 50m Q200 F4 training"
if torchrun --nproc_per_node=2 --master_port="$MASTER_PORT" \
        train.py --config "$CONFIG" --override \
        batch_size=1 data.workers_per_gpu="$WORKERS_PER_GPU" \
        optimizer_config.type=GradientCumulativeFp16OptimizerHook \
        optimizer_config.cumulative_iters=2 \
        > "$RUN_DIR/train.log" 2>&1; then
    touch "$RUN_DIR/TRAINING_DONE"
    log "50m Q200 training completed successfully"
else
    fail "training failed; inspect $RUN_DIR/train.log"
fi

touch "$RUN_DIR/ALL_DONE"
trap - EXIT
