#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
CONFIG=configs/racformer_company_front_50m_q200_f4_30k_tune.py
RESULT_ROOT_DEFAULT=/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning
TEST_ROOT_DEFAULT=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3

usage() {
    cat <<'EOF'
Usage:
  run_company_50m_tuning.sh --profile PROFILE --data-root DIR [--train-ann PKL] [--val-ann PKL] [--gpus 0,1] [--port 30100] --background
  run_company_50m_tuning.sh --run RUN_DIR

Profiles:
  overfit_car_f1_noflip    1 class, 1 frame, no flip, 20 epochs
  overfit_car_f4_noflip    1 class, 4 frames, no flip, 20 epochs
  proxy_all_f4_flip        current 10-class baseline
  proxy_main3_f4_noflip    main3, 4 frames, no flip
  proxy_main3_f1_noflip    main3, 1 frame, no flip
  proxy_main3_f4_flip      main3, 4 frames, flip
  proxy_main3_f4_z         no flip, stronger cz/height and bbox loss
  proxy_main3_f4_lr_half   no flip, lr=2e-4
  proxy_main3_f4_lr_double no flip, lr=8e-4
  proxy_main3_f4_fov120    main3, 4 frames, 120-degree physical FOV
  full_main3_f4_fov120_p15_e8
                           full train set, FOV120, distance power 1.5, 8 epochs
  full_main3_f4_fov120_p15_e20
                           blocked dev split, FOV120, distance power 1.5, 20 epochs
EOF
}

set_profile() {
    export RACFORMER_TUNE_CLASSES=car,truck,bicycle
    export RACFORMER_TUNE_NUM_FRAMES=4
    export RACFORMER_TUNE_RAND_FLIP=0
    export RACFORMER_TUNE_EPOCHS=12
    export RACFORMER_TUNE_EVAL_INTERVAL=2
    export RACFORMER_TUNE_LR=4e-4
    export RACFORMER_TUNE_BBOX_LOSS_WEIGHT=0.25
    export RACFORMER_TUNE_CODE_WEIGHTS=2,2,1,1,1,1,1,1,1,1
    export RACFORMER_TUNE_HORIZONTAL_FOV_DEG=0
    export RACFORMER_TUNE_QUERY_DISTANCE_POWER=1.0
    export RACFORMER_TUNE_CHECKPOINT_INTERVAL=2
    export RACFORMER_TUNE_MAX_KEEP_CKPTS=1
    case "$1" in
        overfit_car_f1_noflip)
            export RACFORMER_TUNE_CLASSES=car
            export RACFORMER_TUNE_NUM_FRAMES=1
            export RACFORMER_TUNE_EPOCHS=20
            ;;
        overfit_car_f4_noflip)
            export RACFORMER_TUNE_CLASSES=car
            export RACFORMER_TUNE_EPOCHS=20
            ;;
        proxy_all_f4_flip)
            export RACFORMER_TUNE_CLASSES=car,truck,trailer,bus,construction_vehicle,bicycle,motorcycle,pedestrian,traffic_cone,barrier
            export RACFORMER_TUNE_RAND_FLIP=1
            ;;
        proxy_main3_f4_noflip) ;;
        proxy_main3_f1_noflip)
            export RACFORMER_TUNE_NUM_FRAMES=1
            ;;
        proxy_main3_f4_flip)
            export RACFORMER_TUNE_RAND_FLIP=1
            ;;
        proxy_main3_f4_z)
            # normalize_bbox order: cx,cy,log(w),log(l),cz,log(h),sin,cos,vx,vy
            export RACFORMER_TUNE_BBOX_LOSS_WEIGHT=0.5
            export RACFORMER_TUNE_CODE_WEIGHTS=2,2,1,1,2,2,1,1,1,1
            ;;
        proxy_main3_f4_lr_half)
            export RACFORMER_TUNE_LR=2e-4
            ;;
        proxy_main3_f4_lr_double)
            export RACFORMER_TUNE_LR=8e-4
            ;;
        proxy_main3_f4_fov120)
            export RACFORMER_TUNE_HORIZONTAL_FOV_DEG=120
            ;;
        full_main3_f4_fov120_p15_e8)
            export RACFORMER_TUNE_HORIZONTAL_FOV_DEG=120
            export RACFORMER_TUNE_QUERY_DISTANCE_POWER=1.5
            export RACFORMER_TUNE_EPOCHS=8
            ;;
        full_main3_f4_fov120_p15_e20)
            export RACFORMER_TUNE_HORIZONTAL_FOV_DEG=120
            export RACFORMER_TUNE_QUERY_DISTANCE_POWER=1.5
            export RACFORMER_TUNE_EPOCHS=20
            export RACFORMER_TUNE_CHECKPOINT_INTERVAL=4
            export RACFORMER_TUNE_MAX_KEEP_CKPTS=4
            ;;
        *) echo "unknown profile: $1" >&2; usage; exit 2 ;;
    esac
}

if [[ ${1:-} == --run ]]; then
    [[ $# == 2 ]] || { usage; exit 2; }
    RUN_DIR=$2
    # shellcheck disable=SC1090
    source "$RUN_DIR/environment.sh"
    cd "$REPO_ROOT"
    log() { echo "$(date --iso-8601=seconds) $*" | tee -a "$RUN_DIR/queue.log"; }
    exec 9>"$RUN_DIR/run.lock"
    flock -n 9 || { echo "another process owns $RUN_DIR/run.lock"; exit 3; }
    exec 8>"$TUNING_LOCK"
    flock -n 8 || { log "FAILED: another tuning run is active"; exit 3; }
    trap 'status=$?; (( status == 0 )) || touch "$RUN_DIR/FAILED"' EXIT
    for path in \
        "$RACFORMER_TUNE_TRAIN_ANN_FILE" \
        "$RACFORMER_TUNE_VAL_ANN_FILE" \
        "$RACFORMER_COMPANY_TEST_ROOT/custom_infos_test_sweep.pkl" \
        "$CONFIG" \
        pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth; do
        [[ -s $path ]] || { echo "missing required file: $path"; exit 3; }
    done
    log "resolving config for profile=$PROFILE GPUs=$CUDA_VISIBLE_DEVICES"
    # MMCV 1.6 calls yapf.FormatCode(..., verify=True), but recent YAPF
    # releases removed that argument.  Dump the resolved ConfigDict directly
    # so experiment startup does not depend on the installed YAPF version.
    python -c "from mmcv import Config; from pprint import pformat; c=Config.fromfile('$CONFIG'); open('$RUN_DIR/resolved_config.py','w').write(pformat(c._cfg_dict.to_dict(), width=120, sort_dicts=False) + '\\n')"
    python tools/smoke_company_training_data.py --config "$CONFIG" --allow-no-empty \
        > "$RUN_DIR/data_smoke.log" 2>&1
    log "data smoke passed"
    torchrun --nproc_per_node="$GPU_COUNT" --master_port=$((MASTER_PORT - 1)) \
        tools/test_nccl_collectives.py > "$RUN_DIR/nccl_test.log" 2>&1
    log "NCCL preflight passed; starting training"
    torchrun --nproc_per_node="$GPU_COUNT" --master_port="$MASTER_PORT" \
        train.py --config "$CONFIG" --work-dir "$RUN_DIR/model" --override \
        batch_size=1 data.workers_per_gpu="$WORKERS_PER_GPU" \
        optimizer_config.type=GradientCumulativeFp16OptimizerHook \
        optimizer_config.cumulative_iters="$CUMULATIVE_ITERS" \
        > "$RUN_DIR/train.log" 2>&1
    touch "$RUN_DIR/ALL_DONE"
    log "training completed"
    trap - EXIT
    exit 0
fi

PROFILE=
DATA_ROOT=
TRAIN_ANN=
VAL_ANN=
GPU_IDS=0,1
MASTER_PORT=30100
WORKERS_PER_GPU=1
BACKGROUND=0
RESULT_ROOT=${COMPANY_TUNING_RESULT_ROOT:-$RESULT_ROOT_DEFAULT}
TEST_ROOT=${RACFORMER_COMPANY_TEST_ROOT:-$TEST_ROOT_DEFAULT}
while (($#)); do
    case "$1" in
        --profile) PROFILE=$2; shift 2 ;;
        --data-root) DATA_ROOT=$2; shift 2 ;;
        --train-ann) TRAIN_ANN=$2; shift 2 ;;
        --val-ann) VAL_ANN=$2; shift 2 ;;
        --gpus) GPU_IDS=$2; shift 2 ;;
        --port) MASTER_PORT=$2; shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU=$2; shift 2 ;;
        --background) BACKGROUND=1; shift ;;
        *) usage; exit 2 ;;
    esac
done
[[ -n $PROFILE && -n $DATA_ROOT && $BACKGROUND == 1 ]] || { usage; exit 2; }
set_profile "$PROFILE"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
GPU_COUNT=${#GPU_LIST[@]}
case "$GPU_COUNT" in
    1) CUMULATIVE_ITERS=4 ;;
    2) CUMULATIVE_ITERS=2 ;;
    4) CUMULATIVE_ITERS=1 ;;
    *) echo "tuning launcher requires 1, 2, or 4 GPUs" >&2; exit 2 ;;
esac
RUN_DIR="$RESULT_ROOT/$PROFILE/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
mkdir -p "$RESULT_ROOT/$PROFILE"
echo "$RUN_DIR" > "$RESULT_ROOT/$PROFILE/latest_run.txt"
export RACFORMER_TUNE_DATA_ROOT=$(readlink -f "$DATA_ROOT")
export RACFORMER_TUNE_TRAIN_ANN_FILE=$(readlink -f "${TRAIN_ANN:-$DATA_ROOT/custom_infos_train_sweep.pkl}")
export RACFORMER_TUNE_VAL_ANN_FILE=$(readlink -f "${VAL_ANN:-$DATA_ROOT/custom_infos_val_sweep.pkl}")
export RACFORMER_COMPANY_TEST_ROOT=$(readlink -f "$TEST_ROOT")
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export GPU_COUNT MASTER_PORT WORKERS_PER_GPU CUMULATIVE_ITERS
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PROFILE
export TUNING_LOCK="$RESULT_ROOT/tuning.lock"
export -p | grep -E 'RACFORMER_|CUDA_VISIBLE_DEVICES|GPU_COUNT|MASTER_PORT|WORKERS_PER_GPU|CUMULATIVE_ITERS|NCCL_|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|PROFILE' \
    > "$RUN_DIR/environment.sh"
setsid bash "$SCRIPT_PATH" --run "$RUN_DIR" \
    > "$RUN_DIR/nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid"
echo "tuning run started: $PROFILE"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid")"
