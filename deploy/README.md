# RaCFormer Deployment Module

This directory contains the deployment-only path for the existing
`racformer_company_front_velocity_v2` checkpoint. It keeps the trained model
unchanged and replaces the validation dataset pipeline with runtime inputs.

## Current Contract

- Use only the left image from the stereo camera.
- Keep `num_cams=1` and `num_frames=8`.
- Images are `uint8` BGR arrays with configured source shape `480x640`.
- Radar points are `float32 [N, 7]` arrays in ego/LiDAR coordinates:
  `x, y, z, RCS, vx, vy, time_lag`.
- `lidar2img` projects ego/LiDAR points into the left image.
- Input order is newest frame followed by seven historical frames.
- Missing history is padded by repeating the oldest available frame.

The runtime path does not load LiDAR, generate `gt_depth`, use
`FrontViewFilter`, or construct OpenMMLab `DataContainer` objects.
The deployment config also sets the offline `val` and `test` dataset pipelines
to empty lists. `offline_demo.py` rejects a non-empty pipeline so these stages
cannot be reintroduced silently by passing the training config.

`radar_depth` and `radar_rcs` remain required. They are runtime radar-fusion
inputs generated from radar points, not LiDAR-derived ground-truth depth.

## Modules

- `input_schema.py`: runtime frame, batch, and detection contracts.
- `temporal_buffer.py`: newest-to-oldest eight-frame buffering.
- `preprocessing.py`: image transform, radar ROI, radar depth/RCS maps, and
  model batch construction.
- `pytorch_runner.py`: unchanged model and checkpoint loading plus inference.
- `postprocessing.py`: framework-independent NumPy outputs.
- `offline_demo.py`: server-only parity check using one existing dataset item.

ROS adapters will be added under `deploy/ros/` after ROS version, topics,
message fields, synchronization, and calibration ownership are confirmed.

## Server Validation

Run from the repository root in the existing `racformer_wp` environment.
First inspect the shared server instead of assuming a fixed physical GPU:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv
nvidia-smi pmon -c 1
```

Confirm process ownership or team reservations, then expose one selected GPU.
Inside the process it becomes logical `cuda:0`:

```bash
read -rp "Physical GPU index approved for this run: " GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -m deploy.offline_demo \
  --config configs/deploy/racformer_company_front_left_pytorch.py \
  --weights /mnt/diskNvme1/hyh/results/RaCFormer/racformer_company_front_velocity_v2/2026-07-07/18-46-40/epoch_36.pth \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --reference-pkl outputs/deploy_baseline/velocity_v2_epoch36_val_preds_gpu3.pkl \
  --out outputs/deploy_baseline/deploy_sample0.npz
```

The `gpu3` suffix in the existing reference filename only records which GPU
created that historical baseline; it does not require future runs to use GPU 3.
List available reference files with
`find outputs/deploy_baseline -maxdepth 1 -name '*.pkl' -type f` before running.

The command must report matching boxes, scores, and labels before this path is
used as the ROS integration base. Default parity tolerances are `5e-3` absolute
for box fields, `2e-4` absolute for scores, and zero relative; labels must be
identical. These account for the model's non-deterministic radar voxelization
and custom CUDA operators without hiding meaningful input differences. Override
them with `--box-atol`, `--score-atol`, and `--rtol` when needed. Real inference
validation is intentionally performed on the GPU server, not on development
machines.

## Deployment Profiling

After parity passes, profile the LiDAR-free path on an approved GPU:

```bash
read -rp "Physical GPU index approved for this run: " GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -m deploy.profile \
  --config configs/deploy/racformer_company_front_left_pytorch.py \
  --weights /mnt/diskNvme1/hyh/results/RaCFormer/racformer_company_front_velocity_v2/2026-07-07/18-46-40/epoch_36.pth \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --warmup 10 \
  --iters 50 \
  --out outputs/deploy_baseline/deploy_profile.txt
```

The report separates three latency definitions:

- Mode A includes offline image/radar file reads and is comparable to the old
  dataset-driven end-to-end profile.
- Mode B assumes synchronized sensor frames already exist and measures runtime
  preprocessing, transfer, model forward, and output transfer/parsing.
- Mode C reuses a prepared GPU batch and measures the model-only floor.

## Forward Submodule Profiling

Use the same approved-GPU procedure, then split the prepared-batch forward:

```bash
python -m deploy.profile_forward \
  --config configs/deploy/racformer_company_front_left_pytorch.py \
  --weights /mnt/diskNvme1/hyh/results/RaCFormer/racformer_company_front_velocity_v2/2026-07-07/18-46-40/epoch_36.pth \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --warmup 10 \
  --iters 50 \
  --out outputs/deploy_baseline/forward_profile.txt
```

The script first records an uninstrumented model baseline. It then temporarily
wraps existing methods with CUDA events to break down image features, eight
radar/LSS frames, the six shared decoder-layer calls, attention/sampling/mixing,
and bbox decode. It restores every wrapped method before exiting and does not
modify model source or checkpoint semantics.

## ONNX Compatibility Audit

The first TensorRT step exports a fixed batch-size-one, left-camera,
eight-frame FP32 graph. The graph ends at raw detector-head tensors
`all_cls_scores` and `all_bbox_preds`; variable-length bbox decode stays outside
the graph. Existing runtime cache optimizations are intentionally disabled for
this first stateless export. Training-only gradient checkpointing is disabled
after checkpoint loading because it does not change eval numerics and the
PyTorch 2.0 legacy ONNX tracer cannot trace it reliably.

MMCV radar voxelization also stays outside the graph: each frame supplies
dynamic `radar_voxels`, `radar_num_points`, and batch-padded `radar_coors`.
Opaque MSMV and deformable-attention CUDA autograd functions switch to their
traceable PyTorch implementations during export so sensor-dependent values are
not frozen into the ONNX graph.

The exporter records a two-forward raw-output comparison. It is informational
by default because radar voxelization and custom CUDA kernels can differ across
independent runs. Add `--strict-boundary-check` only when deterministic kernels
have been established and exceeding `--boundary-atol` should stop export.

Check the server environment, then attempt a standard ONNX export:

```bash
python -c "import torch, onnx; print(torch.__version__, onnx.__version__)"

python -m deploy.export_onnx \
  --config configs/deploy/racformer_company_front_left_pytorch.py \
  --weights /mnt/diskNvme1/hyh/results/RaCFormer/racformer_company_front_velocity_v2/2026-07-07/18-46-40/epoch_36.pth \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --opset 17 \
  --out outputs/deploy_onnx/racformer_raw_fp32.onnx \
  --report outputs/deploy_onnx/export_standard.txt
```

A failed standard export is useful: `export_standard.txt` records the first
unsupported operation and traceback. To retain unsupported operators in an
audit graph where PyTorch permits it, rerun with `--fallthrough` and a distinct
output path:

```bash
python -m deploy.export_onnx \
  --config configs/deploy/racformer_company_front_left_pytorch.py \
  --weights /mnt/diskNvme1/hyh/results/RaCFormer/racformer_company_front_velocity_v2/2026-07-07/18-46-40/epoch_36.pth \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --opset 17 \
  --fallthrough \
  --out outputs/deploy_onnx/racformer_raw_fp32_fallthrough.onnx \
  --report outputs/deploy_onnx/export_fallthrough.txt

python -m deploy.tensorrt.audit_onnx \
  --onnx outputs/deploy_onnx/racformer_raw_fp32_fallthrough.onnx \
  --out outputs/deploy_onnx/onnx_operator_audit.txt
```

If the standard ONNX export succeeds, use that graph for the audit instead.
Do not build or benchmark a TensorRT engine until the ONNX checker passes and
the custom/non-supported operator list has been reviewed.

## Synchronization

Deployment code lives in the same Git repository as training code. Update a
server checkout with:

```bash
git pull origin main
```

Checkpoints, TensorRT engines, datasets, calibration secrets, and files under
`outputs/` must remain outside Git.
