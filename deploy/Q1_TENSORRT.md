# Q1 350 m TensorRT Deployment

Q1 is the four-frame, uniform `30 x 30` query baseline over the 350 m forward
range. It still has 900 queries and six recurrent decoder iterations. The
radar candidate scorer exists in the repository but is not connected to Q1.

Q1 cannot reuse any 200 m fixture, ONNX graph, or engine. It reuses only the
three-engine deployment architecture, export/runtime code, plugin source, and
six-enqueue recurrent scheduling logic.

## 1. Synchronize The Exact Code

Use branch `3dh-query-stage1-radar-candidate-recall` at commit `d4572ea` or a
later commit containing the Q1 deployment scripts on the export server,
TensorRT 8.5 container mount, and Nano:

```bash
git checkout 3dh-query-stage1-radar-candidate-recall
git pull --ff-only origin 3dh-query-stage1-radar-candidate-recall
git rev-parse HEAD
git status --short
```

Do not copy only the Q1 config and checkpoint to an older Nano checkout. The
350 m coordinate conversion depends on the updated model source.

The Q1 deployment config is:

```text
configs/deploy/3dh_query_q1_left_pytorch_f4.py
```

It inherits `configs/3dh_query_q1.py`, clears validation/test pipelines, and
declares four synchronized left-camera/radar frames.

## 2. Native PyTorch Export

Run outside the TensorRT container in the training/export environment:

```bash
cd /path/to/3DH-Query
conda activate racformer_wp

CKPT=/absolute/path/to/q1/epoch_36.pth
CUDA_VISIBLE_DEVICES=<approved_gpu> \
STATIC_RADAR_VOXELS=1024 \
bash scripts/deploy_q1_export.sh "$CKPT"
```

Optional variables are `OUTPUT_ROOT`, `DEVICE`, `SAMPLE_INDEX`, and
`STATIC_RADAR_VOXELS`. Defaults are the repository `outputs` directory,
`cuda:0`, sample 0, and 1024 slots per frame.

Successful export creates these required deployment artifacts under
`outputs/deploy_onnx_q1`:

```text
3dh_query_q1_frontend_precompute_v2_trt85.onnx
3dh_query_q1_decoder_precompute_v2_trt85.onnx
3dh_query_q1_frontend_precompute_sample0.npz
```

The raw model ONNX and model fixture are retained as diagnostic artifacts.

## 3. Audit Radar Capacity

The initial 1024 slots are inherited from the 200 m deployment, not yet proven
for the 350 m ROI. Audit every intended split before declaring the engine
production-ready:

```bash
python -m deploy.audit_radar_voxel_capacity \
  --config configs/deploy/3dh_query_q1_left_pytorch_f4.py \
  --weights "$CKPT" \
  --split val \
  --capacity 1024 \
  --device cuda:0 \
  --fail-if-exceeds \
  --out outputs/deploy_onnx_q1/radar_voxel_capacity_val_1024.txt
```

Repeat with `--split train` and `--split test` when those splits represent the
deployment distribution. Use `--max-samples 20` only for a smoke test; omit it
for the production audit. If any frame exceeds capacity, select a larger fixed
capacity with safety margin and rerun the entire export/build process.

## 4. L20 TensorRT 8.5 Build And Validation

The existing container must mount this Q1 checkout, not an older RaCFormer
checkout. Verify the mount and commit before building:

```bash
docker start racformer_trt85_l20
docker exec -it racformer_trt85_l20 bash

cd /workspace/<mounted-q1-repository>
git rev-parse HEAD
ls /workspace/outputs/deploy_onnx_q1
```

Run the complete subgraph extraction, parser audit, three engine builds, and
decoded validation:

```bash
OUTPUT_ROOT=/workspace/outputs \
WORKSPACE_GB=8 \
WARMUP=20 \
ITERS=20 \
ATOL=0.03 \
bash scripts/deploy_q1_tensorrt.sh l20
```

The generated architecture is:

```text
FP16 image/LSS frontend
  + FP32 radar frontend
  + FP32 fixture initial query
  + FP32 recurrent decoder x 6
```

Do not add `--fp16` to the radar or decoder build. Acceptance requires decoded
count, boxes, scores, and labels to match; parser or build success is not
sufficient.

## 5. Transfer To Nano

Do not transfer L20 engines or its x86 plugin. Transfer these architecture-
independent artifacts from `deploy_onnx_q1`:

```text
3dh_query_q1_frontend_image_lss_trt85.onnx
3dh_query_q1_frontend_radar_trt85.onnx
3dh_query_q1_decoder_precompute_v2_trt85.onnx
3dh_query_q1_frontend_precompute_sample0.npz
```

Put them under the Nano checkout's `outputs/deploy_onnx_q1`. Synchronize the
same Git commit and build/load the Orin plugin locally at:

```text
build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so
```

## 6. Nano Build And Validation

From the synchronized Nano repository root:

```bash
export LD_LIBRARY_PATH="$PWD/build/tensorrt_plugins_orin:/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

OUTPUT_ROOT="$PWD/outputs" \
WORKSPACE_GB=8 \
WARMUP=20 \
ITERS=20 \
ATOL=0.03 \
bash scripts/deploy_q1_tensorrt.sh orin
```

The script accepts the already extracted frontend ONNX files, so the large
full frontend ONNX does not need to be copied to Nano. All engines are rebuilt
locally with the Orin plugin.

Record correctness, stage latency, total engine latency, resident CUDA memory,
and the per-stage memory breakdown. The 350 m BEV has 56,000 cells per frame,
1.75 times the old 32,000-cell grid, so old Nano latency and memory numbers are
not valid for Q1.
