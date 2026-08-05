# Four-Frame TensorRT Bring-Up

This procedure deploys the trained company-front four-frame checkpoint while
keeping the proven split architecture:

```text
FP16 precompute frontend
  + FP32 fixture initial query
  + FP32 recurrent decoder enqueued six times
```

Reducing temporal frames does not make five decoder iterations valid. The
four-frame checkpoint still uses all six trained iterations.

## Paths

Run PyTorch export in the server's native `racformer_wp` environment. Run
TensorRT 8.5 parsing, building, and validation in the
`racformer_trt85_l20` container.

```bash
cd ~/hyh/RaCFormer
conda activate racformer_wp

CONFIG=configs/deploy/racformer_company_front_left_pytorch_f4.py
CKPT=outputs/racformer_company_front_velocity_v2_f4/2026-08-05/11-17-49/epoch_36.pth
OUT=outputs/deploy_onnx
```

The deployment config inherits the exact four-frame training model, changes
the validation/test pipelines to empty lists, and declares the left-camera
four-frame runtime contract.

## Determine Static Radar Capacity

TensorRT 8.5 requires the proven static radar padding path. The successful
eight-frame deployment used **1024 voxel slots per frame**. This is a per-frame
capacity, so the four-frame graph contains 4096 padded slots in total rather
than the old graph's 8192.

```bash
grep -R -h 'static radar voxel slots:' outputs/deploy_onnx \
  --include='*.txt' | sort -u
```

Use the same validated per-frame capacity for initial four-frame bring-up:

```bash
STATIC_RADAR_VOXELS=1024
```

Do not reduce this value from the sample-0 count alone. Audit voxel counts over
the intended deployment dataset before choosing a smaller production capacity.

## 1. Export Four-Frame Model Fixture

```bash
python -m deploy.export_onnx \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --device cuda:0 \
  --split val \
  --sample-index 0 \
  --opset 17 \
  --mixing-chunk-size 32768 \
  --msmv-plugin \
  --single-camera-projection-plugin \
  --fixed-view-geometry \
  --tensorrt-85-compat \
  --static-radar-voxels "$STATIC_RADAR_VOXELS" \
  --out "$OUT/racformer_f4_raw_trt85.onnx" \
  --fixture "$OUT/racformer_f4_model_sample0.npz" \
  --report "$OUT/export_f4_model_trt85.txt"
```

The report must show four radar frames and `status: SUCCESS` before continuing.

## 2. Export FP16-Candidate Precompute Frontend

```bash
python -m deploy.export_frontend_onnx \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$OUT/racformer_f4_model_sample0.npz" \
  --device cuda:0 \
  --opset 17 \
  --precompute-bev-values \
  --out "$OUT/racformer_f4_frontend_precompute_v2_trt85.onnx" \
  --fixture "$OUT/racformer_f4_frontend_precompute_sample0.npz" \
  --report "$OUT/export_f4_frontend_precompute_v2_trt85.txt"
```

The frontend fixture is the shared fixture used by the complete split-pipeline
validator. Learned initial query tensors remain FP32 in this fixture.

## 3. Export FP32 Recurrent Decoder

```bash
python -m deploy.export_decoder_recurrent_layer \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$OUT/racformer_f4_model_sample0.npz" \
  --device cuda:0 \
  --opset 17 \
  --precompute-bev-values \
  --out "$OUT/racformer_f4_decoder_precompute_v2_trt85.onnx" \
  --report "$OUT/export_f4_decoder_precompute_v2_trt85.txt"
```

No separate decoder fixture is required. The exporter can optionally save one
with `--fixture`, but production validation uses the frontend fixture.

## 4. Build In The TensorRT 8.5 Container

```bash
docker start racformer_trt85_l20
docker exec -it racformer_trt85_l20 bash

cd /workspace/RaCFormer
PLUGIN=/workspace/RaCFormer/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so
ONNX=/workspace/outputs/deploy_onnx
TRT=/workspace/outputs/deploy_tensorrt
```

Pull or mount the commit containing the four-frame deployment changes before
running these commands.

Parse both graphs:

```bash
python -m deploy.tensorrt.parse_onnx \
  --onnx "$ONNX/racformer_f4_frontend_precompute_v2_trt85.onnx" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_f4_frontend_precompute_v2_trt852.txt"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$ONNX/racformer_f4_decoder_precompute_v2_trt85.onnx" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_f4_decoder_precompute_v2_trt852.txt"
```

Both reports must show `status: PASS`, `parser errors: 0`, and
`zero-dimension execution tensors: 0` before building either engine.

Build the frontend with FP16 tactics and the decoder in strict FP32:

```bash
python -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_f4_frontend_precompute_v2_trt85.onnx" \
  --engine "$TRT/racformer_f4_frontend_precompute_v2_trt852_fp16.engine" \
  --plugin "$PLUGIN" \
  --fp16 \
  --out "$TRT/build_f4_frontend_precompute_v2_trt852_fp16.txt"

python -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_f4_decoder_precompute_v2_trt85.onnx" \
  --engine "$TRT/racformer_f4_decoder_precompute_v2_trt852_fp32.engine" \
  --plugin "$PLUGIN" \
  --out "$TRT/build_f4_decoder_precompute_v2_trt852_fp32.txt"
```

Do not pass `--fp16` to the decoder build until a four-frame FP16-aware
checkpoint has separately passed decoded validation.

## 5. Validate And Benchmark

```bash
python -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine "$TRT/racformer_f4_frontend_precompute_v2_trt852_fp16.engine" \
  --decoder-engine "$TRT/racformer_f4_decoder_precompute_v2_trt852_fp32.engine" \
  --fixture "$ONNX/racformer_f4_frontend_precompute_sample0.npz" \
  --plugin "$PLUGIN" \
  --initial-query-from-fixture \
  --accept-decoded-match \
  --warmup 5 \
  --iters 20 \
  --profile-stages \
  --out "$TRT/validate_f4_frontend_fp16_decoder_fp32_trt852.txt"
```

Start with the standard `atol=0.006`. If decoded boxes alone exceed that
tolerance while count, labels, and scores remain stable, report the actual
maximum error before deciding whether the Nano-specific 3 cm deployment
tolerance remains appropriate.

Acceptance still requires decoded detection agreement. Parser success, engine
build success, or raw intermediate parity alone is insufficient.
