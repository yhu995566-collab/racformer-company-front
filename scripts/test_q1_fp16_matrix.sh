#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != "l20" ]]; then
  echo "usage: $0 l20" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
ONNX="$OUTPUT_ROOT/deploy_onnx_q1"
TRT="$OUTPUT_ROOT/deploy_tensorrt_q1"
PLUGIN="${PLUGIN:-$REPO_ROOT/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so}"
WORKSPACE_GB="${WORKSPACE_GB:-8}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-20}"
ATOL="${ATOL:-0.03}"

IMAGE_ENGINE="$TRT/3dh_query_q1_frontend_image_lss_trt852_l20_fp16.engine"
RADAR_FP32_ENGINE="$TRT/3dh_query_q1_frontend_radar_trt852_l20_fp32.engine"
DECODER_FP32_ENGINE="$TRT/3dh_query_q1_decoder_precompute_v2_trt852_l20_fp32.engine"
RADAR_ONNX="$ONNX/3dh_query_q1_frontend_radar_trt85.onnx"
DECODER_ONNX="$ONNX/3dh_query_q1_decoder_precompute_v2_trt85.onnx"
FIXTURE="$ONNX/3dh_query_q1_frontend_precompute_sample0.npz"

for path in \
  "$IMAGE_ENGINE" "$RADAR_FP32_ENGINE" "$DECODER_FP32_ENGINE" \
  "$RADAR_ONNX" "$DECODER_ONNX" "$FIXTURE" "$PLUGIN"; do
  if [[ ! -f "$path" ]]; then
    echo "required baseline file not found: $path" >&2
    echo "run scripts/deploy_q1_tensorrt.sh l20 successfully first" >&2
    exit 1
  fi
done

RADAR_FP16_ENGINE="$TRT/3dh_query_q1_frontend_radar_trt852_l20_fp16.engine"
DECODER_FP16_ENGINE="$TRT/3dh_query_q1_decoder_precompute_v2_trt852_l20_fp16.engine"

SUMMARY="$TRT/validate_3dh_query_q1_fp16_matrix_trt852_l20_summary.txt"
{
  echo "=== Q1 TensorRT 8.5 FP16 experiment matrix ==="
  echo "repository: $REPO_ROOT"
  echo "commit: $(git rev-parse HEAD)"
  echo "TensorRT output: $TRT"
  echo "atol: $ATOL"
  echo "warmup: $WARMUP"
  echo "iterations: $ITERS"
  echo "FP16 means TensorRT may select FP16 tactics; graph/plugin constraints may keep individual tensors or layers in FP32."
} > "$SUMMARY"

set +e
python -m deploy.tensorrt.build_engine \
  --onnx "$RADAR_ONNX" \
  --engine "$RADAR_FP16_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb "$WORKSPACE_GB" \
  --fp16 \
  --out "$TRT/build_3dh_query_q1_frontend_radar_trt852_l20_fp16.txt"
RADAR_BUILD_STATUS=$?

python -m deploy.tensorrt.build_engine \
  --onnx "$DECODER_ONNX" \
  --engine "$DECODER_FP16_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb "$WORKSPACE_GB" \
  --fp16 \
  --out "$TRT/build_3dh_query_q1_decoder_trt852_l20_fp16.txt"
DECODER_BUILD_STATUS=$?
set -e

{
  echo "radar FP16 build status: $RADAR_BUILD_STATUS"
  echo "decoder FP16 build status: $DECODER_BUILD_STATUS"
} | tee -a "$SUMMARY"

run_validation() {
  local label="$1"
  local radar_engine="$2"
  local decoder_engine="$3"
  local report="$TRT/validate_3dh_query_q1_${label}_trt852_l20.txt"
  local status

  echo "=== Running $label ==="
  set +e
  python -m deploy.tensorrt.validate_frontend_decoder_numpy \
    --frontend-engine "$IMAGE_ENGINE" \
    --radar-frontend-engine "$radar_engine" \
    --decoder-engine "$decoder_engine" \
    --fixture "$FIXTURE" \
    --plugin "$PLUGIN" \
    --initial-query-from-fixture \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --atol "$ATOL" \
    --accept-decoded-match \
    --profile-stages \
    --out "$report"
  status=$?
  set -e

  {
    echo
    echo "--- $label ---"
    echo "radar engine: $radar_engine"
    echo "decoder engine: $decoder_engine"
    echo "validator exit status: $status"
    grep -E \
      'actual/reference detection count|boxes close:|scores close:|labels equal:|decoded comparison passed:|end-to-end engine GPU latency:|frontend GPU latency:|radar frontend GPU latency:|recurrent decoder GPU latency:|resident CUDA memory delta:|deployment acceptance passed:|status:' \
      "$report" || true
  } | tee -a "$SUMMARY"
}

# Isolate each newly enabled FP16 branch before testing both together.
if [[ "$DECODER_BUILD_STATUS" -eq 0 ]]; then
  run_validation decoder_fp16_only "$RADAR_FP32_ENGINE" "$DECODER_FP16_ENGINE"
else
  echo "decoder_fp16_only: SKIPPED (decoder FP16 build failed)" | tee -a "$SUMMARY"
fi
if [[ "$RADAR_BUILD_STATUS" -eq 0 ]]; then
  run_validation radar_fp16_only "$RADAR_FP16_ENGINE" "$DECODER_FP32_ENGINE"
else
  echo "radar_fp16_only: SKIPPED (radar FP16 build failed)" | tee -a "$SUMMARY"
fi
if [[ "$RADAR_BUILD_STATUS" -eq 0 && "$DECODER_BUILD_STATUS" -eq 0 ]]; then
  run_validation radar_decoder_fp16 "$RADAR_FP16_ENGINE" "$DECODER_FP16_ENGINE"
else
  echo "radar_decoder_fp16: SKIPPED (one or more FP16 builds failed)" | tee -a "$SUMMARY"
fi

echo "Q1 FP16 experiment matrix completed: $SUMMARY"
