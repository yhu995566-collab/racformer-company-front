#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "l20" && "$1" != "orin" ) ]]; then
  echo "usage: $0 l20|orin" >&2
  exit 2
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT"
elif [[ "$TARGET" == "l20" && -d /workspace/outputs ]]; then
  OUTPUT_ROOT=/workspace/outputs
else
  OUTPUT_ROOT="$REPO_ROOT/outputs"
fi

ONNX="$OUTPUT_ROOT/deploy_onnx_q1"
TRT="$OUTPUT_ROOT/deploy_tensorrt_q1"
WORKSPACE_GB="${WORKSPACE_GB:-8}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-20}"
ATOL="${ATOL:-0.03}"

if [[ "$TARGET" == "l20" ]]; then
  ENGINE_TAG=trt852_l20
  DEFAULT_PLUGIN="$REPO_ROOT/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so"
else
  ENGINE_TAG=trt852_orin
  DEFAULT_PLUGIN="$REPO_ROOT/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"
fi
PLUGIN="${PLUGIN:-$DEFAULT_PLUGIN}"

FULL_FRONTEND="$ONNX/3dh_query_q1_frontend_precompute_v2_trt85.onnx"
IMAGE_LSS_ONNX="$ONNX/3dh_query_q1_frontend_image_lss_trt85.onnx"
RADAR_ONNX="$ONNX/3dh_query_q1_frontend_radar_trt85.onnx"
DECODER_ONNX="$ONNX/3dh_query_q1_decoder_precompute_v2_trt85.onnx"
FIXTURE="$ONNX/3dh_query_q1_frontend_precompute_sample0.npz"

for path in "$DECODER_ONNX" "$FIXTURE" "$PLUGIN"; do
  if [[ ! -f "$path" ]]; then
    echo "required file not found: $path" >&2
    exit 1
  fi
done

if [[ -f "$FULL_FRONTEND" ]]; then
  EXTRACT_FRONTENDS=1
elif [[ -f "$IMAGE_LSS_ONNX" && -f "$RADAR_ONNX" ]]; then
  EXTRACT_FRONTENDS=0
else
  echo "provide either the full frontend ONNX or both extracted frontend ONNX files" >&2
  exit 1
fi

mkdir -p "$TRT"

echo "target: $TARGET"
echo "repository: $REPO_ROOT"
echo "commit: $(git rev-parse HEAD)"
echo "ONNX directory: $ONNX"
echo "TensorRT directory: $TRT"
echo "plugin: $PLUGIN"
echo "workspace: $WORKSPACE_GB GB"

if [[ "$EXTRACT_FRONTENDS" -eq 1 ]]; then
  python -m deploy.tensorrt.extract_onnx_subgraph \
    --onnx "$FULL_FRONTEND" \
    --output image_feat_0 \
    --output image_feat_1 \
    --output image_feat_2 \
    --output image_feat_3 \
    --output lss_bev_value \
    --out "$IMAGE_LSS_ONNX" \
    --report "$ONNX/extract_3dh_query_q1_frontend_image_lss_trt85.txt"

  python -m deploy.tensorrt.extract_onnx_subgraph \
    --onnx "$FULL_FRONTEND" \
    --output radar_bev_value \
    --out "$RADAR_ONNX" \
    --report "$ONNX/extract_3dh_query_q1_frontend_radar_trt85.txt"
else
  echo "using previously extracted image/LSS and radar frontend ONNX files"
fi

MANIFEST="$TRT/3dh_query_q1_${ENGINE_TAG}_manifest.txt"
{
  echo "target: $TARGET"
  echo "repository: $REPO_ROOT"
  echo "commit: $(git rev-parse HEAD)"
  echo "plugin: $PLUGIN"
  echo "plugin sha256: $(sha256sum "$PLUGIN" | awk '{print $1}')"
  echo "workspace GB: $WORKSPACE_GB"
  sha256sum "$IMAGE_LSS_ONNX" "$RADAR_ONNX" "$DECODER_ONNX" "$FIXTURE"
} > "$MANIFEST"
echo "manifest: $MANIFEST"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$IMAGE_LSS_ONNX" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_3dh_query_q1_frontend_image_lss_${ENGINE_TAG}.txt"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$RADAR_ONNX" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_3dh_query_q1_frontend_radar_${ENGINE_TAG}.txt"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$DECODER_ONNX" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_3dh_query_q1_decoder_${ENGINE_TAG}.txt"

IMAGE_LSS_ENGINE="$TRT/3dh_query_q1_frontend_image_lss_${ENGINE_TAG}_fp16.engine"
RADAR_ENGINE="$TRT/3dh_query_q1_frontend_radar_${ENGINE_TAG}_fp32.engine"
DECODER_ENGINE="$TRT/3dh_query_q1_decoder_precompute_v2_${ENGINE_TAG}_fp32.engine"

python -m deploy.tensorrt.build_engine \
  --onnx "$IMAGE_LSS_ONNX" \
  --engine "$IMAGE_LSS_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb "$WORKSPACE_GB" \
  --fp16 \
  --out "$TRT/build_3dh_query_q1_frontend_image_lss_${ENGINE_TAG}_fp16.txt"

python -m deploy.tensorrt.build_engine \
  --onnx "$RADAR_ONNX" \
  --engine "$RADAR_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb "$WORKSPACE_GB" \
  --out "$TRT/build_3dh_query_q1_frontend_radar_${ENGINE_TAG}_fp32.txt"

python -m deploy.tensorrt.build_engine \
  --onnx "$DECODER_ONNX" \
  --engine "$DECODER_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb "$WORKSPACE_GB" \
  --out "$TRT/build_3dh_query_q1_decoder_${ENGINE_TAG}_fp32.txt"

REPORT="$TRT/validate_3dh_query_q1_three_engine_${ENGINE_TAG}.txt"
set +e
python -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine "$IMAGE_LSS_ENGINE" \
  --radar-frontend-engine "$RADAR_ENGINE" \
  --decoder-engine "$DECODER_ENGINE" \
  --fixture "$FIXTURE" \
  --plugin "$PLUGIN" \
  --initial-query-from-fixture \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --atol "$ATOL" \
  --accept-decoded-match \
  --profile-stages \
  --out "$REPORT"
VALIDATE_STATUS=$?
set -e

grep -E \
  'actual/reference detection count|boxes close:|scores close:|labels equal:|decoded comparison passed:|end-to-end engine GPU latency:|frontend GPU latency:|radar frontend GPU latency:|recurrent decoder GPU latency:|resident CUDA memory delta:|deployment acceptance passed:|status:' \
  "$REPORT"

if [[ "$VALIDATE_STATUS" -ne 0 ]]; then
  echo "Q1 TensorRT validation failed; inspect: $REPORT" >&2
  exit "$VALIDATE_STATUS"
fi

echo "Q1 TensorRT $TARGET build and validation completed successfully."
