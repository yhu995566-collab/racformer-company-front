#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/or/relative/path/to/q1_epoch.pth" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="$(realpath "$1")"
CONFIG="${CONFIG:-configs/deploy/3dh_query_q1_left_pytorch_f4.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
ONNX="$OUTPUT_ROOT/deploy_onnx_q1"
DEVICE="${DEVICE:-cuda:0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
STATIC_RADAR_VOXELS="${STATIC_RADAR_VOXELS:-1024}"

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "deployment config not found: $CONFIG" >&2
  exit 1
fi
if ! [[ "$STATIC_RADAR_VOXELS" =~ ^[1-9][0-9]*$ ]]; then
  echo "STATIC_RADAR_VOXELS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$ONNX"

MANIFEST="$ONNX/3dh_query_q1_export_manifest.txt"
{
  echo "repository: $REPO_ROOT"
  echo "commit: $(git rev-parse HEAD)"
  echo "config: $CONFIG"
  echo "checkpoint: $CKPT"
  echo "checkpoint sha256: $(sha256sum "$CKPT" | awk '{print $1}')"
  echo "sample index: $SAMPLE_INDEX"
  echo "static radar voxel slots per frame: $STATIC_RADAR_VOXELS"
} > "$MANIFEST"

echo "repository: $REPO_ROOT"
echo "commit: $(git rev-parse HEAD)"
echo "config: $CONFIG"
echo "checkpoint: $CKPT"
echo "output: $ONNX"
echo "device: $DEVICE"
echo "sample index: $SAMPLE_INDEX"
echo "static radar voxel slots per frame: $STATIC_RADAR_VOXELS"
echo "manifest: $MANIFEST"

python -m deploy.export_onnx \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --device "$DEVICE" \
  --split val \
  --sample-index "$SAMPLE_INDEX" \
  --opset 17 \
  --mixing-chunk-size 32768 \
  --msmv-plugin \
  --single-camera-projection-plugin \
  --fixed-view-geometry \
  --tensorrt-85-compat \
  --static-radar-voxels "$STATIC_RADAR_VOXELS" \
  --out "$ONNX/3dh_query_q1_raw_trt85.onnx" \
  --fixture "$ONNX/3dh_query_q1_model_sample0.npz" \
  --report "$ONNX/export_3dh_query_q1_model_trt85.txt"

python -m deploy.export_frontend_onnx \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$ONNX/3dh_query_q1_model_sample0.npz" \
  --device "$DEVICE" \
  --opset 17 \
  --precompute-bev-values \
  --out "$ONNX/3dh_query_q1_frontend_precompute_v2_trt85.onnx" \
  --fixture "$ONNX/3dh_query_q1_frontend_precompute_sample0.npz" \
  --report "$ONNX/export_3dh_query_q1_frontend_precompute_v2_trt85.txt"

python -m deploy.export_decoder_recurrent_layer \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$ONNX/3dh_query_q1_model_sample0.npz" \
  --device "$DEVICE" \
  --opset 17 \
  --precompute-bev-values \
  --out "$ONNX/3dh_query_q1_decoder_precompute_v2_trt85.onnx" \
  --report "$ONNX/export_3dh_query_q1_decoder_precompute_v2_trt85.txt"

grep -H -E 'status:|decoded boundary comparison passed:|frontend output boundary:|PyTorch recurrent loop close:' \
  "$ONNX/export_3dh_query_q1_model_trt85.txt" \
  "$ONNX/export_3dh_query_q1_frontend_precompute_v2_trt85.txt" \
  "$ONNX/export_3dh_query_q1_decoder_precompute_v2_trt85.txt"

echo "Q1 PyTorch/ONNX export completed successfully."
