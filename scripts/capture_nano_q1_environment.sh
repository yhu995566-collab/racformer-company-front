#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DIR="$REPO_ROOT/outputs/deploy_tensorrt_q1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$DEFAULT_DIR/nano_q1_environment_${TIMESTAMP}.txt}"

mkdir -p "$(dirname "$OUT")"
OUT="$(realpath -m "$OUT")"

section() {
  printf '\n=== %s ===\n' "$1"
}

run_optional() {
  local label="$1"
  shift
  printf '\n--- %s ---\n' "$label"
  "$@" 2>&1 || printf '[unavailable or failed, exit=%s]\n' "$?"
}

{
  echo "RaCFormer Q1 Nano environment and artifact inventory"
  echo "captured at: $(date --iso-8601=seconds)"
  echo "repository: $REPO_ROOT"
  echo "report: $OUT"

  section "Device and operating system"
  run_optional "uname" uname -a
  if [[ -r /etc/nv_tegra_release ]]; then
    run_optional "L4T release" cat /etc/nv_tegra_release
  fi
  run_optional "OS release" cat /etc/os-release
  run_optional "architecture" dpkg --print-architecture
  run_optional "memory" free -h
  run_optional "disk" df -h "$REPO_ROOT"
  if command -v nvpmodel >/dev/null 2>&1; then
    run_optional "nvpmodel" nvpmodel -q
  fi
  if command -v jetson_clocks >/dev/null 2>&1; then
    run_optional "jetson clocks" jetson_clocks --show
  fi

  section "CUDA and TensorRT"
  if command -v nvcc >/dev/null 2>&1; then
    run_optional "nvcc" nvcc --version
  else
    echo "nvcc: not found"
  fi
  if command -v trtexec >/dev/null 2>&1; then
    run_optional "trtexec" trtexec --version
  else
    echo "trtexec: not found"
  fi
  run_optional "Python TensorRT/CUDA/NumPy/ONNX" python3 -c \
    "import sys; print('python:', sys.version.replace('\\n', ' ')); import tensorrt as trt; print('tensorrt:', trt.__version__); import numpy; print('numpy:', numpy.__version__); import onnx; print('onnx:', onnx.__version__)"
  run_optional "NVIDIA package versions" dpkg-query -W \
    'nvidia-jetpack' 'nvidia-l4t-*' 'cuda-*' 'libnvinfer*' \
    'python3-libnvinfer*'

  section "Build and Python tools"
  run_optional "git" git --version
  run_optional "cmake" cmake --version
  run_optional "g++" g++ --version
  run_optional "Python pip freeze" python3 -m pip freeze
  run_optional "manually installed apt packages" apt-mark showmanual

  section "Repository"
  run_optional "Git commit" git -C "$REPO_ROOT" rev-parse HEAD
  run_optional "Git branch" git -C "$REPO_ROOT" branch --show-current
  run_optional "Git status" git -C "$REPO_ROOT" status --short

  PLUGIN="$REPO_ROOT/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"
  section "Orin plugin"
  if [[ -f "$PLUGIN" ]]; then
    sha256sum "$PLUGIN"
    run_optional "plugin ELF" file "$PLUGIN"
    run_optional "plugin dependencies" ldd "$PLUGIN"
  else
    echo "missing: $PLUGIN"
  fi

  section "Q1 synchronized inputs and generated outputs"
  shopt -s nullglob
  artifacts=(
    "$REPO_ROOT"/outputs/deploy_onnx_q1/3dh_query_q1_*
    "$REPO_ROOT"/outputs/deploy_tensorrt_q1/3dh_query_q1_*
    "$REPO_ROOT"/outputs/deploy_tensorrt_q1/validate_3dh_query_q1_*
  )
  if (( ${#artifacts[@]} == 0 )); then
    echo "no Q1 artifacts found"
  else
    for artifact in "${artifacts[@]}"; do
      [[ -f "$artifact" && "$artifact" != "$OUT" ]] || continue
      sha256sum "$artifact"
    done
  fi
  shopt -u nullglob
} | tee "$OUT"

echo "Nano environment report written to: $OUT"
