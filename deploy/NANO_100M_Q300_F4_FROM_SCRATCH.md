# 100m-q300 Four-Frame Nano Deployment From Scratch

Last updated: 2026-08-13

This document records a clean, offline-friendly deployment of the four-frame,
300-query RaCFormer TensorRT split pipeline to a Jetson Orin Nano. The current
target was inventoried as an 8GB-class device (`free -h` reports 7.3 GiB), not
the previously assumed 16GB device.

The transfer route is always:

```text
Git/code source -> local workstation -> Nano
server export artifacts -> local workstation -> Nano
```

The Nano is not expected to access GitHub, PyPI, Ubuntu mirrors, or the L20
server directly.

## 1. Accepted Baseline And Scope

The currently accepted precision split is:

```text
image/LSS frontend: FP16
radar frontend: FP32
recurrent decoder: FP32, enqueued 6 times
initial query: FP32 from the frontend precompute fixture
```

The L20 TensorRT 8.5.2 validation passed with decoded detections `6/6`, maximum
decoded box error `0.00146389`, end-to-end GPU latency `29.019 ms`, and resident
CUDA memory delta `980 MB`. These are server reference numbers, not Nano
performance numbers.

Important model identity: the successful checkpoint and artifacts named
`100m_q300_f4` use the legacy training config with `polar_radius=65.0`. The
true-radius-100 config is
`configs/racformer_company_front_100m_q300_r100_f4.py` and requires a newly
trained checkpoint plus a complete new export. Do not mix the two artifact
families.

## 2. Fixed Paths

Server repository:

```text
~/hyh/RaCFormer
```

Local workstation repository:

```text
the path returned by: git rev-parse --show-toplevel
```

Nano repository:

```text
/home/cttest/RaCFormer
```

Nano artifact directories:

```text
/home/cttest/RaCFormer/outputs/deploy_onnx_100m_q300_f4
/home/cttest/RaCFormer/outputs/deploy_tensorrt_100m_q300_f4
```

Never use the server container's `/workspace/...` paths on the Nano.

## 3. Files To Transfer

Transfer code from local Git commit `bb4dc27` or a later reviewed commit that
contains it. Transfer these deployment artifacts from the server through the
local workstation:

```text
outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_image_lss_trt85.onnx
outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_radar_trt85.onnx
outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_decoder_precompute_v2_trt85.onnx
outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_precompute_sample0.npz
```

Also archive the export, extraction, parser, build, and successful L20
validation reports for traceability. The full-model fixture
`racformer_100m_q300_f4_model_sample0.npz` is not the split-pipeline validation
fixture; using it causes a missing `decoder_d_regions` error.

Do not transfer an L20 `.engine` or the x86_64 plugin `.so` for execution on
the Nano. Both must be rebuilt for aarch64/SM87 on the Nano.

## 4. Package Server Artifacts

Run on the server host, outside the TensorRT container:

```bash
cd ~/hyh/RaCFormer

ARTIFACT_ARCHIVE="$HOME/racformer_100m_q300_f4_onnx_fixture_reports.tar.gz"

tar -czf "$ARTIFACT_ARCHIVE" \
  outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_image_lss_trt85.onnx \
  outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_radar_trt85.onnx \
  outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_decoder_precompute_v2_trt85.onnx \
  outputs/deploy_onnx_100m_q300_f4/racformer_100m_q300_f4_frontend_precompute_sample0.npz \
  outputs/deploy_onnx_100m_q300_f4/*.txt \
  outputs/deploy_tensorrt_100m_q300_f4/*.txt

cd "$(dirname "$ARTIFACT_ARCHIVE")"
sha256sum "$(basename "$ARTIFACT_ARCHIVE")" \
  > "$(basename "$ARTIFACT_ARCHIVE").sha256"
ls -lh "$ARTIFACT_ARCHIVE" "${ARTIFACT_ARCHIVE}.sha256"
```

The archive intentionally excludes server `.engine` files and the server
plugin `.so`.

## 5. Prepare The Local Transfer Bundle

Run on the local workstation from its own RaCFormer checkout. Do not reuse a
path copied from another machine. Replace `SERVER_IP` and `NANO_IP` with
addresses reachable from the local workstation.

```bash
cd /actual/local/path/to/RaCFormer
LOCAL_REPO="$(git rev-parse --show-toplevel)"
cd "$LOCAL_REPO"
git status --short
git rev-parse HEAD

TRANSFER_DIR="$LOCAL_REPO/nano_transfer/100m_q300_f4"
mkdir -p "$TRANSFER_DIR"

git archive \
  --format=tar.gz \
  --output="$TRANSFER_DIR/racformer_code_bb4dc27_or_later.tar.gz" \
  HEAD

git rev-parse HEAD > "$TRANSFER_DIR/GIT_COMMIT.txt"

cd "$TRANSFER_DIR"
sha256sum racformer_code_bb4dc27_or_later.tar.gz \
  > racformer_code_bb4dc27_or_later.tar.gz.sha256

scp ubuntu@SERVER_IP:/home/ubuntu/racformer_100m_q300_f4_onnx_fixture_reports.tar.gz \
  "$TRANSFER_DIR/"
scp ubuntu@SERVER_IP:/home/ubuntu/racformer_100m_q300_f4_onnx_fixture_reports.tar.gz.sha256 \
  "$TRANSFER_DIR/"

sha256sum -c racformer_code_bb4dc27_or_later.tar.gz.sha256
sha256sum -c racformer_100m_q300_f4_onnx_fixture_reports.tar.gz.sha256
cat GIT_COMMIT.txt
ls -lh
```

If `git status --short` reports unrelated uncommitted files, `git archive HEAD`
does not include them. This is deliberate: only committed deployment code is
sent to the Nano.

Copy the bundle to the Nano:

```bash
cd "$TRANSFER_DIR"

scp \
  racformer_code_bb4dc27_or_later.tar.gz \
  racformer_code_bb4dc27_or_later.tar.gz.sha256 \
  GIT_COMMIT.txt \
  racformer_100m_q300_f4_onnx_fixture_reports.tar.gz \
  racformer_100m_q300_f4_onnx_fixture_reports.tar.gz.sha256 \
  cttest@NANO_IP:/home/cttest/
```

## 6. Nano Environment Inventory Before Installing Anything

Run on the Nano:

```bash
uname -a
uname -m
cat /etc/nv_tegra_release
python3 --version
nvcc --version
cmake --version
g++ --version
dpkg-query -W nvidia-jetpack tensorrt libnvinfer8 libnvinfer-dev python3-libnvinfer 2>/dev/null
nvidia-smi 2>/dev/null || true

python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"

ldconfig -p | grep -E 'libnvinfer.so.8|libcudart.so'
ls -l /usr/local/cuda/bin/nvcc
ls -l /usr/include/aarch64-linux-gnu/NvInfer.h /usr/lib/aarch64-linux-gnu/libnvinfer.so
```

Required baseline:

```text
architecture: aarch64
L4T: R35.6.1
CUDA: 11.4
TensorRT Python/runtime/development headers: 8.5.2.2-compatible
Python: 3.8
NumPy: importable
CMake, g++, make, and nvcc: available
```

Do not install generic PyPI TensorRT. The TensorRT Python module, runtime, and
headers must come from the matching JetPack/L4T installation. PyTorch, MMCV,
MMDetection, and MMDetection3D are not required to build or validate these
TensorRT artifacts.

If a dependency is missing, record the exact failed command and package query
first. Obtain matching **aarch64 Ubuntu 20.04 / L4T R35.6.1** `.deb` packages or
Python wheels on an online machine, copy them through the local workstation,
and install them offline on the Nano. Do not install x86_64 packages. The
pre-extracted ONNX graphs mean the Nano does not need the Python `onnx` package
for engine build or final validation.

## 7. Install The Transferred Code And Artifacts

On the Nano, first verify both transfer archives:

```bash
cd /home/cttest
sha256sum -c racformer_code_bb4dc27_or_later.tar.gz.sha256
sha256sum -c racformer_100m_q300_f4_onnx_fixture_reports.tar.gz.sha256
```

Inspect the destination before extraction:

```bash
ls -la /home/cttest/RaCFormer 2>/dev/null || true
```

For a clean empty destination:

```bash
mkdir -p /home/cttest/RaCFormer
tar -xzf /home/cttest/racformer_code_bb4dc27_or_later.tar.gz \
  -C /home/cttest/RaCFormer
tar -xzf /home/cttest/racformer_100m_q300_f4_onnx_fixture_reports.tar.gz \
  -C /home/cttest/RaCFormer
```

If `/home/cttest/RaCFormer` already contains files, back it up or choose a new
destination before extraction; do not blindly overwrite an unknown tree.

Verify the required inputs:

```bash
cd /home/cttest/RaCFormer

ONNX="$PWD/outputs/deploy_onnx_100m_q300_f4"
TRT="$PWD/outputs/deploy_tensorrt_100m_q300_f4"

mkdir -p "$TRT"

ls -lh \
  "$ONNX/racformer_100m_q300_f4_frontend_image_lss_trt85.onnx" \
  "$ONNX/racformer_100m_q300_f4_frontend_radar_trt85.onnx" \
  "$ONNX/racformer_100m_q300_f4_decoder_precompute_v2_trt85.onnx" \
  "$ONNX/racformer_100m_q300_f4_frontend_precompute_sample0.npz"

python3 -c "import numpy as np; p='$ONNX/racformer_100m_q300_f4_frontend_precompute_sample0.npz'; f=np.load(p); required=['decoder_d_regions','decoder_pc_range','query_bbox','query_feat']; print('fixture keys:', len(f.files)); print('missing:', [k for k in required if k not in f.files]); print('d_regions:', f['decoder_d_regions']); print('polar_radius:', f['decoder_polar_radius'] if 'decoder_polar_radius' in f.files else 65.0)"
```

The `missing` list must be empty. The printed polar radius identifies whether
this is the legacy 65 m artifact family or a future true-radius-100 export.

## 8. Build The Nano TensorRT Plugin

The Orin Nano GPU architecture is SM87. Build the plugin on the Nano:

```bash
cd /home/cttest/RaCFormer

cmake -S deploy/tensorrt/plugins/bev_pool_v2 \
  -B build/tensorrt_plugins_orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build/tensorrt_plugins_orin --parallel 2

PLUGIN="$PWD/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"

ls -lh "$PLUGIN"
file "$PLUGIN" 2>/dev/null || true
ldd "$PLUGIN"
python3 -c "import ctypes; ctypes.CDLL('$PLUGIN'); print('plugin load: PASS')"
```

`file` must report an aarch64 shared object, and `ldd` must resolve TensorRT and
CUDA libraries.

## 9. Build The Three Nano Engines

Set the runtime library path after each new login shell:

```bash
cd /home/cttest/RaCFormer

export LD_LIBRARY_PATH="$PWD/build/tensorrt_plugins_orin:/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

ONNX="$PWD/outputs/deploy_onnx_100m_q300_f4"
TRT="$PWD/outputs/deploy_tensorrt_100m_q300_f4"
PLUGIN="$PWD/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"

mkdir -p "$TRT"
```

Build sequentially so the unified-memory peak is attributable to one build.
The image/LSS engine is the only FP16 engine in the accepted baseline:

```bash
python3 -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_100m_q300_f4_frontend_image_lss_trt85.onnx" \
  --engine "$TRT/racformer_100m_q300_f4_frontend_image_lss_trt852_fp16_orin.engine" \
  --plugin "$PLUGIN" \
  --fp16 \
  --workspace-gb 2 \
  --out "$TRT/build_100m_q300_f4_frontend_image_lss_trt852_fp16_orin.txt"
```

```bash
python3 -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_100m_q300_f4_frontend_radar_trt85.onnx" \
  --engine "$TRT/racformer_100m_q300_f4_frontend_radar_trt852_fp32_orin.engine" \
  --plugin "$PLUGIN" \
  --workspace-gb 2 \
  --out "$TRT/build_100m_q300_f4_frontend_radar_trt852_fp32_orin.txt"
```

```bash
python3 -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_100m_q300_f4_decoder_precompute_v2_trt85.onnx" \
  --engine "$TRT/racformer_100m_q300_f4_decoder_precompute_v2_trt852_fp32_orin.engine" \
  --plugin "$PLUGIN" \
  --workspace-gb 2 \
  --out "$TRT/build_100m_q300_f4_decoder_precompute_v2_trt852_fp32_orin.txt"
```

Do not run multiple builders concurrently. After each build, require
`status: SUCCESS` and confirm that its engine file is non-empty.

```bash
grep -H -E 'TensorRT version:|status:|precision:|build time:|engine size:|FAILED|RuntimeError|Error' \
  "$TRT"/build_100m_q300_f4_*.txt

ls -lh "$TRT"/*.engine
```

## 10. Correctness Validation

Run a short validation first:

```bash
python3 -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine "$TRT/racformer_100m_q300_f4_frontend_image_lss_trt852_fp16_orin.engine" \
  --radar-frontend-engine "$TRT/racformer_100m_q300_f4_frontend_radar_trt852_fp32_orin.engine" \
  --decoder-engine "$TRT/racformer_100m_q300_f4_decoder_precompute_v2_trt852_fp32_orin.engine" \
  --fixture "$ONNX/racformer_100m_q300_f4_frontend_precompute_sample0.npz" \
  --plugin "$PLUGIN" \
  --initial-query-from-fixture \
  --accept-decoded-match \
  --profile-stages \
  --warmup 5 \
  --iters 5 \
  --atol 0.03 \
  --out "$TRT/validate_100m_q300_f4_three_engine_trt852_orin_short.txt"
```

Acceptance requires all of the following:

```text
actual/reference detection count: equal
boxes close: True
scores close: True
labels equal: True
decoded comparison passed: True
deployment acceptance passed: True
status: SUCCESS
```

Extract the result:

```bash
REPORT="$TRT/validate_100m_q300_f4_three_engine_trt852_orin_short.txt"

grep -E \
'actual/reference detection count|boxes close:|scores close:|labels equal:|decoded comparison passed:|end-to-end engine GPU latency:|frontend GPU latency:|radar frontend GPU latency:|recurrent decoder GPU latency:|resident CUDA memory delta:|deployment acceptance passed:|status:' \
  "$REPORT"
```

## 11. Controlled Nano Benchmark

Record the power mode, clocks, thermal state, and memory before interpreting
latency. Choose the intended power mode based on `nvpmodel` output rather than
assuming a mode number:

```bash
sudo nvpmodel -q --verbose
sudo jetson_clocks
jetson_clocks --show
tegrastats --interval 1000
```

Stop `tegrastats` with Ctrl-C after recording the idle state. Then run:

```bash
python3 -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine "$TRT/racformer_100m_q300_f4_frontend_image_lss_trt852_fp16_orin.engine" \
  --radar-frontend-engine "$TRT/racformer_100m_q300_f4_frontend_radar_trt852_fp32_orin.engine" \
  --decoder-engine "$TRT/racformer_100m_q300_f4_decoder_precompute_v2_trt852_fp32_orin.engine" \
  --fixture "$ONNX/racformer_100m_q300_f4_frontend_precompute_sample0.npz" \
  --plugin "$PLUGIN" \
  --initial-query-from-fixture \
  --accept-decoded-match \
  --profile-stages \
  --warmup 20 \
  --iters 20 \
  --atol 0.03 \
  --out "$TRT/profile_100m_q300_f4_three_engine_trt852_orin.txt"
```

This measures engine execution with preloaded fixture inputs. It does not yet
measure JPG decoding, PLY parsing, four-frame buffering, host preprocessing,
sensor synchronization, result serialization, or end-to-end application I/O.

## 12. Copy Nano Results Back Through Local

Run on the local workstation after validation:

```bash
LOCAL_REPO="$(git rev-parse --show-toplevel)"
RESULT_DIR="$LOCAL_REPO/nano_results/100m_q300_f4"
mkdir -p "$RESULT_DIR"

scp 'cttest@NANO_IP:/home/cttest/RaCFormer/outputs/deploy_tensorrt_100m_q300_f4/*.txt' \
  "$RESULT_DIR/"
```

Keep the Nano build reports, validation report, environment inventory, power
mode, and `tegrastats` sample together. These reports are the source of truth
for Nano performance; L20 timings must never be reported as Nano timings.
