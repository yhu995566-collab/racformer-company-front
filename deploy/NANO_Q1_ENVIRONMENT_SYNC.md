# Q1 Nano 环境与文件同步台账

最后更新：2026-08-11

这份文档是 Q1 在 Jetson Orin Nano 16GB 上重新配置、同步和部署的唯一
台账。每次安装软件、传输文件、重建插件或 Engine、修改部署参数后，都要
更新末尾的“执行记录”，并保存环境采集报告和 SHA256 清单。

完整模型导出和 TensorRT 操作仍以 `deploy/Q1_TENSORRT.md` 为准。

## 1. 已知基线与本次目标

旧部署已经确认过的 Nano 基线是：

- 设备：Jetson Orin Nano 16GB，aarch64
- 系统：L4T R35.6.1 / Ubuntu 20.04
- CUDA：11.4
- TensorRT Python：8.5.2.2
- Python：3.8
- 项目根目录：`/home/cttest/RaCFormer`

这些是旧部署记录，不应直接当作本次重配后的事实。本次必须运行环境采集
脚本重新确认。不要单独升级 CUDA、cuDNN 或 TensorRT，也不要用 PyPI 的
TensorRT 覆盖 JetPack 提供的 aarch64 软件包。

Q1 目标代码版本：

```text
branch: 3dh-query-stage1-radar-candidate-recall
minimum commit containing the deployment workflow: 492d8f6
```

## 2. Nano 固定目录

```text
/home/cttest/RaCFormer
├── outputs/deploy_onnx_q1        # 从服务器同步的 ONNX 和 fixture
├── outputs/deploy_tensorrt_q1    # Nano 本机构建的 Engine、报告和环境快照
└── build/tensorrt_plugins_orin   # Nano 本机构建的 aarch64 TRT 插件
```

Nano 上禁止使用服务器容器的 `/workspace/...` 路径。旧版 200 m 的
`outputs/deploy_onnx` 和 `outputs/deploy_tensorrt` 不能作为 Q1 输入。

## 3. 重配前采集当前状态

如果旧仓库还能运行，先拉取包含采集脚本的提交，再保存一次“重配前”快照：

```bash
cd /home/cttest/RaCFormer
git fetch origin
git checkout 3dh-query-stage1-radar-candidate-recall
git pull --ff-only origin 3dh-query-stage1-radar-candidate-recall

bash scripts/capture_nano_q1_environment.sh \
  outputs/deploy_tensorrt_q1/nano_q1_environment_before_setup.txt
```

如果仓库还不存在，先完成第 5 节的代码同步，再采集快照。

## 4. 系统和 Python 环境记录

先验证 JetPack 基础栈：

```bash
cat /etc/nv_tegra_release
nvcc --version
python3 --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
dpkg-query -W 'libnvinfer*' 'python3-libnvinfer*' 'cuda-*'
```

本流程的 Nano 侧只需要构建插件、解析 ONNX、构建 Engine 和运行 NumPy
验证器，不需要安装 PyTorch 或完整 OpenMMLab 训练环境。需要的软件类别是：

- JetPack 自带的 CUDA 11.4、TensorRT 8.5.2.2 和 Python TensorRT binding
- `git`、`cmake`、`build-essential`、`python3`、`python3-pip`
- Python `numpy` 和 `onnx`
- 编译插件需要的 TensorRT headers，Ubuntu 包通常是 `libnvinfer-dev`

不要先盲目安装。只对检查后缺失的包执行安装，并把真实命令和版本填写到
第 11 节。系统包检查示例：

```bash
dpkg-query -W git cmake build-essential python3-pip libnvinfer-dev \
  python3-libnvinfer
python3 -c "import numpy, onnx; print(numpy.__version__, onnx.__version__)"
```

安装或修复完成后必须保存完整快照：

```bash
bash scripts/capture_nano_q1_environment.sh \
  outputs/deploy_tensorrt_q1/nano_q1_environment_after_setup.txt
```

采集报告包含 JetPack/L4T、CUDA、TensorRT、Python、pip freeze、手动安装的
apt 包、Git 提交、插件依赖以及 Q1 文件 SHA256。

## 5. 同步代码

已有仓库：

```bash
cd /home/cttest/RaCFormer
git fetch origin
git checkout 3dh-query-stage1-radar-candidate-recall
git pull --ff-only origin 3dh-query-stage1-radar-candidate-recall
git rev-parse HEAD
git status --short
```

期望提交至少包含 `492d8f6`。如果 Nano 是全新目录，使用同一个远端仓库
克隆到 `/home/cttest/RaCFormer`，然后切换到上述分支。不要通过复制旧
RaCFormer 目录来代替 Git 同步。

## 6. 服务器生成传输清单

L20 TensorRT 8.5 验收通过后，在服务器的 `deploy_onnx_q1` 目录执行：

```bash
cd /workspace/outputs/deploy_onnx_q1

sha256sum \
  3dh_query_q1_frontend_image_lss_trt85.onnx \
  3dh_query_q1_frontend_radar_trt85.onnx \
  3dh_query_q1_decoder_precompute_v2_trt85.onnx \
  3dh_query_q1_frontend_precompute_sample0.npz \
  > q1_nano_transfer.sha256
```

本次只允许同步下面五个文件：

```text
3dh_query_q1_frontend_image_lss_trt85.onnx
3dh_query_q1_frontend_radar_trt85.onnx
3dh_query_q1_decoder_precompute_v2_trt85.onnx
3dh_query_q1_frontend_precompute_sample0.npz
q1_nano_transfer.sha256
```

不要同步 L20 的 `.engine`、x86_64 `.so`、旧 200 m fixture 或旧四帧 ONNX。

## 7. 传输并校验文件

先在 Nano 创建目标目录：

```bash
mkdir -p /home/cttest/RaCFormer/outputs/deploy_onnx_q1
```

在服务器执行；将 `<NANO_IP>` 替换为实际地址：

```bash
rsync -avP \
  /workspace/outputs/deploy_onnx_q1/3dh_query_q1_frontend_image_lss_trt85.onnx \
  /workspace/outputs/deploy_onnx_q1/3dh_query_q1_frontend_radar_trt85.onnx \
  /workspace/outputs/deploy_onnx_q1/3dh_query_q1_decoder_precompute_v2_trt85.onnx \
  /workspace/outputs/deploy_onnx_q1/3dh_query_q1_frontend_precompute_sample0.npz \
  /workspace/outputs/deploy_onnx_q1/q1_nano_transfer.sha256 \
  cttest@<NANO_IP>:/home/cttest/RaCFormer/outputs/deploy_onnx_q1/
```

在 Nano 校验；任何一个文件失败都不能继续建 Engine：

```bash
cd /home/cttest/RaCFormer/outputs/deploy_onnx_q1
sha256sum -c q1_nano_transfer.sha256
```

## 8. Nano 本地重新构建插件

插件必须在 Nano 本机基于当前源码、CUDA 和 TensorRT headers 重新编译：

```bash
cd /home/cttest/RaCFormer

cmake -S deploy/tensorrt/plugins/bev_pool_v2 \
  -B build/tensorrt_plugins_orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build/tensorrt_plugins_orin --parallel 4
```

验证架构、依赖和加载：

```bash
PLUGIN="$PWD/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"
file "$PLUGIN"
ldd "$PLUGIN"
python3 -c "import ctypes; ctypes.CDLL('$PLUGIN'); print('plugin load: PASS')"
sha256sum "$PLUGIN"
```

## 9. Nano 本地构建 Engine 并验收

```bash
cd /home/cttest/RaCFormer

export LD_LIBRARY_PATH="$PWD/build/tensorrt_plugins_orin:/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

OUTPUT_ROOT="$PWD/outputs" \
WORKSPACE_GB=8 \
WARMUP=20 \
ITERS=20 \
ATOL=0.03 \
bash scripts/deploy_q1_tensorrt.sh orin
```

必须保留以下报告和本机生成的三个 Engine：

```text
outputs/deploy_tensorrt_q1/3dh_query_q1_frontend_image_lss_trt852_orin_fp16.engine
outputs/deploy_tensorrt_q1/3dh_query_q1_frontend_radar_trt852_orin_fp32.engine
outputs/deploy_tensorrt_q1/3dh_query_q1_decoder_precompute_v2_trt852_orin_fp32.engine
outputs/deploy_tensorrt_q1/validate_3dh_query_q1_three_engine_trt852_orin.txt
outputs/deploy_tensorrt_q1/3dh_query_q1_trt852_orin_manifest.txt
```

验收要求是 decoded detection count、boxes、scores、labels 和 deployment
acceptance 全部通过。3 cm 是当前 FP16 image/LSS frontend 的框误差门限。

## 10. 完成后环境快照

```bash
cd /home/cttest/RaCFormer
bash scripts/capture_nano_q1_environment.sh \
  outputs/deploy_tensorrt_q1/nano_q1_environment_after_q1_build.txt
```

将环境报告、传输 SHA256 清单、构建 manifest 和验证报告一起回传到服务器
归档。Engine 和 Orin 插件只属于当前 Nano 环境，不作为跨平台同步文件。

## 11. 执行记录

每完成一步就在表中增加一行，不覆盖历史记录。

| 日期时间 | 设备 | 操作 | 命令/版本/文件 | 结果 | 报告或 SHA256 |
|---|---|---|---|---|---|
| 2026-08-11 | 开发仓库 | 建立 Q1 Nano 重配与同步台账及只读采集脚本 | 当前分支最新提交 | PASS | 本文档、`scripts/capture_nano_q1_environment.sh` |

尚未实测、必须补齐的项目：

- Nano 重配后的 L4T、CUDA、TensorRT、Python、NumPy、ONNX 精确版本
- 实际安装过的 apt/pip 命令及下载包来源
- Nano IP 和服务器到 Nano 的实际传输时间
- 四个输入文件的 SHA256
- Orin 插件 SHA256
- 三个 Orin Engine SHA256、构建耗时和大小
- decoded 验收结果、分阶段延迟、端到端延迟和 CUDA 显存
