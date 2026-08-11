# Q1 Nano 离线环境与文件同步台账

最后更新：2026-08-11

这份文档是 Q1 在 Jetson Orin Nano 16GB 上重新配置、同步和部署的唯一
台账。Nano 默认按无法访问公网处理：不得依赖 Nano 执行 `git fetch`、
`git pull`、`apt update` 或在线 `pip install`。

数据流固定为：

```text
本地 Git 仓库 -> 离线 Git bundle -> Nano
服务器导出目录 -> 本地中转目录 -> Nano
联网侧准备的 aarch64 离线依赖包 -> 本地中转目录 -> Nano
Nano 构建报告/环境快照 -> 本地归档 -> 服务器归档（需要时）
```

完整模型导出和 TensorRT 操作仍以 `deploy/Q1_TENSORRT.md` 为准。每次安装
软件、传输文件、重建插件或 Engine、修改部署参数后，都要更新第 11 节的
执行记录，并保存来源和目标两端的 SHA256。

## 1. 已知基线与离线原则

旧部署记录中的 Nano 基线是：

- Jetson Orin Nano 16GB，aarch64
- L4T R35.6.1 / Ubuntu 20.04
- CUDA 11.4
- TensorRT Python 8.5.2.2
- Python 3.8
- 项目根目录 `/home/cttest/RaCFormer`

这些必须在本次重配后重新采集，不能直接当作当前事实。禁止用 x86_64 的
`.deb`、Python wheel、TensorRT Engine 或插件 `.so` 配置 Nano。不要用
PyPI TensorRT 覆盖 JetPack 自带的 aarch64 TensorRT。

代码分支固定为：

```text
q1-tensorrt-deployment
```

## 2. 固定目录

Nano：

```text
/home/cttest/RaCFormer
├── outputs/deploy_onnx_q1
├── outputs/deploy_tensorrt_q1
└── build/tensorrt_plugins_orin

/home/cttest/q1_offline_transfer
├── code
├── artifacts
└── packages
```

本地电脑建立一个按日期归档的中转目录，例如：

```text
~/q1_nano_transfer/2026-08-11/
├── code
├── artifacts
├── packages
└── nano_reports
```

Nano 禁止使用服务器容器的 `/workspace/...` 路径。旧 200 m 文件不能作为
Q1 输入。

## 3. 本地 Git 生成离线代码包

先在能够访问 GitHub 的本地电脑同步并确认代码：

```bash
cd /home/yanhao/projects/3DH-Query
git fetch origin
git checkout q1-tensorrt-deployment
git pull --ff-only origin q1-tensorrt-deployment
git status --short
git rev-parse HEAD
```

生成包含完整 Git 历史和目标分支的离线 bundle：

```bash
bash scripts/package_q1_nano_offline.sh \
  "$HOME/q1_nano_transfer/2026-08-11/code"
```

脚本生成 `.bundle`、`.bundle.sha256` 和 manifest。它不会包含本地未跟踪
文件、ONNX、fixture 或 Engine。如果存在尚未提交的已跟踪修改，脚本会
拒绝打包。

将这三个文件通过局域网、移动硬盘或其他离线介质传到：

```text
/home/cttest/q1_offline_transfer/code
```

## 4. Nano 离线导入代码

先验证代码包；将文件名替换为本次实际文件名：

```bash
cd /home/cttest/q1_offline_transfer/code
sha256sum -c 3dh_query_q1_code_<commit>.bundle.sha256
git bundle verify 3dh_query_q1_code_<commit>.bundle
```

Nano 已有仓库且 `git status --short` 为空时：

```bash
cd /home/cttest/RaCFormer
git status --short
git fetch /home/cttest/q1_offline_transfer/code/3dh_query_q1_code_<commit>.bundle \
  q1-tensorrt-deployment
git checkout q1-tensorrt-deployment
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
```

如果 `git status --short` 不为空，先记录并人工处理，禁止用 reset/clean
覆盖 Nano 上未知文件。

Nano 没有仓库时：

```bash
git clone \
  /home/cttest/q1_offline_transfer/code/3dh_query_q1_code_<commit>.bundle \
  /home/cttest/RaCFormer
cd /home/cttest/RaCFormer
git checkout q1-tensorrt-deployment
git rev-parse HEAD
```

## 5. 采集重配前环境

代码导入后立即执行：

```bash
cd /home/cttest/RaCFormer
mkdir -p outputs/deploy_tensorrt_q1
bash scripts/capture_nano_q1_environment.sh \
  outputs/deploy_tensorrt_q1/nano_q1_environment_before_setup.txt
```

把报告从 Nano 回传到本地 `nano_reports`，据此决定缺少哪些依赖。

## 6. 离线准备和安装环境依赖

Nano 侧部署只需要构建插件、解析 ONNX、构建 Engine 和运行 NumPy 验证，
不需要 PyTorch 或完整 OpenMMLab。需要检查：

- JetPack 自带 CUDA 11.4、TensorRT 8.5.2.2、Python TensorRT binding
- `git`、`cmake`、`build-essential`、`python3`、`python3-pip`
- Python `numpy`、`onnx`
- TensorRT headers，通常来自 `libnvinfer-dev`

Nano 本机只检查，不联网：

```bash
cat /etc/nv_tegra_release
nvcc --version
python3 --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import numpy, onnx; print(numpy.__version__, onnx.__version__)"
dpkg-query -W git cmake build-essential python3-pip libnvinfer-dev \
  python3-libnvinfer
```

如有缺失，在联网侧按 Nano 的 `aarch64 + Ubuntu 20.04 + L4T R35.6.1 +
Python 3.8` 准备离线 `.deb`/wheel 及全部依赖。x86 本地电脑只能负责下载
和中转，不能把自身安装包直接交给 Nano。所有包放入本地 `packages`，生成：

```bash
cd "$HOME/q1_nano_transfer/2026-08-11/packages"
find . -maxdepth 1 -type f ! -name nano_offline_packages.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > nano_offline_packages.sha256
```

传入 Nano 的 `/home/cttest/q1_offline_transfer/packages` 后先校验，再使用
`apt install ./xxx.deb` 或：

```bash
python3 -m pip install --no-index --find-links . <package-name>
```

每个真实下载来源、文件名、版本、SHA256 和安装命令都必须写入第 11 节。
不要在版本和来源未确认前凭猜测下载包。

安装完成后采集：

```bash
bash /home/cttest/RaCFormer/scripts/capture_nano_q1_environment.sh \
  /home/cttest/RaCFormer/outputs/deploy_tensorrt_q1/nano_q1_environment_after_setup.txt
```

## 7. 服务器产物经本地中转到 Nano

服务器完成 L20 TRT 8.5 验收后，在服务器执行：

```bash
cd /workspace/outputs/deploy_onnx_q1
sha256sum \
  3dh_query_q1_frontend_image_lss_trt85.onnx \
  3dh_query_q1_frontend_radar_trt85.onnx \
  3dh_query_q1_decoder_precompute_v2_trt85.onnx \
  3dh_query_q1_frontend_precompute_sample0.npz \
  > q1_nano_transfer.sha256
```

从服务器同步到本地 `artifacts` 的只有：

```text
3dh_query_q1_frontend_image_lss_trt85.onnx
3dh_query_q1_frontend_radar_trt85.onnx
3dh_query_q1_decoder_precompute_v2_trt85.onnx
3dh_query_q1_frontend_precompute_sample0.npz
q1_nano_transfer.sha256
```

本地进入 `artifacts` 执行 `sha256sum -c q1_nano_transfer.sha256`，通过后再
传到 Nano：

```text
/home/cttest/q1_offline_transfer/artifacts
```

Nano 再次校验并复制到项目目录：

```bash
cd /home/cttest/q1_offline_transfer/artifacts
sha256sum -c q1_nano_transfer.sha256

mkdir -p /home/cttest/RaCFormer/outputs/deploy_onnx_q1
cp -p 3dh_query_q1_frontend_image_lss_trt85.onnx \
  3dh_query_q1_frontend_radar_trt85.onnx \
  3dh_query_q1_decoder_precompute_v2_trt85.onnx \
  3dh_query_q1_frontend_precompute_sample0.npz \
  q1_nano_transfer.sha256 \
  /home/cttest/RaCFormer/outputs/deploy_onnx_q1/

cd /home/cttest/RaCFormer/outputs/deploy_onnx_q1
sha256sum -c q1_nano_transfer.sha256
```

L20 `.engine`、x86_64 插件 `.so`、旧 fixture 和旧 ONNX 禁止传入。

## 8. Nano 本机重建 Orin 插件

```bash
cd /home/cttest/RaCFormer
cmake -S deploy/tensorrt/plugins/bev_pool_v2 \
  -B build/tensorrt_plugins_orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build/tensorrt_plugins_orin --parallel 4

PLUGIN="$PWD/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so"
file "$PLUGIN"
ldd "$PLUGIN"
python3 -c "import ctypes; ctypes.CDLL('$PLUGIN'); print('plugin load: PASS')"
sha256sum "$PLUGIN"
```

## 9. Nano 本机构建 Engine 并验收

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

必须保留三个 Orin Engine、构建 manifest、parser/build 报告以及：

```text
outputs/deploy_tensorrt_q1/validate_3dh_query_q1_three_engine_trt852_orin.txt
```

验收要求 decoded count、boxes、scores、labels 和 deployment acceptance 全部
通过。当前 FP16 image/LSS frontend 框误差门限是 3 cm。

## 10. 完成后快照与反向归档

```bash
cd /home/cttest/RaCFormer
bash scripts/capture_nano_q1_environment.sh \
  outputs/deploy_tensorrt_q1/nano_q1_environment_after_q1_build.txt
```

把环境报告、传输清单、构建 manifest 和验证报告从 Nano 传回本地
`nano_reports`，先生成和校验 SHA256，再按需要同步到服务器。Engine 和
Orin 插件只属于当前 Nano，不作为服务器到其他平台的通用文件。

## 11. 执行记录

每完成一步增加一行，不覆盖历史记录。

| 日期时间 | 来源 -> 目标 | 操作 | 版本/文件/命令 | 结果 | 报告或 SHA256 |
|---|---|---|---|---|---|
| 2026-08-11 | 开发仓库 | 建立 Nano 环境和同步台账 | 环境采集脚本 | PASS | `scripts/capture_nano_q1_environment.sh` |
| 2026-08-11 | 本地 Git -> Nano | 改为离线 Git bundle 同步 | 离线打包脚本 | PASS | `scripts/package_q1_nano_offline.sh` |

仍需补齐：Nano 实测环境版本；实际离线依赖来源、文件和安装命令；代码
bundle SHA256；四个服务器产物 SHA256；Orin 插件和 Engine SHA256；构建
耗时、decoded 验收、分阶段延迟、端到端延迟和 CUDA 显存。
