# Main4 50m-q200 FOV120 固定平台部署交接

最后更新：2026-09-01

本文记录 Main4 四分类模型从 PyTorch checkpoint 到服务器 TensorRT
验证、Nano 重建以及 C++ runtime 交付的完整流程。本文是本次模型族的主交接记录；
旧的 100m-q300 和早期四帧部署记录仅用于参考，不能与本模型的 ONNX、fixture、
engine 或 runtime constants 混用。

## 1. 本次模型身份

模型参数如下：

| 项目 | 值 |
| --- | --- |
| 类别 | `car, truck, bicycle, pedestrian` |
| 类别数 | 4 |
| 检测范围 | `[0, -20, -3, 50, 20, 3]` m |
| Query 数 | 200 |
| Query 初始化 | `front_fov_grid` |
| Query 距离 power | 1.5 |
| 水平 FOV | 120° |
| 时序帧数 | 4 |
| Decoder 层数/迭代数 | 6 |
| 训练原图 | 1920×1080 |
| 网络输入 | 640×256 |
| 部署平台 | 固定不移动 |

Main4 checkpoint：

```text
/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning/full_main4_f4_fov120_p15_e8/20260831_184101/model/best_company/3D_mAP@0.5_epoch_8.pth
```

专用部署配置：

```text
configs/deploy/racformer_company_front_50m_q200_fov120_p15_main4_left_pytorch_f4.py
```

该配置不能替换成旧的 50m-q200、100m-q300 或 Main3 配置。Main3 和 Main4
至少 decoder 分类输出维度不同；不同 checkpoint 的三个 engine 也都不能共用。

## 2. 固定平台部署约束

部署设备固定在原地，因此四帧之间没有自车位姿变化。部署配置已启用：

```python
static_view_geometry = True
```

预处理会把当前帧的 `lidar2img` 和相机内参重复用于四个时序槽，从而满足
TensorRT 固定视角图的要求：

- 不需要向 runtime 输入 odometry 或 ego pose；
- 自车速度为 0，不执行自车运动补偿；
- 雷达与相机的固定安装外参仍然必须使用；
- 四帧图像、雷达数据、时间戳和 radar `time_lag` 仍然保留；
- recurrent decoder 仍运行 6 次，不能因为输入为 4 帧而减少为 4 次。

实现提交：

```text
d96f4e3 feat: support fixed-platform view geometry
```

此前导出失败：

```text
fixed view geometry requires identical frame transforms;
maximum img2lidar difference is 4.34561634
```

原因是训练数据中的历史帧携带移动采集车的不同位姿，而固定几何导出器要求四帧
变换一致。`d96f4e3` 明确实现了固定部署平台的几何契约。重新导出后该差值应为
0 或浮点误差级别。

## 3. 机器、仓库和路径

不同机器的路径不能互相照抄。

| 环境 | 代码路径 | 产物路径/说明 |
| --- | --- | --- |
| 当前开发工作区 | `/home/yanhao/projects/RaCFormer` | 编写、提交部署代码 |
| 公司本地中转仓库 | `~/project/RaCFormer` | 从服务器下载，再通过数据线传 Nano |
| L20 服务器宿主机 | `/home/ubuntu/hyh/RaCFormer` | PyTorch 导出、保存原始产物 |
| TRT 8.5 容器 | `/workspace/RaCFormer` | 代码；TensorRT 产物使用 `/workspace/outputs` |
| Orin Nano | `/home/cttest/RaCFormer` | aarch64 plugin、engine、C++ runtime |

服务器 TensorRT 8.5.2 容器：

```text
racformer_trt85_l20
```

容器启动方式：

```bash
docker start racformer_trt85_l20
docker exec -it racformer_trt85_l20 bash
cd /workspace/RaCFormer
```

容器中的 `/workspace/...` 路径不能用于 Nano。

### 3.1 关键实现文件

```text
configs/deploy/racformer_company_front_50m_q200_fov120_p15_main4_left_pytorch_f4.py
deploy/preprocessing.py
deploy/export_onnx.py
deploy/export_frontend_onnx.py
deploy/export_decoder_recurrent_layer.py
deploy/tensorrt/extract_onnx_subgraph.py
deploy/tensorrt/parse_onnx.py
deploy/tensorrt/build_engine.py
deploy/tensorrt/validate_frontend_decoder_numpy.py
deploy/tensorrt/plugins/bev_pool_v2/
deploy/cpp/
deploy/cpp_visualizer/
```

`deploy/cpp_visualizer` 是独立可视化模块，不参与模型数值计算，也不应成为
TensorRT runtime 的强制依赖。

## 4. 当前服务器文件位置

### 4.1 数据集输入

数据根目录：

```text
/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
```

Main4 验证集 annotation：

```text
/mnt/diskNvme1/hyh/data/company_20260818_30k_tuning/blocked_main4_fov120_v1/custom_infos_val_sweep.pkl
```

`blocked_main4_fov120_v1` 目录只有划分后的 annotation，不是图片数据根目录。
把它设置为 `data_root` 会出现 `images_undistorted/...` 找不到的问题。

导出前必须设置：

```bash
export RACFORMER_DEPLOY_DATA_ROOT=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
export RACFORMER_DEPLOY_ANN_FILE=/mnt/diskNvme1/hyh/data/company_20260818_30k_tuning/blocked_main4_fov120_v1/custom_infos_val_sweep.pkl
```

### 4.2 已有 PyTorch 结果

旧的动态训练几何基线：

```text
outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_traincalib/main4_pytorch_sample0.npz
```

该次结果为 26 个 detection。它使用训练数据中逐帧变化的位姿，不能作为新的静态
几何 TensorRT fixture 的数值参考。

### 4.3 当前静态几何产物目录

服务器宿主机：

```text
/home/ubuntu/hyh/RaCFormer/outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom
```

容器对应路径：

```text
/workspace/outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom
```

静态 PyTorch 基线计划保存为：

```text
main4_pytorch_static_geometry_sample0.npz
```

完整模型导出文件已经生成：

```text
racformer_50m_q200_fov120_p15_main4_e8_raw_trt85.onnx
racformer_50m_q200_fov120_p15_main4_e8_model_sample0.npz
export_main4_model_trt85.txt
```

完整模型导出结果：

```text
fixed view geometry: True
fixed view geometry frame max error: 0.00000000
static radar padding comparison passed: True
fixed BEV rank count: 39944
fixed BEV interval count: 2781
boundary comparison passed: True
legacy/current detection count: 40/40
boxes max_abs_error: 0.00061393
scores max_abs_error: 0.00000748
labels equal: True
decoded boundary comparison passed: True
IsInf nodes remaining: 0
LayerNormalization nodes remaining: 0
onnx checker: PASS
status: SUCCESS
```

model fixture 包含 22 个数组。该结果证明固定几何、静态雷达 padding、TRT 8.5
图改写和完整模型 ONNX 边界都已通过，可以继续拆分三个部署子图。

### 4.4 转换工具与输入输出关系

这里没有使用 `trtexec` 直接把 `.pth` 转成 engine。转换被明确拆成以下阶段：

| 阶段 | Python 工具 | 输入 | 输出 |
| --- | --- | --- | --- |
| PyTorch 基线 | `deploy.offline_demo` | config + `.pth` + dataset | PyTorch detection `.npz` |
| 完整 ONNX | `deploy.export_onnx` | config + `.pth` + dataset | raw ONNX + model fixture + report |
| Frontend ONNX | `deploy.export_frontend_onnx` | config + `.pth` + model fixture | frontend precompute ONNX + shared fixture |
| Decoder ONNX | `deploy.export_decoder_recurrent_layer` | config + `.pth` + model fixture | 单层 recurrent decoder ONNX |
| 提取 Image/LSS | `deploy.tensorrt.extract_onnx_subgraph` | frontend ONNX | Image/LSS ONNX |
| 提取 Radar | `deploy.tensorrt.extract_onnx_subgraph` | frontend ONNX | Radar ONNX |
| TRT 解析 | `deploy.tensorrt.parse_onnx` | 三个拆分 ONNX + plugin | 三份 parser report |
| Engine 构建 | `deploy.tensorrt.build_engine` | 拆分 ONNX + plugin | 三个 `.engine` + build report |
| 精度与测速 | `deploy.tensorrt.validate_frontend_decoder_numpy` | 三 engine + shared fixture + plugin | decoded validation 和分阶段延迟 report |

`.pth` 只在服务器原生 PyTorch 环境中由三个 exporter 读取。TensorRT 容器不读取
checkpoint；它只读取已经导出的 ONNX、fixture 和 plugin。

## 5. 服务器完整导出流程

### 5.1 初始化变量

在服务器宿主机的 `racformer_wp` 环境中执行：

```bash
cd ~/hyh/RaCFormer
conda activate racformer_wp

git pull --ff-only
git merge-base --is-ancestor d96f4e3 HEAD \
  && echo "static geometry commit OK"

CONFIG=configs/deploy/racformer_company_front_50m_q200_fov120_p15_main4_left_pytorch_f4.py
CKPT=/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning/full_main4_f4_fov120_p15_e8/20260831_184101/model/best_company/3D_mAP@0.5_epoch_8.pth
OUT=outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom

export RACFORMER_DEPLOY_DATA_ROOT=/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
export RACFORMER_DEPLOY_ANN_FILE=/mnt/diskNvme1/hyh/data/company_20260818_30k_tuning/blocked_main4_fov120_v1/custom_infos_val_sweep.pkl

mkdir -p "$OUT"
```

### 5.2 PyTorch 静态几何基线

```bash
python -m deploy.offline_demo \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --split val \
  --sample-index 0 \
  --device cuda:0 \
  --out "$OUT/main4_pytorch_static_geometry_sample0.npz"
```

静态几何结果可能与旧的 26 个 detection 不同，这是预处理契约改变后的正常现象。

### 5.3 完整模型 ONNX 与 model fixture

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
  --static-radar-voxels 1024 \
  --out "$OUT/racformer_50m_q200_fov120_p15_main4_e8_raw_trt85.onnx" \
  --fixture "$OUT/racformer_50m_q200_fov120_p15_main4_e8_model_sample0.npz" \
  --report "$OUT/export_main4_model_trt85.txt"
```

继续之前必须看到 `status: SUCCESS`，并确认固定几何差值为 0 或浮点误差级别。

### 5.4 拆分 frontend 和 recurrent decoder

计划文件名统一如下：

```text
racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_v2_trt85.onnx
racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_sample0.npz
racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt85.onnx
export_main4_frontend_precompute_v2_trt85.txt
export_main4_decoder_precompute_v2_trt85.txt
```

实际导出结果（2026-09-01）：

```text
frontend precompute ONNX: 172 MB, onnx checker PASS, status SUCCESS
frontend precompute fixture: 110 MB
recurrent decoder ONNX: 80 MB, onnx checker PASS, status SUCCESS
```

执行：

```bash
python -m deploy.export_frontend_onnx \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$OUT/racformer_50m_q200_fov120_p15_main4_e8_model_sample0.npz" \
  --device cuda:0 \
  --opset 17 \
  --precompute-bev-values \
  --out "$OUT/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_v2_trt85.onnx" \
  --fixture "$OUT/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_sample0.npz" \
  --report "$OUT/export_main4_frontend_precompute_v2_trt85.txt"

python -m deploy.export_decoder_recurrent_layer \
  --config "$CONFIG" \
  --weights "$CKPT" \
  --model-fixture "$OUT/racformer_50m_q200_fov120_p15_main4_e8_model_sample0.npz" \
  --device cuda:0 \
  --opset 17 \
  --precompute-bev-values \
  --out "$OUT/racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt85.onnx" \
  --report "$OUT/export_main4_decoder_precompute_v2_trt85.txt"
```

后续三 engine 验证必须使用 `frontend_precompute_sample0.npz`，不能使用
`model_sample0.npz`。后者缺少 recurrent decoder 调度所需的完整 fixture 字段。

## 6. TensorRT 8.5 容器完整命令

### 6.1 进入容器并定义路径

在服务器宿主机执行前两行，后续命令全部在容器内执行：

```bash
docker start racformer_trt85_l20
docker exec -it racformer_trt85_l20 bash

cd /workspace/RaCFormer

ONNX=/workspace/outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom
TRT=/workspace/outputs/deploy_tensorrt_50m_q200_fov120_p15_main4_e8_staticgeom
PLUGIN=/workspace/RaCFormer/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so

mkdir -p "$TRT"

ls -lh \
  "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_v2_trt85.onnx" \
  "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_sample0.npz" \
  "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt85.onnx" \
  "$PLUGIN"

python -c \
  "import ctypes; ctypes.CDLL('$PLUGIN'); print('plugin load: PASS')"
```

### 6.2 从 frontend ONNX 提取两个独立子图

Image/LSS：

```bash
python -m deploy.tensorrt.extract_onnx_subgraph \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_v2_trt85.onnx" \
  --output image_feat_0 \
  --output image_feat_1 \
  --output image_feat_2 \
  --output image_feat_3 \
  --output lss_bev_value \
  --out "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt85.onnx" \
  --report "$ONNX/extract_main4_frontend_image_lss_trt85.txt"
```

Radar：

```bash
python -m deploy.tensorrt.extract_onnx_subgraph \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_v2_trt85.onnx" \
  --output radar_bev_value \
  --out "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt85.onnx" \
  --report "$ONNX/extract_main4_frontend_radar_trt85.txt"
```

检查：

```bash
grep -H -E 'onnx checker:|status:' \
  "$ONNX"/extract_main4_frontend_*.txt
```

两个报告都必须为 `onnx checker: PASS` 和 `status: SUCCESS`。

### 6.3 用 TensorRT 8.5.2 解析三个 ONNX

```bash
python -m deploy.tensorrt.parse_onnx \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt85.onnx" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_main4_frontend_image_lss_trt852_l20.txt"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt85.onnx" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_main4_frontend_radar_trt852_l20.txt"

python -m deploy.tensorrt.parse_onnx \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt85.onnx" \
  --plugin "$PLUGIN" \
  --fail-on-zero-dim \
  --out "$TRT/parse_main4_decoder_precompute_v2_trt852_l20.txt"
```

统一检查：

```bash
grep -H -E \
'TensorRT version:|status:|parser errors:|zero-dimension tensors:|zero-dimension execution tensors:|RuntimeError|FAILED' \
  "$TRT"/parse_main4_*.txt
```

验收要求是 `status: PASS`、`parser errors: 0`、
`zero-dimension execution tensors: 0`。实际结果为 Image/LSS 0、Radar 15、
Decoder 35 个零维 shape tensor，但三者的 execution tensor 均为 0，已经通过。

### 6.4 构建三个 L20 engine

先统一定义文件名：

```bash
IMAGE_ENGINE="$TRT/racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt852_l20_fp16.engine"
RADAR_ENGINE="$TRT/racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt852_l20_fp16.engine"
DECODER_ENGINE="$TRT/racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt852_l20_fp32.engine"
FIXTURE="$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_precompute_sample0.npz"
```

三个 build 必须串行执行，不要并行争抢 GPU 显存和 TensorRT workspace。

Image/LSS 使用 FP16：

```bash
python -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt85.onnx" \
  --engine "$IMAGE_ENGINE" \
  --plugin "$PLUGIN" \
  --fp16 \
  --workspace-gb 8 \
  --out "$TRT/build_main4_frontend_image_lss_trt852_l20_fp16.txt"
```

Radar 使用 FP16：

```bash
python -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt85.onnx" \
  --engine "$RADAR_ENGINE" \
  --plugin "$PLUGIN" \
  --fp16 \
  --workspace-gb 8 \
  --out "$TRT/build_main4_frontend_radar_trt852_l20_fp16.txt"
```

Decoder 使用严格 FP32，不能添加 `--fp16`：

```bash
python -m deploy.tensorrt.build_engine \
  --onnx "$ONNX/racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt85.onnx" \
  --engine "$DECODER_ENGINE" \
  --plugin "$PLUGIN" \
  --workspace-gb 8 \
  --out "$TRT/build_main4_decoder_precompute_v2_trt852_l20_fp32.txt"
```

检查三次构建：

```bash
grep -H -E \
'TensorRT version:|engine:|precision:|build time:|engine size:|status:|FAILED|RuntimeError|Error' \
  "$TRT"/build_main4_*.txt

ls -lh "$IMAGE_ENGINE" "$RADAR_ENGINE" "$DECODER_ENGINE"
```

首选精度组合是 Image/LSS FP16、Radar FP16、Decoder strict FP32。Radar FP16
若数值验证失败则单独回退 FP32；不能仅凭 build 成功接受 FP16 engine。

实际 L20 构建结果（2026-09-01）：

```text
Image/LSS FP16: SUCCESS, 101.26 MB, 757.552 s
Radar FP16:     SUCCESS,   9.68 MB, 510.844 s
Decoder FP32:   SUCCESS,  80.20 MB, 256.346 s
```

### 6.5 三 engine decoded validation 和分阶段测速

```bash
REPORT="$TRT/validate_main4_three_engine_trt852_l20.txt"

python -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine "$IMAGE_ENGINE" \
  --radar-frontend-engine "$RADAR_ENGINE" \
  --decoder-engine "$DECODER_ENGINE" \
  --fixture "$FIXTURE" \
  --plugin "$PLUGIN" \
  --initial-query-from-fixture \
  --accept-decoded-match \
  --profile-stages \
  --warmup 20 \
  --iters 100 \
  --atol 0.03 \
  --out "$REPORT"
```

提取精度、显存和耗时：

```bash
grep -E \
'all_cls_scores:|all_bbox_preds:|actual/reference detection count:|boxes close:|scores close:|labels equal:|decoded comparison passed:|end-to-end engine GPU latency:|frontend GPU latency:|radar frontend GPU latency:|recurrent decoder GPU latency:|decoder iteration [0-9]+ GPU latency:|resident CUDA memory delta:|deployment acceptance passed:|status:' \
  "$REPORT"
```

验收必须同时满足：

- detection count 相同；
- `boxes close: True`；
- `scores close: True`；
- `labels equal: True`；
- `decoded comparison passed: True`；
- `deployment acceptance passed: True`；
- `status: SUCCESS`。

仅 ONNX checker、parser 或 engine build 成功都不代表部署数值正确。

首次 `Image FP16 + Radar FP16 + Decoder FP32` 验证失败：

```text
decoder polar radius: 50.000000 m
all_cls_scores max_abs_error: 2.04384089
all_bbox_preds max_abs_error: 4.71020412
actual/reference detection count: 39/40
boxes close: False
scores close: False
labels equal: False
decoded comparison passed: False
deployment acceptance passed: False
status: FAILED
```

这不是可用 3 cm 门限接受的边界误差。后续按以下顺序隔离精度来源：

1. 新建 Radar FP32，测试 Image FP16 + Radar FP32 + Decoder FP32；
2. 若仍失败，新建 Image FP32，先验证全 FP32 可靠基线；
3. 全 FP32 通过后，再用 Image FP32 + Radar FP16 判断 Radar FP16 是否可单独使用。

首次失败运行的延迟分布明显双峰（端到端 p50 38.251 ms、p95 112.819 ms），
在精度失败且 GPU 运行状态不稳定时不能作为最终性能数据。

第二次 `Image FP16 + Radar FP32 + Decoder FP32` 验证仍失败：

```text
all_cls_scores max_abs_error: 2.05019331
all_bbox_preds max_abs_error: 1.37683725
actual/reference detection count: 38/40
boxes close: False
scores close: False
labels equal: False
decoded comparison passed: False
deployment acceptance passed: False
status: FAILED
```

Radar 改为 FP32 后 bbox 最大误差从 4.7102 m 降至 1.3768 m，但约 2.05 的
分类最大误差仍然存在，说明 Image FP16 也不可靠；Radar FP16 同时可能贡献了
部分 bbox 误差。下一步必须构建 Image FP32 并先验证全 FP32 基线。

第三次全 FP32 验证仍失败，因此问题不能归因于 FP16：

```text
all_cls_scores max_abs_error: 1.75270605
all_bbox_preds max_abs_error: 0.57533929
actual/reference detection count: 41/40
elements above atol: cls 9/4800, bbox 29/12000
decoded comparison passed: False
deployment acceptance passed: False
status: FAILED
```

绝大多数元素已经接近参考，但 query 34 在第六次 decoder 迭代发生明显分叉。
完整模型 ONNX 的 decoded boundary 曾通过 40/40，因此后续停止精度组合试验，
先用 `validate_decoder_loop_numpy` 绕过两个 frontend，隔离 recurrent decoder
engine；若 decoder-only 通过，再分别检查 Image/LSS 和 Radar frontend 输出边界。

Decoder-only FP32 隔离验证已经成功：

```text
all_cls_scores max_abs_error: 0.00371933
all_bbox_preds max_abs_error: 0.00505805
actual/reference detection count: 40/40
boxes max_abs_error: 0.00100172
scores max_abs_error: 0.00001678
labels equal: True
decoded comparison passed: True
deployment acceptance passed: True
status: SUCCESS
```

因此 recurrent decoder engine、50 m polar radius 和六次调度均正确。问题位于
frontend split 输出进入 decoder 的边界，下一步使用
`--frontend-output-from-fixture` 分别替换 Radar 与 Image/LSS 输出进行隔离。

Frontend 分支隔离结果：

```text
真实 Image/LSS FP32 + fixture Radar:
  detections 41/40, query 34 分叉, status FAILED

fixture Image/LSS + 真实 Radar FP32:
  detections 40/40
  boxes max_abs_error 0.01511276
  scores max_abs_error 0.00081331
  labels equal True
  decoded comparison passed True
  status SUCCESS
```

因此 Radar FP32 已排除，错误明确位于 Image/LSS frontend。下一步分别用 fixture
替换 `lss_bev_value` 和四个 `image_feat_*` 输出，继续区分 image backbone/neck
特征与 LSS BEV 分支。

Image features 与 LSS 分组隔离结果：

```text
真实 image_feat_0..3 + fixture LSS/Radar:
  detections 41/40, query 34 分叉, status FAILED

真实 lss_bev_value + fixture image_feat_0..3/Radar:
  detections 40/40
  all_cls_scores max_abs_error 0.00269413
  all_bbox_preds max_abs_error 0.00713301
  boxes max_abs_error 0.00179803
  labels equal True
  status SUCCESS
```

因此 LSS/BEV pooling 分支已经排除，问题明确位于四个 image backbone/neck
feature 输出。下一步逐个放行 `image_feat_0` 至 `image_feat_3`，判断单一尺度错误
还是多尺度小误差叠加。

逐尺度 decoded 隔离结果：

```text
image_feat_0: 40/40, boxes max error 0.00483435, SUCCESS
image_feat_1: 40/40, boxes max error 0.00653982, SUCCESS
image_feat_2: 41/40, query 34 分叉, FAILED
image_feat_3: 40/40, labels/scores一致, boxes max error 0.03144479, FAILED
```

`image_feat_2` 单独完整复现主故障，是主要根因。`image_feat_3` 未改变 detection
数量、标签或分数，但框最大误差比 3 cm 门限高 1.44 mm，是次要边界问题。
`image_feat_0/1` 已通过。下一步直接比较 Image FP32 engine 的五个原始输出与
frontend fixture，量化各 feature 在进入 decoder 前的实际误差。

Image FP32 engine 原始输出比较：

```text
image_feat_0: max 0.20073843, mean 0.01572888
image_feat_1: max 0.16769409, mean 0.01372290
image_feat_2: max 0.12721252, mean 0.01066186
image_feat_3: max 0.05047607, mean 0.00502714
lss_bev_value: max 0.03595734, mean 0.00005196
```

四个 image feature 都存在 TensorRT 与 fixture 的浮点累计差异；`image_feat_2`
并不是原始误差最大的输出，但 decoder/query 34 只对该尺度表现出灾难性敏感性。
因此当前更准确的根因是“模型对 level-2 特征数值扰动不稳定”，而不是该输出出现
独有的大范围计算错误。下一步用不同 TensorRT tactic（先试 builder optimization
level 0）重建严格 FP32 Image engine，验证能否避开当前数值路径。

TensorRT 8.5.2.2 不暴露 `builder_optimization_level`，因此 opt-level-0 构建在
创建 builder config 后立即失败，没有生成 engine：

```text
RuntimeError: this TensorRT version does not expose builder_optimization_level
```

后续只能使用 TRT 8.5 支持的方式改变 tactic，例如收紧 workspace 或限制 tactic
source。即使某个 L20 tactic 在 sample-0 上通过，也必须在多样本和 Nano 重新验证，
因为 Orin 会重新选择不同的硬件 tactic。

将 workspace 从 8 GB 收紧到 1 GB 后，严格 FP32 Image engine 仍逐项复现原结果：

```text
all_cls_scores max_abs_error: 1.75267267
all_bbox_preds max_abs_error: 0.57531738
detections: 41/40
query 34 分叉
status: FAILED
```

这说明相关层选择没有随 workspace 改变；不再尝试 2 GB、4 GB 等中间值。
下一步检查 `image_feat_2` 的 ONNX producer 与 TensorRT 相邻层，然后选择 identity
fusion barrier 或 tactic-source 限制实验。

`image_feat_2` 的直接尾部已经确认：

```text
/Transpose_2_output_0
  -> ONNX /Transpose_2
  -> ONNX /Reshape_17
  -> image_feat_2

TensorRT layer 1712: SHUFFLE /Reshape_17
```

`Transpose` 与 `Reshape` 不进行浮点算术，因此原始 feature 误差来自
`/Transpose_2_output_0` 更上游。第一轮图修复在 `/Transpose_2` 的 input 0 前插入
`racformer_identity` plugin，建立明确 fusion boundary，再重新 parse、build 和做
decoded validation；若结果不变，再继续沿数据输入向上追踪实际数值 producer。

当前 41/40 数值失败与实机 640×480/约 65.5° 相机不匹配是两个独立问题。
数值比较的 PyTorch reference 和 TensorRT 都读取同一个由 1920×1080 训练图像生成
的 640×256 fixture，实机图像没有进入该测试。因此 FOV 不匹配不是 query 34
分叉的原因；但它仍会影响最终真实输入精度，当前 `staticgeom` 只能作为训练视图下
的 TensorRT 技术基线，不能作为最终实机验收包。

## 7. Nano 阶段与最终目录

服务器的 L20 `.engine` 和 x86_64 plugin `.so` 不能传到 Nano 执行。传输路线固定为：

```text
服务器导出产物 -> 公司本地电脑 -> 数据线/SSH -> Nano
Git 提交/代码     -> 公司本地电脑 -> 数据线/SSH -> Nano
```

Nano 项目根目录：

```text
/home/cttest/RaCFormer
```

计划目录：

```text
/home/cttest/RaCFormer/outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom
/home/cttest/RaCFormer/outputs/deploy_tensorrt_50m_q200_fov120_p15_main4_e8_staticgeom
/home/cttest/RaCFormer/outputs/deploy_runtime_50m_q200_fov120_p15_main4_e8_staticgeom/constants
/home/cttest/RaCFormer/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so
/home/cttest/RaCFormer/build/racformer_runtime/libracformer_runtime.so
```

Nano 上重新构建的计划 engine：

```text
racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt852_orin_fp16.engine
racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt852_orin_fp16.engine
racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt852_orin_fp32.engine
```

Nano 必须完成与 L20 相同的 decoded detection 一致性验证，然后才生成 runtime
constants、编译 C++ runtime 并做四帧真实输入测试。L20 性能不能代替 Nano 性能。

## 8. Runtime 输入输出契约

输入接口使用 JPEG 原始字节和雷达点数组：

```text
camera: timestamp(ns), frame_id, version, data_size, JPEG byte pointer
radar:  timestamp(ns), frame_id, version, point_count, radar point pointer
```

同一帧优先使用 `frame_id` 配对，时间戳单位统一为纳秒。runtime 维护四帧时序环形
缓存，在相机和雷达都到齐后触发推理。

部署雷达坐标定义：

```text
x: 左
y: 前
z: 上
```

模型 ego 坐标定义：

```text
x: 前
y: 左
z: 上
```

因此进入模型前的基本轴变换为：

```text
x_model = y_radar
y_model = x_radar
z_model = z_radar
```

平台固定，`ego_speed=0`。如果雷达只提供径向速度，仍需按点的方位把径向速度
分解到模型平面的 `vx/vy`；不能把同一个径向速度同时填入两个分量。

输出为一帧可变数量的 3D detection 数组，每个检测包含：

```text
x, y, z, dx, dy, dz, yaw, vx, vy, score, label
```

Main4 类别映射必须随部署包交付：

```text
0 car
1 truck
2 bicycle
3 pedestrian
```

## 9. 标定状态和重要限制

当前 `staticgeom` 目录是使用训练数据标定建立的服务器部署基线。它用于验证：

- 固定平台四帧预处理；
- ONNX 导出；
- TensorRT 8.5 解析和建图；
- 三 engine 精度和性能。

它不是最终 640×480 实机相机标定包。

目前已知实机相机内参约为：

```text
fx=497.674873, fy=497.006587
cx=330.025881, cy=241.449375
image size=640×480
distortion=[0.09124804, -0.07861055, 0, 0, 0]
```

目前测得的一组雷达到相机候选变换为：

```text
 0.9998477   0.01745241   0.0000000   0.0000
 0.0000000   0.00000000  -1.0000000   0.0000
-0.01745241  0.99984770   0.0000000   0.2000
 0.0000000   0.00000000   0.0000000   1.0000
```

该矩阵在写入最终 runtime 前必须结合“雷达 x 左、y 前、z 上”的真实轴定义，
通过雷达点叠加图再次确认矩阵方向和左右符号；不能只凭矩阵文件名判断是
`radar_to_camera` 还是其逆矩阵。

该内参对应的水平视场明显小于模型训练使用的 120°。因此这版 Main4 可以完成
TensorRT 流程验证，但实际精度结论必须谨慎。正在训练的约 60° 相机版本完成后，
需要建立新的配置、ONNX、fixture、三个 engine 和 runtime constants，不能复用本次
Main4 engine。

最终实机包还需要确认并固化：

- 640×480 去畸变方式；
- 640×480 到 640×256 的 crop/resize 规则；
- crop/resize 后的新内参；
- 雷达坐标轴变换和雷达到相机外参方向；
- `lidar2img.bin`、`img2lidar.bin`、`mlp_input.bin`；
- runtime 中使用的固定 `radar_to_ego` 配置。

标定或图像预处理改变后，即使网络权重不变，也必须重新生成 fixture、ONNX、
runtime constants 和 engine。

## 10. 交付包内容

每个最终模型包至少包含：

```text
GIT_COMMIT.txt
MODEL_ID.txt
classes.txt
SHA256SUMS

三个拆分 ONNX
frontend_precompute fixture (.npz)
三个 Orin engine
aarch64 TensorRT plugin (.so)
C++ runtime library 和 public headers
runtime constants 及 manifest.tsv
导出、parser、build、L20 validation、Nano validation 报告
环境清单和构建命令记录
```

`MODEL_ID.txt` 应记录 checkpoint 绝对路径、checkpoint SHA256、配置路径、Git
commit、类别、范围、FOV、Query 数、帧数、精度组合、标定版本和构建设备。

## 11. 当前进度

| 阶段 | 状态 |
| --- | --- |
| Main4 专用部署配置 | 完成 |
| 50m 矩形与 FOV120 扇形交集过滤 | 完成 |
| 固定平台静态几何支持 | 完成并推送，commit `d96f4e3` |
| PyTorch 动态训练几何 sample-0 | 完成，26 detections，仅供旧基线参考 |
| PyTorch 静态几何 sample-0 | 正在执行/待确认 |
| 完整模型静态几何 ONNX | 完成；40/40 decoded boundary 一致，`status: SUCCESS` |
| frontend/decoder 拆分导出 | 完成；frontend 172 MB、fixture 110 MB、decoder 80 MB，两个 checker/status 均通过 |
| L20 TRT 8.5 三图 parser | 完成；三图 PASS、parser errors 0、zero-dimension execution tensors 0 |
| L20 TRT 8.5 三 engine 构建 | 完成；Image FP16 101.26 MB、Radar FP16 9.68 MB、Decoder FP32 80.20 MB |
| L20 decoded validation | 主故障定位到 `image_feat_2`；`image_feat_3` 为 3.144 cm 次要边界误差；`image_feat_0/1` 通过 |
| 本地中转归档 | 待执行 |
| Nano plugin/engine 重建 | 待执行 |
| Nano decoded validation 与测速 | 待执行 |
| 最终 640×480 实机标定版本 | 待图像预处理和外参最终确认 |

每完成一步，应把实际文件名、SHA256、报告路径和结果补回本节，避免只在终端或
聊天记录中保存关键信息。
