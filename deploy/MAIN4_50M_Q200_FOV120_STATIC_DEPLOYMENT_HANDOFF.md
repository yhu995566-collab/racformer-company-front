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

### 4.3 当前正在生成的静态几何产物目录

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

完整模型导出文件计划为：

```text
racformer_50m_q200_fov120_p15_main4_e8_raw_trt85.onnx
racformer_50m_q200_fov120_p15_main4_e8_model_sample0.npz
export_main4_model_trt85.txt
```

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

## 6. TensorRT 8.5 容器阶段

容器内定义：

```bash
cd /workspace/RaCFormer

ONNX=/workspace/outputs/deploy_onnx_50m_q200_fov120_p15_main4_e8_staticgeom
TRT=/workspace/outputs/deploy_tensorrt_50m_q200_fov120_p15_main4_e8_staticgeom
PLUGIN=/workspace/RaCFormer/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so

mkdir -p "$TRT"
```

需要从 frontend precompute ONNX 提取两个子图：

```text
racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt85.onnx
racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt85.onnx
```

三张图都必须先通过 `deploy.tensorrt.parse_onnx`，验收要求：

```text
status: PASS
parser errors: 0
zero-dimension execution tensors: 0
```

计划构建的 L20 engine：

```text
racformer_50m_q200_fov120_p15_main4_e8_frontend_image_lss_trt852_l20_fp16.engine
racformer_50m_q200_fov120_p15_main4_e8_frontend_radar_trt852_l20_fp16.engine
racformer_50m_q200_fov120_p15_main4_e8_decoder_precompute_v2_trt852_l20_fp32.engine
```

首选精度组合：

```text
Image/LSS frontend: FP16
Radar frontend: FP16
Recurrent decoder: strict FP32，循环 6 次
Initial query: fixture 中的 FP32
```

Radar FP16 已在上一模型族通过，但本模型仍必须重新验证。若 Radar FP16 未通过，
回退到 Radar FP32；decoder 暂不使用 FP16。

最终 L20 验证使用：

```text
deploy.tensorrt.validate_frontend_decoder_numpy
--initial-query-from-fixture
--accept-decoded-match
--atol 0.03
--profile-stages
```

验收必须同时满足：

- detection count 相同；
- `boxes close: True`；
- `scores close: True`；
- `labels equal: True`；
- `decoded comparison passed: True`；
- `deployment acceptance passed: True`；
- `status: SUCCESS`。

仅 ONNX parser 或 engine build 成功不代表部署成功。

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
| 完整模型静态几何 ONNX | 正在执行/待确认 |
| frontend/decoder 拆分导出 | 待执行 |
| L20 TRT 8.5 三 engine 构建 | 待执行 |
| L20 decoded validation | 待执行 |
| 本地中转归档 | 待执行 |
| Nano plugin/engine 重建 | 待执行 |
| Nano decoded validation 与测速 | 待执行 |
| 最终 640×480 实机标定版本 | 待图像预处理和外参最终确认 |

每完成一步，应把实际文件名、SHA256、报告路径和结果补回本节，避免只在终端或
聊天记录中保存关键信息。
