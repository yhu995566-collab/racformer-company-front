# Q1 服务器 TensorRT 8.5 与 FP16 实验

最后更新：2026-08-11

本阶段只在服务器验证 Q1，不进行 Nano 部署。已知位置：

```text
repository: /home/ubuntu/hyh/3DH-Query
checkpoint: /home/ubuntu/hyh/3DH-Query/outputs/3dh_query_q1/2026-08-10/15-18-38/epoch_36.pth
host output: /home/ubuntu/hyh/3DH-Query/outputs
container code: /workspace/3DH-Query
container output: /workspace/outputs
```

先验证 Q1 混合精度基线，再分别测试 decoder FP16、radar FP16 和二者同时
FP16。每个变量单独测试，避免全 FP16 失败后无法定位来源。

## 1. 宿主机同步和确认 checkpoint

```bash
cd /home/ubuntu/hyh/3DH-Query
git checkout q1-tensorrt-deployment
git pull --ff-only origin q1-tensorrt-deployment
git rev-parse --short HEAD

ls -lh outputs/3dh_query_q1/2026-08-10/15-18-38/epoch_36.pth
sha256sum outputs/3dh_query_q1/2026-08-10/15-18-38/epoch_36.pth
```

## 2. 宿主机导出 Q1 ONNX

在原来能够训练和导出的 Python 环境执行，不在 TRT 容器中执行：

```bash
cd /home/ubuntu/hyh/3DH-Query
conda activate racformer_wp

CKPT="$PWD/outputs/3dh_query_q1/2026-08-10/15-18-38/epoch_36.pth"

CUDA_VISIBLE_DEVICES=0 \
STATIC_RADAR_VOXELS=1024 \
OUTPUT_ROOT="$PWD/outputs" \
bash scripts/deploy_q1_export.sh "$CKPT"
```

检查导出报告：

```bash
grep -H -E 'status:|decoded boundary comparison passed:|PyTorch recurrent loop close:' \
  outputs/deploy_onnx_q1/export_3dh_query_q1_*_trt85.txt
```

## 3. 从旧容器快照并创建新 TRT 8.5 容器

旧容器可能包含创建后安装的软件，因此不要只按原始 image 名重新运行。
先把已验证的 `racformer_trt85_l20` 文件系统提交成新 image。挂载目录不会
进入 image，下一节会重新挂载 Q1 目录。

推荐直接运行带检查和 manifest 的脚本：

```bash
cd /home/ubuntu/hyh/3DH-Query
bash scripts/create_q1_trt85_container.sh
```

脚本从 `racformer_trt85_l20` 快照
`racformer-trt85-q1:trt852-20260811`，创建 `q1_trt85_l20`，挂载 Q1 代码和
输出目录，检查 checkpoint 与 TRT `8.5.2.2`，并写出：

```text
outputs/deploy_tensorrt_q1/q1_trt85_container_manifest.txt
```

如果目标容器已存在，脚本会拒绝覆盖，必须人工判断是否保留。等价的手工
命令仍可通过阅读脚本获得。

## 4. 确认新容器

确认挂载与 TensorRT：

```bash
docker inspect -f \
  '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' \
  q1_trt85_l20

docker exec q1_trt85_l20 \
  python -c 'import tensorrt as trt; print(trt.__version__)'

docker exec q1_trt85_l20 \
  ls -lh /workspace/outputs/deploy_onnx_q1
```

TensorRT 必须显示 `8.5.2.2`。

## 5. 新容器内重新编译 L20 插件

```bash
docker exec -it q1_trt85_l20 bash

cd /workspace/3DH-Query
cmake -S deploy/tensorrt/plugins/bev_pool_v2 \
  -B build/tensorrt_plugins_trt852_l20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/tensorrt_plugins_trt852_l20 --parallel

PLUGIN="$PWD/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so"
file "$PLUGIN"
ldd "$PLUGIN"
python -c "import ctypes; ctypes.CDLL('$PLUGIN'); print('plugin load: PASS')"
```

## 6. 先运行混合精度基线

```bash
cd /workspace/3DH-Query

OUTPUT_ROOT=/workspace/outputs \
WORKSPACE_GB=8 \
WARMUP=20 \
ITERS=20 \
ATOL=0.03 \
bash scripts/deploy_q1_tensorrt.sh l20
```

基线精度是：

```text
image/LSS frontend: FP16 tactics enabled
radar frontend: FP32
initial query: fixture FP32
recurrent decoder x6: FP32
```

只有基线 decoded 验收通过后才进入 FP16 矩阵实验。

## 7. 测试 Q1 FP16 矩阵

```bash
OUTPUT_ROOT=/workspace/outputs \
WORKSPACE_GB=8 \
WARMUP=20 \
ITERS=20 \
ATOL=0.03 \
bash scripts/test_q1_fp16_matrix.sh l20
```

脚本额外构建一次 radar FP16 和 decoder FP16 Engine，然后验证：

1. image FP16 + radar FP32 + decoder FP16
2. image FP16 + radar FP16 + decoder FP32
3. image FP16 + radar FP16 + decoder FP16

某个组合失败后仍继续测试其余组合。汇总报告：

```text
/workspace/outputs/deploy_tensorrt_q1/validate_3dh_query_q1_fp16_matrix_trt852_l20_summary.txt
```

`--fp16` 表示允许 TensorRT 选择 FP16 tactic，不保证每层或插件都变成
FP16。最终以 decoded detections 和 profile 为准，不能只根据 Engine 文件名
判断精度或性能。

## 8. 执行记录

| 日期 | 环境 | 操作 | 结果 | 证据 |
|---|---|---|---|---|
| 2026-08-11 | `q1_trt85_l20` | 编译 L20 TensorRT 插件 | PASS | `libracformer_bev_pool_v2_trt.so`，156368 bytes，`ctypes.CDLL` 加载通过 |
| 2026-08-11 | `q1_trt85_l20` | Q1 混合精度 Engine 构建 | PASS | image/LSS FP16、radar FP32、decoder FP32 均构建成功 |
| 2026-08-11 | `q1_trt85_l20` | 首次联合验证 | VALIDATOR BUG | 分类和检测类别一致，但验证器残留固定 65 m 极径及 200 m 后处理范围；不是 Engine 构建失败 |
| 2026-08-11 | `q1_trt85_l20` | 修正 Q1 范围后重新联合验证 | PASS | 19/19 detections；box 最大误差 0.00666046 m；score 最大误差 0.00023659；labels 一致 |

本次插件依赖记录：

```text
architecture: x86_64
libnvinfer: /lib/x86_64-linux-gnu/libnvinfer.so.8
libcudart: /usr/local/cuda/lib64/libcudart.so.12
plugin: /workspace/3DH-Query/build/tensorrt_plugins_trt852_l20/libracformer_bev_pool_v2_trt.so
```

该 `.so` 只用于服务器 L20 实验，禁止传到 aarch64 Nano。服务器 CUDA
runtime 与 Nano CUDA 11.4 不同，因此服务器结果用于判断 Q1 精度组合是否
可行；最终 Engine、插件、耗时和显存仍必须在 Nano 本地重新生成和测量。

混合精度基线的 raw 输出记录：

```text
all_cls_scores: max_abs_error=0.02356100, mean_abs_error=0.00023250
all_bbox_preds: max_abs_error=0.31193542, mean_abs_error=0.00072453
decoded detections: 19/19
decoded boxes: close=True, max_abs_error=0.00666046 m
decoded scores: close=True, max_abs_error=0.00023659
decoded labels: equal=True
deployment acceptance: True
```

raw bbox 最大误差来自未进入最终 decoded detections 的 query；最终检测框
最大误差为 6.66 mm，低于当前 3 cm 验收门限，因此基线正式通过。
