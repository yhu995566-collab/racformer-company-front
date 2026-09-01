# RaCFormer 公司数据训练交接

最后更新：2026-09-01

本文只记录数据转换、数据划分、训练、验证、可视化、模型选择和 checkpoint。
ONNX、TensorRT、Nano、插件和 C++ runtime 完全不在本文展开，相关内容见
[TensorRT 部署交接](DEPLOYMENT_HANDOFF.md)。训练 checkpoint 与部署 engine 不能仅凭
文件名混用，部署前必须同时核对类别数、检测范围、query 数、时序帧数、FOV、配置和
标定。

## 1. 当前结论和接手优先级

当前产品方向是：

| 项目 | 当前约定 |
| --- | --- |
| 相机 | 前向 60°，采集目录 `front60_camera` |
| 雷达 | 前向 120° |
| 最终检测区域 | 前方 50 m、水平 120° |
| 三维范围 | `[0, -20, -3, 50, 20, 3]` m，再叠加 120° FOV 过滤 |
| Query | 200，`front_fov_grid`，距离 power 1.5 |
| 时序 | 4 帧（当前帧加 3 个 sweep） |
| 类别 | Main3：`car, truck, bicycle` |
| 网络输入 | 原图经标定和增强后输入 640×256 |
| 当前数据 | 2026-08-27 新数据，具有有效自车运动信息 |

最重要的约束如下：

1. **视觉 60°不等于检测 60°。** 中央 60°同时使用图像和雷达；两侧
   `30° < |azimuth| <= 60°` 主要依靠雷达，GT、radar、query 和最终输出仍按
   120°处理。
2. 2026-08-18 数据的自车速度全部为 0，且部分 GT 框不够准确。它适合保留为历史
   对比集，不应再作为最终产品模型的首选训练源。
3. 2026-08-27 数据使用 `front60_camera` 和对应 YAML 标定，是当前正式训练源。
4. 目前稳定的主类别是 car 和 truck；bicycle 在所有实验中都明显难学。Main4 的
   pedestrian 实验也未在独立测试集上学成，暂时停止，等待更可靠的运动信息和 GT。
5. 训练集/验证集必须按时序 block 切分并保留 guard frames，不能逐帧随机拆分，否则
   四帧 sweep 会造成时序泄漏。
6. 模型选型以固定独立测试集为准，不以同路段小验证集的最高数字为准。必须同时报告
   car-only、Main3、0–25 m、25–50 m 和总 0–50 m。

## 2. 指标口径

日志里的 `BEV_AP@0.5` 和 `3D_AP@0.5` 是 IoU 0.5 下的 AP。`precision@0.1` 和
`recall@0.1` 是 score threshold 0.1 下的单点统计，不是 mAP。Main3 mAP 是 car、
truck、bicycle 三类 AP 的算术平均，因此 bicycle 接近 0 会显著拉低 Main3，即使
car/truck 已经达到约 0.4。

统一使用以下 profile：

- `car_only`：只评价 car；
- `main3`：car、truck、bicycle；
- `main4`：car、truck、bicycle、pedestrian，仅用于历史 Main4 实验。

距离桶必须使用修正后的 evaluator。早期曾出现 `0–25 + 25–50` 与 `0–50` 召回数不
自洽的问题，后续已经统一过滤规则。不要使用旧日志重新推导分桶结论。

## 3. 数据集总账

### 3.1 早期约 1,200 帧数据

处理目录：

```text
data/company_dataset_velocity_v2/processed/
```

关键 annotation：

```text
custom_infos_train_sweep.pkl
custom_infos_val_sweep.pkl
```

该数据用于最早的四帧、100m-q300 和 50m-q150/q200 实验。验证集只有 39 个 GT 的
结果数值很高，但样本太少，不能代表独立场景泛化。

### 3.2 2026-08-18 大数据集

原始数据：

```text
/mnt/diskNvme1/DataSet/radar_camera_GT
/mnt/diskNvme2/TruthData/20260818
```

50m train/val 转换结果：

```text
/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_trainval_v1
```

其中：

- train：16,166 帧；
- val：2,820 帧；
- `custom_infos_train_sweep.pkl`：约 131 MB；
- `custom_infos_val_sweep.pkl`：约 13 MB。

独立测试集：

```text
/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed_v3
```

关键文件：

```text
custom_infos_test_sweep.pkl
conversion_summary.json
```

测试集共 7,213 帧，来自三个独立序列：

- `2026-08-18-15-51-14`：2,887 帧；
- `2026-08-18-16-08-55`：2,995 帧；
- `2026-08-18-16-33-50`：1,331 帧。

Main3、FOV120 范围内的测试 GT 数为：car 15,925、truck 4,673、bicycle 3,679，
合计 24,277。该 7,213 帧集合是 2026-08-18 模型的统一独立测试基线，不要改变。

Main4 的 blocked split：

```text
/mnt/diskNvme1/hyh/data/company_20260818_30k_tuning/blocked_main4_fov120_v1
```

它从 18,986 帧中生成 16,012 train / 2,848 val，
`temporal_artifact_overlap=0`。注意该目录主要保存 annotation；图片、雷达和 LiDAR
仍在原 processed root，不能把 blocked split 目录误当完整 `data_root`。

已确认的数据问题：

- 自车速度全部为 0；
- 一部分 GT 框位置和姿态不够准确；
- bicycle 难学；
- 早期 val 只有一个场景且 bicycle 极少，导致 val 和独立 test 差异很大；
- 某些全局坐标 LiDAR 帧在 ROI 内为空，转换器后来增加了 global→ego、空帧保留、
  radar 时间窗过滤、按序列 cache 和断点续转。

### 3.3 2026-08-27 新数据（当前正式数据）

原始数据：

```text
/mnt/diskNvme1/DataSet/radar_camera_GT/2026-08-27-*
/mnt/diskNvme2/TruthData/20260827
```

共 14 个序列、33,335 帧。图像、雷达、GT、LiDAR 已检查对齐；car twist 完整且非零。
Truth LiDAR 是 global 坐标，转换时必须 global→ego。

当前完整转换结果：

```text
/mnt/diskNvme1/hyh/data/company_20260827_33k_front60_q200_f4/processed_raw_v1
```

时序安全的 blocked dev split：

```text
/mnt/diskNvme1/hyh/data/company_20260827_33k_front60_q200_f4/blocked_dev_v1
```

关键文件：

```text
processed_raw_v1/custom_infos_train_sweep.pkl
processed_raw_v1/custom_infos_test_sweep.pkl
processed_raw_v1/conversion_summary.json
blocked_dev_v1/custom_infos_train_sweep.pkl
blocked_dev_v1/custom_infos_val_sweep.pkl
blocked_dev_v1/split_report.json
```

split manifest：

```text
data_splits/company_20260827_33k_v1.json
```

11 个序列进入 train pool，3 个序列固定为独立 test。blocked dev split 再从 train
pool 内按每序列 3 个 block、15% val、3 guard frames 生成 train/val。接手人应从
`split_report.json` 读取最终帧数，不要凭 33,335 总数估算。

Front60 标定通过以下方式进入转换器：

```text
camera directory: front60_camera
camera topic: /front60_camera/compressed
```

代表性 YAML：

```text
/mnt/diskNvme2/TruthData/20260827/20260827_DAY_SUN_URBAN_ROAD_MAINARTERY/2026-08-27-17-39-28/car_chengtai___2026-07-30.yaml
```

相关实现提交：

```text
7bcf310 feat: support calibrated front60 company data
f7b3299 feat: decouple front60 camera from 120deg detection
```

## 4. 模型和配置共同参数

当前 50m 模型的共同参数：

| 参数 | 值 |
| --- | --- |
| `point_cloud_range` | `[0, -20, -3, 50, 20, 3]` |
| voxel | `[0.5, 0.5, 6]` |
| BEV | 100×80 |
| 输入时序 | 4 帧 |
| query | 200 |
| decoder | 6 层/6 次 recurrent iteration |
| 图像原始尺寸 | 1920×1080 |
| 网络图像尺寸 | 640×256 |
| optimizer LR | `4e-4`（除非 profile 显式覆盖） |
| eval interval | 2 epoch |

当前调参配置：

```text
configs/racformer_company_front_50m_q200_f4_30k_tune.py
```

当前启动器：

```text
run_code/run_company_50m_tuning.sh
```

四卡时 `batch_size=1/GPU`，有效 global batch 为 4，梯度累计为 1。两卡时启动器用
梯度累计 2，保持有效 global batch 为 4。历史上 NCCL P2P/IB 在这台服务器上不稳定，
启动器默认配合 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1` 并先跑 collectives preflight。

## 5. 历史模型与实验总账

### 5.1 原始四帧公司模型

```text
config: configs/racformer_company_front_velocity_v2_f4.py
checkpoint: outputs/racformer_company_front_velocity_v2_f4/2026-08-05/11-17-49/epoch_36.pth
```

这是早期 350m/大范围体系的四帧模型，也是最初部署拆图工作的来源。后续 50m 模型
不能复用它的 query、bbox coder 或 TensorRT engine。

### 5.2 100m、300-query、4帧

```text
config: configs/racformer_company_front_100m_q300_f4.py
run: outputs/racformer_company_front_100m_q300_f4/2026-08-12/13-54-13
checkpoint: outputs/racformer_company_front_100m_q300_f4/2026-08-12/13-54-13/latest.pth
range: [0, -20, -3, 100, 20, 3]
BEV: 200×80
query: 300
frames: 4
```

39 GT 小验证集结果：BEV mAP 0.9701、3D mAP 0.9187、recall 1.0。该结果只证明
训练链路可用，不能作为大数据泛化指标。

### 5.3 50m、150-query 与 200-query 小数据版本

```text
150q config: configs/racformer_company_front_50m_q150_f4.py
200q config: configs/racformer_company_front_50m_q200_f4.py
200q checkpoint: outputs/racformer_company_front_50m_q200_f4/2026-08-19/11-23-11/latest.pth
```

150 query 的精度明显低于预期，因此选用 200 query 作为速度/精度折中。200q 小数据
模型用于完成第一版 50m TensorRT 和 Nano 链路，但在 7,213 帧独立测试集上严重失效：

150q 实验没有在现有记录中留下一个经过独立测试确认的 canonical checkpoint 路径；
接手时不要凭目录时间戳猜测。如果服务器仍保留其 run，应先核对其中的
`resolved_config.py` 和 checkpoint metadata，再决定是否归档。

| 口径 | BEV mAP | 3D mAP | precision@0.1 | recall@0.1 |
| --- | ---: | ---: | ---: | ---: |
| car-only | 0.0168 | 0.0000 | 0.1003 | 0.0635 |
| Main3 | 0.0057 | 0.0000 | 0.0832 | 0.0429 |

结论：一千多帧和 39 GT 的结果不能外推到多场景数据；该 checkpoint 只保留作部署
链路回归，不作为产品精度基线。

### 5.4 Q1–Q5：350m query 分布消融

Q1–Q5 不是 1–5 个 query，而是五组 query 分布实验。共同条件为 350m、4帧、10类、
36 epoch。`num_query = num_clusters × num_ray`。

| 实验 | query | cluster×ray | distance power | 最佳保留 epoch | 3D mAP | 0–50m 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 900 | 30×30 | 1.0 | 34 | 0.2866 | 0.4307 |
| Q2 | 900 | 30×30 | 1.5 | 32 | **0.3090** | 0.4636 |
| Q3 | 900 | 30×30 | 2.0 | 32 | 0.2837 | 0.4596 |
| Q4 | 900 | 45×20 | 2.0 | 36 | 0.2987 | 0.4521 |
| Q5 | 1200 | 30×40 | 2.0 | 34 | 0.2534 | 0.3828 |

结果目录：

```text
outputs/3dh_query_company_20260818_q1/2026-08-21/10-12-05
outputs/3dh_query_company_20260818_q2/2026-08-21/12-38-41
outputs/3dh_query_company_20260818_q3/2026-08-21/15-06-03
outputs/3dh_query_company_20260818_q4/2026-08-21/17-33-48
outputs/3dh_query_company_20260818_q5/2026-08-21/20-03-00
```

统一 profile 复评目录：

```text
/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_q1_q5_profiles/20260824_162751
```

结论：增加到 1200 query 没有改善这个小数据实验；Q2 的 power 1.5 是较稳妥的折中，
因此后续 50m FOV query 采用 power 1.5。350m 大数据 Q5 训练后来因近距离模型尚未达到
预期且远距离 GT 极少而停止，不应继续消耗资源。

大数据 350m Q5 的隔离目录为：

```text
config: configs/3dh_query_company_20260818_30k_q5_f4.py
data: /mnt/diskNvme1/hyh/data/company_20260818_30k_q5_350m_f4/processed_trainval_v1
run: /mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_30k_q5_350m/20260825_185948
log: .../train_q5.log
```

它使用 1200 query、power 2.0、4帧和 350m 范围。到 epoch 12 时 car 3D AP 约
0.0104，整体 3D mAP 约 0.0117；50–100m 3D 约 0.0002，100–150m 为 0。该 run
已主动停止，不能当作收敛模型或部署候选。

### 5.5 2026-08-18 30k，矩形/FOV 训练探索

早期 50m Q200 30k 训练目录：

```text
/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_30k_50m_q200_f4/20260825_124708
```

该实验在 epoch 20 的 car 3D AP 达到 0.0716，随后波动。最初矩形训练和实际前向
传感器覆盖不一致，左右近车盲区出现大量 bicycle/pedestrian 误检。

加入 120°训练过滤、但仍使用旧 `front_grid`/power 1.0 的 36 epoch 实验：

```text
run: /mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_30k_50m_q200_f4_fov120/20260827_102049
best: .../model/best_company/3D_mAP@0.5_epoch_8.pth
last: .../model/epoch_36.pth
```

内部 val 的 3D mAP 在 epoch 8 达到 0.0552，epoch 36 降至 0.0230；训练 loss 同期仍
降至约 3.26。这证明“loss 继续下降”不等于泛化继续改善，36 epoch 已明显过拟合。

### 5.6 Main3，FOV120、power 1.5

8 epoch 独立实验：

```text
run: /mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning/full_main3_f4_fov120_p15_e8/20260828_211835
checkpoint: .../model/best_company/3D_mAP@0.5_epoch_8.pth
```

7,213 帧独立测试结果：

| 指标 | 数值 |
| --- | ---: |
| car 3D AP | 0.4266 |
| truck 3D AP | 0.4568 |
| bicycle 3D AP | 0.0063 |
| Main3 BEV mAP | 0.3289 |
| Main3 3D mAP | **0.2966** |
| overall precision / recall | 0.2125 / 0.4479 |
| 0–25m 3D mAP | 0.3876 |
| 25–50m 3D mAP | 0.1225 |

20 epoch 实验：

```text
run: /mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning/full_main3_f4_fov120_p15_e20/20260831_011717
checkpoint: .../model/best_company/3D_mAP@0.5_epoch_20.pth
predictions: .../test_7213_best/predictions.pkl
metrics: .../test_7213_best/metrics_fov120.json
log: .../test_7213_best/eval_fov120.log
```

epoch 20 的 7,213 帧独立测试结果：

| 指标 | 数值 |
| --- | ---: |
| car 3D AP | 0.3955 |
| truck 3D AP | 0.4086 |
| bicycle 3D AP | 0.0018 |
| Main3 BEV mAP | 0.3046 |
| Main3 3D mAP | 0.2686 |
| overall precision / recall | 0.4160 / 0.4165 |
| 0–25m 3D mAP | 0.3631 |
| 25–50m 3D mAP | 0.0996 |

同一 20 epoch run 内，epoch 8/12/16 的独立 Main3 3D mAP 分别约为 0.2716、
0.2782、0.2782。20 epoch 提高了 precision，但没有超过独立 8 epoch run 的 mAP。
结论是模型选择不能只依赖最后一个 epoch；当前旧数据 Main3 精度对比仍以独立 8 epoch
run 的 0.2966 为最好记录，20 epoch checkpoint 保留用于部署和可视化对比。

### 5.7 Main4，加入 pedestrian

```text
run: /mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_50m_q200_tuning/full_main4_f4_fov120_p15_e8/20260831_184101
checkpoint: .../model/best_company/3D_mAP@0.5_epoch_8.pth
```

blocked val 在 epoch 8 的 3D mAP 为 0.2655，但 7,213 帧独立测试结果只有：

| 类别/指标 | epoch 6 | epoch 8 |
| --- | ---: | ---: |
| car 3D AP | 0.3916 | 0.4195 |
| truck 3D AP | 0.4499 | 0.4426 |
| bicycle 3D AP | 0.0032 | 0.0031 |
| pedestrian 3D AP | 0.0000 | 0.0000 |
| Main4 3D mAP | 0.2112 | 0.2163 |
| precision@0.1 | 0.1142 | 0.1261 |
| recall@0.1 | 0.3454 | 0.3427 |

epoch 8 pedestrian：BEV AP 约 0.000066、precision 0.0020、recall 0.0107；模型在
FOV 内输出约 39,223 个 pedestrian prediction，却只召回极少目标。结论：不是简单
增加 epoch 就能解决，暂时停止 Main4；等待有正确自车运动、标注更准的新数据后再做
类别平衡、标签审计和 loss/assigner 调整。

## 6. 当前正在训练：Aug-27 Camera60 / Detection120 Main3

profile：

```text
full_main3_f4_cam60_det120_p15_e20
```

完整身份：

| 参数 | 值 |
| --- | --- |
| 数据 | 2026-08-27 33,335 帧集合 |
| 相机 | front60，光学覆盖 60° |
| 雷达/GT/query/output | 120° |
| 范围 | 50m rectangle ∩ 120° sector |
| 类别 | car, truck, bicycle |
| query | 200，`front_fov_grid`，power 1.5 |
| 时序 | 4帧 |
| epoch | 20 |
| GPU | 物理卡 0,3,4,5 |
| master port | 30327 |
| checkpoint | 每4 epoch；最多保留4个周期 checkpoint；另存 best 和 last |

结果根目录：

```text
/mnt/diskNvme1/hyh/results/RaCFormer/company_20260827_50m_q200_cam60_det120_main3
```

当前 run 不要硬编码时间戳，使用：

```bash
RESULT_ROOT=/mnt/diskNvme1/hyh/results/RaCFormer/company_20260827_50m_q200_cam60_det120_main3
PROFILE=full_main3_f4_cam60_det120_p15_e20
RUN_DIR=$(cat "$RESULT_ROOT/$PROFILE/latest_run.txt")
echo "$RUN_DIR"
```

run 内关键文件：

```text
environment.sh
resolved_config.py
data_smoke.log
nccl_test.log
queue.log
nohup.log
train.log
model/*.pth
model/best_company/*.pth
ALL_DONE 或 FAILED
```

查看训练：

```bash
tail -f "$RUN_DIR/train.log"
grep -E 'Epoch \[[0-9]+/20\]' "$RUN_DIR/train.log" | tail -10
find "$RUN_DIR/model" -type f -name '*.pth' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

训练结束后必须完成：

1. 用 best checkpoint 对固定独立 test 全量推理；
2. 按 Detection120 评价 Main3；
3. 对同一 predictions 再按 Camera60 评价中央融合区域；
4. 单独统计 `30° < |azimuth| <= 60°` 两侧 radar-only wings；
5. 保存 predictions、JSON metrics、完整 log 和可视化；
6. 与 2026-08-18 Main3 8 epoch/20 epoch 模型在相同口径下比较；
7. 确认自车速度进入 temporal/radar compensation，而不是只存在 metadata 中。

在该实验完成并通过独立 test 前，不要称它为最终产品 checkpoint。

## 7. Query 分布结论

早期 `front_grid` 在完整矩形内布点，和真实前向扇区不一致，左右两侧传感器盲区容易
产生误检。当前 `front_fov_grid` 先限制到 120°，再按方位和距离组织 query；power
大于 1 时增加远距离采样密度。

已有消融结论：

- power 1.0 对远处照顾不足；
- power 2.0 在当前数据上没有稳定优于 1.5；
- 1200 query 不一定优于 900 query；
- 50m 下 150 query 精度不足，200 query 是当前部署折中；
- 当前选择：200 query、FOV120、power 1.5。

## 8. 可视化和误检结论

可视化代码：

```text
deploy/cpp_visualizer/
```

训练结果可视化必须同时画：图像投影框、BEV、GT、prediction、score、类别，并分别
抽查 0–25m、25–50m、中央 60°、两侧 radar-only wings。历史可视化观察到：

- 120°相机/雷达假设错误时，矩形左右近车三角区有大量误检；
- 误检主要是 bicycle 和 pedestrian；
- 只在评价阶段裁掉盲区能改善报表，但不能替代训练时一致的 GT/radar/query 过滤；
- 新 Camera60/Detection120 模型仍应在两侧保留 radar-only 检测，不能把输出裁成 60°。

## 9. 标定、分辨率和部署边界

训练和部署必须共享同一个物理坐标定义，但不能简单认为“都朝前”就可以互换外参。
`lidar2img` 同时决定图像投影、深度监督和 LSS 几何。最终产品雷达和相机固定在同一根
杆上是有利条件，但仍必须测量内参、相对旋转、平移和安装高度。

新训练数据图像是 1920×1080，最终摄像头可能直接输出 640×480。网络输入虽然仍可
预处理到 640×256，但 4:3 与 16:9 的裁剪/缩放和有效视场不同。部署时必须使用实际
640×480 相机内参和与训练一致的 resize/crop 几何；如果有效视场、安装姿态或高度与
训练采集杆差异明显，应至少做标定后验证，必要时用最终硬件数据 fine-tune。不能只在
ONNX 导出时替换一个外参文件便默认精度不变。

## 10. Checkpoint 和结果保留规则

每个正式 run 至少保留：

- `environment.sh` 和 `resolved_config.py`；
- `train.log`、`queue.log`、`nccl_test.log`；
- 最佳 checkpoint；
- 最后 checkpoint；
- 用于比较的固定里程碑 checkpoint（如 8/12/16/20）；
- 独立测试 predictions；
- JSON metrics 和 evaluator log；
- 代表性可视化；
- `conversion_summary.json` 和 `split_report.json`。

普通周期 checkpoint 不必全部保留。当前启动器已经通过
`checkpoint_interval=4, max_keep_ckpts=4, save_last=True` 控制数量。删除任何旧 pth
之前，先确认 best/last、配置、日志和独立测试结果均存在；不要删除仅剩的部署来源
checkpoint。

## 11. 关键代码和提交

```text
tools/convert_chengtech_20260818.py
tools/convert_chengtech_20260818_collection.py
tools/create_company_blocked_dev_split.py
tools/smoke_company_training_data.py
tools/evaluate_company_predictions_fov.py
tools/analyze_training_convergence.py
run_code/run_company_50m_tuning.sh
configs/racformer_company_front_50m_q200_f4_30k_tune.py
```

关键提交：

```text
eb76593 feat: add 50m q200 config and fix range counts
12f2d28 feat: add sequence-isolated 30k q200 evaluation
5a6fadd feat: add staged 50m Q200 tuning workflow
a537975 feat: align company training and evaluation to 120deg FOV
ebf82ed Add 8-epoch FOV query-power experiment
2cfd206 Avoid YAPF dependency in tuning launcher
de29942 Add blocked dev split and 20 epoch tuning run
f1dfc88 Add final four-class FOV training profile
7bcf310 feat: support calibrated front60 company data
f7b3299 feat: decouple front60 camera from 120deg detection
```

## 12. 接手检查清单

- [ ] 当前分支包含 `f7b3299` 或其后继提交。
- [ ] Aug-27 raw、blocked dev、independent test 文件都存在。
- [ ] `split_report.json` 的时序重叠为 0。
- [ ] resolved config 显示 camera FOV 60、detection FOV 120。
- [ ] resolved config 显示 Main3、200 query、4 frames、power 1.5。
- [ ] NCCL preflight 通过，训练进程确实占用物理 GPU 0,3,4,5。
- [ ] best checkpoint 按 `company/3D_mAP@0.5` 保存。
- [ ] 独立 test 使用固定 annotation，未混入 train/val。
- [ ] 同时汇报 car-only、Main3、距离桶、中央60°和两侧 wings。
- [ ] 最终 checkpoint 和部署配置的类别、范围、FOV、query、帧数完全一致。
