# 3DH-Query stage 1: radar candidate recall

This experiment measures whether legal nuScenes radar returns geometrically
cover GT object centres. It does not create Gaussians and does not import or
modify the RaCFormer model/training pipeline.

The script uses the official nuScenes devkit radar layout:
`x/y/z=0/1/2`, `rcs=5`, raw velocity `vx/vy=6/7`, and compensated velocity
`vx_comp/vy_comp=8/9`. `RadarPointCloud.from_file()` supplies the devkit-valid
raw returns. The filtered set additionally removes non-finite/out-of-range
points, implausibly large compensated speeds, and the lowest per-sample RCS
percentile. Near-zero velocity is retained.

All sweeps from all five nuScenes radar sensors are transformed through
`radar sensor -> sweep ego -> global -> current ego`. GT annotation centres
are transformed through `global -> current ego`. Metrics therefore use the
same BEV frame (x forward, y left).

## Smoke test (10 samples)

```bash
python tools/analysis/stage1_radar_candidate_recall.py \
  --data-root data/nuscenes \
  --version v1.0-trainval \
  --split val \
  --out-dir outputs/stage1_candidate_recall/smoke_10 \
  --max-samples 10 \
  --use-sweeps 1 \
  --bev-range -54 54 -54 54 \
  --range-bins 0 30 50 70 100 \
  --recall-thresholds 2 4 8 \
  --vis-num 10
```

For a 100-sample pilot, change `--max-samples 10` to `100` and the output
directory to `outputs/stage1_candidate_recall/pilot_100`.

## Full validation split

```bash
python tools/analysis/stage1_radar_candidate_recall.py \
  --data-root data/nuscenes \
  --version v1.0-trainval \
  --split val \
  --out-dir outputs/stage1_candidate_recall \
  --use-sweeps 1 \
  --bev-range -54 54 -54 54 \
  --range-bins 0 30 50 70 100 \
  --recall-thresholds 2 4 8 \
  --vis-num 20
```

`--use-sweeps N` means up to N sweeps per radar sensor, including the current
sweep. Use a larger ego range such as `--bev-range -100 100 -100 100` if the
goal is an untruncated radial 70-100 m measurement; the RaCFormer-like
`[-54, 54]` square only includes the portion of that annulus inside the square.

The output directory contains:

- `candidate_recall_summary.json`
- `candidate_recall_summary.csv`
- `nearest_distance_by_range.csv`
- `per_class_candidate_recall.csv`
- `vis/` when `--vis-num` is positive
