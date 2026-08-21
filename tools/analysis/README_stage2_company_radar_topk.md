# Stage 2: Company Radar Learned Top-K at 200m

This detector-independent experiment trains a shared point MLP on the company
train split and evaluates only on the val split.  Its hard spatial scope is
`x=[0,200]m`, `y=[-20,20]m`, `z=[-3,3]m`; the completed 350m Q1-Q5 configs are
not modified.

The scorer consumes four-frame `[x,y,z,rcs,vx,vy,time_lag]` points.  It learns
class-agnostic objectness and a bounded centre residual.  Targets are built
from distance to oriented BEV GT rectangles, so valid surface returns on large
vehicles are not incorrectly treated as background.

Run in the background on physical GPU 0 by default:

```bash
bash run_code/start_company_radar_topk.sh
```

Override the GPU only when needed:

```bash
COMPANY_RADAR_TOPK_GPU=1 bash run_code/start_company_radar_topk.sh
```

Outputs are placed under
`/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_radar_topk`.  The JSON
compares random, RCS, and learned Top-64/128/256 against the all-candidate
ceiling.  Learned candidates are reported both at the raw return and after the
predicted centre correction, overall and by class/source/50m range bin.

Hard Top-K is only a fixed-shape compute budget for the future Nano/TensorRT
path.  It is not the final fusion: selected embeddings will later contribute
to the original grid queries through soft Gaussian weights and a zero-start
gated residual.
