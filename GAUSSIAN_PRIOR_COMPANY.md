# Company Radar-Guided Gaussian Query Integration

The company-data implementation is staged so the Q1-Q5 grid-query baseline
remains reproducible while the learned radar prior is introduced.

## Stage A: geometric feasibility

Run current-frame versus four-frame radar candidate recall before training a
Gaussian. This measures nearest radar-return distance to every GT centre and
breaks results down by class, source, and range:

```bash
bash run_code/start_company_radar_candidate_recall.sh
```

The output is written under
`/mnt/diskNvme1/hyh/results/RaCFormer/company_20260818_gaussian_analysis` and
contains a JSON summary plus per-GT CSV. It reads data at reduced CPU/I/O
priority and uses no GPU.

The critical comparisons are current versus temporal recall within 1, 2, 4,
and 8 metres, together with nearest-distance p50/p90/p95. If temporal sweeps
improve recall but create a large tail, the Gaussian head needs explicit
time-lag conditioning rather than simply concatenating all radar points.

## Stage B: learned candidate scoring

`models/gaussian_prior/candidate_scorer.py` already implements the first
learned component. It consumes `[x,y,z,rcs,vx,vy,time_lag]`, rejects only
invalid candidates, learns soft objectness from GT distance, and returns a
fixed Top-K. It must be connected to the current-frame raw radar list passed
to `RaCFormer_head`; hand-crafted RCS or velocity ranking is not used.

## Stage C: Gaussian prior over existing queries

Keep the Q1 query count and layout fixed for the first controlled experiment.
For each selected radar candidate, predict:

- centre residual `delta_x, delta_y`;
- positive longitudinal/lateral scales through bounded `softplus`;
- correlation through bounded `tanh`;
- objectness amplitude and a radar embedding.

Evaluate those Gaussians at the existing grid-query centres. The normalized
Gaussian weights aggregate radar embeddings into one context per original
query, followed by a zero-initialized gated residual into `query_feat`.
This preserves the decoder tensor shapes, matching, denoising layout, and Q1
deployment contract. It is safer than replacing grid queries or appending a
dynamic number of radar queries in the first experiment.

Training adds three explicit auxiliary losses:

1. candidate soft-objectness BCE;
2. positive-candidate centre-offset Smooth-L1;
3. Gaussian negative log-likelihood for matched GT centre residuals.

The gate starts at zero so the new config is numerically equivalent to Q1 at
initialization. Baseline Q1 checkpoints can then be loaded with only the new
Gaussian modules missing.

## Stage D: ablation order

Use a new config namespace and do not modify completed Q1-Q5 configs:

1. Q1 baseline;
2. Q1 + current-frame candidate scorer;
3. Q1 + current-frame Gaussian query context;
4. Q1 + four-frame time-conditioned Gaussian context;
5. only then test Gaussian context on Q2-Q5 layouts.

Report candidate recall, detector AP/recall by range, added parameters,
latency, and the learned gate magnitude. The present company validation split
contains almost no GT beyond 150 metres, so claims about long-range benefit
require additional far-range GT even if the engineering ablation succeeds.
