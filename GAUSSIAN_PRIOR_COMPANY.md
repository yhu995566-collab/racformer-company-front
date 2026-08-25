# Company Radar-Guided Gaussian Query Integration

The company-data implementation is staged so the Q1-Q5 grid-query baseline
remains reproducible while the learned radar prior is introduced.

## Stage A: geometric feasibility

For the sequence-disjoint 30k delivery, run the complete 50m Train/Val and
independent Test first:

```bash
bash run_code/start_company_30k_radar_stage1.sh front50
```

After the interrupted 350m conversion has completed, the same launcher can
produce two non-conflicting reports: all classes at 0-200m and car-only at
0-350m.

```bash
bash run_code/start_company_30k_radar_stage1.sh front350
```

Both modes audit every referenced radar path before starting.  The 350m mode
refuses partial conversion output.

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

The 30k 50m experiment trains on Train, selects on Val, and additionally
reports the frozen independent Test. It defaults to physical GPU 1 and uses
batched point-MLP updates:

```bash
bash run_code/start_company_30k_radar_topk.sh
```

The company-data learned Top-K experiment is deliberately capped at the
validated 200m range and leaves all completed 350m Q1-Q5 configs untouched:

```bash
bash run_code/start_company_radar_topk.sh
```

`models/gaussian_prior/candidate_scorer.py` consumes four-frame
`[x,y,z,rcs,vx,vy,time_lag]` points, rejects only invalid candidates, and
learns class-agnostic objectness, a centre residual, and an embedding. Targets
use distance to oriented BEV boxes rather than distance to the geometric
centre, retaining useful surface returns on large vehicles.

Training uses only the train split. Validation compares random, RCS, and
learned Top-64/128/256 selection against the all-candidate ceiling, before and
after learned centre correction. Hard Top-K is a fixed-shape compute budget,
not the final fusion mechanism; Stage C applies soft Gaussian weights to the
unchanged grid queries.

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
latency, and the learned gate magnitude. Company-data Gaussian experiments
currently report only 0-50m, 50-100m, 100-150m, and 150-200m. No claim is made
beyond the user-confirmed accurate 200m range.
