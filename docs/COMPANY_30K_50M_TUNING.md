# Company 30k / 50 m Q200 tuning protocol

This protocol separates data correctness from hyper-parameter search. Do not
compare experiments made with different split files, class profiles, seeds, or
evaluation thresholds.

## Frozen datasets

- Full train/val: `processed_trainval_v1` (16,166 train / 2,820 val).
- Independent test: `processed_v3` (7,213 test).
- Delivery metrics: `car_only` and `main3=(car, truck, bicycle)`.

The test set must never be used for checkpoint selection or tuning.

## Stage 0: establish the real baseline

Run `run_code/eval_company_30k_50m_q200_baseline.sh CHECKPOINT GPU`. It performs
one inference pass, evaluates both delivery profiles, writes a matched-box
geometry audit, and renders 100 evenly spaced samples. A large BEV/3D gap must
be explained from `dz`, `dh`, and yaw errors before learning-rate tuning.

## Stage 1: tiny-set overfit

Create a deterministic 256-frame car subset with `--overfit`, then run these in
order:

1. `overfit_car_f1_noflip`
2. `overfit_car_f4_noflip`

The validation file intentionally contains exactly the training frames. Failure
to reach a high training-set AP indicates a label, geometry, temporal alignment,
model capacity, or optimization defect; it is not a generalization problem.

## Stage 2: proxy experiments

Create one fixed 4,000-frame train / 800-frame val proxy set. Run:

1. `proxy_all_f4_flip` (10-class control)
2. `proxy_main3_f4_noflip` (class filtering)
3. `proxy_main3_f1_noflip` (temporal control)
4. `proxy_main3_f4_flip` (augmentation geometry control)
5. `proxy_main3_f4_z` (stronger `cz`, height, and bbox loss)
6. LR half/double only after the preceding structural checks pass.

Every profile uses the same config. The launcher records all controlling
environment variables and the fully resolved MMCV config in its run directory.
Use two GPUs, one sample per GPU, and accumulation=2 so effective global batch
remains four.

## Stage 3: full-data finalists

Promote at most two or three proxy winners. Train them on the full train set,
select checkpoints only on the frozen val set, then run the independent test
exactly once per finalist. Report car 3D/BEV AP, main3 3D/BEV mAP, precision,
recall, per-range metrics, latency, checkpoint commit, and resolved config.

Query count is not a first tuning variable: average target occupancy is far
below 200 in the current 50 m data. Keep Q200 fixed until data geometry, class
scope, temporal alignment, and loss balance are resolved.
