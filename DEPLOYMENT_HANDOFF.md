# RaCFormer TensorRT Deployment Handoff

Last updated: 2026-08-13

The clean offline transfer, environment inventory, aarch64 plugin build,
three-engine construction, validation, benchmarking, and result-repatriation
procedure for the four-frame 100m-q300 artifact family is recorded in
`deploy/NANO_100M_Q300_F4_FROM_SCRATCH.md`.

## 1. Objective

Deploy RaCFormer on a Jetson Orin Nano 16GB using TensorRT without a
PyTorch runtime dependency. The immediate objective is a correct end-to-end
deployment. The later performance target is approximately 15 Hz, which will
require further architecture and precision optimization.

Do not modify the training behavior for deployment-only changes. Deployment
fallbacks, plugins, graph rewrites, precision controls, and runtime scheduling
must remain opt-in.

## 2. Environments

### L20 server

- GPU: NVIDIA L20
- Native stack: PyTorch 2.0.1+cu118, TensorRT 8.6.1, x86_64
- TensorRT 8.5.2.2 is also available in a container for target-compatible
  testing.
- Container code path: `/workspace/RaCFormer`
- Container output path: `/workspace/outputs`
- The code and output directories are mounted separately. Therefore, container
  artifacts are under `/workspace/outputs`, not
  `/workspace/RaCFormer/outputs`.

### Jetson target

- Device: Jetson Orin Nano 16GB
- Architecture: aarch64
- L4T R35.6.1 / Ubuntu 20.04
- CUDA 11.4
- TensorRT 8.5.2.2
- Target repo path: `/home/cttest/RaCFormer`
- TensorRT engines and plugin shared libraries must be built on the target, or
  in an exactly compatible aarch64 environment. An x86_64 engine or plugin
  cannot be copied to the Jetson.

Confirmed target artifact locations:

- ONNX and fixtures: `/home/cttest/RaCFormer/outputs/deploy_onnx`
- TensorRT engines and reports:
  `/home/cttest/RaCFormer/outputs/deploy_tensorrt`
- Orin plugin:
  `/home/cttest/RaCFormer/build/tensorrt_plugins_orin/libracformer_bev_pool_v2_trt.so`

Do not use the server container's `/workspace/...` paths in Nano commands.

The current precompute V2 artifacts are:

- frontend ONNX:
  `outputs/deploy_onnx/racformer_frontend_precompute_v2_trt85.onnx`
- recurrent decoder ONNX:
  `outputs/deploy_onnx/racformer_decoder_precompute_v2_trt85.onnx`
- shared frontend/decoder fixture:
  `outputs/deploy_onnx/racformer_frontend_precompute_sample0.npz`
- Orin frontend engine:
  `outputs/deploy_tensorrt/racformer_frontend_precompute_v2_trt852_orin.engine`
- Orin decoder engine:
  `outputs/deploy_tensorrt/racformer_decoder_precompute_v2_orin.engine`

The decoder uses the shared frontend precompute fixture. There is no separate
`decoder_precompute_v2.npz` fixture.

The previously validated, non-precompute split artifacts are:

- `outputs/deploy_tensorrt/racformer_frontend_trt852_orin.engine`
- `outputs/deploy_tensorrt/racformer_decoder_recurrent_trt852_orin.engine`
- `outputs/deploy_onnx/racformer_frontend_sample0.npz`

Their reports are `profile_frontend_trt852_orin.txt` and
`profile_decoder_trt852_orin.txt` under `outputs/deploy_tensorrt`.

## 3. Proven Deployment Architecture

The reliable deployment is no longer one monolithic engine. It consists of:

1. A frontend engine that computes image, LSS BEV, radar BEV, and precomputed
   decoder BEV values.
2. One recurrent decoder-layer engine enqueued six times on the same CUDA
   stream, using the six `d_region` values.
3. Runtime orchestration that keeps `query_feat` and `query_bbox` on the GPU
   between decoder iterations.
4. Learned initial query tensors supplied from the fixture in FP32 when the
   frontend engine is FP16.

This split is intentional. TensorRT 8.5.2 builds the six-layer stacked decoder
but produces incorrect detections (`300/8`). The recurrent one-layer decoder
produces the correct `8/8` detections.

Relevant implementation commits on `main`:

- `6d88875 feat: run decoder through recurrent TensorRT layer`
- `1d66b1f feat: chain frontend with recurrent decoder engine`
- `87c931c fix: bind decoder-only pipeline inputs`
- `5c1c18e fix: rewrite frontend graph for TensorRT 8.5`
- `158c93d feat: precompute recurrent decoder BEV values`
- `e759d61 feat: preserve recurrent initial queries in FP32`

## 4. Confirmed Results

### Correctness

The frontend plus recurrent decoder path passed on both the L20 and Jetson:

- Detection count: `8/8`
- Boxes close: `True`
- Scores close: `True`
- Labels equal: `True`
- Decoded comparison: `True`
- Deployment acceptance: `True`

On the Jetson, the FP32 precompute pipeline produced:

- End-to-end engine latency: about `4208.884 ms`
- Resident CUDA memory delta: about `3964.86 MB`
- Frontend latency: about `1593.348 ms`
- Six-layer recurrent decoder latency: about `2624.670 ms`
- Per decoder iteration: about `437.445 ms`

On the L20 TensorRT 8.5.2 container, the FP32 precompute pipeline produced:

- End-to-end latency: about `119.893 ms`
- Detection comparison: pass

### FP16 frontend

The useful hybrid configuration is:

- Frontend engine: FP16
- Initial learned `query_bbox` and `query_feat`: FP32 from fixture
- Recurrent decoder engine: FP32

L20 TensorRT 8.5.2 A/B result:

- FP32 frontend + FP32 query + FP32 decoder: `120.666 ms`, `3254 MB`, pass
- FP16 frontend + FP32 query + FP32 decoder: `104.148 ms`, `2354 MB`, pass

The runtime validator must include:

```bash
--initial-query-from-fixture
```

Without that option, the FP16 frontend changes the learned initial query enough
to cause `7/8` detections.

## 5. Why the Original Monolithic Path Failed

The deployment work encountered several independent TensorRT issues:

- PyTorch 2.0 ONNX export did not support 5D volumetric `GridSample`.
- TensorRT required custom plugins for BEV pooling, MSMV sampling, identity
  barriers, and single-camera projection.
- TensorRT 8.5 does not parse standard `IsInf` and `LayerNormalization` nodes
  used by the original graph. Deployment rewrites replace them with compatible
  primitive operations.
- Zero-dimension execution tensors from empty slices had to be removed or
  rewritten. Zero-length shape tensors are valid and are not execution tensor
  failures.
- Eight dynamic radar voxel counts triggered a TensorRT 8.5 Myelin assertion.
  Static radar padding/scatter removed this build blocker.
- TensorRT 8.5 incorrectly optimized the stacked six-layer decoder. It could
  build an engine, but inference produced `300/8` detections. Splitting out one
  recurrent decoder layer fixed this numerical failure.

Parser success alone is not sufficient. Every engine must also pass decoded
detection validation.

## 6. Current Optimization Work

The next target is reducing recurrent decoder latency with mixed FP16/FP32.

### Full FP16 decoder result

The full FP16 recurrent decoder built successfully:

- Engine size: `54.24 MB`
- Six-layer latency on L20: `47.897 ms`
- Per-layer latency: `7.983 ms`
- Memory delta: `1028 MB`

It was numerically invalid:

- Detection count: `8/8`
- Boxes max error: about `35 m`
- Labels equal: `False`
- Decoded comparison: `False`

Therefore, decoder FP16 is fast but cannot be accepted without selective FP32.

### Failed mixed-precision builds

Mixed attempt 1:

```text
FP32 patterns: position_encoder, self_attn, norm, reg_branch
FP32 constrained layers: 276
Failure: ForeignNode self_attn/Slice ... self_attn/Div_2
```

Mixed attempt 2:

```text
FP32 patterns: position_encoder, norm, reg_branch
FP32 constrained layers: 128
Failure: ForeignNode self_attn in_proj_weight ... attention/Add
```

These failures occur during TensorRT tactic selection. They are not parser or
ONNX-checker failures. The broad `norm` substring creates mixed-precision
boundaries that TensorRT 8.5 propagates into adjacent self-attention fusion
regions. Do not retry these two pattern sets unchanged.

The existing `racformer_identity` plugin is a fusion barrier but supports both
FP16 and FP32. It does not by itself force an FP32 tensor boundary.

## 7. Mixed Decoder Experiment Conclusion

The `reg_branch`-only FP32 experiment has completed. It built successfully but
failed decoded validation:

- Detection count: `8/8`
- Boxes close: `False`, maximum error about `35.0105 m`
- Scores close: `True`, maximum error about `0.00692`
- Labels equal: `False`
- Decoded comparison: `False`
- Six-layer latency: about `181.358 ms` (high run-to-run variance)

This proves that keeping only the final regression branch in FP32 does not fix
the decoder. The material FP16 divergence occurs earlier in the recurrent
query-state path. Do not use or deploy
`racformer_decoder_mixed_regonly_trt852.engine`.

The `norm1`-only FP32 build also failed at the self-attention fusion boundary.
This confirms that TensorRT 8.5 cannot reliably implement the internal
FP16-to-FP32 transition using the current layer-name precision constraints.
Stop trying additional `--fp32-layer-pattern` combinations; later norms are
unlikely to change this limitation and would consume build time without
answering a new question.

This does not mean that the whole deployment must use one precision. The
already validated production-safe configuration is:

- frontend engine: FP16;
- learned initial query state: FP32 fixture tensors;
- recurrent decoder engine: FP32.

This is mixed precision at the engine boundary and has passed decoded
validation. Only the decoder remains entirely FP32 for now.

## 8. Remaining Decoder Optimization Routes

If further decoder optimization becomes necessary, use one of these routes:

1. Add deployment-only explicit FP32 boundaries, such as a dedicated identity
   plugin that accepts only FP32, around the sensitive recurrent state path.
   This requires plugin and ONNX changes and must be revalidated on TRT 8.5.
2. Split the decoder layer into FP16 and FP32 sub-engines and orchestrate them
   at runtime. This is more predictable but adds bindings, buffers, and enqueue
   overhead.
3. Retrain or fine-tune with AMP/FP16-aware numerics so that the complete
   decoder can tolerate FP16. The current FP32 checkpoint demonstrably cannot.

For the current link-validation milestone, keep the decoder FP32 and spend the
next optimization effort on temporal caching, reducing repeated frontend work,
or a separately trained four-frame model. These changes have a clearer payoff
than further TensorRT 8.5 layer-pattern experiments.

## 9. Important Validation Rules

- `status: PASS` from `parse_onnx.py` only proves graph parsing.
- `status: SUCCESS` from `build_engine.py` only proves engine construction.
- Deployment acceptance requires decoded detections to match:
  - detection count equal
  - boxes close
  - scores close
  - labels equal
- Raw tensors can contain a very small number of outliers while decoded
  detections still match. In that case, `--accept-decoded-match` is the intended
  deployment criterion.
- Large output differences, `300/8`, `7/8`, label changes, or tens-of-meters
  box errors are real failures and must not be accepted by increasing `atol`.

## 10. Runtime Notes

- Production inference does not require PyTorch. It requires TensorRT, CUDA,
  the built engines, the target plugin shared library, input preprocessing,
  and postprocessing/runtime orchestration.
- Current benchmark numbers measure GPU engine execution and orchestration, not
  the complete ROS sensor pipeline. Sensor ingestion, synchronization,
  camera/radar preprocessing, host-to-device transfers, and final publication
  must be measured separately.
- The current frontend processes all eight temporal frames. A production
  sliding-window cache that computes only the new frame is still future work.
- ROS 1 versus ROS 2 does not change engine correctness, but it affects the
  surrounding node integration and transport overhead.

## 11. Key Source Files

- `deploy/export_onnx.py`: main deployment ONNX export and compatibility flags
- `deploy/export_decoder_recurrent_layer.py`: one reusable decoder-layer export
- `deploy/export_frontend.py`: frontend export
- `deploy/bev_precompute.py`: decoder BEV value precomputation
- `deploy/tensorrt/rewrite_trt85_onnx.py`: TensorRT 8.5 graph compatibility
- `deploy/tensorrt/build_engine.py`: FP32, FP16, and mixed engine construction
- `deploy/tensorrt/parse_onnx.py`: parser and zero-dimension audit
- `deploy/tensorrt/validate_decoder_loop_numpy.py`: recurrent decoder validation
- `deploy/tensorrt/validate_frontend_decoder_numpy.py`: complete split-pipeline
  validation
- `deploy/tensorrt/validate_engine_numpy.py`: lightweight engine validation
- `deploy/tensorrt/plugins/bev_pool_v2/`: all deployment TensorRT plugins
- `models/racformer_transformer.py`: decoder barriers and deployment branches

## 12. Repository State

- Branch: `main`
- Latest relevant committed training change: `6d28b4f`
- `scripts/` is currently untracked and unrelated to this handoff. Do not add,
  delete, or revert it unless its owner explicitly requests that action.
- Large ONNX, fixture, engine, and report files are deployment artifacts and are
  not expected to be committed to Git.

## 13. Optimization Measurement Update

`deploy/tensorrt/validate_frontend_decoder_numpy.py` now accepts
`--profile-stages`. It reports the frontend, the recurrent decoder total, and
each of the six decoder iterations using CUDA events. The same report splits
resident CUDA memory into engine deserialization, execution contexts, explicit
runtime buffers, and allocations first observed during warmup.

Use the already accepted FP16-frontend/FP32-decoder artifacts as the baseline:

```bash
python -m deploy.tensorrt.validate_frontend_decoder_numpy \
  --frontend-engine <fp16-frontend.engine> \
  --decoder-engine <fp32-recurrent-decoder.engine> \
  --fixture <frontend-decoder-fixture.npz> \
  --plugin <target-plugin.so> \
  --initial-query-from-fixture \
  --accept-decoded-match \
  --profile-stages \
  --warmup 5 \
  --iters 20 \
  --out <nano-profile-report.txt>
```

Stage profiling adds CUDA event overhead, so use its component timings to find
bottlenecks and use a second run without `--profile-stages` for the production
end-to-end latency number. The next optimization should be selected from the
measured memory split and stage timings rather than from the aggregate resident
memory delta alone.

## 14. Nano FP16 Frontend Result And Decoder Early Exit

The Nano-built FP16 precompute V2 frontend was chained to the existing FP32
precompute V2 decoder at:

`outputs/deploy_tensorrt/racformer_decoder_precompute_v2_orin.engine`.

Nano result:

- detections: `8/8`;
- labels equal: `True`;
- score maximum error: about `0.000713`;
- decoded box maximum error: about `0.02463 m`;
- end-to-end latency: about `3109.610 ms`;
- resident CUDA memory delta: about `3261.60 MB`.

The 6 mm validation tolerance rejects the decoded boxes, but a 3 cm decoded
deployment tolerance is appropriate for this FP16 frontend result. Raw tensor
outliers must not be used for acceptance; the accepted criterion remains the
decoded detections.

The validator now also accepts `--decoder-iterations N`. This runs the first N
entries of the six-element `d_region` schedule while comparing decoded output
against the established final six-layer fixture result. Raw parity for an
intermediate layer cannot by itself accept an early-exit deployment.

### Decoder early-exit result

The five-iteration experiment failed decisively in the L20 TensorRT 8.5.2
container:

- decoded detections: `4/8`;
- boxes close: `False`;
- scores close: `False`;
- labels equal: `False`;
- decoded comparison: `False`;
- end-to-end engine latency: about `90.683 ms`.

The sixth decoder iteration is required by the current checkpoint. Do not test
four or three iterations: once five iterations lose half of the detections,
shorter schedules cannot answer a useful deployment question. Keep all six
FP32 recurrent decoder iterations unless the model is retrained explicitly for
early exits.

## 15. Four-Frame Deployment

The trained four-frame checkpoint is:

`outputs/racformer_company_front_velocity_v2_f4/2026-08-05/11-17-49/epoch_36.pth`.

Deployment proceeds without another training/evaluation gate at the owner's
direction. Use
`configs/deploy/racformer_company_front_left_pytorch_f4.py`. Production ONNX
exporters now derive the temporal input list and shapes from the selected
checkpoint/config, while the legacy eight-frame diagnostic input symbol is
retained for compatibility.

The complete native-PyTorch export, TensorRT 8.5 container build, and split
pipeline validation commands are in `deploy/FOUR_FRAME_TENSORRT.md`. Before
export, recover the successful eight-frame `static radar voxel slots` value
from the existing export reports; that historical capacity is not recorded in
the repository and must not be guessed.
