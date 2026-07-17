# TensorRT Bring-Up

This directory starts with compatibility auditing, not engine benchmarking.
The first boundary is fixed batch 1, one left camera, eight frames, FP32, and
raw detector-head outputs. Variable-length bbox decode remains in Python.

1. Run `deploy.export_onnx` in standard mode. Success produces a standard ONNX
   graph; failure identifies the first unsupported export operation.
2. If standard mode fails, rerun with `--fallthrough` to preserve unsupported
   operators when possible, then run `deploy.tensorrt.audit_onnx`.
3. Only after the unsupported-op list is understood should plugins or graph
   rewrites be implemented and a TensorRT engine built.

After the standard ONNX checker passes, audit the installed TensorRT parser
without building an engine:

```bash
python -m deploy.tensorrt.parse_onnx \
  --onnx outputs/deploy_onnx/racformer_raw_fp32.onnx \
  --out outputs/deploy_onnx/tensorrt_parser.txt
```

Use `--plugin /absolute/path/plugin.so` for each TensorRT plugin library once
plugins exist. The current PyTorch `bev_pool_v2_ext` is not a TensorRT plugin
and must not be passed to this option.

The image remains a UINT8 network input, but the model casts it to FP32 before
the first reshape. TensorRT 8.6 permits UINT8 at the network boundary but not as
an intermediate tensor. INT64-initializer downcast warnings can be audited after
the parser reaches the complete graph.

The LSS view transformer exports BEV pooling as the custom ONNX node
`mmdeploy::bev_pool_v2`. Its existing CUDA forward preserves PyTorch export
parity, but TensorRT will require a compatible plugin implementation.
The minimal FP32 TensorRT 8.6 implementation and build instructions live in
`plugins/bev_pool_v2/`.

After parser validation, export a fixture with `deploy.export_onnx --fixture`,
build the first FP32 engine with `deploy.tensorrt.build_engine`, and compare its
raw outputs with `deploy.tensorrt.validate_engine`. The engine builder applies
one shared voxel-count profile to all 24 dynamic radar tensors. Start with a
bounded profile that covers measured data; do not use the model's theoretical
40,000-voxel cap without checking target memory and build time.

If an x86 TensorRT installation was linked against a different cuDNN version
than the only library available in the PyTorch environment,
`build_engine --disable-cudnn-tactics` can isolate that mismatch without
changing the training environment. Such an engine is for functional diagnosis;
its latency is not the deployment performance baseline.

PyTorch 2.0 does not export `aten::atan2` at opset 17. The deployment exporter
lowers it to standard ONNX `Atan`, comparison, and `Where` nodes while retaining
the full quadrant behavior needed by box and polar-coordinate transforms.

Raw radar voxelization remains outside the TensorRT graph. Each of the eight
frames enters the graph as dynamic `voxels`, `num_points`, and batch-padded
`coors` tensors. The ROS/CUDA preprocessing path must reproduce MMCV
voxelization before engine execution. MSMV sampling and deformable attention
switch to their traceable PyTorch implementations only during ONNX export.

Do not commit `.onnx`, `.engine`, `.plan`, or profiling output. TensorRT engines
must ultimately be rebuilt for the target Jetson software and GPU environment.
