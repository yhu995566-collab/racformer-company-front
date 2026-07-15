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

The LSS view transformer exports BEV pooling as the custom ONNX node
`mmdeploy::bev_pool_v2`. Its existing CUDA forward preserves PyTorch export
parity, but TensorRT will require a compatible plugin implementation.

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
