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

Do not commit `.onnx`, `.engine`, `.plan`, or profiling output. TensorRT engines
must ultimately be rebuilt for the target Jetson software and GPU environment.
