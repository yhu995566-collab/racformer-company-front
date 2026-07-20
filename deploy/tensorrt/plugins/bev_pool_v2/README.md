# BEVPoolV2 TensorRT Plugin

This is a minimal TensorRT 8.6 dynamic plugin for the exported
`mmdeploy::bev_pool_v2` node. The first version supports FP32 tensors, INT32
indices, batch size one, and a dynamic number of intervals. It reuses the
RaCFormer BEVPoolV2 forward-kernel algorithm without linking against PyTorch.

Build it separately for each target platform. For the current L20 server:

```bash
cmake -S deploy/tensorrt/plugins/bev_pool_v2 \
  -B build/tensorrt_bev_pool_v2 \
  -DTENSORRT_ROOT=/path/to/TensorRT-8.6.1 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/tensorrt_bev_pool_v2 --parallel
```

If TensorRT headers and libraries are installed under the Conda environment or
standard system paths, `-DTENSORRT_ROOT` can be omitted. For Jetson, rebuild on
the device with its own TensorRT headers and target CUDA architecture; do not
copy this x86_64 plugin binary to Jetson.

The library registers `bev_pool_v2`, `racformer_identity`,
`racformer_msmv_sampling`, and `racformer_single_camera_projection`. The MSMV
plugin reuses the repository's FP32 multi-scale sampling CUDA forward kernel.
The projection plugin replaces the fixed single-camera projection and
coordinate-packing graph that feeds MSMV sampling.

Load the resulting library during parser audit:

```bash
python -m deploy.tensorrt.parse_onnx \
  --onnx outputs/deploy_onnx/racformer_raw_fp32.onnx \
  --plugin build/tensorrt_bev_pool_v2/libracformer_bev_pool_v2_trt.so \
  --out outputs/deploy_onnx/tensorrt_parser_with_bev_pool.txt
```
