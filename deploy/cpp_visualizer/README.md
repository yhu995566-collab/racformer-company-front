# RaCFormer real-time projection visualizer

This component is deliberately independent of TensorRT and the RaCFormer
runtime. It caches camera, radar, and prediction triggers by `(version,
frame_id)`, renders on a dedicated CPU thread, and returns one compressed
640x480 JPEG through a callback. There is no BEV panel.

The rendered image contains:

- raw radar points transformed to current ego and projected into the image;
- predictions within 50 m, drawn as red 3D boxes;
- class name and confidence;
- a small frame/radar/detection status line.

Radar points use the same near-red, middle-yellow, far-blue distance colors as
the repository visualization. Predictions default to score threshold 0.3 and
class-wise rotated BEV NMS IoU 0.2.

## Trigger ownership

The application fans out each camera and radar trigger to inference and this
visualizer. The inference result callback pushes prediction boxes. Input data
are deep-copied before every `racformer_vis_push_*` call returns. Rendering only
starts when all three inputs for one key are present.

The output callback runs on the visualizer worker, not the inference worker.
`output->jpeg_data` is valid only for the callback duration. Copy it before
placing it on another asynchronous queue.

See `apps/integration_example.cpp` for the adapter between the existing model
C ABI and the independent visualization ABI.

## Coordinates and calibration

Radar input uses raw sensor coordinates; `radar_to_ego` converts it to the
current model ego frame. Prediction boxes use the existing runtime output
layout and bottom-center `z` convention.

`ego_to_image` must project current-ego coordinates into the original 640x480
JPEG. For the current artifact, the first matrix in
`constants/lidar2img.bin` targets the cropped 640x256 model image. Load its
first 16 float32 values and set `projection_crop_y=224`; the visualizer restores
the original vertical image coordinate. If an original-image projection is
provided directly, set `projection_crop_y=0`.

## Nano build

Install the C++ OpenCV development package once:

```bash
sudo apt-get install -y libopencv-dev
```

Build independently from the model runtime:

```bash
cd /home/cttest/RaCFormer
cmake -S deploy/cpp_visualizer \
  -B build/racformer_visualizer \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/racformer_visualizer --parallel 2
```

The only production artifact built here is:

```text
build/racformer_visualizer/libracformer_visualizer.so
```

The software process includes `racformer/visualizer_c_api.h`, links this shared
library, and sends the returned JPEG to its existing frame display or transport
path. This library neither opens an HDMI window nor writes files by itself.
