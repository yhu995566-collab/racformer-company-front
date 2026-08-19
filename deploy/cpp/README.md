# RaCFormer C++ sensor runtime

This directory is the Python-free runtime for the company-front four-frame
deployments. Runtime geometry (including the 50 m / 200-query contract) is
loaded from exported fixture constants rather than compiled into the library.
It consumes two callbacks from different producer threads, pairs
them by `frame_id`, keeps four paired frames in a fixed-capacity ring, and runs the image/LSS,
radar, and six-pass recurrent decoder TensorRT engines on one worker thread.

Temporal snapshots use shared ownership. Advancing or resetting the ring does
not copy the previous four JPEG/radar payloads and cannot invalidate a window
already held by the inference worker. This is an input-history optimization;
the current engines still recompute all four frames' frontend features.

## ABI contract

- `timestamp` is unsigned nanoseconds from one shared monotonic clock domain.
- `p_camera_data` points to one complete compressed 640x480 JPEG and
  `data_size` is its byte count.
- `p_radar_data` contains `radar_data_count` contiguous points.
- Both input buffers may be released as soon as their `racformer_push_*` call
  returns; the runtime deep-copies them synchronously.
- Camera and radar are paired by equal `frame_id`; `version` must also match.
- The result callback runs on the inference worker. Result pointers are valid
  only until that callback returns. Copy them if another thread retains them.
- Boxes are `[x,y,z_bottom,dx,dy,dz,yaw,vx,vy]` in the model ego frame.

The runtime ignores the incoming `vx` and `vy`, because the agreed radar source
only guarantees radial `v`. In the native radar frame it computes

```
r = hypot(x, y)
v_comp = v + ego_speed * y / r
vx = v_comp * x / r
vy = v_comp * y / r
```

and rotates this velocity (without translation) into the model ego frame.
Both `v` and `ego_speed` must use metres per second, and `ego_speed` must be
positive in the vehicle-forward direction. `mag`, `snr`, and `label` are not
model inputs in this checkpoint; `rcs` is used.
This sign must still be confirmed using one stationary-object capture: after
compensation, a stationary roadside target should have near-zero ego velocity.

Startup matches the Python implementation by repeating the oldest available
paired frame until four frames exist. Set `pad_startup_frames=0` to suppress
results until the four-frame buffer is full. A voxel count over 1024 is a hard
error and is never silently truncated.

## Prepare constants

Run this once on any machine with NumPy, using the exact fixture that validated
the three engines:

```bash
python3 deploy/cpp/tools/export_runtime_constants.py \
  --fixture outputs/deploy_onnx_50m_q200_f4/racformer_50m_q200_f4_frontend_sample0.npz \
  --voxel-size 0.5 0.5 6.0 \
  --depth-range 1.0 55.0 \
  --max-detections 300 \
  --out-dir outputs/deploy_runtime_50m_q200_f4/constants
```

Transfer the complete constants directory to Nano and verify it with
`sha256sum -c SHA256SUMS` from inside that directory.

## Build on Nano

Install `libjpeg-dev` if it is not already present, then:

```bash
cd /home/cttest/RaCFormer
cmake -S deploy/cpp -B build/racformer_runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/racformer_runtime --parallel 2
```

Link the software team's process against `libracformer_runtime.so` and include
`racformer/c_api.h`. See `apps/offline_demo.cpp` for initialization and callback
ownership. The demo's calibration matrix is the currently audited static
radar-to-ego matrix; production code should pass the versioned calibration
owned by the system instead of copying that sample blindly.

The demo also accepts a recorded `replay_export` root containing
`images/cam_1/NNNNNNN.jpg` and `radar_ply/NNNNNNN.ply`:

```bash
racformer_offline_demo IMAGE.engine RADAR.engine DECODER.engine PLUGIN.so \
  manifest.tsv --sequence /path/to/replay_export 0 4 100000000
```

The last value is the frame period in nanoseconds. The current replay
`sync_index.csv` contains indices but no timestamps, so this must be replaced
with the capture system's real period before treating temporal output as a
parity result. The demo uses `export_index` as the callback `frame_id`; source
`radar_frame_id` is diagnostic only because synchronized exports can repeat it.

## Required parity gates

Before live integration, compare one recorded frame sequence against the Python
fixture path. Check the exact detection count, labels, scores, and boxes with the
existing 0.03 m decoded-box tolerance. Then test asynchronous callback order,
duplicate/missing frame IDs, timestamp delta rejection, startup padding, reset,
and a radar frame exceeding 1024 voxels.
