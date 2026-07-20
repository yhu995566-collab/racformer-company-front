#pragma once

#include <cuda_runtime_api.h>

cudaError_t launch_single_camera_projection(
    const float* sample_points, const float* lidar2img,
    int batch_size, int num_query, int num_frames, int num_groups,
    int num_points, int image_h, int image_w, float* output,
    cudaStream_t stream);
