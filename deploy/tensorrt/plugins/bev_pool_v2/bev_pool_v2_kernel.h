#pragma once

#include <cuda_runtime_api.h>

cudaError_t launch_bev_pool_v2(
    int channels, int intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* output,
    cudaStream_t stream);
