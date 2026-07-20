#pragma once

#include <cuda_runtime_api.h>

cudaError_t launch_msmv_sampling(
    int num_levels, const void* const* inputs, const int* input_dims,
    int batch_size, int num_views, int channels, int num_query, int num_point,
    void* output, cudaStream_t stream);
