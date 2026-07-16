#include "bev_pool_v2_kernel.h"

namespace {

__global__ void bev_pool_v2_kernel(
    int channels, int intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval = index / channels;
  const int channel = index % channels;
  if (interval >= intervals) {
    return;
  }

  const int start = interval_starts[interval];
  const int length = interval_lengths[interval];
  float sum = 0.0F;
  for (int offset = 0; offset < length; ++offset) {
    const int rank = start + offset;
    sum += depth[ranks_depth[rank]] *
           feat[ranks_feat[rank] * channels + channel];
  }
  output[ranks_bev[start] * channels + channel] = sum;
}

}  // namespace

cudaError_t launch_bev_pool_v2(
    int channels, int intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* output,
    cudaStream_t stream) {
  if (intervals == 0 || channels == 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = (intervals * channels + threads - 1) / threads;
  bev_pool_v2_kernel<<<blocks, threads, 0, stream>>>(
      channels, intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, output);
  return cudaPeekAtLastError();
}
