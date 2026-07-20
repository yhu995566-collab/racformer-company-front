#include "single_camera_projection_kernel.h"

#include <cuda_runtime.h>

namespace {

__global__ void single_camera_projection_kernel(
    const int count, const float* sample_points, const float* lidar2img,
    const int num_query, const int num_frames, const int num_groups,
    const int num_points, const float image_h, const float image_w,
    float* output) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    int offset = index;
    const int point = offset % num_points;
    offset /= num_points;
    const int query = offset % num_query;
    offset /= num_query;
    const int group = offset % num_groups;
    offset /= num_groups;
    const int frame = offset % num_frames;
    const int batch = offset / num_frames;

    const int sample_index =
        (((((batch * num_query + query) * num_frames + frame) *
             num_groups + group) * num_points + point) * 3);
    const float x = sample_points[sample_index];
    const float y = sample_points[sample_index + 1];
    const float z = sample_points[sample_index + 2];

    const float* matrix =
        lidar2img + ((batch * num_frames + frame) * 16);
    const float projected_x =
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
    const float projected_y =
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
    const float projected_z =
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
    const float denominator = fmaxf(projected_z, 1e-5F);

    const int output_index =
        (((((batch * num_frames + frame) * num_groups + group) *
             num_query + query) * num_points + point) * 3);
    output[output_index] = projected_x / denominator / image_w;
    output[output_index + 1] = projected_y / denominator / image_h;
    output[output_index + 2] = 0.0F;
  }
}

}  // namespace

cudaError_t launch_single_camera_projection(
    const float* sample_points, const float* lidar2img,
    int batch_size, int num_query, int num_frames, int num_groups,
    int num_points, int image_h, int image_w, float* output,
    cudaStream_t stream) {
  if (batch_size <= 0 || num_query <= 0 || num_frames <= 0 ||
      num_groups <= 0 || num_points <= 0 || image_h <= 0 || image_w <= 0) {
    return cudaErrorInvalidValue;
  }
  const int count =
      batch_size * num_frames * num_groups * num_query * num_points;
  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  single_camera_projection_kernel<<<blocks, threads, 0, stream>>>(
      count, sample_points, lidar2img, num_query, num_frames, num_groups,
      num_points, static_cast<float>(image_h), static_cast<float>(image_w),
      output);
  return cudaPeekAtLastError();
}
