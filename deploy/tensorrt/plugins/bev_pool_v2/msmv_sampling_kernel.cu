#define RACFORMER_TRT_PLUGIN
#include "../../../../models/csrc/msmv_sampling/msmv_sampling_forward.cu"

#include "msmv_sampling_kernel.h"

cudaError_t launch_msmv_sampling(
    int num_levels, const void* const* inputs, const int* input_dims,
    int batch_size, int num_views, int channels, int num_query, int num_point,
    void* output, cudaStream_t stream) {
  const auto* sampling_locations =
      static_cast<const float*>(inputs[num_levels]);
  const auto* scale_weights =
      static_cast<const float*>(inputs[num_levels + 1]);
  auto* output_data = static_cast<float*>(output);

  if (num_levels == 2) {
    ms_deformable_im2col_cuda_c45(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        sampling_locations, scale_weights, batch_size, channels, num_views,
        num_query, num_point, output_data, stream);
  } else if (num_levels == 4) {
    ms_deformable_im2col_cuda_c2345(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        input_dims[4], input_dims[5], input_dims[6], input_dims[7],
        sampling_locations, scale_weights, batch_size, channels, num_views,
        num_query, num_point, output_data, stream);
  } else if (num_levels == 5) {
    ms_deformable_im2col_cuda_c23456(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        static_cast<const float*>(inputs[4]),
        input_dims[0], input_dims[1], input_dims[2], input_dims[3],
        input_dims[4], input_dims[5], input_dims[6], input_dims[7],
        input_dims[8], input_dims[9], sampling_locations, scale_weights,
        batch_size, channels, num_views, num_query, num_point, output_data,
        stream);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}
