#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cassert>
#include <cstring>
#include <string>

#include "msmv_sampling_kernel.h"

namespace racformer {
namespace {

constexpr char kPluginName[] = "racformer_msmv_sampling";
constexpr char kPluginVersion[] = "1";

}  // namespace

class MSMVSamplingPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  MSMVSamplingPlugin() = default;

  MSMVSamplingPlugin(const void* data, size_t length) {
    assert(length == sizeof(num_levels_));
    std::memcpy(&num_levels_, data, sizeof(num_levels_));
  }

  const char* getPluginType() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int input_count,
      nvinfer1::IExprBuilder&) noexcept override {
    assert(output_index == 0);
    assert(input_count == 4 || input_count == 6 || input_count == 7);
    const int num_levels = input_count - 2;
    const auto& sampling_locations = inputs[num_levels];
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = sampling_locations.d[1];
    output.d[2] = inputs[0].d[4];
    output.d[3] = sampling_locations.d[2];
    return output;
  }

  bool supportsFormatCombination(
      int position, const nvinfer1::PluginTensorDesc* tensors,
      int input_count, int output_count) noexcept override {
    assert(output_count == 1);
    assert(input_count == 4 || input_count == 6 || input_count == 7);
    assert(position >= 0 && position < input_count + output_count);
    return tensors[position].format == nvinfer1::TensorFormat::kLINEAR &&
           tensors[position].type == nvinfer1::DataType::kFLOAT;
  }

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc*, int input_count,
      const nvinfer1::DynamicPluginTensorDesc*, int output_count) noexcept override {
    assert(output_count == 1);
    num_levels_ = input_count - 2;
    assert(num_levels_ == 2 || num_levels_ == 4 || num_levels_ == 5);
  }

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc*, int,
      const nvinfer1::PluginTensorDesc*, int) const noexcept override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc*, const void* const* inputs,
              void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (num_levels_ != 2 && num_levels_ != 4 && num_levels_ != 5) {
      return 1;
    }
    const auto& feature_dims = input_desc[0].dims;
    const auto& sampling_dims = input_desc[num_levels_].dims;
    if (feature_dims.nbDims != 5 || sampling_dims.nbDims != 4) {
      return 1;
    }

    int spatial_dims[10]{};
    for (int level = 0; level < num_levels_; ++level) {
      const auto& dims = input_desc[level].dims;
      if (dims.nbDims != 5 || dims.d[0] != feature_dims.d[0] ||
          dims.d[1] != feature_dims.d[1] ||
          dims.d[4] != feature_dims.d[4]) {
        return 1;
      }
      spatial_dims[2 * level] = dims.d[2];
      spatial_dims[2 * level + 1] = dims.d[3];
    }

    const auto status = launch_msmv_sampling(
        num_levels_, inputs, spatial_dims, feature_dims.d[0],
        feature_dims.d[1], feature_dims.d[4], sampling_dims.d[1],
        sampling_dims.d[2], outputs[0], stream);
    return status == cudaSuccess ? 0 : 1;
  }

  size_t getSerializationSize() const noexcept override {
    return sizeof(num_levels_);
  }

  void serialize(void* buffer) const noexcept override {
    std::memcpy(buffer, &num_levels_, sizeof(num_levels_));
  }

  MSMVSamplingPlugin* clone() const noexcept override {
    auto* plugin = new MSMVSamplingPlugin();
    plugin->num_levels_ = num_levels_;
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::DataType getOutputDataType(
      int, const nvinfer1::DataType*, int) const noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }

  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  void destroy() noexcept override { delete this; }

  void setPluginNamespace(const char* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

  void attachToContext(cudnnContext*, cublasContext*,
                       nvinfer1::IGpuAllocator*) noexcept override {}
  void detachFromContext() noexcept override {}

 private:
  int num_levels_{};
  std::string namespace_;
};

class MSMVSamplingPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  MSMVSamplingPluginCreator() {
    field_collection_.nbFields = 0;
    field_collection_.fields = nullptr;
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*) noexcept override {
    auto* plugin = new MSMVSamplingPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char*, const void* data, size_t length) noexcept override {
    auto* plugin = new MSMVSamplingPlugin(data, length);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  void setPluginNamespace(const char* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_{};
};

REGISTER_TENSORRT_PLUGIN(MSMVSamplingPluginCreator);

}  // namespace racformer
