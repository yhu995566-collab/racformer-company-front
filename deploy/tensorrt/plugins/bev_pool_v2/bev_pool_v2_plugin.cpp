#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include "bev_pool_v2_kernel.h"

namespace racformer {
namespace {

constexpr char kPluginName[] = "bev_pool_v2";
constexpr char kIdentityPluginName[] = "racformer_identity";
constexpr char kPluginVersion[] = "1";

template <typename T>
void write_value(char*& buffer, const T& value) {
  std::memcpy(buffer, &value, sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
T read_value(const char*& buffer) {
  T value;
  std::memcpy(&value, buffer, sizeof(T));
  buffer += sizeof(T);
  return value;
}

}  // namespace

class BEVPoolV2Plugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  BEVPoolV2Plugin(int out_height, int out_width)
      : out_height_(out_height), out_width_(out_width) {}

  BEVPoolV2Plugin(const void* data, size_t length) {
    assert(length == getSerializationSize());
    const char* cursor = static_cast<const char*>(data);
    out_height_ = read_value<int>(cursor);
    out_width_ = read_value<int>(cursor);
  }

  const char* getPluginType() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int input_count,
      nvinfer1::IExprBuilder& builder) noexcept override {
    assert(output_index == 0 && input_count == 7);
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = builder.constant(1);
    output.d[1] = builder.constant(out_height_);
    output.d[2] = builder.constant(out_width_);
    output.d[3] = inputs[1].d[3];
    return output;
  }

  bool supportsFormatCombination(
      int position, const nvinfer1::PluginTensorDesc* tensors,
      int input_count, int output_count) noexcept override {
    assert(input_count == 7 && output_count == 1 && position < 8);
    if (tensors[position].format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    if (position == 0 || position == 1 || position == 7) {
      return tensors[position].type == nvinfer1::DataType::kFLOAT;
    }
    return tensors[position].type == nvinfer1::DataType::kINT32;
  }

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc*, int,
      const nvinfer1::DynamicPluginTensorDesc*, int) noexcept override {}

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc*, int,
      const nvinfer1::PluginTensorDesc*, int) const noexcept override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs, void* const* outputs, void*,
              cudaStream_t stream) noexcept override {
    const int channels = input_desc[1].dims.d[3];
    const int intervals = input_desc[6].dims.d[0];
    size_t output_elements = 1;
    for (int index = 0; index < output_desc[0].dims.nbDims; ++index) {
      output_elements *= static_cast<size_t>(output_desc[0].dims.d[index]);
    }
    auto status = cudaMemsetAsync(
        outputs[0], 0, output_elements * sizeof(float), stream);
    if (status != cudaSuccess) {
      return 1;
    }
    status = launch_bev_pool_v2(
        channels, intervals, static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const int*>(inputs[2]),
        static_cast<const int*>(inputs[3]),
        static_cast<const int*>(inputs[4]),
        static_cast<const int*>(inputs[5]),
        static_cast<const int*>(inputs[6]),
        static_cast<float*>(outputs[0]), stream);
    return status == cudaSuccess ? 0 : 1;
  }

  size_t getSerializationSize() const noexcept override {
    return 2 * sizeof(int);
  }

  void serialize(void* buffer) const noexcept override {
    char* cursor = static_cast<char*>(buffer);
    write_value(cursor, out_height_);
    write_value(cursor, out_width_);
  }

  BEVPoolV2Plugin* clone() const noexcept override {
    auto* plugin = new BEVPoolV2Plugin(out_height_, out_width_);
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
  int out_height_{};
  int out_width_{};
  std::string namespace_;
};

class BEVPoolV2PluginCreator final : public nvinfer1::IPluginCreator {
 public:
  BEVPoolV2PluginCreator() {
    fields_.emplace_back(
        nvinfer1::PluginField{"out_height", nullptr,
                             nvinfer1::PluginFieldType::kINT32, 1});
    fields_.emplace_back(
        nvinfer1::PluginField{"out_width", nullptr,
                             nvinfer1::PluginFieldType::kINT32, 1});
    field_collection_.nbFields = static_cast<int>(fields_.size());
    field_collection_.fields = fields_.data();
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection* fields) noexcept override {
    int out_height = 0;
    int out_width = 0;
    for (int index = 0; index < fields->nbFields; ++index) {
      const auto& field = fields->fields[index];
      if (std::strcmp(field.name, "out_height") == 0) {
        out_height = *static_cast<const int*>(field.data);
      } else if (std::strcmp(field.name, "out_width") == 0) {
        out_width = *static_cast<const int*>(field.data);
      }
    }
    if (out_height <= 0 || out_width <= 0) {
      return nullptr;
    }
    auto* plugin = new BEVPoolV2Plugin(out_height, out_width);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char*, const void* data, size_t length) noexcept override {
    auto* plugin = new BEVPoolV2Plugin(data, length);
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
  std::vector<nvinfer1::PluginField> fields_;
  nvinfer1::PluginFieldCollection field_collection_{};
};

REGISTER_TENSORRT_PLUGIN(BEVPoolV2PluginCreator);

class IdentityPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  IdentityPlugin() = default;
  IdentityPlugin(const void*, size_t length) { assert(length == 0); }

  const char* getPluginType() const noexcept override {
    return kIdentityPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int input_count,
      nvinfer1::IExprBuilder&) noexcept override {
    assert(output_index == 0 && input_count == 1);
    return inputs[0];
  }

  bool supportsFormatCombination(
      int position, const nvinfer1::PluginTensorDesc* tensors,
      int input_count, int output_count) noexcept override {
    assert(input_count == 1 && output_count == 1 && position < 2);
    if (tensors[position].format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const auto type = tensors[position].type;
    if (type != nvinfer1::DataType::kFLOAT &&
        type != nvinfer1::DataType::kHALF) {
      return false;
    }
    return position == 0 || type == tensors[0].type;
  }

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc*, int,
      const nvinfer1::DynamicPluginTensorDesc*, int) noexcept override {}

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc*, int,
      const nvinfer1::PluginTensorDesc*, int) const noexcept override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc*, const void* const* inputs,
              void* const* outputs, void*,
              cudaStream_t stream) noexcept override {
    size_t elements = 1;
    for (int index = 0; index < input_desc[0].dims.nbDims; ++index) {
      elements *= static_cast<size_t>(input_desc[0].dims.d[index]);
    }
    const size_t element_size =
        input_desc[0].type == nvinfer1::DataType::kHALF ? 2 : 4;
    const auto status = cudaMemcpyAsync(
        outputs[0], inputs[0], elements * element_size,
        cudaMemcpyDeviceToDevice, stream);
    return status == cudaSuccess ? 0 : 1;
  }

  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void*) const noexcept override {}

  IdentityPlugin* clone() const noexcept override {
    auto* plugin = new IdentityPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::DataType getOutputDataType(
      int, const nvinfer1::DataType* input_types,
      int) const noexcept override {
    return input_types[0];
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
  std::string namespace_;
};

class IdentityPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  IdentityPluginCreator() {
    field_collection_.nbFields = 0;
    field_collection_.fields = nullptr;
  }

  const char* getPluginName() const noexcept override {
    return kIdentityPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*) noexcept override {
    auto* plugin = new IdentityPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char*, const void* data, size_t length) noexcept override {
    auto* plugin = new IdentityPlugin(data, length);
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

REGISTER_TENSORRT_PLUGIN(IdentityPluginCreator);

}  // namespace racformer
