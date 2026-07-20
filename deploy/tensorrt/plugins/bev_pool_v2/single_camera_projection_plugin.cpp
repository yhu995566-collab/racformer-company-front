#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <array>
#include <cassert>
#include <cstring>
#include <string>

#include "single_camera_projection_kernel.h"

namespace racformer {
namespace {

constexpr char kPluginName[] = "racformer_single_camera_projection";
constexpr char kPluginVersion[] = "1";

}  // namespace

class SingleCameraProjectionPlugin final
    : public nvinfer1::IPluginV2DynamicExt {
 public:
  SingleCameraProjectionPlugin(int image_h, int image_w)
      : image_h_(image_h), image_w_(image_w) {}

  SingleCameraProjectionPlugin(const void* data, size_t length) {
    assert(length == 2 * sizeof(int));
    const auto* values = static_cast<const int*>(data);
    image_h_ = values[0];
    image_w_ = values[1];
  }

  const char* getPluginType() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int input_count,
      nvinfer1::IExprBuilder& builder) noexcept override {
    assert(output_index == 0);
    assert(input_count == 2);
    const auto& points = inputs[0];
    assert(points.nbDims == 6);
    const auto* batch_frames = builder.operation(
        nvinfer1::DimensionOperation::kPROD,
        *points.d[0], *points.d[2]);
    const auto* batch_frames_groups = builder.operation(
        nvinfer1::DimensionOperation::kPROD,
        *batch_frames, *points.d[3]);
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = batch_frames_groups;
    output.d[1] = points.d[1];
    output.d[2] = points.d[4];
    output.d[3] = builder.constant(3);
    return output;
  }

  bool supportsFormatCombination(
      int position, const nvinfer1::PluginTensorDesc* tensors,
      int input_count, int output_count) noexcept override {
    assert(input_count == 2);
    assert(output_count == 1);
    assert(position >= 0 && position < input_count + output_count);
    return tensors[position].format == nvinfer1::TensorFormat::kLINEAR &&
           tensors[position].type == nvinfer1::DataType::kFLOAT;
  }

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs, int input_count,
      const nvinfer1::DynamicPluginTensorDesc*, int output_count)
      noexcept override {
    assert(input_count == 2);
    assert(output_count == 1);
    assert(inputs[0].desc.dims.nbDims == 6);
    assert(inputs[1].desc.dims.nbDims == 4);
  }

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc*, int,
      const nvinfer1::PluginTensorDesc*, int) const noexcept override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc*, const void* const* inputs,
              void* const* outputs, void*, cudaStream_t stream)
      noexcept override {
    const auto& point_dims = input_desc[0].dims;
    const auto& matrix_dims = input_desc[1].dims;
    if (point_dims.nbDims != 6 || point_dims.d[5] != 3 ||
        matrix_dims.nbDims != 4 || matrix_dims.d[0] != point_dims.d[0] ||
        matrix_dims.d[1] != point_dims.d[2] ||
        matrix_dims.d[2] != 4 || matrix_dims.d[3] != 4) {
      return 1;
    }
    const auto status = launch_single_camera_projection(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        point_dims.d[0], point_dims.d[1], point_dims.d[2],
        point_dims.d[3], point_dims.d[4], image_h_, image_w_,
        static_cast<float*>(outputs[0]), stream);
    return status == cudaSuccess ? 0 : 1;
  }

  size_t getSerializationSize() const noexcept override {
    return 2 * sizeof(int);
  }

  void serialize(void* buffer) const noexcept override {
    auto* values = static_cast<int*>(buffer);
    values[0] = image_h_;
    values[1] = image_w_;
  }

  SingleCameraProjectionPlugin* clone() const noexcept override {
    auto* plugin = new SingleCameraProjectionPlugin(image_h_, image_w_);
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
  int image_h_{};
  int image_w_{};
  std::string namespace_;
};

class SingleCameraProjectionPluginCreator final
    : public nvinfer1::IPluginCreator {
 public:
  SingleCameraProjectionPluginCreator() {
    fields_[0] = nvinfer1::PluginField{
        "image_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[1] = nvinfer1::PluginField{
        "image_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
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
      const char*, const nvinfer1::PluginFieldCollection* fields)
      noexcept override {
    int image_h = 0;
    int image_w = 0;
    for (int index = 0; fields != nullptr && index < fields->nbFields;
         ++index) {
      const auto& field = fields->fields[index];
      if (field.data == nullptr ||
          field.type != nvinfer1::PluginFieldType::kINT32) {
        continue;
      }
      if (std::strcmp(field.name, "image_h") == 0) {
        image_h = *static_cast<const int*>(field.data);
      } else if (std::strcmp(field.name, "image_w") == 0) {
        image_w = *static_cast<const int*>(field.data);
      }
    }
    if (image_h <= 0 || image_w <= 0) {
      return nullptr;
    }
    auto* plugin = new SingleCameraProjectionPlugin(image_h, image_w);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char*, const void* data, size_t length) noexcept override {
    auto* plugin = new SingleCameraProjectionPlugin(data, length);
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
  std::array<nvinfer1::PluginField, 2> fields_{};
  nvinfer1::PluginFieldCollection field_collection_{};
  std::string namespace_;
};

REGISTER_TENSORRT_PLUGIN(SingleCameraProjectionPluginCreator);

}  // namespace racformer
