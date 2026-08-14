#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace racformer {

enum class DataType { kFloat32, kFloat16, kInt32, kUInt8, kBool };

struct Tensor {
    DataType dtype{DataType::kFloat32};
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
    size_t element_size() const;
    size_t element_count() const;
};

using TensorMap = std::unordered_map<std::string, Tensor>;

TensorMap load_tensor_manifest(const std::string& path);
Tensor make_float_tensor(std::vector<int64_t> shape);
Tensor make_int32_tensor(std::vector<int64_t> shape);
Tensor make_uint8_tensor(std::vector<int64_t> shape);

}  // namespace racformer
