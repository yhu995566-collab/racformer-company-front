#include "tensor_store.hpp"

#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace racformer {
namespace {
DataType parse_dtype(const std::string& value) {
    if (value == "float32") return DataType::kFloat32;
    if (value == "float16") return DataType::kFloat16;
    if (value == "int32") return DataType::kInt32;
    if (value == "uint8") return DataType::kUInt8;
    if (value == "bool") return DataType::kBool;
    throw std::runtime_error("unsupported tensor dtype: " + value);
}
}  // namespace

size_t Tensor::element_size() const {
    switch (dtype) {
        case DataType::kFloat32: return 4;
        case DataType::kFloat16: return 2;
        case DataType::kInt32: return 4;
        case DataType::kUInt8: return 1;
        case DataType::kBool: return 1;
    }
    throw std::runtime_error("invalid tensor dtype");
}

size_t Tensor::element_count() const {
    return std::accumulate(shape.begin(), shape.end(), size_t{1},
                           [](size_t a, int64_t b) {
                               if (b < 0) throw std::runtime_error("negative tensor dimension");
                               return a * static_cast<size_t>(b);
                           });
}

TensorMap load_tensor_manifest(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open constants manifest: " + path);
    const auto base = std::filesystem::absolute(path).parent_path();
    TensorMap result;
    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string name, dtype, shape_text, filename;
        if (!std::getline(fields, name, '\t') || !std::getline(fields, dtype, '\t') ||
            !std::getline(fields, shape_text, '\t') || !std::getline(fields, filename)) {
            throw std::runtime_error("invalid manifest line " + std::to_string(line_number));
        }
        Tensor tensor;
        tensor.dtype = parse_dtype(dtype);
        std::istringstream dimensions(shape_text);
        std::string dimension;
        while (std::getline(dimensions, dimension, ',')) {
            if (!dimension.empty()) tensor.shape.push_back(std::stoll(dimension));
        }
        const size_t expected = tensor.element_count() * tensor.element_size();
        std::ifstream binary(base / filename, std::ios::binary | std::ios::ate);
        if (!binary) throw std::runtime_error("cannot open tensor file: " + filename);
        const auto size = static_cast<size_t>(binary.tellg());
        if (size != expected) throw std::runtime_error("tensor byte count mismatch: " + name);
        tensor.bytes.resize(size);
        binary.seekg(0);
        binary.read(reinterpret_cast<char*>(tensor.bytes.data()), size);
        result.emplace(name, std::move(tensor));
    }
    return result;
}

Tensor make_float_tensor(std::vector<int64_t> shape) {
    Tensor result{DataType::kFloat32, std::move(shape), {}};
    result.bytes.resize(result.element_count() * sizeof(float));
    return result;
}
Tensor make_int32_tensor(std::vector<int64_t> shape) {
    Tensor result{DataType::kInt32, std::move(shape), {}};
    result.bytes.resize(result.element_count() * sizeof(int32_t));
    return result;
}
Tensor make_uint8_tensor(std::vector<int64_t> shape) {
    Tensor result{DataType::kUInt8, std::move(shape), {}};
    result.bytes.resize(result.element_count());
    return result;
}
}  // namespace racformer
