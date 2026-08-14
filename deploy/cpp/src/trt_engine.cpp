#include "trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace racformer {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void TrtLogger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
}

TrtEngine::TrtEngine(const std::string& path, TrtLogger& logger) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open TensorRT engine: " + path);
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), size);
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("engine deserialization failed: " + path);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed: " + path);
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
        names_.emplace_back(engine_->getIOTensorName(i));
    }
}

TrtEngine::~TrtEngine() {
    for (auto& item : owned_) cudaFree(item.second);
}

size_t TrtEngine::volume(const nvinfer1::Dims& dims) {
    size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("unresolved TensorRT dimension");
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
}

DataType TrtEngine::convert(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return DataType::kFloat32;
        case nvinfer1::DataType::kHALF: return DataType::kFloat16;
        case nvinfer1::DataType::kINT32: return DataType::kInt32;
        case nvinfer1::DataType::kINT8: return DataType::kUInt8;
        case nvinfer1::DataType::kBOOL: return DataType::kBool;
        case nvinfer1::DataType::kUINT8: return DataType::kUInt8;
        default: throw std::runtime_error("unsupported TensorRT tensor dtype");
    }
}

void TrtEngine::configure(const TensorMap& available) {
    for (const auto& name : names_) {
        if (!is_input(name)) continue;
        const auto found = available.find(name);
        if (found == available.end()) throw std::runtime_error("missing engine input metadata: " + name);
        if (found->second.dtype != convert(engine_->getTensorDataType(name.c_str())))
            throw std::runtime_error("input dtype mismatch: " + name);
        nvinfer1::Dims dims{};
        dims.nbDims = static_cast<int32_t>(found->second.shape.size());
        if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) throw std::runtime_error("too many dimensions: " + name);
        for (int i = 0; i < dims.nbDims; ++i) dims.d[i] = static_cast<int32_t>(found->second.shape[i]);
        if (!context_->setInputShape(name.c_str(), dims)) throw std::runtime_error("invalid input shape: " + name);
    }
    if (!context_->allInputDimensionsSpecified()) throw std::runtime_error("not all TensorRT shapes specified");
    for (const auto& name : names_) {
        const auto dims = context_->getTensorShape(name.c_str());
        const size_t size = volume(dims) * [&] {
            switch (convert(engine_->getTensorDataType(name.c_str()))) {
                case DataType::kFloat32: return size_t{4};
                case DataType::kFloat16: return size_t{2};
                case DataType::kInt32: return size_t{4};
                case DataType::kUInt8: return size_t{1};
                case DataType::kBool: return size_t{1};
            }
            return size_t{0};
        }();
        void* pointer = nullptr;
        cuda_check(cudaMalloc(&pointer, size), "cudaMalloc tensor");
        owned_[name] = pointer;
        sizes_[name] = size;
        if (!context_->setTensorAddress(name.c_str(), pointer)) throw std::runtime_error("setTensorAddress failed: " + name);
    }
}

void TrtEngine::upload(const std::string& name, const Tensor& value, cudaStream_t stream) {
    if (!is_input(name)) throw std::runtime_error("not an engine input: " + name);
    if (value.bytes.size() != bytes(name)) throw std::runtime_error("input byte count mismatch: " + name);
    cuda_check(cudaMemcpyAsync(pointer(name), value.bytes.data(), value.bytes.size(), cudaMemcpyHostToDevice, stream), "input H2D");
}

void TrtEngine::bind(const std::string& name, void* device_pointer) {
    if (!has(name) || device_pointer == nullptr) throw std::runtime_error("invalid external binding: " + name);
    if (!context_->setTensorAddress(name.c_str(), device_pointer)) throw std::runtime_error("external binding failed: " + name);
}

bool TrtEngine::enqueue(cudaStream_t stream) { return context_->enqueueV3(stream); }

void TrtEngine::download(const std::string& name, void* destination, size_t size, cudaStream_t stream) const {
    if (size != bytes(name)) throw std::runtime_error("output byte count mismatch: " + name);
    cuda_check(cudaMemcpyAsync(destination, pointer(name), size, cudaMemcpyDeviceToHost, stream), "output D2H");
}

bool TrtEngine::has(const std::string& name) const {
    // TensorRT 8.5 logs an internal error when getTensorIOMode() is called
    // with an unknown name. Consult the enumerated I/O list first so probing
    // another split engine remains silent.
    return std::find(names_.begin(), names_.end(), name) != names_.end();
}
bool TrtEngine::is_input(const std::string& name) const {
    return engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
}
void* TrtEngine::pointer(const std::string& name) const {
    const auto it = owned_.find(name);
    if (it == owned_.end()) throw std::runtime_error("unknown tensor: " + name);
    return it->second;
}
size_t TrtEngine::bytes(const std::string& name) const {
    const auto it = sizes_.find(name);
    if (it == sizes_.end()) throw std::runtime_error("unknown tensor size: " + name);
    return it->second;
}
std::vector<int64_t> TrtEngine::shape(const std::string& name) const {
    const auto dims = context_->getTensorShape(name.c_str());
    return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
}
DataType TrtEngine::dtype(const std::string& name) const {
    return convert(engine_->getTensorDataType(name.c_str()));
}
}  // namespace racformer
