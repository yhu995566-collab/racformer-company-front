#pragma once

#include "tensor_store.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace racformer {

class TrtLogger final : public nvinfer1::ILogger {
 public:
    void log(Severity severity, const char* message) noexcept override;
};

class TrtEngine {
 public:
    TrtEngine(const std::string& path, TrtLogger& logger);
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    void configure(const TensorMap& available);
    void upload(const std::string& name, const Tensor& value, cudaStream_t stream);
    void bind(const std::string& name, void* device_pointer);
    bool enqueue(cudaStream_t stream);
    void download(const std::string& name, void* destination, size_t bytes,
                  cudaStream_t stream) const;

    bool has(const std::string& name) const;
    bool is_input(const std::string& name) const;
    void* pointer(const std::string& name) const;
    size_t bytes(const std::string& name) const;
    std::vector<int64_t> shape(const std::string& name) const;
    DataType dtype(const std::string& name) const;
    const std::vector<std::string>& names() const { return names_; }

 private:
    struct DestroyRuntime { void operator()(nvinfer1::IRuntime* p) const { delete p; } };
    struct DestroyEngine { void operator()(nvinfer1::ICudaEngine* p) const { delete p; } };
    struct DestroyContext { void operator()(nvinfer1::IExecutionContext* p) const { delete p; } };
    static size_t volume(const nvinfer1::Dims& dims);
    static DataType convert(nvinfer1::DataType dtype);

    std::unique_ptr<nvinfer1::IRuntime, DestroyRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, DestroyEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, DestroyContext> context_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, void*> owned_;
    std::unordered_map<std::string, size_t> sizes_;
};

void cuda_check(cudaError_t status, const char* operation);

}  // namespace racformer
