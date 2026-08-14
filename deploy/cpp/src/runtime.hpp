#pragma once

#include "preprocess.hpp"
#include "racformer/c_api.h"
#include "tensor_store.hpp"
#include "trt_engine.hpp"

#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace racformer {

struct RuntimeConfig {
    std::string image_engine, radar_engine, decoder_engine, plugin, manifest;
    std::array<float, 16> radar_to_ego{};
    uint64_t max_pair_delta_ns{};
    uint32_t max_pending_frames{16};
    uint32_t warmup_runs{};
    bool pad_startup_frames{true};
};

class Runtime {
 public:
    Runtime(RuntimeConfig config, racformer_result_callback_t callback, void* user_data);
    ~Runtime();
    void push_camera(const camera_data_t& input);
    void push_radar(const radar_data_t& input);
    void reset();
    std::string last_error() const;

 private:
    void pair_locked(uint32_t frame_id);
    void worker();
    void infer(const std::vector<PairedFrame>& frames);
    TensorMap merged_inputs(const TensorMap& dynamic) const;
    std::vector<int64_t> tensor_shape(const Tensor& tensor) const { return tensor.shape; }

    RuntimeConfig config_;
    racformer_result_callback_t callback_{};
    void* user_data_{};
    TensorMap constants_;
    Preprocessor preprocessor_;
    TrtLogger logger_;
    void* plugin_handle_{};
    std::unique_ptr<TrtEngine> image_, radar_, decoder_;
    cudaStream_t stream_{};
    void* bbox_ping_[2]{};
    void* feature_ping_[2]{};

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_{};
    bool configured_{};
    bool warmed_{};
    std::map<uint32_t, CameraCopy> cameras_;
    std::map<uint32_t, RadarCopy> radars_;
    std::deque<PairedFrame> queue_;
    std::vector<PairedFrame> history_;
    std::thread worker_;
    std::string error_;
};

}  // namespace racformer
