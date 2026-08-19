#include "runtime.hpp"

#include <NvInferPlugin.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace racformer {
namespace {
const char* const kStateInputs[] = {"query_bbox", "query_feat"};
const char* const kStateOutputs[] = {"next_query_bbox", "next_query_feat"};
constexpr float kRegions[6] = {0.08F, 0.07F, 0.06F, 0.05F, 0.04F, 0.03F};
constexpr float kTwoPi = 6.2831853071795864769F;

Tensor metadata(DataType dtype, std::vector<int64_t> shape) {
    Tensor result;
    result.dtype = dtype;
    result.shape = std::move(shape);
    return result;
}
const Tensor& choose(const TensorMap& dynamic, const TensorMap& constants,
                     const std::string& name) {
    const auto live = dynamic.find(name);
    if (live != dynamic.end()) return live->second;
    const auto fixed = constants.find(name);
    if (fixed != constants.end()) return fixed->second;
    throw std::runtime_error("no runtime or constant input named " + name);
}
}  // namespace

Runtime::Runtime(RuntimeConfig config, racformer_result_callback_t callback, void* user_data)
    : config_(std::move(config)), callback_(callback), user_data_(user_data),
      constants_(load_tensor_manifest(config_.manifest)),
      preprocessor_(config_.radar_to_ego, constants_) {
    plugin_handle_ = dlopen(config_.plugin.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!plugin_handle_) throw std::runtime_error(std::string("plugin load failed: ") + dlerror());
    if (!initLibNvInferPlugins(&logger_, "")) throw std::runtime_error("initLibNvInferPlugins failed");
    image_ = std::make_unique<TrtEngine>(config_.image_engine, logger_);
    radar_ = std::make_unique<TrtEngine>(config_.radar_engine, logger_);
    decoder_ = std::make_unique<TrtEngine>(config_.decoder_engine, logger_);
    cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    worker_ = std::thread(&Runtime::worker, this);
}

Runtime::~Runtime() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (void* p : bbox_ping_) if (p) cudaFree(p);
    for (void* p : feature_ping_) if (p) cudaFree(p);
    if (stream_) cudaStreamDestroy(stream_);
    decoder_.reset(); radar_.reset(); image_.reset();
    if (plugin_handle_) dlclose(plugin_handle_);
}

void Runtime::push_camera(const camera_data_t& input) {
    if (!input.p_camera_data || input.data_size == 0) throw std::invalid_argument("camera JPEG pointer/size is invalid");
    CameraCopy copy{input.timestamp, input.frame_id, input.version, {}};
    const auto* begin = static_cast<const uint8_t*>(input.p_camera_data);
    copy.jpeg.assign(begin, begin + input.data_size);
    std::lock_guard<std::mutex> lock(mutex_);
    cameras_[copy.frame_id] = std::move(copy);
    pair_locked(input.frame_id);
}

void Runtime::push_radar(const radar_data_t& input) {
    if (input.radar_data_count && !input.p_radar_data) throw std::invalid_argument("radar pointer is null");
    RadarCopy copy{input.timestamp, input.frame_id, input.version, {}};
    if (input.radar_data_count)
        copy.points.assign(input.p_radar_data, input.p_radar_data + input.radar_data_count);
    std::lock_guard<std::mutex> lock(mutex_);
    radars_[copy.frame_id] = std::move(copy);
    pair_locked(input.frame_id);
}

void Runtime::pair_locked(uint32_t id) {
    auto camera = cameras_.find(id);
    auto radar = radars_.find(id);
    if (camera == cameras_.end() || radar == radars_.end()) return;
    const uint64_t delta = camera->second.timestamp > radar->second.timestamp
        ? camera->second.timestamp - radar->second.timestamp
        : radar->second.timestamp - camera->second.timestamp;
    if (config_.max_pair_delta_ns && delta > config_.max_pair_delta_ns) {
        cameras_.erase(camera); radars_.erase(radar);
        error_ = "paired timestamps exceed max_pair_delta_ns at frame " + std::to_string(id);
        return;
    }
    if (camera->second.version != radar->second.version) {
        cameras_.erase(camera); radars_.erase(radar);
        error_ = "camera/radar version mismatch at frame " + std::to_string(id);
        return;
    }
    if (queue_.size() >= config_.max_pending_frames) queue_.pop_front();
    queue_.push_back({std::move(camera->second), std::move(radar->second)});
    cameras_.erase(camera); radars_.erase(radar);
    while (cameras_.size() > config_.max_pending_frames) cameras_.erase(cameras_.begin());
    while (radars_.size() > config_.max_pending_frames) radars_.erase(radars_.begin());
    condition_.notify_one();
}

void Runtime::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    cameras_.clear(); radars_.clear(); queue_.clear(); history_.clear(); error_.clear();
}

std::string Runtime::last_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

TensorMap Runtime::merged_inputs(const TensorMap& dynamic) const {
    TensorMap result = constants_;
    for (const auto& item : dynamic) result[item.first] = item.second;
    return result;
}

void Runtime::worker() {
    for (;;) {
        PairedFrame frame;
        PairedFrameWindow snapshot{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_) return;
            frame = std::move(queue_.front());
            queue_.pop_front();
            if (!history_.empty() &&
                (frame.camera.timestamp < history_.newest().camera.timestamp ||
                 frame.radar.timestamp < history_.newest().radar.timestamp)) {
                error_ = "out-of-order paired frame dropped: " + std::to_string(frame.camera.frame_id);
                continue;
            }
            history_.push(std::move(frame));
            if (history_.size() < kTemporalFrameCount &&
                !config_.pad_startup_frames) continue;
            snapshot = history_.padded_window();
        }
        try {
            infer(snapshot);
        } catch (const std::exception& exception) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = exception.what();
        }
    }
}

void Runtime::infer(const PairedFrameWindow& frames) {
    TensorMap dynamic = preprocessor_.prepare(frames);
    TensorMap available = merged_inputs(dynamic);
    if (!configured_) {
        image_->configure(available);
        radar_->configure(available);
        TensorMap decoder_available = available;
        for (const auto& name : image_->names()) if (!image_->is_input(name))
            decoder_available[name] = metadata(image_->dtype(name), image_->shape(name));
        for (const auto& name : radar_->names()) if (!radar_->is_input(name))
            decoder_available[name] = metadata(radar_->dtype(name), radar_->shape(name));
        decoder_available["d_region"] = make_float_tensor({1});
        decoder_->configure(decoder_available);
        cuda_check(cudaMalloc(&bbox_ping_[0], decoder_->bytes("next_query_bbox")), "bbox state cudaMalloc");
        cuda_check(cudaMalloc(&bbox_ping_[1], decoder_->bytes("next_query_bbox")), "bbox state cudaMalloc");
        cuda_check(cudaMalloc(&feature_ping_[0], decoder_->bytes("next_query_feat")), "feature state cudaMalloc");
        cuda_check(cudaMalloc(&feature_ping_[1], decoder_->bytes("next_query_feat")), "feature state cudaMalloc");
        configured_ = true;
    }
    for (const auto& name : image_->names()) if (image_->is_input(name)) image_->upload(name, choose(dynamic, constants_, name), stream_);
    for (const auto& name : radar_->names()) if (radar_->is_input(name)) radar_->upload(name, choose(dynamic, constants_, name), stream_);
    for (const auto& name : decoder_->names()) {
        if (!decoder_->is_input(name) || name == "query_bbox" || name == "query_feat" || name == "d_region") continue;
        if (radar_->has(name) && !radar_->is_input(name)) {
            if (radar_->bytes(name) != decoder_->bytes(name) || radar_->dtype(name) != decoder_->dtype(name))
                throw std::runtime_error("radar/decoder boundary mismatch: " + name);
            decoder_->bind(name, radar_->pointer(name));
        } else if (image_->has(name) && !image_->is_input(name)) {
            if (image_->bytes(name) != decoder_->bytes(name) || image_->dtype(name) != decoder_->dtype(name))
                throw std::runtime_error("image/decoder boundary mismatch: " + name);
            decoder_->bind(name, image_->pointer(name));
        }
        else decoder_->upload(name, choose(dynamic, constants_, name), stream_);
    }
    if (decoder_->dtype("cls_score") != DataType::kFloat32 ||
        decoder_->dtype("next_query_bbox") != DataType::kFloat32)
        throw std::runtime_error("v1 C++ postprocessor requires an FP32 decoder engine");
    void* final_bbox = nullptr;
    auto execute_pipeline = [&] {
        if (!image_->enqueue(stream_) || !radar_->enqueue(stream_))
            throw std::runtime_error("frontend enqueue failed");
        decoder_->upload("query_bbox", choose(dynamic, constants_, "query_bbox"), stream_);
        decoder_->upload("query_feat", choose(dynamic, constants_, "query_feat"), stream_);
        void* bbox_input = decoder_->pointer("query_bbox");
        void* feature_input = decoder_->pointer("query_feat");
        for (int iteration = 0; iteration < 6; ++iteration) {
            Tensor region = make_float_tensor({1});
            std::memcpy(region.bytes.data(), &kRegions[iteration], sizeof(float));
            decoder_->upload("d_region", region, stream_);
            decoder_->bind("query_bbox", bbox_input);
            decoder_->bind("query_feat", feature_input);
            decoder_->bind("next_query_bbox", bbox_ping_[iteration % 2]);
            decoder_->bind("next_query_feat", feature_ping_[iteration % 2]);
            if (!decoder_->enqueue(stream_))
                throw std::runtime_error("decoder enqueue failed at iteration " + std::to_string(iteration));
            bbox_input = bbox_ping_[iteration % 2];
            feature_input = feature_ping_[iteration % 2];
        }
        final_bbox = bbox_input;
    };
    if (!warmed_) {
        for (uint32_t i = 0; i < config_.warmup_runs; ++i) execute_pipeline();
        cuda_check(cudaStreamSynchronize(stream_), "warmup synchronization");
        warmed_ = true;
    }
    cudaEvent_t start{}, end{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate end");
    cuda_check(cudaEventRecord(start, stream_), "cudaEventRecord start");
    execute_pipeline();
    cuda_check(cudaEventRecord(end, stream_), "cudaEventRecord end");
    std::vector<float> cls(decoder_->bytes("cls_score") / sizeof(float));
    std::vector<float> raw_bbox(decoder_->bytes("next_query_bbox") / sizeof(float));
    decoder_->download("cls_score", cls.data(), cls.size() * sizeof(float), stream_);
    cuda_check(cudaMemcpyAsync(raw_bbox.data(), final_bbox, raw_bbox.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "bbox D2H");
    cuda_check(cudaStreamSynchronize(stream_), "inference synchronization");
    float inference_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&inference_ms, start, end), "cudaEventElapsedTime");
    cudaEventDestroy(start); cudaEventDestroy(end);

    struct Candidate { float score; int query; int label; };
    std::vector<Candidate> candidates;
    candidates.reserve(cls.size());
    for (size_t i = 0; i < cls.size(); ++i)
        candidates.push_back({1.0F / (1.0F + std::exp(-cls[i])), static_cast<int>(i / 10), static_cast<int>(i % 10)});
    const size_t top = std::min<size_t>(300, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + top, candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<racformer_box3d_t> boxes;
    std::vector<float> scores;
    std::vector<int32_t> labels;
    const float polar_radius = [&] {
        const auto it = constants_.find("decoder_polar_radius");
        return it == constants_.end() ? 65.0F : *reinterpret_cast<const float*>(it->second.bytes.data());
    }();
    for (size_t i = 0; i < top; ++i) {
        const Candidate& candidate = candidates[i];
        if (candidate.score <= 0.05F) continue;
        const float* b = raw_bbox.data() + candidate.query * 10;
        const float theta = b[0] * kTwoPi;
        const float x = std::clamp(b[1] * polar_radius * std::cos(theta), 0.0F, 100.0F);
        const float y = std::clamp(b[1] * polar_radius * std::sin(theta), -20.0F, 20.0F);
        const float z_center = b[2] * 6.0F - 3.0F;
        const float dz = std::exp(b[5]);
        if (z_center < -3.0F || z_center > 3.0F) continue;
        boxes.push_back({x, y, z_center - dz * 0.5F, std::exp(b[3]), std::exp(b[4]), dz,
                         std::atan2(b[6], b[7]), b[8], b[9]});
        scores.push_back(candidate.score);
        labels.push_back(candidate.label);
    }
    if (callback_) {
        racformer_result_t result{frames[0]->camera.timestamp, frames[0]->radar.timestamp,
            frames[0]->camera.frame_id, frames[0]->camera.version,
            static_cast<uint32_t>(boxes.size()), boxes.data(), scores.data(), labels.data(), inference_ms};
        callback_(&result, user_data_);
    }
}
}  // namespace racformer
