#include "racformer/c_api.h"
#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

struct racformer_handle {
    std::unique_ptr<racformer::Runtime> runtime;
    std::string error;
};

namespace {
racformer::RuntimeConfig convert(const racformer_config_t& source) {
    if (!source.image_engine_path || !source.radar_engine_path ||
        !source.decoder_engine_path || !source.plugin_path ||
        !source.constants_manifest_path) throw std::invalid_argument("all runtime paths are required");
    racformer::RuntimeConfig result;
    result.image_engine = source.image_engine_path;
    result.radar_engine = source.radar_engine_path;
    result.decoder_engine = source.decoder_engine_path;
    result.plugin = source.plugin_path;
    result.manifest = source.constants_manifest_path;
    std::copy(source.radar_to_ego, source.radar_to_ego + 16, result.radar_to_ego.begin());
    result.max_pair_delta_ns = source.max_pair_delta_ns;
    result.max_pending_frames = source.max_pending_frames ? source.max_pending_frames : 16;
    result.warmup_runs = source.warmup_runs;
    result.pad_startup_frames = source.pad_startup_frames != 0;
    return result;
}
template <typename Function> int protect(racformer_handle_t* handle, Function function) {
    if (!handle || !handle->runtime) return -1;
    try { function(); return 0; }
    catch (const std::exception& error) { handle->error = error.what(); return -2; }
}
}  // namespace

extern "C" racformer_handle_t* racformer_create(
        const racformer_config_t* config, racformer_result_callback_t callback,
        void* user_data) {
    if (!config) return nullptr;
    auto handle = std::make_unique<racformer_handle>();
    try { handle->runtime = std::make_unique<racformer::Runtime>(convert(*config), callback, user_data); }
    catch (...) { return nullptr; }
    return handle.release();
}
extern "C" int racformer_push_camera(racformer_handle_t* handle, const camera_data_t* camera) {
    if (!camera) return -1;
    return protect(handle, [&] { handle->runtime->push_camera(*camera); });
}
extern "C" int racformer_push_radar(racformer_handle_t* handle, const radar_data_t* radar) {
    if (!radar) return -1;
    return protect(handle, [&] { handle->runtime->push_radar(*radar); });
}
extern "C" void racformer_reset(racformer_handle_t* handle) {
    if (handle && handle->runtime) handle->runtime->reset();
}
extern "C" const char* racformer_last_error(racformer_handle_t* handle) {
    if (!handle) return "invalid handle";
    const std::string worker = handle->runtime ? handle->runtime->last_error() : std::string{};
    if (!worker.empty()) handle->error = worker;
    return handle->error.c_str();
}
extern "C" void racformer_destroy(racformer_handle_t* handle) { delete handle; }
