#include "racformer/visualizer_c_api.h"
#include "visualizer.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

struct racformer_vis_handle {
    std::unique_ptr<racformer_vis::Visualizer> visualizer;
    std::string error;
};

namespace {
racformer_vis::Config convert(const racformer_vis_config_t& source) {
    racformer_vis::Config result;
    std::copy(source.radar_to_ego, source.radar_to_ego + 16,
              result.radar_to_ego.begin());
    std::copy(source.ego_to_image, source.ego_to_image + 16,
              result.ego_to_image.begin());
    if (source.projection_crop_y != 0.0F) {
        for (int column = 0; column < 4; ++column)
            result.ego_to_image[4 + column] +=
                source.projection_crop_y * result.ego_to_image[8 + column];
    }
    result.forward_range = source.forward_range_m > 0.0F
        ? source.forward_range_m : 50.0F;
    result.lateral_range = source.lateral_range_m > 0.0F
        ? source.lateral_range_m : 20.0F;
    result.score_threshold = source.score_threshold >= 0.0F
        ? source.score_threshold : 0.3F;
    result.nms_iou_threshold = source.class_nms_iou_threshold;
    result.max_pending = source.max_pending_frames
        ? source.max_pending_frames : 16;
    result.jpeg_quality = source.jpeg_quality
        ? std::min<uint32_t>(source.jpeg_quality, 100) : 90;
    result.radar_point_radius = source.radar_point_radius
        ? std::clamp<uint32_t>(source.radar_point_radius, 1, 10) : 2;
    result.radar_point_alpha = source.radar_point_alpha > 0.0F
        ? std::clamp(source.radar_point_alpha, 0.05F, 1.0F) : 0.45F;
    result.draw_radar = source.draw_radar != 0;
    result.draw_labels = source.draw_labels != 0;
    const bool radar_matrix_empty = std::all_of(
        result.radar_to_ego.begin(), result.radar_to_ego.end(),
        [](float value) { return value == 0.0F; });
    const bool projection_empty = std::all_of(
        result.ego_to_image.begin(), result.ego_to_image.end(),
        [](float value) { return value == 0.0F; });
    if (radar_matrix_empty || projection_empty)
        throw std::invalid_argument(
            "radar_to_ego and ego_to_image matrices are required");
    return result;
}

template <typename Function>
int protect(racformer_vis_handle_t* handle, Function function) {
    if (!handle || !handle->visualizer) return -1;
    try {
        function();
        return 0;
    } catch (const std::exception& error) {
        handle->error = error.what();
        return -2;
    }
}
}  // namespace

extern "C" racformer_vis_handle_t* racformer_vis_create(
        const racformer_vis_config_t* config,
        racformer_vis_output_callback_t callback, void* user_data) {
    if (!config) return nullptr;
    auto handle = std::make_unique<racformer_vis_handle>();
    try {
        handle->visualizer = std::make_unique<racformer_vis::Visualizer>(
            convert(*config), callback, user_data);
    } catch (...) {
        return nullptr;
    }
    return handle.release();
}

extern "C" int racformer_vis_push_camera(
        racformer_vis_handle_t* handle, const racformer_vis_camera_t* camera) {
    if (!camera) return -1;
    return protect(handle, [&] { handle->visualizer->push_camera(*camera); });
}

extern "C" int racformer_vis_push_radar(
        racformer_vis_handle_t* handle, const racformer_vis_radar_t* radar) {
    if (!radar) return -1;
    return protect(handle, [&] { handle->visualizer->push_radar(*radar); });
}

extern "C" int racformer_vis_push_predictions(
        racformer_vis_handle_t* handle,
        const racformer_vis_predictions_t* predictions) {
    if (!predictions) return -1;
    return protect(handle, [&] {
        handle->visualizer->push_predictions(*predictions);
    });
}

extern "C" void racformer_vis_reset(racformer_vis_handle_t* handle) {
    if (handle && handle->visualizer) handle->visualizer->reset();
}

extern "C" const char* racformer_vis_last_error(
        racformer_vis_handle_t* handle) {
    if (!handle) return "invalid handle";
    const std::string worker = handle->visualizer
        ? handle->visualizer->last_error() : std::string{};
    if (!worker.empty()) handle->error = worker;
    return handle->error.c_str();
}

extern "C" void racformer_vis_destroy(racformer_vis_handle_t* handle) {
    delete handle;
}
