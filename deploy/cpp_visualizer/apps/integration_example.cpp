#include "racformer/c_api.h"
#include "racformer/visualizer_c_api.h"

#include <cstring>
#include <vector>

struct Application {
    racformer_handle_t* inference{};
    racformer_vis_handle_t* visualizer{};
};

/* Replace this body with the software team's display/transport function. The
   JPEG pointer is callback-scoped, so a downstream asynchronous queue must
   copy output->jpeg_data before returning. */
void on_visualization(const racformer_vis_output_t* output, void* user_data) {
    (void)output;
    (void)user_data;
}

void on_inference(const racformer_result_t* result, void* user_data) {
    auto* application = static_cast<Application*>(user_data);
    std::vector<racformer_vis_box3d_t> boxes(result->detection_count);
    for (uint32_t index = 0; index < result->detection_count; ++index) {
        const auto& source = result->boxes_3d[index];
        boxes[index] = {
            source.x, source.y, source.z,
            source.dx, source.dy, source.dz, source.yaw,
            result->scores_3d[index], result->labels_3d[index],
        };
    }
    const racformer_vis_predictions_t predictions{
        result->camera_timestamp, result->frame_id, result->version,
        static_cast<uint32_t>(boxes.size()), boxes.data(),
    };
    racformer_vis_push_predictions(application->visualizer, &predictions);
}

void on_camera(Application* application, const camera_data_t* camera) {
    const racformer_vis_camera_t visualization{
        camera->timestamp, camera->frame_id, camera->version,
        camera->data_size, camera->p_camera_data,
    };
    /* Both calls deep-copy before returning. */
    racformer_vis_push_camera(application->visualizer, &visualization);
    racformer_push_camera(application->inference, camera);
}

void on_radar(Application* application, const radar_data_t* radar) {
    std::vector<racformer_vis_radar_point_t> points(radar->radar_data_count);
    for (uint32_t index = 0; index < radar->radar_data_count; ++index) {
        const auto& source = radar->p_radar_data[index];
        points[index] = {source.x, source.y, source.z};
    }
    const racformer_vis_radar_t visualization{
        radar->timestamp, radar->frame_id, radar->version,
        static_cast<uint32_t>(points.size()), points.data(),
    };
    racformer_vis_push_radar(application->visualizer, &visualization);
    racformer_push_radar(application->inference, radar);
}

/* Initialization sketch. model_projection points to the first 16 float32
   values from constants/lidar2img.bin. */
racformer_vis_handle_t* create_visualizer(
        const float* radar_to_ego, const float* model_projection,
        Application* application) {
    racformer_vis_config_t config{};
    std::memcpy(config.radar_to_ego, radar_to_ego, 16 * sizeof(float));
    std::memcpy(config.ego_to_image, model_projection, 16 * sizeof(float));
    config.projection_crop_y = 224.0F;
    config.forward_range_m = 50.0F;
    config.lateral_range_m = 20.0F;
    config.score_threshold = 0.3F;
    config.class_nms_iou_threshold = 0.2F;
    config.max_pending_frames = 16;
    config.jpeg_quality = 90;
    config.radar_point_radius = 2;
    config.radar_point_alpha = 0.45F;
    config.draw_radar = 1;
    config.draw_labels = 1;
    return racformer_vis_create(&config, on_visualization, application);
}
