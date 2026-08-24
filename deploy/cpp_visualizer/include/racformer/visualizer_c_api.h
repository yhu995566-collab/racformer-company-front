#ifndef RACFORMER_VISUALIZER_C_API_H_
#define RACFORMER_VISUALIZER_C_API_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN32)
#define RACFORMER_VIS_API __declspec(dllexport)
#else
#define RACFORMER_VIS_API __attribute__((visibility("default")))
#endif

typedef struct {
    float x;
    float y;
    float z;
} racformer_vis_radar_point_t;

/* Prediction box in current ego coordinates. z is the bottom-center height. */
typedef struct {
    float x, y, z;
    float dx, dy, dz;
    float yaw;
    float score;
    int32_t label;
} racformer_vis_box3d_t;

typedef struct {
    uint64_t timestamp; /* nanoseconds */
    uint32_t frame_id;
    uint32_t version;
    uint32_t data_size;
    const void* jpeg_data; /* complete compressed 640x480 JPEG */
} racformer_vis_camera_t;

typedef struct {
    uint64_t timestamp; /* nanoseconds */
    uint32_t frame_id;
    uint32_t version;
    uint32_t point_count;
    const racformer_vis_radar_point_t* points; /* raw radar coordinates */
} racformer_vis_radar_t;

typedef struct {
    uint64_t timestamp; /* normally the inference result timestamp */
    uint32_t frame_id;
    uint32_t version;
    uint32_t box_count;
    const racformer_vis_box3d_t* boxes;
} racformer_vis_predictions_t;

typedef struct {
    uint64_t timestamp;
    uint32_t frame_id;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t data_size;
    const void* jpeg_data;
    float render_ms;
} racformer_vis_output_t;

typedef void (*racformer_vis_output_callback_t)(
    const racformer_vis_output_t* output, void* user_data);

typedef struct {
    /* Row-major raw-radar -> current-ego transform. */
    float radar_to_ego[16];
    /* Row-major current-ego -> image homogeneous projection. */
    float ego_to_image[16];

    /* If ego_to_image targets a vertically cropped model image, restore its
       original-image coordinates by adding crop_y * projection_row_2 to
       projection_row_1. Use 224 for the current 640x256 model projection and
       0 when an original 640x480 projection is supplied. */
    float projection_crop_y;

    float forward_range_m;       /* default 50 */
    float lateral_range_m;       /* default 20 */
    float score_threshold;       /* default 0.3 */
    float class_nms_iou_threshold; /* default 0.2; <=0 disables */
    uint32_t max_pending_frames; /* default 16 */
    uint32_t jpeg_quality;       /* default 90 */
    uint8_t draw_radar;
    uint8_t draw_labels;
} racformer_vis_config_t;

typedef struct racformer_vis_handle racformer_vis_handle_t;

RACFORMER_VIS_API racformer_vis_handle_t* racformer_vis_create(
    const racformer_vis_config_t* config,
    racformer_vis_output_callback_t callback,
    void* user_data);
RACFORMER_VIS_API int racformer_vis_push_camera(
    racformer_vis_handle_t* handle, const racformer_vis_camera_t* camera);
RACFORMER_VIS_API int racformer_vis_push_radar(
    racformer_vis_handle_t* handle, const racformer_vis_radar_t* radar);
RACFORMER_VIS_API int racformer_vis_push_predictions(
    racformer_vis_handle_t* handle,
    const racformer_vis_predictions_t* predictions);
RACFORMER_VIS_API void racformer_vis_reset(racformer_vis_handle_t* handle);
RACFORMER_VIS_API const char* racformer_vis_last_error(
    racformer_vis_handle_t* handle);
RACFORMER_VIS_API void racformer_vis_destroy(racformer_vis_handle_t* handle);

#if defined(__cplusplus)
}
#endif
#endif
