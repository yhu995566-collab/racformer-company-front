#ifndef RACFORMER_C_API_H_
#define RACFORMER_C_API_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN32)
#define RACFORMER_API __declspec(dllexport)
#else
#define RACFORMER_API __attribute__((visibility("default")))
#endif

typedef struct {
    float x, y, z;
    float v, mag, rcs, snr;
    float vx, vy;
    float ego_speed;
    int32_t label;
} radar_raw_data_t;

typedef struct {
    uint64_t timestamp; /* nanoseconds */
    uint32_t frame_id;
    uint32_t version;
    uint32_t radar_data_count;
    radar_raw_data_t* p_radar_data;
} radar_data_t;

typedef struct {
    uint64_t timestamp; /* nanoseconds */
    uint32_t frame_id;
    uint32_t version;
    uint32_t data_size; /* complete compressed JPEG size in bytes */
    void* p_camera_data;
} camera_data_t;

typedef struct {
    float x, y, z;       /* z is bottom-center */
    float dx, dy, dz;
    float yaw;
    float vx, vy;
} racformer_box3d_t;

typedef struct {
    uint64_t camera_timestamp;
    uint64_t radar_timestamp;
    uint32_t frame_id;
    uint32_t version;
    uint32_t detection_count;
    const racformer_box3d_t* boxes_3d;
    const float* scores_3d;
    const int32_t* labels_3d;
    float inference_ms;    /* TensorRT engines only, retained for ABI compatibility. */
    float preprocessing_ms;
    float postprocessing_ms;
    float end_to_end_ms;
} racformer_result_t;

typedef void (*racformer_result_callback_t)(
    const racformer_result_t* result, void* user_data);

typedef struct {
    const char* image_engine_path;
    const char* radar_engine_path;
    const char* decoder_engine_path;
    const char* plugin_path;
    const char* constants_manifest_path;

    /* Row-major raw-radar -> model-ego rigid transform. */
    float radar_to_ego[16];

    uint64_t max_pair_delta_ns;
    uint32_t max_pending_frames;
    uint32_t warmup_runs;
    uint8_t pad_startup_frames;
} racformer_config_t;

typedef struct racformer_handle racformer_handle_t;

/* Input buffers only need to remain valid until the push call returns. */
RACFORMER_API racformer_handle_t* racformer_create(
    const racformer_config_t* config,
    racformer_result_callback_t callback,
    void* user_data);
RACFORMER_API int racformer_push_camera(
    racformer_handle_t* handle, const camera_data_t* camera);
RACFORMER_API int racformer_push_radar(
    racformer_handle_t* handle, const radar_data_t* radar);
RACFORMER_API void racformer_reset(racformer_handle_t* handle);
RACFORMER_API const char* racformer_last_error(racformer_handle_t* handle);
RACFORMER_API void racformer_destroy(racformer_handle_t* handle);

#if defined(__cplusplus)
}
#endif
#endif
