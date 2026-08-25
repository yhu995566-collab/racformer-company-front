#pragma once

#include "racformer/visualizer_c_api.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace racformer_vis {

struct Config {
    std::array<float, 16> radar_to_ego{};
    std::array<float, 16> ego_to_image{};
    float forward_range{50.0F};
    float lateral_range{20.0F};
    float score_threshold{0.3F};
    float nms_iou_threshold{0.2F};
    uint32_t max_pending{16};
    uint32_t jpeg_quality{90};
    uint32_t radar_point_radius{2};
    float radar_point_alpha{0.45F};
    bool draw_radar{true};
    bool draw_labels{true};
};

struct Key {
    uint32_t version{};
    uint32_t frame_id{};
    bool operator<(const Key& other) const {
        return version < other.version ||
            (version == other.version && frame_id < other.frame_id);
    }
};
struct Camera {
    uint64_t timestamp{};
    std::vector<uint8_t> jpeg;
};
struct Radar {
    uint64_t timestamp{};
    std::vector<racformer_vis_radar_point_t> points;
};
struct Predictions {
    uint64_t timestamp{};
    std::vector<racformer_vis_box3d_t> boxes;
};
struct Frame {
    Key key;
    Camera camera;
    Radar radar;
    Predictions predictions;
};

class Visualizer {
 public:
    Visualizer(Config config, racformer_vis_output_callback_t callback,
               void* user_data);
    ~Visualizer();
    void push_camera(const racformer_vis_camera_t& value);
    void push_radar(const racformer_vis_radar_t& value);
    void push_predictions(const racformer_vis_predictions_t& value);
    void reset();
    std::string last_error() const;

 private:
    void try_queue_locked(const Key& key);
    void prune_locked();
    void worker();
    std::vector<uint8_t> render(const Frame& frame, uint32_t* width,
                                uint32_t* height) const;

    Config config_;
    racformer_vis_output_callback_t callback_{};
    void* user_data_{};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_{};
    std::map<Key, Camera> cameras_;
    std::map<Key, Radar> radars_;
    std::map<Key, Predictions> predictions_;
    std::deque<Frame> queue_;
    std::thread worker_;
    std::string error_;
};

}  // namespace racformer_vis
