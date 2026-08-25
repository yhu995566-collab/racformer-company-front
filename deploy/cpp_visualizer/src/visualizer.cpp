#include "visualizer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace racformer_vis {
namespace {
constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;
constexpr std::array<std::array<int, 2>, 12> kBoxEdges{{
    {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
    {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
    {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
}};
constexpr const char* kClassNames[] = {
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
};

struct EgoPoint { float x, y, z; };
struct Projected { cv::Point2f pixel; float depth; bool valid; };

EgoPoint transform(const std::array<float, 16>& matrix,
                   const racformer_vis_radar_point_t& point) {
    return {
        matrix[0] * point.x + matrix[1] * point.y +
            matrix[2] * point.z + matrix[3],
        matrix[4] * point.x + matrix[5] * point.y +
            matrix[6] * point.z + matrix[7],
        matrix[8] * point.x + matrix[9] * point.y +
            matrix[10] * point.z + matrix[11],
    };
}

Projected project(const std::array<float, 16>& matrix,
                  const EgoPoint& point) {
    const float x = matrix[0] * point.x + matrix[1] * point.y +
        matrix[2] * point.z + matrix[3];
    const float y = matrix[4] * point.x + matrix[5] * point.y +
        matrix[6] * point.z + matrix[7];
    const float depth = matrix[8] * point.x + matrix[9] * point.y +
        matrix[10] * point.z + matrix[11];
    if (depth <= 1e-4F) return {{}, depth, false};
    const cv::Point2f pixel{x / depth, y / depth};
    const bool valid = pixel.x >= 0.0F && pixel.x < kImageWidth &&
        pixel.y >= 0.0F && pixel.y < kImageHeight;
    return {pixel, depth, valid};
}

std::array<EgoPoint, 8> box_corners(const racformer_vis_box3d_t& box) {
    const float hx = box.dx * 0.5F;
    const float hy = box.dy * 0.5F;
    const float bottom = box.z;
    const float top = box.z + box.dz;
    const float cosine = std::cos(box.yaw);
    const float sine = std::sin(box.yaw);
    const std::array<std::array<float, 2>, 4> local{{
        {{-hx, -hy}}, {{-hx, hy}}, {{hx, hy}}, {{hx, -hy}},
    }};
    std::array<EgoPoint, 8> result{};
    for (int index = 0; index < 4; ++index) {
        const float x = cosine * local[index][0] - sine * local[index][1] + box.x;
        const float y = sine * local[index][0] + cosine * local[index][1] + box.y;
        result[index] = {x, y, bottom};
        result[index + 4] = {x, y, top};
    }
    return result;
}

std::vector<cv::Point2f> bev_polygon(const racformer_vis_box3d_t& box) {
    const auto corners = box_corners(box);
    std::vector<cv::Point2f> result;
    result.reserve(4);
    for (int index = 0; index < 4; ++index)
        result.emplace_back(corners[index].x, corners[index].y);
    return result;
}

float rotated_iou(const racformer_vis_box3d_t& first,
                  const racformer_vis_box3d_t& second) {
    const auto first_polygon = bev_polygon(first);
    const auto second_polygon = bev_polygon(second);
    std::vector<cv::Point2f> intersection;
    const float intersection_area = cv::intersectConvexConvex(
        first_polygon, second_polygon, intersection, true);
    const float first_area = std::max(0.0F, first.dx * first.dy);
    const float second_area = std::max(0.0F, second.dx * second.dy);
    const float union_area = first_area + second_area - intersection_area;
    return union_area > 1e-6F ? intersection_area / union_area : 0.0F;
}

std::vector<racformer_vis_box3d_t> select_boxes(
        const std::vector<racformer_vis_box3d_t>& boxes,
        const Config& config) {
    std::vector<racformer_vis_box3d_t> candidates;
    for (const auto& box : boxes) {
        if (box.score >= config.score_threshold && box.x >= 0.0F &&
            box.x <= config.forward_range &&
            std::abs(box.y) <= config.lateral_range && box.dx > 0.0F &&
            box.dy > 0.0F && box.dz > 0.0F)
            candidates.push_back(box);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const auto& first, const auto& second) {
            return first.score > second.score;
        });
    if (config.nms_iou_threshold <= 0.0F) return candidates;
    std::vector<racformer_vis_box3d_t> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& previous : kept) {
            if (candidate.label == previous.label &&
                rotated_iou(candidate, previous) > config.nms_iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}

cv::Scalar distance_color(float forward, float maximum) {
    const float value = std::clamp(forward / std::max(maximum, 1e-6F),
                                   0.0F, 1.0F);
    if (value <= 0.5F) {
        const float t = value * 2.0F;
        return {0, std::round(255.0F * t), 255};
    }
    const float t = (value - 0.5F) * 2.0F;
    return {std::round(255.0F * t), std::round(255.0F * (1.0F - t)),
            std::round(255.0F * (1.0F - t))};
}

std::string box_label(const racformer_vis_box3d_t& box) {
    const std::string name = box.label >= 0 && box.label < 10
        ? kClassNames[box.label] : std::to_string(box.label);
    char score[16];
    std::snprintf(score, sizeof(score), " %.2f", box.score);
    return name + score;
}
}  // namespace

Visualizer::Visualizer(Config config, racformer_vis_output_callback_t callback,
                       void* user_data)
    : config_(std::move(config)), callback_(callback), user_data_(user_data),
      worker_(&Visualizer::worker, this) {}

Visualizer::~Visualizer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void Visualizer::push_camera(const racformer_vis_camera_t& value) {
    if (!value.jpeg_data || !value.data_size)
        throw std::invalid_argument("camera JPEG pointer/size is invalid");
    Camera copy{value.timestamp, {}};
    const auto* begin = static_cast<const uint8_t*>(value.jpeg_data);
    copy.jpeg.assign(begin, begin + value.data_size);
    std::lock_guard<std::mutex> lock(mutex_);
    const Key key{value.version, value.frame_id};
    cameras_[key] = std::move(copy);
    try_queue_locked(key);
}

void Visualizer::push_radar(const racformer_vis_radar_t& value) {
    if (value.point_count && !value.points)
        throw std::invalid_argument("radar point pointer is null");
    Radar copy{value.timestamp, {}};
    if (value.point_count)
        copy.points.assign(value.points, value.points + value.point_count);
    std::lock_guard<std::mutex> lock(mutex_);
    const Key key{value.version, value.frame_id};
    radars_[key] = std::move(copy);
    try_queue_locked(key);
}

void Visualizer::push_predictions(const racformer_vis_predictions_t& value) {
    if (value.box_count && !value.boxes)
        throw std::invalid_argument("prediction box pointer is null");
    Predictions copy{value.timestamp, {}};
    if (value.box_count)
        copy.boxes.assign(value.boxes, value.boxes + value.box_count);
    std::lock_guard<std::mutex> lock(mutex_);
    const Key key{value.version, value.frame_id};
    predictions_[key] = std::move(copy);
    try_queue_locked(key);
}

void Visualizer::try_queue_locked(const Key& key) {
    auto camera = cameras_.find(key);
    auto radar = radars_.find(key);
    auto prediction = predictions_.find(key);
    if (camera == cameras_.end() || radar == radars_.end() ||
        prediction == predictions_.end()) {
        prune_locked();
        return;
    }
    if (queue_.size() >= config_.max_pending) queue_.pop_front();
    queue_.push_back({key, std::move(camera->second), std::move(radar->second),
                      std::move(prediction->second)});
    cameras_.erase(camera);
    radars_.erase(radar);
    predictions_.erase(prediction);
    prune_locked();
    condition_.notify_one();
}

void Visualizer::prune_locked() {
    while (cameras_.size() > config_.max_pending) cameras_.erase(cameras_.begin());
    while (radars_.size() > config_.max_pending) radars_.erase(radars_.begin());
    while (predictions_.size() > config_.max_pending)
        predictions_.erase(predictions_.begin());
}

void Visualizer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    cameras_.clear();
    radars_.clear();
    predictions_.clear();
    queue_.clear();
    error_.clear();
}

std::string Visualizer::last_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void Visualizer::worker() {
    for (;;) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_) return;
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            const auto start = std::chrono::steady_clock::now();
            uint32_t width = 0, height = 0;
            const auto jpeg = render(frame, &width, &height);
            const auto end = std::chrono::steady_clock::now();
            const float render_ms = std::chrono::duration<float, std::milli>(
                end - start).count();
            if (callback_) {
                const racformer_vis_output_t output{
                    frame.camera.timestamp, frame.key.frame_id,
                    frame.key.version, width, height,
                    static_cast<uint32_t>(jpeg.size()), jpeg.data(), render_ms};
                callback_(&output, user_data_);
            }
        } catch (const std::exception& exception) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = exception.what();
        }
    }
}

std::vector<uint8_t> Visualizer::render(const Frame& frame, uint32_t* width,
                                        uint32_t* height) const {
    cv::Mat encoded(1, static_cast<int>(frame.camera.jpeg.size()), CV_8UC1,
                    const_cast<uint8_t*>(frame.camera.jpeg.data()));
    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("failed to decode camera JPEG");
    if (image.cols != kImageWidth || image.rows != kImageHeight)
        throw std::runtime_error("camera JPEG must decode to 640x480");

    if (config_.draw_radar) {
        cv::Mat radar_overlay = image.clone();
        for (const auto& raw : frame.radar.points) {
            const EgoPoint ego = transform(config_.radar_to_ego, raw);
            if (ego.x < 0.0F || ego.x > config_.forward_range ||
                std::abs(ego.y) > config_.lateral_range) continue;
            const Projected projected = project(config_.ego_to_image, ego);
            if (projected.valid)
                cv::circle(radar_overlay, projected.pixel,
                           static_cast<int>(config_.radar_point_radius),
                           distance_color(ego.x, config_.forward_range),
                           cv::FILLED, cv::LINE_AA);
        }
        cv::addWeighted(radar_overlay, config_.radar_point_alpha, image,
                        1.0F - config_.radar_point_alpha, 0.0, image);
    }

    const auto boxes = select_boxes(frame.predictions.boxes, config_);
    for (const auto& box : boxes) {
        const auto corners = box_corners(box);
        std::array<Projected, 8> projected{};
        for (int index = 0; index < 8; ++index)
            projected[index] = project(config_.ego_to_image, corners[index]);
        for (const auto& edge : kBoxEdges) {
            const auto& first = projected[edge[0]];
            const auto& second = projected[edge[1]];
            if (first.depth > 1e-4F && second.depth > 1e-4F)
                cv::line(image, first.pixel, second.pixel, {0, 0, 255}, 2,
                         cv::LINE_AA);
        }
        if (config_.draw_labels) {
            const EgoPoint center{box.x, box.y, box.z + box.dz * 0.5F};
            const auto projected_center = project(config_.ego_to_image, center);
            if (projected_center.valid) {
                const std::string text = box_label(box);
                int baseline = 0;
                const cv::Size size = cv::getTextSize(
                    text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
                const cv::Point origin{
                    static_cast<int>(projected_center.pixel.x),
                    static_cast<int>(projected_center.pixel.y)};
                cv::rectangle(image, origin + cv::Point(0, -size.height - 3),
                              origin + cv::Point(size.width + 3, baseline),
                              {0, 0, 0}, cv::FILLED);
                cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
                            0.45, {0, 0, 255}, 1, cv::LINE_AA);
            }
        }
    }

    const std::string status = "frame " + std::to_string(frame.key.frame_id) +
        "  radar " + std::to_string(frame.radar.points.size()) +
        "  detections " + std::to_string(boxes.size()) + "  range 50m";
    cv::rectangle(image, {0, 0}, {image.cols, 25}, {0, 0, 0}, cv::FILLED);
    cv::putText(image, status, {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {255, 255, 255}, 1, cv::LINE_AA);

    std::vector<uint8_t> output;
    if (!cv::imencode(".jpg", image, output,
                      {cv::IMWRITE_JPEG_QUALITY,
                       static_cast<int>(config_.jpeg_quality)}))
        throw std::runtime_error("failed to encode visualization JPEG");
    *width = static_cast<uint32_t>(image.cols);
    *height = static_cast<uint32_t>(image.rows);
    return output;
}
}  // namespace racformer_vis
