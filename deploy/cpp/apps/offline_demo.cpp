#include "racformer/c_api.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
struct CallbackState {
    std::mutex mutex;
    std::condition_variable condition;
    uint32_t completed{};
};

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    std::vector<uint8_t> result(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), result.size());
    return result;
}

std::string indexed_name(uint32_t index, const char* extension) {
    std::ostringstream result;
    result << std::setfill('0') << std::setw(7) << index << extension;
    return result.str();
}

std::vector<radar_raw_data_t> read_radar_ply(
        const std::filesystem::path& path, uint32_t* radar_frame_id) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    std::string line;
    size_t vertex_count = 0;
    bool ascii = false;
    bool header_complete = false;
    const std::vector<std::string> expected_properties = {
        "x", "y", "z", "v", "mag", "rcs", "snr", "Vx", "Vy",
        "ego_speed", "label"};
    std::vector<std::string> properties;
    while (std::getline(file, line)) {
        if (line == "format ascii 1.0") ascii = true;
        if (line.rfind("comment radar_frame_id ", 0) == 0)
            *radar_frame_id = static_cast<uint32_t>(
                std::stoul(line.substr(std::strlen("comment radar_frame_id "))));
        if (line.rfind("element vertex ", 0) == 0)
            vertex_count = std::stoull(line.substr(std::strlen("element vertex ")));
        if (line.rfind("property ", 0) == 0) {
            std::istringstream property(line);
            std::string keyword, type, name;
            property >> keyword >> type >> name;
            properties.push_back(name);
        }
        if (line == "end_header") { header_complete = true; break; }
    }
    if (!ascii || !header_complete || properties != expected_properties)
        throw std::runtime_error("unsupported radar PLY header: " + path.string());
    std::vector<radar_raw_data_t> points(vertex_count);
    for (size_t index = 0; index < vertex_count; ++index) {
        auto& point = points[index];
        if (!(file >> point.x >> point.y >> point.z >> point.v >> point.mag >>
              point.rcs >> point.snr >> point.vx >> point.vy >>
              point.ego_speed >> point.label))
            throw std::runtime_error("invalid radar vertex " +
                                     std::to_string(index) + " in " + path.string());
    }
    return points;
}

void on_result(const racformer_result_t* result, void* opaque) {
    auto* state = static_cast<CallbackState*>(opaque);
    std::cout << "frame=" << result->frame_id << " detections="
              << result->detection_count << " engine_ms="
              << result->inference_ms << " preprocess_ms="
              << result->preprocessing_ms << " postprocess_ms="
              << result->postprocessing_ms << " end_to_end_ms="
              << result->end_to_end_ms << '\n';
    for (uint32_t i = 0; i < result->detection_count; ++i) {
        const auto& box = result->boxes_3d[i];
        std::cout << i << " label=" << result->labels_3d[i]
                  << " score=" << result->scores_3d[i]
                  << " xyz=" << box.x << ',' << box.y << ',' << box.z << '\n';
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ++state->completed;
    }
    state->condition.notify_all();
}

void usage(const char* program) {
    std::cerr
        << "single-image smoke test:\n  " << program
        << " IMAGE.engine RADAR.engine DECODER.engine PLUGIN.so manifest.tsv frame.jpg\n"
        << "recorded sequence:\n  " << program
        << " IMAGE.engine RADAR.engine DECODER.engine PLUGIN.so manifest.tsv"
           " --sequence REPLAY_ROOT START_INDEX COUNT FRAME_PERIOD_NS\n";
}
}  // namespace

int main(int argc, char** argv) {
    const bool sequence_mode = argc == 11 && std::string(argv[6]) == "--sequence";
    if (argc != 7 && !sequence_mode) { usage(argv[0]); return 2; }
    try {
        racformer_config_t config{};
        config.image_engine_path = argv[1]; config.radar_engine_path = argv[2];
        config.decoder_engine_path = argv[3]; config.plugin_path = argv[4];
        config.constants_manifest_path = argv[5];
        const float radar_to_ego[16] = {
            0.031994525F, 0.99938482F, -0.014363338F, 0.453F,
           -0.99948764F, 0.031978268F, -0.001360151F, -0.0501F,
            0.0009F, -0.014399497F, -0.99989593F, -0.6756F,
            0, 0, 0, 1};
        std::memcpy(config.radar_to_ego, radar_to_ego, sizeof(radar_to_ego));
        config.max_pair_delta_ns = 50'000'000ULL;
        config.max_pending_frames = 16; config.pad_startup_frames = 1;
        CallbackState state;
        racformer_handle_t* runtime = racformer_create(&config, on_result, &state);
        if (!runtime) throw std::runtime_error("runtime creation failed");

        uint32_t expected_results = 1;
        if (!sequence_mode) {
            const auto jpeg = read_file(argv[6]);
            camera_data_t camera{1'000'000'000ULL, 1, 1,
                static_cast<uint32_t>(jpeg.size()), const_cast<uint8_t*>(jpeg.data())};
            radar_data_t radar{1'000'000'000ULL, 1, 1, 0, nullptr};
            if (racformer_push_camera(runtime, &camera) ||
                racformer_push_radar(runtime, &radar))
                throw std::runtime_error(racformer_last_error(runtime));
        } else {
            const std::filesystem::path root(argv[7]);
            const uint32_t start = static_cast<uint32_t>(std::stoul(argv[8]));
            const uint32_t count = static_cast<uint32_t>(std::stoul(argv[9]));
            const uint64_t period_ns = std::stoull(argv[10]);
            if (!count || !period_ns) throw std::runtime_error("COUNT and FRAME_PERIOD_NS must be positive");
            expected_results = count;
            for (uint32_t offset = 0; offset < count; ++offset) {
                const uint32_t index = start + offset;
                const auto jpeg = read_file(
                    root / "images" / "cam_1" / indexed_name(index, ".jpg"));
                uint32_t source_radar_frame = 0;
                auto points = read_radar_ply(
                    root / "radar_ply" / indexed_name(index, ".ply"),
                    &source_radar_frame);
                const uint64_t timestamp = 1'000'000'000ULL +
                    static_cast<uint64_t>(offset) * period_ns;
                camera_data_t camera{timestamp, index, 1,
                    static_cast<uint32_t>(jpeg.size()), const_cast<uint8_t*>(jpeg.data())};
                radar_data_t radar{timestamp, index, 1,
                    static_cast<uint32_t>(points.size()), points.data()};
                std::cout << "submit export_index=" << index
                          << " radar_frame_id=" << source_radar_frame
                          << " radar_points=" << points.size()
                          << " timestamp_ns=" << timestamp << '\n';
                if (racformer_push_camera(runtime, &camera) ||
                    racformer_push_radar(runtime, &radar))
                    throw std::runtime_error(racformer_last_error(runtime));
            }
        }
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.condition.wait_for(lock, std::chrono::minutes(10), [&] {
                return state.completed >= expected_results;
            });
        }
        const char* error = racformer_last_error(runtime);
        if (error && *error) std::cerr << "runtime: " << error << '\n';
        const bool complete = state.completed >= expected_results;
        racformer_destroy(runtime);
        if (!complete) throw std::runtime_error("timed out waiting for inference callbacks");
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
