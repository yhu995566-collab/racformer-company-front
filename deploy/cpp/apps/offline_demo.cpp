#include "racformer/c_api.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace {
std::vector<uint8_t> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<uint8_t> result(static_cast<size_t>(file.tellg()));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), result.size());
    return result;
}
void on_result(const racformer_result_t* result, void*) {
    std::cout << "frame=" << result->frame_id << " detections="
              << result->detection_count << '\n';
    for (uint32_t i = 0; i < result->detection_count; ++i) {
        const auto& box = result->boxes_3d[i];
        std::cout << i << " label=" << result->labels_3d[i]
                  << " score=" << result->scores_3d[i]
                  << " xyz=" << box.x << ',' << box.y << ',' << box.z << '\n';
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " IMAGE.engine RADAR.engine DECODER.engine PLUGIN.so manifest.tsv frame.jpg\n";
        return 2;
    }
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
    const auto jpeg = read_file(argv[6]);
    racformer_handle_t* runtime = racformer_create(&config, on_result, nullptr);
    if (!runtime) { std::cerr << "runtime creation failed\n"; return 1; }
    camera_data_t camera{1'000'000'000ULL, 1, 1,
        static_cast<uint32_t>(jpeg.size()), const_cast<uint8_t*>(jpeg.data())};
    radar_data_t radar{1'000'000'000ULL, 1, 1, 0, nullptr};
    if (racformer_push_camera(runtime, &camera) || racformer_push_radar(runtime, &radar)) {
        std::cerr << racformer_last_error(runtime) << '\n';
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    const char* error = racformer_last_error(runtime);
    if (error && *error) std::cerr << "runtime: " << error << '\n';
    racformer_destroy(runtime);
}
