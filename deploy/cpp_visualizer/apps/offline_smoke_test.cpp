#include "racformer/visualizer_c_api.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kRadarToEgo[16] = {
     0.031994525F,  0.999384820F, -0.014363338F,  0.4530F,
    -0.999487640F,  0.031978268F, -0.001360151F, -0.0501F,
     0.000900000F, -0.014399497F, -0.999895930F, -0.6756F,
     0.000000000F,  0.000000000F,  0.000000000F,  1.0000F,
};

struct OutputState {
    std::mutex mutex;
    std::condition_variable condition;
    std::string output_path;
    std::string error;
    uint32_t width{};
    uint32_t height{};
    uint32_t size{};
    float render_ms{};
    bool finished{};
};

std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open: " + path);
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("file is empty: " + path);
    std::vector<uint8_t> data(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("cannot read: " + path);
    return data;
}

void write_output(const racformer_vis_output_t* output, void* user_data) {
    auto* state = static_cast<OutputState*>(user_data);
    std::lock_guard<std::mutex> lock(state->mutex);
    try {
        std::ofstream file(state->output_path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot create output JPEG");
        file.write(static_cast<const char*>(output->jpeg_data),
                   output->data_size);
        if (!file) throw std::runtime_error("cannot write output JPEG");
        state->width = output->width;
        state->height = output->height;
        state->size = output->data_size;
        state->render_ms = output->render_ms;
    } catch (const std::exception& exception) {
        state->error = exception.what();
    }
    state->finished = true;
    state->condition.notify_one();
}

std::vector<racformer_vis_radar_point_t> read_ascii_ply(
        const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open radar PLY: " + path);

    std::string line;
    std::size_t vertex_count = 0;
    std::vector<std::string> properties;
    bool ascii = false;
    bool vertex_element = false;
    bool header_complete = false;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string first;
        fields >> first;
        if (first == "format") {
            std::string format;
            fields >> format;
            ascii = format == "ascii";
        } else if (first == "element") {
            std::string name;
            fields >> name;
            vertex_element = name == "vertex";
            if (vertex_element) fields >> vertex_count;
        } else if (first == "property" && vertex_element) {
            std::string type;
            std::string name;
            fields >> type >> name;
            properties.push_back(name);
        } else if (first == "end_header") {
            header_complete = true;
            break;
        }
    }
    if (!header_complete || !ascii)
        throw std::runtime_error("radar PLY must use ASCII format");

    int x_index = -1;
    int y_index = -1;
    int z_index = -1;
    for (std::size_t index = 0; index < properties.size(); ++index) {
        if (properties[index] == "x") x_index = static_cast<int>(index);
        if (properties[index] == "y") y_index = static_cast<int>(index);
        if (properties[index] == "z") z_index = static_cast<int>(index);
    }
    if (x_index < 0 || y_index < 0 || z_index < 0)
        throw std::runtime_error("radar PLY is missing x/y/z properties");

    std::vector<racformer_vis_radar_point_t> points;
    points.reserve(vertex_count);
    while (points.size() < vertex_count && std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::vector<float> values(properties.size());
        bool valid = true;
        for (float& value : values) {
            if (!(fields >> value)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            points.push_back({values[x_index], values[y_index],
                              values[z_index]});
        }
    }
    if (points.size() != vertex_count)
        throw std::runtime_error("radar PLY vertex count does not match data");
    return points;
}

std::vector<float> read_projection(const std::string& path) {
    const auto bytes = read_binary(path);
    if (bytes.size() < 16 * sizeof(float))
        throw std::runtime_error("lidar2img.bin contains fewer than 16 float32");
    std::vector<float> projection(16);
    std::memcpy(projection.data(), bytes.data(), 16 * sizeof(float));
    return projection;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " CAMERA.jpg RADAR.ply lidar2img.bin OUTPUT.jpg\n";
        return 2;
    }

    try {
        const auto jpeg = read_binary(argv[1]);
        const auto radar_points = read_ascii_ply(argv[2]);
        const auto projection = read_projection(argv[3]);

        OutputState output;
        output.output_path = argv[4];
        racformer_vis_config_t config{};
        std::memcpy(config.radar_to_ego, kRadarToEgo,
                    sizeof(kRadarToEgo));
        std::memcpy(config.ego_to_image, projection.data(),
                    16 * sizeof(float));
        config.projection_crop_y = 224.0F;
        config.forward_range_m = 50.0F;
        config.lateral_range_m = 20.0F;
        config.score_threshold = 0.3F;
        config.class_nms_iou_threshold = 0.2F;
        config.max_pending_frames = 4;
        config.jpeg_quality = 90;
        config.radar_point_radius = 2;
        config.radar_point_alpha = 0.45F;
        config.draw_radar = 1;
        config.draw_labels = 1;

        racformer_vis_handle_t* handle = racformer_vis_create(
            &config, write_output, &output);
        if (!handle) throw std::runtime_error("visualizer creation failed");

        constexpr uint64_t timestamp_ns = 1000000000ULL;
        constexpr uint32_t frame_id = 0;
        constexpr uint32_t version = 1;
        const racformer_vis_camera_t camera{
            timestamp_ns, frame_id, version,
            static_cast<uint32_t>(jpeg.size()), jpeg.data(),
        };
        const racformer_vis_radar_t radar{
            timestamp_ns, frame_id, version,
            static_cast<uint32_t>(radar_points.size()), radar_points.data(),
        };

        /* This clearly synthetic car checks prediction filtering, 3-D corner
           projection, labels, and JPEG output without loading the model. */
        const racformer_vis_box3d_t synthetic_box{
            20.0F, 0.0F, -2.0F, 4.5F, 2.0F, 1.8F,
            0.0F, 0.95F, 0,
        };
        const racformer_vis_predictions_t predictions{
            timestamp_ns, frame_id, version, 1, &synthetic_box,
        };

        const int camera_status = racformer_vis_push_camera(handle, &camera);
        const int prediction_status =
            racformer_vis_push_predictions(handle, &predictions);
        const int radar_status = racformer_vis_push_radar(handle, &radar);
        if (camera_status || prediction_status || radar_status) {
            const char* message = racformer_vis_last_error(handle);
            const std::string detail = message ? message : "unknown error";
            racformer_vis_destroy(handle);
            throw std::runtime_error("push failed: " + detail);
        }

        bool completed = false;
        {
            std::unique_lock<std::mutex> lock(output.mutex);
            completed = output.condition.wait_for(
                lock, std::chrono::seconds(10),
                [&] { return output.finished; });
        }
        if (!completed) {
            const char* message = racformer_vis_last_error(handle);
            const std::string detail = message ? message : "";
            racformer_vis_destroy(handle);
            throw std::runtime_error("visualization timed out: " + detail);
        }
        racformer_vis_destroy(handle);
        if (!output.error.empty()) throw std::runtime_error(output.error);
        if (output.width != 640 || output.height != 480 || output.size == 0)
            throw std::runtime_error("unexpected visualization dimensions");

        std::cout << "input radar points: " << radar_points.size() << '\n'
                  << "output: " << output.output_path << '\n'
                  << "output dimensions: " << output.width << 'x'
                  << output.height << '\n'
                  << "output bytes: " << output.size << '\n'
                  << "render time: " << output.render_ms << " ms\n"
                  << "synthetic boxes: 1\n"
                  << "status: SUCCESS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "status: FAILED\n" << exception.what() << '\n';
        return 1;
    }
}
