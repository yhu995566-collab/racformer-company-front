#include "preprocess.hpp"

#include <jpeglib.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace racformer {
namespace {
constexpr int kFrames = 4;
constexpr int kSourceWidth = 640;
constexpr int kSourceHeight = 480;
constexpr int kImageWidth = 640;
constexpr int kImageHeight = 256;
constexpr int kCropTop = 224;
constexpr int kRadarSlots = 1024;
constexpr int kMaxPoints = 10;
constexpr int kPointFields = 7;
constexpr int kBevWidth = 200;
constexpr int kBevHeight = 80;

struct JpegError { jpeg_error_mgr base; jmp_buf jump; };
void jpeg_fail(j_common_ptr info) {
    auto* error = reinterpret_cast<JpegError*>(info->err);
    longjmp(error->jump, 1);
}

template <typename T> T* data(Tensor& tensor) {
    return reinterpret_cast<T*>(tensor.bytes.data());
}
}  // namespace

Preprocessor::Preprocessor(const std::array<float, 16>& radar_to_ego,
                           const TensorMap& constants)
    : radar_to_ego_(radar_to_ego) {
    const auto found = constants.find("lidar2img");
    if (found == constants.end() || found->second.dtype != DataType::kFloat32 ||
        found->second.element_count() < lidar2img_.size()) {
        throw std::runtime_error("constants must contain float32 lidar2img [1,4,4,4]");
    }
    std::memcpy(lidar2img_.data(), found->second.bytes.data(), sizeof(lidar2img_));
}

std::vector<uint8_t> Preprocessor::decode_and_crop(const CameraCopy& camera) const {
    jpeg_decompress_struct decoder{};
    JpegError error{};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jpeg_fail;
    if (setjmp(error.jump)) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error("invalid JPEG for frame " + std::to_string(camera.frame_id));
    }
    jpeg_create_decompress(&decoder);
    jpeg_mem_src(&decoder, camera.jpeg.data(), camera.jpeg.size());
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);
    if (decoder.output_width != kSourceWidth || decoder.output_height != kSourceHeight ||
        decoder.output_components != 3) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error("camera JPEG must decode to 640x480 RGB");
    }
    std::vector<uint8_t> row(kSourceWidth * 3);
    std::vector<uint8_t> chw(3 * kImageHeight * kImageWidth);
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW pointer = row.data();
        const int y = static_cast<int>(decoder.output_scanline);
        jpeg_read_scanlines(&decoder, &pointer, 1);
        if (y < kCropTop) continue;
        const int target_y = y - kCropTop;
        for (int x = 0; x < kImageWidth; ++x) {
            // The engine expects uint8 BGR and normalizes/converts internally.
            chw[0 * kImageHeight * kImageWidth + target_y * kImageWidth + x] = row[x * 3 + 2];
            chw[1 * kImageHeight * kImageWidth + target_y * kImageWidth + x] = row[x * 3 + 1];
            chw[2 * kImageHeight * kImageWidth + target_y * kImageWidth + x] = row[x * 3 + 0];
        }
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    return chw;
}

std::vector<Preprocessor::Point> Preprocessor::transform_points(
        const RadarCopy& radar, uint64_t newest_timestamp) const {
    std::vector<Point> result;
    result.reserve(radar.points.size());
    const float lag = static_cast<float>(
        (static_cast<double>(newest_timestamp) - static_cast<double>(radar.timestamp)) * 1e-9);
    for (const auto& source : radar.points) {
        const float radius = std::hypot(source.x, source.y);
        const float compensated = source.v + (radius > 1e-6F ? source.ego_speed * source.y / radius : 0.0F);
        const float raw_vx = radius > 1e-6F ? compensated * source.x / radius : 0.0F;
        const float raw_vy = radius > 1e-6F ? compensated * source.y / radius : 0.0F;
        Point point{};
        point.x = radar_to_ego_[0] * source.x + radar_to_ego_[1] * source.y + radar_to_ego_[2] * source.z + radar_to_ego_[3];
        point.y = radar_to_ego_[4] * source.x + radar_to_ego_[5] * source.y + radar_to_ego_[6] * source.z + radar_to_ego_[7];
        point.z = radar_to_ego_[8] * source.x + radar_to_ego_[9] * source.y + radar_to_ego_[10] * source.z + radar_to_ego_[11];
        point.vx = radar_to_ego_[0] * raw_vx + radar_to_ego_[1] * raw_vy;
        point.vy = radar_to_ego_[4] * raw_vx + radar_to_ego_[5] * raw_vy;
        point.rcs = source.rcs;
        point.lag = lag;
        if (point.x >= 0.0F && point.x <= 100.0F && point.y >= -20.0F &&
            point.y <= 20.0F && point.z >= -3.0F && point.z <= 3.0F) result.push_back(point);
    }
    return result;
}

void Preprocessor::make_maps(const std::vector<Point>& points, const float* matrix,
                             float* depth_map, float* rcs_map) const {
    struct Projection { int u, v; float depth; size_t filtered_index; };
    std::vector<Projection> projected;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        const float px = matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3];
        const float py = matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7];
        const float pz = matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11];
        if (pz < 1.0F || pz >= 105.0F) continue;
        const int u = static_cast<int>(std::nearbyint(px / pz));
        const int v = static_cast<int>(std::nearbyint(py / pz));
        if (u >= 0 && u < kImageWidth && v >= 0 && v < kImageHeight)
            projected.push_back({u, v, pz, projected.size()});
    }
    // Match the Python training/deployment path: rank first, nearer depth first.
    std::stable_sort(projected.begin(), projected.end(), [](const Projection& a, const Projection& b) {
        return (a.u + a.v * kImageWidth + a.depth / 100.0F) <
               (b.u + b.v * kImageWidth + b.depth / 100.0F);
    });
    int previous_rank = -1;
    for (const auto& p : projected) {
        const int rank = p.u + p.v * kImageWidth;
        if (rank == previous_rank) continue;
        previous_rank = rank;
        // Compatibility quirk: Python indexes the unfiltered RCS prefix by
        // post-filter sort indices. Using filtered_index preserves that bug.
        const float rcs = p.filtered_index < points.size() ? points[p.filtered_index].rcs : 0.0F;
        for (int y = 0; y < kImageHeight; ++y) {
            depth_map[y * kImageWidth + p.u] = p.depth;
            rcs_map[y * kImageWidth + p.u] = rcs;
        }
    }
}

void Preprocessor::voxelize(const std::vector<Point>& points, Tensor* voxels,
                            Tensor* counts, Tensor* coordinates) const {
    *voxels = make_float_tensor({kRadarSlots, kMaxPoints, kPointFields});
    *counts = make_int32_tensor({kRadarSlots});
    *coordinates = make_int32_tensor({kRadarSlots, 4});
    float* voxel_data = data<float>(*voxels);
    int32_t* count_data = data<int32_t>(*counts);
    int32_t* coordinate_data = data<int32_t>(*coordinates);
    std::unordered_map<int, int> slots;
    std::vector<bool> used(kBevWidth * kBevHeight, false);
    int slot_count = 0;
    for (const auto& point : points) {
        const int x = std::min(kBevWidth - 1, static_cast<int>(std::floor(point.x / 0.5F)));
        const int y = std::min(kBevHeight - 1, static_cast<int>(std::floor((point.y + 20.0F) / 0.5F)));
        const int key = y * kBevWidth + x;
        auto found = slots.find(key);
        if (found == slots.end()) {
            if (slot_count == kRadarSlots) throw std::runtime_error("radar voxel count exceeds 1024");
            found = slots.emplace(key, slot_count++).first;
            used[key] = true;
            coordinate_data[found->second * 4 + 2] = y;
            coordinate_data[found->second * 4 + 3] = x;
        }
        const int slot = found->second;
        if (count_data[slot] >= kMaxPoints) continue;
        const int offset = (slot * kMaxPoints + count_data[slot]++) * kPointFields;
        const float fields[kPointFields] = {point.x, point.y, 0.0F, point.rcs, point.vx, point.vy, point.lag};
        std::copy(fields, fields + kPointFields, voxel_data + offset);
    }
    int unused = 0;
    for (int slot = slot_count; slot < kRadarSlots; ++slot) {
        while (unused < static_cast<int>(used.size()) && used[unused]) ++unused;
        if (unused == static_cast<int>(used.size())) throw std::runtime_error("insufficient unused BEV padding cells");
        coordinate_data[slot * 4 + 2] = unused / kBevWidth;
        coordinate_data[slot * 4 + 3] = unused % kBevWidth;
        ++unused;
    }
}

TensorMap Preprocessor::prepare(const std::vector<PairedFrame>& frames) const {
    if (frames.size() != kFrames) throw std::runtime_error("preprocessor requires four newest-first frames");
    TensorMap output;
    output["image"] = make_uint8_tensor({1, kFrames, 3, kImageHeight, kImageWidth});
    output["radar_depth"] = make_float_tensor({1, kFrames, kImageHeight, kImageWidth});
    output["radar_rcs"] = make_float_tensor({1, kFrames, kImageHeight, kImageWidth});
    output["time_diff"] = make_float_tensor({1, kFrames});
    output["velocity_time_diff"] = make_float_tensor({1, 1, 1});
    float* time_diff = data<float>(output["time_diff"]);
    const uint64_t newest_radar = frames.front().radar.timestamp;
    const uint64_t newest_image = frames.front().camera.timestamp;
    const size_t image_frame_bytes = 3ULL * kImageHeight * kImageWidth;
    const size_t map_frame_elements = kImageHeight * kImageWidth;
    for (int index = 0; index < kFrames; ++index) {
        const auto image = decode_and_crop(frames[index].camera);
        std::memcpy(output["image"].bytes.data() + index * image_frame_bytes, image.data(), image.size());
        const auto points = transform_points(frames[index].radar, newest_radar);
        make_maps(points, lidar2img_.data() + index * 16,
                  data<float>(output["radar_depth"]) + index * map_frame_elements,
                  data<float>(output["radar_rcs"]) + index * map_frame_elements);
        Tensor voxels, counts, coordinates;
        voxelize(points, &voxels, &counts, &coordinates);
        output["radar_voxels_" + std::to_string(index)] = std::move(voxels);
        output["radar_num_points_" + std::to_string(index)] = std::move(counts);
        output["radar_coors_" + std::to_string(index)] = std::move(coordinates);
        time_diff[index] = static_cast<float>(
            (static_cast<double>(newest_image) -
             static_cast<double>(frames[index].camera.timestamp)) * 1e-9);
    }
    data<float>(output["velocity_time_diff"])[0] = time_diff[1] < 1e-5F ? 1.0F : time_diff[1];
    return output;
}
}  // namespace racformer
