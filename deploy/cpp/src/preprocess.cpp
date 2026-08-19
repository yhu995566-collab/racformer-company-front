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
constexpr int kFrames = static_cast<int>(kTemporalFrameCount);
constexpr int kSourceWidth = 640;
constexpr int kSourceHeight = 480;
constexpr int kImageWidth = 640;
constexpr int kImageHeight = 256;
constexpr int kCropTop = 224;
constexpr int kMaxPoints = 10;
constexpr int kPointFields = 7;

struct JpegError { jpeg_error_mgr base; jmp_buf jump; };
void jpeg_fail(j_common_ptr info) {
    auto* error = reinterpret_cast<JpegError*>(info->err);
    longjmp(error->jump, 1);
}

template <typename T> T* data(Tensor& tensor) {
    return reinterpret_cast<T*>(tensor.bytes.data());
}

template <std::size_t Size>
std::array<float, Size> float_array(
        const TensorMap& constants, const char* name) {
    const auto found = constants.find(name);
    if (found == constants.end() || found->second.dtype != DataType::kFloat32 ||
        found->second.element_count() != Size) {
        throw std::runtime_error(std::string("constants must contain float32 ") +
                                 name + " with " + std::to_string(Size) +
                                 " elements");
    }
    std::array<float, Size> result{};
    std::memcpy(result.data(), found->second.bytes.data(), sizeof(result));
    return result;
}

int int_scalar(const TensorMap& constants, const char* name, int fallback) {
    const auto found = constants.find(name);
    if (found == constants.end()) return fallback;
    if (found->second.dtype != DataType::kInt32 ||
        found->second.element_count() != 1) {
        throw std::runtime_error(std::string("constants must contain int32 scalar ") + name);
    }
    int32_t value{};
    std::memcpy(&value, found->second.bytes.data(), sizeof(value));
    return static_cast<int>(value);
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
    point_cloud_range_ = float_array<6>(constants, "decoder_pc_range");
    const auto voxel = constants.find("runtime_voxel_size");
    voxel_size_ = voxel == constants.end()
        ? std::array<float, 3>{0.5F, 0.5F, 6.0F}
        : float_array<3>(constants, "runtime_voxel_size");
    const auto depth = constants.find("runtime_depth_range");
    depth_range_ = depth == constants.end()
        ? std::array<float, 2>{1.0F, point_cloud_range_[3] + 5.0F}
        : float_array<2>(constants, "runtime_depth_range");
    radar_slots_ = int_scalar(
        constants, "runtime_static_radar_voxels", 1024);
    if (voxel_size_[0] <= 0.0F || voxel_size_[1] <= 0.0F ||
        point_cloud_range_[3] <= point_cloud_range_[0] ||
        point_cloud_range_[4] <= point_cloud_range_[1] ||
        depth_range_[0] < 0.0F || depth_range_[1] <= depth_range_[0] ||
        radar_slots_ <= 0) {
        throw std::runtime_error("invalid runtime geometry constants");
    }
    const float bev_width =
        (point_cloud_range_[3] - point_cloud_range_[0]) / voxel_size_[0];
    const float bev_height =
        (point_cloud_range_[4] - point_cloud_range_[1]) / voxel_size_[1];
    bev_width_ = static_cast<int>(std::lround(bev_width));
    bev_height_ = static_cast<int>(std::lround(bev_height));
    if (bev_width_ <= 0 || bev_height_ <= 0 ||
        std::fabs(bev_width - bev_width_) > 1e-4F ||
        std::fabs(bev_height - bev_height_) > 1e-4F ||
        radar_slots_ > bev_width_ * bev_height_) {
        throw std::runtime_error("runtime range is not aligned to the BEV voxel grid");
    }
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
        if (point.x >= point_cloud_range_[0] &&
            point.x <= point_cloud_range_[3] &&
            point.y >= point_cloud_range_[1] &&
            point.y <= point_cloud_range_[4] &&
            point.z >= point_cloud_range_[2] &&
            point.z <= point_cloud_range_[5]) result.push_back(point);
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
        if (pz < depth_range_[0] || pz >= depth_range_[1]) continue;
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
    *voxels = make_float_tensor({radar_slots_, kMaxPoints, kPointFields});
    *counts = make_int32_tensor({radar_slots_});
    *coordinates = make_int32_tensor({radar_slots_, 4});
    float* voxel_data = data<float>(*voxels);
    int32_t* count_data = data<int32_t>(*counts);
    int32_t* coordinate_data = data<int32_t>(*coordinates);
    std::unordered_map<int, int> slots;
    std::vector<bool> used(bev_width_ * bev_height_, false);
    int slot_count = 0;
    for (const auto& point : points) {
        const int x = std::clamp(static_cast<int>(std::floor(
            (point.x - point_cloud_range_[0]) / voxel_size_[0])), 0, bev_width_ - 1);
        const int y = std::clamp(static_cast<int>(std::floor(
            (point.y - point_cloud_range_[1]) / voxel_size_[1])), 0, bev_height_ - 1);
        const int key = y * bev_width_ + x;
        auto found = slots.find(key);
        if (found == slots.end()) {
            if (slot_count == radar_slots_)
                throw std::runtime_error("radar voxel count exceeds configured capacity " +
                                         std::to_string(radar_slots_));
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
    for (int slot = slot_count; slot < radar_slots_; ++slot) {
        while (unused < static_cast<int>(used.size()) && used[unused]) ++unused;
        if (unused == static_cast<int>(used.size())) throw std::runtime_error("insufficient unused BEV padding cells");
        coordinate_data[slot * 4 + 2] = unused / bev_width_;
        coordinate_data[slot * 4 + 3] = unused % bev_width_;
        ++unused;
    }
}

TensorMap Preprocessor::prepare(const PairedFrameWindow& frames) const {
    for (const auto& frame : frames) {
        if (!frame) throw std::runtime_error("preprocessor received an empty temporal frame");
    }
    TensorMap output;
    output["image"] = make_uint8_tensor({1, kFrames, 3, kImageHeight, kImageWidth});
    output["radar_depth"] = make_float_tensor({1, kFrames, kImageHeight, kImageWidth});
    output["radar_rcs"] = make_float_tensor({1, kFrames, kImageHeight, kImageWidth});
    output["time_diff"] = make_float_tensor({1, kFrames});
    output["velocity_time_diff"] = make_float_tensor({1, 1, 1});
    float* time_diff = data<float>(output["time_diff"]);
    const uint64_t newest_radar = frames.front()->radar.timestamp;
    const uint64_t newest_image = frames.front()->camera.timestamp;
    const size_t image_frame_bytes = 3ULL * kImageHeight * kImageWidth;
    const size_t map_frame_elements = kImageHeight * kImageWidth;
    for (int index = 0; index < kFrames; ++index) {
        const auto image = decode_and_crop(frames[index]->camera);
        std::memcpy(output["image"].bytes.data() + index * image_frame_bytes, image.data(), image.size());
        const auto points = transform_points(frames[index]->radar, newest_radar);
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
             static_cast<double>(frames[index]->camera.timestamp)) * 1e-9);
    }
    data<float>(output["velocity_time_diff"])[0] = time_diff[1] < 1e-5F ? 1.0F : time_diff[1];
    return output;
}
}  // namespace racformer
