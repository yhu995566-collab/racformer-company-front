#pragma once

#include "racformer/c_api.h"
#include "tensor_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace racformer {

struct CameraCopy {
    uint64_t timestamp{};
    uint32_t frame_id{};
    uint32_t version{};
    std::vector<uint8_t> jpeg;
};
struct RadarCopy {
    uint64_t timestamp{};
    uint32_t frame_id{};
    uint32_t version{};
    std::vector<radar_raw_data_t> points;
};
struct PairedFrame { CameraCopy camera; RadarCopy radar; };
constexpr std::size_t kTemporalFrameCount = 4;
using PairedFrameWindow = std::array<
    std::shared_ptr<const PairedFrame>, kTemporalFrameCount>;

class Preprocessor {
 public:
    Preprocessor(const std::array<float, 16>& radar_to_ego,
                 const TensorMap& constants);
    TensorMap prepare(const PairedFrameWindow& newest_first) const;

 private:
    struct Point { float x, y, z, rcs, vx, vy, lag; };
    std::vector<uint8_t> decode_and_crop(const CameraCopy& camera) const;
    std::vector<Point> transform_points(const RadarCopy& radar,
                                        uint64_t newest_timestamp) const;
    void make_maps(const std::vector<Point>& points, const float* lidar2img,
                   float* depth, float* rcs) const;
    void voxelize(const std::vector<Point>& points, Tensor* voxels,
                  Tensor* counts, Tensor* coordinates) const;

    std::array<float, 16> radar_to_ego_{};
    std::array<float, 64> lidar2img_{};
    std::array<float, 6> point_cloud_range_{};
    std::array<float, 3> voxel_size_{};
    std::array<float, 2> depth_range_{};
    int radar_slots_{};
    int bev_width_{};
    int bev_height_{};
};

}  // namespace racformer
