"""Deployment preprocessing equivalent to the company validation pipeline."""

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .input_schema import FrameInput, PreparedBatch


class DeploymentPreprocessor:
    """Build RaCFormer inputs without LiDAR, GT depth, or DataContainer."""

    def __init__(self, cfg):
        self.num_frames = int(cfg.num_frames)
        self.num_cams = int(cfg.num_cams)
        if self.num_cams != 1:
            raise ValueError('left-camera deployment requires num_cams=1')

        ida = cfg.ida_aug_conf
        self.source_height = int(ida['H'])
        self.source_width = int(ida['W'])
        self.final_height, self.final_width = map(int, ida['final_dim'])
        self.depth_min = float(cfg.grid_config['depth'][0])
        self.depth_max = float(cfg.grid_config['depth'][1])
        self.point_cloud_range = np.asarray(
            cfg.point_cloud_range, dtype=np.float32)

        self.resize = max(
            self.final_height / self.source_height,
            self.final_width / self.source_width)
        resized_width = int(self.source_width * self.resize)
        resized_height = int(self.source_height * self.resize)
        self.resize_dims = (resized_width, resized_height)
        crop_y = resized_height - self.final_height
        crop_x = max(0, resized_width - self.final_width) // 2
        self.crop = (
            crop_x, crop_y, crop_x + self.final_width,
            crop_y + self.final_height)

        self.ida_matrix = np.eye(4, dtype=np.float32)
        self.ida_matrix[0, 0] = self.resize
        self.ida_matrix[1, 1] = self.resize
        self.ida_matrix[0, 2] = -crop_x
        self.ida_matrix[1, 2] = -crop_y

    def _validate_frame(self, frame):
        if not isinstance(frame, FrameInput):
            raise TypeError('all frames must be FrameInput instances')
        if frame.image.dtype != np.uint8:
            raise TypeError('left image must use uint8 BGR pixels')
        expected_shape = (self.source_height, self.source_width, 3)
        if frame.image.shape != expected_shape:
            raise ValueError(
                'left image shape {} does not match configured {}'.format(
                    frame.image.shape, expected_shape))
        points = np.asarray(frame.radar_points)
        if points.ndim != 2 or points.shape[1] != 7:
            raise ValueError('radar_points must have shape [N, 7]')
        if np.asarray(frame.lidar2img).shape != (4, 4):
            raise ValueError('lidar2img must have shape [4, 4]')
        if np.asarray(frame.intrinsic).shape not in ((3, 3), (4, 4)):
            raise ValueError('intrinsic must have shape [3, 3] or [4, 4]')

    def _transform_image(self, image):
        # PIL is used by RandomTransformImage; retaining it avoids interpolation
        # differences while validating the deployment path against val.py.
        transformed = Image.fromarray(image).resize(self.resize_dims)
        transformed = transformed.crop(self.crop)
        return np.asarray(transformed, dtype=np.uint8)

    def _filter_radar(self, points):
        points = np.asarray(points, dtype=np.float32).copy()
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        keep = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max))
        return points[keep]

    def _radar_maps(self, points, lidar2img):
        height, width = self.final_height, self.final_width
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        rcs_map = torch.zeros((height, width), dtype=torch.float32)
        if points.shape[0] == 0:
            return depth_map, rcs_map

        points_tensor = torch.from_numpy(points)
        projection = torch.from_numpy(lidar2img).to(torch.float32)
        projected = points_tensor[:, :3].matmul(
            projection[:3, :3].T) + projection[:3, 3].unsqueeze(0)
        projected = torch.cat([
            projected[:, :2] / projected[:, 2:3],
            projected[:, 2:3], points_tensor[:, 3:4]
        ], dim=1)

        coordinates = torch.round(projected[:, :2])
        depth = projected[:, 2]
        rcs = projected[:, 3]
        keep = (
            (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width) &
            (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height) &
            (depth < self.depth_max) & (depth >= self.depth_min))
        coordinates = coordinates[keep]
        depth = depth[keep]
        if coordinates.shape[0] == 0:
            return depth_map, rcs_map

        ranks = coordinates[:, 0] + coordinates[:, 1] * width
        order = (ranks + depth / 100.0).argsort()
        coordinates = coordinates[order]
        depth = depth[order]
        # Keep compatibility with the training pipeline, which indexes the
        # original RCS vector with the post-filter ordering.
        rcs = rcs[order]
        ranks = ranks[order]
        unique = torch.ones(len(coordinates), dtype=torch.bool)
        unique[1:] = ranks[1:] != ranks[:-1]
        coordinates = coordinates[unique].to(torch.long)
        depth = depth[unique]
        rcs = rcs[unique]

        # Preserve the checkpoint's existing RadarPointToMultiViewDepth
        # behavior: each projected point fills its complete image column.
        depth_map[:, coordinates[:, 0]] = depth
        rcs_map[:, coordinates[:, 0]] = rcs
        return depth_map, rcs_map

    @staticmethod
    def _intrinsic_4x4(intrinsic):
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape == (4, 4):
            return intrinsic.copy()
        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = intrinsic
        return output

    def prepare(self, frames: Sequence[FrameInput]):
        if len(frames) != self.num_frames:
            raise ValueError(
                'expected {} newest-first frames, got {}'.format(
                    self.num_frames, len(frames)))
        for frame in frames:
            self._validate_frame(frame)

        reference_radar_time = float(frames[0].radar_timestamp)
        images = []
        radar_points = []
        radar_depth = []
        radar_rcs = []
        lidar2imgs = []
        intrinsics = []

        for frame in frames:
            image = self._transform_image(frame.image)
            lidar2img = self.ida_matrix @ np.asarray(
                frame.lidar2img, dtype=np.float32)
            points = self._filter_radar(frame.radar_points)
            points[:, 6] = reference_radar_time - float(
                frame.radar_timestamp)
            depth_map, rcs_map = self._radar_maps(points, lidar2img)

            images.append(torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))))
            radar_points.append(torch.from_numpy(points))
            radar_depth.append(depth_map)
            radar_rcs.append(rcs_map)
            lidar2imgs.append(lidar2img)
            intrinsics.append(self._intrinsic_4x4(frame.intrinsic))

        image_shapes = [
            (self.final_height, self.final_width, 3)
            for _ in range(self.num_frames)
        ]
        img_meta = dict(
            filename=[frame.frame_id for frame in frames],
            ori_shape=list(image_shapes),
            img_shape=list(image_shapes),
            pad_shape=list(image_shapes),
            lidar2img=lidar2imgs,
            img_timestamp=[float(frame.image_timestamp) for frame in frames],
            intrinsics=intrinsics)

        return PreparedBatch(
            image=torch.stack(images).unsqueeze(0),
            radar_points=radar_points,
            radar_depth=torch.stack(radar_depth).unsqueeze(0),
            radar_rcs=torch.stack(radar_rcs).unsqueeze(0),
            img_meta=img_meta)
