"""Tensor-only ONNX boundary for the RaCFormer deployment model."""

import numpy as np
import torch
from torch import nn


class RaCFormerONNXWrapper(nn.Module):
    """Expose fixed eight-frame inputs and raw detector-head outputs.

    Calibration-derived tensors are explicit inputs so ONNX never has to trace
    NumPy matrix inversion or Python timestamp processing. Bbox decode remains
    outside the graph because it produces variable-length framework objects.
    """

    def __init__(self, model, image_height, image_width):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def forward(self, image, radar_depth, radar_rcs, lidar2img, img2lidar,
                mlp_input, time_diff, radar_points_0, radar_points_1,
                radar_points_2, radar_points_3, radar_points_4,
                radar_points_5, radar_points_6, radar_points_7):
        image_shape = (self.image_height, self.image_width, 3)
        img_meta = dict(
            img_shape=[image_shape] * 8,
            ori_shape=[image_shape] * 8,
            pad_shape=[image_shape] * 8,
            lidar2img=lidar2img[0],
            img2lidar=img2lidar[0],
            mlp_input=mlp_input,
            time_diff=time_diff)
        radar_points = [
            [radar_points_0], [radar_points_1], [radar_points_2],
            [radar_points_3], [radar_points_4], [radar_points_5],
            [radar_points_6], [radar_points_7],
        ]
        img_feats, bev_feats, radar_bev_feats, _ = self.model.extract_feat(
            img=image,
            radar_points=radar_points,
            radar_depth=radar_depth,
            radar_rcs=radar_rcs,
            img_metas=[img_meta])
        outputs = self.model.pts_bbox_head(
            img_feats, bev_feats, radar_bev_feats, [img_meta])
        return outputs['all_cls_scores'], outputs['all_bbox_preds']


def build_export_inputs(batch):
    """Create wrapper inputs from one GPU-resident PreparedBatch."""
    device = batch.image.device
    lidar2img_np = np.asarray(
        batch.img_meta['lidar2img'], dtype=np.float32)
    img2lidar_np = np.linalg.inv(lidar2img_np).astype(np.float32)
    mlp_input_np = img2lidar_np[:, :3, :3].reshape(1, 8, 9)

    timestamps = np.asarray(
        batch.img_meta['img_timestamp'], dtype=np.float64).reshape(1, 8, 1)
    time_diff_np = (timestamps[:, :1] - timestamps).mean(
        axis=-1).astype('float32')

    tensors = (
        batch.image,
        batch.radar_depth,
        batch.radar_rcs,
        torch.from_numpy(lidar2img_np).unsqueeze(0).to(device),
        torch.from_numpy(img2lidar_np).unsqueeze(0).to(device),
        torch.from_numpy(mlp_input_np).to(device),
        torch.from_numpy(time_diff_np).to(device),
    )
    return tensors + tuple(batch.radar_points)


INPUT_NAMES = [
    'image', 'radar_depth', 'radar_rcs', 'lidar2img', 'img2lidar',
    'mlp_input', 'time_diff',
] + ['radar_points_{}'.format(index) for index in range(8)]

OUTPUT_NAMES = ['all_cls_scores', 'all_bbox_preds']
