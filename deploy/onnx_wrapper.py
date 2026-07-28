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

    def __init__(
            self, model, image_height, image_width,
            debug_intermediates=False):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.debug_intermediates = bool(debug_intermediates)
        self.debug_output_names = []
        self._decoder_debug_samples = []
        self._decoder_internal_debug_samples = {}
        self._decoder_internal_modules = []
        if self.debug_intermediates:
            decoder_layer = self.model.pts_bbox_head.transformer.decoder \
                .decoder_layer
            decoder_layer.register_forward_hook(
                self._capture_decoder_query)
            self._decoder_internal_modules = [
                ('position_encoder', decoder_layer.position_encoder),
                ('self_attn', decoder_layer.self_attn),
                ('norm1', decoder_layer.norm1),
                ('sampling_radar_bev', decoder_layer.sampling_radar_bev),
                ('norm_radar_bev', decoder_layer.norm_radar_bev),
                ('sampling_lss_bev', decoder_layer.sampling_lss_bev),
                ('norm_lss_bev', decoder_layer.norm_lss_bev),
                ('sampling_image', decoder_layer.sampling),
                ('mixing', decoder_layer.mixing),
                ('norm2', decoder_layer.norm2),
                ('fusion', decoder_layer.fusion),
                ('norm_fusion', decoder_layer.norm_fusion),
                ('ffn', decoder_layer.ffn),
                ('norm3', decoder_layer.norm3),
                ('cls_branch', decoder_layer.cls_branch),
                ('reg_branch', decoder_layer.reg_branch),
            ]
            for name, module in self._decoder_internal_modules:
                module.register_forward_hook(
                    self._make_decoder_internal_hook(name))

    @staticmethod
    def _sample_tensor(tensor, sample_count=4096):
        flat = tensor.reshape(-1)
        count = min(flat.numel(), int(sample_count))
        if count == flat.numel():
            return flat
        # Fixed Gather indices avoid exporter-dependent strided Slice nodes.
        indices = torch.tensor(
            [
                index * flat.numel() // count
                for index in range(count)
            ],
            dtype=torch.long,
            device=flat.device)
        return torch.index_select(flat, 0, indices)

    def _capture_decoder_query(self, module, inputs, outputs):
        del module, inputs
        self._decoder_debug_samples.append(
            self._sample_tensor(outputs[0]))

    def _make_decoder_internal_hook(self, name):
        def capture(module, inputs, output):
            del module, inputs
            if name in self._decoder_internal_debug_samples:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self._decoder_internal_debug_samples[name] = \
                self._sample_tensor(tensor)
        return capture

    def forward(self, image, radar_depth, radar_rcs, lidar2img, img2lidar,
                mlp_input, time_diff, velocity_time_diff,
                *radar_voxel_inputs):
        image_shape = (self.image_height, self.image_width, 3)
        img_meta = dict(
            img_shape=[image_shape] * 8,
            ori_shape=[image_shape] * 8,
            pad_shape=[image_shape] * 8,
            lidar2img=lidar2img[0],
            decoder_lidar2img=lidar2img,
            img2lidar=img2lidar[0],
            mlp_input=mlp_input,
            time_diff=time_diff,
            velocity_time_diff=velocity_time_diff)
        if len(radar_voxel_inputs) != 24:
            raise ValueError('expected voxels, num_points, and coors for 8 frames')
        radar_points = [
            tuple(radar_voxel_inputs[index:index + 3])
            for index in range(0, 24, 3)
        ]
        img_feats, bev_feats, radar_bev_feats, _ = self.model.extract_feat(
            img=image,
            radar_points=radar_points,
            radar_depth=radar_depth,
            radar_rcs=radar_rcs,
            img_metas=[img_meta])
        debug_outputs = []
        self._decoder_debug_samples = []
        self._decoder_internal_debug_samples = {}
        if self.debug_intermediates:
            debug_outputs.extend(
                self._sample_tensor(feature) for feature in img_feats)
            debug_outputs.extend([
                self._sample_tensor(bev_feats),
                self._sample_tensor(radar_bev_feats),
            ])
        outputs = self.model.pts_bbox_head(
            img_feats, bev_feats, radar_bev_feats, [img_meta])
        if self.debug_intermediates:
            internal_names = [
                name for name, _ in self._decoder_internal_modules
                if name in self._decoder_internal_debug_samples
            ]
            internal_outputs = [
                self._decoder_internal_debug_samples[name]
                for name in internal_names
            ]
            self.debug_output_names = [
                'debug_img_feat_{}'.format(index)
                for index in range(len(img_feats))
            ] + [
                'debug_lss_bev',
                'debug_radar_bev',
            ] + [
                'debug_decoder0_{}'.format(name)
                for name in internal_names
            ] + [
                'debug_decoder_query_{}'.format(index)
                for index in range(len(self._decoder_debug_samples))
            ]
            return tuple(debug_outputs + internal_outputs
                         + self._decoder_debug_samples + [
                outputs['all_cls_scores'],
                outputs['all_bbox_preds'],
            ])
        return outputs['all_cls_scores'], outputs['all_bbox_preds']


def build_export_inputs(batch, model):
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
    velocity_time_diff_np = time_diff_np[:, 1:2, None].copy()
    velocity_time_diff_np[velocity_time_diff_np < 1e-5] = 1.0

    tensors = [
        batch.image,
        batch.radar_depth,
        batch.radar_rcs,
        torch.from_numpy(lidar2img_np).unsqueeze(0).to(device),
        torch.from_numpy(img2lidar_np).unsqueeze(0).to(device),
        torch.from_numpy(mlp_input_np).to(device),
        torch.from_numpy(time_diff_np).to(device),
        torch.from_numpy(velocity_time_diff_np).to(device),
    ]
    for points in batch.radar_points:
        points = points.clone()
        points[:, 2] = 0
        voxels, coors, num_points = model.radar_voxel_layer(points)
        coors = torch.nn.functional.pad(
            coors, (1, 0), mode='constant', value=0)
        tensors.extend((voxels, num_points, coors))
    return tuple(tensors)


INPUT_NAMES = [
    'image', 'radar_depth', 'radar_rcs', 'lidar2img', 'img2lidar',
    'mlp_input', 'time_diff', 'velocity_time_diff',
]
for frame_index in range(8):
    INPUT_NAMES.extend([
        'radar_voxels_{}'.format(frame_index),
        'radar_num_points_{}'.format(frame_index),
        'radar_coors_{}'.format(frame_index),
    ])

OUTPUT_NAMES = ['all_cls_scores', 'all_bbox_preds']
