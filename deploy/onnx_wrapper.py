"""Tensor-only ONNX boundary for the RaCFormer deployment model."""

import numpy as np
import torch
from torch import nn


class RaCFormerONNXWrapper(nn.Module):
    """Expose fixed temporal inputs and raw detector-head outputs.

    Calibration-derived tensors are explicit inputs so ONNX never has to trace
    NumPy matrix inversion or Python timestamp processing. Bbox decode remains
    outside the graph because it produces variable-length framework objects.
    """

    def __init__(
            self, model, image_height, image_width,
            debug_intermediates=False, debug_output_group='all'):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.num_frames = int(
            model.pts_bbox_head.transformer.decoder.decoder_layer
            .sampling.num_frames)
        self.debug_intermediates = bool(debug_intermediates)
        self.debug_output_group = debug_output_group
        self.debug_output_names = []
        self._decoder_debug_samples = []
        self._decoder_internal_debug_samples = {}
        self._decoder_internal_modules = []
        if self.debug_intermediates:
            decoder_layer = self.model.pts_bbox_head.transformer.decoder \
                .decoder_layer
            if self.debug_output_group in ('all', 'core', 'post'):
                decoder_layer.register_forward_hook(
                    self._capture_decoder_query)
            common_modules = [
                ('position_encoder', decoder_layer.position_encoder),
                ('self_attn', decoder_layer.self_attn),
                ('norm1', decoder_layer.norm1),
            ]
            grouped_modules = {
                'radar': [
                    ('radar_sampling_offset',
                     decoder_layer.sampling_radar_bev.sampling_offset),
                    ('radar_ray_points_offset',
                     decoder_layer.sampling_radar_bev.ray_points_offset),
                    ('radar_scale_weights',
                     decoder_layer.sampling_radar_bev.scale_weights),
                    ('sampling_radar_bev',
                     decoder_layer.sampling_radar_bev),
                    ('norm_radar_bev', decoder_layer.norm_radar_bev),
                ],
                'lss': [
                    ('lss_sampling_offset',
                     decoder_layer.sampling_lss_bev.sampling_offset),
                    ('lss_ray_points_offset',
                     decoder_layer.sampling_lss_bev.ray_points_offset),
                    ('lss_scale_weights',
                     decoder_layer.sampling_lss_bev.scale_weights),
                    ('sampling_lss_bev',
                     decoder_layer.sampling_lss_bev),
                    ('norm_lss_bev', decoder_layer.norm_lss_bev),
                ],
                'image': [
                    ('image_sampling_offset',
                     decoder_layer.sampling.sampling_offset),
                    ('image_ray_points_offset',
                     decoder_layer.sampling.ray_points_offset),
                    ('image_scale_weights',
                     decoder_layer.sampling.scale_weights),
                    ('sampling_image', decoder_layer.sampling),
                ],
                'post': [
                    ('mixing', decoder_layer.mixing),
                    ('norm2', decoder_layer.norm2),
                    ('fusion', decoder_layer.fusion),
                    ('norm_fusion', decoder_layer.norm_fusion),
                    ('ffn', decoder_layer.ffn),
                    ('norm3', decoder_layer.norm3),
                    ('cls_branch', decoder_layer.cls_branch),
                    ('reg_branch', decoder_layer.reg_branch),
                ],
            }
            if self.debug_output_group == 'all':
                selected_groups = ('radar', 'lss', 'image', 'post')
            elif self.debug_output_group == 'core':
                selected_groups = ()
            else:
                selected_groups = (self.debug_output_group,)
            self._decoder_internal_modules = list(common_modules)
            for group in selected_groups:
                self._decoder_internal_modules.extend(
                    grouped_modules[group])
            for name, module in self._decoder_internal_modules:
                module.register_forward_hook(
                    self._make_decoder_internal_hook(name))
            if self.debug_output_group in ('all', 'radar'):
                decoder_layer.sampling_radar_bev.attention \
                    .register_forward_pre_hook(
                        self._make_attention_input_hook('radar_attention'))
            if self.debug_output_group in ('all', 'lss'):
                decoder_layer.sampling_lss_bev.attention \
                    .register_forward_pre_hook(
                        self._make_attention_input_hook('lss_attention'))

    @staticmethod
    def _sample_tensor(tensor, sample_count=4096):
        flat = tensor.reshape(-1)
        count = min(flat.numel(), int(sample_count))
        if count == flat.numel():
            return flat
        # Keep diagnostics on a contiguous prefix. Constant Gather outputs can
        # become part of large Myelin fusion regions in TensorRT 8.x and alter
        # both tactic selection and the values being inspected.
        return flat[:count]

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

    def _make_attention_input_hook(self, prefix):
        def capture(module, inputs):
            del module
            input_names = ('query', 'value', 'sampling_points', 'weights')
            for name, tensor in zip(input_names, inputs[:4]):
                key = '{}_{}'.format(prefix, name)
                if tensor is not None and \
                        key not in self._decoder_internal_debug_samples:
                    self._decoder_internal_debug_samples[key] = \
                        self._sample_tensor(tensor)
        return capture

    def forward(self, image, radar_depth, radar_rcs, lidar2img, img2lidar,
                mlp_input, time_diff, velocity_time_diff,
                *radar_voxel_inputs):
        image_shape = (self.image_height, self.image_width, 3)
        img_meta = dict(
            img_shape=[image_shape] * self.num_frames,
            ori_shape=[image_shape] * self.num_frames,
            pad_shape=[image_shape] * self.num_frames,
            lidar2img=lidar2img[0],
            decoder_lidar2img=lidar2img,
            img2lidar=img2lidar[0],
            mlp_input=mlp_input,
            time_diff=time_diff,
            velocity_time_diff=velocity_time_diff)
        expected_radar_inputs = self.num_frames * 3
        if len(radar_voxel_inputs) != expected_radar_inputs:
            raise ValueError(
                'expected voxels, num_points, and coors for {} frames'.format(
                    self.num_frames))
        radar_points = [
            tuple(radar_voxel_inputs[index:index + 3])
            for index in range(0, expected_radar_inputs, 3)
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
        if self.debug_intermediates and \
                self.debug_output_group in ('all', 'core'):
            debug_outputs.extend(
                self._sample_tensor(feature) for feature in img_feats)
            debug_outputs.extend([
                self._sample_tensor(bev_feats),
                self._sample_tensor(radar_bev_feats),
            ])
        outputs = self.model.pts_bbox_head(
            img_feats, bev_feats, radar_bev_feats, [img_meta])
        if self.debug_intermediates:
            internal_names = list(self._decoder_internal_debug_samples)
            internal_outputs = [
                self._decoder_internal_debug_samples[name]
                for name in internal_names
            ]
            base_names = []
            if self.debug_output_group in ('all', 'core'):
                base_names = [
                    'debug_img_feat_{}'.format(index)
                    for index in range(len(img_feats))
                ] + [
                    'debug_lss_bev',
                    'debug_radar_bev',
                ]
            self.debug_output_names = base_names + [
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
    num_frames = len(batch.radar_points)
    if lidar2img_np.shape[0] != num_frames:
        raise ValueError(
            'lidar2img frame count {} does not match radar frame count {}'
            .format(lidar2img_np.shape[0], num_frames))
    mlp_input_np = img2lidar_np[:, :3, :3].reshape(
        1, num_frames, 9)

    timestamps = np.asarray(
        batch.img_meta['img_timestamp'], dtype=np.float64).reshape(
            1, num_frames, 1)
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


def get_input_names(num_frames):
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError('num_frames must be positive')
    names = [
        'image', 'radar_depth', 'radar_rcs', 'lidar2img', 'img2lidar',
        'mlp_input', 'time_diff', 'velocity_time_diff',
    ]
    for frame_index in range(num_frames):
        names.extend([
            'radar_voxels_{}'.format(frame_index),
            'radar_num_points_{}'.format(frame_index),
            'radar_coors_{}'.format(frame_index),
        ])
    return names


# Retain the legacy symbol for diagnostic scripts that are specific to the
# original eight-frame checkpoint. Production exporters derive names from the
# selected config through get_input_names().
INPUT_NAMES = get_input_names(8)

OUTPUT_NAMES = ['all_cls_scores', 'all_bbox_preds']
