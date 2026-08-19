#!/usr/bin/env python3
"""Export a single-frame image encoder and a cache-consuming frontend."""

import argparse
import os
import traceback

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

from deploy.bev_precompute import (
    PRECOMPUTED_BEV_INPUT_NAMES,
    enable_precomputed_bev_values,
    precompute_bev_values,
)
from deploy.export_decoder_stack_probe import (
    DETECTION_OUTPUT_NAMES,
    FEATURE_INPUT_NAMES,
    DecoderStackProbe,
)
from deploy.export_frontend_onnx import (
    FRONTEND_BASE_OUTPUT_NAMES,
    RaCFormerFrontendONNXWrapper,
)
from deploy.export_onnx import (
    disable_gradient_checkpointing,
    enable_fixed_view_geometry,
    enable_single_batch_radar_scatter,
    enable_standard_onnx_fallbacks,
    install_export_symbolics,
)
from deploy.onnx_wrapper import get_input_names
from deploy.pytorch_runner import RaCFormerPyTorchRunner
from deploy.tensorrt.rewrite_trt85_onnx import (
    rewrite_trt85_unsupported_nodes,
)
from deploy.tensorrt.validate_engine_numpy import decode_detections


IMAGE_ENCODER_OUTPUT_NAMES = [
    'image_frame_fpn_0',
    'image_frame_fpn_1',
    'image_frame_fpn_2',
    'image_frame_fpn_3',
    'image_frame_lss',
]
CACHED_IMAGE_FEATURE_NAMES = [
    'cached_image_fpn_0',
    'cached_image_fpn_1',
    'cached_image_fpn_2',
    'cached_image_fpn_3',
    'cached_image_lss',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--model-fixture', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--image-encoder-out', required=True)
    parser.add_argument('--cached-frontend-out', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--feature-atol', type=float, default=2e-2)
    parser.add_argument('--decoded-atol', type=float, default=3e-2)
    return parser.parse_args()


class SingleFrameImageEncoder(nn.Module):

    def __init__(self, model, image_height, image_width):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        data_aug = model.data_aug or {}
        norm = data_aug.get('img_norm_cfg')
        if norm is None:
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
            self.register_buffer('std', torch.ones(1, 3, 1, 1))
            self.to_rgb = False
        else:
            self.register_buffer(
                'mean', torch.tensor(norm['mean']).reshape(1, 3, 1, 1))
            self.register_buffer(
                'std', torch.tensor(norm['std']).reshape(1, 3, 1, 1))
            self.to_rgb = bool(norm.get('to_rgb', False))
        divisor = int(data_aug.get('img_pad_cfg', {}).get(
            'size_divisor', 1))
        self.pad_height = (
            (divisor - self.image_height % divisor) % divisor)
        self.pad_width = (
            (divisor - self.image_width % divisor) % divisor)

    def forward(self, image_frame):
        image = image_frame.float()
        if self.to_rgb:
            image = image[:, [2, 1, 0], :, :]
        image = (image - self.mean) / self.std
        if self.pad_height or self.pad_width:
            image = F.pad(image, (0, self.pad_width, 0, self.pad_height))
        fpn, lss = self.model.extract_img_feat(image)
        expected_fpn_levels = len(IMAGE_ENCODER_OUTPUT_NAMES) - 1
        if len(fpn) != expected_fpn_levels:
            raise ValueError(
                'expected {} FPN levels, got {}'.format(
                    expected_fpn_levels, len(fpn)))
        return tuple(list(fpn) + [lss])


class CachedImageFrontend(nn.Module):

    def __init__(self, model, image_height, image_width):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        decoder = model.pts_bbox_head.transformer.decoder
        self.num_cams = int(decoder.num_cams)
        self.decoder_layer = decoder.decoder_layer
        self.num_frames = int(self.decoder_layer.sampling.num_frames)

    def _organize_image_features(self, image_features):
        organized = []
        for feature in image_features:
            batch, frames, grouped_channels, height, width = feature.shape
            groups = 4
            channels = grouped_channels // groups
            feature = feature.reshape(
                batch, frames, self.num_cams, groups, channels,
                height, width)
            feature = feature.permute(0, 1, 3, 2, 5, 6, 4)
            feature = feature.reshape(
                batch * frames * groups, self.num_cams,
                height, width, channels)
            organized.append(feature.contiguous())
        return organized

    def _initial_queries(self, batch):
        head = self.model.pts_bbox_head
        query_bbox = head.init_query_bbox.weight.clone()
        query_bbox = query_bbox.view(
            1, head.num_query, 10).repeat(batch, 1, 1)
        indicator = torch.zeros(
            [head.num_query, 1], device=query_bbox.device,
            dtype=query_bbox.dtype)
        query_feat = head.label_enc.weight[head.num_classes].repeat(
            head.num_query, 1)
        query_feat = torch.cat([query_feat, indicator], dim=1)
        query_feat = query_feat.repeat(batch, 1, 1)
        return query_bbox, query_feat

    def forward(
            self, cached_image_fpn_0, cached_image_fpn_1,
            cached_image_fpn_2, cached_image_fpn_3, cached_image_lss,
            radar_depth, radar_rcs, lidar2img, img2lidar,
            mlp_input,
            *radar_voxel_inputs):
        image_features = [
            cached_image_fpn_0,
            cached_image_fpn_1,
            cached_image_fpn_2,
            cached_image_fpn_3,
        ]
        image_shape = (self.image_height, self.image_width, 3)
        img_meta = dict(
            img_shape=[image_shape] * self.num_frames,
            ori_shape=[image_shape] * self.num_frames,
            pad_shape=[image_shape] * self.num_frames,
            lidar2img=lidar2img[0],
            decoder_lidar2img=lidar2img,
            img2lidar=img2lidar[0],
            mlp_input=mlp_input)
        expected_radar_inputs = self.num_frames * 3
        if len(radar_voxel_inputs) != expected_radar_inputs:
            raise ValueError(
                'expected {} radar tensors, got {}'.format(
                    expected_radar_inputs, len(radar_voxel_inputs)))
        radar_points = [
            tuple(radar_voxel_inputs[index:index + 3])
            for index in range(0, expected_radar_inputs, 3)
        ]

        lss_bev_frames = []
        radar_bev_frames = []
        batch = cached_image_lss.shape[0]
        for index in range(self.num_frames):
            frame_meta = [dict(
                lidar2img=img_meta['lidar2img'][index:index + 1],
                img2lidar=img_meta['img2lidar'][index:index + 1],
                img_shape=[image_shape])]
            radar_bev = self.model.extract_pts_feat(
                radar_points=radar_points[index])
            if radar_depth.dim() == 4:
                frame_radar_depth = radar_depth[:, index:index + 1]
                frame_radar_rcs = radar_rcs[:, index:index + 1]
            elif radar_depth.dim() == 5:
                frame_radar_depth = radar_depth[:, :, index]
                frame_radar_rcs = radar_rcs[:, :, index]
            else:
                raise ValueError(
                    'expected radar maps with 4 or 5 dimensions, got {}'.format(
                        radar_depth.dim()))
            lss_bev, _ = self.model.img_lss_view_transformer(
                cached_image_lss[:, index:index + 1],
                frame_radar_depth, frame_radar_rcs,
                frame_meta, mlp_input[:, index:index + 1])
            if self.model.pre_process:
                lss_bev = self.model.pre_process_net(lss_bev)[0]
            lss_bev_frames.append(lss_bev)
            radar_bev_frames.append(radar_bev)

        lss_bev = torch.stack(lss_bev_frames, dim=1)
        radar_bev = torch.stack(radar_bev_frames, dim=1)
        lss_value, radar_value = precompute_bev_values(
            self.decoder_layer, lss_bev, radar_bev)
        query_bbox, query_feat = self._initial_queries(batch)
        return tuple([
            query_bbox,
            query_feat,
        ] + self._organize_image_features(image_features) + [
            lss_value,
            radar_value,
        ])


def describe(name, tensor):
    return '{}: shape={}, dtype={}'.format(
        name, tuple(tensor.shape), tensor.dtype)


def comparison(name, actual, reference, atol):
    difference = (actual.float() - reference.float()).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    mean = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(
        actual.float(), reference.float(), atol=atol, rtol=0.0))
    return close, '{}: close={}, max_abs_error={:.8f}, mean_abs_error={:.8f}'.format(
        name, close, maximum, mean)


def numpy_max_error(actual, reference):
    if actual.shape != reference.shape:
        return float('inf')
    if actual.size == 0:
        return 0.0
    return float(np.abs(actual - reference).max())


def save_onnx(module, inputs, input_names, output_names, path, opset):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    install_export_symbolics(opset, tensorrt_85_compat=True)
    torch.onnx.export(
        module, inputs, path, export_params=True, opset_version=opset,
        do_constant_folding=False, input_names=input_names,
        output_names=output_names, verbose=False)
    rewrite = rewrite_trt85_unsupported_nodes(path, path)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if rewrite['isinf_remaining'] or rewrite['layernorm_remaining']:
        raise RuntimeError('TensorRT 8.5 unsupported operators remain')
    actual_outputs = [value.name for value in model.graph.output]
    if actual_outputs != list(output_names):
        raise RuntimeError('unexpected ONNX outputs: {}'.format(
            actual_outputs))
    return rewrite


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Cached frontend export report: {}'.format(path))


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer cached frontend export ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'model fixture: {}'.format(os.path.abspath(args.model_fixture)),
        'device: {}'.format(args.device),
        'opset: {}'.format(args.opset),
        'parity atol: {}'.format(args.atol),
        'feature parity atol: {}'.format(args.feature_atol),
        'decoded parity atol: {}'.format(args.decoded_atol),
    ]
    fixture_data = None
    try:
        runner = RaCFormerPyTorchRunner(
            args.config, args.weights, device=args.device)
        disable_gradient_checkpointing(runner.model)
        decoder = runner.model.pts_bbox_head.transformer.decoder
        num_frames = int(decoder.decoder_layer.sampling.num_frames)
        input_names = get_input_names(num_frames)
        fixture_data = np.load(args.model_fixture)
        missing = [name for name in input_names if name not in fixture_data]
        if missing:
            raise KeyError('model fixture is missing inputs: {}'.format(
                missing))
        model_inputs = tuple(
            torch.from_numpy(np.ascontiguousarray(fixture_data[name])).to(
                runner.device)
            for name in input_names)

        enable_standard_onnx_fallbacks(
            runner.model, mixing_chunk_size=32768,
            use_msmv_plugin=True,
            use_single_camera_projection_plugin=True)
        enable_single_batch_radar_scatter(runner.model)
        enable_fixed_view_geometry(runner.model, model_inputs[4])
        runner.model._deploy_trt_static_radar_padding = True

        height = int(model_inputs[0].shape[-2])
        width = int(model_inputs[0].shape[-1])
        encoder = SingleFrameImageEncoder(
            runner.model, height, width).to(runner.device).eval()
        cached_frontend = CachedImageFrontend(
            runner.model, height, width).to(runner.device).eval()
        baseline_frontend = RaCFormerFrontendONNXWrapper(
            runner.model, height, width,
            precompute_values=True).to(runner.device).eval()
        decoder_probe = DecoderStackProbe(
            decoder.decoder_layer, decoder.num_layers, decoder.pc_range,
            height, width, detection_outputs=True).to(runner.device).eval()
        enable_precomputed_bev_values(decoder_probe.decoder_layer)

        frame_outputs = []
        with torch.no_grad():
            for index in range(num_frames):
                frame_outputs.append(encoder(model_inputs[0][:, index]))
            cached_features = tuple(
                torch.stack([
                    frame_outputs[frame][level]
                    for frame in range(num_frames)
                ], dim=1)
                for level in range(len(IMAGE_ENCODER_OUTPUT_NAMES)))
            cached_inputs = (
                cached_features + model_inputs[1:6] + model_inputs[8:])
            baseline_outputs = baseline_frontend(*model_inputs)
            cached_outputs = cached_frontend(*cached_inputs)
            baseline_detection = decoder_probe(*(
                baseline_outputs + (
                    model_inputs[3], model_inputs[6], model_inputs[7])))
            cached_detection = decoder_probe(*(
                cached_outputs + (
                    model_inputs[3], model_inputs[6], model_inputs[7])))
        torch.cuda.synchronize(runner.device)

        frontend_parity_passed = True
        raw_detection_parity_passed = True
        lines.extend(['', '=== Cached tensor parity ==='])
        frontend_output_names = (
            FRONTEND_BASE_OUTPUT_NAMES + PRECOMPUTED_BEV_INPUT_NAMES)
        for name, actual, reference in zip(
                frontend_output_names, cached_outputs, baseline_outputs):
            tolerance = (
                args.feature_atol
                if name.startswith('image_feat_') or
                name == 'lss_bev_value'
                else args.atol)
            close, line = comparison(
                name, actual, reference, tolerance)
            frontend_parity_passed &= close
            lines.append('{} (atol={})'.format(line, tolerance))
        for name, actual, reference in zip(
                DETECTION_OUTPUT_NAMES,
                cached_detection, baseline_detection):
            close, line = comparison(
                name, actual, reference, args.atol)
            raw_detection_parity_passed &= close
            lines.append(line)

        pc_range = np.asarray(decoder.pc_range, dtype=np.float32)
        cached_detection_np = tuple(
            tensor.detach().cpu().numpy()
            for tensor in cached_detection)
        baseline_detection_np = tuple(
            tensor.detach().cpu().numpy()
            for tensor in baseline_detection)
        actual_decoded = decode_detections(
            cached_detection_np[0], cached_detection_np[1], pc_range)
        reference_decoded = decode_detections(
            baseline_detection_np[0], baseline_detection_np[1], pc_range)
        actual_boxes, actual_scores, actual_labels = actual_decoded
        reference_boxes, reference_scores, reference_labels = \
            reference_decoded
        boxes_match = actual_boxes.shape == reference_boxes.shape and \
            np.allclose(
                actual_boxes, reference_boxes, rtol=0.0,
                atol=args.decoded_atol)
        scores_match = actual_scores.shape == reference_scores.shape and \
            np.allclose(
                actual_scores, reference_scores, rtol=0.0,
                atol=args.decoded_atol)
        labels_match = np.array_equal(actual_labels, reference_labels)
        decoded_passed = boxes_match and scores_match and labels_match
        lines.extend([
            '',
            '=== Decoded detection parity ===',
            'actual/reference detection count: {}/{}'.format(
                len(actual_boxes), len(reference_boxes)),
            'boxes close: {}, max_abs_error={:.8f}'.format(
                boxes_match, numpy_max_error(
                    actual_boxes, reference_boxes)),
            'scores close: {}, max_abs_error={:.8f}'.format(
                scores_match, numpy_max_error(
                    actual_scores, reference_scores)),
            'labels equal: {}'.format(labels_match),
            'raw detection tensor parity: {}'.format(
                'PASS' if raw_detection_parity_passed else 'FAIL'),
            'decoded detection parity: {}'.format(
                'PASS' if decoded_passed else 'FAIL'),
        ])
        parity_passed = frontend_parity_passed and decoded_passed
        lines.append('PyTorch cached parity: {}'.format(
            'PASS' if parity_passed else 'FAIL'))
        if not parity_passed:
            raise RuntimeError('cached frontend PyTorch parity failed')

        fixture_path = os.path.abspath(args.fixture)
        os.makedirs(os.path.dirname(fixture_path) or '.', exist_ok=True)
        single_frame_input = model_inputs[0][:, 0]
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                CACHED_IMAGE_FEATURE_NAMES, cached_features)
        }
        arrays['image_frame'] = single_frame_input.detach().cpu().numpy()
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                IMAGE_ENCODER_OUTPUT_NAMES, frame_outputs[0])
        })
        arrays.update({
            name: fixture_data[name]
            for name in input_names[1:]
        })
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                frontend_output_names, cached_outputs)
        })
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                DETECTION_OUTPUT_NAMES, cached_detection)
        })
        arrays['decoder_d_regions'] = np.asarray(
            decoder.decoder_layer.d_region_list, dtype=np.float32)
        arrays['decoder_pc_range'] = pc_range
        arrays['decoder_polar_radius'] = np.asarray(
            getattr(decoder.decoder_layer, 'polar_radius', 65.0),
            dtype=np.float32)
        np.savez_compressed(fixture_path, **arrays)

        encoder_rewrite = save_onnx(
            encoder, (single_frame_input,), ['image_frame'],
            IMAGE_ENCODER_OUTPUT_NAMES, args.image_encoder_out,
            args.opset)
        cached_input_names = (
            CACHED_IMAGE_FEATURE_NAMES + input_names[1:6] + input_names[8:])
        cached_rewrite = save_onnx(
            cached_frontend, cached_inputs, cached_input_names,
            frontend_output_names, args.cached_frontend_out,
            args.opset)
        lines.extend([
            '',
            '=== Exported tensors ===',
            describe('image_frame', single_frame_input),
        ])
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(
                IMAGE_ENCODER_OUTPUT_NAMES, frame_outputs[0]))
        lines.extend([
            'image encoder IsInf rewritten: {}'.format(
                encoder_rewrite['isinf_rewritten']),
            'cached frontend IsInf rewritten: {}'.format(
                cached_rewrite['isinf_rewritten']),
            'image encoder ONNX: {}'.format(
                os.path.abspath(args.image_encoder_out)),
            'cached frontend ONNX: {}'.format(
                os.path.abspath(args.cached_frontend_out)),
            'fixture: {}'.format(fixture_path),
            'status: SUCCESS',
        ])
    except Exception as error:
        lines.extend([
            '',
            'status: FAILED',
            '{}: {}'.format(type(error).__name__, error),
            traceback.format_exc(),
        ])
        write_report(args.report, lines)
        raise
    finally:
        if fixture_data is not None:
            fixture_data.close()
    write_report(args.report, lines)


if __name__ == '__main__':
    main()
