#!/usr/bin/env python3
"""Export the sensor frontend consumed by the recurrent decoder engine."""

import argparse
import os
import traceback

import numpy as np
import onnx
import torch
from torch import nn

from deploy.export_decoder_stack_probe import (
    DETECTION_OUTPUT_NAMES,
    FEATURE_INPUT_NAMES,
    DecoderStackProbe,
)
from deploy.export_onnx import (
    disable_gradient_checkpointing,
    enable_fixed_view_geometry,
    enable_single_batch_radar_scatter,
    enable_standard_onnx_fallbacks,
    install_export_symbolics,
)
from deploy.onnx_wrapper import INPUT_NAMES
from deploy.pytorch_runner import RaCFormerPyTorchRunner
from deploy.tensorrt.rewrite_trt85_onnx import (
    rewrite_trt85_unsupported_nodes,
)


FRONTEND_OUTPUT_NAMES = [
    'query_bbox',
    'query_feat',
] + FEATURE_INPUT_NAMES + [
    'lss_bev_feats',
    'radar_bev_feats',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--model-fixture', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


class RaCFormerFrontendONNXWrapper(nn.Module):

    def __init__(self, model, image_height, image_width):
        super().__init__()
        self.model = model
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        decoder = model.pts_bbox_head.transformer.decoder
        self.num_cams = int(decoder.num_cams)

    def forward(
            self, image, radar_depth, radar_rcs, lidar2img, img2lidar,
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
            raise ValueError(
                'expected voxels, num_points, and coors for 8 frames')
        radar_points = [
            tuple(radar_voxel_inputs[index:index + 3])
            for index in range(0, 24, 3)
        ]
        image_feats, lss_bev_feats, radar_bev_feats, _ = \
            self.model.extract_feat(
                img=image,
                radar_points=radar_points,
                radar_depth=radar_depth,
                radar_rcs=radar_rcs,
                img_metas=[img_meta])

        organized_image_feats = []
        for feature in image_feats:
            batch, frame_cams, grouped_channels, height, width = feature.shape
            frames = frame_cams // self.num_cams
            groups = 4
            channels = grouped_channels // groups
            feature = feature.reshape(
                batch, frames, self.num_cams, groups, channels,
                height, width)
            feature = feature.permute(0, 1, 3, 2, 5, 6, 4)
            feature = feature.reshape(
                batch * frames * groups, self.num_cams,
                height, width, channels)
            organized_image_feats.append(feature.contiguous())

        head = self.model.pts_bbox_head
        batch = lss_bev_feats.shape[0]
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

        return tuple([
            query_bbox,
            query_feat,
        ] + organized_image_feats + [
            lss_bev_feats,
            radar_bev_feats,
        ])


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Frontend ONNX export report: {}'.format(path))


def describe(name, tensor):
    return '{}: shape={}, dtype={}'.format(
        name, tuple(tensor.shape), tensor.dtype)


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer sensor frontend ONNX export ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'model fixture: {}'.format(os.path.abspath(args.model_fixture)),
        'device: {}'.format(args.device),
        'opset: {}'.format(args.opset),
    ]
    fixture_data = None
    try:
        runner = RaCFormerPyTorchRunner(
            args.config, args.weights, device=args.device)
        disable_gradient_checkpointing(runner.model)
        fixture_data = np.load(args.model_fixture)
        missing = [name for name in INPUT_NAMES if name not in fixture_data]
        if missing:
            raise KeyError('model fixture is missing inputs: {}'.format(
                missing))
        model_inputs = tuple(
            torch.from_numpy(np.ascontiguousarray(fixture_data[name])).to(
                runner.device)
            for name in INPUT_NAMES)

        enable_standard_onnx_fallbacks(
            runner.model, mixing_chunk_size=32768,
            use_msmv_plugin=True,
            use_single_camera_projection_plugin=True)
        enable_single_batch_radar_scatter(runner.model)
        enable_fixed_view_geometry(runner.model, model_inputs[4])
        runner.model._deploy_trt_static_radar_padding = True

        image_height = int(model_inputs[0].shape[-2])
        image_width = int(model_inputs[0].shape[-1])
        frontend = RaCFormerFrontendONNXWrapper(
            runner.model, image_height, image_width).eval()
        decoder = runner.model.pts_bbox_head.transformer.decoder
        decoder_stack = DecoderStackProbe(
            decoder.decoder_layer, decoder.num_layers, decoder.pc_range,
            image_height, image_width,
            detection_outputs=True).to(runner.device).eval()
        with torch.no_grad():
            frontend_outputs = frontend(*model_inputs)
            decoder_inputs = frontend_outputs + (
                model_inputs[3],
                model_inputs[6],
                model_inputs[7],
            )
            detection_outputs = decoder_stack(*decoder_inputs)
        torch.cuda.synchronize(runner.device)

        d_regions = np.asarray(
            decoder.decoder_layer.d_region_list, dtype=np.float32)
        pc_range = np.asarray(decoder.pc_range, dtype=np.float32)
        lines.extend([
            'decoder iterations: {}'.format(decoder.num_layers),
            'd_region schedule: {}'.format(d_regions.tolist()),
            '',
            '=== Frontend tensors ===',
        ])
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(INPUT_NAMES, model_inputs))
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(
                FRONTEND_OUTPUT_NAMES, frontend_outputs))

        fixture_path = os.path.abspath(args.fixture)
        os.makedirs(os.path.dirname(fixture_path) or '.', exist_ok=True)
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(INPUT_NAMES, model_inputs)
        }
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                FRONTEND_OUTPUT_NAMES, frontend_outputs)
        })
        arrays['decoder_d_regions'] = d_regions
        arrays['decoder_pc_range'] = pc_range
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                DETECTION_OUTPUT_NAMES, detection_outputs)
        })
        np.savez_compressed(fixture_path, **arrays)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        install_export_symbolics(args.opset, tensorrt_85_compat=True)
        torch.onnx.export(
            frontend,
            model_inputs,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=INPUT_NAMES,
            output_names=FRONTEND_OUTPUT_NAMES,
            verbose=False)
        rewrite_result = rewrite_trt85_unsupported_nodes(
            output_path, output_path)
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        if (rewrite_result['isinf_remaining'] or
                rewrite_result['layernorm_remaining']):
            raise RuntimeError(
                'TensorRT 8.5 unsupported operators remain after rewrite')
        graph_outputs = [value.name for value in model.graph.output]
        if graph_outputs != FRONTEND_OUTPUT_NAMES:
            raise RuntimeError(
                'unexpected frontend outputs: {}'.format(graph_outputs))
        lines.extend([
            '',
            '=== TensorRT 8.5 compatibility ===',
            'IsInf nodes rewritten: {}'.format(
                rewrite_result['isinf_rewritten']),
            'IsInf nodes remaining: {}'.format(
                rewrite_result['isinf_remaining']),
            'LayerNormalization nodes remaining: {}'.format(
                rewrite_result['layernorm_remaining']),
            'fixture: {}'.format(fixture_path),
            'onnx: {}'.format(output_path),
            'frontend output boundary: PASS',
            'onnx checker: PASS',
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
