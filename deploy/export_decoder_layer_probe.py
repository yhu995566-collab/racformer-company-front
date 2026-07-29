#!/usr/bin/env python3
"""Export the first complete decoder layer as an ONNX probe."""

import argparse
import copy
import os
import traceback

import numpy as np
import onnx
import torch
from torch import nn

from deploy.export_onnx import (
    disable_gradient_checkpointing,
    enable_fixed_view_geometry,
    enable_single_batch_radar_scatter,
    enable_standard_onnx_fallbacks,
    install_export_symbolics,
)
from deploy.onnx_wrapper import INPUT_NAMES, RaCFormerONNXWrapper
from deploy.pytorch_runner import RaCFormerPyTorchRunner


FEATURE_INPUT_NAMES = [
    'image_feat_0',
    'image_feat_1',
    'image_feat_2',
    'image_feat_3',
]
PROBE_INPUT_NAMES = [
    'query_bbox',
    'query_feat',
] + FEATURE_INPUT_NAMES + [
    'lss_bev_feats',
    'radar_bev_feats',
    'lidar2img',
    'time_diff',
    'velocity_time_diff',
]
PROBE_OUTPUT_NAMES = ['decoder_query_output']


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


class DecoderLayerProbe(nn.Module):

    def __init__(self, decoder_layer, image_height, image_width):
        super().__init__()
        self.decoder_layer = copy.deepcopy(decoder_layer)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def forward(
            self, query_bbox, query_feat,
            image_feat_0, image_feat_1, image_feat_2, image_feat_3,
            lss_bev_feats, radar_bev_feats,
            lidar2img, time_diff, velocity_time_diff):
        image_shape = (
            self.image_height, self.image_width, 3)
        img_metas = [dict(
            img_shape=[image_shape] * 8,
            lidar2img=lidar2img,
            time_diff=time_diff,
            velocity_time_diff=velocity_time_diff)]
        image_feats = [
            image_feat_0,
            image_feat_1,
            image_feat_2,
            image_feat_3,
        ]
        query_output, _, _ = self.decoder_layer(
            query_bbox,
            query_feat,
            image_feats,
            lss_bev_feats,
            radar_bev_feats,
            None,
            img_metas,
            layer=0)
        return query_output


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Decoder layer probe export report: {}'.format(path))


def describe(name, tensor):
    return '{}: shape={}, dtype={}'.format(
        name, tuple(tensor.shape), tensor.dtype)


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer decoder layer ONNX probe ===',
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
        wrapper = RaCFormerONNXWrapper(
            runner.model, image_height, image_width).eval()
        decoder_layer = runner.model.pts_bbox_head.transformer.decoder \
            .decoder_layer
        captured = {}

        def capture_first(module, inputs):
            del module
            if captured:
                return
            image_feats = inputs[2]
            if len(image_feats) != len(FEATURE_INPUT_NAMES):
                raise RuntimeError(
                    'expected {} image feature levels, got {}'.format(
                        len(FEATURE_INPUT_NAMES), len(image_feats)))
            img_meta = inputs[6][0]
            captured.update({
                'query_bbox': inputs[0].detach().clone(),
                'query_feat': inputs[1].detach().clone(),
                'lss_bev_feats': inputs[3].detach().clone(),
                'radar_bev_feats': inputs[4].detach().clone(),
                'lidar2img': img_meta['lidar2img'].detach().clone(),
                'time_diff': img_meta['time_diff'].detach().clone(),
                'velocity_time_diff':
                    img_meta['velocity_time_diff'].detach().clone(),
            })
            for name, tensor in zip(FEATURE_INPUT_NAMES, image_feats):
                captured[name] = tensor.detach().clone()

        handle = decoder_layer.register_forward_pre_hook(capture_first)
        try:
            with torch.no_grad():
                wrapper(*model_inputs)
            torch.cuda.synchronize(runner.device)
        finally:
            handle.remove()
        missing = [name for name in PROBE_INPUT_NAMES if name not in captured]
        if missing:
            raise RuntimeError(
                'did not capture decoder layer inputs: {}'.format(missing))

        probe = DecoderLayerProbe(
            decoder_layer, image_height,
            image_width).to(runner.device).eval()
        probe_inputs = tuple(captured[name] for name in PROBE_INPUT_NAMES)
        with torch.no_grad():
            probe_output = probe(*probe_inputs)
        torch.cuda.synchronize(runner.device)

        lines.extend(['', '=== Probe tensors ==='])
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(PROBE_INPUT_NAMES, probe_inputs))
        lines.append(describe(PROBE_OUTPUT_NAMES[0], probe_output))

        fixture_path = os.path.abspath(args.fixture)
        os.makedirs(os.path.dirname(fixture_path) or '.', exist_ok=True)
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(PROBE_INPUT_NAMES, probe_inputs)
        }
        arrays[PROBE_OUTPUT_NAMES[0]] = \
            probe_output.detach().cpu().numpy()
        np.savez_compressed(fixture_path, **arrays)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        install_export_symbolics(args.opset, tensorrt_85_compat=True)
        torch.onnx.export(
            probe,
            probe_inputs,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=PROBE_INPUT_NAMES,
            output_names=PROBE_OUTPUT_NAMES,
            verbose=False)
        onnx.checker.check_model(onnx.load(output_path))
        lines.extend([
            '',
            'fixture: {}'.format(fixture_path),
            'onnx: {}'.format(output_path),
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
