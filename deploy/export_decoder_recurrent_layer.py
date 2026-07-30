#!/usr/bin/env python3
"""Export one reusable decoder layer for six GPU-resident iterations."""

import argparse
import copy
import os
import traceback

import numpy as np
import onnx
import torch
from torch import nn

from deploy.export_decoder_stack_probe import (
    DETECTION_OUTPUT_NAMES,
    FEATURE_INPUT_NAMES,
    PROBE_INPUT_NAMES as STACK_INPUT_NAMES,
    DecoderStackProbe,
)
from deploy.export_onnx import (
    disable_gradient_checkpointing,
    enable_fixed_view_geometry,
    enable_single_batch_radar_scatter,
    enable_standard_onnx_fallbacks,
    install_export_symbolics,
)
from deploy.onnx_wrapper import INPUT_NAMES, RaCFormerONNXWrapper
from deploy.pytorch_runner import RaCFormerPyTorchRunner
from models.bbox.utils import theta_d2xy_coods


RECURRENT_INPUT_NAMES = STACK_INPUT_NAMES + ['d_region']
RECURRENT_OUTPUT_NAMES = [
    'next_query_feat',
    'cls_score',
    'next_query_bbox',
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


class RecurrentDecoderLayer(nn.Module):

    def __init__(self, decoder_layer, image_height, image_width):
        super().__init__()
        self.decoder_layer = copy.deepcopy(decoder_layer)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def forward(
            self, query_bbox, query_feat,
            image_feat_0, image_feat_1, image_feat_2, image_feat_3,
            lss_bev_feats, radar_bev_feats,
            lidar2img, time_diff, velocity_time_diff, d_region):
        image_shape = (self.image_height, self.image_width, 3)
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
        return self.decoder_layer(
            query_bbox,
            query_feat,
            image_feats,
            lss_bev_feats,
            radar_bev_feats,
            None,
            img_metas,
            layer=0,
            d_region_override=d_region)


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Recurrent decoder layer export report: {}'.format(path))


def describe(name, tensor):
    return '{}: shape={}, dtype={}'.format(
        name, tuple(tensor.shape), tensor.dtype)


def to_detection_bbox(bbox_preds, pc_range):
    bbox_preds = theta_d2xy_coods(bbox_preds, pc_range)
    x = bbox_preds[..., 0:1] * (
        pc_range[3] - pc_range[0]) + pc_range[0]
    y = bbox_preds[..., 1:2] * (
        pc_range[4] - pc_range[1]) + pc_range[1]
    z = bbox_preds[..., 2:3] * (
        pc_range[5] - pc_range[2]) + pc_range[2]
    return torch.cat([
        x, y, bbox_preds[..., 3:5], z, bbox_preds[..., 5:10],
    ], dim=-1)


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer recurrent decoder layer ONNX export ===',
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
        decoder = runner.model.pts_bbox_head.transformer.decoder
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

        handle = decoder.decoder_layer.register_forward_pre_hook(
            capture_first)
        try:
            with torch.no_grad():
                wrapper(*model_inputs)
            torch.cuda.synchronize(runner.device)
        finally:
            handle.remove()
        missing = [
            name for name in STACK_INPUT_NAMES if name not in captured]
        if missing:
            raise RuntimeError(
                'did not capture decoder inputs: {}'.format(missing))

        recurrent = RecurrentDecoderLayer(
            decoder.decoder_layer, image_height,
            image_width).to(runner.device).eval()
        stack = DecoderStackProbe(
            decoder.decoder_layer, decoder.num_layers, decoder.pc_range,
            image_height, image_width,
            detection_outputs=True).to(runner.device).eval()
        shared_inputs = tuple(
            captured[name] for name in STACK_INPUT_NAMES)
        d_regions = torch.as_tensor(
            decoder.decoder_layer.d_region_list,
            dtype=shared_inputs[0].dtype, device=runner.device)
        export_inputs = shared_inputs + (d_regions[0:1],)

        with torch.no_grad():
            query_bbox = shared_inputs[0]
            query_feat = shared_inputs[1]
            recurrent_cls = []
            recurrent_bbox = []
            recurrent_outputs = None
            for d_region in d_regions:
                recurrent_outputs = recurrent(
                    query_bbox, query_feat, *shared_inputs[2:],
                    d_region.reshape(1))
                query_feat, cls_score, query_bbox = recurrent_outputs
                recurrent_cls.append(cls_score)
                recurrent_bbox.append(to_detection_bbox(
                    query_bbox, decoder.pc_range))
            loop_outputs = (
                torch.stack(recurrent_cls),
                torch.stack(recurrent_bbox),
            )
            reference_outputs = stack(*shared_inputs)
        torch.cuda.synchronize(runner.device)
        loop_errors = [
            float((actual - reference).abs().max())
            for actual, reference in zip(loop_outputs, reference_outputs)
        ]
        loop_close = [
            bool(torch.allclose(
                actual, reference, rtol=0.0, atol=6e-3))
            for actual, reference in zip(loop_outputs, reference_outputs)
        ]
        if not all(loop_close):
            raise RuntimeError(
                'recurrent PyTorch loop does not match decoder stack: '
                'close={}, max_abs_error={}'.format(
                    loop_close, loop_errors))

        lines.extend([
            'decoder iterations: {}'.format(decoder.num_layers),
            'd_region schedule: {}'.format(
                [float(value) for value in d_regions.cpu()]),
            'PyTorch recurrent loop close: {}'.format(loop_close),
            'PyTorch recurrent loop max abs error: {}'.format(loop_errors),
            '',
            '=== Recurrent engine tensors ===',
        ])
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(
                RECURRENT_INPUT_NAMES, export_inputs))
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(
                RECURRENT_OUTPUT_NAMES, recurrent_outputs))

        fixture_path = os.path.abspath(args.fixture)
        os.makedirs(os.path.dirname(fixture_path) or '.', exist_ok=True)
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(STACK_INPUT_NAMES, shared_inputs)
        }
        arrays['decoder_d_regions'] = d_regions.detach().cpu().numpy()
        arrays['decoder_pc_range'] = np.asarray(
            decoder.pc_range, dtype=np.float32)
        arrays.update({
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                DETECTION_OUTPUT_NAMES, reference_outputs)
        })
        np.savez_compressed(fixture_path, **arrays)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        install_export_symbolics(args.opset, tensorrt_85_compat=True)
        torch.onnx.export(
            recurrent,
            export_inputs,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=RECURRENT_INPUT_NAMES,
            output_names=RECURRENT_OUTPUT_NAMES,
            verbose=False)
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        graph_inputs = [value.name for value in model.graph.input]
        if 'd_region' not in graph_inputs:
            raise RuntimeError('d_region was folded out of the ONNX graph')
        lines.extend([
            '',
            'fixture: {}'.format(fixture_path),
            'onnx: {}'.format(output_path),
            'd_region ONNX input: True',
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
