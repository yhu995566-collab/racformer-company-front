#!/usr/bin/env python3
"""Export the first decoder LSS attention as a standalone ONNX probe."""

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
)
from deploy.onnx_wrapper import INPUT_NAMES, RaCFormerONNXWrapper
from deploy.pytorch_runner import RaCFormerPyTorchRunner


PROBE_INPUT_NAMES = [
    'query',
    'value',
    'sampling_points',
    'attention_weights',
]
PROBE_OUTPUT_NAMES = ['attention_output']


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


class LSSAttentionProbe(nn.Module):

    def __init__(self, attention, spatial_shape):
        super().__init__()
        self.attention = copy.deepcopy(attention)
        self.attention._deploy_onnx_fallback = True
        self.spatial_shape = tuple(int(value) for value in spatial_shape)

    def forward(self, query, value, sampling_points, attention_weights):
        return self.attention(
            query,
            value,
            sampling_points,
            attention_weights,
            spatial_shapes=self.spatial_shape)


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('LSS attention probe export report: {}'.format(path))


def describe(name, tensor):
    return '{}: shape={}, dtype={}'.format(
        name, tuple(tensor.shape), tensor.dtype)


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer LSS attention ONNX probe ===',
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

        wrapper = RaCFormerONNXWrapper(
            runner.model,
            image_height=int(model_inputs[0].shape[-2]),
            image_width=int(model_inputs[0].shape[-1])).eval()
        decoder_layer = runner.model.pts_bbox_head.transformer.decoder \
            .decoder_layer
        attention = decoder_layer.sampling_lss_bev.attention
        captured = {}

        def capture_first(module, inputs):
            del module
            if captured:
                return
            for name, tensor in zip(PROBE_INPUT_NAMES, inputs[:4]):
                if tensor is None:
                    raise RuntimeError(
                        'LSS attention input {} is None'.format(name))
                captured[name] = tensor.detach().clone()

        handle = attention.register_forward_pre_hook(capture_first)
        try:
            with torch.no_grad():
                wrapper(*model_inputs)
            torch.cuda.synchronize(runner.device)
        finally:
            handle.remove()
        if list(captured) != PROBE_INPUT_NAMES:
            raise RuntimeError(
                'did not capture all LSS attention inputs: {}'.format(
                    list(captured)))

        value = captured['value']
        spatial_shape = value.shape[-2:]
        probe = LSSAttentionProbe(
            attention, spatial_shape).to(runner.device).eval()
        probe_inputs = tuple(captured[name] for name in PROBE_INPUT_NAMES)
        with torch.no_grad():
            probe_output = probe(*probe_inputs)
        torch.cuda.synchronize(runner.device)

        lines.extend(['', '=== Probe tensors ==='])
        lines.extend(
            describe(name, tensor)
            for name, tensor in zip(PROBE_INPUT_NAMES, probe_inputs))
        lines.append(describe(PROBE_OUTPUT_NAMES[0], probe_output))
        lines.append(
            'spatial shape: {}'.format(tuple(int(v) for v in spatial_shape)))

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
