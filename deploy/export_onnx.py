#!/usr/bin/env python3
"""Export one fixed-shape RaCFormer sample and retain failure diagnostics."""

import argparse
import copy
import importlib
import os
import sys
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.offline_demo import load_frames
from deploy.onnx_wrapper import (
    INPUT_NAMES, OUTPUT_NAMES, RaCFormerONNXWrapper, build_export_inputs)
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export fixed-input FP32 RaCFormer raw outputs to ONNX')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--boundary-atol', type=float, default=5e-3)
    parser.add_argument(
        '--strict-boundary-check', action='store_true',
        help='Fail when two full PyTorch forwards exceed boundary-atol')
    parser.add_argument(
        '--fallthrough', action='store_true',
        help='Preserve unsupported operators for graph auditing')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


def describe_tensor(name, tensor):
    return '{}: shape={}, dtype={}, device={}'.format(
        name, tuple(tensor.shape), tensor.dtype, tensor.device)


def write_report(path, lines):
    path = os.path.abspath(path)
    mmcv.mkdir_or_exist(os.path.dirname(path))
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('Export report: {}'.format(path))


def legacy_raw_outputs(model, batch):
    """Run the original NumPy-metadata path up to the detector head."""
    img_meta = copy.deepcopy(batch.img_meta)
    radar_points = [[points] for points in batch.radar_points]
    img_feats, bev_feats, radar_bev_feats, _ = model.extract_feat(
        img=batch.image,
        radar_points=radar_points,
        radar_depth=batch.radar_depth,
        radar_rcs=batch.radar_rcs,
        img_metas=[img_meta])
    outputs = model.pts_bbox_head(
        img_feats, bev_feats, radar_bev_feats, [img_meta])
    return outputs['all_cls_scores'], outputs['all_bbox_preds']


def disable_gradient_checkpointing(model):
    """Disable training-only recomputation that the legacy exporter cannot trace."""
    disabled = []
    for name, module in model.named_modules():
        if getattr(module, 'with_cp', False):
            module.with_cp = False
            disabled.append(name or '<root>')
    return disabled


def main():
    args = parse_args()
    report = [
        '=== RaCFormer FP32 ONNX export ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'sample index: {}'.format(args.sample_index),
        'opset: {}'.format(args.opset),
        'operator mode: {}'.format(
            'ONNX_FALLTHROUGH' if args.fallthrough else 'ONNX'),
        'output boundary: raw all_cls_scores + all_bbox_preds (decode excluded)',
    ]
    try:
        cfg = Config.fromfile(args.config)
        importlib.import_module('models')
        importlib.import_module('loaders')
        dataset = build_dataset(cfg.data[args.split])
        if args.sample_index < 0 or args.sample_index >= len(dataset):
            raise IndexError('sample index is outside the dataset')

        preprocessor = DeploymentPreprocessor(cfg)
        frames = load_frames(dataset, args.sample_index, preprocessor.num_frames)
        cpu_batch = preprocessor.prepare(frames)
        runner = RaCFormerPyTorchRunner(
            args.config, args.weights, device=args.device)
        checkpoint_modules = disable_gradient_checkpointing(runner.model)
        report.extend([
            '', '=== Export preparation ===',
            'disabled gradient-checkpoint modules: {}'.format(
                len(checkpoint_modules)),
        ])
        report.extend(
            'checkpoint disabled: {}'.format(name)
            for name in checkpoint_modules)
        batch = runner.prepare(cpu_batch)
        wrapper = RaCFormerONNXWrapper(
            runner.model, preprocessor.final_height,
            preprocessor.final_width).eval()
        inputs = build_export_inputs(batch)

        report.extend(['', '=== Inputs ==='])
        report.extend(
            describe_tensor(name, tensor)
            for name, tensor in zip(INPUT_NAMES, inputs))

        with torch.no_grad():
            legacy_outputs = legacy_raw_outputs(runner.model, batch)
            outputs = wrapper(*inputs)
        torch.cuda.synchronize(runner.device)
        report.extend(['', '=== PyTorch raw outputs ==='])
        report.extend(
            describe_tensor(name, tensor)
            for name, tensor in zip(OUTPUT_NAMES, outputs))
        report.extend(['', '=== Tensor metadata boundary check ==='])
        boundary_passed = True
        for name, legacy, current in zip(
                OUTPUT_NAMES, legacy_outputs, outputs):
            difference = (legacy - current).abs()
            close = torch.allclose(
                legacy, current, rtol=0.0, atol=args.boundary_atol)
            boundary_passed = boundary_passed and close
            report.append(
                '{}: close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    name, close, difference.max().item(),
                    difference.mean().item()))
        report.append('boundary atol: {}'.format(args.boundary_atol))
        report.append('boundary comparison passed: {}'.format(
            boundary_passed))
        if not boundary_passed and args.strict_boundary_check:
            raise RuntimeError(
                'tensor metadata boundary does not match the legacy path')
        if not boundary_passed:
            report.append(
                'warning: continuing because radar voxelization and custom '
                'CUDA kernels can vary across independent full forwards')

        output_path = os.path.abspath(args.out)
        mmcv.mkdir_or_exist(os.path.dirname(output_path))
        operator_type = torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH \
            if args.fallthrough else torch.onnx.OperatorExportTypes.ONNX
        dynamic_axes = {
            'radar_points_{}'.format(index): {0: 'radar_points_{}_count'.format(index)}
            for index in range(8)
        }
        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            operator_export_type=operator_type,
            verbose=False)
        report.extend([
            '', '=== Export result ===', 'status: SUCCESS',
            'onnx: {}'.format(output_path),
            'next: python -m deploy.tensorrt.audit_onnx --onnx {} --out {}.audit.txt'.format(
                output_path, output_path),
        ])
    except Exception as error:
        report.extend([
            '', '=== Export result ===', 'status: FAILED',
            'exception: {}: {}'.format(type(error).__name__, error),
            '', '=== Traceback ===', traceback.format_exc(),
            'A failed standard export is an expected audit result when the '
            'graph reaches an unsupported custom CUDA operator.',
        ])
        write_report(args.report, report)
        raise

    write_report(args.report, report)


if __name__ == '__main__':
    main()
