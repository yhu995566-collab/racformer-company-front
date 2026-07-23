#!/usr/bin/env python3
"""Compare the legacy and TensorRT 8.5 single-batch radar scatter paths."""

import argparse
import importlib
import os

import mmcv
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.export_onnx import single_batch_radar_scatter
from deploy.offline_demo import load_frames
from deploy.onnx_wrapper import build_export_inputs
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    dataset = build_dataset(cfg.data[args.split])
    preprocessor = DeploymentPreprocessor(cfg)
    frames = load_frames(dataset, args.sample_index, preprocessor.num_frames)
    cpu_batch = preprocessor.prepare(frames)
    runner = RaCFormerPyTorchRunner(
        args.config, args.weights, device=args.device)
    batch = runner.prepare(cpu_batch)
    inputs = build_export_inputs(batch, runner.model)

    model = runner.model.eval()
    height, width = model.radar_output_shape
    channels = model.radar_middle_channels
    lines = [
        '=== Radar scatter equivalence check ===',
        'sample index: {}'.format(args.sample_index),
        'shape: channels={}, height={}, width={}'.format(
            channels, height, width),
        'atol: {}'.format(args.atol),
    ]
    passed = True
    with torch.no_grad():
        for frame_index in range(8):
            offset = 8 + frame_index * 3
            voxels, num_points, coors = inputs[offset:offset + 3]
            features = model.radar_voxel_encoder(
                voxels, num_points, coors).to(torch.float32)
            if features.dim() == 3 and features.shape[1] == 1:
                features = features.squeeze(1)
            legacy = model.radar_middle_encoder(features, coors, 1)
            deployment = single_batch_radar_scatter(
                features, coors, height, width, channels)
            difference = (legacy - deployment).abs()
            close = torch.allclose(
                legacy, deployment, rtol=0.0, atol=args.atol)
            spatial_coors = coors[:, -2:]
            unique_count = torch.unique(
                spatial_coors, dim=0).shape[0]
            duplicate_count = spatial_coors.shape[0] - unique_count
            passed = passed and close and duplicate_count == 0
            lines.append(
                'frame {}: voxels={}, unique={}, duplicates={}, close={}, '
                'max_abs_error={:.8f}, mean_abs_error={:.8f}'.format(
                    frame_index, spatial_coors.shape[0], unique_count,
                    duplicate_count, close, difference.max().item(),
                    difference.mean().item()))
    torch.cuda.synchronize(runner.device)
    lines.append('scatter equivalence passed: {}'.format(passed))

    output_path = os.path.abspath(args.out)
    mmcv.mkdir_or_exist(os.path.dirname(output_path))
    with open(output_path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Radar scatter report: {}'.format(output_path))
    if not passed:
        raise RuntimeError('single-batch radar scatter is not equivalent')


if __name__ == '__main__':
    main()
