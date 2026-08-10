#!/usr/bin/env python3
"""Audit per-frame radar voxel counts for a deployment dataset split."""

import argparse
import importlib
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.offline_demo import load_frames
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        default='val')
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--progress-interval', type=int, default=100)
    parser.add_argument('--fail-if-exceeds', action='store_true')
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    mmcv.mkdir_or_exist(os.path.dirname(path))
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Radar voxel capacity report: {}'.format(path))


def main():
    args = parse_args()
    if args.capacity <= 0:
        raise ValueError('capacity must be positive')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('max-samples must be positive')
    if args.progress_interval <= 0:
        raise ValueError('progress-interval must be positive')

    cfg = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    dataset = build_dataset(cfg.data[args.split])
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)

    preprocessor = DeploymentPreprocessor(cfg)
    runner = RaCFormerPyTorchRunner(
        args.config, args.weights, device=args.device)
    counts = []
    exceedances = []

    with torch.no_grad():
        for sample_index in range(sample_count):
            frames = load_frames(
                dataset, sample_index, preprocessor.num_frames)
            batch = preprocessor.prepare(frames)
            for frame_index, cpu_points in enumerate(batch.radar_points):
                points = cpu_points.to(runner.device).clone()
                points[:, 2] = 0
                voxels, _, _ = runner.model.radar_voxel_layer(points)
                count = int(voxels.shape[0])
                counts.append(count)
                if count > args.capacity:
                    exceedances.append((sample_index, frame_index, count))
            if ((sample_index + 1) % args.progress_interval == 0 or
                    sample_index + 1 == sample_count):
                print('audited {}/{} samples'.format(
                    sample_index + 1, sample_count), flush=True)

    values = np.asarray(counts, dtype=np.int64)
    if values.size == 0:
        raise RuntimeError('radar voxel audit produced no frame observations')
    percentiles = (50, 90, 95, 99, 99.9, 100)
    lines = [
        '=== Radar voxel capacity audit ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'split: {}'.format(args.split),
        'samples: {}'.format(sample_count),
        'frames per sample: {}'.format(preprocessor.num_frames),
        'frame observations: {}'.format(values.size),
        'tested capacity: {}'.format(args.capacity),
        'minimum voxels: {}'.format(int(values.min())),
        'mean voxels: {:.3f}'.format(float(values.mean())),
    ]
    lines.extend(
        'p{} voxels: {:.3f}'.format(percentile, value)
        for percentile, value in zip(
            percentiles, np.percentile(values, percentiles)))
    lines.extend([
        'frames exceeding capacity: {}'.format(len(exceedances)),
        'capacity passed: {}'.format(not exceedances),
    ])
    if exceedances:
        lines.append('first exceedances (sample, frame, voxels): {}'.format(
            exceedances[:20]))
    lines.append('status: {}'.format(
        'PASS' if not exceedances else 'EXCEEDED'))
    write_report(args.out, lines)

    if exceedances and args.fail_if_exceeds:
        raise RuntimeError(
            '{} frame observations exceed capacity {}'.format(
                len(exceedances), args.capacity))


if __name__ == '__main__':
    main()
