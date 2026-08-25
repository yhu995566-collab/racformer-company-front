#!/usr/bin/env python3
"""Load regular and empty-LiDAR samples through a company train pipeline."""

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def unwrap(value):
    return getattr(value, 'data', value)


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    dataset = build_dataset(config.data.train)

    regular_indices = []
    empty_index = None
    for index, info in enumerate(dataset.data_infos):
        path = Path(dataset._resolve_path(info['lidar_path']))
        points = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(points) == 0 and empty_index is None:
            empty_index = index
        elif len(points) > 0 and len(regular_indices) < 8:
            regular_indices.append(index)
        if len(regular_indices) >= 8 and empty_index is not None:
            break
    if not regular_indices or empty_index is None:
        raise RuntimeError(
            'training set must contain regular and empty-LiDAR samples')

    regular = None
    for index in regular_indices:
        sample = dataset[index]
        if sample is None:
            continue
        depth = unwrap(sample['gt_depth'])
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth)
        if int(torch.count_nonzero(depth).item()) > 0:
            regular = (index, sample, depth)
            break
    if regular is None:
        raise RuntimeError(
            'none of the regular LiDAR samples produced depth supervision')

    empty_sample = dataset[empty_index]
    checks = [('regular', regular[0], regular[1], regular[2]),
              ('empty', empty_index, empty_sample, None)]
    for label, index, sample, known_depth in checks:
        if sample is None:
            raise RuntimeError('{} sample {} was filtered'.format(label, index))
        required = {'img', 'gt_depth', 'radar_points', 'radar_depth',
                    'radar_rcs', 'gt_bboxes_3d', 'gt_labels_3d'}
        missing = required - set(sample)
        if missing:
            raise RuntimeError(
                '{} sample missing keys {}'.format(label, sorted(missing)))
        depth = known_depth if known_depth is not None else unwrap(
            sample['gt_depth'])
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth)
        nonzero_depth = int(torch.count_nonzero(depth).item())
        if label == 'empty' and nonzero_depth != 0:
            raise RuntimeError('empty-LiDAR sample produced nonzero gt_depth')
        if label == 'regular' and nonzero_depth == 0:
            raise RuntimeError('regular sample produced empty gt_depth')
        print('{} index={} token={} gt_depth_shape={} nonzero_depth={}'.format(
            label, index, dataset.data_infos[index]['token'],
            tuple(depth.shape), nonzero_depth))
    print('company training data smoke: PASS')


if __name__ == '__main__':
    main()
