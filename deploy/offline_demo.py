#!/usr/bin/env python3
"""Run the deployment path on one company validation sample."""

import argparse
import importlib
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.input_schema import FrameInput
from deploy.postprocessing import parse_detection_result
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one offline sample through the deployment path')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-pkl')
    parser.add_argument('--out')
    return parser.parse_args()


def padded_entries(current, sweeps, num_frames):
    entries = [current]
    history = list(sweeps[:num_frames - 1])
    if not history:
        history = [current] * (num_frames - 1)
    elif len(history) < num_frames - 1:
        history.extend(
            [history[-1]] * (num_frames - 1 - len(history)))
    entries.extend(history)
    return entries


def load_radar(entry, reference_timestamp):
    path = entry['data_path']
    if path.endswith('.npy'):
        points = np.load(path)
    else:
        points = np.fromfile(path, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 7)

    if not entry.get('radar_in_ego', True):
        transform = np.asarray(entry['radar2ego'], dtype=np.float32)
        xyz1 = np.concatenate([
            points[:, :3], np.ones((len(points), 1), dtype=np.float32)
        ], axis=1)
        points[:, :3] = (xyz1 @ transform.T)[:, :3]
        velocity = np.concatenate([
            points[:, 4:6], np.zeros((len(points), 1), dtype=np.float32)
        ], axis=1)
        points[:, 4:6] = (velocity @ transform[:3, :3].T)[:, :2]

    points[:, 6] = (reference_timestamp - entry['timestamp']) / 1e6
    return points


def load_frames(dataset, sample_index, num_frames):
    info = dataset.get_data_info(sample_index)
    cameras = padded_entries(
        info['front_camera'], info.get('camera_sweeps', []), num_frames)
    radars = padded_entries(
        info['front_radar'], info.get('radar_sweeps', []), num_frames)
    reference_radar_time = radars[0]['timestamp']
    frames = []
    for camera, radar in zip(cameras, radars):
        frames.append(FrameInput(
            image=mmcv.imread(camera['data_path'], 'color'),
            radar_points=load_radar(radar, reference_radar_time),
            image_timestamp=camera['timestamp'] / 1e6,
            radar_timestamp=radar['timestamp'] / 1e6,
            lidar2img=np.asarray(camera['lidar2img'], dtype=np.float32),
            intrinsic=np.asarray(camera['cam_intrinsic'], dtype=np.float32),
            frame_id=camera['data_path']))
    return frames


def compare_reference(prediction, reference_path, sample_index):
    reference_results = mmcv.load(reference_path)
    reference = parse_detection_result([reference_results[sample_index]])
    same_shape = prediction.boxes_3d.shape == reference.boxes_3d.shape
    boxes_close = same_shape and np.allclose(
        prediction.boxes_3d, reference.boxes_3d, rtol=1e-4, atol=1e-4)
    scores_close = prediction.scores_3d.shape == reference.scores_3d.shape and \
        np.allclose(
            prediction.scores_3d, reference.scores_3d,
            rtol=1e-4, atol=1e-4)
    labels_equal = np.array_equal(
        prediction.labels_3d, reference.labels_3d)
    print('reference boxes close: {}'.format(boxes_close))
    print('reference scores close: {}'.format(scores_close))
    print('reference labels equal: {}'.format(labels_equal))
    if not (boxes_close and scores_close and labels_equal):
        raise RuntimeError('deployment output does not match reference PKL')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    dataset = build_dataset(cfg.data[args.split])
    transforms = getattr(getattr(dataset, 'pipeline', None), 'transforms', [])
    if transforms:
        raise RuntimeError(
            'deployment dataset pipeline must be empty; use the deploy config')
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError('sample index is outside the dataset')

    preprocessor = DeploymentPreprocessor(cfg)
    frames = load_frames(dataset, args.sample_index, preprocessor.num_frames)
    batch = preprocessor.prepare(frames)
    runner = RaCFormerPyTorchRunner(
        args.config, args.weights, device=args.device)
    prediction = runner.infer(runner.prepare(batch))

    print('boxes_3d shape: {}'.format(prediction.boxes_3d.shape))
    print('scores_3d shape: {}'.format(prediction.scores_3d.shape))
    print('labels_3d shape: {}'.format(prediction.labels_3d.shape))
    print('detection count: {}'.format(prediction.count))

    if args.reference_pkl:
        compare_reference(prediction, args.reference_pkl, args.sample_index)
    if args.out:
        output_path = os.path.abspath(args.out)
        mmcv.mkdir_or_exist(os.path.dirname(output_path))
        np.savez(
            output_path,
            boxes_3d=prediction.boxes_3d,
            scores_3d=prediction.scores_3d,
            labels_3d=prediction.labels_3d)
        print('prediction saved to {}'.format(output_path))


if __name__ == '__main__':
    main()
