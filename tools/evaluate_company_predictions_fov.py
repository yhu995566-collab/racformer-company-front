#!/usr/bin/env python3
"""Re-evaluate saved company predictions with a physical horizontal FOV."""

import argparse
import importlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

from fov_geometry import front_fov_mask


PROFILES = {
    'car_only': ('car',),
    'main3': ('car', 'truck', 'bicycle'),
    'main4': ('car', 'truck', 'bicycle', 'pedestrian'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--ann-file', required=True, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--horizontal-fov-deg', type=float, required=True)
    parser.add_argument('--profiles', nargs='+', choices=tuple(PROFILES),
                        default=('car_only', 'main3'))
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--nms-iou-threshold', type=float, default=0.2)
    parser.add_argument('--prediction-only', action='store_true',
                        help='Filter predictions but retain rectangular-ROI GT.')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def blind_counts(dataset, predictions, classes, fov, score_threshold,
                 nms_iou_threshold):
    class_to_id = {name: index for index, name in enumerate(dataset.CLASSES)}
    selected_ids = {class_to_id[name]: name for name in classes}
    counts = Counter()
    for frame_index, result in enumerate(predictions):
        boxes, scores, labels = dataset._result_fields(
            result, nms_iou_threshold)
        if len(boxes):
            inside = front_fov_mask(boxes.gravity_center[:, :2], fov) \
                .detach().cpu().numpy()
            for label, score, valid in zip(labels, scores, inside):
                if int(label) in selected_ids and score >= score_threshold:
                    key = 'inside_predictions' if valid else 'blind_predictions'
                    counts[(selected_ids[int(label)], key)] += 1
        gt_boxes, gt_labels = dataset._filtered_gt(frame_index, None)
        if len(gt_boxes):
            gt_inside = front_fov_mask(gt_boxes.gravity_center[:, :2], fov) \
                .detach().cpu().numpy()
            for label, valid in zip(gt_labels, gt_inside):
                if int(label) in selected_ids:
                    key = 'inside_gt' if valid else 'blind_gt'
                    counts[(selected_ids[int(label)], key)] += 1
    return {
        class_name: {
            key: counts[(class_name, key)]
            for key in ('inside_gt', 'blind_gt', 'inside_predictions',
                        'blind_predictions')}
        for class_name in classes}


def main():
    args = parse_args()
    importlib.import_module('models')
    importlib.import_module('loaders')
    config = Config.fromfile(args.config)
    dataset_config = config.data[args.split].copy()
    dataset_config['ann_file'] = str(args.ann_file.resolve())
    dataset_config['data_root'] = str(args.ann_file.resolve().parent) + '/'
    # Pass the FOV explicitly to evaluate so old rectangular configs work.
    dataset_config['horizontal_fov_deg'] = None
    dataset = build_dataset(dataset_config)
    with args.predictions.open('rb') as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(dataset):
        raise ValueError('prediction count {} != dataset count {}'.format(
            len(predictions), len(dataset)))

    report = {
        'horizontal_fov_deg': args.horizontal_fov_deg,
        'prediction_only': args.prediction_only,
        'score_threshold': args.score_threshold,
        'profiles': {},
    }
    all_classes = sorted(set().union(*(PROFILES[name] for name in args.profiles)))
    report['fov_counts'] = blind_counts(
        dataset, predictions, all_classes, args.horizontal_fov_deg,
        args.score_threshold, args.nms_iou_threshold)
    for profile in args.profiles:
        metrics = dataset.evaluate(
            predictions,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            eval_classes=PROFILES[profile],
            metric_prefix='company/{}/fov{:g}'.format(
                profile, args.horizontal_fov_deg),
            horizontal_fov_deg=args.horizontal_fov_deg,
            filter_gt_by_fov=not args.prediction_only)
        report['profiles'][profile] = {
            key: float(value) if isinstance(value, (float, int, np.number)) else value
            for key, value in metrics.items()}
        for key, value in metrics.items():
            print('{}: {:.4f}'.format(key, value)
                  if isinstance(value, (float, int, np.number))
                  else '{}: {}'.format(key, value))
    print('FOV counts:')
    print(json.dumps(report['fov_counts'], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('report:', args.output.resolve())


if __name__ == '__main__':
    main()
