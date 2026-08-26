#!/usr/bin/env python3
"""Summarize RaCFormer loss/validation trends and flag convergence states."""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


EPOCH_RE = re.compile(
    r'Epoch \[(?P<epoch>\d+)/(?:\d+)\]\[(?P<iteration>\d+)/(?P<total>\d+)\] '
    r'(?P<body>.*)')
VALUE_RE = re.compile(
    r'(?P<key>[A-Za-z0-9_.]+): '
    r'(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
METRIC_RE = re.compile(
    r'(?P<key>company/[A-Za-z0-9_./@-]+): '
    r'(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
DEFAULT_PRIMARY = 'company/car_3D_AP@0.5'
DISPLAY_METRICS = (
    'company/car_3D_AP@0.5',
    'company/car_BEV_AP@0.5',
    'company/3D_mAP@0.5',
    'company/BEV_mAP@0.5',
    'company/overall_recall@0.1',
    'company/range_0_50m_3D_mAP@0.5',
    'company/range_50_100m_3D_mAP@0.5',
    'company/range_100_150m_3D_mAP@0.5',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run', action='append', required=True, metavar='LABEL=LOG',
        help='Repeat for each run, for example 50m=/path/train.log')
    parser.add_argument('--primary', default=DEFAULT_PRIMARY)
    parser.add_argument(
        '--patience', type=int, default=3,
        help='Number of validation snapshots without a significant new best')
    parser.add_argument(
        '--min-delta', type=float, default=0.005,
        help='Absolute AP improvement considered significant')
    return parser.parse_args()


def parse_run(value):
    if '=' not in value:
        raise ValueError('--run must use LABEL=LOG syntax: {}'.format(value))
    label, path = value.split('=', 1)
    if not label or not path:
        raise ValueError('--run must use non-empty LABEL=LOG syntax')
    return label, Path(path)


def parse_log(path):
    losses = defaultdict(list)
    metrics = defaultdict(dict)
    current_epoch = None
    nonfinite_lines = []
    for line_number, line in enumerate(
            path.open(errors='replace'), start=1):
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group('epoch'))
            iteration = int(epoch_match.group('iteration'))
            total = int(epoch_match.group('total'))
            values = {
                match.group('key'): float(match.group('value'))
                for match in VALUE_RE.finditer(epoch_match.group('body'))
            }
            if iteration >= max(1, int(total * 0.8)):
                losses[current_epoch].append(values)
            lowered = line.lower()
            if 'nan' in lowered or 'inf' in lowered:
                nonfinite_lines.append(line_number)
            continue
        metric_match = METRIC_RE.search(line)
        if metric_match and current_epoch is not None:
            metrics[current_epoch][metric_match.group('key')] = float(
                metric_match.group('value'))
    return losses, metrics, nonfinite_lines


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float('nan')


def epoch_loss_means(losses):
    result = {}
    for epoch, records in losses.items():
        keys = set().union(*(record.keys() for record in records))
        result[epoch] = {
            key: mean([record[key] for record in records if key in record])
            for key in keys
        }
    return result


def slope(points):
    if len(points) < 2:
        return float('nan')
    x_mean = mean([point[0] for point in points])
    y_mean = mean([point[1] for point in points])
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in points) / denominator


def classify(eval_points, loss_points, patience, min_delta):
    if len(eval_points) < max(4, patience + 1):
        return 'NOT_ENOUGH_EVALS', '至少需要{}次验证'.format(
            max(4, patience + 1))
    running_best = -float('inf')
    last_significant_index = 0
    for index, (_, value) in enumerate(eval_points):
        if value > running_best + min_delta:
            running_best = value
            last_significant_index = index
        running_best = max(running_best, value)
    stale = len(eval_points) - 1 - last_significant_index
    best_epoch, best_value = max(eval_points, key=lambda item: item[1])
    last_epoch, last_value = eval_points[-1]
    recent = eval_points[-min(4, len(eval_points)):]
    metric_slope = slope(recent)
    degradation = best_value - last_value
    loss_slope = slope(loss_points[-min(4, len(loss_points)):])

    details = (
        'best=e{}:{:.4f}, last=e{}:{:.4f}, stale_evals={}, '
        'recent_metric_slope={:+.6f}/epoch, loss_slope={:+.6f}/epoch'.format(
            best_epoch, best_value, last_epoch, last_value, stale,
            metric_slope, loss_slope))
    if stale >= patience and degradation >= max(0.02, 2 * min_delta):
        return 'OVERFITTING_RISK', details
    if stale >= patience and abs(metric_slope) <= min_delta / 2.0:
        return 'CONVERGED_PLATEAU', details
    if stale < patience and metric_slope > 0:
        return 'STILL_IMPROVING', details
    return 'UNSTABLE_OR_INCONCLUSIVE', details


def metric_short_name(key):
    return key.replace('company/', '').replace('_mAP@0.5', '')


def report(label, path, primary, patience, min_delta):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    losses, metrics, nonfinite = parse_log(path)
    loss_means = epoch_loss_means(losses)
    eval_points = [
        (epoch, values[primary])
        for epoch, values in sorted(metrics.items()) if primary in values]
    loss_points = [
        (epoch, values['loss'])
        for epoch, values in sorted(loss_means.items()) if 'loss' in values]

    print('\n' + '=' * 88)
    print('{}: {}'.format(label, path))
    print('latest_training_epoch:', max(losses) if losses else 'NOT FOUND')
    print('validation_snapshots:', len(eval_points))
    print('nonfinite_training_lines:', len(nonfinite))
    if nonfinite:
        print('nonfinite_line_examples:', nonfinite[:10])

    print('\nRECENT LOSSES (mean over final 20% logged iterations per epoch)')
    print('epoch      loss  loss_cls loss_bbox loss_depth')
    for epoch, values in sorted(loss_means.items())[-6:]:
        print('{:>5} {:>9} {:>9} {:>9} {:>10}'.format(
            epoch,
            '{:.4f}'.format(values['loss']) if 'loss' in values else '-',
            '{:.4f}'.format(values['loss_cls'])
            if 'loss_cls' in values else '-',
            '{:.4f}'.format(values['loss_bbox'])
            if 'loss_bbox' in values else '-',
            '{:.4f}'.format(values['loss_depth'])
            if 'loss_depth' in values else '-'))

    available = [key for key in DISPLAY_METRICS
                 if any(key in values for values in metrics.values())]
    print('\nVALIDATION HISTORY')
    print('epoch ' + ' '.join(
        '{:>17}'.format(metric_short_name(key)) for key in available))
    for epoch, values in sorted(metrics.items()):
        if primary not in values:
            continue
        print('{:>5} '.format(epoch) + ' '.join(
            '{:>17}'.format(
                '{:.4f}'.format(values[key]) if key in values else '-')
            for key in available))

    if not eval_points:
        print('\nSTATUS: NO_PRIMARY_METRIC')
        print('missing primary metric:', primary)
        print('available examples:', sorted(
            set().union(*(value.keys() for value in metrics.values())))[:20])
        return
    status, details = classify(
        eval_points, loss_points, patience, min_delta)
    print('\nPRIMARY:', primary)
    print('STATUS:', status)
    print('DETAIL:', details)


def main():
    args = parse_args()
    if args.patience < 1 or args.min_delta <= 0:
        raise ValueError('--patience and --min-delta must be positive')
    for value in args.run:
        label, path = parse_run(value)
        report(label, path, args.primary, args.patience, args.min_delta)


if __name__ == '__main__':
    main()
