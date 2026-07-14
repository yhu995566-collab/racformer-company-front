#!/usr/bin/env python3
"""Profile the LiDAR-free deployment path on one offline sample."""

import argparse
import contextlib
import importlib
import os
import sys
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.offline_demo import load_frames
from deploy.postprocessing import parse_detection_result
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner


class Tee:
    def __init__(self, stream, output_file):
        self.stream = stream
        self.output_file = output_file

    def write(self, text):
        self.stream.write(text)
        self.output_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.output_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile the RaCFormer deployment path')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--cache-radar-temporal', action='store_true')
    parser.add_argument(
        '--cache-bev-value-projections', action='store_true')
    parser.add_argument(
        '--out', default='outputs/deploy_baseline/deploy_profile.txt')
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error('--sample-index must be non-negative')
    if args.warmup < 0:
        parser.error('--warmup must be non-negative')
    if args.iters <= 0:
        parser.error('--iters must be positive')
    return args


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        min=float(values.min()),
        max=float(values.max()))


def print_summary(name, values):
    stats = summary(values)
    print(
        '{}: mean={mean:.3f} ms, p50={p50:.3f} ms, '
        'p95={p95:.3f} ms, min={min:.3f} ms, max={max:.3f} ms'.format(
            name, **stats))


def append(timings, name, elapsed_ms):
    timings.setdefault(name, []).append(elapsed_ms)


def profile_iteration(runner, preprocessor, frames, start_event, end_event):
    timings = {}

    start = time.perf_counter()
    batch = preprocessor.prepare(frames)
    timings['preprocessing'] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    prepared = runner.prepare(batch)
    torch.cuda.synchronize()
    timings['move-to-GPU'] = (time.perf_counter() - start) * 1000.0

    start_event.record()
    raw_result = runner.infer_raw(prepared)
    end_event.record()
    torch.cuda.synchronize()
    timings['model forward (includes decode)'] = start_event.elapsed_time(
        end_event)

    start = time.perf_counter()
    prediction = parse_detection_result(raw_result)
    timings['output parsing / GPU-to-CPU'] = (
        time.perf_counter() - start) * 1000.0
    return timings, prediction


def print_mode(name, timings, allocated_mb, reserved_mb):
    print('\n=== {} ==='.format(name))
    for stage, values in timings.items():
        print_summary(stage, values)
    print('max_memory_allocated: {:.2f} MB'.format(allocated_mb))
    print('max_memory_reserved: {:.2f} MB'.format(reserved_mb))


def profile(args):
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')

    cfg = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    dataset = build_dataset(cfg.data[args.split])
    transforms = getattr(getattr(dataset, 'pipeline', None), 'transforms', [])
    if transforms:
        raise RuntimeError(
            'deployment dataset pipeline must be empty; use the deploy config')
    if args.sample_index >= len(dataset):
        raise IndexError('sample index is outside the dataset')

    preprocessor = DeploymentPreprocessor(cfg)
    runner = RaCFormerPyTorchRunner(
        args.config, args.weights, device=args.device,
        cache_radar_temporal=args.cache_radar_temporal,
        cache_bev_value_projections=args.cache_bev_value_projections)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print('=== RaCFormer deployment profile ===')
    print('CUDA_VISIBLE_DEVICES: {}'.format(
        os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')))
    print('CUDA device: {}'.format(torch.cuda.get_device_name(runner.device)))
    print('visible device index: {}'.format(runner.device))
    print('sample index: {}'.format(args.sample_index))
    print('warmup iterations: {}'.format(args.warmup))
    print('profile iterations per mode: {}'.format(args.iters))
    print('radar temporal cache: {}'.format(
        'enabled' if args.cache_radar_temporal else 'disabled'))
    print('BEV value projection caches: {}'.format(
        'enabled' if args.cache_bev_value_projections else 'disabled'))
    print('model forward includes get_bboxes() and bbox3d2result() decode')

    cached_frames = load_frames(
        dataset, args.sample_index, preprocessor.num_frames)
    for _ in range(args.warmup):
        profile_iteration(
            runner, preprocessor, cached_frames, start_event, end_event)
    torch.cuda.synchronize()

    # Mode A includes offline disk reads. It is useful for comparison with the
    # old dataset profile, but it is not the expected ROS callback latency.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    offline_timings = {}
    offline_prediction = None
    for _ in range(args.iters):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        frames = load_frames(
            dataset, args.sample_index, preprocessor.num_frames)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        stages, offline_prediction = profile_iteration(
            runner, preprocessor, frames, start_event, end_event)
        append(offline_timings, 'offline image/radar file loading', load_ms)
        for name, value in stages.items():
            append(offline_timings, name, value)
        append(
            offline_timings, 'offline end-to-end',
            (time.perf_counter() - total_start) * 1000.0)
    offline_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    offline_reserved = torch.cuda.max_memory_reserved() / (1024.0 ** 2)

    # Mode B represents frames already supplied by a future ROS synchronizer.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runtime_timings = {}
    runtime_prediction = None
    for _ in range(args.iters):
        total_start = time.perf_counter()
        stages, runtime_prediction = profile_iteration(
            runner, preprocessor, cached_frames, start_event, end_event)
        for name, value in stages.items():
            append(runtime_timings, name, value)
        append(
            runtime_timings, 'deployment runtime end-to-end',
            (time.perf_counter() - total_start) * 1000.0)
    runtime_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    runtime_reserved = torch.cuda.max_memory_reserved() / (1024.0 ** 2)

    # Mode C caches preprocessing and transfer, measuring the model floor.
    cached_batch = runner.prepare(preprocessor.prepare(cached_frames))
    for _ in range(args.warmup):
        runner.infer_raw(cached_batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cached_forward = []
    cached_wall = []
    cached_prediction = None
    for _ in range(args.iters):
        wall_start = time.perf_counter()
        start_event.record()
        raw_result = runner.infer_raw(cached_batch)
        end_event.record()
        torch.cuda.synchronize()
        cached_forward.append(start_event.elapsed_time(end_event))
        cached_wall.append((time.perf_counter() - wall_start) * 1000.0)
        cached_prediction = parse_detection_result(raw_result)
        del raw_result
    cached_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    cached_reserved = torch.cuda.max_memory_reserved() / (1024.0 ** 2)

    print_mode(
        'Mode A: offline files + deployment path', offline_timings,
        offline_allocated, offline_reserved)
    print_mode(
        'Mode B: deployment runtime (sensor frames already available)',
        runtime_timings, runtime_allocated, runtime_reserved)
    print('\n=== Mode C: prepared GPU batch ===')
    print_summary('cached GPU forward', cached_forward)
    print_summary('cached synchronized wall', cached_wall)
    print('max_memory_allocated: {:.2f} MB'.format(cached_allocated))
    print('max_memory_reserved: {:.2f} MB'.format(cached_reserved))

    prediction = cached_prediction or runtime_prediction or offline_prediction
    print('\n=== Output structure ===')
    print('boxes_3d shape: {}'.format(prediction.boxes_3d.shape))
    print('scores_3d shape: {}'.format(prediction.scores_3d.shape))
    print('labels_3d shape: {}'.format(prediction.labels_3d.shape))
    print('detection count: {}'.format(prediction.count))


def main():
    args = parse_args()
    output_path = os.path.abspath(args.out)
    mmcv.mkdir_or_exist(os.path.dirname(output_path))
    with open(output_path, 'w', buffering=1) as output_file:
        stdout_tee = Tee(sys.stdout, output_file)
        stderr_tee = Tee(sys.stderr, output_file)
        with contextlib.redirect_stdout(stdout_tee), \
                contextlib.redirect_stderr(stderr_tee):
            print('Profiling report: {}'.format(output_path))
            profile(args)
            print('\nProfiling completed successfully')


if __name__ == '__main__':
    main()
