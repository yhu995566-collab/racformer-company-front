#!/usr/bin/env python3
"""Break down cached RaCFormer forward latency with CUDA events."""

import argparse
import contextlib
import functools
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


class EventRecorder:
    """Collect nested CUDA event pairs and aggregate them per iteration."""

    def __init__(self):
        self.active = False
        self.current = []
        self.totals = {}
        self.calls = {}
        self.indexed_calls = {}

    def begin(self):
        self.current = []
        self.active = True

    def add(self, label, start_event, end_event):
        self.current.append((label, start_event, end_event))

    def finish(self):
        self.active = False
        grouped = {}
        for label, start_event, end_event in self.current:
            elapsed = start_event.elapsed_time(end_event)
            grouped.setdefault(label, []).append(elapsed)
            self.calls.setdefault(label, []).append(elapsed)
        for label, values in grouped.items():
            self.totals.setdefault(label, []).append(sum(values))
            for index, value in enumerate(values):
                key = '{} [{}]'.format(label, index)
                self.indexed_calls.setdefault(key, []).append(value)


class MethodPatches:
    """Temporarily wrap methods and restore every original on exit."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.originals = []
        self.keys = set()

    def wrap(self, obj, attribute, label):
        key = (id(obj), attribute)
        if key in self.keys:
            return
        self.keys.add(key)
        original = getattr(obj, attribute)
        recorder = self.recorder

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if not recorder.active:
                return original(*args, **kwargs)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                return original(*args, **kwargs)
            finally:
                end_event.record()
                recorder.add(label, start_event, end_event)

        self.originals.append((obj, attribute, original))
        setattr(obj, attribute, measured)

    def restore(self):
        for obj, attribute, original in reversed(self.originals):
            setattr(obj, attribute, original)
        self.originals = []
        self.keys = set()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Break down RaCFormer model forward latency')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--cache-radar-temporal', action='store_true')
    parser.add_argument(
        '--out', default='outputs/deploy_baseline/forward_profile.txt')
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error('--sample-index must be non-negative')
    if args.warmup < 0:
        parser.error('--warmup must be non-negative')
    if args.iters <= 0:
        parser.error('--iters must be positive')
    return args


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        min=float(values.min()),
        max=float(values.max()))


def print_stats(label, values):
    result = stats(values)
    print(
        '{}: mean={mean:.3f} ms, p50={p50:.3f} ms, '
        'p95={p95:.3f} ms, min={min:.3f} ms, max={max:.3f} ms'.format(
            label, **result))


def residual(parent, children):
    size = len(parent)
    output = []
    for index in range(size):
        child_total = sum(child[index] for child in children)
        output.append(max(0.0, parent[index] - child_total))
    return output


def values(recorder, label):
    return recorder.totals.get(label, [])


def install_patches(model, recorder):
    patches = MethodPatches(recorder)
    view = model.img_lss_view_transformer
    head = model.pts_bbox_head
    transformer = head.transformer
    decoder = transformer.decoder
    layer = decoder.decoder_layer
    radar_sampling = layer.sampling_radar_bev
    lss_sampling = layer.sampling_lss_bev

    patches.wrap(model, 'extract_feat', 'extract_feat total')
    patches.wrap(model, 'extract_img_feat', 'image feature extraction')
    patches.wrap(model.img_backbone, 'forward', 'image backbone')
    patches.wrap(model.img_neck, 'forward', 'image FPN')
    patches.wrap(model.img_lss_neck, 'forward', 'image LSS neck')
    patches.wrap(view, 'get_mlp_input', 'calibration MLP input')
    patches.wrap(view, 'forward', 'LSS view transform per frame')

    patches.wrap(model, 'extract_pts_feat', 'radar branch per frame')
    patches.wrap(model, 'radar_voxelize', 'radar voxelization per frame')
    patches.wrap(
        model.radar_voxel_encoder, 'forward',
        'radar pillar encoder per frame')
    patches.wrap(
        model.radar_middle_encoder, 'forward',
        'radar scatter per frame')
    patches.wrap(
        model.radar_bev_conv, 'forward', 'radar BEV conv per frame')

    patches.wrap(head, 'forward', 'detection head forward')
    patches.wrap(transformer, 'forward', 'RaCFormer transformer')
    patches.wrap(decoder, 'forward', 'transformer decoder')
    patches.wrap(layer, 'forward', 'decoder layer')
    patches.wrap(layer.self_attn, 'forward', 'decoder self-attention')
    patches.wrap(
        layer.sampling, 'forward', 'decoder camera sampling')
    patches.wrap(
        radar_sampling, 'forward',
        'decoder radar-BEV sampling')
    patches.wrap(
        lss_sampling, 'forward',
        'decoder LSS-BEV sampling')

    patches.wrap(
        radar_sampling.temporal_encoder, 'forward',
        'radar-BEV temporal encoder')
    patches.wrap(
        radar_sampling.temporal_encoder.downsample, 'forward',
        'radar temporal downsample')
    patches.wrap(
        radar_sampling.temporal_encoder.convGRU, 'forward',
        'radar temporal ConvGRU')
    patches.wrap(
        radar_sampling.temporal_encoder.upsample, 'forward',
        'radar temporal upsample')
    patches.wrap(
        radar_sampling.temporal_encoder.temporal_fusion, 'forward',
        'radar temporal fusion')

    for sampling, prefix in (
            (radar_sampling, 'radar-BEV'),
            (lss_sampling, 'LSS-BEV')):
        patches.wrap(
            sampling.ray_points_offset, 'forward',
            '{} ray offset linear'.format(prefix))
        patches.wrap(
            sampling.sampling_offset, 'forward',
            '{} sampling offset linear'.format(prefix))
        patches.wrap(
            sampling.scale_weights, 'forward',
            '{} scale weight linear'.format(prefix))
        patches.wrap(
            sampling.positional_encoding, 'forward',
            '{} positional encoding'.format(prefix))
        patches.wrap(
            sampling.attention, 'forward',
            '{} deformable attention total'.format(prefix))
        patches.wrap(
            sampling.attention.value_proj, 'forward',
            '{} attention value projection'.format(prefix))
        patches.wrap(
            sampling.attention.bev_queue_weight, 'forward',
            '{} attention queue weight'.format(prefix))
        patches.wrap(
            sampling.attention.output_proj, 'forward',
            '{} attention output projection'.format(prefix))
    patches.wrap(layer.mixing, 'forward', 'decoder adaptive mixing')
    patches.wrap(layer.ffn, 'forward', 'decoder FFN')
    patches.wrap(head, 'get_bboxes', 'get_bboxes / decode')
    patches.wrap(head.bbox_coder, 'decode', 'bbox coder decode')
    return patches


def print_indexed(recorder, label, expected_count, unit_name):
    print('\n{} indexed breakdown:'.format(label))
    for index in range(expected_count):
        key = '{} [{}]'.format(label, index)
        if key in recorder.indexed_calls:
            print_stats('{} {}'.format(unit_name, index),
                        recorder.indexed_calls[key])


def print_breakdown(recorder, full_times, instrumented_times,
                    num_frames, num_decoder_layers):
    print('\n=== Uninstrumented model baseline ===')
    print_stats('cached model forward', full_times)
    print('\n=== Instrumented model total ===')
    print_stats('instrumented model forward', instrumented_times)

    extract_feat = values(recorder, 'extract_feat total')
    head = values(recorder, 'detection head forward')
    decode = values(recorder, 'get_bboxes / decode')
    print('\n=== Top-level forward breakdown ===')
    print_stats('extract_feat total', extract_feat)
    print_stats('detection head forward', head)
    print_stats('get_bboxes / decode', decode)
    print_stats(
        'top-level orchestration residual',
        residual(instrumented_times, [extract_feat, head, decode]))

    image = values(recorder, 'image feature extraction')
    calibration = values(recorder, 'calibration MLP input')
    radar = values(recorder, 'radar branch per frame')
    view = values(recorder, 'LSS view transform per frame')
    print('\n=== extract_feat breakdown ===')
    print_stats('image feature extraction', image)
    image_components = [
        values(recorder, 'image backbone'),
        values(recorder, 'image FPN'),
        values(recorder, 'image LSS neck'),
    ]
    print_stats('  image backbone', image_components[0])
    print_stats('  image FPN', image_components[1])
    print_stats('  image LSS neck', image_components[2])
    print_stats(
        '  image feature residual', residual(image, image_components))
    print_stats('calibration MLP input', calibration)
    print_stats('radar branch, {}-frame total'.format(num_frames), radar)
    print_stats(
        'LSS view transform, {}-frame total'.format(num_frames), view)
    print_stats(
        'extract_feat residual',
        residual(extract_feat, [image, calibration, radar, view]))
    print_indexed(
        recorder, 'radar branch per frame', num_frames, 'radar frame')
    print_indexed(
        recorder, 'LSS view transform per frame', num_frames, 'LSS frame')

    print('\n=== Radar branch nested breakdown, {}-frame totals ==='.format(
        num_frames))
    radar_components = [
        values(recorder, 'radar voxelization per frame'),
        values(recorder, 'radar pillar encoder per frame'),
        values(recorder, 'radar scatter per frame'),
        values(recorder, 'radar BEV conv per frame'),
    ]
    print_stats('radar voxelization', radar_components[0])
    print_stats('radar pillar encoder', radar_components[1])
    print_stats('radar scatter', radar_components[2])
    print_stats('radar BEV conv', radar_components[3])
    print_stats('radar branch residual', residual(radar, radar_components))

    transformer = values(recorder, 'RaCFormer transformer')
    decoder = values(recorder, 'transformer decoder')
    layers = values(recorder, 'decoder layer')
    components = [
        values(recorder, 'decoder self-attention'),
        values(recorder, 'decoder camera sampling'),
        values(recorder, 'decoder radar-BEV sampling'),
        values(recorder, 'decoder LSS-BEV sampling'),
        values(recorder, 'decoder adaptive mixing'),
        values(recorder, 'decoder FFN'),
    ]
    print('\n=== Detection head / Transformer breakdown ===')
    print_stats('RaCFormer transformer', transformer)
    print_stats('transformer decoder', decoder)
    print_stats(
        '{} decoder layers total'.format(num_decoder_layers), layers)
    print_stats('  self-attention total', components[0])
    print_stats('  camera sampling total', components[1])
    print_stats('  radar-BEV sampling total', components[2])
    print_stats('  LSS-BEV sampling total', components[3])
    print_stats('  adaptive mixing total', components[4])
    print_stats('  FFN total', components[5])
    print_stats('  decoder-layer residual', residual(layers, components))
    print_stats(
        'head residual outside transformer', residual(head, [transformer]))
    print_stats('bbox coder decode', values(recorder, 'bbox coder decode'))
    print_indexed(
        recorder, 'decoder layer', num_decoder_layers, 'decoder layer')

    print('\n=== Radar-BEV sampling nested breakdown ===')
    radar_sampling = values(recorder, 'decoder radar-BEV sampling')
    radar_sampling_components = [
        values(recorder, 'radar-BEV temporal encoder'),
        values(recorder, 'radar-BEV ray offset linear'),
        values(recorder, 'radar-BEV sampling offset linear'),
        values(recorder, 'radar-BEV scale weight linear'),
        values(recorder, 'radar-BEV positional encoding'),
        values(recorder, 'radar-BEV deformable attention total'),
    ]
    labels = (
        'temporal encoder',
        'ray offset linear',
        'sampling offset linear',
        'scale weight linear',
        'positional encoding',
        'deformable attention total',
    )
    for label, component in zip(labels, radar_sampling_components):
        print_stats(label, component)
    print_stats(
        'sampling coordinate / tensor operations residual',
        residual(radar_sampling, radar_sampling_components))

    temporal = radar_sampling_components[0]
    temporal_components = [
        values(recorder, 'radar temporal downsample'),
        values(recorder, 'radar temporal ConvGRU'),
        values(recorder, 'radar temporal upsample'),
        values(recorder, 'radar temporal fusion'),
    ]
    print('\nRadar temporal encoder nested breakdown:')
    for label, component in zip(
            ('downsample', 'ConvGRU', 'upsample', 'temporal fusion'),
            temporal_components):
        print_stats(label, component)
    print_stats(
        'temporal encoder residual', residual(temporal, temporal_components))

    print('\nRadar deformable attention nested breakdown:')
    print_attention_breakdown(recorder, 'radar-BEV')

    print('\n=== LSS-BEV sampling nested breakdown ===')
    lss_sampling = values(recorder, 'decoder LSS-BEV sampling')
    lss_sampling_components = [
        values(recorder, 'LSS-BEV ray offset linear'),
        values(recorder, 'LSS-BEV sampling offset linear'),
        values(recorder, 'LSS-BEV scale weight linear'),
        values(recorder, 'LSS-BEV positional encoding'),
        values(recorder, 'LSS-BEV deformable attention total'),
    ]
    for label, component in zip(labels[1:], lss_sampling_components):
        print_stats(label, component)
    print_stats(
        'sampling coordinate / tensor operations residual',
        residual(lss_sampling, lss_sampling_components))
    print('\nLSS deformable attention nested breakdown:')
    print_attention_breakdown(recorder, 'LSS-BEV')


def print_attention_breakdown(recorder, prefix):
    attention = values(
        recorder, '{} deformable attention total'.format(prefix))
    components = [
        values(recorder, '{} attention value projection'.format(prefix)),
        values(recorder, '{} attention queue weight'.format(prefix)),
        values(recorder, '{} attention output projection'.format(prefix)),
    ]
    for label, component in zip(
            ('value projection', 'queue weight', 'output projection'),
            components):
        print_stats(label, component)
    print_stats(
        'CUDA deformable attention / tensor operations residual',
        residual(attention, components))


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
        cache_radar_temporal=args.cache_radar_temporal)
    frames = load_frames(dataset, args.sample_index, preprocessor.num_frames)
    batch = runner.prepare(preprocessor.prepare(frames))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print('=== RaCFormer forward submodule profile ===')
    print('CUDA_VISIBLE_DEVICES: {}'.format(
        os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')))
    print('CUDA device: {}'.format(torch.cuda.get_device_name(runner.device)))
    print('visible device index: {}'.format(runner.device))
    print('sample index: {}'.format(args.sample_index))
    print('warmup iterations: {}'.format(args.warmup))
    print('profile iterations: {}'.format(args.iters))
    print('execution context: runner.infer_raw() with torch.no_grad()')
    print('radar temporal cache: {}'.format(
        'enabled' if args.cache_radar_temporal else 'disabled'))

    for _ in range(args.warmup):
        runner.infer_raw(batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    full_times = []
    result = None
    for _ in range(args.iters):
        start_event.record()
        result = runner.infer_raw(batch)
        end_event.record()
        torch.cuda.synchronize()
        full_times.append(start_event.elapsed_time(end_event))
    baseline_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    baseline_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024.0 ** 2))

    recorder = EventRecorder()
    patches = install_patches(runner.model, recorder)
    instrumented_times = []
    try:
        for _ in range(args.warmup):
            runner.infer_raw(batch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(args.iters):
            recorder.begin()
            start_event.record()
            result = runner.infer_raw(batch)
            end_event.record()
            torch.cuda.synchronize()
            instrumented_times.append(
                start_event.elapsed_time(end_event))
            recorder.finish()
    finally:
        patches.restore()

    print_breakdown(
        recorder, full_times, instrumented_times,
        num_frames=batch.image.shape[1],
        num_decoder_layers=runner.model.pts_bbox_head.transformer.decoder.num_layers)
    instrumented_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    instrumented_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024.0 ** 2))
    print('\n=== Memory and output ===')
    print('Memory peaks exclude model/checkpoint loading and warmup.')
    print('baseline max_memory_allocated: {:.2f} MB'.format(
        baseline_allocated_mb))
    print('baseline max_memory_reserved: {:.2f} MB'.format(
        baseline_reserved_mb))
    print('instrumented max_memory_allocated: {:.2f} MB'.format(
        instrumented_allocated_mb))
    print('instrumented max_memory_reserved: {:.2f} MB'.format(
        instrumented_reserved_mb))
    prediction = parse_detection_result(result)
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
            start = time.perf_counter()
            profile(args)
            print('\nProfiler wall time: {:.1f} s'.format(
                time.perf_counter() - start))
            print('Profiling completed successfully')


if __name__ == '__main__':
    main()
