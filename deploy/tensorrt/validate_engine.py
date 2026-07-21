#!/usr/bin/env python3
"""Validate and benchmark a TensorRT engine against an exported NPZ fixture."""

import argparse
import ctypes
import importlib
import os

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--atol', type=float, default=5e-3)
    parser.add_argument(
        '--decode-config',
        help='Optional model config used to compare final decoded detections')
    parser.add_argument(
        '--accept-decoded-match', action='store_true',
        help='Accept the engine when decoded boxes, scores, and labels match '
             'even if diagnostic raw tensors exceed atol')
    return parser.parse_args()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return 'mean={:.3f} ms, p50={:.3f} ms, p95={:.3f} ms, min={:.3f} ms, max={:.3f} ms'.format(
        values.mean(), np.percentile(values, 50), np.percentile(values, 95),
        values.min(), values.max())


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('TensorRT validation report: {}'.format(path))


def torch_dtype(trt, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    if dtype not in mapping:
        raise TypeError('unsupported TensorRT dtype: {}'.format(dtype))
    return mapping[dtype]


def append_comparison_details(lines, name, actual, reference, atol):
    difference = np.abs(actual - reference)
    flat = difference.reshape(-1)
    max_index = np.unravel_index(int(flat.argmax()), difference.shape)
    lines.extend([
        '{} error percentiles: p50={:.8f}, p95={:.8f}, p99={:.8f}, '
        'p99.9={:.8f}'.format(
            name, np.percentile(flat, 50), np.percentile(flat, 95),
            np.percentile(flat, 99), np.percentile(flat, 99.9)),
        '{} elements above atol: {}/{} ({:.6f}%)'.format(
            name, int((flat > atol).sum()), flat.size,
            100.0 * float((flat > atol).sum()) / flat.size),
        '{} max error index: {}, actual={:.8f}, reference={:.8f}'.format(
            name, max_index, float(actual[max_index]),
            float(reference[max_index])),
    ])
    if difference.ndim >= 1:
        for layer_index in range(difference.shape[0]):
            layer_error = difference[layer_index].reshape(-1)
            lines.append(
                '{} layer {}: max={:.8f}, mean={:.8f}, p99={:.8f}, '
                'above_atol={}/{}'.format(
                    name, layer_index, layer_error.max(), layer_error.mean(),
                    np.percentile(layer_error, 99),
                    int((layer_error > atol).sum()), layer_error.size))


def decode_detections(config_path, cls_scores, bbox_preds, device):
    from mmcv import Config
    from mmdet.core.bbox.builder import build_bbox_coder

    importlib.import_module('models')
    cfg = Config.fromfile(config_path)
    coder = build_bbox_coder(cfg.model.pts_bbox_head.bbox_coder)
    predictions = coder.decode({
        'all_cls_scores': torch.from_numpy(
            np.ascontiguousarray(cls_scores)).to(device=device),
        'all_bbox_preds': torch.from_numpy(
            np.ascontiguousarray(bbox_preds)).to(device=device),
    })[0]
    boxes = predictions['bboxes'].detach().cpu().numpy().copy()
    boxes[:, 2] -= boxes[:, 5] * 0.5
    return (
        boxes,
        predictions['scores'].detach().cpu().numpy(),
        predictions['labels'].detach().cpu().numpy())


def main():
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    import tensorrt as trt

    lines = [
        '=== RaCFormer TensorRT validation ===',
        'TensorRT version: {}'.format(trt.__version__),
        'engine: {}'.format(os.path.abspath(args.engine)),
        'fixture: {}'.format(os.path.abspath(args.fixture)),
        'device: {}'.format(args.device),
    ]
    try:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.synchronize(device)
        free_before, total_memory = torch.cuda.mem_get_info(device)
        lines.append('CUDA device: {}'.format(torch.cuda.get_device_name(device)))
        for path in args.plugin:
            path = os.path.abspath(path)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            lines.append('loaded plugin: {}'.format(path))
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with open(args.engine, 'rb') as stream:
            engine = runtime.deserialize_cuda_engine(stream.read())
        if engine is None:
            raise RuntimeError('failed to deserialize TensorRT engine')
        context = engine.create_execution_context()
        fixture = np.load(args.fixture)
        stream = torch.cuda.current_stream(device)

        tensors = {}
        output_names = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if name not in fixture:
                    raise KeyError('fixture is missing input {}'.format(name))
                array = np.ascontiguousarray(fixture[name])
                tensor = torch.from_numpy(array).to(device=device)
                expected_dtype = torch_dtype(trt, engine.get_tensor_dtype(name))
                if tensor.dtype != expected_dtype:
                    tensor = tensor.to(expected_dtype)
                if not context.set_input_shape(name, tuple(tensor.shape)):
                    raise RuntimeError('invalid input shape for {}'.format(name))
                tensors[name] = tensor.contiguous()
            else:
                output_names.append(name)

        missing = context.infer_shapes()
        if missing:
            raise RuntimeError('shape inference needs: {}'.format(missing))
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(
                    'unresolved output shape {}: {}'.format(name, shape))
            tensors[name] = torch.empty(
                shape, dtype=torch_dtype(trt, engine.get_tensor_dtype(name)),
                device=device)
        for name, tensor in tensors.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError('failed to bind {}'.format(name))

        def execute():
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')

        for _ in range(args.warmup):
            execute()
        stream.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            execute()
            end.record(stream)
            end.synchronize()
            latencies.append(start.elapsed_time(end))
        free_after, _ = torch.cuda.mem_get_info(device)

        lines.extend(['', '=== Numerical comparison ==='])
        comparison_passed = True
        decoded_comparison_passed = None
        actual_outputs = {}
        for name in output_names:
            if name not in fixture:
                raise KeyError('fixture is missing reference {}'.format(name))
            actual = tensors[name].detach().cpu().numpy()
            reference = fixture[name]
            actual_outputs[name] = actual
            difference = np.abs(actual - reference)
            close = np.allclose(actual, reference, rtol=0.0, atol=args.atol)
            comparison_passed = comparison_passed and close
            lines.append(
                '{}: shape={}, close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    name, actual.shape, close, difference.max(),
                    difference.mean()))
            append_comparison_details(
                lines, name, actual, reference, args.atol)
        if args.decode_config:
            required_outputs = {'all_cls_scores', 'all_bbox_preds'}
            if not required_outputs.issubset(actual_outputs):
                raise RuntimeError(
                    'decoded comparison requires {}'.format(
                        sorted(required_outputs)))
            actual_decoded = decode_detections(
                args.decode_config, actual_outputs['all_cls_scores'],
                actual_outputs['all_bbox_preds'], device)
            reference_decoded = decode_detections(
                args.decode_config, fixture['all_cls_scores'],
                fixture['all_bbox_preds'], device)
            actual_boxes, actual_scores, actual_labels = actual_decoded
            ref_boxes, ref_scores, ref_labels = reference_decoded
            boxes_match = actual_boxes.shape == ref_boxes.shape and np.allclose(
                actual_boxes, ref_boxes, rtol=0.0, atol=args.atol)
            scores_match = actual_scores.shape == ref_scores.shape and np.allclose(
                actual_scores, ref_scores, rtol=0.0, atol=args.atol)
            labels_match = np.array_equal(actual_labels, ref_labels)
            decoded_comparison_passed = (
                boxes_match and scores_match and labels_match)
            lines.extend([
                '', '=== Decoded detection comparison ===',
                'actual/reference detection count: {}/{}'.format(
                    len(actual_boxes), len(ref_boxes)),
                'boxes close: {}, max_abs_error={:.8f}'.format(
                    boxes_match,
                    np.abs(actual_boxes - ref_boxes).max()
                    if actual_boxes.shape == ref_boxes.shape else float('inf')),
                'scores close: {}, max_abs_error={:.8f}'.format(
                    scores_match,
                    np.abs(actual_scores - ref_scores).max()
                    if actual_scores.shape == ref_scores.shape else float('inf')),
                'labels equal: {}'.format(labels_match),
                'decoded comparison passed: {}'.format(
                    decoded_comparison_passed),
            ])
        lines.extend([
            'atol: {}'.format(args.atol),
            'comparison passed: {}'.format(comparison_passed),
            '', '=== Performance ===',
            'engine GPU latency: {}'.format(stats(latencies)),
            'resident CUDA memory delta: {:.2f} MB'.format(
                max(0, free_before - free_after) / (1024 ** 2)),
            'device memory total: {:.2f} MB'.format(
                total_memory / (1024 ** 2)),
            'PyTorch I/O max_memory_allocated: {:.2f} MB'.format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            'PyTorch I/O max_memory_reserved: {:.2f} MB'.format(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
        ])
        accepted = comparison_passed or (
            args.accept_decoded_match and decoded_comparison_passed is True)
        lines.extend([
            '', '=== Acceptance ===',
            'raw tensor comparison passed: {}'.format(comparison_passed),
            'decoded comparison passed: {}'.format(
                decoded_comparison_passed),
            'accept decoded match: {}'.format(args.accept_decoded_match),
            'deployment acceptance passed: {}'.format(accepted),
        ])
        if not accepted:
            raise RuntimeError('TensorRT output comparison failed')
        lines.extend(['', 'status: SUCCESS'])
    except Exception as error:
        lines.extend([
            '', 'status: FAILED',
            '{}: {}'.format(type(error).__name__, error),
        ])
        write_report(args.out, lines)
        raise
    write_report(args.out, lines)


if __name__ == '__main__':
    main()
