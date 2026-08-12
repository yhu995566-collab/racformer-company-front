#!/usr/bin/env python3
"""Validate a TensorRT engine without PyTorch or OpenMMLab."""

import argparse
import ctypes
import os

import numpy as np


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--atol', type=float, default=6e-3)
    parser.add_argument('--save-outputs')
    parser.add_argument(
        '--skip-decode', action='store_true',
        help='Validate arbitrary tensor outputs without detection decoding')
    parser.add_argument(
        '--accept-decoded-match', action='store_true',
        help='Accept matching decoded detections when raw tensors exceed atol')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('TensorRT validation report: {}'.format(path))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return (
        'mean={:.3f} ms, p50={:.3f} ms, p95={:.3f} ms, '
        'min={:.3f} ms, max={:.3f} ms'
    ).format(
        values.mean(), np.percentile(values, 50),
        np.percentile(values, 95), values.min(), values.max())


class CudaRuntime:
    def __init__(self):
        errors = []
        for name in ('libcudart.so', 'libcudart.so.11.0'):
            try:
                self.lib = ctypes.CDLL(name)
                self.library_name = name
                break
            except OSError as error:
                errors.append('{}: {}'.format(name, error))
        else:
            raise RuntimeError(
                'could not load CUDA runtime: {}'.format('; '.join(errors)))

        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(void_p), size_t]
        self.lib.cudaFree.argtypes = [void_p]
        self.lib.cudaMemcpyAsync.argtypes = [
            void_p, void_p, size_t, ctypes.c_int, void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(void_p)]
        self.lib.cudaStreamDestroy.argtypes = [void_p]
        self.lib.cudaStreamSynchronize.argtypes = [void_p]
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(void_p)]
        self.lib.cudaEventDestroy.argtypes = [void_p]
        self.lib.cudaEventRecord.argtypes = [void_p, void_p]
        self.lib.cudaEventSynchronize.argtypes = [void_p]
        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float), void_p, void_p]
        self.lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(size_t), ctypes.POINTER(size_t)]

    def check(self, status, operation):
        if status:
            message = self.lib.cudaGetErrorString(status)
            message = message.decode() if message else 'unknown CUDA error'
            raise RuntimeError(
                '{} failed with CUDA error {}: {}'.format(
                    operation, status, message))

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.cudaMalloc(ctypes.byref(pointer), size), 'cudaMalloc')
        return pointer

    def free(self, pointer):
        if pointer and pointer.value:
            self.check(self.lib.cudaFree(pointer), 'cudaFree')

    def create_stream(self):
        stream = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreate(ctypes.byref(stream)),
            'cudaStreamCreate')
        return stream

    def create_event(self):
        event = ctypes.c_void_p()
        self.check(
            self.lib.cudaEventCreate(ctypes.byref(event)), 'cudaEventCreate')
        return event

    def memory_info(self):
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self.check(
            self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            'cudaMemGetInfo')
        return free.value, total.value


def numpy_dtype(trt, dtype):
    mapping = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32: np.int32,
        trt.int8: np.int8,
        trt.bool: np.bool_,
        trt.uint8: np.uint8,
    }
    if dtype not in mapping:
        raise TypeError('unsupported TensorRT dtype: {}'.format(dtype))
    return mapping[dtype]


def append_comparison_details(lines, name, actual, reference, atol):
    difference = np.abs(
        actual.astype(np.float64) - reference.astype(np.float64))
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


def decode_detections(cls_scores, bbox_preds, center_range=None):
    """NumPy equivalent of the company-front NMSFreeCoder."""
    cls_scores = cls_scores[-1, 0]
    bbox_preds = bbox_preds[-1, 0]
    probabilities = 1.0 / (1.0 + np.exp(-cls_scores))
    flat = probabilities.reshape(-1)
    max_num = min(300, flat.size)
    top_indices = np.argpartition(flat, -max_num)[-max_num:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    scores = flat[top_indices]
    labels = (top_indices % 10).astype(np.int64)
    selected = bbox_preds[top_indices // 10]

    boxes = np.concatenate([
        selected[:, 0:1],
        selected[:, 1:2],
        selected[:, 4:5],
        np.exp(selected[:, 2:3]),
        np.exp(selected[:, 3:4]),
        np.exp(selected[:, 5:6]),
        np.arctan2(selected[:, 6:7], selected[:, 7:8]),
        selected[:, 8:9],
        selected[:, 9:10],
    ], axis=-1)
    if center_range is None:
        center_range = [0.0, -20.0, -3.0, 200.0, 20.0, 3.0]
    center_range = np.asarray(center_range, dtype=boxes.dtype)
    if center_range.shape != (6,):
        raise ValueError(
            'detection center range must contain 6 values, got {}'.format(
                center_range.shape))
    mask = np.all(boxes[:, :3] >= center_range[:3], axis=1)
    mask &= np.all(boxes[:, :3] <= center_range[3:], axis=1)
    mask &= scores > 0.05
    boxes = boxes[mask]
    boxes[:, 2] -= boxes[:, 5] * 0.5
    return boxes, scores[mask], labels[mask]


def main():
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')

    import tensorrt as trt

    lines = [
        '=== RaCFormer lightweight TensorRT validation ===',
        'TensorRT version: {}'.format(trt.__version__),
        'engine: {}'.format(os.path.abspath(args.engine)),
        'fixture: {}'.format(os.path.abspath(args.fixture)),
        'PyTorch required: False',
    ]
    cuda = None
    stream = None
    events = []
    allocations = []
    fixture = None
    try:
        cuda = CudaRuntime()
        lines.append('CUDA runtime: {}'.format(cuda.library_name))
        for path in args.plugin:
            path = os.path.abspath(path)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            lines.append('loaded plugin: {}'.format(path))

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        free_before, total_memory = cuda.memory_info()
        with open(args.engine, 'rb') as file:
            engine = runtime.deserialize_cuda_engine(file.read())
        if engine is None:
            raise RuntimeError('failed to deserialize TensorRT engine')
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError('failed to create TensorRT execution context')
        fixture = np.load(args.fixture)
        stream = cuda.create_stream()

        host_tensors = {}
        output_names = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            dtype = numpy_dtype(trt, engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                if name not in fixture:
                    raise KeyError('fixture is missing input {}'.format(name))
                array = np.ascontiguousarray(fixture[name], dtype=dtype)
                if not context.set_input_shape(name, tuple(array.shape)):
                    raise RuntimeError(
                        'invalid input shape {}: {}'.format(name, array.shape))
                host_tensors[name] = array
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
            host_tensors[name] = np.empty(
                shape, dtype=numpy_dtype(trt, engine.get_tensor_dtype(name)))

        for name, array in host_tensors.items():
            pointer = cuda.malloc(array.nbytes)
            allocations.append(pointer)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cuda.check(cuda.lib.cudaMemcpyAsync(
                    pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE, stream), 'input H2D')
            if not context.set_tensor_address(name, pointer.value):
                raise RuntimeError('failed to bind {}'.format(name))
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'input synchronization')

        def execute():
            if not context.execute_async_v3(stream.value):
                raise RuntimeError('TensorRT execution failed')

        for _ in range(args.warmup):
            execute()
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'warmup synchronization')

        latencies = []
        for _ in range(args.iters):
            start = cuda.create_event()
            end = cuda.create_event()
            events.extend([start, end])
            cuda.check(cuda.lib.cudaEventRecord(start, stream), 'event record')
            execute()
            cuda.check(cuda.lib.cudaEventRecord(end, stream), 'event record')
            cuda.check(
                cuda.lib.cudaEventSynchronize(end), 'event synchronization')
            elapsed = ctypes.c_float()
            cuda.check(cuda.lib.cudaEventElapsedTime(
                ctypes.byref(elapsed), start, end), 'event elapsed time')
            latencies.append(elapsed.value)
            cuda.check(cuda.lib.cudaEventDestroy(start), 'cudaEventDestroy')
            cuda.check(cuda.lib.cudaEventDestroy(end), 'cudaEventDestroy')
            events = events[:-2]

        for name in output_names:
            array = host_tensors[name]
            index = list(host_tensors).index(name)
            pointer = allocations[index]
            cuda.check(cuda.lib.cudaMemcpyAsync(
                ctypes.c_void_p(array.ctypes.data), pointer, array.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST, stream), 'output D2H')
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'output synchronization')
        free_after, _ = cuda.memory_info()

        actual_outputs = {}
        raw_passed = True
        lines.extend(['', '=== Numerical comparison ==='])
        for name in output_names:
            if name not in fixture:
                raise KeyError('fixture is missing reference {}'.format(name))
            actual = host_tensors[name]
            reference = fixture[name]
            actual_outputs[name] = actual
            difference = np.abs(
                actual.astype(np.float64) - reference.astype(np.float64))
            close = np.allclose(actual, reference, rtol=0.0, atol=args.atol)
            raw_passed = raw_passed and close
            lines.append(
                '{}: shape={}, close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    name, actual.shape, close, difference.max(),
                    difference.mean()))
            append_comparison_details(
                lines, name, actual, reference, args.atol)

        decoded_passed = None
        if args.skip_decode:
            lines.extend([
                '', '=== Decoded detection comparison ===',
                'skipped: True',
            ])
        else:
            center_range = (
                fixture['decoder_pc_range']
                if 'decoder_pc_range' in fixture else None)
            actual_decoded = decode_detections(
                actual_outputs['all_cls_scores'],
                actual_outputs['all_bbox_preds'], center_range)
            reference_decoded = decode_detections(
                fixture['all_cls_scores'], fixture['all_bbox_preds'],
                center_range)
            actual_boxes, actual_scores, actual_labels = actual_decoded
            ref_boxes, ref_scores, ref_labels = reference_decoded
            boxes_match = \
                actual_boxes.shape == ref_boxes.shape and np.allclose(
                    actual_boxes, ref_boxes, rtol=0.0, atol=args.atol)
            scores_match = \
                actual_scores.shape == ref_scores.shape and np.allclose(
                    actual_scores, ref_scores, rtol=0.0, atol=args.atol)
            labels_match = np.array_equal(actual_labels, ref_labels)
            decoded_passed = boxes_match and scores_match and labels_match
            lines.extend([
                '', '=== Decoded detection comparison ===',
                'actual/reference detection count: {}/{}'.format(
                    len(actual_boxes), len(ref_boxes)),
                'boxes close: {}, max_abs_error={:.8f}'.format(
                    boxes_match,
                    np.abs(actual_boxes - ref_boxes).max()
                    if actual_boxes.shape == ref_boxes.shape
                    else float('inf')),
                'scores close: {}, max_abs_error={:.8f}'.format(
                    scores_match,
                    np.abs(actual_scores - ref_scores).max()
                    if actual_scores.shape == ref_scores.shape
                    else float('inf')),
                'labels equal: {}'.format(labels_match),
                'decoded comparison passed: {}'.format(decoded_passed),
            ])
        lines.extend([
            'atol: {}'.format(args.atol),
            '', '=== Performance ===',
            'engine GPU latency: {}'.format(stats(latencies)),
            'resident CUDA memory delta: {:.2f} MB'.format(
                max(0, free_before - free_after) / (1024 ** 2)),
            'device memory total: {:.2f} MB'.format(
                total_memory / (1024 ** 2)),
        ])
        if args.save_outputs:
            output_path = os.path.abspath(args.save_outputs)
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            np.savez_compressed(output_path, **actual_outputs)
            lines.append('actual outputs: {}'.format(output_path))

        accepted = raw_passed or (
            args.accept_decoded_match and decoded_passed is True)
        lines.extend([
            '', '=== Acceptance ===',
            'raw tensor comparison passed: {}'.format(raw_passed),
            'decoded comparison passed: {}'.format(decoded_passed),
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
    finally:
        if fixture is not None:
            fixture.close()
        if cuda is not None:
            for event in events:
                cuda.lib.cudaEventDestroy(event)
            for pointer in allocations:
                cuda.lib.cudaFree(pointer)
            if stream is not None:
                cuda.lib.cudaStreamDestroy(stream)
    write_report(args.out, lines)


if __name__ == '__main__':
    main()
