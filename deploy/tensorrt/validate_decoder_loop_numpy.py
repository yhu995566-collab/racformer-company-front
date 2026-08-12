#!/usr/bin/env python3
"""Run one TensorRT decoder-layer engine six times on one CUDA stream."""

import argparse
import ctypes
import os

import numpy as np

from deploy.tensorrt.validate_engine_numpy import (
    CUDA_MEMCPY_DEVICE_TO_HOST,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    append_comparison_details,
    decode_detections,
    numpy_dtype,
    stats,
)


STATE_INPUTS = ('query_bbox', 'query_feat')
STATE_OUTPUTS = ('next_query_bbox', 'next_query_feat')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--atol', type=float, default=6e-3)
    parser.add_argument('--accept-decoded-match', action='store_true')
    parser.add_argument('--save-outputs')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Decoder loop validation report: {}'.format(path))


def recurrent_bbox_to_detection(bbox_preds, pc_range):
    theta = bbox_preds[..., 0:1]
    distance = bbox_preds[..., 1:2]
    x = distance * 65.0 * np.cos(theta * (2.0 * np.pi))
    y = distance * 65.0 * np.sin(theta * (2.0 * np.pi))
    x = np.clip(
        (x - pc_range[0]) / (pc_range[3] - pc_range[0]), 0.0, 1.0)
    y = np.clip(
        (y - pc_range[1]) / (pc_range[4] - pc_range[1]), 0.0, 1.0)
    xy_bbox = np.concatenate([x, y, bbox_preds[..., 2:]], axis=-1)
    x = xy_bbox[..., 0:1] * (
        pc_range[3] - pc_range[0]) + pc_range[0]
    y = xy_bbox[..., 1:2] * (
        pc_range[4] - pc_range[1]) + pc_range[1]
    z = xy_bbox[..., 2:3] * (
        pc_range[5] - pc_range[2]) + pc_range[2]
    return np.concatenate([
        x, y, xy_bbox[..., 3:5], z, xy_bbox[..., 5:10],
    ], axis=-1)


def main():
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')

    import tensorrt as trt

    lines = [
        '=== RaCFormer recurrent decoder TensorRT validation ===',
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
        d_regions = np.ascontiguousarray(
            fixture['decoder_d_regions'], dtype=np.float32)
        pc_range = np.asarray(fixture['decoder_pc_range'], dtype=np.float32)
        iterations = int(d_regions.size)
        if iterations <= 0:
            raise RuntimeError('decoder_d_regions is empty')
        stream = cuda.create_stream()

        io_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
        ]
        expected = set(STATE_INPUTS + STATE_OUTPUTS + (
            'd_region', 'cls_score'))
        missing = expected.difference(io_names)
        if missing:
            raise RuntimeError(
                'engine is missing recurrent tensors: {}'.format(
                    sorted(missing)))

        shared_inputs = {}
        for name in io_names:
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            if name in STATE_INPUTS or name == 'd_region':
                continue
            if name not in fixture:
                raise KeyError('fixture is missing input {}'.format(name))
            dtype = numpy_dtype(trt, engine.get_tensor_dtype(name))
            array = np.ascontiguousarray(fixture[name], dtype=dtype)
            if not context.set_input_shape(name, tuple(array.shape)):
                raise RuntimeError(
                    'invalid input shape {}: {}'.format(name, array.shape))
            shared_inputs[name] = array

        state_arrays = {}
        for name in STATE_INPUTS:
            dtype = numpy_dtype(trt, engine.get_tensor_dtype(name))
            state_arrays[name] = np.ascontiguousarray(
                fixture[name], dtype=dtype)
            if not context.set_input_shape(
                    name, tuple(state_arrays[name].shape)):
                raise RuntimeError(
                    'invalid input shape {}: {}'.format(
                        name, state_arrays[name].shape))
        d_region_dtype = numpy_dtype(
            trt, engine.get_tensor_dtype('d_region'))
        d_region_arrays = [
            np.ascontiguousarray([value], dtype=d_region_dtype)
            for value in d_regions
        ]
        if not context.set_input_shape(
                'd_region', tuple(d_region_arrays[0].shape)):
            raise RuntimeError('invalid d_region input shape')
        shape_inputs = context.infer_shapes()
        if shape_inputs:
            raise RuntimeError('shape inference needs: {}'.format(shape_inputs))

        def allocate(size):
            pointer = cuda.malloc(size)
            allocations.append(pointer)
            return pointer

        shared_pointers = {}
        for name, array in shared_inputs.items():
            pointer = allocate(array.nbytes)
            shared_pointers[name] = pointer
            cuda.check(cuda.lib.cudaMemcpyAsync(
                pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE, stream), 'shared input H2D')
            if not context.set_tensor_address(name, pointer.value):
                raise RuntimeError('failed to bind {}'.format(name))

        initial_pointers = {}
        for name, array in state_arrays.items():
            pointer = allocate(array.nbytes)
            initial_pointers[name] = pointer
            cuda.check(cuda.lib.cudaMemcpyAsync(
                pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE, stream), 'initial state H2D')
        d_region_pointers = []
        for array in d_region_arrays:
            pointer = allocate(array.nbytes)
            d_region_pointers.append(pointer)
            cuda.check(cuda.lib.cudaMemcpyAsync(
                pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE, stream), 'd_region H2D')

        output_shapes = {
            name: tuple(context.get_tensor_shape(name))
            for name in STATE_OUTPUTS + ('cls_score',)
        }
        for name, shape in output_shapes.items():
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(
                    'unresolved output shape {}: {}'.format(name, shape))
        output_dtypes = {
            name: numpy_dtype(trt, engine.get_tensor_dtype(name))
            for name in output_shapes
        }
        feature_buffers = [
            allocate(np.empty(
                output_shapes['next_query_feat'],
                dtype=output_dtypes['next_query_feat']).nbytes)
            for _ in range(2)
        ]
        bbox_buffers = [
            allocate(np.empty(
                output_shapes['next_query_bbox'],
                dtype=output_dtypes['next_query_bbox']).nbytes)
            for _ in range(iterations)
        ]
        cls_buffers = [
            allocate(np.empty(
                output_shapes['cls_score'],
                dtype=output_dtypes['cls_score']).nbytes)
            for _ in range(iterations)
        ]
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'input synchronization')

        def execute_loop():
            bbox_input = initial_pointers['query_bbox']
            feature_input = initial_pointers['query_feat']
            for index in range(iterations):
                feature_output = feature_buffers[index % 2]
                context.set_tensor_address('query_bbox', bbox_input.value)
                context.set_tensor_address('query_feat', feature_input.value)
                context.set_tensor_address(
                    'd_region', d_region_pointers[index].value)
                context.set_tensor_address(
                    'next_query_bbox', bbox_buffers[index].value)
                context.set_tensor_address(
                    'next_query_feat', feature_output.value)
                context.set_tensor_address(
                    'cls_score', cls_buffers[index].value)
                if not context.execute_async_v3(stream.value):
                    raise RuntimeError(
                        'TensorRT execution failed at decoder iteration '
                        '{}'.format(index))
                bbox_input = bbox_buffers[index]
                feature_input = feature_output

        for _ in range(args.warmup):
            execute_loop()
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'warmup synchronization')

        latencies = []
        for _ in range(args.iters):
            start = cuda.create_event()
            end = cuda.create_event()
            events.extend([start, end])
            cuda.check(cuda.lib.cudaEventRecord(start, stream), 'event record')
            execute_loop()
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

        cls_layers = []
        bbox_layers = []
        for index in range(iterations):
            cls = np.empty(
                output_shapes['cls_score'], dtype=output_dtypes['cls_score'])
            bbox = np.empty(
                output_shapes['next_query_bbox'],
                dtype=output_dtypes['next_query_bbox'])
            cuda.check(cuda.lib.cudaMemcpyAsync(
                ctypes.c_void_p(cls.ctypes.data), cls_buffers[index],
                cls.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream),
                'cls_score D2H')
            cuda.check(cuda.lib.cudaMemcpyAsync(
                ctypes.c_void_p(bbox.ctypes.data), bbox_buffers[index],
                bbox.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream),
                'next_query_bbox D2H')
            cls_layers.append(cls)
            bbox_layers.append(bbox)
        cuda.check(
            cuda.lib.cudaStreamSynchronize(stream), 'output synchronization')
        free_after, _ = cuda.memory_info()

        actual_outputs = {
            'all_cls_scores': np.stack(cls_layers),
            'all_bbox_preds': recurrent_bbox_to_detection(
                np.stack(bbox_layers), pc_range),
        }
        raw_passed = True
        lines.extend([
            'decoder iterations: {}'.format(iterations),
            'd_region schedule: {}'.format(d_regions.tolist()),
            '',
            '=== Numerical comparison ===',
        ])
        for name, actual in actual_outputs.items():
            reference = fixture[name]
            difference = np.abs(
                actual.astype(np.float64) - reference.astype(np.float64))
            close = np.allclose(
                actual, reference, rtol=0.0, atol=args.atol)
            raw_passed = raw_passed and close
            lines.append(
                '{}: shape={}, close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    name, actual.shape, close, difference.max(),
                    difference.mean()))
            append_comparison_details(
                lines, name, actual, reference, args.atol)

        actual_decoded = decode_detections(
            actual_outputs['all_cls_scores'],
            actual_outputs['all_bbox_preds'], pc_range)
        reference_decoded = decode_detections(
            fixture['all_cls_scores'], fixture['all_bbox_preds'], pc_range)
        actual_boxes, actual_scores, actual_labels = actual_decoded
        ref_boxes, ref_scores, ref_labels = reference_decoded
        boxes_match = actual_boxes.shape == ref_boxes.shape and np.allclose(
            actual_boxes, ref_boxes, rtol=0.0, atol=args.atol)
        scores_match = \
            actual_scores.shape == ref_scores.shape and np.allclose(
                actual_scores, ref_scores, rtol=0.0, atol=args.atol)
        labels_match = np.array_equal(actual_labels, ref_labels)
        decoded_passed = boxes_match and scores_match and labels_match
        lines.extend([
            '',
            '=== Decoded detection comparison ===',
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
            'decoded comparison passed: {}'.format(decoded_passed),
            'atol: {}'.format(args.atol),
            '',
            '=== Performance ===',
            'six-layer decoder GPU latency: {}'.format(stats(latencies)),
            'per-layer average latency: {:.3f} ms'.format(
                float(np.mean(latencies)) / iterations),
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
            args.accept_decoded_match and decoded_passed)
        lines.extend([
            '',
            '=== Acceptance ===',
            'raw tensor comparison passed: {}'.format(raw_passed),
            'decoded comparison passed: {}'.format(decoded_passed),
            'accept decoded match: {}'.format(args.accept_decoded_match),
            'deployment acceptance passed: {}'.format(accepted),
        ])
        if not accepted:
            raise RuntimeError(
                'recurrent decoder output comparison failed')
        lines.extend(['', 'status: SUCCESS'])
    except Exception as error:
        lines.extend([
            '',
            'status: FAILED',
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
