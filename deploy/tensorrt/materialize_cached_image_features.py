#!/usr/bin/env python3
"""Run the single-frame TensorRT encoder over a temporal image fixture."""

import argparse
import ctypes
import os

import numpy as np

from deploy.tensorrt.validate_engine_numpy import (
    CUDA_MEMCPY_DEVICE_TO_HOST,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    numpy_dtype,
)


OUTPUT_TO_CACHE = {
    'image_frame_fpn_0': 'cached_image_fpn_0',
    'image_frame_fpn_1': 'cached_image_fpn_1',
    'image_frame_fpn_2': 'cached_image_fpn_2',
    'image_frame_fpn_3': 'cached_image_fpn_3',
    'image_frame_lss': 'cached_image_lss',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--out-fixture', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--max-abs-error', type=float, default=0.5)
    parser.add_argument('--max-mean-error', type=float, default=0.02)
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Cached image materialization report: {}'.format(path))


def main():
    args = parse_args()
    import tensorrt as trt

    lines = [
        '=== TensorRT cached image feature materialization ===',
        'TensorRT version: {}'.format(trt.__version__),
        'engine: {}'.format(os.path.abspath(args.engine)),
        'fixture: {}'.format(os.path.abspath(args.fixture)),
        'output fixture: {}'.format(os.path.abspath(args.out_fixture)),
    ]
    cuda = None
    stream = None
    allocations = []
    fixture = None
    try:
        cuda = CudaRuntime()
        for path in args.plugin:
            path = os.path.abspath(path)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            lines.append('loaded plugin: {}'.format(path))
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with open(args.engine, 'rb') as file:
            engine = runtime.deserialize_cuda_engine(file.read())
        if engine is None:
            raise RuntimeError('failed to deserialize image encoder')
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError('failed to create image encoder context')
        fixture = np.load(args.fixture)
        if 'image' not in fixture:
            raise KeyError(
                'fixture is missing the four-frame image tensor; rerun the '
                'cached frontend exporter')
        images = np.asarray(fixture['image'])
        if images.ndim != 5:
            raise ValueError(
                'image tensor must have shape [B,T,C,H,W], got {}'.format(
                    images.shape))
        batch, frames = images.shape[:2]
        if batch != 1 or frames <= 0:
            raise ValueError(
                'expected batch-size-one temporal images, got {}'.format(
                    images.shape))

        io_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
        ]
        input_names = [
            name for name in io_names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            name for name in io_names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if input_names != ['image_frame']:
            raise RuntimeError(
                'unexpected image encoder inputs: {}'.format(input_names))
        if set(output_names) != set(OUTPUT_TO_CACHE):
            raise RuntimeError(
                'unexpected image encoder outputs: {}'.format(output_names))

        input_dtype = numpy_dtype(
            trt, engine.get_tensor_dtype('image_frame'))
        frame_input = np.ascontiguousarray(images[:, 0], dtype=input_dtype)
        if not context.set_input_shape('image_frame', frame_input.shape):
            raise RuntimeError(
                'failed to set image frame shape {}'.format(
                    frame_input.shape))
        missing = context.infer_shapes()
        if missing:
            raise RuntimeError('shape inference needs: {}'.format(missing))

        stream = cuda.create_stream()
        input_pointer = cuda.malloc(frame_input.nbytes)
        allocations.append(input_pointer)
        if not context.set_tensor_address(
                'image_frame', input_pointer.value):
            raise RuntimeError('failed to bind image_frame')

        host_outputs = {}
        output_pointers = {}
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            dtype = numpy_dtype(trt, engine.get_tensor_dtype(name))
            host_outputs[name] = np.empty(shape, dtype=dtype)
            pointer = cuda.malloc(host_outputs[name].nbytes)
            allocations.append(pointer)
            output_pointers[name] = pointer
            if not context.set_tensor_address(name, pointer.value):
                raise RuntimeError('failed to bind {}'.format(name))

        temporal_outputs = {name: [] for name in output_names}
        for frame_index in range(frames):
            frame_input = np.ascontiguousarray(
                images[:, frame_index], dtype=input_dtype)
            cuda.check(cuda.lib.cudaMemcpyAsync(
                input_pointer, ctypes.c_void_p(frame_input.ctypes.data),
                frame_input.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream),
                'image frame H2D')
            if not context.execute_async_v3(stream.value):
                raise RuntimeError(
                    'image encoder failed for frame {}'.format(frame_index))
            for name in output_names:
                output = host_outputs[name]
                cuda.check(cuda.lib.cudaMemcpyAsync(
                    ctypes.c_void_p(output.ctypes.data),
                    output_pointers[name], output.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST, stream),
                    '{} D2H'.format(name))
            cuda.check(
                cuda.lib.cudaStreamSynchronize(stream),
                'image frame synchronization')
            for name in output_names:
                temporal_outputs[name].append(host_outputs[name].copy())

        arrays = {name: fixture[name] for name in fixture.files}
        passed = True
        lines.extend(['', '=== TensorRT feature cache comparison ==='])
        for output_name, cache_name in OUTPUT_TO_CACHE.items():
            actual = np.ascontiguousarray(np.stack(
                temporal_outputs[output_name], axis=1))
            reference = np.asarray(fixture[cache_name])
            if actual.shape != reference.shape:
                raise RuntimeError(
                    '{} shape mismatch: {} vs {}'.format(
                        cache_name, actual.shape, reference.shape))
            difference = np.abs(
                actual.astype(np.float64) - reference.astype(np.float64))
            maximum = float(difference.max())
            mean = float(difference.mean())
            close = maximum <= args.max_abs_error and \
                mean <= args.max_mean_error
            passed &= close
            lines.append(
                '{}: shape={}, close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    cache_name, actual.shape, close, maximum, mean))
            arrays[cache_name] = actual
        lines.append('feature cache sanity: {}'.format(
            'PASS' if passed else 'FAIL'))
        if not passed:
            raise RuntimeError('TensorRT image feature cache sanity failed')

        output_path = os.path.abspath(args.out_fixture)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        lines.extend([
            'materialized frames: {}'.format(frames),
            'status: SUCCESS',
        ])
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
            for pointer in allocations:
                cuda.lib.cudaFree(pointer)
            if stream is not None:
                cuda.lib.cudaStreamDestroy(stream)
    write_report(args.out, lines)


if __name__ == '__main__':
    main()
