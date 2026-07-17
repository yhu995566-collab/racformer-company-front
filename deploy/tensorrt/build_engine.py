#!/usr/bin/env python3
"""Build a fixed-batch FP32 RaCFormer TensorRT engine."""

import argparse
import ctypes
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--min-voxels', type=int, default=1)
    parser.add_argument('--opt-voxels', type=int, default=1024)
    parser.add_argument('--max-voxels', type=int, default=4096)
    parser.add_argument('--workspace-gb', type=float, default=8.0)
    parser.add_argument(
        '--builder-optimization-level', type=int, choices=range(6),
        help='TensorRT builder optimization level (0-5)')
    parser.add_argument(
        '--fusion-break-before', action='append', default=[],
        help='Mark input 0 of matching TensorRT layers as an engine output')
    parser.add_argument(
        '--disable-cudnn-tactics', action='store_true',
        help='Exclude cuDNN tactics when the loaded cuDNN does not match TRT')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('TensorRT build report: {}'.format(path))


def main():
    args = parse_args()
    if not (0 < args.min_voxels <= args.opt_voxels <= args.max_voxels):
        raise ValueError('voxel profile must satisfy 0 < min <= opt <= max')

    import tensorrt as trt

    class ReportLogger(trt.ILogger):
        def __init__(self, report, minimum_severity):
            super().__init__()
            self.report = report
            self.minimum_severity = minimum_severity

        def log(self, severity, message):
            if severity > self.minimum_severity:
                return
            entry = '[TRT] [{}] {}'.format(severity, message)
            self.report.append(entry)
            print(entry, flush=True)

    lines = [
        '=== RaCFormer FP32 TensorRT engine build ===',
        'TensorRT version: {}'.format(trt.__version__),
        'onnx: {}'.format(os.path.abspath(args.onnx)),
        'engine: {}'.format(os.path.abspath(args.engine)),
        'radar voxel profile: min={}, opt={}, max={}'.format(
            args.min_voxels, args.opt_voxels, args.max_voxels),
        'workspace: {:.2f} GB'.format(args.workspace_gb),
    ]
    try:
        for path in args.plugin:
            path = os.path.abspath(path)
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            lines.append('loaded plugin: {}'.format(path))

        logger = ReportLogger(lines, trt.ILogger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        explicit_batch = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(os.path.abspath(args.onnx)):
            for index in range(parser.num_errors):
                lines.append('parser error {}: {}'.format(
                    index, parser.get_error(index)))
            raise RuntimeError('TensorRT could not parse the ONNX graph')

        for pattern in args.fusion_break_before:
            matches = []
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                if pattern not in layer.name:
                    continue
                tensor = layer.get_input(0)
                if tensor is None:
                    raise RuntimeError(
                        'fusion-break layer has no input 0: {}'.format(
                            layer.name))
                if not tensor.is_network_output:
                    network.mark_output(tensor)
                matches.append((index, layer.name, tensor.name,
                                tuple(tensor.shape)))
            if not matches:
                raise RuntimeError(
                    'fusion-break pattern matched no layers: {}'.format(
                        pattern))
            lines.append('fusion break before {!r}: {}'.format(
                pattern, matches))

        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        if args.builder_optimization_level is not None:
            if not hasattr(config, 'builder_optimization_level'):
                raise RuntimeError(
                    'this TensorRT version does not expose '
                    'builder_optimization_level')
            config.builder_optimization_level = args.builder_optimization_level
            lines.append('builder optimization level: {}'.format(
                args.builder_optimization_level))
        else:
            lines.append('builder optimization level: TensorRT default')
        if args.disable_cudnn_tactics:
            tactic_sources = config.get_tactic_sources()
            tactic_sources &= ~(1 << int(trt.TacticSource.CUDNN))
            config.set_tactic_sources(tactic_sources)
            lines.append('cuDNN tactics: disabled')
        else:
            lines.append('cuDNN tactics: enabled')
        workspace_bytes = int(args.workspace_gb * (1024 ** 3))
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        profile = builder.create_optimization_profile()
        dynamic_inputs = 0
        lines.extend(['', '=== Optimization profile ==='])
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shape = tuple(tensor.shape)
            if -1 not in shape:
                continue
            if not tensor.name.startswith('radar_') or shape[0] != -1:
                raise RuntimeError(
                    'unsupported dynamic input {}: {}'.format(
                        tensor.name, shape))
            min_shape = (args.min_voxels,) + shape[1:]
            opt_shape = (args.opt_voxels,) + shape[1:]
            max_shape = (args.max_voxels,) + shape[1:]
            profile.set_shape(
                tensor.name, min_shape, opt_shape, max_shape)
            dynamic_inputs += 1
            lines.append('{}: {} / {} / {}'.format(
                tensor.name, min_shape, opt_shape, max_shape))
        if dynamic_inputs != 24:
            raise RuntimeError(
                'expected 24 dynamic radar inputs, found {}'.format(
                    dynamic_inputs))
        if not profile:
            raise RuntimeError('TensorRT rejected the optimization profile')
        config.add_optimization_profile(profile)

        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - start
        if serialized is None:
            raise RuntimeError('TensorRT engine build returned None')
        engine_path = os.path.abspath(args.engine)
        os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
        with open(engine_path, 'wb') as stream:
            stream.write(serialized)
        lines.extend([
            '', '=== Build result ===', 'status: SUCCESS',
            'precision: strict FP32 (TF32 disabled)',
            'build time: {:.3f} s'.format(elapsed),
            'engine size: {:.2f} MB'.format(
                os.path.getsize(engine_path) / (1024 ** 2)),
        ])
    except Exception as error:
        lines.extend([
            '', '=== Build result ===', 'status: FAILED',
            '{}: {}'.format(type(error).__name__, error),
        ])
        write_report(args.out, lines)
        raise
    write_report(args.out, lines)


if __name__ == '__main__':
    main()
