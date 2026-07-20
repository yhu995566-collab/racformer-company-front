#!/usr/bin/env python3
"""Run the TensorRT ONNX parser without building or benchmarking an engine."""

import argparse
import ctypes
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse RaCFormer ONNX with the installed TensorRT')
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--plugin', action='append', default=[],
        help='TensorRT plugin shared library to load before parsing')
    parser.add_argument(
        '--layer-index', action='append', type=int, default=[],
        help='TensorRT network layer index to inspect after parsing')
    parser.add_argument(
        '--layer-match', action='append', default=[],
        help='Substring used to find TensorRT layer or tensor names')
    parser.add_argument('--layer-radius', type=int, default=8)
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('TensorRT parser report: {}'.format(path))


def describe_tensor(tensor):
    if tensor is None:
        return '<none>'
    return '{} shape={} dtype={}'.format(
        tensor.name, tuple(tensor.shape), tensor.dtype)


def describe_layer(network, index):
    layer = network.get_layer(index)
    lines = [
        'layer {}: type={}, name={}'.format(index, layer.type, layer.name),
    ]
    for input_index in range(layer.num_inputs):
        lines.append('  input {}: {}'.format(
            input_index, describe_tensor(layer.get_input(input_index))))
    for output_index in range(layer.num_outputs):
        lines.append('  output {}: {}'.format(
            output_index, describe_tensor(layer.get_output(output_index))))
    return lines


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer TensorRT parser audit ===',
        'onnx: {}'.format(os.path.abspath(args.onnx)),
    ]
    try:
        import tensorrt as trt
    except ImportError as error:
        lines.extend([
            'status: FAILED',
            'TensorRT Python import failed: {}'.format(error),
        ])
        write_report(args.out, lines)
        raise

    lines.append('TensorRT version: {}'.format(trt.__version__))
    for plugin_path in args.plugin:
        plugin_path = os.path.abspath(plugin_path)
        ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        lines.append('loaded plugin: {}'.format(plugin_path))

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    creator = trt.get_plugin_registry().get_plugin_creator(
        'bev_pool_v2', '1', '')
    lines.append('bev_pool_v2 plugin registered: {}'.format(
        creator is not None))
    identity_creator = trt.get_plugin_registry().get_plugin_creator(
        'racformer_identity', '1', '')
    lines.append('racformer_identity plugin registered: {}'.format(
        identity_creator is not None))
    msmv_creator = trt.get_plugin_registry().get_plugin_creator(
        'racformer_msmv_sampling', '1', '')
    lines.append('racformer_msmv_sampling plugin registered: {}'.format(
        msmv_creator is not None))
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(os.path.abspath(args.onnx))

    lines.extend([
        'status: {}'.format('PASS' if parsed else 'FAILED'),
        'parser errors: {}'.format(parser.num_errors),
    ])
    for index in range(parser.num_errors):
        lines.append('error {}: {}'.format(index, parser.get_error(index)))

    if parsed:
        lines.extend([
            'network inputs: {}'.format(network.num_inputs),
            'network outputs: {}'.format(network.num_outputs),
            'network layers: {}'.format(network.num_layers),
            '', '=== TensorRT inputs ===',
        ])
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            lines.append('{}: {} {}'.format(
                tensor.name, tuple(tensor.shape), tensor.dtype))
        lines.extend(['', '=== TensorRT outputs ==='])
        for index in range(network.num_outputs):
            tensor = network.get_output(index)
            lines.append('{}: {} {}'.format(
                tensor.name, tuple(tensor.shape), tensor.dtype))

        inspect_indices = set()
        radius = max(0, args.layer_radius)
        for center in args.layer_index:
            start = max(0, center - radius)
            end = min(network.num_layers, center + radius + 1)
            inspect_indices.update(range(start, end))
        matched_centers = []
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            names = [layer.name]
            names.extend(
                tensor.name for tensor_index in range(layer.num_inputs)
                for tensor in [layer.get_input(tensor_index)]
                if tensor is not None)
            names.extend(
                tensor.name for tensor_index in range(layer.num_outputs)
                for tensor in [layer.get_output(tensor_index)]
                if tensor is not None)
            if any(pattern in name for pattern in args.layer_match
                   for name in names):
                matched_centers.append(index)
                start = max(0, index - radius)
                end = min(network.num_layers, index + radius + 1)
                inspect_indices.update(range(start, end))
        if inspect_indices:
            lines.extend([
                '', '=== TensorRT layer inspection ===',
                'requested indices: {}'.format(args.layer_index),
                'matched centers: {}'.format(matched_centers),
                'radius: {}'.format(radius),
            ])
            for index in sorted(inspect_indices):
                lines.extend(describe_layer(network, index))

    write_report(args.out, lines)
    if not parsed:
        raise RuntimeError('TensorRT could not parse the ONNX graph')


if __name__ == '__main__':
    main()
