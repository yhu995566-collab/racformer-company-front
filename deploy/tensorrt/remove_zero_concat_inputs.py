#!/usr/bin/env python3
"""Remove every statically empty TensorRT tensor used only by ONNX Concat."""

import argparse
import ctypes
import os
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--plugin', action='append', default=[],
        help='TensorRT plugin shared library to load before parsing')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX zero-input removal report: {}'.format(path))


def find_zero_dim_tensors(network):
    tensors = {}
    shape_tensors = {}
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None and 0 in tuple(tensor.shape):
                target = shape_tensors \
                    if tensor.is_shape_tensor else tensors
                target[tensor.name] = (
                    tuple(tensor.shape), layer_index, layer.name)
    return tensors, shape_tensors


def find_onnx_shape_parameter_tensors(model):
    producer_types = {
        output: node.op_type
        for node in model.graph.node
        for output in node.output
    }
    uses = {}
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            uses.setdefault(name, []).append((node.op_type, input_index))

    shape_inputs = {
        'ConstantOfShape': {0},
        'Expand': {1},
        'Reshape': {1},
        'Slice': {1, 2, 3, 4},
        'Tile': {1},
    }
    return {
        name for name, producer_type in producer_types.items()
        if producer_type == 'Shape'
        and uses.get(name)
        and all(
            input_index in shape_inputs.get(op_type, set())
            for op_type, input_index in uses[name])
    }


def prune_unreachable_nodes(model):
    required_tensors = {output.name for output in model.graph.output}
    kept_reversed = []
    for node in reversed(model.graph.node):
        if not any(output in required_tensors for output in node.output):
            continue
        kept_reversed.append(node)
        required_tensors.update(
            name for name in node.input if name)

    kept_nodes = list(reversed(kept_reversed))
    removed_count = len(model.graph.node) - len(kept_nodes)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return removed_count


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer ONNX zero-length Concat input removal ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
    ]
    try:
        import onnx
        import tensorrt as trt

        for plugin_path in args.plugin:
            plugin_path = os.path.abspath(plugin_path)
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
            lines.append('loaded plugin: {}'.format(plugin_path))

        logger = trt.Logger(trt.Logger.WARNING)
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

        model = onnx.load(args.onnx, load_external_data=True)
        zero_tensors, zero_shape_tensors = find_zero_dim_tensors(network)
        semantic_shape_names = find_onnx_shape_parameter_tensors(model)
        for name in set(zero_tensors) & semantic_shape_names:
            zero_shape_tensors[name] = zero_tensors.pop(name)
        lines.append('zero-dimension execution tensors found: {}'.format(
            len(zero_tensors)))
        lines.append('zero-length shape tensors ignored: {}'.format(
            len(zero_shape_tensors)))
        for name in sorted(zero_tensors):
            shape, layer_index, layer_name = zero_tensors[name]
            lines.append(
                '{} shape={} producer=layer {} {!r}'.format(
                    name, shape, layer_index, layer_name))

        removals = []
        unresolved = []
        for node in model.graph.node:
            empty_indices = [
                index for index, name in enumerate(node.input)
                if name in zero_tensors
            ]
            if not empty_indices:
                continue
            if node.op_type != 'Concat':
                unresolved.append(
                    '{} ({}) inputs {}'.format(
                        node.name, node.op_type, empty_indices))
                continue
            remaining = [
                name for index, name in enumerate(node.input)
                if index not in empty_indices
            ]
            if not remaining:
                unresolved.append(
                    '{} (Concat) would have no inputs'.format(node.name))
                continue
            removed_names = [node.input[index] for index in empty_indices]
            del node.input[:]
            node.input.extend(remaining)
            removals.append(
                '{}: removed inputs {} {}; remaining {}'.format(
                    node.name, empty_indices, removed_names, len(remaining)))

        graph_outputs = {
            output.name for output in model.graph.output
            if output.name in zero_tensors
        }
        if graph_outputs:
            unresolved.append(
                'zero-dimension graph outputs: {}'.format(
                    sorted(graph_outputs)))
        if unresolved:
            lines.extend([
                'unresolved zero-dimension uses: {}'.format(len(unresolved)),
                *unresolved,
            ])
            raise RuntimeError(
                'zero-dimension tensors are not limited to removable '
                'Concat inputs')
        if not zero_tensors:
            raise RuntimeError('TensorRT graph has no zero-dimension tensors')
        if not removals:
            raise RuntimeError('no zero-dimension Concat inputs were removed')

        removed_dead_nodes = prune_unreachable_nodes(model)
        onnx.checker.check_model(model)
        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(model, output_path)
        lines.extend([
            'removed Concat inputs: {}'.format(len(removals)),
            *removals,
            'removed unreachable nodes: {}'.format(removed_dead_nodes),
            'unresolved zero-dimension uses: 0',
            'onnx checker: PASS',
            'status: SUCCESS',
        ])
    except Exception as error:
        lines.extend([
            'status: FAILED',
            '{}: {}'.format(type(error).__name__, error),
            traceback.format_exc(),
        ])
        write_report(args.report, lines)
        raise

    write_report(args.report, lines)


if __name__ == '__main__':
    main()
