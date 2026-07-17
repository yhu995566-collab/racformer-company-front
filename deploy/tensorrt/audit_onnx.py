#!/usr/bin/env python3
"""Validate an ONNX graph and summarize operators requiring TensorRT work."""

import argparse
import collections
import os


STANDARD_DOMAINS = ('', 'ai.onnx', 'ai.onnx.ml')


def parse_args():
    parser = argparse.ArgumentParser(description='Audit an exported ONNX graph')
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--node-name', action='append', default=[],
        help='Exact ONNX node name whose dependency neighborhood is printed')
    parser.add_argument('--node-radius', type=int, default=1)
    return parser.parse_args()


def describe_value(onnx, value):
    tensor_type = value.type.tensor_type
    dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField('dim_value'):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField('dim_param'):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append('?')
    return '{}: [{}] {}'.format(
        value.name, ', '.join(dimensions), dtype)


def describe_node(node, index):
    return [
        'node {}: domain={}, op={}, name={}'.format(
            index, node.domain or 'ai.onnx', node.op_type, node.name),
        '  inputs: {}'.format(list(node.input)),
        '  outputs: {}'.format(list(node.output)),
    ]


def dependency_neighborhood(nodes, centers, radius):
    producers = {}
    consumers = collections.defaultdict(set)
    for index, node in enumerate(nodes):
        for name in node.output:
            producers[name] = index
        for name in node.input:
            consumers[name].add(index)

    selected = set(centers)
    frontier = set(centers)
    for _ in range(max(0, radius)):
        neighbors = set()
        for index in frontier:
            node = nodes[index]
            neighbors.update(
                producers[name] for name in node.input if name in producers)
            for name in node.output:
                neighbors.update(consumers[name])
        frontier = neighbors - selected
        selected.update(frontier)
    return sorted(selected)


def main():
    args = parse_args()
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            'The audit command requires the onnx Python package') from error

    model = onnx.load(args.onnx, load_external_data=True)
    checker_status = 'PASS'
    try:
        onnx.checker.check_model(model)
    except Exception as error:
        checker_status = 'FAIL: {}: {}'.format(type(error).__name__, error)

    counts = collections.Counter(
        ((node.domain or 'ai.onnx'), node.op_type) for node in model.graph.node)
    custom = [
        (domain, operator, count)
        for (domain, operator), count in sorted(counts.items())
        if domain not in STANDARD_DOMAINS
    ]
    lines = [
        '=== RaCFormer ONNX operator audit ===',
        'onnx: {}'.format(os.path.abspath(args.onnx)),
        'onnx checker: {}'.format(checker_status),
        'IR version: {}'.format(model.ir_version),
        'opsets: {}'.format(', '.join(
            '{}={}'.format(item.domain or 'ai.onnx', item.version)
            for item in model.opset_import)),
        'nodes: {}'.format(len(model.graph.node)),
        '', '=== Inputs ===',
    ]
    lines.extend(describe_value(onnx, value) for value in model.graph.input)
    lines.extend(['', '=== Outputs ==='])
    lines.extend(describe_value(onnx, value) for value in model.graph.output)
    lines.extend(['', '=== Operators ==='])
    lines.extend(
        '{}::{} x{}'.format(domain, operator, count)
        for (domain, operator), count in sorted(counts.items()))
    lines.extend(['', '=== Non-standard domains ==='])
    if custom:
        lines.extend(
            '{}::{} x{}'.format(domain, operator, count)
            for domain, operator, count in custom)
    else:
        lines.append('none found')

    if args.node_name:
        nodes = list(model.graph.node)
        centers = [
            index for index, node in enumerate(nodes)
            if node.name in args.node_name
        ]
        missing = sorted(set(args.node_name) - {
            nodes[index].name for index in centers
        })
        selected = dependency_neighborhood(
            nodes, centers, args.node_radius)
        lines.extend([
            '', '=== Node dependency inspection ===',
            'requested names: {}'.format(args.node_name),
            'matched centers: {}'.format(centers),
            'missing names: {}'.format(missing),
            'dependency radius: {}'.format(max(0, args.node_radius)),
        ])
        for index in selected:
            lines.extend(describe_node(nodes[index], index))

    lines.extend([
        '', 'A clean ONNX checker result does not guarantee TensorRT support.',
        'Use trtexec parser output to identify unsupported standard ONNX ops.',
    ])

    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Audit report: {}'.format(output_path))


if __name__ == '__main__':
    main()
