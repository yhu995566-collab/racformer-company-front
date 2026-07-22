#!/usr/bin/env python3
"""Rewrite unsupported ONNX operators for TensorRT 8.5."""

import argparse
import collections
import os
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('TensorRT 8.5 rewrite report: {}'.format(path))


def _attribute_int(node, name, default):
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def rewrite_isinf_nodes(model):
    """Replace IsInf with Abs/Greater or directional comparisons."""
    import onnx
    from onnx import helper

    existing_names = {
        name
        for node in model.graph.node
        for name in list(node.input) + list(node.output)
    }
    max_name = '__racformer_trt85_float_max'
    min_name = '__racformer_trt85_float_min'
    suffix = 0
    while max_name in existing_names or min_name in existing_names:
        suffix += 1
        max_name = '__racformer_trt85_float_max_{}'.format(suffix)
        min_name = '__racformer_trt85_float_min_{}'.format(suffix)

    max_float = 3.4028234663852886e38
    model.graph.initializer.extend([
        helper.make_tensor(max_name, onnx.TensorProto.FLOAT, [], [max_float]),
        helper.make_tensor(min_name, onnx.TensorProto.FLOAT, [], [-max_float]),
    ])

    rewritten = []
    replacement_nodes = []
    for index, node in enumerate(model.graph.node):
        if (node.domain not in ('', 'ai.onnx') or
                node.op_type != 'IsInf'):
            replacement_nodes.append(node)
            continue

        detect_negative = _attribute_int(node, 'detect_negative', 1)
        detect_positive = _attribute_int(node, 'detect_positive', 1)
        if not detect_negative and not detect_positive:
            raise RuntimeError(
                'IsInf node {!r} disables both detection directions'.format(
                    node.name))

        base_name = node.name or 'IsInf_{}'.format(index)
        if detect_negative and detect_positive:
            absolute_output = '{}__trt85_abs'.format(node.output[0])
            replacement_nodes.extend([
                helper.make_node(
                    'Abs', [node.input[0]], [absolute_output],
                    name='{}__trt85_abs'.format(base_name)),
                helper.make_node(
                    'Greater', [absolute_output, max_name], [node.output[0]],
                    name='{}__trt85_isinf'.format(base_name)),
            ])
        elif detect_positive:
            replacement_nodes.append(helper.make_node(
                'Greater', [node.input[0], max_name], [node.output[0]],
                name='{}__trt85_isposinf'.format(base_name)))
        else:
            replacement_nodes.append(helper.make_node(
                'Less', [node.input[0], min_name], [node.output[0]],
                name='{}__trt85_isneginf'.format(base_name)))
        rewritten.append(node.name or '<unnamed node {}>'.format(index))

    if not rewritten:
        del model.graph.initializer[-2:]
        return rewritten
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    return rewritten


def rewrite_trt85_unsupported_nodes(input_path, output_path):
    import onnx

    model = onnx.load(input_path, load_external_data=True)
    before = collections.Counter(node.op_type for node in model.graph.node)
    rewritten_isinf = rewrite_isinf_nodes(model)
    onnx.checker.check_model(model)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    onnx.save(model, output_path)
    after = collections.Counter(node.op_type for node in model.graph.node)
    return {
        'input_nodes': sum(before.values()),
        'output_nodes': sum(after.values()),
        'isinf_rewritten': len(rewritten_isinf),
        'isinf_remaining': after['IsInf'],
        'layernorm_remaining': after['LayerNormalization'],
        'onnx_checker': 'PASS',
    }


def main():
    args = parse_args()
    lines = [
        '=== TensorRT 8.5 ONNX compatibility rewrite ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
    ]
    try:
        result = rewrite_trt85_unsupported_nodes(args.onnx, args.out)
        lines.extend([
            'input nodes: {}'.format(result['input_nodes']),
            'output nodes: {}'.format(result['output_nodes']),
            'IsInf nodes rewritten: {}'.format(result['isinf_rewritten']),
            'IsInf nodes remaining: {}'.format(result['isinf_remaining']),
            'LayerNormalization nodes remaining: {}'.format(
                result['layernorm_remaining']),
            'onnx checker: {}'.format(result['onnx_checker']),
        ])
        if result['isinf_remaining'] or result['layernorm_remaining']:
            raise RuntimeError(
                'TensorRT 8.5 unsupported operators remain in the graph')
        lines.append('status: SUCCESS')
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
