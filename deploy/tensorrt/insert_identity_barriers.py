#!/usr/bin/env python3
"""Insert TensorRT identity-plugin barriers into an existing ONNX graph."""

import argparse
import os
import re
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--before-node-regex', action='append', required=True,
        help='Regex selecting ONNX nodes whose input receives a barrier')
    parser.add_argument(
        '--input-index', type=int, default=0,
        help='Selected node input to pass through the identity plugin')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX barrier insertion report: {}'.format(path))


def unique_tensor_name(base, used_names):
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = '{}_{}'.format(base, suffix)
        suffix += 1
    used_names.add(candidate)
    return candidate


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer ONNX identity barrier insertion ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
        'before-node regexes: {}'.format(args.before_node_regex),
        'input index: {}'.format(args.input_index),
    ]
    try:
        import onnx

        patterns = [re.compile(pattern) for pattern in args.before_node_regex]
        model = onnx.load(args.onnx, load_external_data=True)
        used_names = {
            name
            for node in model.graph.node
            for name in list(node.input) + list(node.output)
        }
        updated_nodes = []
        matched = []
        for node in model.graph.node:
            if not any(pattern.search(node.name) for pattern in patterns):
                updated_nodes.append(node)
                continue
            if args.input_index < 0 or args.input_index >= len(node.input):
                raise RuntimeError(
                    'input index {} is invalid for node {!r}'.format(
                        args.input_index, node.name))

            original_input = node.input[args.input_index]
            barrier_output = unique_tensor_name(
                '{}__racformer_identity'.format(original_input), used_names)
            barrier = onnx.helper.make_node(
                'racformer_identity',
                [original_input],
                [barrier_output],
                name='{}/input_{}_racformer_identity'.format(
                    node.name, args.input_index),
                domain='mmdeploy')
            node.input[args.input_index] = barrier_output
            updated_nodes.extend([barrier, node])
            matched.append(
                '{} input {}: {} -> {}'.format(
                    node.name, args.input_index,
                    original_input, barrier_output))

        if not matched:
            raise RuntimeError('no ONNX nodes matched the requested regexes')

        del model.graph.node[:]
        model.graph.node.extend(updated_nodes)
        if not any(item.domain == 'mmdeploy' for item in model.opset_import):
            model.opset_import.extend([
                onnx.helper.make_opsetid('mmdeploy', 1)])
        onnx.checker.check_model(model)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(model, output_path)
        lines.extend([
            'matched nodes: {}'.format(len(matched)),
            *matched,
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
