#!/usr/bin/env python3
"""Remove known-empty inputs from selected ONNX Concat nodes."""

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
        '--node-regex', action='append', required=True,
        help='Regex selecting ONNX Concat node names')
    parser.add_argument(
        '--input-index', type=int, required=True,
        help='Concat input index to remove')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX Concat input removal report: {}'.format(path))


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer ONNX Concat input removal ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
        'node regexes: {}'.format(args.node_regex),
        'removed input index: {}'.format(args.input_index),
    ]
    try:
        import onnx

        patterns = [re.compile(pattern) for pattern in args.node_regex]
        model = onnx.load(args.onnx, load_external_data=True)
        matched = []
        for node in model.graph.node:
            if not any(pattern.search(node.name) for pattern in patterns):
                continue
            if node.op_type != 'Concat':
                raise RuntimeError(
                    'matched node {!r} is {}, not Concat'.format(
                        node.name, node.op_type))
            if args.input_index < 0 or args.input_index >= len(node.input):
                raise RuntimeError(
                    'input index {} is invalid for node {!r}'.format(
                        args.input_index, node.name))
            if len(node.input) <= 1:
                raise RuntimeError(
                    'cannot remove the only input of {!r}'.format(node.name))

            inputs = list(node.input)
            removed = inputs.pop(args.input_index)
            del node.input[:]
            node.input.extend(inputs)
            matched.append(
                '{}: removed input {} {!r}; remaining inputs {}'.format(
                    node.name, args.input_index, removed, len(inputs)))

        if not matched:
            raise RuntimeError('no ONNX nodes matched the requested regexes')

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
