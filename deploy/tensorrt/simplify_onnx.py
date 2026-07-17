#!/usr/bin/env python3
"""Fold constant ONNX shape subgraphs without executing CUDA tensors."""

import argparse
import os
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--skip-shape-inference', action='store_true',
        help='Skip ONNX shape inference when custom operators block it')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX simplification report: {}'.format(path))


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer ONNX simplification ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
        'skip shape inference: {}'.format(args.skip_shape_inference),
    ]
    try:
        import onnx
        try:
            from onnxsim import simplify
        except ImportError as error:
            raise RuntimeError(
                'onnxsim is required; install it with '
                '`python -m pip install onnxsim==0.4.36`') from error

        model = onnx.load(args.onnx, load_external_data=True)
        before_nodes = len(model.graph.node)
        simplified, checked = simplify(
            model,
            check_n=0,
            skip_shape_inference=args.skip_shape_inference)
        if not checked:
            raise RuntimeError('onnxsim equivalence check failed')
        onnx.checker.check_model(simplified)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(simplified, output_path)
        after_nodes = len(simplified.graph.node)
        lines.extend([
            'input nodes: {}'.format(before_nodes),
            'output nodes: {}'.format(after_nodes),
            'removed nodes: {}'.format(before_nodes - after_nodes),
            'onnxsim check: PASS',
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
