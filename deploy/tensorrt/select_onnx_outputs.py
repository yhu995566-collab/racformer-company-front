#!/usr/bin/env python3
"""Keep selected ONNX graph outputs without changing model computation."""

import argparse
import os
import re
import traceback

import onnx


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--keep-regex', action='append', required=True,
        help='Regular expression matched against graph output names')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX output selection report: {}'.format(path))


def main():
    args = parse_args()
    lines = [
        '=== ONNX graph output selection ===',
        'input: {}'.format(os.path.abspath(args.onnx)),
        'output: {}'.format(os.path.abspath(args.out)),
        'keep regex: {}'.format(args.keep_regex),
    ]
    try:
        patterns = [re.compile(pattern) for pattern in args.keep_regex]
        model = onnx.load(os.path.abspath(args.onnx))
        original_outputs = list(model.graph.output)
        matched_by_pattern = [
            [
                value.name for value in original_outputs
                if pattern.search(value.name)
            ]
            for pattern in patterns
        ]
        missing_patterns = [
            pattern.pattern for pattern, matches
            in zip(patterns, matched_by_pattern) if not matches
        ]
        if missing_patterns:
            raise RuntimeError(
                'patterns matched no outputs: {}'.format(missing_patterns))

        kept_outputs = [
            value for value in original_outputs
            if any(pattern.search(value.name) for pattern in patterns)
        ]
        if not kept_outputs:
            raise RuntimeError('no graph outputs selected')
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
        onnx.checker.check_model(model)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(model, output_path)
        lines.extend([
            'input outputs: {}'.format(len(original_outputs)),
            'kept outputs: {}'.format(len(kept_outputs)),
            'removed outputs: {}'.format(
                len(original_outputs) - len(kept_outputs)),
            'onnx checker: PASS',
            '',
            '=== Kept outputs ===',
        ])
        lines.extend(value.name for value in kept_outputs)
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
