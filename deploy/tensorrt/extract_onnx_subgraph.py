#!/usr/bin/env python3
"""Extract the minimal ONNX subgraph required by selected graph outputs."""

import argparse
import os
import traceback

import onnx
from onnx import utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--output', action='append', required=True, dest='outputs',
        help='Exact graph output name to retain; may be repeated')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('ONNX subgraph extraction report: {}'.format(path))


def required_graph_inputs(model, output_names):
    graph_input_names = {value.name for value in model.graph.input}
    initializer_names = {value.name for value in model.graph.initializer}
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }

    required_inputs = set()
    visited = set()
    pending = list(output_names)
    while pending:
        name = pending.pop()
        if not name or name in visited:
            continue
        visited.add(name)
        if name in graph_input_names:
            if name not in initializer_names:
                required_inputs.add(name)
            continue
        if name in initializer_names:
            continue
        producer = producers.get(name)
        if producer is None:
            raise RuntimeError(
                "tensor '{}' has no graph input, initializer, or producer"
                .format(name))
        pending.extend(producer.input)

    return [
        value.name for value in model.graph.input
        if value.name in required_inputs
    ]


def main():
    args = parse_args()
    input_path = os.path.abspath(args.onnx)
    output_path = os.path.abspath(args.out)
    lines = [
        '=== ONNX dependency subgraph extraction ===',
        'input: {}'.format(input_path),
        'output: {}'.format(output_path),
        'requested outputs: {}'.format(args.outputs),
    ]
    try:
        if len(args.outputs) != len(set(args.outputs)):
            raise ValueError('duplicate output names are not allowed')
        model = onnx.load(input_path)
        graph_outputs = {value.name for value in model.graph.output}
        missing = set(args.outputs).difference(graph_outputs)
        if missing:
            raise RuntimeError(
                'requested names are not graph outputs: {}'.format(
                    sorted(missing)))

        input_names = required_graph_inputs(model, args.outputs)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        utils.extract_model(
            input_path, output_path, input_names, args.outputs,
            check_model=True)
        extracted = onnx.load(output_path, load_external_data=False)
        onnx.checker.check_model(extracted)
        lines.extend([
            'input nodes: {}'.format(len(model.graph.node)),
            'output nodes: {}'.format(len(extracted.graph.node)),
            'removed nodes: {}'.format(
                len(model.graph.node) - len(extracted.graph.node)),
            'required inputs: {}'.format(input_names),
            'kept outputs: {}'.format(args.outputs),
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
