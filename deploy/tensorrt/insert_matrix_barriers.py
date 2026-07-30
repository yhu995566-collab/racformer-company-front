#!/usr/bin/env python3
"""Insert TensorRT identity barriers after ONNX matrix operations."""

import argparse
import collections
import os
import traceback

import onnx
from onnx import helper


MATRIX_OPS = frozenset(('Gemm', 'MatMul'))
IDENTITY_DOMAIN = 'mmdeploy'
IDENTITY_OP = 'racformer_identity'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--reshape-only', action='store_true',
        help='Only isolate matrix outputs consumed directly by Reshape')
    return parser.parse_args()


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Matrix barrier report: {}'.format(path))


def is_barrier(node):
    return node.domain == IDENTITY_DOMAIN and node.op_type == IDENTITY_OP


def main():
    args = parse_args()
    input_path = os.path.abspath(args.onnx)
    output_path = os.path.abspath(args.out)
    lines = [
        '=== ONNX matrix barrier insertion ===',
        'input: {}'.format(input_path),
        'output: {}'.format(output_path),
        'scope: {}'.format(
            'direct Reshape consumers'
            if args.reshape_only else 'all Gemm/MatMul outputs'),
    ]
    try:
        model = onnx.load(input_path)
        consumers = collections.defaultdict(list)
        for node in model.graph.node:
            for input_index, tensor_name in enumerate(node.input):
                if tensor_name:
                    consumers[tensor_name].append((node, input_index))

        targets = {}
        producer_counts = collections.Counter()
        for node_index, node in enumerate(model.graph.node):
            if node.op_type not in MATRIX_OPS:
                continue
            for output_index, tensor_name in enumerate(node.output):
                if not tensor_name:
                    continue
                tensor_consumers = consumers.get(tensor_name, [])
                non_barrier_consumers = [
                    item for item in tensor_consumers
                    if not is_barrier(item[0])
                ]
                if not non_barrier_consumers:
                    continue
                if args.reshape_only and not any(
                        consumer.op_type == 'Reshape'
                        for consumer, _ in non_barrier_consumers):
                    continue
                barrier_output = '{}__racformer_matrix_barrier'.format(
                    tensor_name)
                targets[(node_index, output_index)] = (
                    tensor_name,
                    barrier_output,
                    non_barrier_consumers,
                )
                producer_counts[node.op_type] += 1

        if not targets:
            raise RuntimeError('no matrix outputs require barriers')

        inserted = []
        rewired_edges = 0
        for (node_index, output_index), (
                tensor_name, barrier_output,
                tensor_consumers) in targets.items():
            del output_index
            producer = model.graph.node[node_index]
            barrier = helper.make_node(
                IDENTITY_OP,
                inputs=[tensor_name],
                outputs=[barrier_output],
                name='MatrixBarrier_{}'.format(len(inserted)),
                domain=IDENTITY_DOMAIN,
                plugin_version='1',
                plugin_namespace='')
            inserted.append((node_index, barrier))
            for consumer, input_index in tensor_consumers:
                consumer.input[input_index] = barrier_output
                rewired_edges += 1

        barriers_by_producer = collections.defaultdict(list)
        for node_index, barrier in inserted:
            barriers_by_producer[node_index].append(barrier)
        rewritten_nodes = []
        for node_index, node in enumerate(model.graph.node):
            rewritten_nodes.append(node)
            rewritten_nodes.extend(barriers_by_producer.get(node_index, []))
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

        if not any(
                opset.domain == IDENTITY_DOMAIN
                for opset in model.opset_import):
            model.opset_import.append(
                helper.make_opsetid(IDENTITY_DOMAIN, 1))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        onnx.checker.check_model(onnx.load(output_path))
        lines.extend([
            'producer counts: {}'.format(dict(producer_counts)),
            'inserted barriers: {}'.format(len(inserted)),
            'rewired consumer edges: {}'.format(rewired_edges),
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
