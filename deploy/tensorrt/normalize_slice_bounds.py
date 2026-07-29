#!/usr/bin/env python3
"""Clamp oversized INT64 Slice bounds to TensorRT's INT32 range."""

import argparse
import collections
import os
import traceback

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper


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
    print('Slice-bound normalization report: {}'.format(path))


def main():
    args = parse_args()
    input_path = os.path.abspath(args.onnx)
    output_path = os.path.abspath(args.out)
    lines = [
        '=== ONNX Slice-bound normalization ===',
        'input: {}'.format(input_path),
        'output: {}'.format(output_path),
    ]
    try:
        model = onnx.load(input_path)
        consumers = collections.defaultdict(list)
        for node in model.graph.node:
            for name in node.input:
                if name:
                    consumers[name].append(node.op_type)

        int32_info = np.iinfo(np.int32)
        changed_tensors = 0
        changed_elements = 0

        def normalize(name, array, replace):
            nonlocal changed_tensors, changed_elements
            array = np.asarray(array)
            if array.dtype != np.int64 or not array.size:
                return
            mask = (array < int32_info.min) | (array > int32_info.max)
            if not mask.any():
                return
            tensor_consumers = consumers.get(name, [])
            if not tensor_consumers:
                raise RuntimeError(
                    'oversized INT64 tensor has no consumer: {}'.format(name))
            non_slice = [
                op_type for op_type in tensor_consumers
                if op_type != 'Slice'
            ]
            if non_slice:
                raise RuntimeError(
                    'oversized INT64 tensor {} has non-Slice consumers: {}'
                    .format(name, non_slice))
            normalized = np.clip(
                array, int32_info.min, int32_info.max).astype(
                    np.int64, copy=False)
            replace(normalized)
            changed_tensors += 1
            changed_elements += int(mask.sum())

        for index, tensor in enumerate(model.graph.initializer):
            def replace_initializer(array, index=index, name=tensor.name):
                replacement = numpy_helper.from_array(array, name=name)
                model.graph.initializer[index].CopyFrom(replacement)

            normalize(
                tensor.name,
                numpy_helper.to_array(tensor),
                replace_initializer)

        for node in model.graph.node:
            if node.op_type != 'Constant' or not node.output:
                continue
            output_name = node.output[0]
            for attribute in node.attribute:
                if attribute.type == AttributeProto.TENSOR:
                    def replace_tensor(array, attribute=attribute):
                        replacement = numpy_helper.from_array(array)
                        attribute.t.CopyFrom(replacement)

                    normalize(
                        output_name,
                        numpy_helper.to_array(attribute.t),
                        replace_tensor)
                elif attribute.type == AttributeProto.INTS:
                    def replace_ints(array, attribute=attribute):
                        del attribute.ints[:]
                        attribute.ints.extend(array.reshape(-1).tolist())

                    normalize(
                        output_name,
                        np.asarray(attribute.ints, dtype=np.int64),
                        replace_ints)
                elif attribute.type == AttributeProto.INT:
                    def replace_int(array, attribute=attribute):
                        attribute.i = int(array.reshape(-1)[0])

                    normalize(
                        output_name,
                        np.asarray(attribute.i, dtype=np.int64),
                        replace_int)

        if changed_tensors == 0:
            raise RuntimeError(
                'no oversized INT64 Slice bounds were found')

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        onnx.checker.check_model(onnx.load(output_path))
        lines.extend([
            'changed tensors: {}'.format(changed_tensors),
            'changed elements: {}'.format(changed_elements),
            'INT64 dtype preserved: True',
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
