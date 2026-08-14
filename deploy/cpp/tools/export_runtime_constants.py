#!/usr/bin/env python3
"""Export production constants from a deployment NPZ without an NPZ C++ dependency."""

import argparse
import hashlib
import os

import numpy as np


REQUIRED = (
    'lidar2img',
    'img2lidar',
    'mlp_input',
    'query_bbox',
    'query_feat',
    'decoder_d_regions',
    'decoder_pc_range',
    'decoder_polar_radius',
)
SUPPORTED_DTYPES = {
    np.dtype('float32'): 'float32',
    np.dtype('float16'): 'float16',
    np.dtype('int32'): 'int32',
    np.dtype('uint8'): 'uint8',
    np.dtype('bool'): 'bool',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = os.path.abspath(args.out_dir)
    os.makedirs(output, exist_ok=True)
    manifest_lines = [
        '# name\tdtype\tcomma-separated-shape\tfile',
    ]
    checksums = []
    with np.load(args.fixture) as fixture:
        missing = [name for name in REQUIRED if name not in fixture]
        if missing:
            raise KeyError('fixture is missing constants: {}'.format(missing))
        for name in REQUIRED:
            array = np.ascontiguousarray(fixture[name])
            dtype = SUPPORTED_DTYPES.get(array.dtype)
            if dtype is None:
                raise TypeError('{} uses unsupported dtype {}'.format(
                    name, array.dtype))
            filename = '{}.bin'.format(name)
            path = os.path.join(output, filename)
            array.tofile(path)
            shape = ','.join(str(value) for value in array.shape)
            manifest_lines.append(
                '{}\t{}\t{}\t{}'.format(name, dtype, shape, filename))
            with open(path, 'rb') as stream:
                checksums.append('{}  {}'.format(
                    hashlib.sha256(stream.read()).hexdigest(), filename))
    manifest_path = os.path.join(output, 'manifest.tsv')
    with open(manifest_path, 'w') as stream:
        stream.write('\n'.join(manifest_lines) + '\n')
    with open(os.path.join(output, 'SHA256SUMS'), 'w') as stream:
        stream.write('\n'.join(checksums) + '\n')
    print('runtime constants: {}'.format(output))
    print('manifest: {}'.format(manifest_path))
    print('status: SUCCESS')


if __name__ == '__main__':
    main()
