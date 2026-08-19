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
    parser.add_argument(
        '--voxel-size', type=float, nargs=3,
        default=(0.5, 0.5, 6.0), metavar=('X', 'Y', 'Z'))
    parser.add_argument(
        '--depth-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        help='Radar projection depth range. Defaults to [1, x_max + 5].')
    parser.add_argument('--max-detections', type=int, default=300)
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
        arrays = {name: fixture[name] for name in REQUIRED}
        # Older, already-validated 100m/q300 fixtures predate this metadata
        # field. That checkpoint was explicitly trained/exported with the
        # legacy 65m polar decoder radius, matching the Python validator's
        # compatibility fallback.
        arrays['decoder_polar_radius'] = (
            fixture['decoder_polar_radius']
            if 'decoder_polar_radius' in fixture
            else np.asarray(65.0, dtype=np.float32))
        pc_range = np.asarray(arrays['decoder_pc_range'], dtype=np.float32)
        if pc_range.shape != (6,):
            raise ValueError('decoder_pc_range must have shape (6,)')
        if args.max_detections <= 0:
            raise ValueError('--max-detections must be positive')
        depth_range = (
            args.depth_range if args.depth_range is not None
            else (1.0, float(pc_range[3]) + 5.0))
        if not 0.0 <= depth_range[0] < depth_range[1]:
            raise ValueError('--depth-range must satisfy 0 <= MIN < MAX')
        radar_voxels = (
            fixture['radar_voxels_0']
            if 'radar_voxels_0' in fixture else None)
        if radar_voxels is None or radar_voxels.ndim < 1:
            raise KeyError('fixture is missing radar_voxels_0 capacity metadata')
        arrays.update({
            'runtime_voxel_size': np.asarray(
                args.voxel_size, dtype=np.float32),
            'runtime_depth_range': np.asarray(
                depth_range, dtype=np.float32),
            'runtime_static_radar_voxels': np.asarray(
                radar_voxels.shape[0], dtype=np.int32),
            'runtime_max_detections': np.asarray(
                args.max_detections, dtype=np.int32),
        })
        for name, source in arrays.items():
            array = np.ascontiguousarray(source)
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
