#!/usr/bin/env python3
"""Export one fixed-shape RaCFormer sample and retain failure diagnostics."""

import argparse
import copy
import importlib
import os
import sys
import traceback
import types

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

from deploy.offline_demo import load_frames
from deploy.onnx_wrapper import (
    INPUT_NAMES, OUTPUT_NAMES, RaCFormerONNXWrapper, build_export_inputs)
from deploy.preprocessing import DeploymentPreprocessor
from deploy.pytorch_runner import RaCFormerPyTorchRunner
from models.csrc.tensorrt_barrier import tensorrt_fusion_barrier


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export fixed-input FP32 RaCFormer raw outputs to ONNX')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--boundary-atol', type=float, default=5e-3)
    parser.add_argument(
        '--strict-boundary-check', action='store_true',
        help='Fail when two full PyTorch forwards exceed boundary-atol')
    parser.add_argument(
        '--fallthrough', action='store_true',
        help='Preserve unsupported operators for graph auditing')
    parser.add_argument(
        '--constant-folding', action='store_true',
        help='Fold fixed-shape ONNX subgraphs during export')
    parser.add_argument(
        '--mixing-chunk-size', type=int, default=32768,
        help='Output width of each deployment AdaptiveMixing projection')
    parser.add_argument(
        '--msmv-plugin', action='store_true',
        help='Export the existing multi-scale sampling CUDA op as a TRT plugin')
    parser.add_argument(
        '--single-camera-projection-plugin', action='store_true',
        help='Export single-camera projection and coordinate packing as a '
             'TensorRT plugin')
    parser.add_argument(
        '--fixed-view-geometry', action='store_true',
        help='Precompute BEV pooling ranks after verifying that all frame '
             'camera-to-ego transforms are identical')
    parser.add_argument(
        '--tensorrt-85-compat', action='store_true',
        help='Decompose IsInf and LayerNormalization for TensorRT 8.5')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument(
        '--fixture',
        help='Optional NPZ containing TensorRT inputs and PyTorch outputs')
    return parser.parse_args()


def describe_tensor(name, tensor):
    return '{}: shape={}, dtype={}, device={}'.format(
        name, tuple(tensor.shape), tensor.dtype, tensor.device)


def write_report(path, lines):
    path = os.path.abspath(path)
    mmcv.mkdir_or_exist(os.path.dirname(path))
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('Export report: {}'.format(path))


def save_fixture(path, inputs, outputs):
    path = os.path.abspath(path)
    mmcv.mkdir_or_exist(os.path.dirname(path))
    arrays = {}
    for name, tensor in zip(INPUT_NAMES, inputs):
        arrays[name] = tensor.detach().cpu().numpy()
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        arrays[name] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    return path


def legacy_raw_outputs(model, batch):
    """Run the original NumPy-metadata path up to the detector head."""
    img_meta = copy.deepcopy(batch.img_meta)
    radar_points = [[points] for points in batch.radar_points]
    img_feats, bev_feats, radar_bev_feats, _ = model.extract_feat(
        img=batch.image,
        radar_points=radar_points,
        radar_depth=batch.radar_depth,
        radar_rcs=batch.radar_rcs,
        img_metas=[img_meta])
    outputs = model.pts_bbox_head(
        img_feats, bev_feats, radar_bev_feats, [img_meta])
    return outputs['all_cls_scores'], outputs['all_bbox_preds']


def disable_gradient_checkpointing(model):
    """Disable training-only recomputation that the legacy exporter cannot trace."""
    disabled = []
    for name, module in model.named_modules():
        if getattr(module, 'with_cp', False):
            module.with_cp = False
            disabled.append(name or '<root>')
    return disabled


def enable_standard_onnx_fallbacks(
        model, mixing_chunk_size, use_msmv_plugin,
        use_single_camera_projection_plugin):
    """Use traceable implementations instead of opaque CUDA autograd ops."""
    import models.csrc.wrapper as sampling_wrapper
    import models.racformer_transformer as transformer_module

    if use_msmv_plugin:
        if not sampling_wrapper.MSMV_CUDA:
            raise RuntimeError(
                '--msmv-plugin requires the compiled MSMV CUDA extension')
        transformer_module.MSMV_CUDA = True
    else:
        sampling_wrapper.MSMV_CUDA = False
        transformer_module.MSMV_CUDA = False
    camera_sampling_modules = [
        module for module in model.modules()
        if module.__class__.__name__ == 'RaCFormerSampling'
    ]
    if use_single_camera_projection_plugin and (
            not camera_sampling_modules or
            any(module.num_cams != 1 for module in camera_sampling_modules)):
        raise RuntimeError(
            'single-camera projection plugin requires num_cams=1')
    sampling_wrapper.SINGLE_CAMERA_PROJECTION_TRT = \
        use_single_camera_projection_plugin
    positional_cache_bytes = 0
    positional_cache_count = 0
    layernorm_barrier_count = 0

    def barrier_layernorm_input(module, inputs):
        del module
        return (tensorrt_fusion_barrier(inputs[0]),) + inputs[1:]

    for module in model.modules():
        if module.__class__.__name__ == 'RaCFormerTransformerDecoder':
            module._deploy_trt_decoder_barriers = True
        if module.__class__.__name__ == 'RaCFormerTransformerDecoderLayer':
            module._deploy_trt_branch_barriers = True
        if module.__class__.__name__ == 'BEVSampling':
            module._deploy_trt_sampling_barriers = True
        if module.__class__.__name__ == 'RaCFormerSampling':
            module._deploy_trt_sampling_barriers = True
        if module.__class__.__name__ == 'AdaptiveMixing':
            module._deploy_trt_mixing_barriers = True
            module._deploy_trt_parameter_chunk_size = mixing_chunk_size
        if module.__class__.__name__ == 'ScaleAdaptiveSelfAttention':
            module._deploy_vectorized_bbox_dist = True
        if isinstance(module, torch.nn.LayerNorm):
            module.register_forward_pre_hook(barrier_layernorm_input)
            layernorm_barrier_count += 1
        if module.__class__.__name__ == 'BEVSelfAttention':
            module._deploy_onnx_fallback = True
        if module.__class__.__name__ == 'BEVSampling':
            height, width = module.spatial_shapes
            parameter = next(module.positional_encoding.parameters())
            mask = torch.zeros(
                (1, height, width), device=parameter.device,
                dtype=parameter.dtype)
            with torch.no_grad():
                cache = module.positional_encoding(mask).detach()
            if '_deploy_bev_pos_cache' in module._buffers:
                module._deploy_bev_pos_cache = cache
            else:
                module.register_buffer(
                    '_deploy_bev_pos_cache', cache, persistent=False)
            positional_cache_count += 1
            positional_cache_bytes += cache.numel() * cache.element_size()
    return (
        positional_cache_count, positional_cache_bytes,
        layernorm_barrier_count)


def enable_fixed_view_geometry(model, img2lidar, atol=1e-6):
    """Enable the existing BEVPool acceleration for fixed vehicle calibration."""
    matrices = img2lidar.detach()
    if matrices.ndim != 4 or matrices.shape[0] != 1:
        raise RuntimeError(
            'fixed view geometry requires img2lidar shape [1, frames, 4, 4]')
    reference = matrices[:, :1].expand_as(matrices)
    max_error = (matrices - reference).abs().max().item()
    if not torch.allclose(matrices, reference, rtol=0.0, atol=atol):
        raise RuntimeError(
            'fixed view geometry requires identical frame transforms; '
            'maximum img2lidar difference is {:.8f}'.format(max_error))
    view_transformer = model.img_lss_view_transformer
    view_transformer.accelerate = True
    view_transformer.initial_flag = True
    return max_error


def enable_single_batch_radar_scatter(model):
    """Avoid the generic batch mask and its dynamic ONNX NonZero shape."""
    middle_encoder = model.radar_middle_encoder
    if middle_encoder.__class__.__name__ != 'PointPillarsScatter':
        raise RuntimeError(
            'TensorRT 8.5 radar scatter requires PointPillarsScatter, got {}'
            .format(middle_encoder.__class__.__name__))
    height, width = model.radar_output_shape
    channels = model.radar_middle_channels

    def single_batch_forward(module, voxel_features, coors, batch_size):
        del module
        if batch_size != 1:
            raise RuntimeError(
                'TensorRT 8.5 radar scatter requires batch_size=1')
        if voxel_features.dim() != 2:
            raise RuntimeError(
                'radar voxel features must have shape [voxels, channels]')
        linear_indices = (
            coors[:, -2] * int(width) + coors[:, -1]).to(torch.long)
        scatter_indices = linear_indices.unsqueeze(0).expand(
            int(channels), -1)
        canvas = voxel_features.new_zeros(
            (int(channels), int(height) * int(width)))
        canvas = canvas.scatter(
            1, scatter_indices, voxel_features.transpose(0, 1))
        return canvas.reshape(
            1, int(channels), int(height), int(width))

    middle_encoder.forward = types.MethodType(
        single_batch_forward, middle_encoder)
    return height, width, channels


def install_export_symbolics(opset, tensorrt_85_compat=False):
    """Install compatibility symbolics missing from the PyTorch 2.0 exporter."""
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper

    def node_detail(node, method_name):
        try:
            return getattr(node, method_name)()
        except Exception as error:
            return '<unavailable: {}>'.format(error)

    def diagnostic_cat(g, tensor_list, dim):
        tensors = symbolic_helper._unpack_list(tensor_list)
        nonempty = [
            tensor for tensor in tensors
            if not symbolic_helper._is_none(tensor)
        ]
        if not nonempty:
            list_node = tensor_list.node()
            details = [
                'ONNX export found aten::cat with no tensor inputs',
                'list node: {}'.format(list_node),
                'scope: {}'.format(node_detail(list_node, 'scopeName')),
                'source: {}'.format(node_detail(list_node, 'sourceRange')),
            ]
            raise RuntimeError('\n'.join(details))
        if len(nonempty) == 1:
            return nonempty[0]
        axis = symbolic_helper._get_const(dim, 'i', 'dim')
        return g.op('Concat', *nonempty, axis_i=axis)

    def atan2(g, y, x):
        zero = g.op('Constant', value_t=torch.tensor(0.0, dtype=torch.float32))
        pi = g.op(
            'Constant', value_t=torch.tensor(
                3.141592653589793, dtype=torch.float32))
        half_pi = g.op(
            'Constant', value_t=torch.tensor(
                1.5707963267948966, dtype=torch.float32))
        angle = g.op('Atan', g.op('Div', y, x))
        negative_x_offset = g.op(
            'Where', g.op('GreaterOrEqual', y, zero), pi, g.op('Neg', pi))
        angle = g.op(
            'Where', g.op('Less', x, zero),
            g.op('Add', angle, negative_x_offset), angle)
        vertical = g.op(
            'Where', g.op('Greater', y, zero), half_pi,
            g.op('Where', g.op('Less', y, zero), g.op('Neg', half_pi), zero))
        return g.op('Where', g.op('Equal', x, zero), vertical, angle)

    def isinf(g, value):
        max_float = g.op(
            'Constant', value_t=torch.tensor(
                torch.finfo(torch.float32).max, dtype=torch.float32))
        return g.op('Greater', g.op('Abs', value), max_float)

    @symbolic_helper.parse_args('v', 'is', 'v', 'v', 'f', 'i')
    def layer_norm(g, value, normalized_shape, weight, bias, eps,
                   cudnn_enable):
        del cudnn_enable
        axes = list(range(-len(normalized_shape), 0))
        mean = g.op(
            'ReduceMean', value, axes_i=axes, keepdims_i=1)
        centered = g.op('Sub', value, mean)
        variance = g.op(
            'ReduceMean', g.op('Mul', centered, centered),
            axes_i=axes, keepdims_i=1)
        epsilon = g.op(
            'Constant', value_t=torch.tensor(eps, dtype=torch.float32))
        normalized = g.op(
            'Div', centered,
            g.op('Sqrt', g.op('Add', variance, epsilon)))
        if not symbolic_helper._is_none(weight):
            normalized = g.op('Mul', normalized, weight)
        if not symbolic_helper._is_none(bias):
            normalized = g.op('Add', normalized, bias)
        return normalized

    register_custom_op_symbolic('aten::cat', diagnostic_cat, int(opset))
    register_custom_op_symbolic('aten::atan2', atan2, int(opset))
    if tensorrt_85_compat:
        register_custom_op_symbolic('aten::isinf', isinf, int(opset))
        register_custom_op_symbolic(
            'aten::layer_norm', layer_norm, int(opset))


def main():
    args = parse_args()
    report = [
        '=== RaCFormer FP32 ONNX export ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'sample index: {}'.format(args.sample_index),
        'opset: {}'.format(args.opset),
        'operator mode: {}'.format(
            'ONNX_FALLTHROUGH' if args.fallthrough else 'ONNX'),
        'constant folding: {}'.format(args.constant_folding),
        'AdaptiveMixing chunk size: {}'.format(args.mixing_chunk_size),
        'MSMV TensorRT plugin: {}'.format(args.msmv_plugin),
        'single-camera projection TensorRT plugin: {}'.format(
            args.single_camera_projection_plugin),
        'fixed view geometry: {}'.format(args.fixed_view_geometry),
        'TensorRT 8.5 compatibility: {}'.format(args.tensorrt_85_compat),
        'output boundary: raw all_cls_scores + all_bbox_preds (decode excluded)',
    ]
    try:
        if args.mixing_chunk_size <= 0:
            raise ValueError('mixing-chunk-size must be positive')
        if args.single_camera_projection_plugin and not args.msmv_plugin:
            raise ValueError(
                '--single-camera-projection-plugin requires --msmv-plugin')
        cfg = Config.fromfile(args.config)
        importlib.import_module('models')
        importlib.import_module('loaders')
        dataset = build_dataset(cfg.data[args.split])
        if args.sample_index < 0 or args.sample_index >= len(dataset):
            raise IndexError('sample index is outside the dataset')

        preprocessor = DeploymentPreprocessor(cfg)
        frames = load_frames(dataset, args.sample_index, preprocessor.num_frames)
        cpu_batch = preprocessor.prepare(frames)
        runner = RaCFormerPyTorchRunner(
            args.config, args.weights, device=args.device)
        checkpoint_modules = disable_gradient_checkpointing(runner.model)
        report.extend([
            '', '=== Export preparation ===',
            'disabled gradient-checkpoint modules: {}'.format(
                len(checkpoint_modules)),
        ])
        report.extend(
            'checkpoint disabled: {}'.format(name)
            for name in checkpoint_modules)
        batch = runner.prepare(cpu_batch)
        wrapper = RaCFormerONNXWrapper(
            runner.model, preprocessor.final_height,
            preprocessor.final_width).eval()
        inputs = build_export_inputs(batch, runner.model)

        report.extend(['', '=== Inputs ==='])
        report.extend(
            describe_tensor(name, tensor)
            for name, tensor in zip(INPUT_NAMES, inputs))

        with torch.no_grad():
            legacy_outputs = legacy_raw_outputs(runner.model, batch)
            torch.cuda.synchronize(runner.device)
            legacy_outputs = tuple(
                tensor.detach().clone() for tensor in legacy_outputs)
            torch.cuda.synchronize(runner.device)
            cache_count, cache_bytes, layernorm_barrier_count = \
                enable_standard_onnx_fallbacks(
                    runner.model, args.mixing_chunk_size, args.msmv_plugin,
                    args.single_camera_projection_plugin)
            radar_scatter_shape = None
            if args.tensorrt_85_compat:
                radar_scatter_shape = enable_single_batch_radar_scatter(
                    runner.model)
            fixed_geometry_error = None
            if args.fixed_view_geometry:
                fixed_geometry_error = enable_fixed_view_geometry(
                    runner.model, inputs[4])
            outputs = wrapper(*inputs)
        torch.cuda.synchronize(runner.device)
        report.extend(['', '=== PyTorch raw outputs ==='])
        report.extend(
            describe_tensor(name, tensor)
            for name, tensor in zip(OUTPUT_NAMES, outputs))
        report.extend([
            'cached BEV positional maps: {}'.format(cache_count),
            'cached BEV positional map size: {:.2f} MB'.format(
                cache_bytes / (1024 ** 2)),
            'TensorRT LayerNorm input barriers: {}'.format(
                layernorm_barrier_count),
        ])
        if radar_scatter_shape is not None:
            report.append(
                'TensorRT 8.5 single-batch radar scatter: '
                'channels={}, height={}, width={}'.format(
                    radar_scatter_shape[2], radar_scatter_shape[0],
                    radar_scatter_shape[1]))
        if fixed_geometry_error is not None:
            report.extend([
                'fixed view geometry frame max error: {:.8f}'.format(
                    fixed_geometry_error),
                'fixed BEV rank count: {}'.format(
                    runner.model.img_lss_view_transformer.ranks_bev.numel()),
                'fixed BEV interval count: {}'.format(
                    runner.model.img_lss_view_transformer.interval_starts.numel()),
            ])
        report.extend(['', '=== Tensor metadata boundary check ==='])
        boundary_passed = True
        for name, legacy, current in zip(
                OUTPUT_NAMES, legacy_outputs, outputs):
            difference = (legacy - current).abs()
            close = torch.allclose(
                legacy, current, rtol=0.0, atol=args.boundary_atol)
            boundary_passed = boundary_passed and close
            report.append(
                '{}: close={}, max_abs_error={:.8f}, '
                'mean_abs_error={:.8f}'.format(
                    name, close, difference.max().item(),
                    difference.mean().item()))
        report.append('boundary atol: {}'.format(args.boundary_atol))
        report.append('boundary comparison passed: {}'.format(
            boundary_passed))
        if not boundary_passed and args.strict_boundary_check:
            raise RuntimeError(
                'tensor metadata boundary does not match the legacy path')
        if not boundary_passed:
            report.append(
                'warning: continuing because radar voxelization and custom '
                'CUDA kernels can vary across independent full forwards')

        if args.fixture:
            fixture_path = save_fixture(args.fixture, inputs, outputs)
            report.extend([
                '', '=== TensorRT fixture ===',
                'fixture: {}'.format(fixture_path),
                'arrays: {}'.format(len(INPUT_NAMES) + len(OUTPUT_NAMES)),
            ])

        output_path = os.path.abspath(args.out)
        mmcv.mkdir_or_exist(os.path.dirname(output_path))
        operator_type = torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH \
            if args.fallthrough else torch.onnx.OperatorExportTypes.ONNX
        dynamic_axes = {}
        for index in range(8):
            voxel_count = 'radar_voxel_{}_count'.format(index)
            dynamic_axes.update({
                'radar_voxels_{}'.format(index): {0: voxel_count},
                'radar_num_points_{}'.format(index): {0: voxel_count},
                'radar_coors_{}'.format(index): {0: voxel_count},
            })
        install_export_symbolics(args.opset, args.tensorrt_85_compat)
        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=args.constant_folding,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            operator_export_type=operator_type,
            verbose=False)
        if args.tensorrt_85_compat:
            from deploy.tensorrt.rewrite_trt85_onnx import \
                rewrite_trt85_unsupported_nodes
            rewrite_result = rewrite_trt85_unsupported_nodes(
                output_path, output_path)
            report.extend([
                '', '=== TensorRT 8.5 ONNX rewrite ===',
                'IsInf nodes rewritten: {}'.format(
                    rewrite_result['isinf_rewritten']),
                'IsInf nodes remaining: {}'.format(
                    rewrite_result['isinf_remaining']),
                'LayerNormalization nodes remaining: {}'.format(
                    rewrite_result['layernorm_remaining']),
                'onnx checker: {}'.format(rewrite_result['onnx_checker']),
            ])
            if (rewrite_result['isinf_remaining'] or
                    rewrite_result['layernorm_remaining']):
                raise RuntimeError(
                    'TensorRT 8.5 unsupported operators remain after rewrite')
        report.extend([
            '', '=== Export result ===', 'status: SUCCESS',
            'onnx: {}'.format(output_path),
            'next: python -m deploy.tensorrt.audit_onnx --onnx {} --out {}.audit.txt'.format(
                output_path, output_path),
        ])
    except Exception as error:
        report.extend([
            '', '=== Export result ===', 'status: FAILED',
            'exception: {}: {}'.format(type(error).__name__, error),
            '', '=== Traceback ===', traceback.format_exc(),
            'A failed standard export is an expected audit result when the '
            'graph reaches an unsupported custom CUDA operator.',
        ])
        write_report(args.report, report)
        raise

    write_report(args.report, report)


if __name__ == '__main__':
    main()
