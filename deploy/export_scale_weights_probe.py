#!/usr/bin/env python3
"""Export the decoder sampling-weight projections as a small ONNX probe."""

import argparse
import copy
import os
import traceback

import numpy as np
import torch
from torch import nn

from deploy.pytorch_runner import RaCFormerPyTorchRunner


OUTPUT_NAMES = [
    'radar_logits',
    'radar_weights',
    'lss_logits',
    'lss_weights',
    'image_logits',
    'image_weights',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


class ScaleWeightsProbe(nn.Module):

    def __init__(self, decoder_layer):
        super().__init__()
        self.radar = copy.deepcopy(
            decoder_layer.sampling_radar_bev.scale_weights)
        self.lss = copy.deepcopy(
            decoder_layer.sampling_lss_bev.scale_weights)
        self.image = copy.deepcopy(decoder_layer.sampling.scale_weights)
        self.radar_shape = self.bev_shape(
            decoder_layer.sampling_radar_bev)
        self.lss_shape = self.bev_shape(
            decoder_layer.sampling_lss_bev)
        image = decoder_layer.sampling
        self.image_shape = (
            int(image.num_groups), int(image.num_frames),
            int(image.depth_num), int(image.num_points),
            int(image.num_levels))

    @staticmethod
    def bev_shape(sampling):
        return (
            int(sampling.num_heads), int(sampling.num_frames),
            int(sampling.num_levels), int(sampling.depth_num),
            int(sampling.num_points))

    @staticmethod
    def bev_weights(logits, shape):
        heads, frames, levels, depth, points = shape
        batch, queries, _ = logits.shape
        weights = logits.view(
            batch, queries, heads, 1, levels, depth * points)
        weights = torch.softmax(weights, dim=-1)
        return weights.expand(
            batch, queries, heads, frames, levels,
            depth * points).contiguous()

    @staticmethod
    def image_weights(logits, shape):
        groups, frames, depth, points, levels = shape
        batch, queries, _ = logits.shape
        weights = logits.view(
            batch, queries, groups, frames, depth * points, levels)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.permute(0, 2, 3, 1, 4, 5)
        return weights.reshape(
            batch * groups * frames, queries, depth * points, levels)

    def forward(self, query_feat):
        radar_logits = self.radar(query_feat)
        lss_logits = self.lss(query_feat)
        image_logits = self.image(query_feat)
        return (
            radar_logits,
            self.bev_weights(radar_logits, self.radar_shape),
            lss_logits,
            self.bev_weights(lss_logits, self.lss_shape),
            image_logits,
            self.image_weights(image_logits, self.image_shape),
        )


def write_report(path, lines):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Scale-weight probe export report: {}'.format(path))


def main():
    args = parse_args()
    lines = [
        '=== RaCFormer scale-weight ONNX probe ===',
        'config: {}'.format(os.path.abspath(args.config)),
        'weights: {}'.format(os.path.abspath(args.weights)),
        'device: {}'.format(args.device),
        'opset: {}'.format(args.opset),
    ]
    try:
        runner = RaCFormerPyTorchRunner(
            args.config, args.weights, device=args.device)
        decoder_layer = runner.model.pts_bbox_head.transformer.decoder \
            .decoder_layer
        probe = ScaleWeightsProbe(decoder_layer).to(runner.device).eval()
        num_query = int(runner.model.pts_bbox_head.num_query)
        embed_dims = int(decoder_layer.embed_dims)
        generator = torch.Generator(device=runner.device)
        generator.manual_seed(20260729)
        query_feat = torch.randn(
            (1, num_query, embed_dims), generator=generator,
            device=runner.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = probe(query_feat)
        torch.cuda.synchronize(runner.device)

        fixture_path = os.path.abspath(args.fixture)
        os.makedirs(os.path.dirname(fixture_path) or '.', exist_ok=True)
        arrays = {'query_feat': query_feat.detach().cpu().numpy()}
        for name, tensor in zip(OUTPUT_NAMES, outputs):
            arrays[name] = tensor.detach().cpu().numpy()
            lines.append(
                '{}: shape={}, min={:.8f}, max={:.8f}'.format(
                    name, tuple(tensor.shape), tensor.min().item(),
                    tensor.max().item()))
        np.savez_compressed(fixture_path, **arrays)

        output_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torch.onnx.export(
            probe,
            query_feat,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=['query_feat'],
            output_names=OUTPUT_NAMES,
            verbose=False)
        lines.extend([
            'fixture: {}'.format(fixture_path),
            'onnx: {}'.format(output_path),
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
