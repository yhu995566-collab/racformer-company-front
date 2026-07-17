"""TensorRT-only identity operation used to prevent oversized fusion groups."""

import torch


class TensorRTFusionBarrier(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, tensor):
        output = graph.op(
            'mmdeploy::racformer_identity',
            tensor,
            plugin_version_s='1',
            plugin_namespace_s='')
        output.setType(tensor.type())
        return output

    @staticmethod
    def forward(ctx, tensor):
        return tensor.clone()


def tensorrt_fusion_barrier(tensor):
    return TensorRTFusionBarrier.apply(tensor)
