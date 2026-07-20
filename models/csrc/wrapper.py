import torch
import torch.nn.functional as F

SINGLE_CAMERA_PROJECTION_TRT = False

try:
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_forward, _ms_deform_attn_cuda_c2345_backward
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_forward, _ms_deform_attn_cuda_c23456_backward
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c45_forward, _ms_deform_attn_cuda_c45_backward
    MSMV_CUDA = True
except ImportError as e:
    print('Warning: failed to load one or more CUDA extensions, performance may be hurt.')
    print('Error message:', e)
    MSMV_CUDA = False


def _msmv_sampling_symbolic(g, *inputs):
    output = g.op('mmdeploy::racformer_msmv_sampling', *inputs)
    feature = inputs[0]
    sampling_locations = inputs[-2]
    try:
        feature_sizes = feature.type().sizes()
        sampling_sizes = sampling_locations.type().sizes()
        output.setType(feature.type().with_sizes([
            feature_sizes[0], sampling_sizes[1],
            feature_sizes[4], sampling_sizes[2],
        ]))
    except (AttributeError, TypeError):
        output.setType(feature.type())
    return output


class SingleCameraProjection(torch.autograd.Function):
    image_h = 0
    image_w = 0

    @staticmethod
    def symbolic(g, sample_points, lidar2img):
        output = g.op(
            'mmdeploy::racformer_single_camera_projection',
            sample_points, lidar2img,
            image_h_i=SingleCameraProjection.image_h,
            image_w_i=SingleCameraProjection.image_w)
        try:
            sizes = sample_points.type().sizes()
            output.setType(sample_points.type().with_sizes([
                sizes[0] * sizes[2] * sizes[3],
                sizes[1], sizes[4], 3,
            ]))
        except (AttributeError, TypeError):
            output.setType(sample_points.type())
        return output

    @staticmethod
    def forward(ctx, sample_points, lidar2img):
        del ctx
        points = sample_points.permute(0, 2, 1, 3, 4, 5)
        homogeneous = torch.cat(
            [points, torch.ones_like(points[..., :1])], dim=-1)
        projection = lidar2img[:, :, None, None, None, :, :]
        projected = (
            projection * homogeneous[..., None, :]).sum(dim=-1)
        depth = torch.maximum(
            projected[..., 2:3],
            torch.zeros_like(projected[..., 2:3]) + 1e-5)
        xy = projected[..., 0:2] / depth
        xy = torch.stack([
            xy[..., 0] / SingleCameraProjection.image_w,
            xy[..., 1] / SingleCameraProjection.image_h,
        ], dim=-1)
        locations = torch.cat(
            [xy, torch.zeros_like(xy[..., :1])], dim=-1)
        return locations.permute(0, 1, 3, 2, 4, 5).reshape(
            sample_points.shape[0] * sample_points.shape[2] *
            sample_points.shape[3],
            sample_points.shape[1], sample_points.shape[4], 3)


def single_camera_projection_enabled():
    return SINGLE_CAMERA_PROJECTION_TRT


def project_single_camera(sample_points, lidar2img, image_h, image_w):
    SingleCameraProjection.image_h = int(image_h)
    SingleCameraProjection.image_w = int(image_w)
    return SingleCameraProjection.apply(sample_points, lidar2img)


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 3]
    scale_weights: [B, Q, P, 4]
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, C, _, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    sampling_locations = sampling_locations * 2 - 1
    single_camera = mlvl_feats[0].shape[2] == 1
    if single_camera:
        sampling_grid = sampling_locations[..., :2]
    else:
        sampling_grid = sampling_locations[:, :, :, None, :]

    final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)

    for lvl, feat in enumerate(mlvl_feats):
        if single_camera:
            out = F.grid_sample(
                feat[:, :, 0, :, :], sampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=True)
        else:
            out = F.grid_sample(
                feat, sampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=True)[..., 0]
        out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final += out

    return final.permute(0, 2, 1, 3)

def msmv_sampling_pytorch_v2(mlvl_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 3]
    scale_weights: [B, Q, P, 4]
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, C, _, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    sampling_locations = sampling_locations * 2 - 1
    single_camera = mlvl_feats[0].shape[2] == 1
    if single_camera:
        sampling_grid = sampling_locations[..., :2]
    else:
        sampling_grid = sampling_locations[:, :, :, None, :]

    # final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)
    final = []
    for lvl, feat in enumerate(mlvl_feats):
        if single_camera:
            out = F.grid_sample(
                feat[:, :, 0, :, :], sampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=True)
        else:
            out = F.grid_sample(
                feat, sampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=True)[..., 0]
        # out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        # final += out
        final.append(out)
    final = torch.stack(final, dim=-1)
    max_indices = torch.argmax(scale_weights, dim=-1)  # [B, Q, P]
    idx = max_indices.unsqueeze(1).unsqueeze(-1).expand(-1, C, -1, -1, -1)  # [B, C, Q, P, 1]

    B_idx = torch.arange(B).view(B, 1, 1, 1, 1).to(max_indices.device)
    C_idx = torch.arange(C).view(1, C, 1, 1, 1).to(max_indices.device)
    Q_idx = torch.arange(Q).view(1, 1, Q, 1, 1).to(max_indices.device)
    P_idx = torch.arange(P).view(1, 1, 1, P, 1).to(max_indices.device)

    final = final[B_idx, C_idx, Q_idx, P_idx, idx][..., 0]

    return final.permute(0, 2, 1, 3)

class MSMVSamplingC2345(torch.autograd.Function):
    symbolic = staticmethod(_msmv_sampling_symbolic)

    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c2345_forward)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c2345_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c2345_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight

class MSMVSamplingC45(torch.autograd.Function):
    symbolic = staticmethod(_msmv_sampling_symbolic)

    @staticmethod
    def forward(ctx, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c4, feat_c5, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c45_forward)
        return _ms_deform_attn_cuda_c45_forward(
            feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c45_backward)
        grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c45_backward(grad_output.contiguous(), 
            feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        
        return grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight
    
class MSMVSamplingC23456(torch.autograd.Function):
    symbolic = staticmethod(_msmv_sampling_symbolic)

    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c23456_forward)
        return _ms_deform_attn_cuda_c23456_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c23456_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c23456_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) == 2 and MSMV_CUDA:
        return MSMVSamplingC45.apply(*mlvl_feats, sampling_locations, scale_weights)
    if len(mlvl_feats) == 4 and MSMV_CUDA:
        return MSMVSamplingC2345.apply(*mlvl_feats, sampling_locations, scale_weights)
    elif len(mlvl_feats) == 5 and MSMV_CUDA:
        return MSMVSamplingC23456.apply(*mlvl_feats, sampling_locations, scale_weights)
    else:
        return msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights)

def msmv_sampling_v2(mlvl_feats, sampling_locations, scale_weights):
    return msmv_sampling_pytorch_v2(mlvl_feats, sampling_locations, scale_weights)
