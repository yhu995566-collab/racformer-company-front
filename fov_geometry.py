"""Shared front sensor field-of-view geometry for NumPy and PyTorch."""

import math

import numpy as np
import torch


def validate_horizontal_fov(horizontal_fov_deg):
    if horizontal_fov_deg is None:
        return None
    value = float(horizontal_fov_deg)
    if not 0.0 < value < 180.0:
        raise ValueError(
            'horizontal_fov_deg must be between 0 and 180 degrees')
    return value


def front_fov_mask(xy, horizontal_fov_deg):
    """Return a center mask for a FOV symmetric around ego +X."""
    horizontal_fov_deg = validate_horizontal_fov(horizontal_fov_deg)
    if horizontal_fov_deg is None:
        if isinstance(xy, torch.Tensor):
            return torch.ones(xy.shape[:-1], dtype=torch.bool, device=xy.device)
        return np.ones(np.asarray(xy).shape[:-1], dtype=bool)
    limit = math.tan(math.radians(horizontal_fov_deg * 0.5))
    if isinstance(xy, torch.Tensor):
        return (xy[..., 0] >= 0) & (xy[..., 1].abs() <= xy[..., 0] * limit)
    xy = np.asarray(xy)
    return (xy[..., 0] >= 0) & (np.abs(xy[..., 1]) <= xy[..., 0] * limit)
