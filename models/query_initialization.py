"""Query anchor initialization helpers with no OpenMMLab dependencies."""

import math

import torch


def generate_front_grid_points(
        num_depth, num_lateral, distance_power=1.0, dtype=torch.float):
    """Generate normalized Cartesian anchors over a front-view rectangle.

    ``distance_power=1`` produces uniform longitudinal spacing. Values greater
    than one preserve the full near-to-far coverage while moving anchors toward
    the far boundary.
    """
    if num_depth <= 0 or num_lateral <= 0:
        raise ValueError('query grid dimensions must be positive')
    if distance_power < 1.0:
        raise ValueError('distance_power must be >= 1.0')

    depth = torch.linspace(
        0, 1, num_depth + 2, dtype=dtype)[1:-1]
    if distance_power != 1.0:
        depth = 1.0 - (1.0 - depth).pow(distance_power)
    lateral = torch.linspace(0, 1, num_lateral, dtype=dtype)

    x = depth.view(1, num_depth).expand(num_lateral, num_depth)
    y = lateral.view(num_lateral, 1).expand(num_lateral, num_depth)
    return torch.stack([x, y], dim=-1).flatten(0, 1)


def generate_front_fov_grid_points(
        num_depth, num_lateral, point_cloud_range,
        horizontal_fov_deg=120.0, distance_power=1.0, dtype=torch.float):
    """Generate normalized anchors in a rectangular ROI intersected by FOV."""
    if num_depth <= 0 or num_lateral <= 0:
        raise ValueError('query grid dimensions must be positive')
    if len(point_cloud_range) != 6:
        raise ValueError('point_cloud_range must contain six values')
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError('horizontal_fov_deg must be between 0 and 180')
    x_min, y_min, _, x_max, y_max, _ = map(float, point_cloud_range)
    if x_min < 0 or x_max <= x_min or not y_min < 0 < y_max:
        raise ValueError('front FOV grid requires a front ROI spanning y=0')
    normalized_x = torch.linspace(
        0, 1, num_depth + 2, dtype=dtype)[1:-1]
    if distance_power < 1.0:
        raise ValueError('distance_power must be >= 1.0')
    if distance_power != 1.0:
        normalized_x = 1.0 - (1.0 - normalized_x).pow(distance_power)
    x = x_min + normalized_x * (x_max - x_min)
    sensor_y_limit = x * math.tan(math.radians(horizontal_fov_deg * 0.5))
    roi_y_limit = min(-y_min, y_max)
    y_limit = torch.clamp(sensor_y_limit, max=roi_y_limit)
    lateral = torch.linspace(-1, 1, num_lateral, dtype=dtype)
    actual_y = y_limit[:, None] * lateral[None, :]
    normalized_y = (actual_y - y_min) / (y_max - y_min)
    normalized_x = normalized_x[:, None].expand(num_depth, num_lateral)
    return torch.stack([normalized_x, normalized_y], dim=-1).flatten(0, 1)
