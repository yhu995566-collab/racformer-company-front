"""Query anchor initialization helpers with no OpenMMLab dependencies."""

import torch


def generate_front_grid_points(
        num_depth, num_lateral, distance_power=1.0, dtype=torch.float):
    """Generate normalized front-view Cartesian anchors.

    ``distance_power=1`` is a uniform longitudinal grid. Values greater than
    one retain full near-to-far coverage while progressively reducing spacing
    toward the far boundary.
    """
    if num_depth <= 0 or num_lateral <= 0:
        raise ValueError('query grid dimensions must be positive')
    if distance_power < 1.0:
        raise ValueError('distance_power must be >= 1.0')
    u = torch.linspace(0, 1, num_depth + 2, dtype=dtype)[1:-1]
    if distance_power != 1.0:
        u = 1.0 - (1.0 - u).pow(distance_power)
    lateral = torch.linspace(0, 1, num_lateral, dtype=dtype)
    x = u.view(1, num_depth).expand(num_lateral, num_depth)
    y = lateral.view(num_lateral, 1).expand(num_lateral, num_depth)
    return torch.stack([x, y], dim=-1).flatten(0, 1)
