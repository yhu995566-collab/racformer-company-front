import math

import numpy as np
import torch

from fov_geometry import front_fov_mask
from models.query_initialization import generate_front_fov_grid_points


def test_front_120_degree_mask_matches_physical_boundary():
    points = np.asarray([
        [5.0, 8.0],
        [5.0, 9.0],
        [10.0, 17.0],
        [10.0, 18.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    assert front_fov_mask(points, 120).tolist() == [
        True, False, True, False, False]


def test_front_fov_query_grid_stays_inside_roi_and_fov():
    roi = [0.0, -20.0, -3.0, 50.0, 20.0, 3.0]
    points = generate_front_fov_grid_points(10, 20, roi, 120.0)
    assert points.shape == (200, 2)
    actual_x = points[:, 0] * 50.0
    actual_y = points[:, 1] * 40.0 - 20.0
    assert torch.all(actual_x > 0)
    assert torch.all(actual_x < 50)
    assert torch.all(actual_y.abs() <= 20.0 + 1e-5)
    assert torch.all(
        actual_y.abs() <= actual_x * math.tan(math.radians(60)) + 1e-5)
