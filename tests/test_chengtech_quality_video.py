import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_video_module():
    path = ROOT / "tools" / "render_chengtech_quality_video.py"
    spec = importlib.util.spec_from_file_location("chengtech_quality_video", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_box_corners_use_geometric_center_and_lwh_axes():
    module = load_video_module()
    corners = module.box_corners_3d([10, 2, 1, 4, 2, 2, 0])
    assert np.allclose(corners.min(axis=0), [8, 1, 0])
    assert np.allclose(corners.max(axis=0), [12, 3, 2])


def test_bev_projector_places_forward_up_and_positive_y_left():
    module = load_video_module()
    projector = module.BevProjector(
        width=640, height=1080,
        x_min=0, x_max=350, y_min=-100, y_max=100)
    origin, forward, left = projector.pixels(np.asarray([
        [0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]))
    assert forward[1] < origin[1]
    assert left[0] < origin[0]
