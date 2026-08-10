import ast
import importlib.util
import math
import runpy
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_bbox_utils():
    if "mmcv.runner" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        runner = types.ModuleType("mmcv.runner")

        def auto_fp16(*args, **kwargs):
            del args, kwargs
            return lambda function: function

        runner.BaseModule = torch.nn.Module
        runner.auto_fp16 = auto_fp16
        mmcv.runner = runner
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.runner"] = runner
    path = ROOT / "models" / "bbox" / "utils.py"
    spec = importlib.util.spec_from_file_location("bbox_utils_long_range", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_query_initialization():
    path = ROOT / "models" / "query_initialization.py"
    spec = importlib.util.spec_from_file_location("query_initialization", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dynamic_radius_round_trip_preserves_350m_coordinates():
    utils = load_bbox_utils()
    pc_range = [0.0, -20.0, -3.0, 350.0, 20.0, 3.0]
    xy = torch.tensor([[0.1, 0.5], [0.5, 0.25], [0.999, 0.75]])
    polar = utils.xy2theta_d_coods(xy, pc_range)
    restored = utils.theta_d2xy_coods(polar, pc_range)
    assert polar[:, 1].min() >= 0.0
    assert polar[:, 1].max() <= 1.0
    assert torch.allclose(restored, xy, atol=1e-5)
    assert math.isclose(utils._max_radius(pc_range), math.sqrt(350 ** 2 + 20 ** 2))


def test_30_by_30_grid_has_900_queries_and_more_far_anchors():
    initialization = load_query_initialization()
    grid = initialization.generate_front_grid_points(
        30, 30, distance_power=2.0)
    assert grid.shape == (900, 2)
    unique_x = torch.unique(grid[:, 0]).sort().values
    assert len(unique_x) == 30
    # Far-field spacing is smaller than near-field spacing.
    assert unique_x[-1] - unique_x[-2] < unique_x[1] - unique_x[0]
    assert unique_x[0] > 0.0 and unique_x[-1] < 1.0


def test_experiment_config_declares_consistent_long_range_geometry():
    path = ROOT / "configs" / "3dh_query_company_front_350m_f4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"point_cloud_range", "voxel_size"}
    }
    assert assignments["point_cloud_range"] == [0.0, -20.0, -3.0, 350.0, 20.0, 3.0]
    assert assignments["voxel_size"] == [0.5, 0.5, 6.0]


def test_q1_to_q5_configs_keep_expected_query_budgets():
    expected = {
        "q1": (900, 30, 30, 1.0),
        "q2": (900, 30, 30, 1.5),
        "q3": (900, 30, 30, 2.0),
        "q4": (900, 45, 20, 2.0),
        "q5": (1200, 30, 40, 2.0),
    }
    for name, values in expected.items():
        config = runpy.run_path(str(ROOT / "configs" / f"3dh_query_{name}.py"))
        head = config["model"]["pts_bbox_head"]
        actual = (
            head["num_query"], head["num_clusters"],
            head["transformer"]["num_ray"], head["query_distance_power"])
        assert actual == values
        assert actual[0] == actual[1] * actual[2]


def test_long_range_evaluation_bins_cover_350m_and_overall_far_range():
    config = runpy.run_path(
        str(ROOT / "configs" / "3dh_query_company_front_350m_f4.py"))
    ranges = config["evaluation_distance_ranges"]
    assert (200, 250) in ranges
    assert (250, 300) in ranges
    assert (300, 350) in ranges
    assert (200, 350) in ranges
