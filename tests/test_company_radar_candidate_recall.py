import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools" / "analysis" / "company_radar_candidate_recall.py"
    spec = importlib.util.spec_from_file_location("company_candidate_recall", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nearest_distances_and_empty_radar():
    module = load_module()
    gt = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    radar = np.asarray([[1.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    assert np.allclose(module.nearest_distances(gt, radar), [1.0, 3.0])
    assert np.isinf(module.nearest_distances(gt, np.empty((0, 3)))).all()


def test_temporal_entries_are_deduplicated_and_transformed(tmp_path):
    module = load_module()
    radar_path = tmp_path / "radar.npy"
    np.save(radar_path, np.asarray([[1, 0, 0, 0, 0, 0, 0]], np.float32))
    transform = np.eye(4)
    transform[0, 3] = 2.0
    current = {
        "data_path": str(radar_path), "timestamp": 100,
        "radar_in_ego": True, "radar2ego": np.eye(4)}
    previous = {
        "data_path": str(radar_path), "timestamp": 0,
        "radar_in_ego": False, "radar2ego": transform}
    info = {
        "rads": {"RADAR_FRONT": current},
        "sweeps": [
            {"RADAR_FRONT": previous},
            {"RADAR_FRONT": previous},
        ],
    }
    now, temporal, count = module.collect_radar(
        tmp_path, info, 4, [0, -20, -3, 350, 20, 3])
    assert count == 2
    assert np.allclose(now[:, 0], [1.0])
    assert np.allclose(temporal[:, 0], [1.0, 3.0])


def test_summary_reports_threshold_recall():
    module = load_module()
    records = [
        {"nearest_current_m": 0.5},
        {"nearest_current_m": 3.0},
        {"nearest_current_m": float("inf")},
    ]
    summary = module.summarize(records, "nearest_current_m", [1.0, 4.0])
    assert np.isclose(summary["recall"]["within_1.0m"], 1 / 3)
    assert np.isclose(summary["recall"]["within_4.0m"], 2 / 3)
