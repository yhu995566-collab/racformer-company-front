import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "analysis" / "train_company_radar_topk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("company_radar_topk", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_temporal_features_transform_velocity_and_set_seconds(tmp_path):
    module = load_module()
    radar_path = tmp_path / "radar.npy"
    np.save(radar_path, np.asarray([[1, 0, 0, 5, 2, 0, 99]], np.float32))
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 3
    transform[:2, :2] = np.asarray([[0, -1], [1, 0]], np.float32)
    entry = {
        "data_path": str(radar_path), "timestamp": 800000,
        "radar_in_ego": False, "radar2ego": transform,
    }
    points = module.load_radar_entry(
        tmp_path, entry, 1000000, [0, -20, -3, 200, 20, 3])
    assert np.allclose(points[0, :2], [3, 1])
    assert np.allclose(points[0, 4:6], [0, 2])
    assert np.isclose(points[0, 6], 0.2)


def test_top_indices_returns_descending_largest_values():
    module = load_module()
    values = np.asarray([0.1, 0.9, 0.4, 0.7])
    assert module.top_indices(values, 3).tolist() == [1, 3, 2]


def make_info(token, radar_path):
    return {
        "token": str(token),
        "timestamp": 1000000 + token * 100000,
        "rads": {"RADAR_FRONT": {
            "data_path": str(radar_path),
            "timestamp": 1000000 + token * 100000,
            "radar_in_ego": True,
            "radar2ego": np.eye(4, dtype=np.float32),
        }},
        "sweeps": [],
        "gt_boxes": np.asarray([[10, 0, 0, 4, 2, 2, 0]], np.float32),
        "gt_names": np.asarray(["car"], dtype=object),
        "gt_sources": np.asarray([1], dtype=np.int64),
    }


def test_cpu_smoke_trains_and_reports_all_baselines(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    radar_path = processed / "radar.npy"
    np.save(radar_path, np.asarray([
        [9.5, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0],
        [10.5, 0.5, 0.0, 10.0, 0.0, 0.0, 0.0],
        [50.0, 5.0, 0.0, -5.0, 0.0, 0.0, 0.0],
        [80.0, -5.0, 0.0, -8.0, 0.0, 0.0, 0.0],
    ], np.float32))
    for split, tokens in (("train", [0, 1]), ("val", [2])):
        with (processed / "custom_infos_{}_sweep.pkl".format(split)).open("wb") as f:
            pickle.dump({"infos": [make_info(token, radar_path)
                                    for token in tokens]}, f)

    output = tmp_path / "output"
    subprocess.run([
        sys.executable, str(SCRIPT),
        "--processed-root", str(processed),
        "--out-dir", str(output),
        "--device", "cpu",
        "--epochs", "1",
        "--hidden-dim", "8",
        "--embedding-dim", "8",
        "--topk", "2", "4",
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    summary = json.loads((output / "company_radar_topk_summary.json").read_text())
    assert summary["train_frames"] == 2
    assert summary["val_frames"] == 1
    assert summary["point_cloud_range"][3] == 200.0
    assert "all_candidates" in summary["metrics"]
    assert "random_top2" in summary["metrics"]
    assert "rcs_top2" in summary["metrics"]
    assert "mlp_top2" in summary["metrics"]
    assert "mlp_top2_corrected" in summary["metrics"]
    assert (output / "radar_candidate_scorer_200m.pth").is_file()
