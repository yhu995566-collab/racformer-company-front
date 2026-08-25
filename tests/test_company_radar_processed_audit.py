import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "analysis" / "audit_company_radar_processed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("company_radar_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_info(token, radar_path, x=10.0):
    return {
        "token": token,
        "timestamp": int(token) * 100000,
        "rads": {"RADAR_FRONT": {"data_path": str(radar_path)}},
        "sweeps": [],
        "gt_boxes": np.asarray([[x, 0, 0, 4, 2, 2, 0]], np.float32),
        "gt_names": np.asarray(["car"], dtype=object),
        "gt_sources": np.asarray([1]),
    }


def write_split(root, split, infos):
    with (root / "custom_infos_{}_sweep.pkl".format(split)).open("wb") as f:
        pickle.dump({"infos": infos}, f)


def test_audit_counts_splits_ranges_and_existing_radar(tmp_path):
    module = load_module()
    radar = tmp_path / "radar.npy"
    np.save(radar, np.zeros((1, 7), np.float32))
    write_split(tmp_path, "train", [make_info("1", radar, 10)])
    write_split(tmp_path, "val", [make_info("2", radar, 175)])
    result = module.audit(tmp_path, ["train", "val"])
    assert result["passed"]
    assert result["splits"]["train"]["forward_range_counts"]["0-50m"] == 1
    assert result["splits"]["val"]["forward_range_counts"]["150-200m"] == 1


def test_audit_rejects_overlap_and_missing_radar(tmp_path):
    module = load_module()
    missing = tmp_path / "missing.npy"
    write_split(tmp_path, "train", [make_info("1", missing)])
    write_split(tmp_path, "val", [make_info("1", missing)])
    result = module.audit(tmp_path, ["train", "val"])
    assert not result["passed"]
    assert result["token_overlap_count"] == 1
    assert result["missing_radar_path_count"] == 2
