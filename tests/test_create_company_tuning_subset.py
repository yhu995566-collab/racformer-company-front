import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "create_company_tuning_subset.py"
    spec = importlib.util.spec_from_file_location("tuning_subset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def info(token, names, centers):
    boxes = np.zeros((len(names), 7), dtype=np.float32)
    boxes[:, :3] = centers
    return {
        "token": token,
        "gt_boxes": boxes,
        "gt_names": np.asarray(names),
        "valid_flag": np.ones(len(names), dtype=bool),
        "lidar_path": "lidar/{}.npy".format(token),
        "cams": {"CAM_FRONT": {"data_path": "images/{}.jpeg".format(token)}},
        "rads": {"RADAR_FRONT": {"data_path": "radar/{}.npy".format(token)}},
        "sweeps": [],
    }


def test_balanced_subset_covers_sequences_and_target_classes():
    tool = load_tool()
    infos = [
        info("seq-a-000000", ["car"], [[10, 0, 0]]),
        info("seq-a-000001", ["bicycle"], [[20, 0, 0]]),
        info("seq-b-000000", ["truck"], [[30, 0, 0]]),
        info("seq-b-000001", ["pedestrian"], [[10, 0, 0]]),
    ]
    selected = tool.balanced_select(
        infos, 3, ("car", "truck", "bicycle"),
        (0, -20, -3, 50, 20, 3), seed=0)
    assert selected == [0, 1, 2]


def test_subset_rewrites_artifacts_to_source_absolute_paths(tmp_path):
    tool = load_tool()
    source = tmp_path / "processed" / "custom_infos_train_sweep.pkl"
    source.parent.mkdir()
    payload = {"infos": [info("seq-a-000000", ["car"], [[10, 0, 0]])]}
    with source.open("wb") as stream:
        pickle.dump(payload, stream)
    output = tool.subset_payload(payload, [0], source)
    selected = output["infos"][0]
    assert Path(selected["lidar_path"]).is_absolute()
    assert selected["lidar_path"] == str(
        source.parent / "lidar/seq-a-000000.npy")
    assert Path(selected["cams"]["CAM_FRONT"]["data_path"]).is_absolute()
