import pickle

import numpy as np

from tools.audit_company_split_distribution import summarize


def test_summarize_applies_roi_fov_and_class_filters(tmp_path):
    path = tmp_path / "infos.pkl"
    infos = [{
        "token": "sequence-a-000001",
        "gt_boxes": np.asarray([
            [10, 0, 0, 4, 2, 2, 0],       # kept car
            [10, 19, 0, 4, 2, 2, 0],      # outside 120-degree FOV
            [30, 0, 0, 1, 1, 2, 0],       # kept bicycle, far bin
            [60, 0, 0, 4, 2, 2, 0],       # outside range
        ], dtype=np.float32),
        "gt_names": np.asarray(["car", "truck", "bicycle", "car"]),
        "gt_velocity": np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32),
    }]
    with path.open("wb") as stream:
        pickle.dump({"infos": infos}, stream)

    result = summarize(path, ("car", "truck", "bicycle"),
                       (0, -20, -3, 50, 20, 3), 120)

    assert result["frames"] == 1
    assert result["sequences"] == 1
    assert result["gt"] == {"car": 1, "bicycle": 1}
    assert result["distance_bins"] == {"bicycle:25_50m": 1, "car:0_25m": 1}
    assert result["velocity"] == {"finite": 2, "nonzero": 1}
