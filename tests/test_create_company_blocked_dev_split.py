from tools.create_company_blocked_dev_split import artifacts, split_sequence


def make_info(index):
    def frame_path(frame, kind):
        return "{}/{}.bin".format(kind, max(frame, 0))
    return {
        "token": "sequence-a-{:06d}".format(index),
        "sequence": "sequence-a",
        "lidar_path": frame_path(index, "lidar"),
        "radar_path": frame_path(index, "radar"),
        "cams": {"CAM_FRONT": {"data_path": frame_path(index, "image")}},
        "sweeps": [{
            "CAM_FRONT": {"data_path": frame_path(index - offset, "image")},
            "RADAR_FRONT": {"data_path": frame_path(index - offset, "radar")},
        } for offset in range(1, 4)],
    }


def test_blocked_split_covers_sequence_without_temporal_leakage():
    source = [make_info(index) for index in range(100)]
    train, val, dropped, windows = split_sequence(
        source, val_fraction=0.15, blocks=3, guard=3)

    assert len(val) == 15
    assert len(windows) == 3
    assert dropped == 18
    train_paths = set().union(*(artifacts(info) for info in train))
    val_paths = set().union(*(artifacts(info) for info in val))
    assert not train_paths & val_paths
