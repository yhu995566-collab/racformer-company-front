import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_collection_converter():
    path = ROOT / "tools" / "convert_chengtech_20260818_collection.py"
    spec = importlib.util.spec_from_file_location(
        "convert_chengtech_20260818_collection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_collection_split_has_no_sequence_leakage():
    manifest = json.loads((
        ROOT / "data_splits" / "company_20260818_30k_v1.json").read_text())
    groups = [set(manifest[name]) for name in ("train", "val", "test")]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert len(groups[2]) == 3


def test_modality_frame_ids_may_have_a_fixed_offset():
    converter = load_collection_converter()
    assert converter.contiguous_start({3, 4, 5}, "image", "sequence") == 3
    assert converter.contiguous_start({0, 1, 2}, "lidar", "sequence") == 0


def test_result_lidar_adds_zero_ring_channel(tmp_path):
    converter = load_collection_converter()
    fields = {
        "x": np.asarray([1.0, 2.0], dtype=np.float32),
        "y": np.asarray([3.0, 4.0], dtype=np.float32),
        "z": np.asarray([5.0, 6.0], dtype=np.float32),
        "intensity": np.asarray([7.0, 8.0], dtype=np.float32),
    }
    converter.single.read_binary_compressed_pcd = lambda _: fields
    frame = type("Frame", (), {"lidar_path": Path("input.pcd"),
                                "sample_id": "sequence-000000"})()
    output, audit = converter.convert_result_lidar(
        frame, tmp_path, (0.0, -10.0, -10.0, 10.0, 10.0, 10.0), "ego")
    points = np.load(output)
    assert points.shape == (2, 5)
    assert np.array_equal(points[:, :4], np.column_stack(list(fields.values())))
    assert np.count_nonzero(points[:, 4]) == 0
    assert audit["source_points"] == 2
    assert audit["cropped_points"] == 2


def test_global_result_lidar_is_transformed_to_ego(tmp_path):
    converter = load_collection_converter()
    fields = {
        "x": np.asarray([11.0], dtype=np.float32),
        "y": np.asarray([2.0], dtype=np.float32),
        "z": np.asarray([3.0], dtype=np.float32),
        "intensity": np.asarray([7.0], dtype=np.float32),
    }
    converter.single.read_binary_compressed_pcd = lambda _: fields
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, 3] = [10.0, 0.0, 0.0]
    frame = type("Frame", (), {
        "lidar_path": Path("input.pcd"),
        "sample_id": "sequence-000000",
        "ego2global": ego2global,
    })()
    output, audit = converter.convert_result_lidar(
        frame, tmp_path,
        (0.0, -10.0, -10.0, 10.0, 10.0, 10.0), "global")
    points = np.load(output)
    assert np.allclose(points[0, :3], [1.0, 2.0, 3.0])
    assert audit["coordinate_frame"] == "global"
    assert audit["cropped_points"] == 1


def test_empty_result_lidar_can_be_audited_without_output(tmp_path):
    converter = load_collection_converter()
    fields = {
        "x": np.asarray([100.0], dtype=np.float32),
        "y": np.asarray([100.0], dtype=np.float32),
        "z": np.asarray([100.0], dtype=np.float32),
        "intensity": np.asarray([7.0], dtype=np.float32),
    }
    converter.single.read_binary_compressed_pcd = lambda _: fields
    frame = type("Frame", (), {
        "lidar_path": Path("empty.pcd"),
        "sample_id": "sequence-000001",
    })()
    output, audit = converter.convert_result_lidar(
        frame, tmp_path,
        (0.0, -10.0, -10.0, 10.0, 10.0, 10.0), "ego",
        allow_empty=True)
    assert output is None
    assert audit["source_points"] == 1
    assert audit["cropped_points"] == 0
    assert audit["xyz_min"] is None


def test_existing_result_lidar_is_validated_and_reused(tmp_path):
    converter = load_collection_converter()
    output = tmp_path / "lidar" / "sequence-000002.npy"
    output.parent.mkdir()
    expected = np.asarray([[1.0, 2.0, 3.0, 4.0, 0.0]], dtype=np.float32)
    np.save(output, expected)
    frame = type("Frame", (), {
        "lidar_path": Path("source-must-not-be-read.pcd"),
        "sample_id": "sequence-000002",
    })()
    converter.single.read_binary_compressed_pcd = lambda _: (_ for _ in ()).throw(
        AssertionError("source PCD was unexpectedly read"))
    actual, audit = converter.convert_result_lidar(
        frame, tmp_path,
        (0.0, -10.0, -10.0, 10.0, 10.0, 10.0), "ego",
        reuse_existing=True)
    assert actual == output.resolve()
    assert audit["reused_existing"] is True
    assert audit["cropped_points"] == 1
