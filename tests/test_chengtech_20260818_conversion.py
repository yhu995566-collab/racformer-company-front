import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_converter():
    return load_module(
        "convert_chengtech_20260818",
        "tools/convert_chengtech_20260818.py")


def literal_lzf(data):
    encoded = bytearray()
    for offset in range(0, len(data), 32):
        chunk = data[offset:offset + 32]
        encoded.append(len(chunk) - 1)
        encoded.extend(chunk)
    return bytes(encoded)


def test_binary_compressed_pcd_preserves_all_declared_types(tmp_path):
    converter = load_converter()
    fields = {
        "x": np.asarray([1.0, 2.0], dtype="<f4"),
        "y": np.asarray([3.0, 4.0], dtype="<f4"),
        "z": np.asarray([5.0, 6.0], dtype="<f4"),
        "intensity": np.asarray([7.0, 8.0], dtype="<f4"),
        "ring": np.asarray([9, 127], dtype="<u2"),
        "timestamp": np.asarray([1.5e9, 1.6e9], dtype="<f8"),
    }
    decoded = b"".join(value.tobytes() for value in fields.values())
    compressed = literal_lzf(decoded)
    header = (
        "# .PCD v0.7\nVERSION 0.7\n"
        "FIELDS x y z intensity ring timestamp\n"
        "SIZE 4 4 4 4 2 8\n"
        "TYPE F F F F U F\n"
        "COUNT 1 1 1 1 1 1\n"
        "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary_compressed\n"
    ).encode("ascii")
    path = tmp_path / "0.pcd"
    path.write_bytes(
        header + struct.pack("<II", len(compressed), len(decoded)) + compressed)

    actual = converter.read_binary_compressed_pcd(path)
    assert set(actual) == set(fields)
    for name, expected in fields.items():
        assert actual[name].dtype == expected.dtype
        assert np.array_equal(actual[name], expected)


def test_projected_pixel_rounding_stays_inside_image_bounds():
    audit = load_module(
        "audit_chengtech_20260818",
        "tools/audit_chengtech_20260818.py")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    points = np.asarray([[1.9999, 1.9999, 1.0, 0.0, 0.0]])
    rendered, count = audit.project_lidar(image, points, np.eye(4), 10)
    assert count == 1
    assert rendered.shape == image.shape


def test_company_gt_mapping_uses_center_dimensions_and_ground_velocity():
    converter = load_converter()
    payload = {
        "objects": [{
            "type": 3,
            "source": 1,
            "center.x": 12.0,
            "center.y": -1.5,
            "center.z": 0.8,
            "length": 4.5,
            "width": 1.9,
            "height": 1.6,
            "obj_yaw": 0.25,
            "project_velocity.x": 7.0,
            "project_velocity.y": -0.5,
        }, {
            "type": 14,
            "source": 0,
            "center.x": 5.0,
            "center.y": 0.0,
            "center.z": 3.0,
            "length": 0.5,
            "width": 0.5,
            "height": 1.0,
            "obj_yaw": 0.0,
            "project_velocity.x": 0.0,
            "project_velocity.y": 0.0,
        }]
    }
    frame = converter.Frame(
        index=0, sample_id="sample", timestamp_us=0,
        image_path=Path("image"), radar_path=Path("radar"),
        lidar_path=Path("lidar"), gt_path=Path("gt"), gt=payload,
        ego2global=np.eye(4), car_twist=0.0)
    boxes, names, velocities, sources, audit = converter.convert_gt(frame)
    assert np.allclose(boxes, [[12.0, -1.5, 0.8, 4.5, 1.9, 1.6, 0.25]])
    assert names.tolist() == ["car"]
    assert np.allclose(velocities, [[7.0, -0.5]])
    assert sources.tolist() == [1]
    assert audit["ignored_types"] == {14: 1}


def test_radar_stationary_forward_point_is_zero_after_host_compensation():
    converter = load_converter()
    # Pick a radar-frame ray aligned with vehicle +X so a stationary target's
    # relative radial speed exactly cancels the host projection.
    direction_radar = converter.T_RADAR_TO_VEHICLE[:3, :3].T @ np.asarray(
        [1.0, 0.0, 0.0])
    xyz = direction_radar * 20.0
    points = np.asarray([[xyz[0], xyz[1], xyz[2], -10.0, 5.0]])
    names = ["x", "y", "z", "rspDetVelocity", "rspDetRCS"]
    converted = converter.convert_radar(points, names, car_twist=10.0)
    assert np.linalg.norm(converted[0, 4:6]) < 1e-5
    assert converted[0, 3] == 5.0


def test_radar_microsecond_timestamp_applies_sync_difference():
    converter = load_converter()
    timestamp, audit = converter.aligned_radar_timestamp_us({
        "rspTimestamp": "1758238003300083",
        "syncTimediff": "28802088411108",
    }, gt_timestamp_us=1787040091710000)
    assert timestamp == 1787040091711191
    assert audit["radar_timestamp_unit"] == "microseconds"
    assert np.isclose(audit["radar_to_gt_difference_ms"], 1.191)


def test_company_q_configs_keep_the_five_query_layouts():
    expected = {
        1: (900, 30, 30, 1.0),
        2: (900, 30, 30, 1.5),
        3: (900, 30, 30, 2.0),
        4: (900, 45, 20, 2.0),
        5: (1200, 30, 40, 2.0),
    }
    for q, values in expected.items():
        namespace = {}
        path = ROOT / "configs" / "3dh_query_company_20260818_q{}.py".format(q)
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        head = namespace["model"]["pts_bbox_head"]
        assert (head["num_query"], head["num_clusters"],
                head["transformer"]["num_ray"],
                head["query_distance_power"]) == values
