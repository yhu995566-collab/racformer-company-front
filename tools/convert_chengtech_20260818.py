#!/usr/bin/env python3
"""Convert the synchronized 2026-08-18 ChengTech capture for RaCFormer.

The input contract is deliberately strict: frame ``N`` is composed from the
root-level ``lidar_middle/N.pcd`` and the camera/radar/GT files whose suffix is
``-NNNNNN``.  Subdirectories below ``lidar_middle`` are never scanned.
"""

import argparse
import json
import pickle
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


CAMERA_K = np.asarray([
    [1016.6968662245, 0.0, 965.0321173800],
    [0.0, 1016.5654215078, 537.9014922432],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
CAMERA_DISTORTION = np.asarray([
    -0.1342809090, -0.2376811870, -0.0001393982, 0.0000050977,
    -0.0058570910, 0.2581241099, -0.3799765692, -0.0521208096,
], dtype=np.float64)
CAMERA_IMAGE_SIZE = (1920, 1080)

# Confirmed camera -> vehicle pose, Euler R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
CAMERA_TRANSLATION = (1.067547570351529, 0.02643990469655807,
                      1.78008907572419)
CAMERA_RPY = (1.607537010631503, 3.140413538237299,
              1.569174828378515)

# Common radar -> vehicle pose for all supplied radar PLY files.
RADAR_TRANSLATION = (0.1051289439201355, -0.1236444115638733,
                     -0.04894282296299934)
RADAR_RPY = (0.006052746437489986, -0.0007436465626268648,
             -0.00304712844081223)

CLASS_NAMES = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)
COMPANY_TYPE_TO_CLASS = {
    1: "pedestrian",
    2: "bicycle",
    3: "car",
    4: "truck",
    5: "bus",
    6: "truck",
    7: "construction_vehicle",
    8: "bicycle",
    11: "traffic_cone",
    12: "traffic_cone",
    13: "truck",
}


@dataclass(frozen=True)
class Frame:
    index: int
    sample_id: str
    timestamp_us: int
    image_path: Path
    radar_path: Path
    lidar_path: Path
    gt_path: Path
    gt: Dict
    ego2global: np.ndarray
    car_twist: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=1216)
    parser.add_argument("--limit", type=int,
                        help="Convert only the first N frames for an audit run")
    parser.add_argument("--num-sweeps", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-undistort", action="store_true",
                        help="Audit-only: reference original distorted JPEGs")
    return parser.parse_args()


def euler_transform(translation: Sequence[float],
                    rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


T_CAMERA_TO_VEHICLE = euler_transform(CAMERA_TRANSLATION, CAMERA_RPY)
T_VEHICLE_TO_CAMERA = np.linalg.inv(T_CAMERA_TO_VEHICLE)
T_RADAR_TO_VEHICLE = euler_transform(RADAR_TRANSLATION, RADAR_RPY)


def _lzf_decompress(payload: bytes, expected_size: int) -> bytes:
    """Pure-Python LZF decompressor for PCD ``binary_compressed`` payloads."""
    output = bytearray()
    cursor = 0
    while cursor < len(payload):
        control = payload[cursor]
        cursor += 1
        if control < 32:
            length = control + 1
            end = cursor + length
            if end > len(payload):
                raise ValueError("truncated LZF literal run")
            output.extend(payload[cursor:end])
            cursor = end
            continue

        length = control >> 5
        reference = len(output) - ((control & 0x1F) << 8) - 1
        if length == 7:
            if cursor >= len(payload):
                raise ValueError("truncated LZF extended length")
            length += payload[cursor]
            cursor += 1
        if cursor >= len(payload):
            raise ValueError("truncated LZF back reference")
        reference -= payload[cursor]
        cursor += 1
        length += 2
        if reference < 0:
            raise ValueError("invalid LZF back reference")
        for _ in range(length):
            output.append(output[reference])
            reference += 1

    if len(output) != expected_size:
        raise ValueError(
            "PCD LZF size mismatch: expected {}, decoded {}".format(
                expected_size, len(output)))
    return bytes(output)


def _pcd_dtype(type_code: str, size: int) -> np.dtype:
    table = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("U", 1): np.dtype("u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("I", 1): np.dtype("i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
    }
    try:
        return table[(type_code.upper(), int(size))]
    except KeyError as error:
        raise ValueError(
            "unsupported PCD TYPE/SIZE: {}/{}".format(type_code, size)) \
            from error


def read_binary_compressed_pcd(path: Path) -> Dict[str, np.ndarray]:
    """Read PCD v0.7 while preserving every field's declared dtype."""
    header: Dict[str, List[str]] = {}
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("{} has no DATA line".format(path))
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        if header["DATA"] != ["binary_compressed"]:
            raise ValueError(
                "{} must use DATA binary_compressed".format(path))
        sizes = stream.read(8)
        if len(sizes) != 8:
            raise ValueError("{} has truncated compressed sizes".format(path))
        compressed_size, uncompressed_size = struct.unpack("<II", sizes)
        compressed = stream.read(compressed_size)
        if len(compressed) != compressed_size:
            raise ValueError("{} has truncated compressed payload".format(path))

    fields = header.get("FIELDS") or header.get("FIELD")
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get(
        "COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("inconsistent PCD field metadata in {}".format(path))

    decoded = _lzf_decompress(compressed, uncompressed_size)
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for name, size, type_code, count in zip(fields, sizes, types, counts):
        dtype = _pcd_dtype(type_code, size)
        byte_count = points * count * size
        field_data = np.frombuffer(
            decoded, dtype=dtype, count=points * count, offset=offset).copy()
        offset += byte_count
        if count != 1:
            field_data = field_data.reshape(points, count)
        result[name] = field_data
    if offset != len(decoded):
        raise ValueError("PCD decoded byte count does not match fields")
    return result


def read_ply_vertices(path: Path) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    properties: List[Tuple[str, str]] = []
    comments: Dict[str, str] = {}
    vertex_count = 0
    with path.open("rb") as stream:
        if stream.readline().decode("ascii").strip() != "ply":
            raise ValueError("{} is not PLY".format(path))
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("format ") and line.split()[1] != "ascii":
                raise ValueError("only ASCII radar PLY is supported")
            if line.startswith("comment "):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    comments[parts[1]] = parts[2]
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            elif line.startswith("property "):
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        data = np.loadtxt(stream, max_rows=vertex_count, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, [name for _, name in properties], comments


def aligned_radar_timestamp_us(comments: Dict[str, str],
                               gt_timestamp_us: int) -> Tuple[int, Dict]:
    """Apply the PLY sync offset; both radar header values are microseconds."""
    try:
        raw_timestamp_us = int(comments["rspTimestamp"])
        sync_difference_us = int(comments["syncTimediff"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "radar PLY requires integer rspTimestamp and syncTimediff comments"
        ) from error
    aligned_timestamp_us = raw_timestamp_us + sync_difference_us
    difference_us = aligned_timestamp_us - gt_timestamp_us
    if abs(difference_us) > 250000:
        raise ValueError(
            "aligned radar timestamp differs from GT by {:.3f} ms".format(
                difference_us / 1000.0))
    return aligned_timestamp_us, {
        "radar_timestamp_unit": "microseconds",
        "radar_raw_timestamp_us": raw_timestamp_us,
        "radar_sync_difference_us": sync_difference_us,
        "radar_aligned_timestamp_us": aligned_timestamp_us,
        "radar_to_gt_difference_ms": difference_us / 1000.0,
    }


def _field(points: np.ndarray, names: Sequence[str],
           aliases: Iterable[str]) -> np.ndarray:
    indices = {name.lower(): index for index, name in enumerate(names)}
    for alias in aliases:
        if alias.lower() in indices:
            return points[:, indices[alias.lower()]]
    raise ValueError("missing fields {}; available={}".format(
        list(aliases), list(names)))


def convert_radar(points: np.ndarray, names: Sequence[str],
                  car_twist: float) -> np.ndarray:
    xyz_radar = np.column_stack([
        _field(points, names, ("x",)),
        _field(points, names, ("y",)),
        _field(points, names, ("z",)),
    ])
    rcs = _field(points, names, ("rspDetRCS", "rsp_velocity_rcs", "rcs"))
    relative_radial = _field(
        points, names, ("rspDetVelocity", "rsp_velocity", "v"))

    radius = np.linalg.norm(xyz_radar, axis=1)
    direction_radar = np.zeros_like(xyz_radar)
    valid = radius > 1e-6
    direction_radar[valid] = xyz_radar[valid] / radius[valid, None]
    host_vehicle = np.asarray([car_twist, 0.0, 0.0], dtype=np.float64)
    host_radar = T_RADAR_TO_VEHICLE[:3, :3].T @ host_vehicle
    absolute_radial = relative_radial + direction_radar @ host_radar
    velocity_radar = direction_radar * absolute_radial[:, None]

    xyz1 = np.column_stack([xyz_radar, np.ones(len(xyz_radar))])
    xyz_vehicle = (xyz1 @ T_RADAR_TO_VEHICLE.T)[:, :3]
    velocity_vehicle = velocity_radar @ T_RADAR_TO_VEHICLE[:3, :3].T
    result = np.zeros((len(points), 7), dtype=np.float32)
    result[:, :3] = xyz_vehicle
    result[:, 3] = rcs
    result[:, 4:6] = velocity_vehicle[:, :2]
    return result


def parse_pose(frame_info: Dict) -> np.ndarray:
    return euler_transform(
        (frame_info["pose.pos.x"], frame_info["pose.pos.y"],
         frame_info["pose.pos.z"]),
        (frame_info["roll"], frame_info["pitch"], frame_info["yaw"]))


def discover_frames(data_root: Path, count: int) -> List[Frame]:
    image_files = sorted((data_root / "front120_camera").glob("*.jpeg"))
    radar_files = sorted((data_root / "radar_front").glob("*.ply"))
    gt_files = sorted((data_root / "GT").glob("*.json"))
    if not (len(image_files) == len(radar_files) == len(gt_files) == count):
        raise ValueError(
            "expected {0} image/radar/GT files; got image={1}, radar={2}, "
            "GT={3}".format(count, len(image_files), len(radar_files),
                            len(gt_files)))

    frames: List[Frame] = []
    for index in range(count):
        suffix = "-{:06d}".format(index)
        image, radar, gt_path = image_files[index], radar_files[index], gt_files[index]
        if not all(path.stem.endswith(suffix) for path in (image, radar, gt_path)):
            raise ValueError("frame {} suffix mismatch: {}, {}, {}".format(
                index, image.name, radar.name, gt_path.name))
        lidar = data_root / "lidar_middle" / "{}.pcd".format(index)
        if not lidar.is_file():
            raise FileNotFoundError(str(lidar))
        gt = json.loads(gt_path.read_text())
        if int(gt["frame_num"]) != index:
            raise ValueError("{} declares frame_num {}".format(
                gt_path, gt["frame_num"]))
        frames.append(Frame(
            index=index,
            sample_id=image.stem,
            timestamp_us=int(round(float(gt["stamp_sec"]) * 1e6)),
            image_path=image.resolve(),
            radar_path=radar.resolve(),
            lidar_path=lidar.resolve(),
            gt_path=gt_path.resolve(),
            gt=gt,
            ego2global=parse_pose(gt["frame_info"]),
            car_twist=float(gt["frame_info"]["car_twist"]),
        ))
    timestamps = np.asarray([frame.timestamp_us for frame in frames])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("GT timestamps must be strictly increasing")
    return frames


def prepare_image(frame: Frame, out_root: Path, quality: int,
                  skip_undistort: bool) -> Path:
    if skip_undistort:
        return frame.image_path
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for image undistortion") from error
    output = out_root / "images_undistorted" / (frame.sample_id + ".jpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        expected_shape = (CAMERA_IMAGE_SIZE[1], CAMERA_IMAGE_SIZE[0])
        if image is None or image.shape[:2] != expected_shape:
            raise ValueError("unexpected image shape for {}: {}".format(
                frame.image_path, None if image is None else image.shape))
        rectified = cv2.undistort(
            image, CAMERA_K, CAMERA_DISTORTION, None, CAMERA_K)
        if not cv2.imwrite(str(output), rectified,
                           [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise IOError("failed to write {}".format(output))
    return output.resolve()


def convert_lidar(frame: Frame, out_root: Path) -> Tuple[Path, Dict]:
    fields = read_binary_compressed_pcd(frame.lidar_path)
    required = {"x", "y", "z", "intensity", "ring", "timestamp"}
    if set(fields) != required:
        raise ValueError("{} fields must be exactly {}; got {}".format(
            frame.lidar_path, sorted(required), sorted(fields)))
    count = len(fields["x"])
    if count != 230400 or any(len(value) != count for value in fields.values()):
        raise ValueError("{} must contain 230400 complete points".format(
            frame.lidar_path))
    if fields["ring"].dtype != np.dtype("<u2"):
        raise ValueError("ring dtype must be uint16")
    if fields["timestamp"].dtype != np.dtype("<f8"):
        raise ValueError("timestamp dtype must be float64")
    output_points = np.column_stack([
        fields["x"], fields["y"], fields["z"], fields["intensity"],
        fields["ring"].astype(np.float32),
    ]).astype(np.float32)
    output = out_root / "lidar" / (frame.sample_id + ".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, output_points)
    timestamp_raw = fields["timestamp"]
    frame_seconds = frame.timestamp_us / 1e6
    raw_median = float(np.median(timestamp_raw))
    if abs(raw_median - frame_seconds) <= 2.0:
        timestamp_unit = "seconds"
        timestamp_seconds = timestamp_raw
    elif abs(raw_median / 1e9 - frame_seconds) <= 2.0:
        timestamp_unit = "nanoseconds"
        timestamp_seconds = timestamp_raw / 1e9
    else:
        raise ValueError(
            "{} point timestamps match neither seconds nor nanoseconds "
            "relative to GT stamp_sec".format(frame.lidar_path))
    audit = {
        "points": count,
        "ring_min": int(fields["ring"].min()),
        "ring_max": int(fields["ring"].max()),
        "timestamp_detected_unit": timestamp_unit,
        "timestamp_raw_min": float(timestamp_raw.min()),
        "timestamp_raw_max": float(timestamp_raw.max()),
        "timestamp_sec_min": float(timestamp_seconds.min()),
        "timestamp_sec_max": float(timestamp_seconds.max()),
        "intensity_min": float(fields["intensity"].min()),
        "intensity_max": float(fields["intensity"].max()),
    }
    return output.resolve(), audit


def convert_gt(frame: Frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, Dict]:
    boxes, names, velocities, sources = [], [], [], []
    ignored: Dict[int, int] = {}
    for item in frame.gt["objects"]:
        company_type = int(item["type"])
        name = COMPANY_TYPE_TO_CLASS.get(company_type)
        if name is None:
            ignored[company_type] = ignored.get(company_type, 0) + 1
            continue
        box = [
            item["center.x"], item["center.y"], item["center.z"],
            item["length"], item["width"], item["height"], item["obj_yaw"],
        ]
        velocity = [item["project_velocity.x"], item["project_velocity.y"]]
        if not np.isfinite(box + velocity).all() or min(box[3:6]) <= 0:
            continue
        boxes.append(box)
        names.append(name)
        velocities.append(velocity)
        sources.append(int(item["source"]))
    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    names_array = np.asarray(names, dtype=object)
    velocities_array = np.asarray(velocities, dtype=np.float32).reshape(-1, 2)
    sources_array = np.asarray(sources, dtype=np.int8)
    duplicate_candidates = 0
    for left in np.flatnonzero(sources_array == 0):
        for right in np.flatnonzero(sources_array == 1):
            if names_array[left] == names_array[right] and \
                    np.linalg.norm(boxes_array[left, :2] -
                                   boxes_array[right, :2]) < 1.0:
                duplicate_candidates += 1
    return boxes_array, names_array, velocities_array, sources_array, {
        "ignored_types": ignored,
        "duplicate_cross_source_candidates": duplicate_candidates,
    }


def relative_path(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        # With --skip-undistort the original JPEG remains outside out_root.
        return str(path)


def camera_entry(image: Path, timestamp_us: int,
                 current_to_camera_vehicle: np.ndarray) -> Dict:
    projection = np.eye(4, dtype=np.float64)
    projection[:3, :3] = CAMERA_K
    projection = projection @ T_VEHICLE_TO_CAMERA @ current_to_camera_vehicle
    return {
        "data_path": str(image),
        "timestamp": int(timestamp_us),
        "cam_intrinsic": CAMERA_K.astype(np.float32),
        "lidar2img": projection.astype(np.float32),
    }


def split_infos(infos: List[Dict], val_ratio: float,
                test_ratio: float) -> Dict[str, List[Dict]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("ratios must be non-negative and sum to less than one")
    count = len(infos)
    test_count = int(round(count * test_ratio))
    val_count = int(round(count * val_ratio))
    train_count = count - val_count - test_count
    return {
        "train": infos[:train_count],
        "val": infos[train_count:train_count + val_count],
        "test": infos[train_count + val_count:],
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(data_root, args.num_frames)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        frames = frames[:args.limit]

    converted: Dict[int, Dict] = {}
    audits = []
    for frame in tqdm(frames, desc="Converting synchronized frames"):
        image = prepare_image(
            frame, out_root, args.jpeg_quality, args.skip_undistort)
        lidar, lidar_audit = convert_lidar(frame, out_root)
        radar_raw, radar_names, radar_comments = read_ply_vertices(
            frame.radar_path)
        radar_timestamp_us, radar_audit = aligned_radar_timestamp_us(
            radar_comments, frame.timestamp_us)
        radar = convert_radar(radar_raw, radar_names, frame.car_twist)
        radar_path = out_root / "radar" / (frame.sample_id + ".npy")
        radar_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(radar_path, radar)
        boxes, names, velocities, sources, gt_audit = convert_gt(frame)
        converted[frame.index] = {
            "image": image,
            "lidar": lidar,
            "radar": radar_path.resolve(),
            "radar_timestamp_us": radar_timestamp_us,
            "boxes": boxes,
            "names": names,
            "velocities": velocities,
            "sources": sources,
        }
        audits.append({
            "frame": frame.index,
            "lidar": lidar_audit,
            "radar_points": int(len(radar)),
            **radar_audit,
            "gt_boxes": int(len(boxes)),
            "class_counts": dict(Counter(names.tolist())),
            "gt_source_0": int(np.count_nonzero(sources == 0)),
            "gt_source_1": int(np.count_nonzero(sources == 1)),
            **gt_audit,
        })

    infos = []
    for local_index, frame in enumerate(frames):
        item = converted[frame.index]
        current_camera = camera_entry(
            Path(relative_path(item["image"], out_root)), frame.timestamp_us,
            np.eye(4))
        sweeps = []
        for offset in range(1, args.num_sweeps + 1):
            previous_local_index = max(0, local_index - offset)
            previous = frames[previous_local_index]
            previous_item = converted[previous.index]
            current_to_previous = np.linalg.inv(
                previous.ego2global) @ frame.ego2global
            previous_to_current = np.linalg.inv(
                frame.ego2global) @ previous.ego2global
            sweeps.append({
                "CAM_FRONT": camera_entry(
                    Path(relative_path(previous_item["image"], out_root)),
                    previous.timestamp_us, current_to_previous),
                "RADAR_FRONT": {
                    "data_path": relative_path(previous_item["radar"], out_root),
                    "timestamp": int(previous_item["radar_timestamp_us"]),
                    "radar_in_ego": bool(np.allclose(
                        previous_to_current, np.eye(4), atol=1e-6)),
                    "radar2ego": previous_to_current.astype(np.float32),
                },
            })
        infos.append({
            "token": frame.sample_id,
            "timestamp": int(frame.timestamp_us),
            "lidar_path": relative_path(item["lidar"], out_root),
            "lidar_in_ego": True,
            "lidar2ego": np.eye(4, dtype=np.float32),
            "radar_path": relative_path(item["radar"], out_root),
            "ego2global": frame.ego2global.astype(np.float32),
            "ego2global_translation": frame.ego2global[:3, 3].astype(np.float32),
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "lidar2ego_translation": np.zeros(3, dtype=np.float32),
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "cams": {"CAM_FRONT": current_camera},
            "rads": {"RADAR_FRONT": {
                "data_path": relative_path(item["radar"], out_root),
                "timestamp": int(item["radar_timestamp_us"]),
                "radar_in_ego": True,
                "radar2ego": np.eye(4, dtype=np.float32),
            }},
            "sweeps": sweeps,
            "gt_boxes": item["boxes"],
            "gt_names": item["names"],
            "gt_velocity": item["velocities"],
            "gt_sources": item["sources"],
            "num_lidar_pts": np.zeros(len(item["boxes"]), dtype=np.int32),
            "num_radar_pts": np.zeros(len(item["boxes"]), dtype=np.int32),
            "valid_flag": np.ones(len(item["boxes"]), dtype=bool),
        })

    splits = split_infos(infos, args.val_ratio, args.test_ratio)
    metadata = {
        "dataset": "chengtech_20260818",
        "info_version": "racformer_chengtech_20260818_v1",
        "classes": CLASS_NAMES,
        "source_frames": len(frames),
        "undistorted": not args.skip_undistort,
    }
    for name, split in splits.items():
        with (out_root / "custom_infos_{}_sweep.pkl".format(name)).open("wb") as stream:
            pickle.dump({"infos": split, "metadata": metadata}, stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
    class_counts = Counter()
    ignored_type_counts = Counter()
    for audit in audits:
        class_counts.update(audit["class_counts"])
        ignored_type_counts.update(audit["ignored_types"])
    timestamp_deltas_ms = np.diff(
        np.asarray([frame.timestamp_us for frame in frames], dtype=np.int64)
    ) / 1000.0
    summary = {
        "source_root": str(data_root),
        "output_root": str(out_root),
        "frames": len(frames),
        "num_sweeps": args.num_sweeps,
        "split_counts": {name: len(value) for name, value in splits.items()},
        "gt_boxes": int(sum(item["gt_boxes"] for item in audits)),
        "class_counts": dict(sorted(class_counts.items())),
        "ignored_type_counts": dict(sorted(ignored_type_counts.items())),
        "gt_source_0": int(sum(item["gt_source_0"] for item in audits)),
        "gt_source_1": int(sum(item["gt_source_1"] for item in audits)),
        "duplicate_cross_source_candidates": int(sum(
            item["duplicate_cross_source_candidates"] for item in audits)),
        "camera_intrinsic": CAMERA_K.tolist(),
        "camera_distortion": CAMERA_DISTORTION.tolist(),
        "camera_to_vehicle": T_CAMERA_TO_VEHICLE.tolist(),
        "radar_to_vehicle": T_RADAR_TO_VEHICLE.tolist(),
        "frame_interval_ms": {
            "min": float(timestamp_deltas_ms.min()) if len(timestamp_deltas_ms) else None,
            "median": float(np.median(timestamp_deltas_ms)) if len(timestamp_deltas_ms) else None,
            "max": float(timestamp_deltas_ms.max()) if len(timestamp_deltas_ms) else None,
        },
        "first_frame_audit": audits[0],
        "last_frame_audit": audits[-1],
    }
    (out_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
