#!/usr/bin/env python3
"""Convert company tri-modal frames into RaCFormer-style info files.

This is a first-pass bridge script for data that already has images, radar
PLY, LiDAR PLY, calibration, timestamps, sweeps, and optional GT. It does not
try to fabricate the full nuScenes database. Instead it writes a compact
`*_infos_sweep.pkl` that a custom RaCFormer loader can consume.

The current RaCFormer pipeline still hardcodes six nuScenes cameras, five radar
channels, and calls NuScenes.get() for radar aggregation. Use this converter as
the data-prep half; the loader still needs a small custom path to read the
generated radar files directly.

Example with a manifest:
    python tools/convert_company_to_racformer.py \
        --manifest /path/to/frames.csv \
        --out-root /path/to/racformer_company \
        --repeat-single-camera-to-six \
        --repeat-single-radar-to-five

Manifest columns:
    sample_id,timestamp,image_path,radar_ply,lidar_ply,gt_path,ego_pose_path

`timestamp` may be seconds, milliseconds, microseconds, or nanoseconds. It is
normalized to nuScenes-like integer microseconds.
"""

import argparse
import csv
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_K = np.array(
    [
        [1481.62, 0.0, 979.99],
        [0.0, 1491.83, 502.51],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

DEFAULT_T_RADAR_TO_CAMERA = np.array(
    [
        [0.923739, -0.380076, 0.047419, -1.0],
        [0.016124, -0.085105, -0.996241, -1.9],
        [0.382683, 0.921032, -0.072487, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

NUSC_CAM_KEYS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUSC_RADAR_KEYS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

NUSC_CLASSES = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]


@dataclass
class FrameRecord:
    sample_id: str
    timestamp_us: int
    image_path: Path
    radar_ply: Path
    lidar_ply: Path
    gt_path: Optional[Path] = None
    ego2global: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RaCFormer-style custom info pkl files from company data."
    )
    src = parser.add_argument_group("inputs")
    src.add_argument("--manifest", type=Path, help="CSV with sample metadata.")
    src.add_argument("--image-dir", type=Path, help="Directory of camera images.")
    src.add_argument("--radar-dir", type=Path, help="Directory of radar PLY files.")
    src.add_argument("--lidar-dir", type=Path, help="Directory of LiDAR PLY files.")
    src.add_argument("--gt-dir", type=Path, help="Optional directory of GT json/txt files.")
    src.add_argument("--out-root", type=Path, required=True, help="Output dataset root.")

    calib = parser.add_argument_group("calibration")
    calib.add_argument("--intrinsic", type=Path, help="JSON/TXT/NPY camera intrinsic matrix.")
    calib.add_argument(
        "--radar-to-camera",
        type=Path,
        help="JSON/TXT/NPY 4x4 T_radar_to_camera. Defaults to current known value.",
    )
    calib.add_argument(
        "--lidar-to-camera",
        type=Path,
        help="JSON/TXT/NPY 4x4 T_lidar_to_camera. Defaults to T_radar_to_camera.",
    )
    calib.add_argument(
        "--radar-to-ego",
        type=Path,
        help="Optional direct 4x4 T_radar_to_ego. Derived from camera transforms otherwise.",
    )
    calib.add_argument(
        "--lidar-to-ego",
        type=Path,
        help="Optional 4x4 T_lidar_to_ego. Defaults to identity.",
    )
    calib.add_argument(
        "--radar-in-ego",
        action="store_true",
        help="Radar PLY coordinates are already ego coordinates; skip transformation.",
    )
    calib.add_argument(
        "--assume-radar-lidar-same-frame",
        action="store_true",
        help="Use identity radar->lidar. This should be enabled only if confirmed.",
    )

    split = parser.add_argument_group("split and sweeps")
    split.add_argument("--val-ratio", type=float, default=0.2)
    split.add_argument("--test-ratio", type=float, default=0.0)
    split.add_argument("--keyframe-stride", type=int, default=1)
    split.add_argument("--num-sweeps", type=int, default=7)
    split.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="Unit for manifest timestamps.",
    )

    compat = parser.add_argument_group("RaCFormer compatibility helpers")
    compat.add_argument("--camera-key", default="CAM_FRONT")
    compat.add_argument("--radar-key", default="RADAR_FRONT")
    compat.add_argument(
        "--repeat-single-camera-to-six",
        action="store_true",
        help="Duplicate one camera into six nuScenes camera keys for fixed pipelines.",
    )
    compat.add_argument(
        "--repeat-single-radar-to-five",
        action="store_true",
        help="Duplicate one radar into five nuScenes radar keys for fixed pipelines.",
    )

    io = parser.add_argument_group("point conversion")
    io.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images under out-root/images instead of referencing originals.",
    )
    io.add_argument(
        "--point-format",
        choices=["npy", "bin"],
        default="npy",
        help="Converted point cloud format.",
    )
    io.add_argument("--lidar-dim", type=int, default=5)
    io.add_argument("--radar-dim", type=int, default=7)
    return parser.parse_args()


def load_matrix(path: Optional[Path], default: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if path is None:
        return default.astype(np.float32)
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        text = path.read_text().strip()
        try:
            value = np.array(json.loads(text), dtype=np.float32)
        except json.JSONDecodeError:
            rows = [
                [float(item) for item in line.replace(",", " ").split()]
                for line in text.splitlines()
            ]
            value = np.array(rows, dtype=np.float32)
    value = value.reshape(shape).astype(np.float32)
    return value


def normalize_timestamp(value: str, unit: str) -> int:
    ts = float(value)
    if unit == "auto":
        if ts > 1e17:
            unit = "ns"
        elif ts > 1e14:
            unit = "us"
        elif ts > 1e11:
            unit = "ms"
        else:
            unit = "s"
    if unit == "s":
        return int(round(ts * 1_000_000))
    if unit == "ms":
        return int(round(ts * 1_000))
    if unit == "us":
        return int(round(ts))
    if unit == "ns":
        return int(round(ts / 1_000))
    raise ValueError(f"Unsupported timestamp unit: {unit}")


def read_manifest(path: Path, timestamp_unit: str) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "timestamp", "image_path", "radar_ply", "lidar_ply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            gt = row.get("gt_path") or None
            ego_pose = row.get("ego_pose_path") or None
            records.append(
                FrameRecord(
                    sample_id=row["sample_id"],
                    timestamp_us=normalize_timestamp(row["timestamp"], timestamp_unit),
                    image_path=Path(row["image_path"]),
                    radar_ply=Path(row["radar_ply"]),
                    lidar_ply=Path(row["lidar_ply"]),
                    gt_path=Path(gt) if gt else None,
                    ego2global=load_matrix(
                        Path(ego_pose) if ego_pose else None,
                        np.eye(4, dtype=np.float32), (4, 4)),
                )
            )
    return sorted(records, key=lambda rec: rec.timestamp_us)


def scan_dirs(args: argparse.Namespace) -> List[FrameRecord]:
    if not (args.image_dir and args.radar_dir and args.lidar_dir):
        raise ValueError("Provide either --manifest or --image-dir/--radar-dir/--lidar-dir.")
    images = sorted(
        p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    radars = sorted(p for p in args.radar_dir.iterdir() if p.suffix.lower() == ".ply")
    lidars = sorted(p for p in args.lidar_dir.iterdir() if p.suffix.lower() == ".ply")
    if not (len(images) == len(radars) == len(lidars)):
        raise ValueError(
            f"Directory counts differ: images={len(images)}, radars={len(radars)}, lidars={len(lidars)}"
        )

    records: List[FrameRecord] = []
    for idx, (img, radar, lidar) in enumerate(zip(images, radars, lidars)):
        sample_id = img.stem
        gt_path = None
        if args.gt_dir:
            for suffix in (".json", ".txt", ".csv"):
                candidate = args.gt_dir / f"{sample_id}{suffix}"
                if candidate.exists():
                    gt_path = candidate
                    break
        records.append(
            FrameRecord(
                sample_id=sample_id,
                timestamp_us=idx * 100_000,
                image_path=img,
                radar_ply=radar,
                lidar_ply=lidar,
                gt_path=gt_path,
                ego2global=np.eye(4, dtype=np.float32),
            )
        )
    return records


def parse_ply_header(f) -> Tuple[str, int, List[Tuple[str, str]], int]:
    raw_first = f.readline()
    first = raw_first.decode("ascii").strip()
    if first != "ply":
        raise ValueError("Not a PLY file.")
    fmt = ""
    vertex_count = 0
    properties: List[Tuple[str, str]] = []
    header_len = len(raw_first)
    in_vertex = False
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header ended unexpectedly.")
        header_len += len(raw)
        line = raw.decode("ascii").strip()
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            properties.append((parts[1], parts[2]))
        elif line == "end_header":
            return fmt, vertex_count, properties, header_len


PLY_DTYPE_MAP = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "u2",
    "uint16": "u2",
    "short": "i2",
    "int16": "i2",
    "uint": "u4",
    "uint32": "u4",
    "int": "i4",
    "int32": "i4",
}


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        fmt, vertex_count, properties, header_len = parse_ply_header(f)
        names = [name for _, name in properties]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data.astype(np.float32)
        if fmt not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format: {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + PLY_DTYPE_MAP[prop]) for prop, name in properties])
        f.seek(header_len)
        structured = np.fromfile(f, dtype=dtype, count=vertex_count)
        return np.column_stack([structured[name] for name in names]).astype(np.float32)


def pad_or_trim(points: np.ndarray, dim: int) -> np.ndarray:
    if points.shape[1] >= dim:
        return points[:, :dim].astype(np.float32)
    pad = np.zeros((points.shape[0], dim - points.shape[1]), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), pad], axis=1)


def convert_radar_points(points: np.ndarray, dim: int) -> np.ndarray:
    """Map raw radar PLY columns to RaCFormer-friendly columns.

    Current assumption:
        raw columns start with x, y, z, ...
        if there are 4 columns, column 3 is velocity or intensity-like
        if there are 5 columns, column 4 is rcs

    Final columns are:
        x, y, z, rcs, vx, vy, time_lag

    TODO: replace this mapping once the exact radar PLY property names and
    velocity definition are confirmed.
    """
    out = np.zeros((points.shape[0], dim), dtype=np.float32)
    out[:, : min(3, points.shape[1])] = points[:, : min(3, points.shape[1])]
    if dim > 3 and points.shape[1] >= 5:
        out[:, 3] = points[:, 4]
    if dim > 4 and points.shape[1] >= 4:
        out[:, 4] = points[:, 3]
    return out


def write_points(points: np.ndarray, path: Path, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, points.astype(np.float32))
    elif fmt == "bin":
        points.astype(np.float32).tofile(path)
    else:
        raise ValueError(fmt)
    return path


def maybe_copy_image(src: Path, out_root: Path, copy_images: bool) -> Path:
    if not copy_images:
        return src
    dst = out_root / "images" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def transform_info_from_lidar_to_camera(t_lidar_to_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t_camera_to_lidar = np.linalg.inv(t_lidar_to_camera)
    return t_camera_to_lidar[:3, :3].astype(np.float32), t_camera_to_lidar[:3, 3].astype(np.float32)


def make_cam_entry(
    image_path: Path,
    timestamp_us: int,
    k: np.ndarray,
    t_ego_to_camera: np.ndarray,
    ego2global_rotation: np.ndarray,
    ego2global_translation: np.ndarray,
) -> Dict:
    sensor2lidar_rotation, sensor2lidar_translation = transform_info_from_lidar_to_camera(
        t_ego_to_camera
    )
    # The model-facing coordinate frame is ego; legacy field names are kept so
    # existing RaCFormer projection code can consume the metadata.
    sensor2global_rotation = sensor2lidar_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2lidar_translation @ ego2global_rotation.T + ego2global_translation
    return {
        "data_path": str(image_path),
        "timestamp": int(timestamp_us),
        "cam_intrinsic": k.astype(np.float32),
        "sensor2lidar_rotation": sensor2lidar_rotation,
        "sensor2lidar_translation": sensor2lidar_translation,
        "sensor2global_rotation": sensor2global_rotation.astype(np.float32),
        "sensor2global_translation": sensor2global_translation.astype(np.float32),
    }


def load_gt(gt_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GT boxes if available.

    Supported JSON format:
        [
          {"box": [x, y, z, dx, dy, dz, yaw], "name": "car", "valid": true}
        ]

    GT is expected in current keyframe LiDAR coordinates. If your annotation
    tool exports camera/global boxes, convert them before writing the final pkl.
    """
    if gt_path is None or not gt_path.exists():
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.array([], dtype=object),
            np.zeros((0,), dtype=bool),
        )
    if gt_path.suffix.lower() == ".json":
        items = json.loads(gt_path.read_text())
        boxes, names, valid = [], [], []
        for item in items:
            boxes.append(item["box"])
            names.append(item.get("name", "car"))
            valid.append(bool(item.get("valid", True)))
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
            np.asarray(names, dtype=object),
            np.asarray(valid, dtype=bool),
        )
    raise ValueError(
        f"Unsupported GT format {gt_path.suffix}. Use JSON first, or extend load_gt()."
    )


def transform_boxes_to_ego(boxes: np.ndarray, lidar2ego: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or np.allclose(lidar2ego, np.eye(4), atol=1e-6):
        return boxes
    boxes = boxes.copy()
    centers = np.concatenate(
        [boxes[:, :3], np.ones((len(boxes), 1), dtype=np.float32)], axis=1)
    boxes[:, :3] = (centers @ lidar2ego.T)[:, :3]
    headings = np.stack(
        [np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))],
        axis=1)
    headings = headings @ lidar2ego[:3, :3].T
    boxes[:, 6] = np.arctan2(headings[:, 1], headings[:, 0])
    return boxes


def select_keyframes(records: Sequence[FrameRecord], stride: int) -> List[int]:
    return list(range(0, len(records), max(1, stride)))


def previous_indices(index: int, num_sweeps: int) -> List[int]:
    return [max(0, index - offset) for offset in range(1, num_sweeps + 1)]


def split_infos(infos: List[Dict], val_ratio: float, test_ratio: float) -> Dict[str, List[Dict]]:
    n = len(infos)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)
    return {
        "train": infos[:n_train],
        "val": infos[n_train : n_train + n_val],
        "test": infos[n_train + n_val :],
    }


def dump_info(path: Path, infos: List[Dict]) -> None:
    payload = {
        "infos": infos,
        "metadata": {
            "dataset": "company_custom",
            "info_version": "racformer_custom_v0",
            "classes": NUSC_CLASSES,
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    k = load_matrix(args.intrinsic, DEFAULT_K, (3, 3))
    t_radar_to_camera = load_matrix(args.radar_to_camera, DEFAULT_T_RADAR_TO_CAMERA, (4, 4))
    t_lidar_to_camera = load_matrix(args.lidar_to_camera, t_radar_to_camera, (4, 4))
    t_lidar_to_ego = load_matrix(
        args.lidar_to_ego, np.eye(4, dtype=np.float32), (4, 4))
    derived_radar_to_ego = (
        t_lidar_to_ego @ np.linalg.inv(t_lidar_to_camera) @ t_radar_to_camera)
    t_radar_to_ego = load_matrix(
        args.radar_to_ego, derived_radar_to_ego, (4, 4))
    t_ego_to_camera = t_lidar_to_camera @ np.linalg.inv(t_lidar_to_ego)
    radar_in_ego = args.radar_in_ego or args.assume_radar_lidar_same_frame
    if not radar_in_ego and args.radar_to_ego is None:
        print("INFO: derived T_radar_to_ego from camera calibration matrices.")

    records = read_manifest(args.manifest, args.timestamp_unit) if args.manifest else scan_dirs(args)
    if not records:
        raise ValueError("No frames found.")

    point_suffix = ".npy" if args.point_format == "npy" else ".bin"
    keyframe_ids = select_keyframes(records, args.keyframe_stride)
    infos: List[Dict] = []

    for index in keyframe_ids:
        rec = records[index]
        current_ego2global = rec.ego2global
        ego2global_translation = current_ego2global[:3, 3].astype(np.float32)
        ego2global_rotation = np.eye(3, dtype=np.float32)
        lidar2ego_translation = t_lidar_to_ego[:3, 3].astype(np.float32)
        lidar2ego_rotation = np.eye(3, dtype=np.float32)

        lidar_points = pad_or_trim(read_ply_vertices(rec.lidar_ply), args.lidar_dim)
        radar_points = convert_radar_points(read_ply_vertices(rec.radar_ply), args.radar_dim)

        lidar_path = write_points(
            lidar_points,
            args.out_root / "lidar" / f"{rec.sample_id}{point_suffix}",
            args.point_format,
        )
        radar_path = write_points(
            radar_points,
            args.out_root / "radar" / f"{rec.sample_id}{point_suffix}",
            args.point_format,
        )
        image_path = maybe_copy_image(rec.image_path, args.out_root, args.copy_images)

        cam_entry = make_cam_entry(
            image_path,
            rec.timestamp_us,
            k,
            t_ego_to_camera,
            ego2global_rotation,
            ego2global_translation,
        )
        cam_keys = NUSC_CAM_KEYS if args.repeat_single_camera_to_six else [args.camera_key]
        radar_keys = NUSC_RADAR_KEYS if args.repeat_single_radar_to_five else [args.radar_key]
        cams = {key: dict(cam_entry) for key in cam_keys}
        rads = {
            key: {
                "data_path": str(radar_path),
                "timestamp": int(rec.timestamp_us),
                "radar_in_ego": radar_in_ego,
                "radar2ego": t_radar_to_ego.astype(np.float32),
            }
            for key in radar_keys
        }

        sweeps = []
        for prev_idx in previous_indices(index, args.num_sweeps):
            prev = records[prev_idx]
            current_ego_to_prev_ego = (
                np.linalg.inv(prev.ego2global) @ current_ego2global)
            prev_ego_to_current_ego = np.linalg.inv(current_ego_to_prev_ego)
            prev_ego_to_camera = t_ego_to_camera
            current_ego_to_prev_camera = (
                prev_ego_to_camera @ current_ego_to_prev_ego)
            prev_image = maybe_copy_image(prev.image_path, args.out_root, args.copy_images)
            prev_cam_entry = make_cam_entry(
                prev_image,
                prev.timestamp_us,
                k,
                current_ego_to_prev_camera,
                ego2global_rotation,
                ego2global_translation,
            )
            prev_radar_path = args.out_root / "radar" / f"{prev.sample_id}{point_suffix}"
            if not prev_radar_path.exists():
                prev_radar_points = convert_radar_points(
                    read_ply_vertices(prev.radar_ply), args.radar_dim
                )
                write_points(prev_radar_points, prev_radar_path, args.point_format)
            sweep = {key: dict(prev_cam_entry) for key in cam_keys}
            raw_radar_to_prev_ego = (
                np.eye(4, dtype=np.float32) if radar_in_ego
                else t_radar_to_ego)
            raw_radar_to_current_ego = (
                prev_ego_to_current_ego @ raw_radar_to_prev_ego)
            sweep_radar_in_current_ego = np.allclose(
                raw_radar_to_current_ego, np.eye(4), atol=1e-6)
            for key in radar_keys:
                sweep[key] = {
                    "data_path": str(prev_radar_path),
                    "timestamp": int(prev.timestamp_us),
                    "radar_in_ego": sweep_radar_in_current_ego,
                    "radar2ego": raw_radar_to_current_ego.astype(np.float32),
                }
            sweeps.append(sweep)

        gt_boxes, gt_names, valid_flag = load_gt(rec.gt_path)
        gt_boxes = transform_boxes_to_ego(gt_boxes, t_lidar_to_ego)
        infos.append(
            {
                "token": rec.sample_id,
                "timestamp": int(rec.timestamp_us),
                "lidar_path": str(lidar_path),
                "lidar_in_ego": args.lidar_to_ego is None,
                "lidar2ego": t_lidar_to_ego.astype(np.float32),
                "radar_path": str(radar_path),
                "ego2global_translation": ego2global_translation,
                "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego2global": current_ego2global.astype(np.float32),
                "lidar2ego_translation": lidar2ego_translation,
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "cams": cams,
                "rads": rads,
                "sweeps": sweeps,
                "gt_boxes": gt_boxes,
                "gt_names": gt_names,
                "gt_velocity": np.zeros((gt_boxes.shape[0], 2), dtype=np.float32),
                "num_lidar_pts": np.zeros((gt_boxes.shape[0],), dtype=np.int32),
                "num_radar_pts": np.zeros((gt_boxes.shape[0],), dtype=np.int32),
                "valid_flag": valid_flag,
            }
        )

    splits = split_infos(infos, args.val_ratio, args.test_ratio)
    for split_name, split_infos_list in splits.items():
        dump_info(args.out_root / f"custom_infos_{split_name}_sweep.pkl", split_infos_list)

    print(f"Converted {len(infos)} keyframes from {len(records)} raw frames.")
    print(f"Output root: {args.out_root}")
    for split_name, split_infos_list in splits.items():
        print(f"  {split_name}: {len(split_infos_list)} samples")
    print("NOTE: GT is empty unless per-sample JSON files were provided.")
    if all(np.allclose(record.ego2global, np.eye(4)) for record in records):
        print("NOTE: all ego poses are identity; temporal motion compensation is disabled.")


if __name__ == "__main__":
    main()
