#!/usr/bin/env python3
"""Convert company tri-modal frames into RaCFormer-style info files.

This bridge consumes the complete temporal sequence, but emits training
samples only for labeled keyframes. Unlabeled frames remain available as
camera/radar sweeps. Missing ego poses are represented by identity matrices,
which deliberately disables temporal motion compensation.

Example with a manifest:
    python tools/convert_company_to_racformer.py \
        --manifest /path/to/frames.csv \
        --calibration /path/to/company_calibration_20260630.json \
        --out-root /path/to/racformer_company

Manifest columns:
    sample_id,timestamp,image_path,radar_path,lidar_path,gt_path,is_keyframe,ego_pose_path

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
from tqdm import tqdm


DEFAULT_K = np.array(
    [
        [524.9667, 0.0, 329.8176],
        [0.0, 529.8227, 237.9376],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

DEFAULT_T_CAMERA_TO_LIDAR = np.array(
    [
        [0.984808, 0.166921, 0.047864, 0.0],
        [0.0, 0.275637, -0.961262, 0.0],
        [-0.173648, 0.946658, 0.271450, -2.9],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# Euler convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), radians.
DEFAULT_T_RADAR_TO_LIDAR = np.array(
    [
        [0.999487639, -0.031978269, 0.001360151, 0.0501],
        [0.031994526, 0.999384833, -0.014363338, 0.4530],
        [-0.000900000, 0.014399497, 0.999895917, 0.6756],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# Native LiDAR axes: +x right, +y backward, +z downward.
# Canonical model ego axes: +X forward, +Y left, +Z upward.
DEFAULT_T_LIDAR_TO_EGO = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
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
    is_keyframe: bool = False


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
    calib.add_argument(
        "--calibration",
        type=Path,
        help="Combined company calibration JSON containing camera, radar, and lidar matrices.",
    )
    calib.add_argument("--intrinsic", type=Path, help="JSON/TXT/NPY camera intrinsic matrix.")
    calib.add_argument(
        "--camera-to-lidar",
        type=Path,
        help="JSON/TXT/NPY 4x4 T_camera_to_lidar. Defaults to the 2026-06-30 calibration.",
    )
    calib.add_argument(
        "--radar-to-lidar",
        type=Path,
        help="JSON/TXT/NPY 4x4 T_radar_to_lidar. Defaults to the current radar-front calibration.",
    )
    calib.add_argument(
        "--lidar-to-camera",
        type=Path,
        help="Optional direct T_lidar_to_camera. Defaults to inverse(T_camera_to_lidar).",
    )
    calib.add_argument(
        "--radar-to-ego",
        type=Path,
        help="Optional direct 4x4 T_radar_to_ego. Derived from camera transforms otherwise.",
    )
    calib.add_argument(
        "--lidar-to-ego",
        type=Path,
        help="Optional T_lidar_to_ego. Defaults to x-right/y-back/z-down -> x-front/y-left/z-up.",
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
    split.add_argument("--frame-rate", type=float, default=4.0)
    split.add_argument(
        "--all-frames-as-keyframes",
        action="store_true",
        help="Use every frame as a keyframe. By default only labeled/marked frames are used.",
    )
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


def load_calibration_bundle(path: Optional[Path]) -> Dict[str, np.ndarray]:
    values = {
        "intrinsic": DEFAULT_K,
        "camera_to_lidar": DEFAULT_T_CAMERA_TO_LIDAR,
        "radar_to_lidar": DEFAULT_T_RADAR_TO_LIDAR,
        "lidar_to_ego": DEFAULT_T_LIDAR_TO_EGO,
    }
    if path is None:
        return values

    payload = json.loads(path.read_text())
    values.update({
        "intrinsic": np.asarray(
            payload["camera"]["intrinsic"], dtype=np.float32).reshape(3, 3),
        "camera_to_lidar": np.asarray(
            payload["camera"]["camera_to_lidar"], dtype=np.float32).reshape(4, 4),
        "radar_to_lidar": np.asarray(
            payload["radar"]["radar_to_lidar"], dtype=np.float32).reshape(4, 4),
        "lidar_to_ego": np.asarray(
            payload["lidar"]["lidar_to_ego"], dtype=np.float32).reshape(4, 4),
    })
    return values


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
    path = path.resolve()

    def resolve(value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        item = Path(value)
        return item.resolve() if item.is_absolute() else (path.parent / item).resolve()

    records: List[FrameRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "timestamp", "image_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            gt = row.get("gt_path") or None
            ego_pose = row.get("ego_pose_path") or None
            radar_path = row.get("radar_path") or row.get("radar_ply")
            lidar_path = row.get("lidar_path") or row.get("lidar_ply")
            if not radar_path or not lidar_path:
                raise ValueError(
                    "Each manifest row needs radar_path/lidar_path "
                    "(legacy radar_ply/lidar_ply is also accepted).")
            marked = str(row.get("is_keyframe", "")).strip().lower()
            is_keyframe = marked in {"1", "true", "yes", "y"} or gt is not None
            records.append(
                FrameRecord(
                    sample_id=row["sample_id"],
                    timestamp_us=normalize_timestamp(row["timestamp"], timestamp_unit),
                    image_path=resolve(row["image_path"]),
                    radar_ply=resolve(radar_path),
                    lidar_ply=resolve(lidar_path),
                    gt_path=resolve(gt),
                    ego2global=load_matrix(
                        resolve(ego_pose),
                        np.eye(4, dtype=np.float32), (4, 4)),
                    is_keyframe=is_keyframe,
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
        stems = (img.stem, radar.stem, lidar.stem)
        if len(set(stems)) != 1:
            raise ValueError(
                "Frame names do not match after sorting: "
                f"image={img.name}, radar={radar.name}, lidar={lidar.name}")
        sample_id = img.stem
        gt_path = None
        if args.gt_dir:
            for suffix in (".json", ".txt", ".csv"):
                candidate = args.gt_dir / f"{sample_id}{suffix}"
                if candidate.exists():
                    gt_path = candidate.resolve()
                    break
        records.append(
            FrameRecord(
                sample_id=sample_id,
                timestamp_us=int(round(idx * 1_000_000 / args.frame_rate)),
                image_path=img.resolve(),
                radar_ply=radar.resolve(),
                lidar_ply=lidar.resolve(),
                gt_path=gt_path,
                ego2global=np.eye(4, dtype=np.float32),
                is_keyframe=gt_path is not None,
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


def read_ply_vertices(path: Path) -> Tuple[np.ndarray, List[str]]:
    with path.open("rb") as f:
        fmt, vertex_count, properties, header_len = parse_ply_header(f)
        names = [name for _, name in properties]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data.astype(np.float32), names
        if fmt not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format: {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + PLY_DTYPE_MAP[prop]) for prop, name in properties])
        f.seek(header_len)
        structured = np.fromfile(f, dtype=dtype, count=vertex_count)
        data = np.column_stack([structured[name] for name in names]).astype(np.float32)
        return data, names


def columns_by_name(
    points: np.ndarray, names: Sequence[str], required: Sequence[str]
) -> Dict[str, np.ndarray]:
    index = {name.lower(): idx for idx, name in enumerate(names)}
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(
            f"PLY is missing fields {missing}; available fields are {list(names)}")
    return {name: points[:, index[name]] for name in required}


def convert_lidar_points(
    points: np.ndarray, names: Sequence[str], dim: int
) -> np.ndarray:
    if dim < 4:
        raise ValueError("LiDAR output dimension must be at least 4")
    fields = columns_by_name(points, names, ("x", "y", "z", "intensity"))
    out = np.zeros((points.shape[0], dim), dtype=np.float32)
    out[:, 0] = fields["x"]
    out[:, 1] = fields["y"]
    out[:, 2] = fields["z"]
    out[:, 3] = fields["intensity"]
    return out


def convert_radar_points(
    points: np.ndarray, names: Sequence[str], dim: int
) -> np.ndarray:
    """Map raw radar PLY columns to RaCFormer-friendly columns.

    Final columns are:
        x, y, z, rcs, vx, vy, time_lag

    Raw `v` is treated as radial velocity. Its sign convention still needs to
    be checked against the radar documentation; the geometric decomposition
    into the native radar x/y axes is deterministic.
    """
    if dim < 7:
        raise ValueError("Radar output dimension must be at least 7")
    fields = columns_by_name(points, names, ("x", "y", "z", "v", "rcs"))
    out = np.zeros((points.shape[0], dim), dtype=np.float32)
    out[:, 0] = fields["x"]
    out[:, 1] = fields["y"]
    out[:, 2] = fields["z"]
    out[:, 3] = fields["rcs"]
    distance_xy = np.hypot(fields["x"], fields["y"])
    valid = distance_xy > 1e-6
    out[valid, 4] = fields["v"][valid] * fields["x"][valid] / distance_xy[valid]
    out[valid, 5] = fields["v"][valid] * fields["y"][valid] / distance_xy[valid]
    # Column 6 is filled with the actual time lag by LoadCompanyRadarSweeps.
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

    Supported compact JSON format:
        [
          {"box": [x, y, z, dx, dy, dz, yaw], "name": "car", "valid": true}
        ]

    Supported SUSTechPOINTS JSON format:
        [{"obj_type": "Car", "psr": {"position": ..., "scale": ...,
          "rotation": {"z": yaw}}}]

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
            if "psr" in item:
                psr = item["psr"]
                position = psr["position"]
                scale = psr["scale"]
                rotation = psr["rotation"]
                boxes.append([
                    position["x"], position["y"], position["z"],
                    scale["x"], scale["y"], scale["z"], rotation["z"],
                ])
                raw_name = item.get("obj_type", "car")
            else:
                boxes.append(item["box"])
                raw_name = item.get("name", "car")
            names.append(str(raw_name).strip().lower().replace(" ", "_"))
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


def select_keyframes(
    records: Sequence[FrameRecord], stride: int, all_frames: bool
) -> List[int]:
    candidates = list(range(len(records))) if all_frames else [
        idx for idx, record in enumerate(records) if record.is_keyframe]
    return candidates[::max(1, stride)]


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
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    calibration_path = args.calibration.resolve() if args.calibration else None
    calibration = load_calibration_bundle(calibration_path)
    if calibration_path:
        print(f"Using calibration: {calibration_path}")

    k = load_matrix(args.intrinsic, calibration["intrinsic"], (3, 3))
    t_camera_to_lidar = load_matrix(
        args.camera_to_lidar, calibration["camera_to_lidar"], (4, 4))
    t_lidar_to_camera = load_matrix(
        args.lidar_to_camera, np.linalg.inv(t_camera_to_lidar), (4, 4))
    t_radar_to_lidar = load_matrix(
        args.radar_to_lidar, calibration["radar_to_lidar"], (4, 4))
    t_lidar_to_ego = load_matrix(
        args.lidar_to_ego, calibration["lidar_to_ego"], (4, 4))
    if args.assume_radar_lidar_same_frame:
        t_radar_to_lidar = np.eye(4, dtype=np.float32)
    derived_radar_to_ego = t_lidar_to_ego @ t_radar_to_lidar
    t_radar_to_ego = load_matrix(
        args.radar_to_ego, derived_radar_to_ego, (4, 4))
    t_ego_to_camera = t_lidar_to_camera @ np.linalg.inv(t_lidar_to_ego)
    radar_in_ego = args.radar_in_ego or np.allclose(
        t_radar_to_ego, np.eye(4), atol=1e-6)
    if not radar_in_ego and args.radar_to_ego is None:
        print("INFO: derived T_radar_to_ego from T_radar_to_lidar and T_lidar_to_ego.")

    records = read_manifest(args.manifest, args.timestamp_unit) if args.manifest else scan_dirs(args)
    if not records:
        raise ValueError("No frames found.")

    point_suffix = ".npy" if args.point_format == "npy" else ".bin"
    keyframe_ids = select_keyframes(
        records, args.keyframe_stride, args.all_frames_as_keyframes)
    if not keyframe_ids:
        raise ValueError(
            "No keyframes found. Provide gt_path/is_keyframe in the manifest, "
            "a matching --gt-dir, or use --all-frames-as-keyframes.")
    infos: List[Dict] = []

    for index in tqdm(
        keyframe_ids,
        desc="Converting keyframes",
        unit="frame",
        dynamic_ncols=True,
    ):
        rec = records[index]
        current_ego2global = rec.ego2global
        ego2global_translation = current_ego2global[:3, 3].astype(np.float32)
        ego2global_rotation = np.eye(3, dtype=np.float32)
        lidar2ego_translation = t_lidar_to_ego[:3, 3].astype(np.float32)
        lidar2ego_rotation = np.eye(3, dtype=np.float32)

        lidar_raw, lidar_fields = read_ply_vertices(rec.lidar_ply)
        radar_raw, radar_fields = read_ply_vertices(rec.radar_ply)
        lidar_points = convert_lidar_points(
            lidar_raw, lidar_fields, args.lidar_dim)
        radar_points = convert_radar_points(
            radar_raw, radar_fields, args.radar_dim)

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
                prev_radar_raw, prev_radar_fields = read_ply_vertices(
                    prev.radar_ply)
                prev_radar_points = convert_radar_points(
                    prev_radar_raw, prev_radar_fields, args.radar_dim)
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
                "lidar_in_ego": np.allclose(
                    t_lidar_to_ego, np.eye(4), atol=1e-6),
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
