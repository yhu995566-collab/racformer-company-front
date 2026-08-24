#!/usr/bin/env python3
"""Convert sequence-disjoint ChengTech 2026-08-18 splits for RaCFormer.

Camera, radar and GT files live below ``data-root/<sequence>``.  The matching
front LiDAR cloud lives below exactly one
``truth-root/<scenario>/<sequence>/result/pcd/<frame>.pcd`` directory.  Sweeps
are constructed only within one sequence, so a temporal sample can never
cross a train/validation/test boundary.
"""

import argparse
import importlib.util
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _load_single_converter():
    path = Path(__file__).with_name("convert_chengtech_20260818.py")
    spec = importlib.util.spec_from_file_location(
        "convert_chengtech_20260818", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


single = _load_single_converter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("test",), help="Only selected splits are converted")
    parser.add_argument("--num-sweeps", type=int, default=3)
    parser.add_argument(
        "--point-cloud-range", type=float, nargs=6,
        default=(0.0, -20.0, -3.0, 50.0, 20.0, 3.0),
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Crop LiDAR and radar before serialization (default: front 50 m)")
    parser.add_argument("--limit-per-sequence", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-undistort", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate split membership and modality frame IDs without conversion")
    return parser.parse_args()


def _indexed_files(directory: Path, suffix: str) -> Dict[int, Path]:
    result = {}
    for path in directory.glob("*" + suffix):
        try:
            index = int(path.stem.rsplit("-", 1)[-1])
        except ValueError as error:
            raise ValueError("cannot parse frame index from {}".format(path)) from error
        if index in result:
            raise ValueError("duplicate frame {} below {}".format(index, directory))
        result[index] = path.resolve()
    return result


def find_truth_sequence(truth_root: Path, sequence: str) -> Tuple[Path, str]:
    matches = [path for path in truth_root.glob("*/" + sequence)
               if (path / "result" / "pcd").is_dir()]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one truth directory for {}; got {}".format(
                sequence, [str(path) for path in matches]))
    return matches[0].resolve(), matches[0].parent.name


def read_ply_comments(path: Path) -> Dict[str, str]:
    """Read only an ASCII PLY header, without loading its point payload."""
    comments = {}
    with path.open("rb") as stream:
        if stream.readline().decode("ascii").strip() != "ply":
            raise ValueError("{} is not PLY".format(path))
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("{} has no end_header".format(path))
            line = raw.decode("ascii").strip()
            if line.startswith("comment "):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    comments[parts[1]] = parts[2]
            elif line == "end_header":
                return comments


def radar_aligned_timestamp_us(path: Path) -> int:
    comments = read_ply_comments(path)
    try:
        return int(comments["rspTimestamp"]) + int(comments["syncTimediff"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "{} requires integer rspTimestamp and syncTimediff comments".format(
                path)) from error


def _timestamp_stats(deltas_ms: np.ndarray) -> Dict:
    absolute = np.abs(deltas_ms)
    return {
        "min_ms": float(deltas_ms.min()),
        "median_ms": float(np.median(deltas_ms)),
        "max_ms": float(deltas_ms.max()),
        "abs_p50_ms": float(np.percentile(absolute, 50)),
        "abs_p95_ms": float(np.percentile(absolute, 95)),
        "abs_p99_ms": float(np.percentile(absolute, 99)),
        "abs_max_ms": float(absolute.max()),
        "count_abs_gt_50ms": int(np.count_nonzero(absolute > 50.0)),
        "count_abs_gt_100ms": int(np.count_nonzero(absolute > 100.0)),
        "count_abs_gt_250ms": int(np.count_nonzero(absolute > 250.0)),
    }


def audit_radar_timestamps(frames: Sequence) -> Dict:
    gt_us = np.asarray([frame.timestamp_us for frame in frames], dtype=np.int64)
    radar_us = np.asarray([
        radar_aligned_timestamp_us(frame.radar_path) for frame in frames
    ], dtype=np.int64)
    same_deltas_ms = (radar_us - gt_us) / 1000.0

    if np.any(np.diff(radar_us) <= 0):
        monotonic = False
        nearest_indices = np.asarray([
            int(np.argmin(np.abs(radar_us - timestamp))) for timestamp in gt_us
        ], dtype=np.int64)
    else:
        monotonic = True
        right = np.searchsorted(radar_us, gt_us, side="left")
        right = np.clip(right, 0, len(radar_us) - 1)
        left = np.clip(right - 1, 0, len(radar_us) - 1)
        choose_left = np.abs(radar_us[left] - gt_us) <= np.abs(
            radar_us[right] - gt_us)
        nearest_indices = np.where(choose_left, left, right)
    nearest_deltas_ms = (
        radar_us[nearest_indices] - gt_us) / 1000.0
    offsets = nearest_indices - np.arange(len(frames), dtype=np.int64)
    offset_counts = Counter(offsets.tolist())
    worst = np.argsort(np.abs(same_deltas_ms))[-10:][::-1]
    return {
        "radar_timestamps_strictly_increasing": monotonic,
        "same_ordinal": _timestamp_stats(same_deltas_ms),
        "nearest_timestamp": _timestamp_stats(nearest_deltas_ms),
        "nearest_unique_radar_frames": int(len(np.unique(nearest_indices))),
        "nearest_reused_radar_frames": int(
            len(nearest_indices) - len(np.unique(nearest_indices))),
        "nearest_index_offset_counts": {
            str(key): int(value) for key, value in sorted(offset_counts.items())
        },
        "worst_same_ordinal": [{
            "local_index": int(index),
            "radar_file": frames[index].radar_path.name,
            "gt_file": frames[index].gt_path.name,
            "delta_ms": float(same_deltas_ms[index]),
            "nearest_radar_local_index": int(nearest_indices[index]),
            "nearest_delta_ms": float(nearest_deltas_ms[index]),
        } for index in worst],
    }


def contiguous_start(indices, modality: str, sequence: str) -> int:
    ordered = sorted(indices)
    if not ordered:
        raise ValueError("{} has no {} frames".format(sequence, modality))
    expected = list(range(ordered[0], ordered[0] + len(ordered)))
    if ordered != expected:
        raise ValueError(
            "{} {} frame IDs are not contiguous: first={}, last={}, count={}".format(
                sequence, modality, ordered[0], ordered[-1], len(ordered)))
    return ordered[0]


def discover_sequence(data_root: Path, truth_root: Path, sequence: str):
    source = data_root / sequence
    truth, scenario = find_truth_sequence(truth_root, sequence)
    images = _indexed_files(source / "front120_camera", ".jpeg")
    radars = _indexed_files(source / "radar_front", ".ply")
    labels = _indexed_files(source / "GT", ".json")
    lidar_dir = truth / "result" / "pcd"
    lidars = {int(path.stem): path.resolve() for path in lidar_dir.glob("*.pcd")}
    modalities = {
        "image": images, "radar": radars, "GT": labels, "lidar": lidars,
    }
    counts = {name: len(files) for name, files in modalities.items()}
    if not counts["GT"] or len(set(counts.values())) != 1:
        raise ValueError(
            "{} modalities do not contain identical frame counts: {}".format(
                sequence, counts))
    starts = {
        name: contiguous_start(files, name, sequence)
        for name, files in modalities.items()
    }
    frames = []
    declared_frame_start = None
    for index in range(counts["GT"]):
        image_path = images[index + starts["image"]]
        radar_path = radars[index + starts["radar"]]
        gt_path = labels[index + starts["GT"]]
        lidar_path = lidars[index + starts["lidar"]]
        gt = json.loads(gt_path.read_text())
        declared_frame = int(gt["frame_num"])
        if declared_frame_start is None:
            declared_frame_start = declared_frame
        if declared_frame != declared_frame_start + index:
            raise ValueError(
                "{} declares non-contiguous frame_num {}; expected {}".format(
                    gt_path, declared_frame, declared_frame_start + index))
        frames.append(single.Frame(
            index=index,
            sample_id="{}-{:06d}".format(sequence, index),
            timestamp_us=int(round(float(gt["stamp_sec"]) * 1e6)),
            image_path=image_path, radar_path=radar_path,
            lidar_path=lidar_path, gt_path=gt_path, gt=gt,
            ego2global=single.parse_pose(gt["frame_info"]),
            car_twist=float(gt["frame_info"]["car_twist"]),
        ))
    timestamps = np.asarray([frame.timestamp_us for frame in frames])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("{} GT timestamps are not strictly increasing".format(sequence))
    alignment = {
        "image_file_index_offset": starts["image"],
        "radar_file_index_offset": starts["radar"],
        "gt_file_index_offset": starts["GT"],
        "lidar_file_index_offset": starts["lidar"],
        "gt_declared_frame_offset": declared_frame_start,
    }
    return frames, scenario, alignment


def roi_mask(points: np.ndarray, point_cloud_range: Sequence[float]) -> np.ndarray:
    roi = np.asarray(point_cloud_range, dtype=np.float32)
    return (np.isfinite(points[:, :3]).all(axis=1) &
            (points[:, 0] >= roi[0]) & (points[:, 0] <= roi[3]) &
            (points[:, 1] >= roi[1]) & (points[:, 1] <= roi[4]) &
            (points[:, 2] >= roi[2]) & (points[:, 2] <= roi[5]))


def convert_result_lidar(frame, out_root: Path,
                         point_cloud_range: Sequence[float]) -> Tuple[Path, Dict]:
    fields = single.read_binary_compressed_pcd(frame.lidar_path)
    required = {"x", "y", "z", "intensity"}
    if set(fields) != required:
        raise ValueError("{} fields must be {}; got {}".format(
            frame.lidar_path, sorted(required), sorted(fields)))
    count = len(fields["x"])
    if any(len(value) != count for value in fields.values()):
        raise ValueError("{} has incomplete point fields".format(frame.lidar_path))
    points = np.zeros((count, 5), dtype=np.float32)
    points[:, :4] = np.column_stack(
        [fields["x"], fields["y"], fields["z"], fields["intensity"]])
    source_count = count
    source_finite = np.isfinite(points[:, :3]).all(axis=1)
    source_xyz = points[source_finite, :3]
    points = points[roi_mask(points, point_cloud_range)]
    output = out_root / "lidar" / (frame.sample_id + ".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, points)
    return output.resolve(), {
        "source_points": source_count,
        "cropped_points": len(points),
        "cropped_ratio": float(len(points)) / max(1, source_count),
        "source_xyz_min": source_xyz.min(axis=0).tolist()
        if len(source_xyz) else None,
        "source_xyz_max": source_xyz.max(axis=0).tolist()
        if len(source_xyz) else None,
        "xyz_min": points[:, :3].min(axis=0).tolist() if len(points) else None,
        "xyz_max": points[:, :3].max(axis=0).tolist() if len(points) else None,
    }


def convert_sequence(frames: Sequence, out_root: Path, args) -> Tuple[List[Dict], Dict]:
    converted = []
    class_counts = Counter()
    ignored_counts = Counter()
    duplicate_candidates = 0
    for frame in tqdm(frames, desc=frames[0].sample_id.rsplit("-", 1)[0]):
        image = single.prepare_image(
            frame, out_root, args.jpeg_quality, args.skip_undistort)
        lidar, lidar_audit = convert_result_lidar(
            frame, out_root, args.point_cloud_range)
        radar_raw, radar_names, radar_comments = single.read_ply_vertices(
            frame.radar_path)
        radar_timestamp_us, radar_audit = single.aligned_radar_timestamp_us(
            radar_comments, frame.timestamp_us)
        radar = single.convert_radar(radar_raw, radar_names, frame.car_twist)
        radar_source_points = len(radar)
        radar = radar[roi_mask(radar, args.point_cloud_range)]
        radar_audit.update({
            "source_points": radar_source_points,
            "cropped_points": len(radar),
            "cropped_ratio": float(len(radar)) / max(1, radar_source_points),
        })
        radar_path = out_root / "radar" / (frame.sample_id + ".npy")
        radar_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(radar_path, radar)
        boxes, names, velocities, sources, gt_audit = single.convert_gt(frame)
        class_counts.update(names.tolist())
        ignored_counts.update(gt_audit["ignored_types"])
        duplicate_candidates += gt_audit["duplicate_cross_source_candidates"]
        converted.append({
            "image": image, "lidar": lidar, "radar": radar_path.resolve(),
            "radar_timestamp_us": radar_timestamp_us, "boxes": boxes,
            "names": names, "velocities": velocities, "sources": sources,
            "lidar_audit": lidar_audit, "radar_audit": radar_audit,
        })

    infos = []
    for local_index, (frame, item) in enumerate(zip(frames, converted)):
        current_camera = single.camera_entry(
            Path(single.relative_path(item["image"], out_root)),
            frame.timestamp_us, np.eye(4))
        sweeps = []
        for offset in range(1, args.num_sweeps + 1):
            previous_index = max(0, local_index - offset)
            previous, previous_item = frames[previous_index], converted[previous_index]
            current_to_previous = np.linalg.inv(
                previous.ego2global) @ frame.ego2global
            previous_to_current = np.linalg.inv(
                frame.ego2global) @ previous.ego2global
            sweeps.append({
                "CAM_FRONT": single.camera_entry(
                    Path(single.relative_path(previous_item["image"], out_root)),
                    previous.timestamp_us, current_to_previous),
                "RADAR_FRONT": {
                    "data_path": single.relative_path(previous_item["radar"], out_root),
                    "timestamp": int(previous_item["radar_timestamp_us"]),
                    "radar_in_ego": bool(np.allclose(
                        previous_to_current, np.eye(4), atol=1e-6)),
                    "radar2ego": previous_to_current.astype(np.float32),
                },
            })
        infos.append({
            "token": frame.sample_id, "sequence": frame.sample_id[:-7],
            "timestamp": int(frame.timestamp_us),
            "lidar_path": single.relative_path(item["lidar"], out_root),
            "lidar_in_ego": True, "lidar2ego": np.eye(4, dtype=np.float32),
            "radar_path": single.relative_path(item["radar"], out_root),
            "ego2global": frame.ego2global.astype(np.float32),
            "ego2global_translation": frame.ego2global[:3, 3].astype(np.float32),
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "lidar2ego_translation": np.zeros(3, dtype=np.float32),
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "cams": {"CAM_FRONT": current_camera},
            "rads": {"RADAR_FRONT": {
                "data_path": single.relative_path(item["radar"], out_root),
                "timestamp": int(item["radar_timestamp_us"]),
                "radar_in_ego": True,
                "radar2ego": np.eye(4, dtype=np.float32),
            }},
            "sweeps": sweeps, "gt_boxes": item["boxes"],
            "gt_names": item["names"], "gt_velocity": item["velocities"],
            "gt_sources": item["sources"],
            "num_lidar_pts": np.zeros(len(item["boxes"]), dtype=np.int32),
            "num_radar_pts": np.zeros(len(item["boxes"]), dtype=np.int32),
            "valid_flag": np.ones(len(item["boxes"]), dtype=bool),
        })
    return infos, {
        "frames": len(frames), "class_counts": dict(sorted(class_counts.items())),
        "ignored_type_counts": dict(sorted(ignored_counts.items())),
        "duplicate_cross_source_candidates": int(duplicate_candidates),
        "first_lidar": converted[0]["lidar_audit"],
        "last_lidar": converted[-1]["lidar_audit"],
        "first_radar": converted[0]["radar_audit"],
        "last_radar": converted[-1]["radar_audit"],
    }


def main() -> None:
    args = parse_args()
    if args.num_sweeps < 0:
        raise ValueError("--num-sweeps must be non-negative")
    manifest = json.loads(args.split_manifest.read_text())
    selected = [sequence for split in args.splits for sequence in manifest[split]]
    if len(selected) != len(set(selected)):
        raise ValueError("selected splits contain duplicate sequences")
    all_declared = [sequence for name in ("train", "val", "test")
                    for sequence in manifest.get(name, [])]
    if len(all_declared) != len(set(all_declared)):
        raise ValueError("split manifest assigns a sequence more than once")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "chengtech_20260818_collection",
        "info_version": "racformer_chengtech_20260818_collection_v1",
        "classes": single.CLASS_NAMES, "num_sweeps": args.num_sweeps,
        "point_cloud_range": list(args.point_cloud_range),
        "split_manifest": str(args.split_manifest.resolve()),
    }
    summary = {"source_root": str(args.data_root.resolve()),
               "truth_root": str(args.truth_root.resolve()),
               "output_root": str(out_root), "splits": {}}
    for split in args.splits:
        split_infos, sequence_summaries = [], {}
        for sequence in manifest[split]:
            frames, scenario, alignment = discover_sequence(
                args.data_root.resolve(), args.truth_root.resolve(), sequence)
            if args.limit_per_sequence is not None:
                frames = frames[:args.limit_per_sequence]
            if args.dry_run:
                sequence_summaries[sequence] = {
                    "scenario": scenario, "frames": len(frames),
                    "alignment": alignment,
                    "radar_timestamp_audit": audit_radar_timestamps(frames)}
                continue
            infos, audit = convert_sequence(frames, out_root, args)
            split_infos.extend(infos)
            sequence_summaries[sequence] = {
                "scenario": scenario, "alignment": alignment, **audit}
        if args.dry_run:
            summary["splits"][split] = {
                "frames": sum(item["frames"] for item in sequence_summaries.values()),
                "sequences": sequence_summaries,
            }
            continue
        payload = {"infos": split_infos,
                   "metadata": {**metadata, "split": split,
                                "sequences": list(manifest[split])}}
        info_path = out_root / "custom_infos_{}_sweep.pkl".format(split)
        with info_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        summary["splits"][split] = {
            "frames": len(split_infos), "sequences": sequence_summaries,
            "info_file": str(info_path),
        }
    summary_name = "dry_run_summary.json" if args.dry_run else "conversion_summary.json"
    (out_root / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
