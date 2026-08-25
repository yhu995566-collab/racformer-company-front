#!/usr/bin/env python3
"""Measure company-radar geometric coverage of GT centres before Gaussian fusion.

The report compares current-frame radar candidates with the temporal set used
by the four-frame model.  It operates on converted RaCFormer info files, keeps
all valid radar returns inside the configured ROI, and reports nearest-return
recall by range, class, and GT source.  It does not run or modify the detector.
"""

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "val", "test"),
                        default="all")
    parser.add_argument("--class-names", nargs="+",
                        help="Optional reliable GT classes to include")
    parser.add_argument("--use-frames", type=int, default=4,
                        help="Current radar frame plus previous sweeps")
    parser.add_argument("--point-cloud-range", type=float, nargs=6,
                        default=(0.0, -20.0, -3.0, 350.0, 20.0, 3.0))
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(1.0, 2.0, 4.0, 8.0))
    parser.add_argument("--range-bins", type=float, nargs="+",
                        default=(0, 50, 100, 150, 200, 250, 300, 350))
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_infos(root, split):
    splits = ("train", "val") if split == "all" else (split,)
    infos = []
    for name in splits:
        path = root / "custom_infos_{}_sweep.pkl".format(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        infos.extend(payload["infos"])
    by_token = {str(info["token"]): info for info in infos}
    return sorted(by_token.values(), key=lambda item: int(item["timestamp"]))


def roi_mask(points, point_cloud_range):
    pc = np.asarray(point_cloud_range, dtype=np.float64)
    return (
        np.isfinite(points[:, :3]).all(axis=1) &
        (points[:, 0] >= pc[0]) & (points[:, 0] <= pc[3]) &
        (points[:, 1] >= pc[1]) & (points[:, 1] <= pc[4]) &
        (points[:, 2] >= pc[2]) & (points[:, 2] <= pc[5]))


def load_radar_entry(root, entry, point_cloud_range):
    points = np.load(resolve(root, entry["data_path"]), mmap_mode="r")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    if not entry.get("radar_in_ego", True):
        transform = np.asarray(entry["radar2ego"], dtype=np.float64)
        xyz = (np.column_stack([xyz, np.ones(len(xyz))]) @ transform.T)[:, :3]
    return xyz[roi_mask(xyz, point_cloud_range)]


def collect_radar(root, info, use_frames, point_cloud_range):
    entries = [info["rads"]["RADAR_FRONT"]]
    entries.extend(
        sweep["RADAR_FRONT"] for sweep in info.get("sweeps", [])
        if "RADAR_FRONT" in sweep)
    unique_entries = []
    seen = set()
    for entry in entries:
        key = (str(entry["data_path"]), int(entry.get("timestamp", -1)))
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
        if len(unique_entries) == use_frames:
            break
    chunks = [
        load_radar_entry(root, entry, point_cloud_range)
        for entry in unique_entries
    ]
    current = chunks[0] if chunks else np.empty((0, 3), dtype=np.float64)
    temporal = np.concatenate(chunks, axis=0) if chunks else current
    return current, temporal, len(unique_entries)


def nearest_distances(gt_xy, radar_xyz):
    if len(gt_xy) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(radar_xyz) == 0:
        return np.full((len(gt_xy),), np.inf, dtype=np.float64)
    # A frame has only a few thousand radar points and tens of GT boxes, so a
    # direct matrix is faster and clearer than adding a spatial-tree dependency.
    delta = gt_xy[:, None, :] - radar_xyz[None, :, :2]
    return np.sqrt(np.square(delta).sum(axis=-1)).min(axis=1)


def percentile_or_none(values, percentile):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.percentile(finite, percentile))


def summarize(records, distance_key, thresholds):
    distances = np.asarray([item[distance_key] for item in records], dtype=np.float64)
    result = {
        "gt_count": len(records),
        "finite_nearest_count": int(np.isfinite(distances).sum()),
        "nearest_distance_m": {
            "p50": percentile_or_none(distances, 50),
            "p90": percentile_or_none(distances, 90),
            "p95": percentile_or_none(distances, 95),
        },
        "recall": {},
    }
    for threshold in thresholds:
        result["recall"]["within_{}m".format(threshold)] = (
            float(np.mean(distances <= threshold)) if len(distances) else None)
    return result


def grouped_summaries(records, distance_key, thresholds, range_bins):
    output = {"overall": summarize(records, distance_key, thresholds)}
    for field in ("class_name", "source"):
        groups = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(record)
        output["by_{}".format(field)] = {
            key: summarize(value, distance_key, thresholds)
            for key, value in sorted(groups.items())
        }
    range_groups = {}
    for low, high in zip(range_bins[:-1], range_bins[1:]):
        selected = [
            record for record in records
            if low <= record["forward_x_m"] < high
        ]
        range_groups["{}-{}m".format(low, high)] = summarize(
            selected, distance_key, thresholds)
    output["by_range"] = range_groups
    return output


def main():
    args = parse_args()
    if args.use_frames <= 0:
        raise ValueError("--use-frames must be positive")
    if sorted(args.thresholds) != list(args.thresholds) or min(args.thresholds) <= 0:
        raise ValueError("--thresholds must be sorted positive values")
    if sorted(args.range_bins) != list(args.range_bins) or len(args.range_bins) < 2:
        raise ValueError("--range-bins must be sorted and contain at least two values")

    root = args.processed_root.resolve()
    infos = load_infos(root, args.split)
    if args.max_frames is not None:
        infos = infos[:args.max_frames]
    records = []
    frame_stats = []
    pc = np.asarray(args.point_cloud_range, dtype=np.float64)

    for info in tqdm(infos, desc="Measuring radar candidate recall"):
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        names = np.asarray(info["gt_names"], dtype=object)
        sources = np.asarray(info.get("gt_sources", np.full(len(boxes), -1)))
        gt_valid = roi_mask(boxes[:, :3], pc)
        if args.class_names:
            gt_valid &= np.isin(names, np.asarray(args.class_names, dtype=object))
        boxes, names, sources = boxes[gt_valid], names[gt_valid], sources[gt_valid]
        current, temporal, frame_count = collect_radar(
            root, info, args.use_frames, pc)
        current_distance = nearest_distances(boxes[:, :2], current)
        temporal_distance = nearest_distances(boxes[:, :2], temporal)
        frame_stats.append({
            "token": str(info["token"]),
            "current_candidates": len(current),
            "temporal_candidates": len(temporal),
            "unique_radar_frames": frame_count,
            "gt_count": len(boxes),
        })
        for index, box in enumerate(boxes):
            records.append({
                "token": str(info["token"]),
                "class_name": str(names[index]),
                "source": int(sources[index]),
                "x": float(box[0]),
                "y": float(box[1]),
                "forward_x_m": float(box[0]),
                "range_m": float(np.linalg.norm(box[:2])),
                "nearest_current_m": float(current_distance[index]),
                "nearest_temporal_m": float(temporal_distance[index]),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_fields = ("current_candidates", "temporal_candidates")
    candidate_stats = {
        field: {
            "mean": float(np.mean([item[field] for item in frame_stats])),
            "p50": float(np.percentile([item[field] for item in frame_stats], 50)),
            "p95": float(np.percentile([item[field] for item in frame_stats], 95)),
        } for field in candidate_fields
    }
    summary = {
        "processed_root": str(root),
        "split": args.split,
        "frames": len(infos),
        "gt_count": len(records),
        "use_frames": args.use_frames,
        "point_cloud_range": list(args.point_cloud_range),
        "thresholds_m": list(args.thresholds),
        "class_names": args.class_names,
        "candidate_counts_per_frame": candidate_stats,
        "current_frame": grouped_summaries(
            records, "nearest_current_m", args.thresholds, args.range_bins),
        "temporal": grouped_summaries(
            records, "nearest_temporal_m", args.thresholds, args.range_bins),
    }
    summary_path = args.out_dir / "company_radar_candidate_recall_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    records_path = args.out_dir / "company_radar_candidate_recall_records.csv"
    with records_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [
            "token", "class_name", "source", "x", "y", "forward_x_m", "range_m",
            "nearest_current_m", "nearest_temporal_m"])
        writer.writeheader()
        writer.writerows(records)

    print("frames: {}".format(len(infos)))
    print("GT centres: {}".format(len(records)))
    print("summary: {}".format(summary_path.resolve()))
    print("records: {}".format(records_path.resolve()))
    print(json.dumps({
        "current": summary["current_frame"]["overall"],
        "temporal": summary["temporal"]["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
