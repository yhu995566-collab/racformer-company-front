#!/usr/bin/env python3
"""Measure geometric radar-candidate coverage of nuScenes GT centres.

This is a stage-1 analysis only: radar returns are candidate locations, not
Gaussian centres.  No model, training pipeline, or detector forward is used.

Coordinate convention
---------------------
Every radar sweep is transformed as

    radar sensor -> sweep ego -> global -> current-sample ego

and annotation centres are transformed as ``global -> current-sample ego``.
Consequently both inputs to the metric are in the same nuScenes ego frame
(x forward, y left, z up).  Compensated radar velocity (fields 8/9) is rotated
through the same frames, without translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# Official nuScenes RadarPointCloud layout.
RADAR_FIELDS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "dyn_prop": 3,
    "id": 4,
    "rcs": 5,
    "vx": 6,
    "vy": 7,
    "vx_comp": 8,
    "vy_comp": 9,
    "is_quality_valid": 10,
    "ambig_state": 11,
    "x_rms": 12,
    "y_rms": 13,
    "invalid_state": 14,
    "pdh0": 15,
    "vx_rms": 16,
    "vy_rms": 17,
}

CLASS_PREFIXES = {
    "car": ("vehicle.car",),
    "truck": ("vehicle.truck",),
    "bus": ("vehicle.bus",),
    "pedestrian": ("human.pedestrian",),
    "motorcycle": ("vehicle.motorcycle",),
    "bicycle": ("vehicle.bicycle",),
}


@dataclass
class GTRecord:
    sample_token: str
    annotation_token: str
    class_name: str
    center: np.ndarray
    size: np.ndarray
    yaw: float
    radial_range: float


@dataclass
class RadarCandidates:
    """Radar fields after alignment to the current ego frame."""

    xyz: np.ndarray
    velocity_xy: np.ndarray
    rcs: np.ndarray


def _rotation_matrix(quaternion_values: Sequence[float]) -> np.ndarray:
    """Return a rotation matrix for a nuScenes [w, x, y, z] quaternion."""
    from pyquaternion import Quaternion

    return Quaternion(quaternion_values).rotation_matrix


def _global_to_ego(points: np.ndarray, ego_pose: Mapping[str, Any]) -> np.ndarray:
    rotation = _rotation_matrix(ego_pose["rotation"])
    translation = np.asarray(ego_pose["translation"], dtype=np.float64)
    return (points - translation) @ rotation


def _category_to_detection_class(category: str) -> Optional[str]:
    for class_name, prefixes in CLASS_PREFIXES.items():
        if any(category.startswith(prefix) for prefix in prefixes):
            return class_name
    return None


def _inside_bev(xy: np.ndarray, bev_range: Sequence[float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bev_range
    return (
        (xy[:, 0] >= xmin)
        & (xy[:, 0] <= xmax)
        & (xy[:, 1] >= ymin)
        & (xy[:, 1] <= ymax)
    )


def collect_gt_centers(
    nusc: Any,
    sample: Mapping[str, Any],
    bev_range: Sequence[float],
    max_range: float,
) -> List[GTRecord]:
    """Collect detection GT centres in the current sample ego frame."""
    ref_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    records: List[GTRecord] = []

    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)
        class_name = _category_to_detection_class(annotation["category_name"])
        if class_name is None:
            continue
        center = _global_to_ego(
            np.asarray(annotation["translation"], dtype=np.float64)[None, :],
            ref_pose,
        )[0]
        radial_range = float(np.linalg.norm(center[:2]))
        if radial_range > max_range or not _inside_bev(center[None, :2], bev_range)[0]:
            continue

        global_rotation = _rotation_matrix(annotation["rotation"])
        ego_rotation = _rotation_matrix(ref_pose["rotation"])
        local_rotation = ego_rotation.T @ global_rotation
        yaw = float(math.atan2(local_rotation[1, 0], local_rotation[0, 0]))
        records.append(
            GTRecord(
                sample_token=sample["token"],
                annotation_token=annotation_token,
                class_name=class_name,
                center=center,
                size=np.asarray(annotation["size"], dtype=np.float64),
                yaw=yaw,
                radial_range=radial_range,
            )
        )
    return records


def _load_one_radar_sweep(
    nusc: Any,
    sample_data: Mapping[str, Any],
    current_ego_pose: Mapping[str, Any],
) -> RadarCandidates:
    from nuscenes.utils.data_classes import RadarPointCloud

    path = Path(nusc.dataroot) / sample_data["filename"]
    # The devkit's default filters define legal nuScenes radar returns.  The
    # additional quality definition is applied later and never removes points
    # merely because their velocity is near zero.
    points = RadarPointCloud.from_file(str(path)).points
    if points.shape[1] == 0:
        return RadarCandidates(
            np.empty((0, 3)), np.empty((0, 2)), np.empty((0,))
        )

    calibrated = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    sweep_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    sensor_rotation = _rotation_matrix(calibrated["rotation"])
    sweep_rotation = _rotation_matrix(sweep_pose["rotation"])
    current_rotation = _rotation_matrix(current_ego_pose["rotation"])

    xyz_sensor = points[:3].T
    xyz_sweep_ego = xyz_sensor @ sensor_rotation.T + np.asarray(
        calibrated["translation"], dtype=np.float64
    )
    xyz_global = xyz_sweep_ego @ sweep_rotation.T + np.asarray(
        sweep_pose["translation"], dtype=np.float64
    )
    xyz_current_ego = _global_to_ego(xyz_global, current_ego_pose)

    velocity_sensor = np.column_stack(
        (points[RADAR_FIELDS["vx_comp"]], points[RADAR_FIELDS["vy_comp"]],
         np.zeros(points.shape[1]))
    )
    velocity_current = (
        velocity_sensor @ sensor_rotation.T @ sweep_rotation.T @ current_rotation
    )
    return RadarCandidates(
        xyz=xyz_current_ego,
        velocity_xy=velocity_current[:, :2],
        rcs=points[RADAR_FIELDS["rcs"]].astype(np.float64, copy=False),
    )


def collect_radar_candidates(
    nusc: Any,
    sample: Mapping[str, Any],
    use_sweeps: int,
    bev_range: Sequence[float],
    min_range: float,
    max_range: float,
    max_abs_speed: float,
    rcs_drop_percentile: float,
) -> Tuple[RadarCandidates, RadarCandidates]:
    """Collect raw/legal and simply filtered candidates in current ego frame."""
    ref_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    chunks: List[RadarCandidates] = []

    for channel in RADAR_CHANNELS:
        token = sample["data"].get(channel, "")
        for _ in range(use_sweeps):
            if not token:
                break
            sample_data = nusc.get("sample_data", token)
            chunks.append(_load_one_radar_sweep(nusc, sample_data, current_pose))
            token = sample_data["prev"]

    if not chunks:
        empty = RadarCandidates(np.empty((0, 3)), np.empty((0, 2)), np.empty((0,)))
        return empty, empty

    all_candidates = RadarCandidates(
        xyz=np.concatenate([chunk.xyz for chunk in chunks], axis=0),
        velocity_xy=np.concatenate([chunk.velocity_xy for chunk in chunks], axis=0),
        rcs=np.concatenate([chunk.rcs for chunk in chunks], axis=0),
    )
    radial_range = np.linalg.norm(all_candidates.xyz[:, :2], axis=1)
    finite = (
        np.isfinite(all_candidates.xyz).all(axis=1)
        & np.isfinite(all_candidates.velocity_xy).all(axis=1)
        & np.isfinite(all_candidates.rcs)
    )
    spatial = (
        _inside_bev(all_candidates.xyz[:, :2], bev_range)
        & (radial_range >= min_range)
        & (radial_range <= max_range)
    )
    raw_mask = finite & spatial
    raw = _subset_candidates(all_candidates, raw_mask)

    speed = np.linalg.norm(raw.velocity_xy, axis=1)
    filtered_mask = speed <= max_abs_speed
    if len(raw.rcs) and rcs_drop_percentile > 0:
        rcs_cutoff = np.percentile(raw.rcs, rcs_drop_percentile)
        filtered_mask &= raw.rcs >= rcs_cutoff
    filtered = _subset_candidates(raw, filtered_mask)
    return raw, filtered


def _subset_candidates(candidates: RadarCandidates, mask: np.ndarray) -> RadarCandidates:
    return RadarCandidates(
        xyz=candidates.xyz[mask],
        velocity_xy=candidates.velocity_xy[mask],
        rcs=candidates.rcs[mask],
    )


def compute_nearest_distances(
    gt_centers: np.ndarray, candidate_xy: np.ndarray, chunk_size: int = 4096
) -> np.ndarray:
    """Compute exact nearest BEV distance without a scipy dependency."""
    if len(gt_centers) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(candidate_xy) == 0:
        return np.full((len(gt_centers),), np.inf, dtype=np.float64)
    nearest = np.full((len(gt_centers),), np.inf, dtype=np.float64)
    for start in range(0, len(candidate_xy), chunk_size):
        chunk = candidate_xy[start : start + chunk_size]
        squared = np.sum(
            (gt_centers[:, None, :2] - chunk[None, :, :2]) ** 2, axis=2
        )
        nearest = np.minimum(nearest, np.sqrt(np.min(squared, axis=1)))
    return nearest


def compute_recall_metrics(
    nearest_distances: Sequence[float], recall_thresholds: Sequence[float]
) -> Dict[str, Optional[float]]:
    """Return recall and finite nearest-distance distribution statistics."""
    distances = np.asarray(nearest_distances, dtype=np.float64)
    result: Dict[str, Optional[float]] = {"num_gt": int(len(distances))}
    for threshold in recall_thresholds:
        result[f"recall@{_number_label(threshold)}m"] = (
            float(np.mean(distances <= threshold)) if len(distances) else None
        )
    finite = distances[np.isfinite(distances)]
    result.update(
        {
            "num_gt_with_candidate": int(len(finite)),
            "mean_nearest_distance": float(np.mean(finite)) if len(finite) else None,
            "median_nearest_distance": float(np.median(finite)) if len(finite) else None,
            "p90_nearest_distance": (
                float(np.percentile(finite, 90)) if len(finite) else None
            ),
        }
    )
    return result


def _number_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _range_groups(range_bins: Sequence[float]) -> List[Tuple[str, float, float]]:
    groups = [
        (f"{_number_label(lo)}-{_number_label(hi)}m", lo, hi)
        for lo, hi in zip(range_bins[:-1], range_bins[1:])
    ]
    if range_bins[0] <= 50 and range_bins[-1] >= 100:
        groups.append(("50-100m_overall", 50.0, 100.0))
    return groups


def _in_half_open_range(value: float, lo: float, hi: float, final_hi: float) -> bool:
    return lo <= value <= hi if hi == final_hi else lo <= value < hi


def _metric_rows(
    observations: Sequence[Mapping[str, Any]],
    range_bins: Sequence[float],
    thresholds: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    for definition in ("raw", "filtered"):
        for range_name, lo, hi in _range_groups(range_bins):
            selected = [
                row[f"nearest_{definition}"]
                for row in observations
                if _in_half_open_range(row["gt_range"], lo, hi, range_bins[-1])
            ]
            summary_rows.append(
                {
                    "candidate_definition": definition,
                    "range_bin": range_name,
                    **compute_recall_metrics(selected, thresholds),
                }
            )
            for class_name in CLASS_PREFIXES:
                class_selected = [
                    row[f"nearest_{definition}"]
                    for row in observations
                    if row["class_name"] == class_name
                    and _in_half_open_range(row["gt_range"], lo, hi, range_bins[-1])
                ]
                class_rows.append(
                    {
                        "candidate_definition": definition,
                        "range_bin": range_name,
                        "class_name": class_name,
                        **compute_recall_metrics(class_selected, thresholds),
                    }
                )
    return summary_rows, class_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for empty CSV {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    out_dir: Path,
    metadata: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    range_bins: Sequence[float],
    thresholds: Sequence[float],
) -> None:
    """Save the four required machine-readable result files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, class_rows = _metric_rows(observations, range_bins, thresholds)
    payload = {"metadata": dict(metadata), "metrics": summary_rows}
    with (out_dir / "candidate_recall_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    _write_csv(out_dir / "candidate_recall_summary.csv", summary_rows)
    _write_csv(out_dir / "per_class_candidate_recall.csv", class_rows)

    distance_rows = []
    for row in observations:
        common = {
            "sample_token": row["sample_token"],
            "annotation_token": row["annotation_token"],
            "class_name": row["class_name"],
            "gt_range_m": row["gt_range"],
            "range_bin": row["range_bin"],
        }
        for definition in ("raw", "filtered"):
            value = row[f"nearest_{definition}"]
            distance_rows.append(
                {
                    **common,
                    "candidate_definition": definition,
                    "nearest_distance_m": value if math.isfinite(value) else "inf",
                }
            )
    _write_csv(
        out_dir / "nearest_distance_by_range.csv",
        distance_rows,
        fieldnames=(
            "sample_token", "annotation_token", "class_name", "gt_range_m",
            "range_bin", "candidate_definition", "nearest_distance_m",
        ),
    )


def visualize_bev_case(
    output_path: Path,
    sample_token: str,
    gt_records: Sequence[GTRecord],
    raw: RadarCandidates,
    filtered: RadarCandidates,
    match_threshold: float,
    bev_range: Sequence[float],
) -> None:
    """Render one optional diagnostic view; matplotlib is imported lazily."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, axis = plt.subplots(figsize=(9, 9))
    if len(raw.xyz):
        axis.scatter(raw.xyz[:, 0], raw.xyz[:, 1], s=2, c="0.75", label="raw radar")
    if len(filtered.xyz):
        axis.scatter(
            filtered.xyz[:, 0], filtered.xyz[:, 1], s=4, c="tab:blue",
            alpha=0.7, label="filtered radar"
        )
    centers = np.asarray([gt.center for gt in gt_records], dtype=np.float64)
    nearest = compute_nearest_distances(centers, filtered.xyz[:, :2])
    for gt, distance in zip(gt_records, nearest):
        width, length = gt.size[:2]
        local = np.asarray(
            [[length / 2, width / 2], [length / 2, -width / 2],
             [-length / 2, -width / 2], [-length / 2, width / 2]]
        )
        rotation = np.asarray(
            [[math.cos(gt.yaw), -math.sin(gt.yaw)],
             [math.sin(gt.yaw), math.cos(gt.yaw)]]
        )
        corners = local @ rotation.T + gt.center[:2]
        color = "tab:green" if distance <= match_threshold else "tab:red"
        axis.add_patch(Polygon(corners, closed=True, fill=False, ec=color, lw=1.2))
        axis.scatter(gt.center[0], gt.center[1], marker="x", c=color, s=28)
    for radius in (50, 70, 100):
        axis.add_patch(Circle((0, 0), radius, fill=False, ls="--", ec="0.45", lw=0.8))
    axis.set(xlim=bev_range[:2], ylim=bev_range[2:], aspect="equal", xlabel="ego x / forward (m)", ylabel="ego y / left (m)")
    axis.set_title(f"{sample_token} | green=matched@{match_threshold:g}m, red=missed")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", markerscale=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scene_names_for_split(version: str, split: str) -> set:
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    aliases = {"mini_train": "mini_train", "mini_val": "mini_val"}
    requested = aliases.get(split, split)
    if requested not in splits:
        raise ValueError(f"Unknown split {split!r}; available: {sorted(splits)}")
    if version == "v1.0-mini" and requested in ("train", "val"):
        requested = f"mini_{requested}"
    return set(splits[requested])


def _samples_for_split(nusc: Any, split: str) -> List[Mapping[str, Any]]:
    scene_names = _scene_names_for_split(nusc.version, split)
    scene_tokens = {
        scene["token"] for scene in nusc.scene if scene["name"] in scene_names
    }
    samples = [sample for sample in nusc.sample if sample["scene_token"] in scene_tokens]
    return sorted(samples, key=lambda row: row["timestamp"])


def _range_bin_label(value: float, bins: Sequence[float]) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if _in_half_open_range(value, lo, hi, bins[-1]):
            return f"{_number_label(lo)}-{_number_label(hi)}m"
    return "out_of_range"


def _config_defaults(config_path: Optional[str]) -> Dict[str, str]:
    """Read simple path/version assignments without importing an MMEngine config."""
    if not config_path:
        return {}
    text = Path(config_path).read_text(encoding="utf-8")
    result: Dict[str, str] = {}
    for key, target in (("dataset_root", "data_root"), ("data_root", "data_root"), ("nu_version", "version")):
        match = re.search(rf"^\s*{key}\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
        if match and target not in result:
            result[target] = match.group(1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default=None, help="Optional config used only for simple data-root/version defaults.")
    parser.add_argument("--out-dir", default="outputs/stage1_candidate_recall")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-sweeps", type=int, default=1, help="Sweeps per each of the five radar sensors, including current.")
    parser.add_argument("--bev-range", nargs=4, type=float, default=[-54, 54, -54, 54], metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--range-bins", nargs="+", type=float, default=[0, 30, 50, 70, 100])
    parser.add_argument("--recall-thresholds", nargs="+", type=float, default=[2, 4, 8])
    parser.add_argument("--min-radar-range", type=float, default=1.0)
    parser.add_argument("--max-abs-speed", type=float, default=100.0, help="Only removes implausible compensated speeds; zero-speed returns are retained.")
    parser.add_argument("--rcs-drop-percentile", type=float, default=5.0)
    parser.add_argument("--vis-num", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.data_root:
        raise ValueError("--data-root is required unless it can be read from --config")
    if args.use_sweeps < 1:
        raise ValueError("--use-sweeps must be >= 1")
    if len(args.range_bins) < 2 or any(b >= c for b, c in zip(args.range_bins, args.range_bins[1:])):
        raise ValueError("--range-bins must be strictly increasing")
    if any(value <= 0 for value in args.recall_thresholds):
        raise ValueError("--recall-thresholds must be positive")
    if not 0 <= args.rcs_drop_percentile < 100:
        raise ValueError("--rcs-drop-percentile must be in [0, 100)")


def main() -> None:
    args = parse_args()
    defaults = _config_defaults(args.config)
    args.data_root = args.data_root or defaults.get("data_root")
    args.version = args.version or defaults.get("version", "v1.0-trainval")
    _validate_args(args)

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=args.verbose)
    samples = _samples_for_split(nusc, args.split)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    observations: List[Dict[str, Any]] = []
    total_raw = total_filtered = 0
    out_dir = Path(args.out_dir)

    for sample_index, sample in enumerate(samples):
        gt_records = collect_gt_centers(
            nusc, sample, args.bev_range, args.range_bins[-1]
        )
        raw, filtered = collect_radar_candidates(
            nusc=nusc,
            sample=sample,
            use_sweeps=args.use_sweeps,
            bev_range=args.bev_range,
            min_range=args.min_radar_range,
            max_range=args.range_bins[-1],
            max_abs_speed=args.max_abs_speed,
            rcs_drop_percentile=args.rcs_drop_percentile,
        )
        total_raw += len(raw.xyz)
        total_filtered += len(filtered.xyz)
        centers = np.asarray([record.center for record in gt_records], dtype=np.float64)
        if not len(centers):
            centers = np.empty((0, 3), dtype=np.float64)
        raw_distances = compute_nearest_distances(centers, raw.xyz[:, :2])
        filtered_distances = compute_nearest_distances(centers, filtered.xyz[:, :2])
        for gt, raw_distance, filtered_distance in zip(
            gt_records, raw_distances, filtered_distances
        ):
            observations.append(
                {
                    "sample_token": gt.sample_token,
                    "annotation_token": gt.annotation_token,
                    "class_name": gt.class_name,
                    "gt_range": gt.radial_range,
                    "range_bin": _range_bin_label(gt.radial_range, args.range_bins),
                    "nearest_raw": float(raw_distance),
                    "nearest_filtered": float(filtered_distance),
                }
            )
        if sample_index < args.vis_num:
            visualize_bev_case(
                out_dir / "vis" / f"{sample_index:04d}_{sample['token']}.png",
                sample["token"], gt_records, raw, filtered,
                match_threshold=4.0, bev_range=args.bev_range,
            )
        if args.verbose and (sample_index + 1) % 10 == 0:
            print(f"processed {sample_index + 1}/{len(samples)} samples")

    metadata = {
        "data_root": str(Path(args.data_root).resolve()),
        "version": args.version,
        "split": args.split,
        "num_samples": len(samples),
        "num_gt": len(observations),
        "num_raw_candidates": total_raw,
        "num_filtered_candidates": total_filtered,
        "radar_sweeps_per_sensor": args.use_sweeps,
        "radar_channels": list(RADAR_CHANNELS),
        "bev_range": args.bev_range,
        "range_bins": args.range_bins,
        "recall_thresholds": args.recall_thresholds,
        "max_abs_speed": args.max_abs_speed,
        "rcs_drop_percentile": args.rcs_drop_percentile,
        "coordinate_frame": "current sample ego (x forward, y left, z up)",
        "raw_definition": "finite, devkit-valid radar points inside BEV/range",
        "filtered_definition": "raw plus speed <= max_abs_speed and per-sample RCS percentile filter",
    }
    save_summary(out_dir, metadata, observations, args.range_bins, args.recall_thresholds)
    print(f"Saved {len(observations)} GT observations from {len(samples)} samples to {out_dir}")


if __name__ == "__main__":
    main()
