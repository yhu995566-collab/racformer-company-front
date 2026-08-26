#!/usr/bin/env python3
"""Create deterministic, sequence/class-balanced RaCFormer tuning subsets."""

import argparse
import copy
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = ("car", "truck", "bicycle")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--val-source", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit", action="store_true",
        help="Use the exact selected training samples as validation data.")
    parser.add_argument(
        "--point-cloud-range", nargs=6, type=float,
        default=(0.0, -20.0, -3.0, 50.0, 20.0, 3.0))
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError("{} is not an infos payload".format(path))
    return payload


def sequence_name(info):
    token = str(info.get("token", "unknown"))
    head, separator, tail = token.rpartition("-")
    return head if separator and tail.isdigit() else token


def classes_in_roi(info, selected_classes, point_cloud_range):
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
    names = np.asarray(info.get("gt_names", []), dtype=object)
    valid = np.asarray(
        info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
    if len(boxes) == 0:
        return ()
    roi = np.asarray(point_cloud_range, dtype=np.float32)
    centers = boxes[:, :3]
    inside = (
        (centers[:, 0] >= roi[0]) & (centers[:, 0] <= roi[3]) &
        (centers[:, 1] >= roi[1]) & (centers[:, 1] <= roi[4]) &
        (centers[:, 2] >= roi[2]) & (centers[:, 2] <= roi[5]))
    present = set(names[valid & inside].tolist()) & set(selected_classes)
    return tuple(name for name in selected_classes if name in present)


def balanced_select(infos, size, classes, point_cloud_range, seed):
    if size <= 0:
        raise ValueError("subset size must be positive")
    strata = defaultdict(list)
    for index, info in enumerate(infos):
        present = classes_in_roi(info, classes, point_cloud_range)
        if not present:
            continue
        # The least common present target class defines the class stratum.
        class_name = present[-1]
        strata[(sequence_name(info), class_name)].append(index)
    if not strata:
        raise ValueError("no samples contain the requested classes in the ROI")

    rng = random.Random(seed)
    for indices in strata.values():
        rng.shuffle(indices)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected = []
    cursor = {key: 0 for key in keys}
    while len(selected) < min(size, sum(map(len, strata.values()))):
        made_progress = False
        for key in keys:
            offset = cursor[key]
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                cursor[key] += 1
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            break
    if len(selected) < size:
        raise ValueError(
            "requested {} samples but only {} eligible samples exist".format(
                size, len(selected)))
    return sorted(selected)


def _absolute_artifact_paths(info, source_root):
    def absolute(path):
        path = Path(path)
        return str(path if path.is_absolute() else (source_root / path).resolve())

    if info.get("lidar_path"):
        info["lidar_path"] = absolute(info["lidar_path"])
    if info.get("radar_path"):
        info["radar_path"] = absolute(info["radar_path"])
    for cam in info.get("cams", {}).values():
        if cam.get("data_path"):
            cam["data_path"] = absolute(cam["data_path"])
    for radar in info.get("rads", {}).values():
        if radar.get("data_path"):
            radar["data_path"] = absolute(radar["data_path"])
    for sweep in info.get("sweeps", []):
        for sensor in sweep.values():
            if isinstance(sensor, dict) and sensor.get("data_path"):
                sensor["data_path"] = absolute(sensor["data_path"])
    return info


def subset_payload(payload, indices, source):
    output = dict(payload)
    source_root = source.resolve().parent
    output["infos"] = [
        _absolute_artifact_paths(copy.deepcopy(payload["infos"][index]), source_root)
        for index in indices]
    metadata = dict(output.get("metadata", {}))
    metadata["subset_source"] = str(source.resolve())
    metadata["subset_indices"] = indices
    output["metadata"] = metadata
    return output


def describe(infos, classes, point_cloud_range):
    sequences = Counter()
    class_frames = Counter()
    class_instances = Counter()
    for info in infos:
        sequences[sequence_name(info)] += 1
        present = classes_in_roi(info, classes, point_cloud_range)
        class_frames.update(present)
        boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
        names = np.asarray(info.get("gt_names", []), dtype=object)
        valid = np.asarray(
            info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
        if len(boxes):
            roi = np.asarray(point_cloud_range, dtype=np.float32)
            xyz = boxes[:, :3]
            inside = valid & (
                (xyz[:, 0] >= roi[0]) & (xyz[:, 0] <= roi[3]) &
                (xyz[:, 1] >= roi[1]) & (xyz[:, 1] <= roi[4]) &
                (xyz[:, 2] >= roi[2]) & (xyz[:, 2] <= roi[5]))
            class_instances.update(
                name for name in names[inside].tolist() if name in classes)
    return {
        "frames": len(infos),
        "sequences": dict(sorted(sequences.items())),
        "target_class_frames": dict(class_frames),
        "target_class_instances": dict(class_instances),
    }


def dump_pickle(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def main():
    args = parse_args()
    train_payload = load_payload(args.train_source)
    train_indices = balanced_select(
        train_payload["infos"], args.train_size, tuple(args.classes),
        args.point_cloud_range, args.seed)
    train_output = subset_payload(
        train_payload, train_indices, args.train_source)

    if args.overfit:
        val_output = subset_payload(
            train_payload, train_indices, args.train_source)
    else:
        if args.val_source is None:
            raise ValueError("--val-source is required unless --overfit is used")
        val_payload = load_payload(args.val_source)
        val_size = min(args.val_size, len(val_payload["infos"]))
        val_indices = balanced_select(
            val_payload["infos"], val_size, tuple(args.classes),
            args.point_cloud_range, args.seed + 1)
        val_output = subset_payload(val_payload, val_indices, args.val_source)

    args.output_root.mkdir(parents=True, exist_ok=True)
    dump_pickle(train_output, args.output_root / "custom_infos_train_sweep.pkl")
    dump_pickle(val_output, args.output_root / "custom_infos_val_sweep.pkl")
    manifest = {
        "seed": args.seed,
        "classes": args.classes,
        "point_cloud_range": args.point_cloud_range,
        "overfit": args.overfit,
        "train": describe(
            train_output["infos"], args.classes, args.point_cloud_range),
        "val": describe(
            val_output["infos"], args.classes, args.point_cloud_range),
    }
    (args.output_root / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
