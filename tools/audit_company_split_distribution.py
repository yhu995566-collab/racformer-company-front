#!/usr/bin/env python3
"""Audit class and geometry drift between company train/val/test splits."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", action="append", required=True, metavar="NAME=PKL",
        help="Repeat for every split to compare.")
    parser.add_argument("--classes", default="car,truck,bicycle")
    parser.add_argument("--horizontal-fov-deg", type=float, default=120.0)
    parser.add_argument("--range", nargs=6, type=float, default=(0, -20, -3, 50, 20, 3))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sequence_name(info):
    token = str(info.get("token", info.get("sample_idx", "unknown")))
    return token.rsplit("-", 1)[0]


def summarize(path, classes, pc_range, fov_deg):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    x0, y0, z0, x1, y1, z1 = pc_range
    slope = np.tan(np.deg2rad(fov_deg / 2.0))
    totals = Counter()
    frame_counts = Counter()
    distance_bins = Counter()
    velocity = Counter()
    per_sequence = defaultdict(lambda: {
        "frames": 0, "gt": Counter(), "frames_with_gt": Counter()})

    for info in infos:
        sequence = sequence_name(info)
        per_sequence[sequence]["frames"] += 1
        boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
        names = np.asarray(info.get("gt_names", []), dtype=object)
        valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
        keep = valid & np.isin(names, classes)
        if len(boxes):
            keep &= ((boxes[:, 0] >= x0) & (boxes[:, 0] <= x1)
                     & (boxes[:, 1] >= y0) & (boxes[:, 1] <= y1)
                     & (boxes[:, 2] >= z0) & (boxes[:, 2] <= z1)
                     & (np.abs(boxes[:, 1]) <= boxes[:, 0] * slope))
        boxes, names = boxes[keep], names[keep]
        velocities = np.asarray(
            info.get("gt_velocity", np.zeros((len(valid), 2))), dtype=np.float32)
        velocities = velocities[keep] if len(velocities) == len(valid) else np.zeros((len(boxes), 2))
        present = set()
        for box, name, vel in zip(boxes, names, velocities):
            name = str(name)
            totals[name] += 1
            per_sequence[sequence]["gt"][name] += 1
            present.add(name)
            distance_bins[(name, "0_25" if box[0] < 25 else "25_50")] += 1
            velocity["finite"] += int(np.isfinite(vel).all())
            velocity["nonzero"] += int(np.linalg.norm(np.nan_to_num(vel)) > 1e-3)
        for name in present:
            frame_counts[name] += 1
            per_sequence[sequence]["frames_with_gt"][name] += 1

    result = {
        "path": str(path.resolve()), "frames": len(infos),
        "sequences": len(per_sequence), "gt": dict(totals),
        "frames_with_gt": dict(frame_counts),
        "gt_per_frame": {name: totals[name] / max(len(infos), 1) for name in classes},
        "distance_bins": {"{}:{}m".format(*key): value
                          for key, value in sorted(distance_bins.items())},
        "velocity": dict(velocity),
        "per_sequence": {
            name: {"frames": item["frames"], "gt": dict(item["gt"]),
                   "frames_with_gt": dict(item["frames_with_gt"])}
            for name, item in sorted(per_sequence.items())},
    }
    return result


def main():
    args = parse_args()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    report = {"classes": classes, "horizontal_fov_deg": args.horizontal_fov_deg,
              "range": args.range, "splits": {}}
    for value in args.split:
        if "=" not in value:
            raise ValueError("--split must use NAME=PKL: {!r}".format(value))
        name, raw_path = value.split("=", 1)
        report["splits"][name] = summarize(
            Path(raw_path), classes, args.range, args.horizontal_fov_deg)

    for name, item in report["splits"].items():
        print("\n=== {} ===".format(name))
        print("frames={} sequences={}".format(item["frames"], item["sequences"]))
        print("gt={}".format(item["gt"]))
        print("gt_per_frame={}".format({key: round(value, 4)
                                         for key, value in item["gt_per_frame"].items()}))
        for sequence, sequence_item in item["per_sequence"].items():
            print("  {} frames={} gt={}".format(
                sequence, sequence_item["frames"], sequence_item["gt"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("\nreport={}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
