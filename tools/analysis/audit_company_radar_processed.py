#!/usr/bin/env python3
"""Fail-fast metadata audit for converted company radar splits."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-missing-examples", type=int, default=20)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def range_label(x):
    for low in range(0, 350, 50):
        high = low + 50
        if low <= x < high or (high == 350 and x == high):
            return "{}-{}m".format(low, high)
    return "outside_0_350m"


def audit(root, splits, max_missing_examples=20):
    token_owner, overlap, missing, split_summary = {}, [], [], {}
    missing_count = 0
    for split in splits:
        info_path = root / "custom_infos_{}_sweep.pkl".format(split)
        if not info_path.is_file() or info_path.stat().st_size == 0:
            raise FileNotFoundError(str(info_path))
        with info_path.open("rb") as stream:
            infos = pickle.load(stream).get("infos", [])
        if not infos:
            raise ValueError("{} contains no infos".format(info_path))
        classes, sources, ranges = Counter(), Counter(), Counter()
        timestamps, radar_entries = [], 0
        for info in infos:
            token = str(info["token"])
            if token in token_owner:
                overlap.append({"token": token, "first": token_owner[token],
                                "second": split})
            token_owner[token] = split
            timestamps.append(int(info["timestamp"]))
            boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
            names = np.asarray(info.get("gt_names", []), dtype=object)
            gt_sources = np.asarray(info.get(
                "gt_sources", np.full(len(boxes), -1)))
            classes.update(str(name) for name in names)
            sources.update(str(int(value)) for value in gt_sources)
            ranges.update(range_label(float(box[0])) for box in boxes)
            entries = [info["rads"]["RADAR_FRONT"]]
            entries.extend(
                sweep["RADAR_FRONT"] for sweep in info.get("sweeps", [])
                if "RADAR_FRONT" in sweep)
            for entry in entries:
                radar_entries += 1
                path = resolve(root, entry["data_path"])
                if not path.is_file() or path.stat().st_size == 0:
                    missing_count += 1
                    if len(missing) < max_missing_examples:
                        missing.append(str(path))
        split_summary[split] = {
            "frames": len(infos),
            "gt_count": int(sum(classes.values())),
            "class_counts": dict(sorted(classes.items())),
            "source_counts": dict(sorted(sources.items())),
            "forward_range_counts": dict(sorted(ranges.items())),
            "radar_entries_checked": radar_entries,
            "timestamp_min": min(timestamps),
            "timestamp_max": max(timestamps),
        }
    return {
        "processed_root": str(root),
        "splits": split_summary,
        "token_overlap_count": len(overlap),
        "token_overlap_examples": overlap[:20],
        "missing_radar_path_count": missing_count,
        "missing_radar_path_examples": missing,
        "passed": not overlap and missing_count == 0,
    }


def main():
    args = parse_args()
    result = audit(args.processed_root.resolve(), args.splits,
                   args.max_missing_examples)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
