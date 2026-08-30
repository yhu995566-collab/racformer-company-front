#!/usr/bin/env python3
"""Create leakage-safe, sequence-balanced train/dev metadata splits.

The source artifacts are reused in place.  Validation keyframes are selected
from several evenly spaced contiguous blocks in every source sequence.  A
guard on both sides of each block prevents four-frame samples from sharing
camera/radar artifacts across train and validation.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--blocks-per-sequence", type=int, default=3)
    parser.add_argument("--guard-frames", type=int, default=3)
    return parser.parse_args()


def sequence_name(info):
    if info.get("sequence"):
        return str(info["sequence"])
    token = str(info.get("token", "unknown"))
    head, separator, tail = token.rpartition("-")
    return head if separator and tail.isdigit() else token


def load_sources(paths):
    infos = []
    metadata = {}
    seen = set()
    for path in paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or "infos" not in payload:
            raise ValueError("{} is not an infos payload".format(path))
        metadata.update(payload.get("metadata", {}))
        for info in payload["infos"]:
            token = str(info["token"])
            if token in seen:
                raise ValueError("duplicate token across sources: {}".format(token))
            seen.add(token)
            infos.append(info)
    return infos, metadata


def distributed_lengths(total, blocks):
    base, remainder = divmod(total, blocks)
    return [base + int(index < remainder) for index in range(blocks)]


def split_sequence(infos, val_fraction, blocks, guard):
    infos = sorted(infos, key=lambda item: str(item["token"]))
    count = len(infos)
    target = max(blocks, int(round(count * val_fraction)))
    lengths = distributed_lengths(target, blocks)
    train_mask = [True] * count
    val_mask = [False] * count
    windows = []
    previous_padded_end = -1
    for block_index, length in enumerate(lengths):
        center = int(round((block_index + 1) * count / (blocks + 1)))
        start = max(guard, center - length // 2)
        start = min(start, count - guard - length)
        padded_start = start - guard
        if padded_start <= previous_padded_end:
            start += previous_padded_end - padded_start + 1
        end = start + length
        if end + guard > count:
            raise ValueError("sequence too short for requested blocked split")
        for index in range(start, end):
            val_mask[index] = True
        for index in range(start - guard, end + guard):
            train_mask[index] = False
        previous_padded_end = end + guard - 1
        windows.append({
            "first_token": infos[start]["token"],
            "last_token": infos[end - 1]["token"],
            "val_frames": length,
            "guard_before": guard,
            "guard_after": guard,
        })
    train = [item for index, item in enumerate(infos) if train_mask[index]]
    val = [item for index, item in enumerate(infos) if val_mask[index]]
    dropped = count - len(train) - len(val)
    return train, val, dropped, windows


def artifacts(info):
    values = {
        str(info.get("lidar_path", "")), str(info.get("radar_path", "")),
        str(info.get("cams", {}).get("CAM_FRONT", {}).get("data_path", "")),
    }
    for sweep in info.get("sweeps", []):
        values.add(str(sweep.get("CAM_FRONT", {}).get("data_path", "")))
        values.add(str(sweep.get("RADAR_FRONT", {}).get("data_path", "")))
    values.discard("")
    return values


def atomic_pickle(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as stream:
        temporary = Path(stream.name)
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if args.blocks_per_sequence < 1 or args.guard_frames < 0:
        raise ValueError("blocks must be positive and guard must be nonnegative")
    infos, metadata = load_sources(args.source)
    grouped = defaultdict(list)
    for info in infos:
        grouped[sequence_name(info)].append(info)

    train, val = [], []
    report = {"method": "evenly_spaced_contiguous_blocks",
              "val_fraction": args.val_fraction,
              "blocks_per_sequence": args.blocks_per_sequence,
              "guard_frames": args.guard_frames, "sequences": {}}
    for sequence in sorted(grouped):
        seq_train, seq_val, dropped, windows = split_sequence(
            grouped[sequence], args.val_fraction,
            args.blocks_per_sequence, args.guard_frames)
        train.extend(seq_train)
        val.extend(seq_val)
        report["sequences"][sequence] = {
            "source_frames": len(grouped[sequence]),
            "train_frames": len(seq_train), "val_frames": len(seq_val),
            "dropped_guard_frames": dropped, "windows": windows,
        }

    train_artifacts = set().union(*(artifacts(info) for info in train))
    val_artifacts = set().union(*(artifacts(info) for info in val))
    overlap = train_artifacts & val_artifacts
    if overlap:
        raise RuntimeError(
            "temporal artifact leakage detected ({} paths), example={}".format(
                len(overlap), sorted(overlap)[0]))
    train.sort(key=lambda item: str(item["token"]))
    val.sort(key=lambda item: str(item["token"]))
    report.update({"source_frames": len(infos), "train_frames": len(train),
                   "val_frames": len(val),
                   "dropped_guard_frames": len(infos) - len(train) - len(val),
                   "temporal_artifact_overlap": 0})
    common = {**metadata, "split_method": report["method"],
              "split_report": "split_report.json"}
    atomic_pickle({"infos": train, "metadata": {**common, "split": "train"}},
                  args.output_dir / "custom_infos_train_sweep.pkl")
    atomic_pickle({"infos": val, "metadata": {**common, "split": "val"}},
                  args.output_dir / "custom_infos_val_sweep.pkl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("BLOCKED DEV SPLIT: PASS")


if __name__ == "__main__":
    main()
