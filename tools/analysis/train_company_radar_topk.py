#!/usr/bin/env python3
"""Train and evaluate the 200m company-radar learned Top-K baseline.

This is a detector-independent stage-2 experiment.  A shared point MLP learns
class-agnostic objectness and a BEV centre residual from the train split.  The
val split compares random, RCS, and learned Top-K selection at several K values
against the all-candidate geometric ceiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = ROOT / "models" / "gaussian_prior" / "candidate_scorer.py"
SPEC = importlib.util.spec_from_file_location("company_candidate_scorer", SCORER_PATH)
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCORER_MODULE
SPEC.loader.exec_module(SCORER_MODULE)

RadarCandidateScorer = SCORER_MODULE.RadarCandidateScorer
build_box_candidate_targets = SCORER_MODULE.build_box_candidate_targets
candidate_center_offset_loss = SCORER_MODULE.candidate_center_offset_loss
candidate_scoring_loss = SCORER_MODULE.candidate_scoring_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path,
                        help="Optional independent root containing test info")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--target-sigma", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--offset-min-target", type=float, default=0.5)
    parser.add_argument("--use-frames", type=int, default=4)
    parser.add_argument("--topk", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(1.0, 2.0, 4.0, 8.0))
    parser.add_argument("--point-cloud-range", type=float, nargs=6,
                        default=(0.0, -20.0, -3.0, 200.0, 20.0, 3.0))
    parser.add_argument("--range-bins", type=float, nargs="+",
                        default=(0.0, 50.0, 100.0, 150.0, 200.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-val-frames", type=int)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_infos(root, split):
    path = root / "custom_infos_{}_sweep.pkl".format(split)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return sorted(payload["infos"], key=lambda item: int(item["timestamp"]))


def point_roi_mask(points, point_cloud_range):
    pc = np.asarray(point_cloud_range, dtype=np.float32)
    return (
        np.isfinite(points[:, :7]).all(axis=1)
        & (points[:, 0] >= pc[0]) & (points[:, 0] <= pc[3])
        & (points[:, 1] >= pc[1]) & (points[:, 1] <= pc[4])
        & (points[:, 2] >= pc[2]) & (points[:, 2] <= pc[5])
    )


def box_roi_mask(boxes, point_cloud_range):
    pc = np.asarray(point_cloud_range, dtype=np.float32)
    return (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 0] >= pc[0]) & (boxes[:, 0] <= pc[3])
        & (boxes[:, 1] >= pc[1]) & (boxes[:, 1] <= pc[4])
        & (boxes[:, 2] >= pc[2]) & (boxes[:, 2] <= pc[5])
    )


def load_radar_entry(root, entry, reference_timestamp, point_cloud_range):
    points = np.asarray(np.load(resolve(root, entry["data_path"])),
                        dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] < 7:
        raise ValueError("radar points must have shape [N,>=7]")
    points = points[:, :7]
    if not entry.get("radar_in_ego", True):
        transform = np.asarray(entry["radar2ego"], dtype=np.float32)
        xyz1 = np.column_stack(
            [points[:, :3], np.ones(len(points), dtype=np.float32)])
        points[:, :3] = (xyz1 @ transform.T)[:, :3]
        velocity3 = np.column_stack(
            [points[:, 4:6], np.zeros(len(points), dtype=np.float32)])
        points[:, 4:6] = (velocity3 @ transform[:3, :3].T)[:, :2]
    points[:, 6] = (int(reference_timestamp) - int(entry["timestamp"])) / 1e6
    return points[point_roi_mask(points, point_cloud_range)]


def collect_radar_features(root, info, use_frames, point_cloud_range):
    current = info["rads"]["RADAR_FRONT"]
    entries = [current]
    entries.extend(
        sweep["RADAR_FRONT"] for sweep in info.get("sweeps", [])
        if "RADAR_FRONT" in sweep)
    unique_entries, seen = [], set()
    for entry in entries:
        key = (str(entry["data_path"]), int(entry.get("timestamp", -1)))
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
        if len(unique_entries) == use_frames:
            break
    chunks = [
        load_radar_entry(root, entry, current["timestamp"], point_cloud_range)
        for entry in unique_entries
    ]
    return (np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            if chunks else np.empty((0, 7), dtype=np.float32))


def frame_gt(info, point_cloud_range):
    boxes = np.asarray(info["gt_boxes"], dtype=np.float32).reshape(-1, 7)
    names = np.asarray(info["gt_names"], dtype=object)
    sources = np.asarray(info.get("gt_sources", np.full(len(boxes), -1)))
    keep = box_roi_mask(boxes, point_cloud_range)
    return boxes[keep], names[keep], sources[keep]


def nearest_distances(gt_xy, candidates_xy):
    if len(gt_xy) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(candidates_xy) == 0:
        return np.full((len(gt_xy),), np.inf, dtype=np.float64)
    delta = gt_xy[:, None, :] - candidates_xy[None, :, :]
    return np.sqrt(np.square(delta).sum(axis=-1)).min(axis=1)


def stable_rng(seed, token):
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    token_seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng((int(seed) + token_seed) % (2 ** 63 - 1))


def top_indices(values, count):
    count = min(int(count), len(values))
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    # argpartition avoids sorting thousands of discarded returns.
    selected = np.argpartition(values, len(values) - count)[-count:]
    return selected[np.argsort(values[selected])[::-1]]


def summarize(records, key, thresholds):
    distance = np.asarray([record[key] for record in records], dtype=np.float64)
    finite = distance[np.isfinite(distance)]
    return {
        "gt_count": len(records),
        "finite_nearest_count": int(len(finite)),
        "nearest_distance_m": {
            "p50": None if not len(finite) else float(np.percentile(finite, 50)),
            "p90": None if not len(finite) else float(np.percentile(finite, 90)),
            "p95": None if not len(finite) else float(np.percentile(finite, 95)),
        },
        "recall": {
            "within_{}m".format(threshold): (
                None if not len(distance) else float(np.mean(distance <= threshold)))
            for threshold in thresholds
        },
    }


def grouped_summary(records, key, thresholds, range_bins):
    result = {"overall": summarize(records, key, thresholds)}
    for field in ("class_name", "source"):
        groups = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(record)
        result["by_{}".format(field)] = {
            name: summarize(items, key, thresholds)
            for name, items in sorted(groups.items())
        }
    result["by_range"] = {}
    for low, high in zip(range_bins[:-1], range_bins[1:]):
        items = [item for item in records if low <= item["forward_x_m"] < high]
        result["by_range"]["{:g}-{:g}m".format(low, high)] = summarize(
            items, key, thresholds)
    return result


def train_epoch(model, optimizer, root, infos, args, device, epoch):
    model.train()
    order = np.random.default_rng(args.seed + epoch).permutation(len(infos))
    totals = defaultdict(float)
    batches = [order[start:start + args.batch_size]
               for start in range(0, len(order), args.batch_size)]
    progress = tqdm(batches, desc="Train epoch {}/{}".format(epoch, args.epochs))
    for batch_indices in progress:
        point_tensors, box_tensors = [], []
        for info_index in batch_indices:
            info = infos[int(info_index)]
            points = collect_radar_features(
                root, info, args.use_frames, args.point_cloud_range)
            boxes, _, _ = frame_gt(info, args.point_cloud_range)
            point_tensors.append(torch.from_numpy(points).to(device))
            box_tensors.append(torch.from_numpy(boxes).to(device))
        output = model(point_tensors)
        targets = build_box_candidate_targets(
            output["points"], output["candidate_mask"], box_tensors,
            target_sigma=args.target_sigma)
        score_loss = candidate_scoring_loss(
            output["objectness_logits"], targets["objectness_targets"],
            output["candidate_mask"], positive_weight=args.positive_weight)
        offset_loss = candidate_center_offset_loss(
            output["center_offsets"], targets["center_offsets"],
            targets["objectness_targets"], output["candidate_mask"],
            min_target=args.offset_min_target)
        loss = score_loss + args.offset_loss_weight * offset_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["score_loss"] += float(score_loss.detach())
        totals["offset_loss"] += float(offset_loss.detach())
        progress.set_postfix(loss="{:.4f}".format(float(loss.detach())))
    return {key: value / max(len(batches), 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, root, infos, args, device, split_name="val"):
    model.eval()
    records = []
    selection = defaultdict(lambda: defaultdict(float))
    max_k = max(args.topk)
    for info in tqdm(infos, desc="Evaluate {} Top-K".format(split_name)):
        points = collect_radar_features(
            root, info, args.use_frames, args.point_cloud_range)
        boxes, names, sources = frame_gt(info, args.point_cloud_range)
        point_tensor = torch.from_numpy(points).to(device)
        box_tensor = torch.from_numpy(boxes).to(device)
        output = model([point_tensor])
        count = len(points)
        logits = output["objectness_logits"][0, :count].cpu().numpy()
        offsets = output["center_offsets"][0, :count].cpu().numpy()
        targets = build_box_candidate_targets(
            output["points"], output["candidate_mask"], [box_tensor],
            target_sigma=args.target_sigma)
        target_score = targets["objectness_targets"][0, :count].cpu().numpy()
        rng = stable_rng(args.seed, info["token"])
        random_order = rng.permutation(count)
        ranks = {
            "random": random_order,
            "rcs": top_indices(points[:, 3], max_k),
            "mlp": top_indices(logits, max_k),
        }
        method_candidates = {"all_candidates": points[:, :2]}
        for method, ranking in ranks.items():
            for k in args.topk:
                chosen = ranking[:min(k, len(ranking))]
                key = "{}_top{}".format(method, k)
                method_candidates[key] = points[chosen, :2]
                selection[key]["selected"] += len(chosen)
                selection[key]["positive"] += float(
                    np.count_nonzero(target_score[chosen] >= args.offset_min_target))
                if method == "mlp":
                    method_candidates[key + "_corrected"] = (
                        points[chosen, :2] + offsets[chosen])

        distances = {
            key: nearest_distances(boxes[:, :2], candidate_xy)
            for key, candidate_xy in method_candidates.items()
        }
        for gt_index, box in enumerate(boxes):
            record = {
                "token": str(info["token"]),
                "class_name": str(names[gt_index]),
                "source": int(sources[gt_index]),
                "forward_x_m": float(box[0]),
                "y_m": float(box[1]),
            }
            for key, values in distances.items():
                record[key] = float(values[gt_index])
            records.append(record)

    metric_keys = [key for key in records[0] if key not in {
        "token", "class_name", "source", "forward_x_m", "y_m"}]
    metrics = {
        key: grouped_summary(records, key, args.thresholds, args.range_bins)
        for key in metric_keys
    }
    selection_summary = {}
    for key, values in selection.items():
        selected = values["selected"]
        selection_summary[key] = {
            "selected_candidates": int(selected),
            "positive_candidates": int(values["positive"]),
            "positive_fraction": float(values["positive"] / max(selected, 1.0)),
        }
    return records, metrics, selection_summary


def main():
    args = parse_args()
    if args.epochs <= 0 or args.use_frames <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, use_frames, and batch_size must be positive")
    if sorted(set(args.topk)) != sorted(args.topk) or min(args.topk) <= 0:
        raise ValueError("topk must contain unique positive values in ascending order")
    if args.point_cloud_range[3] not in (50.0, 200.0, 350.0):
        raise ValueError("company experiment xmax must be 50m, 200m, or 350m")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    root = args.processed_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_infos = load_infos(root, "train")
    val_infos = load_infos(root, "val")
    test_root = args.test_root.resolve() if args.test_root else None
    test_infos = load_infos(test_root, "test") if test_root else []
    if args.max_train_frames is not None:
        train_infos = train_infos[:args.max_train_frames]
    if args.max_val_frames is not None:
        val_infos = val_infos[:args.max_val_frames]
    print("train frames: {}".format(len(train_infos)), flush=True)
    print("val frames: {}".format(len(val_infos)), flush=True)
    if test_root:
        print("independent test frames: {}".format(len(test_infos)), flush=True)
    print("device: {}".format(device), flush=True)

    model = RadarCandidateScorer(
        point_cloud_range=args.point_cloud_range,
        topk=max(args.topk), hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = train_epoch(
            model, optimizer, root, train_infos, args, device, epoch)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        print("epoch {}: {}".format(epoch, json.dumps(epoch_metrics,
                                                       sort_keys=True)), flush=True)

    range_tag = int(args.point_cloud_range[3])
    checkpoint_path = args.out_dir / "radar_candidate_scorer_{}m.pth".format(
        range_tag)
    torch.save({
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "config": vars(args),
        "history": history,
    }, checkpoint_path)
    records, metrics, selection = evaluate(
        model, root, val_infos, args, device, split_name="val")
    test_records, test_metrics, test_selection = ([], {}, {})
    if test_root:
        test_records, test_metrics, test_selection = evaluate(
            model, test_root, test_infos, args, device, split_name="test")
    summary = {
        "experiment": "company_radar_learned_topk_{}m".format(range_tag),
        "processed_root": str(root),
        "train_frames": len(train_infos),
        "val_frames": len(val_infos),
        "val_gt_count": len(records),
        "test_root": None if test_root is None else str(test_root),
        "test_frames": len(test_infos),
        "test_gt_count": len(test_records),
        "point_cloud_range": list(args.point_cloud_range),
        "use_frames": args.use_frames,
        "topk": list(args.topk),
        "thresholds_m": list(args.thresholds),
        "target_sigma_m": args.target_sigma,
        "history": history,
        "selection": selection,
        "metrics": metrics,
        "test_selection": test_selection,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    summary_path = args.out_dir / "company_radar_topk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                       default=str) + "\n")
    records_path = args.out_dir / "company_radar_topk_records.csv"
    with records_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    test_records_path = None
    if test_records:
        test_records_path = args.out_dir / "company_radar_topk_test_records.csv"
        with test_records_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(test_records[0]))
            writer.writeheader()
            writer.writerows(test_records)

    print("checkpoint: {}".format(checkpoint_path.resolve()), flush=True)
    print("summary: {}".format(summary_path.resolve()), flush=True)
    print("records: {}".format(records_path.resolve()), flush=True)
    if test_records_path:
        print("test records: {}".format(test_records_path.resolve()), flush=True)
    compact = {
        name: value["overall"] for name, value in metrics.items()
        if name == "all_candidates" or name.startswith("mlp_top")
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
