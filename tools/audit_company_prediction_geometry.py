#!/usr/bin/env python3
"""Measure matched RaCFormer box errors to explain BEV/3D AP gaps."""

import argparse
import importlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--ann-file", type=Path)
    parser.add_argument("--classes", nargs="+", default=("car", "truck", "bicycle"))
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--match-bev-iou", type=float, default=0.1)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sequence_name(token):
    head, separator, tail = str(token).rpartition("-")
    return head if separator and tail.isdigit() else str(token)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def greedy_matches(bev_iou, scores, threshold):
    matches = []
    used_gt = set()
    for pred_index in np.argsort(-scores):
        overlaps = bev_iou[pred_index].copy()
        if used_gt:
            overlaps[list(used_gt)] = -1.0
        gt_index = int(overlaps.argmax()) if overlaps.size else -1
        if gt_index >= 0 and overlaps[gt_index] >= threshold:
            used_gt.add(gt_index)
            matches.append((int(pred_index), gt_index))
    return matches


def empty_accumulator():
    return {
        "gt": 0, "predictions": 0, "matches": 0,
        "dx": [], "dy": [], "dz": [], "dw": [], "dl": [], "dh": [],
        "dyaw_deg": [], "bev_iou": [], "iou_3d": [], "score": [],
    }


def add_frame(accumulator, pred_boxes, pred_scores, gt_boxes, matches,
              bev_iou, iou_3d):
    accumulator["gt"] += len(gt_boxes)
    accumulator["predictions"] += len(pred_boxes)
    accumulator["matches"] += len(matches)
    pred = pred_boxes.tensor.detach().cpu().numpy()
    gt = gt_boxes.tensor.detach().cpu().numpy()
    for pred_index, gt_index in matches:
        delta = pred[pred_index, :6] - gt[gt_index, :6]
        for key, value in zip(("dx", "dy", "dz", "dw", "dl", "dh"), delta):
            accumulator[key].append(float(value))
        accumulator["dyaw_deg"].append(
            math.degrees(wrap_angle(float(pred[pred_index, 6] - gt[gt_index, 6]))))
        accumulator["bev_iou"].append(float(bev_iou[pred_index, gt_index]))
        accumulator["iou_3d"].append(float(iou_3d[pred_index, gt_index]))
        accumulator["score"].append(float(pred_scores[pred_index]))


def summarize(accumulator):
    summary = {
        key: int(accumulator[key])
        for key in ("gt", "predictions", "matches")}
    summary["matched_gt_fraction"] = (
        accumulator["matches"] / max(accumulator["gt"], 1))
    for key in ("dx", "dy", "dz", "dw", "dl", "dh", "dyaw_deg"):
        values = np.asarray(accumulator[key], dtype=np.float64)
        summary[key] = {
            "signed_mean": float(values.mean()) if len(values) else None,
            "abs_p50": float(np.percentile(np.abs(values), 50)) if len(values) else None,
            "abs_p90": float(np.percentile(np.abs(values), 90)) if len(values) else None,
            "abs_p95": float(np.percentile(np.abs(values), 95)) if len(values) else None,
        }
    for key in ("bev_iou", "iou_3d", "score"):
        values = np.asarray(accumulator[key], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "p10": float(np.percentile(values, 10)) if len(values) else None,
            "p50": float(np.percentile(values, 50)) if len(values) else None,
        }
    return summary


def main():
    args = parse_args()
    importlib.import_module("models")
    importlib.import_module("loaders")
    config = Config.fromfile(args.config)
    dataset_config = config.data[args.split].copy()
    if args.ann_file is not None:
        dataset_config["ann_file"] = str(args.ann_file.resolve())
        dataset_config["data_root"] = str(args.ann_file.resolve().parent) + "/"
    dataset = build_dataset(dataset_config)
    with args.predictions.open("rb") as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(dataset):
        raise ValueError(
            "prediction count {} != dataset count {}".format(
                len(predictions), len(dataset)))

    class_to_id = {name: index for index, name in enumerate(dataset.CLASSES)}
    unknown = set(args.classes) - set(class_to_id)
    if unknown:
        raise ValueError("unknown classes: {}".format(sorted(unknown)))
    totals = defaultdict(empty_accumulator)

    for frame_index, result in enumerate(predictions):
        pred_boxes, scores, pred_labels = dataset._result_fields(
            result, args.nms_iou_threshold)
        keep = scores >= args.score_threshold
        pred_boxes = pred_boxes[np.flatnonzero(keep).tolist()]
        scores = scores[keep]
        pred_labels = pred_labels[keep]
        gt_boxes, gt_labels = dataset._filtered_gt(frame_index)
        token = dataset.data_infos[frame_index].get("token", frame_index)
        sequence = sequence_name(token)
        for class_name in args.classes:
            class_id = class_to_id[class_name]
            pred_indices = np.flatnonzero(pred_labels == class_id)
            gt_indices = np.flatnonzero(gt_labels == class_id)
            class_pred = pred_boxes[pred_indices.tolist()]
            class_scores = scores[pred_indices]
            class_gt = gt_boxes[gt_indices.tolist()]
            bev_iou = dataset._bev_iou(class_pred, class_gt).numpy()
            iou_3d = dataset._iou_3d(class_pred, class_gt).numpy()
            matches = greedy_matches(
                bev_iou, class_scores, args.match_bev_iou)
            add_frame(
                totals[("class", class_name)], class_pred, class_scores,
                class_gt, matches, bev_iou, iou_3d)
            add_frame(
                totals[("sequence", sequence)], class_pred, class_scores,
                class_gt, matches, bev_iou, iou_3d)
            add_frame(
                totals[("overall", "main")], class_pred, class_scores,
                class_gt, matches, bev_iou, iou_3d)

    report = {
        "config": str(Path(args.config).resolve()),
        "predictions": str(args.predictions.resolve()),
        "split": args.split,
        "classes": args.classes,
        "score_threshold": args.score_threshold,
        "match_bev_iou": args.match_bev_iou,
        "overall": summarize(totals[("overall", "main")]),
        "by_class": {
            name: summarize(totals[("class", name)]) for name in args.classes},
        "by_sequence": {
            name: summarize(value)
            for (kind, name), value in sorted(totals.items())
            if kind == "sequence"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    overall = report["overall"]
    print("GT/pred/matched: {}/{}/{}".format(
        overall["gt"], overall["predictions"], overall["matches"]))
    for key in ("dx", "dy", "dz", "dw", "dl", "dh", "dyaw_deg"):
        item = overall[key]
        print("{} signed_mean={} abs_p50={} abs_p95={}".format(
            key, item["signed_mean"], item["abs_p50"], item["abs_p95"]))
    print("BEV IoU:", overall["bev_iou"])
    print("3D IoU:", overall["iou_3d"])
    print("report:", args.output.resolve())


if __name__ == "__main__":
    main()
