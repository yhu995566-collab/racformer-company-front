#!/usr/bin/env python3
"""Render company GT and RaCFormer predictions in image and front BEV."""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualize_company_alignment import (
    ROI,
    draw_bev_boxes,
    draw_image_boxes,
    filter_roi,
    lidar2img,
    load_points,
    project,
    select_indices,
    subsample,
    transform_points,
)


DEFAULT_CLASSES = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize company front-view predictions")
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--camera-key", default="CAM_FRONT")
    parser.add_argument("--max-lidar-points", type=int, default=30000)
    return parser.parse_args()


def prediction_fields(result, score_threshold):
    result = result.get("pts_bbox", result)
    boxes_3d = result["boxes_3d"]
    boxes = np.concatenate([
        boxes_3d.gravity_center.detach().cpu().numpy(),
        boxes_3d.dims.detach().cpu().numpy(),
        boxes_3d.yaw.detach().cpu().numpy()[:, None],
    ], axis=1)
    scores = result["scores_3d"].detach().cpu().numpy()
    labels = result["labels_3d"].detach().cpu().numpy()
    keep = scores >= score_threshold
    return boxes[keep], scores[keep], labels[keep]


def render(info, result, output_path, args, class_names):
    cam = info["cams"][args.camera_key]
    image = np.asarray(Image.open(cam["data_path"]).convert("RGB"))
    projection = lidar2img(cam)

    lidar = load_points(info["lidar_path"], 5)
    if not info.get("lidar_in_ego", True):
        lidar = transform_points(lidar, info["lidar2ego"])
    lidar = subsample(filter_roi(lidar), args.max_lidar_points)

    gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
    gt_names = np.asarray(info.get("gt_names", []))
    pred_boxes, pred_scores, pred_labels = prediction_fields(
        result, args.score_threshold)
    pred_names = np.asarray([
        f"{class_names[label] if label < len(class_names) else label} {score:.2f}"
        for label, score in zip(pred_labels, pred_scores)
    ])

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    image_axis, bev_axis = axes
    image_axis.imshow(image)
    uv, depth, valid = project(lidar[:, :3], projection, image.shape)
    image_axis.scatter(
        uv[valid, 0], uv[valid, 1], c=depth[valid], cmap="turbo_r",
        s=1, alpha=0.35, vmin=ROI[0], vmax=ROI[3])
    draw_image_boxes(
        image_axis, gt_boxes, gt_names, projection, image.shape, color="lime")
    draw_image_boxes(
        image_axis, pred_boxes, pred_names, projection, image.shape, color="red")
    image_axis.set_xlim(0, image.shape[1])
    image_axis.set_ylim(image.shape[0], 0)
    image_axis.set_title("Image: GT green, prediction red")
    image_axis.axis("off")

    bev_axis.scatter(
        -lidar[:, 1], lidar[:, 0], s=0.3, c="gray", alpha=0.35,
        label="LiDAR")
    draw_bev_boxes(bev_axis, gt_boxes, gt_names, color="lime")
    draw_bev_boxes(bev_axis, pred_boxes, pred_names, color="red")
    bev_axis.scatter(0.0, 0.0, marker="^", s=80, c="blue", label="Ego")
    bev_axis.annotate(
        "FRONT", xy=(0.0, 4.0), xytext=(0.0, 0.8), ha="center",
        color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
    bev_axis.set_xlim(-ROI[4], -ROI[1])
    bev_axis.set_ylim(ROI[0], ROI[3])
    bev_axis.set_aspect("equal", adjustable="box")
    bev_axis.set_xlabel("Vehicle lateral: left (-), right (+) [m]")
    bev_axis.set_ylabel("Ego X: forward (+) [m]")
    bev_axis.set_title("Front BEV: GT green, prediction red")
    bev_axis.grid(True, linewidth=0.4, alpha=0.4)
    bev_axis.legend(loc="upper right")

    figure.suptitle(
        f"sample={info['token']} | GT={len(gt_boxes)} "
        f"pred@{args.score_threshold:g}={len(pred_boxes)}")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    with args.ann_file.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    class_names = tuple(payload.get("metadata", {}).get(
        "classes", DEFAULT_CLASSES)) if isinstance(payload, dict) else DEFAULT_CLASSES
    with args.predictions.open("rb") as handle:
        predictions = pickle.load(handle)
    if len(predictions) != len(infos):
        raise ValueError(
            f"Prediction count {len(predictions)} != sample count {len(infos)}")

    indices = select_indices(len(infos), args.indices, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        output_path = args.output_dir / f"{index:04d}_{infos[index]['token']}.png"
        render(infos[index], predictions[index], output_path, args, class_names)
        print(f"[{index + 1}/{len(infos)}] {output_path}")
    print(f"Wrote {len(indices)} visualizations to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
