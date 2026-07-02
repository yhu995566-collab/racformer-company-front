#!/usr/bin/env python3
"""Render synchronized camera detection and radar-focused BEV video."""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from render_company_radar_video import (
    BevCanvas,
    draw_lidar,
    draw_points,
    draw_predictions,
    points_inside_predictions,
    subsample,
)
from visualize_company_alignment import (
    BOX_EDGES,
    box_corners,
    lidar2img,
    load_points,
    project,
    resolve_path,
    transform_points,
)
from visualize_company_predictions import prediction_fields


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render camera predictions beside radar front BEV")
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-key", default="CAM_FRONT")
    parser.add_argument("--radar-key", default="RADAR_FRONT")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--forward-range", type=float, default=100.0)
    parser.add_argument("--lateral-range", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--bev-width", type=int, default=360)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--max-lidar-points", type=int, default=50000)
    parser.add_argument("--max-radar-points", type=int, default=15000)
    parser.add_argument("--radar-radius", type=int, default=1)
    parser.add_argument("--highlight-radius", type=int, default=2)
    return parser.parse_args()


def camera_panel(image, radar_xyz, highlighted, boxes, scores, projection,
                 width, height):
    panel = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    source_height, source_width = panel.shape[:2]
    scale_x = width / source_width
    scale_y = height / source_height
    panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)

    uv, _, valid = project(radar_xyz, projection, (source_height, source_width))
    for index in np.flatnonzero(valid & ~highlighted):
        point = (int(round(uv[index, 0] * scale_x)),
                 int(round(uv[index, 1] * scale_y)))
        cv2.circle(panel, point, 1, (255, 255, 255), -1, cv2.LINE_AA)
    for index in np.flatnonzero(valid & highlighted):
        point = (int(round(uv[index, 0] * scale_x)),
                 int(round(uv[index, 1] * scale_y)))
        cv2.circle(panel, point, 2, (0, 255, 0), -1, cv2.LINE_AA)

    for box, score in zip(boxes, scores):
        corners = box_corners(box)
        box_uv, depth, _ = project(
            corners, projection, (source_height, source_width))
        box_uv[:, 0] *= scale_x
        box_uv[:, 1] *= scale_y
        for start, end in BOX_EDGES:
            if depth[start] <= 1e-4 or depth[end] <= 1e-4:
                continue
            start_point = tuple(np.rint(box_uv[start]).astype(np.int32))
            end_point = tuple(np.rint(box_uv[end]).astype(np.int32))
            cv2.line(
                panel, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)
        center_uv, _, center_valid = project(
            np.asarray([box[:3]], dtype=np.float32), projection,
            (source_height, source_width))
        if center_valid[0]:
            text_point = (
                int(round(center_uv[0, 0] * scale_x)),
                int(round(center_uv[0, 1] * scale_y)))
            cv2.putText(
                panel, f"{score:.2f}", text_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                cv2.LINE_AA)

    cv2.putText(
        panel, "CAMERA DETECTIONS", (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return panel


def main():
    args = parse_args()
    with args.ann_file.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    with args.predictions.open("rb") as handle:
        predictions = pickle.load(handle)
    if len(predictions) != len(infos):
        raise ValueError(
            f"Prediction count {len(predictions)} != frame count {len(infos)}")

    class CanvasArgs:
        width = args.bev_width
        height = args.height
        forward_range = args.forward_range
        lateral_range = args.lateral_range
        margin = args.margin

    canvas = BevCanvas(CanvasArgs)
    output_size = (args.camera_width + args.bev_width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {args.output}")

    data_root = args.ann_file.resolve().parent
    try:
        for frame_id, (info, prediction) in enumerate(zip(infos, predictions)):
            cam = info["cams"][args.camera_key]
            image = Image.open(resolve_path(cam["data_path"], data_root)).convert("RGB")
            projection = lidar2img(cam)

            lidar = load_points(info["lidar_path"], 5, data_root)
            if not info.get("lidar_in_ego", True):
                lidar = transform_points(lidar, info["lidar2ego"])
            lidar = subsample(lidar, args.max_lidar_points)

            radar_info = info["rads"][args.radar_key]
            radar = load_points(radar_info["data_path"], 7, data_root)
            if not radar_info.get("radar_in_ego", True):
                radar = transform_points(radar, radar_info["radar2ego"])
            radar_xyz = subsample(radar[:, :3], args.max_radar_points)
            radar_mask = (
                (radar_xyz[:, 0] >= 0.0) &
                (radar_xyz[:, 0] <= args.forward_range) &
                (np.abs(radar_xyz[:, 1]) <= args.lateral_range))
            radar_xyz = radar_xyz[radar_mask]

            boxes, scores, _ = prediction_fields(
                prediction, args.score_threshold)
            box_mask = (
                (boxes[:, 0] >= 0.0) &
                (boxes[:, 0] <= args.forward_range) &
                (np.abs(boxes[:, 1]) <= args.lateral_range))
            boxes = boxes[box_mask]
            scores = scores[box_mask]
            highlighted = points_inside_predictions(radar_xyz, boxes)

            left = camera_panel(
                image, radar_xyz, highlighted, boxes, scores, projection,
                args.camera_width, args.height)
            right = canvas.blank()
            draw_lidar(right, canvas, lidar[:, :3])
            draw_points(
                right, canvas, radar_xyz[~highlighted],
                (255, 255, 255), args.radar_radius)
            draw_points(
                right, canvas, radar_xyz[highlighted],
                (0, 255, 0), args.highlight_radius)
            draw_predictions(right, canvas, boxes, scores)
            cv2.putText(
                right, "RADAR FRONT BEV", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                cv2.LINE_AA)
            cv2.putText(
                right, f"{frame_id + 1}/{len(infos)}  {info['token']}",
                (12, args.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

            writer.write(np.concatenate([left, right], axis=1))
            if frame_id % 25 == 0 or frame_id + 1 == len(infos):
                print(f"Rendered {frame_id + 1}/{len(infos)}")
    finally:
        writer.release()
    print(f"Wrote {len(infos)} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
