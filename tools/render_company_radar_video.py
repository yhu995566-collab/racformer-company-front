#!/usr/bin/env python3
"""Render a radar-focused front-BEV MP4 for the full company sequence."""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from visualize_company_alignment import load_points, transform_points
from visualize_company_predictions import prediction_fields


def parse_args():
    parser = argparse.ArgumentParser(description="Render company radar BEV video")
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radar-key", default="RADAR_FRONT")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--forward-range", type=float, default=100.0)
    parser.add_argument("--lateral-range", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--margin", type=int, default=35)
    parser.add_argument("--max-lidar-points", type=int, default=50000)
    parser.add_argument("--max-radar-points", type=int, default=15000)
    parser.add_argument("--radar-radius", type=int, default=1)
    parser.add_argument("--highlight-radius", type=int, default=2)
    return parser.parse_args()


def subsample(points, maximum):
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


class BevCanvas:
    def __init__(self, args):
        self.width = args.width
        self.height = args.height
        self.forward = args.forward_range
        self.lateral = args.lateral_range
        self.margin = args.margin
        self.scale = min(
            (self.width - 2 * self.margin) / (2 * self.lateral),
            (self.height - 2 * self.margin) / self.forward)
        self.center_u = self.width // 2
        self.bottom_v = self.height - self.margin

    def pixels(self, xyz):
        # Driver-facing view: vehicle-right (-ego Y) is screen-right.
        u = np.rint(self.center_u - xyz[:, 1] * self.scale).astype(np.int32)
        v = np.rint(self.bottom_v - xyz[:, 0] * self.scale).astype(np.int32)
        valid = (
            (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= self.forward) &
            (np.abs(xyz[:, 1]) <= self.lateral) &
            (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height))
        return u, v, valid

    def blank(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        left = int(round(self.center_u - self.lateral * self.scale))
        right = int(round(self.center_u + self.lateral * self.scale))
        top = int(round(self.bottom_v - self.forward * self.scale))
        cv2.rectangle(frame, (left, top), (right, self.bottom_v), (35, 35, 35), 1)
        for distance in range(50, int(self.forward) + 1, 50):
            y = int(round(self.bottom_v - distance * self.scale))
            cv2.line(frame, (left, y), (right, y), (25, 25, 25), 1)
            cv2.putText(
                frame, f"{distance}m", (right + 6, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1,
                cv2.LINE_AA)
        return frame


def box_bev_corners(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array(
        [[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]],
        dtype=np.float32) * 0.5
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    corners = local @ rotation.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_points(frame, canvas, points, color, radius):
    if len(points) == 0:
        return
    u, v, valid = canvas.pixels(points)
    for px, py in zip(u[valid], v[valid]):
        cv2.circle(frame, (int(px), int(py)), radius, color, -1, cv2.LINE_AA)


def draw_lidar(frame, canvas, points):
    if len(points) == 0:
        return
    u, v, valid = canvas.pixels(points)
    frame[v[valid], u[valid]] = (65, 65, 65)


def draw_predictions(frame, canvas, boxes, scores):
    for box, score in zip(boxes, scores):
        corners = box_bev_corners(box)
        xyz = np.column_stack(
            [corners[:, 0], corners[:, 1], np.zeros(4, dtype=np.float32)])
        u, v, valid = canvas.pixels(xyz)
        if not valid.any():
            continue
        polygon = np.column_stack([u, v]).reshape(-1, 1, 2)
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame, f"{score:.2f}", tuple(polygon[0, 0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def points_inside_predictions(points, boxes):
    """Return a BEV mask for radar points enclosed by any predicted box."""
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        x, y, _, dx, dy, _, yaw = box[:7]
        delta_x = points[:, 0] - x
        delta_y = points[:, 1] - y
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        inside |= (
            (np.abs(local_x) <= dx * 0.5) &
            (np.abs(local_y) <= dy * 0.5))
    return inside


def main():
    args = parse_args()
    with args.ann_file.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    with args.predictions.open("rb") as handle:
        predictions = pickle.load(handle)
    if len(predictions) != len(infos):
        raise ValueError(
            f"Prediction count {len(predictions)} != frame count {len(infos)}. "
            "Run inference on the 1041-frame all-keyframe info file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {args.output}")

    canvas = BevCanvas(args)
    data_root = args.ann_file.resolve().parent
    try:
        for frame_id, (info, prediction) in enumerate(zip(infos, predictions)):
            lidar = load_points(info["lidar_path"], 5, data_root)
            if not info.get("lidar_in_ego", True):
                lidar = transform_points(lidar, info["lidar2ego"])
            if len(lidar) > args.max_lidar_points:
                indices = np.linspace(
                    0, len(lidar) - 1, args.max_lidar_points, dtype=np.int64)
                lidar = lidar[indices]

            radar_info = info["rads"][args.radar_key]
            radar = load_points(radar_info["data_path"], 7, data_root)
            if not radar_info.get("radar_in_ego", True):
                radar = transform_points(radar, radar_info["radar2ego"])
            radar_xyz = subsample(radar[:, :3], args.max_radar_points)

            boxes, scores, _ = prediction_fields(
                prediction, args.score_threshold)
            highlighted = points_inside_predictions(radar_xyz, boxes)
            frame = canvas.blank()
            draw_lidar(frame, canvas, lidar[:, :3])
            draw_points(
                frame, canvas, radar_xyz[~highlighted],
                (255, 255, 255), args.radar_radius)
            draw_points(
                frame, canvas, radar_xyz[highlighted],
                (0, 255, 0), args.highlight_radius)
            draw_predictions(frame, canvas, boxes, scores)
            cv2.putText(
                frame, f"frame {frame_id + 1}/{len(infos)}  token {info['token']}",
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(frame)
            if frame_id % 25 == 0 or frame_id + 1 == len(infos):
                print(f"Rendered {frame_id + 1}/{len(infos)}")
    finally:
        writer.release()
    print(f"Wrote {len(infos)} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
