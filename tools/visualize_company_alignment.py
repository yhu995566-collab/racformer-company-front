#!/usr/bin/env python3
"""Visualize company camera, point clouds, and GT in image and front BEV."""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROI = (0.0, -12.0, -3.0, 50.0, 12.0, 3.0)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render company front-view alignment diagnostics")
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-key", default="CAM_FRONT")
    parser.add_argument("--radar-key", default="RADAR_FRONT")
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-lidar-points", type=int, default=30000)
    parser.add_argument("--max-radar-points", type=int, default=10000)
    return parser.parse_args()


def load_points(path, dim):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, dim)
    return np.asarray(points, dtype=np.float32)


def transform_points(points, transform):
    points = points.copy()
    xyz1 = np.concatenate(
        [points[:, :3], np.ones((len(points), 1), dtype=np.float32)], axis=1)
    points[:, :3] = (xyz1 @ np.asarray(transform, dtype=np.float32).T)[:, :3]
    return points


def filter_roi(points):
    x_min, y_min, z_min, x_max, y_max, z_max = ROI
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def subsample(points, maximum):
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def lidar2img(cam_info):
    sensor2lidar_r = np.asarray(
        cam_info["sensor2lidar_rotation"], dtype=np.float32)
    sensor2lidar_t = np.asarray(
        cam_info["sensor2lidar_translation"], dtype=np.float32)
    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :3] = lidar2cam_r.T
    lidar2cam[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32)
    return viewpad @ lidar2cam.T


def project(points_xyz, projection, image_shape):
    xyz1 = np.concatenate(
        [points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)], axis=1)
    projected = xyz1 @ projection.T
    depth = projected[:, 2]
    valid_depth = depth > 1e-4
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-4)
    height, width = image_shape[:2]
    valid = (
        valid_depth & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv, depth, valid


def box_corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([
        [-dx, -dy, -dz], [-dx, dy, -dz], [dx, dy, -dz], [dx, -dy, -dz],
        [-dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz], [dx, -dy, dz],
    ], dtype=np.float32) * 0.5
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return local @ rotation.T + np.array([x, y, z], dtype=np.float32)


def draw_image_boxes(axis, boxes, names, projection, image_shape):
    for box, name in zip(boxes, names):
        corners = box_corners(box)
        uv, depth, _ = project(corners, projection, image_shape)
        for start, end in BOX_EDGES:
            if depth[start] > 1e-4 and depth[end] > 1e-4:
                axis.plot(
                    [uv[start, 0], uv[end, 0]],
                    [uv[start, 1], uv[end, 1]],
                    color="lime", linewidth=1.5)
        center_uv, _, center_valid = project(
            np.asarray([box[:3]], dtype=np.float32), projection, image_shape)
        if center_valid[0]:
            axis.text(
                center_uv[0, 0], center_uv[0, 1], str(name), color="lime",
                fontsize=7, bbox=dict(facecolor="black", alpha=0.5, pad=1))


def draw_bev_boxes(axis, boxes, names):
    for box, name in zip(boxes, names):
        corners = box_corners(box)[:4]
        polygon = np.concatenate([corners, corners[:1]], axis=0)
        axis.plot(-polygon[:, 1], polygon[:, 0], color="lime", linewidth=1.5)
        axis.text(-box[1], box[0], str(name), color="darkgreen", fontsize=7)


def select_indices(length, requested, count):
    if requested:
        invalid = [index for index in requested if index < 0 or index >= length]
        if invalid:
            raise IndexError(f"Indices outside [0, {length - 1}]: {invalid}")
        return requested
    count = min(max(1, count), length)
    return np.unique(np.linspace(0, length - 1, count, dtype=np.int64)).tolist()


def render_sample(info, output_path, camera_key, radar_key,
                  max_lidar_points, max_radar_points):
    cam = info["cams"][camera_key]
    radar_info = info["rads"][radar_key]
    image = np.asarray(Image.open(cam["data_path"]).convert("RGB"))

    lidar = load_points(info["lidar_path"], 5)
    if not info.get("lidar_in_ego", True):
        lidar = transform_points(lidar, info["lidar2ego"])
    radar = load_points(radar_info["data_path"], 7)
    if not radar_info.get("radar_in_ego", True):
        radar = transform_points(radar, radar_info["radar2ego"])

    lidar = subsample(filter_roi(lidar), max_lidar_points)
    radar = subsample(filter_roi(radar), max_radar_points)
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
    names = np.asarray(info.get("gt_names", []))
    projection = lidar2img(cam)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    image_axis, bev_axis = axes
    image_axis.imshow(image)

    lidar_uv, lidar_depth, lidar_valid = project(lidar[:, :3], projection, image.shape)
    radar_uv, _, radar_valid = project(radar[:, :3], projection, image.shape)
    image_axis.scatter(
        lidar_uv[lidar_valid, 0], lidar_uv[lidar_valid, 1],
        c=lidar_depth[lidar_valid], cmap="turbo_r", s=1, alpha=0.45,
        vmin=ROI[0], vmax=ROI[3], label="LiDAR")
    image_axis.scatter(
        radar_uv[radar_valid, 0], radar_uv[radar_valid, 1],
        c="cyan", s=7, alpha=0.8, label="Radar")
    draw_image_boxes(image_axis, boxes, names, projection, image.shape)
    image_axis.set_xlim(0, image.shape[1])
    image_axis.set_ylim(image.shape[0], 0)
    image_axis.set_title("Front image projection")
    image_axis.axis("off")
    image_axis.legend(loc="upper right")

    # Screen-right is vehicle-right (-ego Y), while forward (+ego X) is up.
    bev_axis.scatter(-lidar[:, 1], lidar[:, 0], s=0.3, c="gray", alpha=0.4,
                     label="LiDAR")
    bev_axis.scatter(-radar[:, 1], radar[:, 0], s=5, c="tab:blue", alpha=0.8,
                     label="Radar")
    draw_bev_boxes(bev_axis, boxes, names)
    bev_axis.scatter(0.0, 0.0, marker="^", s=80, c="red", label="Ego")
    bev_axis.annotate(
        "FRONT", xy=(0.0, 4.0), xytext=(0.0, 0.8), ha="center",
        color="red", arrowprops=dict(arrowstyle="->", color="red"))
    bev_axis.set_xlim(-ROI[4], -ROI[1])
    bev_axis.set_ylim(ROI[0], ROI[3])
    bev_axis.set_aspect("equal", adjustable="box")
    bev_axis.set_xlabel("Vehicle lateral: left (-), right (+) [m]")
    bev_axis.set_ylabel("Ego X: forward (+) [m]")
    bev_axis.set_title("Front BEV")
    bev_axis.grid(True, linewidth=0.4, alpha=0.4)
    bev_axis.legend(loc="upper right")

    figure.suptitle(
        f"sample={info['token']} | lidar={len(lidar)} radar={len(radar)} "
        f"GT={len(boxes)}")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    with args.ann_file.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    indices = select_indices(len(infos), args.indices, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in indices:
        info = infos[index]
        output_path = args.output_dir / f"{index:04d}_{info['token']}.png"
        render_sample(
            info, output_path, args.camera_key, args.radar_key,
            args.max_lidar_points, args.max_radar_points)
        print(f"[{index + 1}/{len(infos)}] {output_path}")

    print(f"Wrote {len(indices)} visualizations to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
