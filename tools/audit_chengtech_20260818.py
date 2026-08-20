#!/usr/bin/env python3
"""Render camera projection and front-BEV audits for converted company data."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-lidar-points", type=int, default=50000)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_info(root, frame_index):
    suffix = "-{:06d}".format(frame_index)
    for split in ("train", "val", "test"):
        path = root / "custom_infos_{}_sweep.pkl".format(split)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        for info in payload["infos"]:
            if str(info["token"]).endswith(suffix):
                return split, info
    raise KeyError("frame {} is absent from all splits".format(frame_index))


def project_lidar(image, points, projection, maximum):
    import cv2

    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite]
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum).astype(np.int64)
        points = points[indices]
    homogeneous = np.column_stack([points[:, :3], np.ones(len(points))])
    camera = homogeneous @ projection.T
    depth = camera[:, 2]
    valid = depth > 0.5
    uv = camera[valid, :2] / depth[valid, None]
    depth = depth[valid]
    height, width = image.shape[:2]
    valid = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
             (uv[:, 1] >= 0) & (uv[:, 1] < height))
    uv, depth = uv[valid], depth[valid]
    # Draw far points first so a near point wins when pixels overlap.  Direct
    # NumPy indexing is much faster than tens of thousands of cv2.circle calls.
    order = np.argsort(depth)[::-1]
    pixels = np.rint(uv[order]).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    color_values = np.rint(
        255.0 * (1.0 - np.clip(depth[order] / 350.0, 0.0, 1.0))
    ).astype(np.uint8)
    colors = cv2.applyColorMap(
        color_values.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    image[pixels[:, 1], pixels[:, 0]] = colors
    return image, len(uv)


def bev_xy(points, scale, margin, height):
    x = points[:, 0]
    y = points[:, 1]
    pixels_x = margin + np.rint(x * scale).astype(np.int32)
    pixels_y = height - margin - np.rint((y + 20.0) * scale).astype(np.int32)
    return np.column_stack([pixels_x, pixels_y])


def box_corners(box):
    x, y, _, length, width, _, yaw = box
    local = np.asarray([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    return local @ rotation.T + np.asarray([x, y])


def render_bev(lidar, radar, boxes, names, sources):
    import cv2

    scale, margin = 3.0, 20
    width = int(350 * scale) + 2 * margin
    height = int(40 * scale) + 2 * margin
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    roi_lidar = lidar[
        (lidar[:, 0] >= 0) & (lidar[:, 0] <= 350) &
        (lidar[:, 1] >= -20) & (lidar[:, 1] <= 20)]
    if len(roi_lidar) > 100000:
        roi_lidar = roi_lidar[np.linspace(
            0, len(roi_lidar) - 1, 100000).astype(np.int64)]
    pixels = bev_xy(roi_lidar, scale, margin, height)
    canvas[pixels[:, 1], pixels[:, 0]] = (170, 170, 170)

    roi_radar = radar[
        (radar[:, 0] >= 0) & (radar[:, 0] <= 350) &
        (radar[:, 1] >= -20) & (radar[:, 1] <= 20)]
    for pixel in bev_xy(roi_radar, scale, margin, height):
        cv2.circle(canvas, tuple(pixel), 1, (255, 80, 0), -1)

    for box, name, source in zip(boxes, names, sources):
        if not (0 <= box[0] <= 350 and -20 <= box[1] <= 20):
            continue
        corners = bev_xy(box_corners(box), scale, margin, height)
        color = (0, 0, 220) if int(source) == 1 else (0, 150, 0)
        cv2.polylines(canvas, [corners], True, color, 2)
        cv2.putText(canvas, "{}:s{}".format(name, source),
                    tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA)
    cv2.line(canvas, (margin, margin), (margin, height - margin),
             (0, 0, 0), 1)
    return canvas, len(roi_lidar), len(roi_radar)


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for audit rendering") from error
    root = args.processed_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split, info = load_info(root, args.frame_index)
    lidar = np.load(resolve(root, info["lidar_path"]))
    radar = np.load(resolve(
        root, info["rads"]["RADAR_FRONT"]["data_path"]))
    image = cv2.imread(str(resolve(
        root, info["cams"]["CAM_FRONT"]["data_path"])), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("failed to read converted image")
    projection = np.asarray(
        info["cams"]["CAM_FRONT"]["lidar2img"], dtype=np.float64)
    projected, projected_count = project_lidar(
        image.copy(), lidar, projection, args.max_lidar_points)
    bev, lidar_roi_count, radar_roi_count = render_bev(
        lidar, radar, np.asarray(info["gt_boxes"]),
        np.asarray(info["gt_names"]), np.asarray(info["gt_sources"]))
    camera_path = args.out_dir / "frame_{:06d}_camera.jpg".format(
        args.frame_index)
    bev_path = args.out_dir / "frame_{:06d}_bev.jpg".format(
        args.frame_index)
    cv2.imwrite(str(camera_path), projected)
    cv2.imwrite(str(bev_path), bev)
    print("split: {}".format(split))
    print("token: {}".format(info["token"]))
    print("projected lidar points: {}".format(projected_count))
    print("front ROI lidar points: {}".format(lidar_roi_count))
    print("front ROI radar points: {}".format(radar_roi_count))
    print("GT boxes: {}".format(len(info["gt_boxes"])))
    print("camera audit: {}".format(camera_path.resolve()))
    print("BEV audit: {}".format(bev_path.resolve()))


if __name__ == "__main__":
    main()
