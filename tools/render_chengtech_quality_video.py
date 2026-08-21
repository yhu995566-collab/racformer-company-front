#!/usr/bin/env python3
"""Render a side-by-side human quality-control video for company data.

The left panel contains only the undistorted camera image and projected 3D GT
boxes.  The right panel contains the current-frame LiDAR, radar, and GT boxes
in a metric bird's-eye view.  Point colors are fixed (never distance-coded),
and GT colors encode annotation source 0 versus source 1.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


SOURCE_COLORS = {
    0: (0, 220, 0),       # green: image-generated GT
    1: (0, 0, 255),       # red: LiDAR-generated GT
}
UNKNOWN_SOURCE_COLOR = (255, 0, 255)
LIDAR_COLOR = (175, 175, 175)
RADAR_COLOR = (0, 200, 255)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=0,
                        help="First chronological frame index")
    parser.add_argument("--stop", type=int,
                        help="Exclusive chronological stop index")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=350.0)
    parser.add_argument("--y-min", type=float, default=-100.0)
    parser.add_argument("--y-max", type=float, default=100.0)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC codec, for example mp4v or avc1")
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_infos(root):
    by_token = {}
    for split in ("train", "val", "test"):
        path = root / "custom_infos_{}_sweep.pkl".format(split)
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        for info in payload["infos"]:
            by_token[str(info["token"])] = info
    if not by_token:
        raise FileNotFoundError("no custom_infos_*_sweep.pkl under {}".format(root))
    return sorted(by_token.values(), key=lambda item: int(item["timestamp"]))


def box_corners_3d(box):
    x, y, z, length, width, height, yaw = np.asarray(box, dtype=np.float64)
    local = np.asarray([
        [length / 2, width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [length / 2, width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [-length / 2, -width / 2, height / 2],
        [-length / 2, width / 2, height / 2],
    ], dtype=np.float64)
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    local[:, :2] = local[:, :2] @ rotation.T
    local += np.asarray([x, y, z])
    return local


def _clip_camera_segment(left, right, near=0.1):
    left = left.copy()
    right = right.copy()
    if left[2] <= near and right[2] <= near:
        return None
    if left[2] <= near:
        ratio = (near - left[2]) / (right[2] - left[2])
        left += ratio * (right - left)
    elif right[2] <= near:
        ratio = (near - right[2]) / (left[2] - right[2])
        right += ratio * (left - right)
    return left, right


def draw_projected_boxes(image, boxes, names, sources, projection):
    import cv2

    height, width = image.shape[:2]
    drawn = 0
    for box, name, source in zip(boxes, names, sources):
        color = SOURCE_COLORS.get(int(source), UNKNOWN_SOURCE_COLOR)
        corners = box_corners_3d(box)
        homogeneous = np.column_stack([corners, np.ones(8)])
        camera = homogeneous @ np.asarray(projection, dtype=np.float64).T
        visible_pixels = []
        edge_drawn = False
        for first, second in BOX_EDGES:
            clipped = _clip_camera_segment(camera[first], camera[second])
            if clipped is None:
                continue
            left, right = clipped
            uv_left = left[:2] / left[2]
            uv_right = right[:2] / right[2]
            if not np.isfinite([uv_left, uv_right]).all():
                continue
            endpoints = np.clip(
                np.rint([uv_left, uv_right]), -1000000, 1000000
            ).astype(np.int32)
            accepted, point_left, point_right = cv2.clipLine(
                (0, 0, width, height), tuple(endpoints[0]), tuple(endpoints[1]))
            if accepted:
                cv2.line(image, point_left, point_right, color, 2, cv2.LINE_AA)
                visible_pixels.extend((point_left, point_right))
                edge_drawn = True
        if edge_drawn:
            drawn += 1
            anchor = min(visible_pixels, key=lambda point: point[1])
            cv2.putText(
                image, "{} s{}".format(name, int(source)),
                (anchor[0], max(18, anchor[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image, drawn


class BevProjector:
    def __init__(self, width, height, x_min, x_max, y_min, y_max,
                 margin=24, header=90):
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("invalid BEV bounds")
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.margin = margin
        self.header = header
        usable_width = width - 2 * margin
        usable_height = height - header - margin
        self.scale = min(
            usable_width / (y_max - y_min),
            usable_height / (x_max - x_min))
        metric_width = (y_max - y_min) * self.scale
        self.left = (width - metric_width) / 2.0
        self.bottom = height - margin

    def mask(self, points):
        return (
            np.isfinite(points[:, :3]).all(axis=1) &
            (points[:, 0] >= self.x_min) &
            (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) &
            (points[:, 1] <= self.y_max))

    def pixels(self, xy):
        # Vehicle +Y is shown to the left; vehicle +X points upward.
        pixel_x = self.left + (self.y_max - xy[:, 1]) * self.scale
        pixel_y = self.bottom - (xy[:, 0] - self.x_min) * self.scale
        pixels = np.rint(np.column_stack([pixel_x, pixel_y])).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, self.width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, self.height - 1)
        return pixels


def render_bev(projector, lidar, radar, boxes, names, sources, token):
    import cv2

    canvas = np.full(
        (projector.height, projector.width, 3), 18, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "BEV: LiDAR + Radar + GT", (18, 28),
                font, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, str(token), (18, 54),
                font, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT s0", (18, 78), font, 0.5,
                SOURCE_COLORS[0], 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT s1", (100, 78), font, 0.5,
                SOURCE_COLORS[1], 2, cv2.LINE_AA)
    cv2.putText(canvas, "LiDAR", (182, 78), font, 0.5,
                LIDAR_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Radar", (260, 78), font, 0.5,
                RADAR_COLOR, 2, cv2.LINE_AA)

    for distance in np.arange(
            np.ceil(projector.x_min / 50.0) * 50.0,
            projector.x_max + 1.0, 50.0):
        endpoints = projector.pixels(np.asarray([
            [distance, projector.y_min], [distance, projector.y_max]]))
        cv2.line(canvas, tuple(endpoints[0]), tuple(endpoints[1]),
                 (45, 45, 45), 1)
        cv2.putText(canvas, "{}m".format(int(distance)),
                    tuple(endpoints[0] + np.asarray([4, -3])),
                    font, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

    lidar_roi = lidar[projector.mask(lidar)]
    lidar_pixels = projector.pixels(lidar_roi[:, :2])
    canvas[lidar_pixels[:, 1], lidar_pixels[:, 0]] = LIDAR_COLOR

    radar_roi = radar[projector.mask(radar)]
    radar_pixels = projector.pixels(radar_roi[:, :2])
    canvas[radar_pixels[:, 1], radar_pixels[:, 0]] = RADAR_COLOR
    if len(radar_pixels):
        radar_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        radar_mask[radar_pixels[:, 1], radar_pixels[:, 0]] = 255
        radar_mask = cv2.dilate(radar_mask, np.ones((3, 3), dtype=np.uint8))
        canvas[radar_mask > 0] = RADAR_COLOR

    for box, name, source in zip(boxes, names, sources):
        center = np.asarray(box[:2], dtype=np.float64)
        if not (projector.x_min <= center[0] <= projector.x_max and
                projector.y_min <= center[1] <= projector.y_max):
            continue
        corners = box_corners_3d(box)[:4, :2]
        pixels = projector.pixels(corners)
        color = SOURCE_COLORS.get(int(source), UNKNOWN_SOURCE_COLOR)
        cv2.polylines(canvas, [pixels], True, color, 2, cv2.LINE_AA)
        anchor = tuple(pixels[0])
        cv2.putText(canvas, "{} s{}".format(name, int(source)), anchor,
                    font, 0.42, color, 1, cv2.LINE_AA)

    vehicle = projector.pixels(np.asarray([[0.0, 0.0]]))[0]
    cv2.drawMarker(canvas, tuple(vehicle), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 14, 2)
    return canvas, len(lidar_roi), len(radar_roi)


def fit_panel(image, width, height):
    import cv2

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (int(round(image.shape[1] * scale)),
            int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    left = (width - size[0]) // 2
    top = (height - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    return canvas


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for video rendering") from error
    if args.step <= 0 or args.fps <= 0:
        raise ValueError("--step and --fps must be positive")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    root = args.processed_root.resolve()
    infos = load_infos(root)
    stop = len(infos) if args.stop is None else min(args.stop, len(infos))
    selected = infos[args.start:stop:args.step]
    if not selected:
        raise ValueError("selected frame range is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_size = (args.camera_width + args.bev_width, args.output_height)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*args.codec),
        args.fps, output_size)
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer for {}".format(args.output))
    projector = BevProjector(
        args.bev_width, args.output_height,
        args.x_min, args.x_max, args.y_min, args.y_max)
    preview_path = args.output.with_suffix(".preview.jpg")

    try:
        for output_index, info in enumerate(tqdm(selected, desc="Rendering QC video")):
            cam = info["cams"]["CAM_FRONT"]
            image = cv2.imread(
                str(resolve(root, cam["data_path"])), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError("failed to read {}".format(cam["data_path"]))
            lidar = np.load(resolve(root, info["lidar_path"]), mmap_mode="r")
            radar = np.load(resolve(
                root, info["rads"]["RADAR_FRONT"]["data_path"]),
                mmap_mode="r")
            boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 7)
            names = np.asarray(info["gt_names"], dtype=object)
            sources = np.asarray(
                info.get("gt_sources", np.full(len(boxes), -1)), dtype=np.int8)

            camera_panel, projected_boxes = draw_projected_boxes(
                image.copy(), boxes, names, sources, cam["lidar2img"])
            camera_panel = fit_panel(
                camera_panel, args.camera_width, args.output_height)
            cv2.putText(
                camera_panel, "Camera + projected GT only", (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                2, cv2.LINE_AA)
            cv2.putText(
                camera_panel,
                "frame {}/{} | projected GT {}/{}".format(
                    args.start + output_index * args.step,
                    len(infos) - 1, projected_boxes, len(boxes)),
                (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

            bev_panel, lidar_count, radar_count = render_bev(
                projector, lidar, radar, boxes, names, sources, info["token"])
            cv2.putText(
                bev_panel, "points: L{} R{} | GT {}".format(
                    lidar_count, radar_count, len(boxes)),
                (350, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)
            composite = np.concatenate([camera_panel, bev_panel], axis=1)
            writer.write(composite)
            if output_index == 0:
                cv2.imwrite(str(preview_path), composite)
    finally:
        writer.release()

    print("frames: {}".format(len(selected)))
    print("fps: {}".format(args.fps))
    print("video: {}".format(args.output.resolve()))
    print("first-frame preview: {}".format(preview_path.resolve()))


if __name__ == "__main__":
    main()
