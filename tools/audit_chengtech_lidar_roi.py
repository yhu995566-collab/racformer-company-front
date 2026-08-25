#!/usr/bin/env python3
"""Compare global- and ego-frame interpretations of ChengTech LiDAR PCDs."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_collection_converter():
    path = Path(__file__).with_name("convert_chengtech_20260818_collection.py")
    spec = importlib.util.spec_from_file_location(
        "convert_chengtech_20260818_collection_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--point-cloud-range", type=float, nargs=6,
        default=(0.0, -20.0, -3.0, 50.0, 20.0, 3.0))
    return parser.parse_args()


def main():
    args = parse_args()
    converter = load_collection_converter()
    frames, scenario, alignment = converter.discover_sequence(
        args.data_root.resolve(), args.truth_root.resolve(), args.sequence)
    print("sequence:", args.sequence)
    print("scenario:", scenario)
    print("alignment:", alignment)
    print("frames:", len(frames))
    print("index source_xyz_center pose_xyz global_roi ego_roi")
    for index in args.indices:
        if index < 0 or index >= len(frames):
            raise ValueError(
                "index {} outside [0, {})".format(index, len(frames)))
        frame = frames[index]
        fields = converter.single.read_binary_compressed_pcd(frame.lidar_path)
        source = np.column_stack(
            [fields["x"], fields["y"], fields["z"]]).astype(np.float32)
        finite = np.isfinite(source).all(axis=1)
        source = source[finite]
        homogeneous = np.column_stack([
            source, np.ones(len(source), dtype=np.float32)])
        global_to_ego = np.linalg.inv(frame.ego2global)
        transformed = (homogeneous @ global_to_ego.T)[:, :3]
        global_roi = int(np.count_nonzero(converter.roi_mask(
            transformed, args.point_cloud_range)))
        ego_roi = int(np.count_nonzero(converter.roi_mask(
            source, args.point_cloud_range)))
        center = np.median(source, axis=0)
        pose = frame.ego2global[:3, 3]
        print(
            "{} {} {} {} {}".format(
                index,
                np.round(center, 3).tolist(),
                np.round(pose, 3).tolist(),
                global_roi,
                ego_roi))


if __name__ == "__main__":
    main()
