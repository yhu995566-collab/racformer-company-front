"""Typed contracts shared by offline, ROS, and PyTorch deployment paths."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch


@dataclass(frozen=True)
class FrameInput:
    """One synchronized left-camera and front-radar observation.

    The image must be uint8 BGR. Radar points must be float32 in ego/LiDAR
    coordinates with columns x, y, z, RCS, vx, vy, time_lag. The preprocessor
    overwrites time_lag using radar_timestamp and the newest buffered frame.
    """

    image: np.ndarray
    radar_points: np.ndarray
    image_timestamp: float
    radar_timestamp: float
    lidar2img: np.ndarray
    intrinsic: np.ndarray
    frame_id: str = ''


@dataclass
class PreparedBatch:
    """Batch-size-one, eight-frame inputs before or after GPU transfer."""

    image: torch.Tensor
    radar_points: List[torch.Tensor]
    radar_depth: torch.Tensor
    radar_rcs: torch.Tensor
    img_meta: Dict[str, Any]

    def to(self, device, non_blocking=False):
        return PreparedBatch(
            image=self.image.to(device=device, non_blocking=non_blocking),
            radar_points=[
                points.to(device=device, non_blocking=non_blocking)
                for points in self.radar_points
            ],
            radar_depth=self.radar_depth.to(
                device=device, non_blocking=non_blocking),
            radar_rcs=self.radar_rcs.to(
                device=device, non_blocking=non_blocking),
            img_meta=self.img_meta)


@dataclass(frozen=True)
class DetectionResult:
    """Device-independent RaCFormer prediction arrays."""

    boxes_3d: np.ndarray
    scores_3d: np.ndarray
    labels_3d: np.ndarray

    @property
    def count(self):
        return int(self.boxes_3d.shape[0])
