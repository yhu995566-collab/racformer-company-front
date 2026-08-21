"""Learned radar-candidate scoring and top-K selection.

This module is deliberately independent from MMDetection3D so it can be
tested before the complete company dataset arrives.  Hand-written rules only
construct a validity mask.  The final top-K ordering always comes from the
MLP's learned objectness logits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PointBatch = Union[Tensor, Sequence[Tensor]]


class RadarCandidateScorer(nn.Module):
    """Score variable-length company radar points and select learned top-K.

    Expected point fields are ``[x, y, z, rcs, vx, vy, time_lag]`` by default.
    The input can be a padded ``[B, N, D]`` tensor or a list of ``[Ni, D]``
    tensors.  A list is convenient for RaCFormer's existing radar pipeline.

    Args:
        point_cloud_range: ``[xmin, ymin, zmin, xmax, ymax, zmax]`` in ego.
        topk: Number of learned candidates returned for every batch item.
        hidden_dim: Width of the point MLP.
        embedding_dim: Feature width retained for future Gaussian heads.
        max_abs_speed: Only reject clearly implausible velocity magnitudes.
            Stationary points are explicitly retained.
        min_range: Reject sensor-near artifacts below this radial distance.
        field_indices: Indices for x, y, z, rcs, vx, vy, and time_lag.
    """

    DEFAULT_FIELDS = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rcs": 3,
        "vx": 4,
        "vy": 5,
        "time_lag": 6,
    }

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        topk: int = 256,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        max_abs_speed: float = 100.0,
        min_range: float = 0.5,
        rcs_scale: float = 64.0,
        velocity_scale: float = 50.0,
        time_scale: float = 1.0,
        field_indices: Optional[Dict[str, int]] = None,
        dropout: float = 0.0,
        max_center_offset: float = 20.0,
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError("point_cloud_range must contain 6 values")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("MLP dimensions must be positive")
        if min(rcs_scale, velocity_scale, time_scale) <= 0:
            raise ValueError("feature scales must be positive")
        if max_center_offset <= 0:
            raise ValueError("max_center_offset must be positive")

        fields = dict(self.DEFAULT_FIELDS)
        if field_indices is not None:
            fields.update(field_indices)
        missing = set(self.DEFAULT_FIELDS) - set(fields)
        if missing:
            raise ValueError("missing radar fields: {}".format(sorted(missing)))

        self.topk = int(topk)
        self.max_abs_speed = float(max_abs_speed)
        self.min_range = float(min_range)
        self.rcs_scale = float(rcs_scale)
        self.velocity_scale = float(velocity_scale)
        self.time_scale = float(time_scale)
        self.max_center_offset = float(max_center_offset)
        self.field_indices = fields
        self.required_dim = max(fields.values()) + 1

        pc_range = torch.as_tensor(point_cloud_range, dtype=torch.float32)
        if torch.any(pc_range[3:] <= pc_range[:3]):
            raise ValueError("point_cloud_range maxima must exceed minima")
        self.register_buffer("point_cloud_range", pc_range)

        # Ten normalized features: xyz, rcs, vx/vy, time, radial range,
        # sin(azimuth), and cos(azimuth).
        self.feature_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(embedding_dim, 1)
        self.offset_head = nn.Linear(embedding_dim, 2)
        # Start with a conservative ~0.1 objectness probability.
        nn.init.constant_(self.score_head.bias, -2.1972246)
        # A fresh scorer starts at the measured radar position.  The bounded
        # residual is learned only where GT association supplies supervision.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def _pad_points(
        self, points: PointBatch, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return padded points, validity mask, and original point counts."""
        if torch.is_tensor(points):
            if points.dim() != 3:
                raise ValueError("tensor points must have shape [B, N, D]")
            batch_points = points
            if valid_mask is None:
                mask = torch.ones(
                    points.shape[:2], dtype=torch.bool, device=points.device
                )
            else:
                if valid_mask.shape != points.shape[:2]:
                    raise ValueError("valid_mask must have shape [B, N]")
                mask = valid_mask.to(device=points.device, dtype=torch.bool)
            counts = mask.sum(dim=1)
        else:
            point_list = list(points)
            if not point_list:
                raise ValueError("points list must contain at least one batch item")
            if valid_mask is not None:
                raise ValueError("valid_mask is only supported with tensor input")
            feature_dim = point_list[0].shape[-1]
            if any(item.dim() != 2 for item in point_list):
                raise ValueError("list items must have shape [Ni, D]")
            if any(item.shape[-1] != feature_dim for item in point_list):
                raise ValueError("all list items must use the same point dimension")
            device, dtype = point_list[0].device, point_list[0].dtype
            if any(item.device != device or item.dtype != dtype for item in point_list):
                raise ValueError("all list items must share device and dtype")
            max_points = max(max(item.shape[0] for item in point_list), self.topk)
            batch_points = torch.zeros(
                (len(point_list), max_points, feature_dim),
                device=device,
                dtype=dtype,
            )
            mask = torch.zeros(
                (len(point_list), max_points), device=device, dtype=torch.bool
            )
            for batch_index, item in enumerate(point_list):
                count = item.shape[0]
                batch_points[batch_index, :count] = item
                mask[batch_index, :count] = True
            counts = torch.as_tensor(
                [item.shape[0] for item in point_list], device=device, dtype=torch.long
            )

        if batch_points.shape[-1] < self.required_dim:
            raise ValueError(
                "radar point dimension {} is smaller than required {}".format(
                    batch_points.shape[-1], self.required_dim
                )
            )
        # Tensor inputs can also contain N < K. Pad here to keep a stable K.
        if batch_points.shape[1] < self.topk:
            amount = self.topk - batch_points.shape[1]
            batch_points = F.pad(batch_points, (0, 0, 0, amount))
            mask = F.pad(mask, (0, amount), value=False)
        return batch_points, mask, counts

    def coarse_filter(self, points: Tensor, input_mask: Tensor) -> Tensor:
        """Build a non-learned validity mask without ranking candidates."""
        f = self.field_indices
        required = [f[name] for name in self.DEFAULT_FIELDS]
        selected = points[..., required]
        finite = torch.isfinite(selected).all(dim=-1)
        safe = torch.where(torch.isfinite(points), points, torch.zeros_like(points))

        x, y, z = safe[..., f["x"]], safe[..., f["y"]], safe[..., f["z"]]
        vx, vy = safe[..., f["vx"]], safe[..., f["vy"]]
        pc = self.point_cloud_range.to(device=points.device, dtype=points.dtype)
        in_roi = (
            (x >= pc[0]) & (x <= pc[3])
            & (y >= pc[1]) & (y <= pc[4])
            & (z >= pc[2]) & (z <= pc[5])
        )
        radial_range = torch.sqrt(x.square() + y.square())
        speed = torch.sqrt(vx.square() + vy.square())
        return (
            input_mask
            & finite
            & in_roi
            & (radial_range >= self.min_range)
            & (speed <= self.max_abs_speed)
        )

    def build_features(self, points: Tensor) -> Tensor:
        """Construct normalized geometry and radar attributes for the MLP."""
        f = self.field_indices
        safe = torch.where(torch.isfinite(points), points, torch.zeros_like(points))
        pc = self.point_cloud_range.to(device=points.device, dtype=points.dtype)
        xyz = torch.stack(
            (safe[..., f["x"]], safe[..., f["y"]], safe[..., f["z"]]), dim=-1
        )
        xyz_normalized = 2.0 * (xyz - pc[:3]) / (pc[3:] - pc[:3]) - 1.0
        x, y = xyz[..., 0], xyz[..., 1]
        radial_range = torch.sqrt(x.square() + y.square())
        radius_denom = torch.sqrt(
            torch.maximum(pc[0].abs(), pc[3].abs()).square()
            + torch.maximum(pc[1].abs(), pc[4].abs()).square()
        ).clamp_min(1.0)
        safe_range = radial_range.clamp_min(1e-6)
        azimuth_sin = y / safe_range
        azimuth_cos = x / safe_range

        attributes = torch.stack(
            (
                (safe[..., f["rcs"]] / self.rcs_scale).clamp(-4.0, 4.0),
                (safe[..., f["vx"]] / self.velocity_scale).clamp(-4.0, 4.0),
                (safe[..., f["vy"]] / self.velocity_scale).clamp(-4.0, 4.0),
                (safe[..., f["time_lag"]] / self.time_scale).clamp(-4.0, 4.0),
                radial_range / radius_denom,
                azimuth_sin,
                azimuth_cos,
            ),
            dim=-1,
        )
        return torch.cat((xyz_normalized, attributes), dim=-1)

    def forward(
        self, points: PointBatch, valid_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        padded_points, input_mask, input_counts = self._pad_points(
            points, valid_mask
        )
        candidate_mask = self.coarse_filter(padded_points, input_mask)
        features = self.build_features(padded_points)
        embeddings = self.feature_encoder(features)
        logits = self.score_head(embeddings).squeeze(-1)
        center_offsets = self.max_center_offset * torch.tanh(
            self.offset_head(embeddings)
        )

        # Invalid candidates cannot enter top-K.  There is deliberately no
        # hand-crafted ranking term here: valid ordering is 100% learned.
        ranking_logits = logits.masked_fill(
            ~candidate_mask, torch.finfo(logits.dtype).min
        )
        _, topk_indices = torch.topk(ranking_logits, self.topk, dim=1)
        topk_mask = torch.gather(candidate_mask, 1, topk_indices)
        topk_scores = torch.gather(logits, 1, topk_indices)

        point_index = topk_indices.unsqueeze(-1).expand(
            -1, -1, padded_points.shape[-1]
        )
        feature_index = topk_indices.unsqueeze(-1).expand(
            -1, -1, embeddings.shape[-1]
        )
        topk_points = torch.gather(padded_points, 1, point_index)
        topk_embeddings = torch.gather(embeddings, 1, feature_index)
        offset_index = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        topk_center_offsets = torch.gather(center_offsets, 1, offset_index)
        topk_points = topk_points * topk_mask.unsqueeze(-1).to(topk_points.dtype)
        topk_embeddings = topk_embeddings * topk_mask.unsqueeze(-1).to(
            topk_embeddings.dtype
        )
        topk_center_offsets = topk_center_offsets * topk_mask.unsqueeze(-1).to(
            topk_center_offsets.dtype
        )
        topk_scores = topk_scores.masked_fill(~topk_mask, 0.0)

        return {
            "points": padded_points,
            "input_counts": input_counts,
            "candidate_mask": candidate_mask,
            "candidate_embeddings": embeddings,
            "objectness_logits": logits,
            "center_offsets": center_offsets,
            "topk_indices": topk_indices,
            "topk_mask": topk_mask,
            "topk_scores": topk_scores,
            "topk_points": topk_points,
            "topk_embeddings": topk_embeddings,
            "topk_center_offsets": topk_center_offsets,
        }


@torch.no_grad()
def build_box_candidate_targets(
    candidate_points: Tensor,
    candidate_mask: Tensor,
    gt_boxes: Sequence[Tensor],
    target_sigma: float = 2.0,
) -> Dict[str, Tensor]:
    """Associate radar points with oriented BEV boxes.

    Company boxes use ``[x, y, z, length, width, height, yaw]`` with yaw
    measured from vehicle +X toward +Y.  Objectness is one inside a box and
    decays with Euclidean distance to its oriented rectangle.  Centre-offset
    supervision still points to the geometric box centre, allowing returns on
    a vehicle surface to become useful Gaussian centres.
    """
    if target_sigma <= 0:
        raise ValueError("target_sigma must be positive")
    if candidate_points.dim() != 3 or candidate_mask.shape != candidate_points.shape[:2]:
        raise ValueError("candidate tensors must have shapes [B,N,D] and [B,N]")
    if len(gt_boxes) != candidate_points.shape[0]:
        raise ValueError("gt_boxes length must equal batch size")

    batch_size, num_candidates = candidate_mask.shape
    device, dtype = candidate_points.device, candidate_points.dtype
    targets = torch.zeros((batch_size, num_candidates), device=device, dtype=dtype)
    offsets = torch.zeros((batch_size, num_candidates, 2), device=device, dtype=dtype)
    distances = torch.full(
        (batch_size, num_candidates), float("inf"), device=device, dtype=dtype
    )
    matched_indices = torch.full(
        (batch_size, num_candidates), -1, device=device, dtype=torch.long
    )

    for batch_index, boxes in enumerate(gt_boxes):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=dtype).reshape(-1, 7)
        candidate_xy = candidate_points[batch_index, :, :2]
        delta = candidate_xy[:, None, :] - boxes[None, :, :2]
        yaw = boxes[:, 6]
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        local_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
        local_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
        outside_x = (local_x.abs() - boxes[:, 3] * 0.5).clamp_min(0.0)
        outside_y = (local_y.abs() - boxes[:, 4] * 0.5).clamp_min(0.0)
        box_distance = torch.sqrt(outside_x.square() + outside_y.square())
        nearest_distance, nearest_index = box_distance.min(dim=1)
        nearest_center = boxes[nearest_index, :2]
        valid = candidate_mask[batch_index]
        distances[batch_index, valid] = nearest_distance[valid]
        matched_indices[batch_index, valid] = nearest_index[valid]
        offsets[batch_index, valid] = nearest_center[valid] - candidate_xy[valid]
        targets[batch_index, valid] = torch.exp(
            -0.5 * (nearest_distance[valid] / target_sigma).square()
        )
    return {
        "objectness_targets": targets,
        "center_offsets": offsets,
        "nearest_box_distances": distances,
        "matched_gt_indices": matched_indices,
    }


def candidate_center_offset_loss(
    predicted_offsets: Tensor,
    target_offsets: Tensor,
    objectness_targets: Tensor,
    candidate_mask: Tensor,
    min_target: float = 0.5,
) -> Tensor:
    """Weighted Smooth-L1 centre regression for target-associated returns."""
    if predicted_offsets.shape != target_offsets.shape:
        raise ValueError("predicted and target offsets must have identical shapes")
    if objectness_targets.shape != candidate_mask.shape:
        raise ValueError("objectness_targets and candidate_mask must match")
    if predicted_offsets.shape[:2] != candidate_mask.shape:
        raise ValueError("offset tensors must begin with [B,N]")
    if not 0 <= min_target <= 1:
        raise ValueError("min_target must be in [0,1]")
    positive = candidate_mask.to(torch.bool) & (objectness_targets >= min_target)
    if not torch.any(positive):
        return predicted_offsets.sum() * 0.0
    loss = F.smooth_l1_loss(
        predicted_offsets[positive], target_offsets[positive], reduction="none"
    ).mean(dim=-1)
    weights = objectness_targets[positive]
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def build_candidate_targets(
    candidate_points: Tensor,
    candidate_mask: Tensor,
    gt_centers: Sequence[Tensor],
    target_sigma: float = 4.0,
) -> Dict[str, Tensor]:
    """Build soft objectness and nearest-centre offset supervision.

    This helper is intentionally dataset-agnostic.  GT centres only need to be
    supplied in the same ego/BEV frame as radar points.
    """
    if target_sigma <= 0:
        raise ValueError("target_sigma must be positive")
    if candidate_points.dim() != 3 or candidate_mask.shape != candidate_points.shape[:2]:
        raise ValueError("candidate tensors must have shapes [B,N,D] and [B,N]")
    if len(gt_centers) != candidate_points.shape[0]:
        raise ValueError("gt_centers length must equal batch size")

    batch_size, num_candidates = candidate_mask.shape
    device, dtype = candidate_points.device, candidate_points.dtype
    targets = torch.zeros((batch_size, num_candidates), device=device, dtype=dtype)
    offsets = torch.zeros((batch_size, num_candidates, 2), device=device, dtype=dtype)
    distances = torch.full(
        (batch_size, num_candidates), float("inf"), device=device, dtype=dtype
    )
    matched_indices = torch.full(
        (batch_size, num_candidates), -1, device=device, dtype=torch.long
    )

    for batch_index, centers in enumerate(gt_centers):
        if centers.numel() == 0:
            continue
        centers = centers.to(device=device, dtype=dtype)[..., :2]
        candidate_xy = candidate_points[batch_index, :, :2]
        pairwise = torch.cdist(candidate_xy.unsqueeze(0), centers.unsqueeze(0))[0]
        nearest_distance, nearest_index = pairwise.min(dim=1)
        nearest_center = centers[nearest_index]
        valid = candidate_mask[batch_index]
        distances[batch_index, valid] = nearest_distance[valid]
        matched_indices[batch_index, valid] = nearest_index[valid]
        offsets[batch_index, valid] = nearest_center[valid] - candidate_xy[valid]
        targets[batch_index, valid] = torch.exp(
            -0.5 * (nearest_distance[valid] / target_sigma).square()
        )
    return {
        "objectness_targets": targets,
        "center_offsets": offsets,
        "nearest_distances": distances,
        "matched_gt_indices": matched_indices,
    }


def candidate_scoring_loss(
    objectness_logits: Tensor,
    objectness_targets: Tensor,
    candidate_mask: Tensor,
    positive_weight: float = 4.0,
) -> Tensor:
    """Masked soft-label BCE used to train all coarse-valid candidates."""
    if objectness_logits.shape != objectness_targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if candidate_mask.shape != objectness_logits.shape:
        raise ValueError("candidate_mask must match logits")
    if positive_weight < 1.0:
        raise ValueError("positive_weight must be >= 1")
    valid = candidate_mask.to(torch.bool)
    if not torch.any(valid):
        return objectness_logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        objectness_logits, objectness_targets, reduction="none"
    )
    weights = 1.0 + (positive_weight - 1.0) * objectness_targets
    return (losses[valid] * weights[valid]).mean()
