import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "gaussian_prior"
    / "candidate_scorer.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_scorer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RadarCandidateScorer = MODULE.RadarCandidateScorer
build_candidate_targets = MODULE.build_candidate_targets
build_box_candidate_targets = MODULE.build_box_candidate_targets
candidate_center_offset_loss = MODULE.candidate_center_offset_loss
candidate_scoring_loss = MODULE.candidate_scoring_loss


def make_scorer(topk=3):
    torch.manual_seed(7)
    return RadarCandidateScorer(
        point_cloud_range=[0.0, -50.0, -3.0, 350.0, 50.0, 3.0],
        topk=topk,
        hidden_dim=16,
        embedding_dim=8,
        max_abs_speed=100.0,
    )


def test_coarse_filter_retains_stationary_and_rejects_only_invalid_points():
    scorer = make_scorer()
    points = torch.tensor(
        [
            [250.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],  # stationary: valid
            [351.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0],  # outside ROI
            [200.0, 0.0, 0.0, 5.0, 101.0, 0.0, 0.0],  # abnormal speed
            [200.0, float("nan"), 0.0, 5.0, 1.0, 0.0, 0.0],
        ]
    )
    output = scorer([points])
    assert output["candidate_mask"][0, :4].tolist() == [True, False, False, False]
    assert output["topk_mask"].sum().item() == 1
    assert torch.isfinite(output["candidate_embeddings"]).all()


def test_topk_order_is_exactly_the_learned_score_order():
    scorer = make_scorer(topk=2)
    points = torch.tensor(
        [
            [100.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0],
            [200.0, 1.0, 0.0, 40.0, 0.0, 0.0, 0.0],
            [300.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    output = scorer([points])
    valid_logits = output["objectness_logits"][0, :3]
    expected = torch.topk(valid_logits, 2).indices
    assert torch.equal(output["topk_indices"][0], expected)


def test_variable_length_batch_always_returns_fixed_topk_shape():
    scorer = make_scorer(topk=4)
    first = torch.zeros((0, 7))
    second = torch.tensor([[250.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    output = scorer([first, second])
    assert output["topk_points"].shape == (2, 4, 7)
    assert output["topk_embeddings"].shape == (2, 4, 8)
    assert output["topk_center_offsets"].shape == (2, 4, 2)
    assert output["topk_mask"].sum(dim=1).tolist() == [0, 1]


def test_targets_and_loss_train_the_candidate_score_head():
    scorer = make_scorer(topk=2)
    points = torch.tensor(
        [
            [100.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [110.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        ]
    )
    output = scorer([points])
    targets = build_candidate_targets(
        output["points"], output["candidate_mask"],
        [torch.tensor([[102.0, 0.0, 0.0]])], target_sigma=4.0,
    )
    assert targets["objectness_targets"][0, 0] > targets["objectness_targets"][0, 1]
    assert torch.allclose(
        targets["center_offsets"][0, 0], torch.tensor([2.0, 0.0])
    )
    loss = candidate_scoring_loss(
        output["objectness_logits"], targets["objectness_targets"],
        output["candidate_mask"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert scorer.score_head.weight.grad is not None
    assert scorer.score_head.weight.grad.abs().sum() > 0


def test_oriented_box_targets_accept_surface_returns_and_regress_center():
    points = torch.tensor([[[10.0, 2.9, 0.0, 5.0, 0.0, 0.0, 0.0],
                            [10.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    # A 6m x 2m box rotated 90 degrees has its long side along global Y.
    boxes = [torch.tensor([[10.0, 0.0, 0.0, 6.0, 2.0, 2.0, 1.5707963]])]
    targets = build_box_candidate_targets(points, mask, boxes, target_sigma=1.0)
    assert torch.isclose(targets["objectness_targets"][0, 0], torch.tensor(1.0))
    assert targets["objectness_targets"][0, 1] < 1.0
    assert torch.allclose(
        targets["center_offsets"][0, 0], torch.tensor([0.0, -2.9]), atol=1e-5
    )


def test_center_offset_loss_trains_offset_head():
    scorer = make_scorer(topk=2)
    points = torch.tensor([[10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    output = scorer([points])
    target_offsets = torch.zeros_like(output["center_offsets"])
    target_offsets[0, 0] = torch.tensor([2.0, 1.0])
    objectness = torch.zeros_like(output["objectness_logits"])
    objectness[0, 0] = 1.0
    loss = candidate_center_offset_loss(
        output["center_offsets"], target_offsets, objectness,
        output["candidate_mask"])
    loss.backward()
    assert scorer.offset_head.weight.grad is not None
    assert scorer.offset_head.weight.grad.abs().sum() > 0
