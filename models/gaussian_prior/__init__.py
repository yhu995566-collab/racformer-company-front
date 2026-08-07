"""Building blocks for the 3DH-Query radar-guided Gaussian prior."""

from .candidate_scorer import (
    RadarCandidateScorer,
    build_candidate_targets,
    candidate_scoring_loss,
)

__all__ = [
    "RadarCandidateScorer",
    "build_candidate_targets",
    "candidate_scoring_loss",
]
