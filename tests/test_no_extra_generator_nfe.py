"""Tests that CRH-CFG matches ordinary CFG generator forward counts."""

import torch
import torch.nn as nn

from src.crh_cfg import CRHCFGConfig, CRHCFGController
from src.sampling_hw4 import sample_hw4


class CountingVelocity(nn.Module):
    """Return condition-dependent constant velocities with a discoverable device."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        embedded_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a constant field derived from condition ids."""
        value = condition.float().sum(dim=1).view(-1, 1, 1, 1)
        return torch.zeros_like(state) + value + self.anchor * 0


def simple_evaluator(images: torch.Tensor) -> torch.Tensor:
    """Return two deterministic logits for controller-count tests."""
    mean = images.mean(dim=(1, 2, 3))
    return torch.stack([mean, -mean], dim=1)


def test_crh_cfg_adds_no_generator_forwards() -> None:
    """Feedback evaluator work must not change generator NFEs relative to static CFG."""
    model = CountingVelocity()
    condition = torch.tensor([[2, 0, 0], [2, 0, 0]])
    seeds = [11, 12]
    baseline = sample_hw4(
        model,
        condition,
        seeds,
        method="hw3_baseline",
        num_steps=3,
        sampler="heun",
        image_shape=(1, 2, 2),
    )
    controller = CRHCFGController(CRHCFGConfig(
        candidate_scales=(0.5, 1.0, 2.0),
        target_index=0,
        target_sign=1,
        protected_indices=(1,),
        reliability_gate=False,
    ))
    feedback = sample_hw4(
        model,
        condition,
        seeds,
        method="crh_cfg",
        num_steps=3,
        sampler="heun",
        image_shape=(1, 2, 2),
        controller=controller,
        evaluator=simple_evaluator,
    )

    assert baseline.counters.raw_generator_forwards == 12
    assert feedback.counters.raw_generator_forwards == baseline.counters.raw_generator_forwards
    assert feedback.counters.guided_field_evaluations == baseline.counters.guided_field_evaluations
    assert feedback.counters.evaluator_forwards == 3
