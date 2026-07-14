"""Tests for CRH-CFG infeasible-candidate fallback behavior."""

import torch

from src.crh_cfg import CRHCFGConfig, CRHCFGController


def test_controller_always_returns_candidate_when_constraints_are_infeasible() -> None:
    """An empty feasible set must trigger the fallback and still select a valid scale."""
    config = CRHCFGConfig(
        candidate_scales=(0.5, 1.0, 2.0),
        target_index=0,
        target_sign=1,
        protected_indices=(1,),
        m_early=10.0,
        m_late=10.0,
        epsilon_early=0.0,
        epsilon_late=0.0,
        reliability_gate=False,
    )
    controller = CRHCFGController(config)
    logits = torch.tensor([
        [[-2.0, 2.0], [-3.0, 1.0]],
        [[-1.0, 0.0], [-2.0, 0.0]],
        [[0.0, 3.0], [-1.0, 4.0]],
    ])

    decision = controller.select(logits, torch.ones(2), tau=0.5)

    assert decision.indices.min().item() >= 0
    assert decision.indices.max().item() < len(config.candidate_scales)
    assert decision.fallback_selected.all()
    assert not decision.feasible.any()
