"""Tests for the frozen HW3 classifier-free guidance convention."""

import torch

from src.crh_cfg import cfg_candidate_velocities


def test_cfg_zero_is_unconditional_and_one_is_conditional() -> None:
    """Candidate endpoints w=0 and w=1 must preserve the documented semantics."""
    unconditional = torch.randn(2, 3, 4, 4)
    conditional = torch.randn(2, 3, 4, 4)
    candidates = cfg_candidate_velocities(unconditional, conditional, [0.0, 1.0])

    torch.testing.assert_close(candidates[0], unconditional)
    torch.testing.assert_close(candidates[1], conditional)
