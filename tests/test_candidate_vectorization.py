"""Tests for vectorized CFG construction and endpoint prediction."""

import torch

from src.crh_cfg import cfg_candidate_velocities
from src.endpoint_prediction import native_candidate_data_predictions, native_data_prediction


def test_candidate_batching_matches_loop() -> None:
    """Batched candidate operations must numerically match a reference loop."""
    unconditional = torch.randn(3, 2, 4, 4)
    conditional = torch.randn_like(unconditional)
    state = torch.randn_like(unconditional)
    native_s = torch.tensor([0.1, 0.5, 0.9])
    scales = [0.5, 1.0, 2.0, 4.0]

    velocities = cfg_candidate_velocities(unconditional, conditional, scales)
    endpoints = native_candidate_data_predictions(state, native_s, velocities)
    loop_velocities = torch.stack([
        unconditional + scale * (conditional - unconditional) for scale in scales
    ])
    loop_endpoints = torch.stack([
        native_data_prediction(state, native_s, velocity) for velocity in loop_velocities
    ])

    torch.testing.assert_close(velocities, loop_velocities)
    torch.testing.assert_close(endpoints, loop_endpoints)
