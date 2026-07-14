"""Analytic tests for straight-path clean endpoint prediction."""

import torch

from src.endpoint_prediction import native_data_prediction, prompt_data_prediction


def test_prompt_straight_path_recovers_clean_endpoint() -> None:
    """The required mapping x0_hat=x_t-t*u_t must be exact on analytic paths."""
    clean = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(clean)
    t = torch.tensor([0.0, 0.2, 0.7, 1.0])
    t_view = t.view(-1, 1, 1, 1)
    state = (1.0 - t_view) * clean + t_view * noise
    forward_velocity = noise - clean

    torch.testing.assert_close(prompt_data_prediction(state, t, forward_velocity), clean)


def test_native_straight_path_recovers_clean_endpoint() -> None:
    """The frozen model's opposite orientation must also recover the same data."""
    clean = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(clean)
    native_s = torch.tensor([0.0, 0.2, 0.7, 1.0])
    s_view = native_s.view(-1, 1, 1, 1)
    state = (1.0 - s_view) * noise + s_view * clean
    native_velocity = clean - noise

    torch.testing.assert_close(native_data_prediction(state, native_s, native_velocity), clean)
