"""Tests for the explicit HW4 and inherited HW3 time conventions."""

import torch

from src.endpoint_prediction import native_data_prediction, prompt_data_prediction


def test_time_orientation_maps_native_progress_to_prompt_time() -> None:
    """Native noise-to-data progress s must equal generation progress tau=1-t."""
    native_s = torch.tensor([0.0, 0.25, 1.0])
    prompt_t = 1.0 - native_s
    tau = 1.0 - prompt_t

    torch.testing.assert_close(tau, native_s)
    assert prompt_t[0].item() == 1.0
    assert prompt_t[-1].item() == 0.0


def test_native_and_prompt_endpoint_forms_are_equivalent() -> None:
    """Negating the native velocity must recover the prompt forward orientation."""
    data = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    native_s = torch.tensor([0.4])
    prompt_t = 1.0 - native_s
    native_velocity = data - noise
    prompt_velocity = -native_velocity
    state = (1.0 - native_s.view(1, 1, 1, 1)) * noise + native_s.view(1, 1, 1, 1) * data

    torch.testing.assert_close(native_data_prediction(state, native_s, native_velocity), data)
    torch.testing.assert_close(prompt_data_prediction(state, prompt_t, prompt_velocity), data)
