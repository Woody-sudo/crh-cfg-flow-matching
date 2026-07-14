"""Endpoint prediction utilities for the frozen HW3 Flow Matching model."""

from __future__ import annotations

import torch


def prompt_data_prediction(
    x_t: torch.Tensor,
    t: torch.Tensor | float,
    forward_velocity: torch.Tensor,
) -> torch.Tensor:
    """Predict clean data for the prompt convention ``x_t=(1-t)x0+t*x1``.

    Args:
        x_t: State at prompt time ``t``, where ``t=0`` is data and ``t=1`` is noise.
        t: Scalar or batch tensor in the prompt time convention.
        forward_velocity: Velocity oriented from data to noise.

    Returns:
        Predicted clean endpoint with the same shape as ``x_t``.
    """
    time = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
    while time.ndim < x_t.ndim:
        time = time.unsqueeze(-1)
    return x_t - time * forward_velocity


def native_data_prediction(
    x_s: torch.Tensor,
    native_s: torch.Tensor | float,
    native_velocity: torch.Tensor,
) -> torch.Tensor:
    """Predict clean data from the frozen model's native noise-to-data velocity.

    The HW3 implementation uses ``x_s=(1-s)noise+s*data`` and predicts
    ``data-noise``. Its native time is therefore ``s=1-t`` relative to the HW4
    prompt convention.

    Args:
        x_s: State at native time ``s``.
        native_s: Scalar or batch tensor, with zero at noise and one at data.
        native_velocity: Frozen model velocity oriented from noise to data.

    Returns:
        Predicted clean endpoint with the same shape as ``x_s``.
    """
    time = torch.as_tensor(native_s, device=x_s.device, dtype=x_s.dtype)
    while time.ndim < x_s.ndim:
        time = time.unsqueeze(-1)
    return x_s + (1.0 - time) * native_velocity


def native_candidate_data_predictions(
    x_s: torch.Tensor,
    native_s: torch.Tensor | float,
    candidate_velocities: torch.Tensor,
) -> torch.Tensor:
    """Vectorize native endpoint prediction over candidate CFG velocities.

    Args:
        x_s: Current state with shape ``(B, C, H, W)``.
        native_s: Scalar or tensor with shape ``(B,)``.
        candidate_velocities: Tensor with shape ``(M, B, C, H, W)``.

    Returns:
        Candidate clean predictions with shape ``(M, B, C, H, W)``.
    """
    if candidate_velocities.ndim != x_s.ndim + 1:
        raise ValueError(
            "candidate_velocities must add exactly one candidate dimension; "
            f"got x_s={tuple(x_s.shape)} and candidates={tuple(candidate_velocities.shape)}."
        )
    if candidate_velocities.shape[1:] != x_s.shape:
        raise ValueError(
            "candidate batch/image shape must match x_s; "
            f"got x_s={tuple(x_s.shape)} and candidates={tuple(candidate_velocities.shape)}."
        )

    time = torch.as_tensor(native_s, device=x_s.device, dtype=x_s.dtype)
    if time.ndim == 0:
        time = time.expand(x_s.shape[0])
    if time.shape != (x_s.shape[0],):
        raise ValueError(f"native_s must be scalar or shape ({x_s.shape[0]},), got {time.shape}.")
    time = time.view(1, x_s.shape[0], *((1,) * (x_s.ndim - 1)))
    return x_s.unsqueeze(0) + (1.0 - time) * candidate_velocities
