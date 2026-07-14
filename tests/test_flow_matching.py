"""Tests for Flow Matching training and sampling behavior."""

import torch
import torch.nn as nn

from src.data import derive_celeba_condition
from src.methods import FlowMatching
from src.models import UNet


class TimeVelocity(nn.Module):
    """Return a spatially constant velocity equal to continuous time."""

    def forward(self, x: torch.Tensor, embedded_time: torch.Tensor) -> torch.Tensor:
        """Predict dx/dt=t using the method's default time embedding scale."""
        time = embedded_time / 1000.0
        return time.view(-1, *((1,) * (x.ndim - 1))).expand_as(x)


class ConditionVelocity(nn.Module):
    """Return velocity from the sum of condition ids for CFG tests."""

    def forward(
        self,
        x: torch.Tensor,
        embedded_time: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a constant field tied to the provided semantic condition."""
        if condition is None:
            value = torch.zeros(x.shape[0], device=x.device)
        else:
            value = condition.float().sum(dim=1)
        return value.view(-1, *((1,) * (x.ndim - 1))).expand_as(x)


def test_heun_integrates_linear_time_field_exactly(monkeypatch) -> None:
    """Heun should exactly integrate dx/dt=t while Euler has first-order error."""
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    method = FlowMatching(TimeVelocity(), torch.device("cpu"))

    euler = method.sample(1, (1, 1, 1), num_steps=4, sampler="euler")
    heun = method.sample(1, (1, 1, 1), num_steps=4, sampler="heun")

    torch.testing.assert_close(euler, torch.tensor([[[[0.375]]]]))
    torch.testing.assert_close(heun, torch.tensor([[[[0.5]]]]))


def test_unknown_sampler_is_rejected() -> None:
    """Invalid sampler names should fail before model evaluation."""
    method = FlowMatching(TimeVelocity(), torch.device("cpu"))

    try:
        method.sample(1, (1, 1, 1), num_steps=1, sampler="unknown")
    except ValueError as error:
        assert "euler" in str(error)
        assert "heun" in str(error)
    else:
        raise AssertionError("Expected an invalid Flow Matching sampler to raise ValueError.")


def test_celeba_condition_schema_handles_ambiguous_hair() -> None:
    """Ambiguous hair labels should not train a false single-color condition."""
    condition = derive_celeba_condition({
        "Smiling": 1,
        "Bangs": 0,
        "Black_Hair": 1,
        "Blond_Hair": 1,
        "Brown_Hair": 0,
    })

    torch.testing.assert_close(condition, torch.tensor([2, 1, 0]))


def test_unet_accepts_semantic_condition() -> None:
    """Conditional U-Net should preserve the image tensor shape."""
    model = UNet(
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_resolutions=[],
        num_heads=1,
        condition_num_classes=[3, 3, 5],
    )
    x = torch.randn(2, 3, 64, 64)
    t = torch.zeros(2)
    condition = torch.tensor([[2, 1, 2], [1, 2, 3]])

    output = model(x, t, condition=condition)

    assert output.shape == x.shape


def test_cfg_combines_conditional_and_unconditional_velocity(monkeypatch) -> None:
    """CFG should compose unconditional and conditional velocity fields."""
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    method = FlowMatching(ConditionVelocity(), torch.device("cpu"), conditional=True)

    sample = method.sample(
        1,
        (1, 1, 1),
        num_steps=1,
        sampler="euler",
        condition=torch.tensor([[2, 1, 2]]),
        guidance_scale=2.0,
    )

    torch.testing.assert_close(sample, torch.tensor([[[[1.0]]]]))
