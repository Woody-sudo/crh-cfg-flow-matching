"""
Flow Matching generative method.

This module implements conditional flow matching on straight paths between
Gaussian noise and data. The model learns the velocity field used by an Euler
sampler to transport noise into samples.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    """
    Flow matching with a straight-line probability path.

    Args:
        model: Neural network predicting the velocity field.
        device: Device used for training and sampling.
        time_scale: Scale applied to continuous t before the U-Net time embedding.
        min_time: Lower bound for training times.
        max_time: Upper bound for training times.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        time_scale: float = 1000.0,
        min_time: float = 0.0,
        max_time: float = 1.0,
        conditional: bool = False,
        condition_dropout: float = 0.15,
        field_dropout: float = 0.20,
    ):
        super().__init__(model, device)
        if not 0.0 <= min_time < max_time <= 1.0:
            raise ValueError(
                "FlowMatching expects 0 <= min_time < max_time <= 1, "
                f"got min_time={min_time}, max_time={max_time}."
            )

        self.time_scale = float(time_scale)
        self.min_time = float(min_time)
        self.max_time = float(max_time)
        self.conditional = bool(conditional)
        self.condition_dropout = float(condition_dropout)
        self.field_dropout = float(field_dropout)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous flow time to the U-Net's sinusoidal time scale."""
        return t * self.time_scale

    def _expand_time(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape a batch of scalar times for image broadcasting."""
        return t.reshape(t.shape[0], *((1,) * (x.ndim - 1)))

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract image and optional condition tensors from supported batch formats.

        Args:
            batch: Tensor, tuple/list, or dictionary from a DataLoader.

        Returns:
            Tuple of image tensor and optional condition tensor.
        """
        if isinstance(batch, dict):
            image = batch["image"]
            condition = batch.get("condition")
            return image, condition
        if isinstance(batch, (tuple, list)):
            image = batch[0]
            condition = batch[1] if len(batch) > 1 else None
            return image, condition
        return batch, None

    def _drop_conditions(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free training dropout to discrete condition fields.

        Args:
            condition: Long tensor of shape (batch_size, num_fields).

        Returns:
            Condition tensor with some rows or fields replaced by the null id 0.
        """
        condition = condition.clone()
        if self.condition_dropout > 0:
            row_mask = torch.rand(condition.shape[0], device=condition.device) < self.condition_dropout
            condition[row_mask] = 0
        if self.field_dropout > 0:
            field_mask = torch.rand(condition.shape, device=condition.device) < self.field_dropout
            condition[field_mask] = 0
        return condition

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict velocity with optional classifier-free guidance.

        Args:
            x_t: Current ODE state.
            t: Continuous time tensor.
            condition: Optional semantic condition ids.
            guidance_scale: CFG scale; ``None`` disables guidance composition.

        Returns:
            Predicted velocity tensor.
        """
        embedded_time = self._embed_time(t)
        if condition is None:
            return self.model(x_t, embedded_time)
        if guidance_scale is None:
            return self.model(x_t, embedded_time, condition=condition)

        scale = float(guidance_scale)
        if scale == 1.0:
            return self.model(x_t, embedded_time, condition=condition)

        null_condition = torch.zeros_like(condition)
        unconditional = self.model(x_t, embedded_time, condition=null_condition)
        if scale == 0.0:
            return unconditional

        conditional = self.model(x_t, embedded_time, condition=condition)
        return unconditional + scale * (conditional - unconditional)

    def sample_time(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous training times.

        Returns:
            Tensor of shape (batch_size,) with values in [min_time, max_time].
        """
        t = torch.rand(batch_size, device=self.device)
        return self.min_time + (self.max_time - self.min_time) * t

    def interpolate(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate from noise at t=0 to data at t=1.

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width).
            t: Continuous times of shape (batch_size,).
            noise: Optional starting noise with the same shape as x_0.

        Returns:
            x_t: Samples on the straight path.
            velocity: Constant target velocity x_0 - noise.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t_view = self._expand_time(t, x_0)
        velocity = x_0 - noise
        x_t = (1.0 - t_view) * noise + t_view * x_0
        return x_t, velocity

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching MSE loss.

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width).
            **kwargs: Additional method-specific arguments.

        Returns:
            loss: Scalar MSE between predicted and target velocities.
            metrics: Dictionary of scalar metrics for logging.
        """
        x_0, condition = self._unpack_batch(x_0)
        x_0 = x_0.to(self.device)
        if condition is not None:
            condition = condition.to(self.device, dtype=torch.long)
        batch_size = x_0.shape[0]
        t = self.sample_time(batch_size)
        x_t, target_velocity = self.interpolate(x_0, t)

        train_condition = None
        if self.conditional and condition is not None:
            train_condition = self._drop_conditions(condition)

        predicted_velocity = self._predict_velocity(x_t, t, condition=train_condition)
        loss = F.mse_loss(predicted_velocity, target_velocity)

        metrics = {
            "loss": loss.detach(),
            "mse": loss.detach(),
            "velocity_norm": target_velocity.detach().pow(2).mean().sqrt(),
        }
        if train_condition is not None:
            metrics["null_field_fraction"] = (train_condition == 0).float().mean().detach()
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples by fixed-step ODE integration from t=0 to t=1.

        Args:
            batch_size: Number of samples to generate.
            image_shape: Shape of each image as (channels, height, width).
            num_steps: Number of integration steps.
            **kwargs: Additional method-specific arguments. ``sampler`` may be
                ``"euler"`` or ``"heun"``.

        Returns:
            Generated samples of shape (batch_size, *image_shape).
        """
        self.eval_mode()
        steps = int(num_steps or kwargs.get("sampling_steps", 100))
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}.")

        sampler = kwargs.get("sampler") or "euler"
        if sampler not in {"euler", "heun"}:
            raise ValueError(
                "FlowMatching sampler must be 'euler' or 'heun', "
                f"got {sampler!r}."
            )

        condition = kwargs.get("condition")
        guidance_scale = kwargs.get("guidance_scale", kwargs.get("cfg_scale", None))
        if condition is not None:
            condition = torch.as_tensor(condition, device=self.device, dtype=torch.long)
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            if condition.shape[0] == 1 and batch_size > 1:
                condition = condition.repeat(batch_size, 1)
            if condition.shape[0] != batch_size:
                raise ValueError(
                    f"condition batch size must be 1 or {batch_size}, got {condition.shape[0]}."
                )
        elif self.conditional and guidance_scale is not None:
            raise ValueError("CFG sampling requires a non-null condition tensor.")

        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        dt = 1.0 / steps

        for step in range(steps):
            t = torch.full((batch_size,), step * dt, device=self.device)
            velocity = self._predict_velocity(
                x_t,
                t,
                condition=condition,
                guidance_scale=guidance_scale,
            )

            if sampler == "euler":
                x_t = x_t + dt * velocity
                continue

            # Heun averages endpoint slopes, reducing integration error without retraining.
            x_pred = x_t + dt * velocity
            t_next = torch.full((batch_size,), (step + 1) * dt, device=self.device)
            velocity_next = self._predict_velocity(
                x_pred,
                t_next,
                condition=condition,
                guidance_scale=guidance_scale,
            )
            x_t = x_t + 0.5 * dt * (velocity + velocity_next)

        return x_t.clamp(-1.0, 1.0)

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["time_scale"] = self.time_scale
        state["min_time"] = self.min_time
        state["max_time"] = self.max_time
        state["conditional"] = self.conditional
        state["condition_dropout"] = self.condition_dropout
        state["field_dropout"] = self.field_dropout
        return state

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ) -> "FlowMatching":
        """
        Create a FlowMatching instance from a training config.

        Args:
            model: Neural network predicting velocity.
            config: Full YAML config or a flow_matching section.
            device: Device used by the method.

        Returns:
            Configured FlowMatching method.
        """
        flow_config = config.get("flow_matching", config)
        return cls(
            model=model,
            device=device,
            time_scale=flow_config.get("time_scale", 1000.0),
            min_time=flow_config.get("min_time", 0.0),
            max_time=flow_config.get("max_time", 1.0),
            conditional=flow_config.get("conditional", config.get("data", {}).get("conditional", False)),
            condition_dropout=flow_config.get("condition_dropout", 0.15),
            field_dropout=flow_config.get("field_dropout", 0.20),
        )
