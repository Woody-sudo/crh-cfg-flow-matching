"""Paired HW4 samplers for static, interval, and CRH-CFG guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .crh_cfg import (
    CRHCFGController,
    cfg_candidate_velocities,
    decision_to_rows,
    gather_candidate,
)
from .endpoint_prediction import native_candidate_data_predictions


AttributeEvaluator = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class SamplingCounters:
    """Track generator and controller work for one sampling batch."""

    guided_field_evaluations: int = 0
    raw_generator_forwards: int = 0
    evaluator_forwards: int = 0
    evaluator_examples: int = 0


@dataclass
class SamplingResult:
    """Return generated samples, controller schedules, and compute counters."""

    samples: torch.Tensor
    schedule_rows: list[dict] = field(default_factory=list)
    counters: SamplingCounters = field(default_factory=SamplingCounters)


def seeded_noise(
    seeds: Sequence[int],
    image_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create batch-independent initial noise from an explicit ordered seed list.

    Args:
        seeds: Per-sample integer seeds.
        image_shape: Image shape as ``(C,H,W)``.
        device: Destination device.
        dtype: Destination floating-point dtype.

    Returns:
        Noise tensor shaped ``(len(seeds), C, H, W)``.
    """
    samples = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        samples.append(torch.randn(image_shape, generator=generator, dtype=dtype))
    if not samples:
        raise ValueError("seeds must not be empty.")
    return torch.stack(samples).to(device=device)


def compose_cfg_velocity(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    scales: torch.Tensor | float,
) -> torch.Tensor:
    """Compose one per-sample CFG velocity using ``v_u+w(v_c-v_u)``."""
    scale_tensor = torch.as_tensor(scales, device=unconditional.device, dtype=unconditional.dtype)
    if scale_tensor.ndim == 0:
        scale_tensor = scale_tensor.expand(unconditional.shape[0])
    if scale_tensor.shape != (unconditional.shape[0],):
        raise ValueError(
            f"scales must be scalar or shape ({unconditional.shape[0]},), got {scale_tensor.shape}."
        )
    scale_view = scale_tensor.view(unconditional.shape[0], *((1,) * (unconditional.ndim - 1)))
    return unconditional + scale_view * (conditional - unconditional)


class FrozenCFGFields:
    """Evaluate conditional and unconditional fields while counting frozen-model NFEs."""

    def __init__(self, model: nn.Module, time_scale: float = 1000.0):
        """Store the frozen model and its native time-embedding scale."""
        self.model = model
        self.time_scale = float(time_scale)
        self.counters = SamplingCounters()

    def predict(
        self,
        state: torch.Tensor,
        native_s: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ``v_u`` and ``v_c`` exactly once each at the supplied state."""
        embedded_time = native_s * self.time_scale
        null_condition = torch.zeros_like(condition)
        unconditional = self.model(state, embedded_time, condition=null_condition)
        conditional = self.model(state, embedded_time, condition=condition)
        self.counters.guided_field_evaluations += 1
        self.counters.raw_generator_forwards += 2
        return unconditional, conditional


def _method_scales(
    method: str,
    batch_size: int,
    tau: float,
    device: torch.device,
    dtype: torch.dtype,
    static_scale: float,
    interval_start: float,
    interval_end: float,
) -> torch.Tensor:
    """Return scales for non-feedback baselines."""
    if method in {"hw3_baseline", "static_cfg"}:
        scale = static_scale
    elif method == "interval_cfg":
        scale = static_scale if interval_start <= tau <= interval_end else 1.0
    else:
        raise ValueError(f"Method {method!r} requires a CRH-CFG controller.")
    return torch.full((batch_size,), scale, device=device, dtype=dtype)


@torch.no_grad()
def sample_hw4(
    model: nn.Module,
    condition: torch.Tensor,
    seeds: Sequence[int],
    *,
    method: str,
    num_steps: int = 50,
    sampler: str = "heun",
    image_shape: tuple[int, int, int] = (3, 64, 64),
    time_scale: float = 1000.0,
    static_scale: float = 2.0,
    interval_start: float = 0.2,
    interval_end: float = 0.8,
    controller: CRHCFGController | None = None,
    evaluator: AttributeEvaluator | None = None,
    controller_frequency: int = 1,
    endpoint_clip: tuple[float, float] = (-1.0, 1.0),
) -> SamplingResult:
    """Generate paired samples with unchanged Euler or Heun solver updates.

    Args:
        model: Frozen HW3 conditional velocity model.
        condition: Discrete condition tensor shaped ``(B,F)``.
        seeds: Ordered per-sample initial-noise seeds.
        method: ``hw3_baseline``, ``static_cfg``, ``interval_cfg``, or ``crh_cfg``.
        num_steps: Number of original HW3 solver macro-steps.
        sampler: Original solver, ``euler`` or ``heun``.
        image_shape: Generated image shape.
        time_scale: Frozen model's native time embedding scale.
        static_scale: Scale for static/HW3 and active interval guidance.
        interval_start: Inclusive interval start in generation progress ``tau``.
        interval_end: Inclusive interval end in generation progress ``tau``.
        controller: CRH-CFG controller for feedback sampling.
        evaluator: Frozen control evaluator returning signed attribute logits ``(B,K)``.
        controller_frequency: Evaluate the controller every N macro-steps and hold otherwise.
        endpoint_clip: Evaluator-domain clipping range for predicted endpoints.

    Returns:
        Generated samples, controller schedule rows, and exact per-batch counters.
    """
    if sampler not in {"euler", "heun"}:
        raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler!r}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if controller_frequency <= 0:
        raise ValueError(f"controller_frequency must be positive, got {controller_frequency}.")
    if condition.ndim != 2 or condition.shape[0] != len(seeds):
        raise ValueError(
            f"condition must have shape ({len(seeds)},F), got {tuple(condition.shape)}."
        )
    feedback = method == "crh_cfg"
    if feedback and (controller is None or evaluator is None):
        raise ValueError("crh_cfg requires both controller and evaluator.")
    if not feedback and method not in {"hw3_baseline", "static_cfg", "interval_cfg"}:
        raise ValueError(f"Unknown HW4 sampling method: {method!r}.")

    model.eval()
    device = next(model.parameters()).device
    condition = condition.to(device=device, dtype=torch.long)
    state = seeded_noise(seeds, image_shape, device)
    fields = FrozenCFGFields(model, time_scale=time_scale)
    schedule_rows: list[dict] = []
    previous_scales = torch.ones(len(seeds), device=device, dtype=state.dtype)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        native_s_value = step * dt
        # Under the prompt convention t=1-s, generation progress tau=1-t equals native s.
        tau = native_s_value
        native_s = torch.full((len(seeds),), native_s_value, device=device, dtype=state.dtype)
        unconditional, conditional = fields.predict(state, native_s, condition)

        if feedback and step % controller_frequency == 0:
            assert controller is not None and evaluator is not None
            candidates = cfg_candidate_velocities(
                unconditional,
                conditional,
                controller.config.candidate_scales,
            )
            predicted_data = native_candidate_data_predictions(state, native_s, candidates)
            clipped = predicted_data.clamp(*endpoint_clip)
            flat = clipped.reshape(-1, *clipped.shape[2:])
            flat_logits = evaluator(flat)
            if flat_logits.ndim != 2:
                raise ValueError(f"evaluator must return shape (B,K), got {flat_logits.shape}.")
            candidate_logits = flat_logits.reshape(
                len(controller.config.candidate_scales),
                len(seeds),
                flat_logits.shape[-1],
            )
            fields.counters.evaluator_forwards += 1
            fields.counters.evaluator_examples += flat.shape[0]
            decision = controller.select(candidate_logits, previous_scales, tau)
            scales = decision.scales
            velocity = gather_candidate(candidates, decision.indices)
            schedule_rows.extend(decision_to_rows(decision, seeds, step, tau))
        elif feedback:
            scales = previous_scales
            velocity = compose_cfg_velocity(unconditional, conditional, scales)
        else:
            scales = _method_scales(
                method,
                len(seeds),
                tau,
                device,
                state.dtype,
                static_scale,
                interval_start,
                interval_end,
            )
            velocity = compose_cfg_velocity(unconditional, conditional, scales)

        if sampler == "euler":
            state = state + dt * velocity
            previous_scales = scales
            continue

        # Protocol B holds the selected scale across the Heun predictor/corrector pair.
        predicted_state = state + dt * velocity
        next_s = torch.full(
            (len(seeds),),
            (step + 1) * dt,
            device=device,
            dtype=state.dtype,
        )
        next_unconditional, next_conditional = fields.predict(predicted_state, next_s, condition)
        next_velocity = compose_cfg_velocity(next_unconditional, next_conditional, scales)
        state = state + 0.5 * dt * (velocity + next_velocity)
        previous_scales = scales

    return SamplingResult(
        samples=state.clamp(-1.0, 1.0),
        schedule_rows=schedule_rows,
        counters=fields.counters,
    )
