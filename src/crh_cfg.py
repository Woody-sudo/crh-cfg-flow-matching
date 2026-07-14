"""Constraint-aware receding-horizon classifier-free guidance controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass(frozen=True)
class CRHCFGConfig:
    """Store validation-frozen CRH-CFG controller parameters.

    Args:
        candidate_scales: Candidate CFG scales; the set must contain ``w=1``.
        target_index: Column of the target attribute logit.
        target_sign: Desired sign, either ``-1`` or ``+1``.
        protected_indices: Non-target attribute columns used as preservation proxies.
        m_early: Target margin threshold at generation progress ``tau=0``.
        m_late: Target margin threshold at generation progress ``tau=1``.
        gamma: Exponent controlling threshold tightening.
        epsilon_early: Protected drift budget at ``tau=0``.
        epsilon_late: Protected drift budget at ``tau=1``.
        alpha: Target violation penalty.
        preservation_weight: Preservation violation penalty.
        rho: Minimum-intervention penalty relative to ``w=1``.
        eta: Temporal smoothness penalty relative to the previous scale.
        default_scale: Scale used before an optional reliability gate activates.
        tau_on: Generation progress at which the controller becomes active.
        reliability_gate: Whether to use ``default_scale`` before ``tau_on``.
        fixed_threshold: Whether to use ``m_late`` throughout instead of tightening.
    """

    candidate_scales: tuple[float, ...]
    target_index: int
    target_sign: int
    protected_indices: tuple[int, ...] = ()
    m_early: float = -0.5
    m_late: float = 1.0
    gamma: float = 1.5
    epsilon_early: float = 2.0
    epsilon_late: float = 0.5
    alpha: float = 1.0
    preservation_weight: float = 1.0
    rho: float = 0.05
    eta: float = 0.05
    default_scale: float = 2.0
    tau_on: float = 0.1
    reliability_gate: bool = True
    fixed_threshold: bool = False

    def __post_init__(self) -> None:
        """Validate invariants that would otherwise silently change CFG semantics."""
        if not self.candidate_scales:
            raise ValueError("candidate_scales must not be empty.")
        if not any(abs(scale - 1.0) < 1e-7 for scale in self.candidate_scales):
            raise ValueError("candidate_scales must contain w=1 for the preservation reference.")
        if self.target_sign not in {-1, 1}:
            raise ValueError(f"target_sign must be -1 or +1, got {self.target_sign}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}.")
        if self.epsilon_late > self.epsilon_early:
            raise ValueError("epsilon_late must not exceed epsilon_early.")
        if not 0.0 <= self.tau_on <= 1.0:
            raise ValueError(f"tau_on must be in [0,1], got {self.tau_on}.")

    def target_threshold(self, tau: float) -> float:
        """Return the validation-frozen target threshold at generation progress ``tau``."""
        if self.fixed_threshold:
            return self.m_late
        progress = min(1.0, max(0.0, float(tau)))
        return self.m_early + (self.m_late - self.m_early) * progress**self.gamma

    def preservation_budget(self, tau: float) -> float:
        """Return the linearly tightening protected-attribute drift budget."""
        progress = min(1.0, max(0.0, float(tau)))
        return self.epsilon_early + (self.epsilon_late - self.epsilon_early) * progress


@dataclass
class ControllerDecision:
    """Contain selected scales and diagnostics for one controller decision."""

    scales: torch.Tensor
    indices: torch.Tensor
    target_margin: torch.Tensor
    protected_drift: torch.Tensor
    feasible: torch.Tensor
    target_infeasible: torch.Tensor
    preservation_infeasible: torch.Tensor
    fallback_selected: torch.Tensor
    gate_inactive: torch.Tensor


def cfg_candidate_velocities(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    scales: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Construct all candidate CFG velocities without extra generator calls.

    Args:
        unconditional: Unconditional velocity ``v_u`` with shape ``(B, ...)``.
        conditional: Conditional velocity ``v_c`` with the same shape.
        scales: Candidate CFG scales.

    Returns:
        Tensor with shape ``(M, B, ...)`` containing ``v_u+w(v_c-v_u)``.
    """
    if unconditional.shape != conditional.shape:
        raise ValueError(
            f"conditional and unconditional shapes differ: {conditional.shape} vs {unconditional.shape}."
        )
    scale_tensor = torch.as_tensor(scales, device=unconditional.device, dtype=unconditional.dtype)
    if scale_tensor.ndim != 1:
        raise ValueError(f"scales must be one-dimensional, got shape {scale_tensor.shape}.")
    scale_view = scale_tensor.view(-1, *((1,) * unconditional.ndim))
    return unconditional.unsqueeze(0) + scale_view * (conditional - unconditional).unsqueeze(0)


def gather_candidate(candidate_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one candidate per batch element from a tensor shaped ``(M,B,...)``."""
    if candidate_tensor.ndim < 2:
        raise ValueError("candidate_tensor must have candidate and batch dimensions.")
    if indices.shape != (candidate_tensor.shape[1],):
        raise ValueError(
            f"indices must have shape ({candidate_tensor.shape[1]},), got {tuple(indices.shape)}."
        )
    batch_indices = torch.arange(candidate_tensor.shape[1], device=candidate_tensor.device)
    return candidate_tensor[indices, batch_indices]


class CRHCFGController:
    """Select minimum-intervention CFG scales from evaluator logits."""

    def __init__(self, config: CRHCFGConfig):
        """Initialize the controller from validation-frozen parameters."""
        self.config = config

    def select(
        self,
        candidate_logits: torch.Tensor,
        previous_scales: torch.Tensor,
        tau: float,
    ) -> ControllerDecision:
        """Select one valid candidate per sample and return decision diagnostics.

        Args:
            candidate_logits: Signed attribute logits shaped ``(M,B,K)``.
            previous_scales: Previously selected scales shaped ``(B,)``.
            tau: Generation progress, zero at noise and one at clean data.

        Returns:
            Per-sample selected scale and feasibility/fallback diagnostics.
        """
        if candidate_logits.ndim != 3:
            raise ValueError(
                f"candidate_logits must have shape (M,B,K), got {candidate_logits.shape}."
            )
        candidate_count, batch_size, attribute_count = candidate_logits.shape
        if previous_scales.shape != (batch_size,):
            raise ValueError(
                f"previous_scales must have shape ({batch_size},), got {previous_scales.shape}."
            )
        if not 0 <= self.config.target_index < attribute_count:
            raise IndexError(
                f"target_index {self.config.target_index} is outside K={attribute_count}."
            )

        scales = torch.as_tensor(
            self.config.candidate_scales,
            device=candidate_logits.device,
            dtype=candidate_logits.dtype,
        )
        if scales.numel() != candidate_count:
            raise ValueError(
                f"Expected {scales.numel()} candidates from config, got {candidate_count}."
            )

        target_margin = self.config.target_sign * candidate_logits[:, :, self.config.target_index]
        reference_index = int(torch.argmin(torch.abs(scales - 1.0)).item())
        if self.config.protected_indices:
            protected = candidate_logits[:, :, list(self.config.protected_indices)]
            reference = candidate_logits[reference_index, :, list(self.config.protected_indices)]
            protected_drift = torch.mean(torch.abs(protected - reference.unsqueeze(0)), dim=-1)
        else:
            protected_drift = torch.zeros_like(target_margin)

        threshold = self.config.target_threshold(tau)
        budget = self.config.preservation_budget(tau)
        target_ok = target_margin >= threshold
        preservation_ok = protected_drift <= budget
        feasible_mask = target_ok & preservation_ok

        intervention = self.config.rho * (scales[:, None] - 1.0).square()
        smoothness = self.config.eta * (scales[:, None] - previous_scales[None, :]).square()
        regularization = intervention + smoothness

        feasible_cost = torch.where(
            feasible_mask,
            regularization,
            torch.full_like(regularization, torch.inf),
        )
        feasible_choice = torch.argmin(feasible_cost, dim=0)

        fallback_cost = (
            self.config.alpha * torch.relu(threshold - target_margin)
            + self.config.preservation_weight * torch.relu(protected_drift - budget)
            + regularization
        )
        fallback_choice = torch.argmin(fallback_cost, dim=0)
        any_feasible = feasible_mask.any(dim=0)
        selected = torch.where(any_feasible, feasible_choice, fallback_choice)

        gate_inactive = torch.full(
            (batch_size,),
            self.config.reliability_gate and tau < self.config.tau_on,
            device=candidate_logits.device,
            dtype=torch.bool,
        )
        if gate_inactive.any():
            default_index = torch.argmin(torch.abs(scales - self.config.default_scale))
            selected = torch.full_like(selected, int(default_index.item()))

        batch_indices = torch.arange(batch_size, device=candidate_logits.device)
        selected_margin = target_margin[selected, batch_indices]
        selected_drift = protected_drift[selected, batch_indices]
        selected_feasible = feasible_mask[selected, batch_indices] & ~gate_inactive
        fallback_selected = ~any_feasible & ~gate_inactive

        return ControllerDecision(
            scales=scales[selected],
            indices=selected,
            target_margin=selected_margin,
            protected_drift=selected_drift,
            feasible=selected_feasible,
            target_infeasible=(~target_ok.any(dim=0)) & ~gate_inactive,
            preservation_infeasible=(~preservation_ok.any(dim=0)) & ~gate_inactive,
            fallback_selected=fallback_selected,
            gate_inactive=gate_inactive,
        )


def decision_to_rows(
    decision: ControllerDecision,
    seeds: Sequence[int],
    step: int,
    tau: float,
) -> list[Dict[str, int | float | bool]]:
    """Convert a batched controller decision into serializable schedule rows."""
    if len(seeds) != decision.scales.shape[0]:
        raise ValueError("The number of seeds must match the controller batch size.")
    rows: list[Dict[str, int | float | bool]] = []
    for index, seed in enumerate(seeds):
        rows.append({
            "seed": int(seed),
            "step": int(step),
            "tau": float(tau),
            "scale": float(decision.scales[index].item()),
            "target_margin": float(decision.target_margin[index].item()),
            "protected_drift": float(decision.protected_drift[index].item()),
            "feasible": bool(decision.feasible[index].item()),
            "target_infeasible": bool(decision.target_infeasible[index].item()),
            "preservation_infeasible": bool(decision.preservation_infeasible[index].item()),
            "fallback_selected": bool(decision.fallback_selected[index].item()),
            "gate_inactive": bool(decision.gate_inactive[index].item()),
        })
    return rows
