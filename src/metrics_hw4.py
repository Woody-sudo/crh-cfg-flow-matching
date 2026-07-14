"""Quantitative metrics for paired HW4 attribute-guidance experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def paired_attribute_metrics(
    method_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    *,
    target_index: int,
    target_sign: int,
    protected_indices: Sequence[int],
) -> Dict[str, float]:
    """Compute target and protected-attribute metrics against paired baseline samples.

    Args:
        method_logits: Final independent-evaluator margins for the method.
        baseline_logits: Margins for baseline samples sharing seeds and conditions.
        target_index: Target attribute column.
        target_sign: Desired target sign, ``-1`` or ``+1``.
        protected_indices: Protected non-target columns.

    Returns:
        Target success/logit and paired protected drift/flip metrics.
    """
    if method_logits.shape != baseline_logits.shape or method_logits.ndim != 2:
        raise ValueError(
            "method_logits and baseline_logits must have identical (B,K) shapes; "
            f"got {method_logits.shape} and {baseline_logits.shape}."
        )
    if target_sign not in {-1, 1}:
        raise ValueError(f"target_sign must be -1 or +1, got {target_sign}.")
    signed_target = target_sign * method_logits[:, target_index]
    metrics = {
        "target_success": float((signed_target >= 0).float().mean().item()),
        "target_accuracy": float((signed_target >= 0).float().mean().item()),
        "signed_target_logit_mean": float(signed_target.mean().item()),
        "signed_target_logit_std": float(signed_target.std(unbiased=False).item()),
        "target_margin_q10": float(torch.quantile(signed_target, 0.1).item()),
        "target_margin_median": float(torch.quantile(signed_target, 0.5).item()),
        "target_margin_q90": float(torch.quantile(signed_target, 0.9).item()),
    }
    if protected_indices:
        protected = list(protected_indices)
        drift = torch.abs(method_logits[:, protected] - baseline_logits[:, protected])
        flips = (method_logits[:, protected] >= 0) != (baseline_logits[:, protected] >= 0)
        metrics["protected_logit_drift_mean"] = float(drift.mean().item())
        metrics["protected_flip_rate"] = float(flips.float().mean().item())
        for local_index, attribute_index in enumerate(protected):
            metrics[f"protected_{attribute_index}_drift"] = float(drift[:, local_index].mean().item())
            metrics[f"protected_{attribute_index}_flip_rate"] = float(
                flips[:, local_index].float().mean().item()
            )
    else:
        metrics["protected_logit_drift_mean"] = 0.0
        metrics["protected_flip_rate"] = 0.0
    return metrics


def feature_pairwise_diversity(
    features: torch.Tensor,
    *,
    max_pairs: int = 4096,
    seed: int = 0,
) -> float:
    """Estimate diversity using mean cosine distance between random feature pairs."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"features must have shape (B,D) with B>=2, got {features.shape}.")
    normalized = F.normalize(features.float(), dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(features.shape[0], (max_pairs,), generator=generator)
    second = torch.randint(features.shape[0] - 1, (max_pairs,), generator=generator)
    second = second + (second >= first).long()
    distances = 1.0 - (normalized[first.to(features.device)] * normalized[second.to(features.device)]).sum(dim=1)
    return float(distances.mean().item())


def controller_summary(schedule_rows: list[dict]) -> Dict[str, float]:
    """Aggregate scale and feasibility diagnostics from serialized schedules."""
    if not schedule_rows:
        return {
            "mean_scale": float("nan"),
            "max_scale": float("nan"),
            "feasible_rate": float("nan"),
            "fallback_rate": float("nan"),
            "gate_inactive_rate": float("nan"),
        }
    scales = torch.tensor([float(row["scale"]) for row in schedule_rows])
    return {
        "mean_scale": float(scales.mean().item()),
        "max_scale": float(scales.max().item()),
        "feasible_rate": sum(bool(row["feasible"]) for row in schedule_rows) / len(schedule_rows),
        "fallback_rate": sum(bool(row["fallback_selected"]) for row in schedule_rows)
        / len(schedule_rows),
        "gate_inactive_rate": sum(bool(row["gate_inactive"]) for row in schedule_rows)
        / len(schedule_rows),
    }
