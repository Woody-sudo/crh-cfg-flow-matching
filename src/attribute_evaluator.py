"""Frozen CelebA attribute evaluator adapters used by CRH-CFG and final metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn


ATTRIBUTE_NAMES = ("smiling", "bangs", "black_hair", "blond_hair", "brown_hair")


def binary_logit_margin(logits: torch.Tensor) -> torch.Tensor:
    """Convert two-class logits into a signed present-versus-absent margin."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected binary logits with shape (B,2), got {logits.shape}.")
    return logits[:, 1] - logits[:, 0]


def outputs_to_attribute_margins(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack supported evaluator outputs in the documented HW4 attribute order."""
    missing = [name for name in ATTRIBUTE_NAMES if name not in outputs]
    if missing:
        raise KeyError(f"Evaluator outputs are missing required binary heads: {missing}.")
    return torch.stack([binary_logit_margin(outputs[name]) for name in ATTRIBUTE_NAMES], dim=1)


class FrozenAttributeEvaluator(nn.Module):
    """Expose signed attribute margins from a frozen HW3-style evaluator."""

    def __init__(self, model: nn.Module):
        """Freeze the supplied evaluator and switch it to evaluation mode."""
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return signed attribute margins for normalized images in ``[-1,1]``."""
        outputs = self.model(images.clamp(-1.0, 1.0))
        return outputs_to_attribute_margins(outputs)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Return frozen backbone features for the diversity proxy."""
        if not hasattr(self.model, "backbone"):
            raise AttributeError("The wrapped evaluator does not expose a backbone.")
        return self.model.backbone(images.clamp(-1.0, 1.0))


def load_frozen_attribute_evaluator(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FrozenAttributeEvaluator, Dict]:
    """Load a binary-head CelebA evaluator checkpoint and freeze its parameters.

    Args:
        checkpoint_path: Path to an evaluator checkpoint produced by the HW3 script.
        device: Destination device.

    Returns:
        Frozen evaluator adapter and the checkpoint metadata dictionary.
    """
    from scripts.celeba_controllability import CelebAAttributeClassifier

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hair_mode = checkpoint.get("hair_mode", checkpoint.get("args", {}).get("hair_mode", "binary"))
    if hair_mode != "binary":
        raise ValueError("CRH-CFG requires the binary-hair evaluator checkpoint.")
    model = CelebAAttributeClassifier(pretrained=False, hair_mode=hair_mode).to(device)
    model.load_state_dict(checkpoint["model"])
    return FrozenAttributeEvaluator(model).to(device), checkpoint


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary AUROC using average ranks, including tied scores.

    Args:
        scores: Signed positive-class scores.
        labels: Binary labels containing zeros and ones.

    Returns:
        AUROC in ``[0,1]`` or ``nan`` when one class is absent.
    """
    scores = scores.detach().flatten().to(dtype=torch.float64, device="cpu")
    labels = labels.detach().flatten().to(dtype=torch.long, device="cpu")
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[labels == 1].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def per_attribute_metrics(margins: torch.Tensor, labels: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Compute accuracy and AUROC for each evaluator attribute column."""
    if margins.shape != labels.shape or margins.ndim != 2:
        raise ValueError(
            f"margins and labels must have matching (B,K) shapes, got {margins.shape} and {labels.shape}."
        )
    if margins.shape[1] != len(ATTRIBUTE_NAMES):
        raise ValueError(f"Expected {len(ATTRIBUTE_NAMES)} attributes, got {margins.shape[1]}.")
    predictions = margins >= 0
    return {
        name: {
            "accuracy": float((predictions[:, index] == labels[:, index].bool()).float().mean().item()),
            "auroc": binary_auroc(margins[:, index], labels[:, index]),
        }
        for index, name in enumerate(ATTRIBUTE_NAMES)
    }
