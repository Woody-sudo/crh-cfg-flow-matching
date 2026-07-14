"""Frozen HW3 evaluator split utilities and HW4 test-isolation checks."""

from __future__ import annotations

from typing import Iterable

import torch


FROZEN_SPLIT_SEED = 42


def frozen_split_indices(
    dataset_size: int,
    seed: int = FROZEN_SPLIT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the inherited deterministic 80/10/10 evaluator split.

    Args:
        dataset_size: Number of examples in the frozen CelebA subset.
        seed: Frozen permutation seed; defaults to the HW3 value.

    Returns:
        Train, validation, and final-test index lists.
    """
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_end = int(dataset_size * 0.8)
    validation_end = int(dataset_size * 0.9)
    return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]


def assert_test_isolation(calibration_indices: Iterable[int], test_indices: Iterable[int]) -> None:
    """Raise when controller calibration includes any frozen final-test example."""
    overlap = set(calibration_indices).intersection(test_indices)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Calibration/test overlap detected; first overlapping indices: {preview}.")
