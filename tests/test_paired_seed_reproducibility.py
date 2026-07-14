"""Tests for paired initial-noise reproducibility across HW4 methods."""

import torch

from src.sampling_hw4 import seeded_noise


def test_seeded_noise_is_reproducible_and_ordered() -> None:
    """Identical ordered seeds must reproduce exact per-sample initial noise."""
    first = seeded_noise([10, 20, 30], (3, 4, 4), torch.device("cpu"))
    second = seeded_noise([10, 20, 30], (3, 4, 4), torch.device("cpu"))
    reordered = seeded_noise([30, 10, 20], (3, 4, 4), torch.device("cpu"))

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(reordered[0], first[2])
    torch.testing.assert_close(reordered[1], first[0])
