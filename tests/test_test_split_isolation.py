"""Tests for inherited split isolation during HW4 calibration."""

import pytest

from src.hw4_splits import assert_test_isolation, frozen_split_indices


def test_calibration_uses_only_non_test_indices() -> None:
    """Inherited train/validation indices must remain disjoint from final test."""
    train, validation, test = frozen_split_indices(100)

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert_test_isolation(validation, test)


def test_overlap_is_rejected() -> None:
    """The isolation guard must fail loudly on accidental final-test tuning."""
    with pytest.raises(ValueError, match="overlap"):
        assert_test_isolation([1, 2, 3], [3, 4, 5])
