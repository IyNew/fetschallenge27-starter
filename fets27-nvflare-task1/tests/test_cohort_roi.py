"""Regression tests for the training/inference ROI contract."""

from __future__ import annotations

import pytest

from fets27_challenge.cohort_registry import COHORT_REGISTRY


@pytest.mark.parametrize("cohort_name", sorted(COHORT_REGISTRY))
def test_infer_roi_matches_training_roi(cohort_name: str) -> None:
    """Validation inference must use the same ROI as training."""
    spec = COHORT_REGISTRY[cohort_name]
    assert spec.infer_roi_size == spec.roi_size


def test_brain_tumor_uses_128_cubed_inference() -> None:
    """Pin the brain-tumor cohort to 128^3 training and inference."""
    spec = COHORT_REGISTRY["brain_tumor"]
    assert spec.roi_size == (128, 128, 128)
    assert spec.infer_roi_size == (128, 128, 128)
