"""Parametrized conformance test: one case per ROI.

Run with:

    RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
    RTMASK_CONFORMANCE_GROUNDTRUTH=./fixture/groundtruth \
    pytest --pyargs rtmask_conformance.tests.test_conformance

A user-supplied threshold YAML can be picked up via the
``RTMASK_CONFORMANCE_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance.verify import Status, evaluate_one


@pytest.fixture(scope="session")
def conformance_config():
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_conformance(roi: str, predictions_dir: Path, groundtruth_dir: Path, conformance_config):
    gt_path = groundtruth_dir / f"{roi}.nii.gz"
    assert gt_path.is_file(), f"ground-truth missing at {gt_path}"

    pred_path = predictions_dir / f"{roi}.nii.gz"
    if not pred_path.is_file():
        # case-insensitive fallback handled by evaluate_one's caller; here we
        # still want a clear failure message rather than a silent skip.
        pred_path = next(
            (p for p in predictions_dir.glob("*.nii.gz") if p.name.lower() == f"{roi}.nii.gz".lower()),
            None,
        )

    result = evaluate_one(roi, pred_path, gt_path, conformance_config)

    if result.status == Status.MISSING:
        pytest.fail(f"prediction file missing: {predictions_dir / (roi + '.nii.gz')}")
    if result.status == Status.GEOMETRY_MISMATCH:
        pytest.fail(f"geometry mismatch for {roi}: {result.geometry_diagnostic}")
    if result.status == Status.FAIL:
        pytest.fail(
            f"{roi} failed thresholds: {result.violations}; metrics={result.metrics}"
        )
    assert result.status == Status.PASS
