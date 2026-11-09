"""Parametrized conformance test: one case per graded ROI.

Run with:

    RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
    RTMASK_CONFORMANCE_FIXTURE=./fixture \
    pytest --pyargs rtmask_conformance.tests.test_conformance

With a fixture, the ROI list is the manifest's graded ROIs and each ROI's
ground truth is the manifest's ``truth_path``. Without one (the v0.1 form,
``RTMASK_CONFORMANCE_GROUNDTRUTH=./fixture/groundtruth``), the ROI list is
``CONFORMANCE_ROIS``.

A user-supplied threshold YAML can be picked up via the
``RTMASK_CONFORMANCE_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.verify import Status, evaluate_one


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "roi" not in metafunc.fixturenames:
        return
    raw = metafunc.config.getoption("--fixture") or os.environ.get("RTMASK_CONFORMANCE_FIXTURE")
    if raw is not None and (Path(raw) / "manifest.json").is_file():
        rois = read_manifest(raw).graded_rois
    else:
        rois = list(CONFORMANCE_ROIS)
    metafunc.parametrize("roi", rois)


@pytest.fixture(scope="session")
def conformance_config():
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    return load_config(config_path)


def test_conformance(
    roi: str,
    predictions_dir: Path,
    groundtruth_dir: Path,
    fixture_manifest,
    conformance_config,
):
    if fixture_manifest is not None:
        fixture_dir, manifest = fixture_manifest
        gt_path = fixture_dir / manifest.rois[roi].truth_path
    else:
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
