"""Parametrized conformance test: one case per scored ROI.

Run with:

    RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
    RTMASK_CONFORMANCE_FIXTURE=./fixture \
    pytest --pyargs rtmask_conformance.tests.test_conformance

With a fixture, the ROI list is the one ``verify --fixture`` scores (every
manifest ROI with a binary-mask ground truth, graded or not) and each ROI's
ground truth is the manifest's ``truth_path``. Without one (the v0.1 form,
``RTMASK_CONFORMANCE_GROUNDTRUTH=./fixture/groundtruth``), the ROI list is
``CONFORMANCE_ROIS``.

Statuses map onto pytest outcomes as they map onto the CLI exit code:
``PASS`` passes; ``FAIL``, ``MISSING`` and ``GEOMETRY_MISMATCH`` fail;
``UNSUPPORTED_TYPE`` (a type the threshold YAML does not claim, and no
prediction) and ``REPORT_ONLY`` (scored, never graded) are skipped with the
reason in the skip message.

A user-supplied threshold YAML can be picked up via the
``RTMASK_CONFORMANCE_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.verify import (
    Status,
    _resolve_prediction_path,
    evaluate_one,
    fixture_targets,
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "roi" not in metafunc.fixturenames:
        return
    raw = metafunc.config.getoption("--fixture") or os.environ.get("RTMASK_CONFORMANCE_FIXTURE")
    if raw is not None and (Path(raw) / "manifest.json").is_file():
        rois = [t.roi for t in fixture_targets(raw, read_manifest(raw))]
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
        target = {t.roi: t for t in fixture_targets(fixture_dir, manifest)}[roi]
        gt_path = target.gt_path
        kwargs = {"geometric_type": target.geometric_type, "truth_kind": target.truth_kind,
                  "graded": target.graded}
    else:
        gt_path = groundtruth_dir / f"{roi}.nii.gz"
        kwargs = {}  # geometric type from the suite's own ROI table; graded
    assert gt_path.is_file(), f"ground-truth missing at {gt_path}"

    pred_path = _resolve_prediction_path(predictions_dir, roi)
    result = evaluate_one(roi, pred_path, gt_path, conformance_config, **kwargs)

    if result.status == Status.UNSUPPORTED_TYPE:
        pytest.skip(f"{roi}: geometric type {result.geometric_type} not claimed, no prediction")
    if result.status == Status.REPORT_ONLY:
        detail = result.geometry_diagnostic or result.violations or "within thresholds"
        pytest.skip(f"{roi}: report-only, not graded; metrics={result.metrics}; {detail}")
    if result.status == Status.MISSING:
        pytest.fail(f"prediction file missing: {predictions_dir / (roi + '.nii.gz')}")
    if result.status == Status.GEOMETRY_MISMATCH:
        pytest.fail(f"geometry mismatch for {roi}: {result.geometry_diagnostic}")
    if result.status == Status.FAIL:
        pytest.fail(
            f"{roi} failed thresholds: {result.violations}; metrics={result.metrics}"
        )
    assert result.status == Status.PASS
