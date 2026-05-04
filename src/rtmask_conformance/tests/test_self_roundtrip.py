"""Self-roundtrip: rasterize the analytic primitives via the same vendored
voxelizer that built the ground truth, and confirm verify() reports Dice = 1.0.

This is the suite's own correctness test — if it ever fails, ``verify.py``
or the vendored metric/voxelizer code has a bug. It runs end-to-end
(``generate`` → re-rasterize → ``verify``) on a tiny grid so it stays under a
few seconds in CI.

Skipped by default when ``RTMASK_CONFORMANCE_SKIP_E2E=1`` is set; otherwise
runs unconditionally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import SimpleITK as sitk

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance._roi_set import build_conformance_primitives
from rtmask_conformance._vendor.common.io import sitk_image_to_geometry, write_nifti
from rtmask_conformance._vendor.groundtruth.partial_volume import (
    binary_threshold,
    partial_volume_mask,
)
from rtmask_conformance.generate import GenerateOptions, generate_fixture
from rtmask_conformance.verify import Status, verify_predictions

pytestmark = pytest.mark.skipif(
    os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set",
)


def test_self_roundtrip(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    # Primitive positions are calibrated to the canonical 512x512x200 grid; smaller
    # grids would clip several ROIs. Drop n_quadrature to keep this end-to-end
    # test under a minute on a typical CI runner.
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)

    # Rasterize each primitive on the same geometry as the GT and dump as a
    # prediction. By construction these masks are byte-identical to the GT.
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    primitives = build_conformance_primitives()
    gt_dir = fixture / "groundtruth"

    # Pull the geometry back from the GT NIfTI rather than reconstructing it,
    # since a real consumer reads it that way.
    sample = sitk.ReadImage(str(gt_dir / f"{primitives[0].name}.nii.gz"))
    geometry = sitk_image_to_geometry(sample)

    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=options.n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        write_nifti(binary, geometry, pred_dir / f"{primitive.name}.nii.gz")

    report = verify_predictions(pred_dir, gt_dir, load_config())
    assert report.overall_pass, [r.violations for r in report.results if r.status != Status.PASS]
    for record in report.results:
        assert record.status == Status.PASS, (record.roi, record.violations)
        assert record.metrics["dice"] == pytest.approx(1.0)


def test_all_seven_rois_present_in_groundtruth(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)
    gt_dir = fixture / "groundtruth"
    for roi in CONFORMANCE_ROIS:
        assert (gt_dir / f"{roi}.nii.gz").is_file(), roi
