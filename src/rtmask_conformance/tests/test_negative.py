"""Negative tests: confirm verify() reports the right status for corrupted predictions.

* zero out one prediction        -> FAIL
* shift another by 5 voxels      -> FAIL (low surface metrics)
* swap geometry of a third       -> GEOMETRY_MISMATCH
* delete a fourth                -> MISSING

Like ``test_self_roundtrip``, this generates a small fixture into ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance._roi_set import build_conformance_primitives
from rtmask_conformance._vendor.common.io import sitk_image_to_geometry
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


def _ideal_predictions(predictions_dir: Path, geometry, options: GenerateOptions) -> None:
    for primitive in build_conformance_primitives():
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=options.n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        img = sitk.GetImageFromArray(binary)
        img.SetSpacing(geometry.spacing)
        img.SetOrigin(geometry.origin)
        img.SetDirection(geometry.direction)
        sitk.WriteImage(img, str(predictions_dir / f"{primitive.name}.nii.gz"))


def test_negative_paths(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)
    gt_dir = fixture / "groundtruth"

    sample = sitk.ReadImage(str(gt_dir / f"{CONFORMANCE_ROIS[0]}.nii.gz"))
    geometry = sitk_image_to_geometry(sample)

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _ideal_predictions(pred_dir, geometry, options)

    # Sabotage one prediction per failure mode. Pick distinct ROIs so the four
    # cases don't interact.
    zero_roi = CONFORMANCE_ROIS[0]      # FAIL via empty mask
    shift_roi = CONFORMANCE_ROIS[1]     # FAIL via shifted mask
    geom_roi = CONFORMANCE_ROIS[2]      # GEOMETRY_MISMATCH
    miss_roi = CONFORMANCE_ROIS[3]      # MISSING

    # 1) Zero out
    z_path = pred_dir / f"{zero_roi}.nii.gz"
    img = sitk.ReadImage(str(z_path))
    arr = np.zeros_like(sitk.GetArrayFromImage(img))
    new = sitk.GetImageFromArray(arr)
    new.CopyInformation(img)
    sitk.WriteImage(new, str(z_path))

    # 2) Shift by 5 voxels along x
    s_path = pred_dir / f"{shift_roi}.nii.gz"
    img = sitk.ReadImage(str(s_path))
    arr = sitk.GetArrayFromImage(img)
    arr = np.roll(arr, shift=5, axis=2)
    arr[:, :, :5] = 0
    new = sitk.GetImageFromArray(arr)
    new.CopyInformation(img)
    sitk.WriteImage(new, str(s_path))

    # 3) Swap geometry: write with shifted origin so geometry precheck fires
    g_path = pred_dir / f"{geom_roi}.nii.gz"
    img = sitk.ReadImage(str(g_path))
    new = sitk.GetImageFromArray(sitk.GetArrayFromImage(img))
    new.SetSpacing(img.GetSpacing())
    new.SetDirection(img.GetDirection())
    new.SetOrigin((img.GetOrigin()[0] + 1.0, img.GetOrigin()[1], img.GetOrigin()[2]))
    sitk.WriteImage(new, str(g_path))

    # 4) Missing: delete the file
    (pred_dir / f"{miss_roi}.nii.gz").unlink()

    report = verify_predictions(pred_dir, gt_dir, load_config())
    statuses = {r.roi: r.status for r in report.results}

    assert statuses[zero_roi] == Status.FAIL
    assert statuses[shift_roi] == Status.FAIL
    assert statuses[geom_roi] == Status.GEOMETRY_MISMATCH
    assert statuses[miss_roi] == Status.MISSING
    # Untouched ROIs should still pass.
    for roi in CONFORMANCE_ROIS[4:]:
        assert statuses[roi] == Status.PASS, (roi, [r for r in report.results if r.roi == roi])

    assert not report.overall_pass
    assert report.summary["failed"] >= 2
    assert report.summary["geometry_mismatch"] == 1
    assert report.summary["missing"] == 1
