"""Tests for the plugin evaluator API (:mod:`rtmask_conformance.evaluator`).

Covers all three accepted input forms (ndarray / path / SimpleITK.Image) and
both functions: ``evaluate_masks`` (raw metrics) and
``evaluate_masks_with_thresholds`` (graded ResultRecord).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtmask_conformance import (
    GeometryMismatchError,
    MaskMetrics,
    ResultRecord,
    evaluate_masks,
    evaluate_masks_with_thresholds,
    load_config,
)
from rtmask_conformance.thresholds import Thresholds
from rtmask_conformance.verify import Status


def _cube(shape: tuple[int, int, int], origin: tuple[int, int, int], side: int) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    z, y, x = origin
    arr[z:z + side, y:y + side, x:x + side] = 1
    return arr


def _write_nifti(
    arr: np.ndarray,
    path: Path,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(path))
    return path


# ---------------------------------------------------------------------------
# evaluate_masks — input-form coverage
# ---------------------------------------------------------------------------


def test_evaluate_masks_ndarray_inputs() -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    m = evaluate_masks(a, a, spacing_xyz=(1.0, 1.0, 1.0))
    assert isinstance(m, MaskMetrics)
    assert m.dice == 1.0
    assert m.volume_rel_err == pytest.approx(0.0)
    assert m.spacing_xyz == (1.0, 1.0, 1.0)


def test_evaluate_masks_ndarray_inputs_require_spacing() -> None:
    a = _cube((10, 10, 10), (2, 2, 2), 5)
    with pytest.raises(ValueError, match="spacing_xyz is required"):
        evaluate_masks(a, a)


def test_evaluate_masks_path_inputs(tmp_path: Path) -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    pred = _write_nifti(a, tmp_path / "pred.nii.gz", spacing=(0.5, 1.0, 2.0))
    gt = _write_nifti(a, tmp_path / "gt.nii.gz", spacing=(0.5, 1.0, 2.0))
    m = evaluate_masks(pred, gt)
    assert m.dice == 1.0
    assert m.spacing_xyz == (0.5, 1.0, 2.0)
    assert m.tool_volume_mm3 == pytest.approx(1000.0 * 0.5 * 1.0 * 2.0)


def test_evaluate_masks_sitk_image_inputs() -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((1.0, 1.0, 1.0))
    m = evaluate_masks(img, img)
    assert m.dice == 1.0


def test_evaluate_masks_mixed_ndarray_and_path(tmp_path: Path) -> None:
    """ndarray on one side, NIfTI path on the other — spacing inferred from path."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    gt_path = _write_nifti(a, tmp_path / "gt.nii.gz", spacing=(0.5, 0.5, 0.5))
    m = evaluate_masks(a, gt_path)
    assert m.dice == 1.0
    assert m.spacing_xyz == (0.5, 0.5, 0.5)


def test_evaluate_masks_explicit_spacing_overrides_image_spacing(tmp_path: Path) -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    pred = _write_nifti(a, tmp_path / "p.nii.gz", spacing=(1.0, 1.0, 1.0))
    gt = _write_nifti(a, tmp_path / "g.nii.gz", spacing=(1.0, 1.0, 1.0))
    m = evaluate_masks(pred, gt, spacing_xyz=(2.0, 2.0, 2.0))
    assert m.spacing_xyz == (2.0, 2.0, 2.0)
    assert m.tool_volume_mm3 == pytest.approx(1000.0 * 8.0)  # 8 mm³ per voxel


def test_evaluate_masks_unsupported_input_type() -> None:
    a = _cube((10, 10, 10), (2, 2, 2), 5)
    with pytest.raises(TypeError, match="Unsupported mask input"):
        evaluate_masks([1, 2, 3], a, spacing_xyz=(1.0, 1.0, 1.0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# evaluate_masks — geometry checking
# ---------------------------------------------------------------------------


def test_evaluate_masks_geometry_mismatch_raises(tmp_path: Path) -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    pred = _write_nifti(a, tmp_path / "p.nii.gz", origin=(0.0, 0.0, 0.0))
    gt = _write_nifti(a, tmp_path / "g.nii.gz", origin=(1.0, 0.0, 0.0))
    with pytest.raises(GeometryMismatchError):
        evaluate_masks(pred, gt)


def test_evaluate_masks_geometry_mismatch_skipped_when_disabled(tmp_path: Path) -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    pred = _write_nifti(a, tmp_path / "p.nii.gz", origin=(0.0, 0.0, 0.0))
    gt = _write_nifti(a, tmp_path / "g.nii.gz", origin=(1.0, 0.0, 0.0))
    m = evaluate_masks(pred, gt, check_geometry=False)
    assert m.dice == 1.0  # arrays are identical even though origins disagree


def test_evaluate_masks_geometry_check_skipped_for_ndarray_input() -> None:
    """Geometry precheck only fires when both inputs carry geometry."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    m = evaluate_masks(a, a, spacing_xyz=(1.0, 1.0, 1.0), check_geometry=True)
    assert m.dice == 1.0


# ---------------------------------------------------------------------------
# evaluate_masks — values match the underlying metric functions
# ---------------------------------------------------------------------------


def test_evaluate_masks_half_overlap_dice() -> None:
    a = _cube((20, 20, 20), (5, 5, 0), 10)
    b = _cube((20, 20, 20), (5, 5, 5), 10)
    m = evaluate_masks(a, b, spacing_xyz=(1.0, 1.0, 1.0))
    assert m.dice == pytest.approx(0.5)
    assert m.volume_rel_err == pytest.approx(0.0)  # equal volumes


def test_evaluate_masks_surface_tolerance_argument_propagates() -> None:
    """Passing surface_dsc_tolerance_mm should change the surface DSC value
    when the configured tolerance is the discriminating factor."""
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b = _cube((30, 30, 30), (10, 10, 12), 10)  # +2 voxel x-shift, 1 mm spacing

    tight = evaluate_masks(a, b, spacing_xyz=(1.0, 1.0, 1.0), surface_dsc_tolerance_mm=1.0)
    loose = evaluate_masks(a, b, spacing_xyz=(1.0, 1.0, 1.0), surface_dsc_tolerance_mm=3.0)

    assert tight.surface_dice is not None and loose.surface_dice is not None
    assert loose.surface_dice >= tight.surface_dice
    assert loose.surface_dice == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_masks_with_thresholds
# ---------------------------------------------------------------------------


def test_thresholded_pass_with_custom_dict() -> None:
    """Half-overlap fails the shipped Dice threshold but passes a 0.4 floor."""
    a = _cube((20, 20, 20), (5, 5, 0), 10)
    b = _cube((20, 20, 20), (5, 5, 5), 10)

    rec = evaluate_masks_with_thresholds(
        a,
        b,
        spacing_xyz=(1.0, 1.0, 1.0),
        thresholds={"dice": 0.4, "surface_dice_1mm": 0.0, "hd95_mm": 999.0,
                    "msd_mm": 999.0, "volume_rel_err": 1.0},
    )
    assert isinstance(rec, ResultRecord)
    assert rec.status == Status.PASS
    assert rec.violations == []
    assert rec.metrics["dice"] == pytest.approx(0.5)
    assert rec.thresholds["dice"] == 0.4


def test_thresholded_fail_populates_violations() -> None:
    """Half-overlap fails Dice ≥ 0.95 (the shipped default)."""
    a = _cube((20, 20, 20), (5, 5, 0), 10)
    b = _cube((20, 20, 20), (5, 5, 5), 10)

    rec = evaluate_masks_with_thresholds(a, b, spacing_xyz=(1.0, 1.0, 1.0))
    assert rec.status == Status.FAIL
    assert any("dice" in v for v in rec.violations)


def test_thresholded_partial_dict_falls_back_to_defaults() -> None:
    """A partial dict shallow-merges over the shipped defaults; missing keys
    are filled in rather than raising."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)

    rec = evaluate_masks_with_thresholds(
        a, a, spacing_xyz=(1.0, 1.0, 1.0), thresholds={"dice": 0.999}
    )
    assert rec.status == Status.PASS  # identical arrays; everything passes
    cfg_defaults = load_config().defaults
    # Filled-in keys come from the shipped defaults.
    assert rec.thresholds["hd95_mm"] == cfg_defaults["hd95_mm"]
    assert rec.thresholds["msd_mm"] == cfg_defaults["msd_mm"]
    assert rec.thresholds["dice"] == 0.999


def test_thresholded_uses_roi_specific_thresholds() -> None:
    """Asking for ``roi='sphere'`` should pull the per-ROI thresholds from
    the shipped YAML rather than the bare defaults."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    rec = evaluate_masks_with_thresholds(a, a, spacing_xyz=(1.0, 1.0, 1.0), roi="sphere")
    cfg = load_config()
    assert rec.thresholds == cfg.thresholds_for("sphere").as_dict()


def test_thresholded_unknown_roi_falls_back_to_defaults() -> None:
    """An unknown ROI name (not in CONFORMANCE_ROIS) uses bare defaults."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    rec = evaluate_masks_with_thresholds(a, a, spacing_xyz=(1.0, 1.0, 1.0), roi="not_a_roi")
    assert rec.thresholds == load_config().defaults


def test_thresholded_thresholds_object_argument_passes_through() -> None:
    """Passing a Thresholds instance directly bypasses any merging."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    custom = Thresholds(
        dice=0.5, surface_dice_1mm=0.5, hd95_mm=100.0, msd_mm=100.0, volume_rel_err=1.0
    )
    rec = evaluate_masks_with_thresholds(
        a, a, spacing_xyz=(1.0, 1.0, 1.0), thresholds=custom
    )
    assert rec.thresholds == custom.as_dict()
    assert rec.status == Status.PASS


def test_thresholded_geometry_mismatch_returns_record(tmp_path: Path) -> None:
    """Unlike ``evaluate_masks`` (which raises), the thresholded variant
    returns a ResultRecord with Status.GEOMETRY_MISMATCH — same shape as
    ``evaluate_one`` for direct interop with existing reporting code."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    pred = _write_nifti(a, tmp_path / "p.nii.gz", origin=(0.0, 0.0, 0.0))
    gt = _write_nifti(a, tmp_path / "g.nii.gz", origin=(1.0, 0.0, 0.0))

    rec = evaluate_masks_with_thresholds(pred, gt)
    assert rec.status == Status.GEOMETRY_MISMATCH
    assert rec.geometry_diagnostic is not None
    assert "origin" in rec.geometry_diagnostic
