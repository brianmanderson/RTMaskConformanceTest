"""End-to-end offset / overlap validation tests.

These tests build a real conformance fixture, write predictions that are the
ideal ground truth perturbed by a known transformation (translation,
erosion), and assert on the *quantitative* metrics — not just PASS/FAIL
status. A regression in :func:`binary_dsc` or :func:`all_surface_metrics`
that survives ``test_negative`` because it still flips status correctly
will trip these tests on the numeric value.

Skipped under ``RTMASK_CONFORMANCE_SKIP_E2E=1`` (matches the rest of the
e2e suite — fixture generation is the slow part, not the assertions).

The fixture is generated once per module via a session-scoped fixture and
shared across the four perturbation tests below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

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
    """Write the ideal rasterised mask for every conformance primitive."""
    for primitive in build_conformance_primitives():
        fractions = partial_volume_mask(
            primitive, geometry, n_quadrature=options.n_quadrature
        )
        binary = binary_threshold(fractions, threshold=0.5)
        img = sitk.GetImageFromArray(binary)
        img.SetSpacing(geometry.spacing)
        img.SetOrigin(geometry.origin)
        img.SetDirection(geometry.direction)
        sitk.WriteImage(img, str(predictions_dir / f"{primitive.name}.nii.gz"))


def _shifted_predictions(
    src_dir: Path, dst_dir: Path, *, shift_voxels: int, axis: int
) -> None:
    """Copy each prediction with ``np.roll`` along ``axis`` (numpy array axis,
    so axis=2 is the SimpleITK x dimension)."""
    for src in src_dir.glob("*.nii.gz"):
        img = sitk.ReadImage(str(src))
        arr = sitk.GetArrayFromImage(img)
        rolled = np.roll(arr, shift=shift_voxels, axis=axis)
        # Zero out the wrapped slab so we get a true translation, not a wrap.
        slicer = [slice(None)] * 3
        if shift_voxels > 0:
            slicer[axis] = slice(0, shift_voxels)
        else:
            slicer[axis] = slice(shift_voxels, None)
        rolled[tuple(slicer)] = 0
        new = sitk.GetImageFromArray(rolled)
        new.CopyInformation(img)
        sitk.WriteImage(new, str(dst_dir / src.name))


def _eroded_predictions(src_dir: Path, dst_dir: Path, *, iterations: int = 1) -> None:
    for src in src_dir.glob("*.nii.gz"):
        img = sitk.ReadImage(str(src))
        arr = sitk.GetArrayFromImage(img).astype(bool)
        eroded = binary_erosion(arr, iterations=iterations).astype(np.uint8)
        new = sitk.GetImageFromArray(eroded)
        new.CopyInformation(img)
        sitk.WriteImage(new, str(dst_dir / src.name))


@pytest.fixture(scope="module")
def fixture_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate one fixture and the four perturbed prediction sets.

    Module-scoped so the slow ``generate_fixture`` runs once even though
    four tests consume it.
    """
    root = tmp_path_factory.mktemp("e2e_offset_overlap")
    fixture = root / "fixture"
    options = GenerateOptions(
        voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2
    )
    generate_fixture(fixture, options=options)
    gt_dir = fixture / "groundtruth"

    sample = sitk.ReadImage(str(gt_dir / f"{CONFORMANCE_ROIS[0]}.nii.gz"))
    geometry = sitk_image_to_geometry(sample)

    ideal = root / "ideal"
    ideal.mkdir()
    _ideal_predictions(ideal, geometry, options)

    shift1 = root / "shift1"
    shift1.mkdir()
    _shifted_predictions(ideal, shift1, shift_voxels=1, axis=2)

    shift3 = root / "shift3"
    shift3.mkdir()
    _shifted_predictions(ideal, shift3, shift_voxels=3, axis=2)

    eroded = root / "eroded"
    eroded.mkdir()
    _eroded_predictions(ideal, eroded, iterations=1)

    return {
        "gt": gt_dir,
        "ideal": ideal,
        "shift1": shift1,
        "shift3": shift3,
        "eroded": eroded,
    }


def _metrics_by_roi(report) -> dict[str, dict[str, float | None]]:
    return {r.roi: r.metrics for r in report.results}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def test_baseline_ideal_predictions_score_perfect(fixture_dirs: dict[str, Path]) -> None:
    """Sanity: GT-identical predictions → every ROI Dice ≥ 0.999."""
    report = verify_predictions(fixture_dirs["ideal"], fixture_dirs["gt"], load_config())
    metrics = _metrics_by_roi(report)
    for roi in CONFORMANCE_ROIS:
        dice = metrics[roi]["dice"]
        assert dice is not None and dice >= 0.999, (roi, dice)
        hd95 = metrics[roi]["hd95_mm"]
        assert hd95 is not None and hd95 <= 1.0, (roi, hd95)
    assert report.summary["passed"] == report.summary["total"]


def test_one_voxel_shift_produces_nontrivial_hd95(fixture_dirs: dict[str, Path]) -> None:
    """1-voxel shift along x at 1 mm spacing → HD95 should land near 1.0 mm.

    Voxelisation noise can push HD95 a fraction either side of 1.0; we
    bracket it generously to guard against regressions while staying robust
    to per-shape geometry quirks.
    """
    report = verify_predictions(fixture_dirs["shift1"], fixture_dirs["gt"], load_config())
    metrics = _metrics_by_roi(report)
    for roi in CONFORMANCE_ROIS:
        hd95 = metrics[roi]["hd95_mm"]
        assert hd95 is not None
        # 1-voxel translation at 1 mm spacing — HD95 sits near 1.0 mm but
        # discrete-cube edge effects can push it up to ~2 mm on torus/straw.
        assert 0.5 <= hd95 <= 3.0, (roi, hd95)

        dice = metrics[roi]["dice"]
        assert dice is not None and dice < 1.0, (roi, dice)


def test_dice_degrades_monotonically_with_larger_shift(
    fixture_dirs: dict[str, Path],
) -> None:
    """Per ROI, Dice(3-voxel shift) < Dice(1-voxel shift) < Dice(ideal).

    A directional regression in ``binary_dsc`` (e.g., swapped numerator and
    denominator counts) would either invert this ordering or collapse it.
    """
    cfg = load_config()
    ideal = _metrics_by_roi(verify_predictions(fixture_dirs["ideal"], fixture_dirs["gt"], cfg))
    s1 = _metrics_by_roi(verify_predictions(fixture_dirs["shift1"], fixture_dirs["gt"], cfg))
    s3 = _metrics_by_roi(verify_predictions(fixture_dirs["shift3"], fixture_dirs["gt"], cfg))

    for roi in CONFORMANCE_ROIS:
        d_ideal, d1, d3 = ideal[roi]["dice"], s1[roi]["dice"], s3[roi]["dice"]
        assert d_ideal is not None and d1 is not None and d3 is not None
        assert d3 < d1 < d_ideal, (roi, d_ideal, d1, d3)

        h1, h3 = s1[roi]["hd95_mm"], s3[roi]["hd95_mm"]
        assert h1 is not None and h3 is not None
        assert h3 > h1, (roi, h1, h3)


def test_three_voxel_shift_fails_conformance(fixture_dirs: dict[str, Path]) -> None:
    """A 3-voxel shift is well outside the shipped tolerance for every ROI."""
    report = verify_predictions(fixture_dirs["shift3"], fixture_dirs["gt"], load_config())
    failed_rois = [r.roi for r in report.results if r.status == Status.FAIL]
    # All seven primitives should fail at this shift (tolerances vary, but
    # 3 mm exceeds the most lenient hd95_mm threshold of 2.5 mm).
    assert set(failed_rois) == set(CONFORMANCE_ROIS), failed_rois


def test_one_voxel_erosion_drives_volume_underreport(
    fixture_dirs: dict[str, Path],
) -> None:
    """Eroding by 1 voxel removes a thin surface shell → tool volume < GT volume.

    Asserts ``volume_rel_err > 0`` and ``tool_volume_mm3 < reference_volume_mm3``
    for every ROI. The relative loss is dominated by surface-area / volume,
    which depends on absolute shape dimensions as well as hollowness — so we
    check direction (under-report) rather than ordering between shapes.
    """
    report = verify_predictions(fixture_dirs["eroded"], fixture_dirs["gt"], load_config())
    metrics = _metrics_by_roi(report)

    for roi in CONFORMANCE_ROIS:
        rel = metrics[roi]["volume_rel_err"]
        tool_vol = metrics[roi]["tool_volume_mm3"]
        ref_vol = metrics[roi]["reference_volume_mm3"]
        assert rel is not None and tool_vol is not None and ref_vol is not None
        assert rel > 0.0, (roi, rel)
        assert tool_vol < ref_vol, (roi, tool_vol, ref_vol)
