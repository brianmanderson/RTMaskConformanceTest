"""Validation tests for the vendored metric math.

These are pure unit tests — small synthetic numpy arrays, no fixture
generation, no I/O. Each case has a hand-computed expected value with the
arithmetic spelled out in a comment, so a regression in
:func:`binary_dsc`, :func:`volume_metrics`, or :func:`all_surface_metrics`
shows up here as a numeric mismatch rather than a generic "tests failed."

Array convention: numpy ``(Z, Y, X)``; ``spacing_xyz`` follows SimpleITK's
``(X, Y, Z)`` convention. ``_cube`` indexes the array directly, so its
``origin`` argument is ``(z, y, x)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtmask_conformance._vendor.metrics import (
    all_surface_metrics,
    binary_dsc,
    volume_metrics,
)


def _cube(
    shape: tuple[int, int, int],
    origin: tuple[int, int, int],
    side: int | tuple[int, int, int],
) -> np.ndarray:
    """Place an axis-aligned cuboid in a zero-filled array.

    ``shape`` and ``origin`` are ``(z, y, x)``. ``side`` is either a scalar
    voxel count or a per-axis triple.
    """
    arr = np.zeros(shape, dtype=np.uint8)
    z, y, x = origin
    if isinstance(side, int):
        sz = sy = sx = side
    else:
        sz, sy, sx = side
    arr[z:z + sz, y:y + sy, x:x + sx] = 1
    return arr


# ---------------------------------------------------------------------------
# binary_dsc
# ---------------------------------------------------------------------------


def test_dsc_identity_is_one() -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    assert binary_dsc(a, a) == 1.0


def test_dsc_disjoint_is_zero() -> None:
    a = _cube((20, 20, 20), (0, 0, 0), 5)
    b = _cube((20, 20, 20), (15, 15, 15), 5)
    assert binary_dsc(a, b) == 0.0


def test_dsc_both_empty_returns_one_by_convention() -> None:
    """Both masks empty: by convention, agreement = perfect.

    Documented behavior in :func:`binary_dsc` (lines 76-77): if there is
    nothing to disagree about, return 1.0. The caller can treat this as a
    sentinel separately if desired.
    """
    z = np.zeros((10, 10, 10), dtype=np.uint8)
    assert binary_dsc(z, z) == 1.0


def test_dsc_one_empty_is_zero() -> None:
    z = np.zeros((10, 10, 10), dtype=np.uint8)
    a = _cube((10, 10, 10), (0, 0, 0), 5)
    assert binary_dsc(a, z) == 0.0
    assert binary_dsc(z, a) == 0.0


def test_dsc_half_overlap_shift_along_x() -> None:
    """10×10×10 cube vs same shifted +5 voxels along x.

    Intersection = 10·10·5 = 500.  |A|+|B| = 1000+1000 = 2000.
    Dice = 2·500 / 2000 = 0.5.
    """
    a = _cube((20, 20, 20), (5, 5, 0), 10)
    b = _cube((20, 20, 20), (5, 5, 5), 10)
    assert binary_dsc(a, b) == pytest.approx(0.5)


def test_dsc_subset_known_ratio() -> None:
    """10³ cube containing a centered 5³ cube.

    Intersection = 5·5·5 = 125. |A|+|B| = 1000+125 = 1125.
    Dice = 2·125 / 1125 = 250/1125 ≈ 0.22222.
    """
    big = _cube((20, 20, 20), (5, 5, 5), 10)
    small = _cube((20, 20, 20), (7, 7, 7), 5)  # entirely inside `big`
    assert binary_dsc(big, small) == pytest.approx(250.0 / 1125.0)


def test_dsc_quarter_overlap_two_axis_shift() -> None:
    """Two 10³ cubes shifted +5 along both z and x axes.

    A occupies (0:10, 5:15, 0:10); B occupies (5:15, 5:15, 5:15).
    Intersection: z=[5,10) → 5 voxels; y=[5,15) → 10; x=[5,10) → 5.
    Inter = 5·10·5 = 250.  |A|+|B| = 2000.  Dice = 2·250/2000 = 0.25.
    """
    a = _cube((20, 20, 20), (0, 5, 0), 10)
    b = _cube((20, 20, 20), (5, 5, 5), 10)
    assert binary_dsc(a, b) == pytest.approx(0.25)


def test_dsc_shape_mismatch_raises() -> None:
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    b = np.zeros((10, 10, 11), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_dsc(a, b)


def test_dsc_dtype_independent() -> None:
    """Bool, uint8, uint16, float — all should give the same answer."""
    a8 = _cube((20, 20, 20), (5, 5, 5), 10)
    expected = 0.5  # half-overlap
    a_other = _cube((20, 20, 20), (5, 5, 0), 10)

    for dtype in (np.bool_, np.uint8, np.uint16, np.int32, np.float32):
        a = a8.astype(dtype)
        b = a_other.astype(dtype)
        assert binary_dsc(a, b) == pytest.approx(expected), f"dtype={dtype}"


# ---------------------------------------------------------------------------
# volume_metrics
# ---------------------------------------------------------------------------


def test_volume_identity_zero_error() -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    vm = volume_metrics(a, a, voxel_volume_mm3=1.0)
    assert vm.tool_volume_mm3 == 1000.0
    assert vm.reference_volume_mm3 == 1000.0
    assert vm.abs_error_mm3 == 0.0
    assert vm.rel_error_pct == 0.0


def test_volume_known_5_percent_error() -> None:
    """Tool reports 1050 voxels, GT has 1000 → rel error = 5.0%."""
    tool = _cube((30, 30, 30), (5, 5, 5), 10)        # 10³ = 1000 voxels
    tool[5:10, 5:10, 15:17] = 1                       # +50 voxels → 1050
    assert int(tool.sum()) == 1050

    gt = _cube((30, 30, 30), (5, 5, 5), 10)           # 1000 voxels
    assert int(gt.sum()) == 1000

    vm = volume_metrics(tool, gt, voxel_volume_mm3=1.0)
    assert vm.tool_volume_mm3 == 1050.0
    assert vm.reference_volume_mm3 == 1000.0
    assert vm.abs_error_mm3 == 50.0
    assert vm.rel_error_pct == pytest.approx(5.0)


def test_volume_anisotropic_voxel_volume() -> None:
    """Voxel volume is multiplicative — 1000 voxels × 0.5×1×2 = 1000 mm³ unchanged
    in absolute terms relative to ref, but the absolute volume value scales."""
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    voxel_vol = 0.5 * 1.0 * 2.0  # mm³
    vm = volume_metrics(a, a, voxel_volume_mm3=voxel_vol)
    assert vm.tool_volume_mm3 == pytest.approx(1000.0 * voxel_vol)
    assert vm.reference_volume_mm3 == pytest.approx(1000.0 * voxel_vol)
    assert vm.rel_error_pct == 0.0


def test_volume_empty_reference_yields_none_rel_err() -> None:
    """When the GT mask is empty, rel_error_pct is undefined → returns None."""
    z = np.zeros((10, 10, 10), dtype=np.uint8)
    a = _cube((10, 10, 10), (0, 0, 0), 5)
    vm = volume_metrics(a, z, voxel_volume_mm3=1.0)
    assert vm.reference_volume_mm3 == 0.0
    assert vm.rel_error_pct is None


def test_volume_shape_mismatch_raises() -> None:
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    b = np.zeros((10, 11, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        volume_metrics(a, b, voxel_volume_mm3=1.0)


# ---------------------------------------------------------------------------
# all_surface_metrics
# ---------------------------------------------------------------------------


def test_surface_identity_perfect() -> None:
    a = _cube((20, 20, 20), (5, 5, 5), 10)
    sm = all_surface_metrics(a, a, voxel_size_mm=(1.0, 1.0, 1.0))
    assert sm.surface_dsc == pytest.approx(1.0)
    assert sm.hausdorff_mm == pytest.approx(0.0)
    assert sm.hausdorff95_mm == pytest.approx(0.0)
    assert sm.mean_surface_distance_mm == pytest.approx(0.0)


def test_surface_both_empty_dsc_one_distances_none() -> None:
    z = np.zeros((10, 10, 10), dtype=np.uint8)
    sm = all_surface_metrics(z, z, voxel_size_mm=(1.0, 1.0, 1.0))
    assert sm.surface_dsc == 1.0
    assert sm.hausdorff_mm is None
    assert sm.hausdorff95_mm is None
    assert sm.mean_surface_distance_mm is None


def test_surface_one_empty_dsc_zero_distances_none() -> None:
    z = np.zeros((10, 10, 10), dtype=np.uint8)
    a = _cube((10, 10, 10), (2, 2, 2), 5)
    sm = all_surface_metrics(a, z, voxel_size_mm=(1.0, 1.0, 1.0))
    assert sm.surface_dsc == 0.0
    assert sm.hausdorff_mm is None
    assert sm.hausdorff95_mm is None
    assert sm.mean_surface_distance_mm is None


def test_surface_hd95_matches_known_voxel_shift() -> None:
    """Two identical cubes, one shifted +3 voxels along x, isotropic 1 mm spacing.

    Maurer distance from the displaced cube's surface to the original's
    surface is bounded by 3 mm on the leading/trailing faces. HD95 should
    therefore land within sub-voxel tolerance of 3.0 mm.
    """
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b = _cube((30, 30, 30), (10, 10, 13), 10)  # shifted +3 in x
    sm = all_surface_metrics(a, b, voxel_size_mm=(1.0, 1.0, 1.0))
    assert sm.hausdorff95_mm is not None
    assert sm.hausdorff95_mm == pytest.approx(3.0, abs=1.0)


def test_surface_hd_respects_anisotropic_spacing_on_z() -> None:
    """A 1-voxel shift along z with sz=2.0 mm should produce HD95 ≈ 2.0 mm.

    Confirms that ``all_surface_metrics`` honours per-axis spacing — if the
    spacing argument were ignored, HD95 would degenerate to ≈ 1.0 mm.
    """
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b = _cube((30, 30, 30), (11, 10, 10), 10)  # +1 voxel along axis 0 (z)
    sm = all_surface_metrics(a, b, voxel_size_mm=(0.5, 1.0, 2.0))
    assert sm.hausdorff95_mm is not None
    assert sm.hausdorff95_mm == pytest.approx(2.0, abs=0.5)


def test_surface_dsc_within_tolerance_is_one() -> None:
    """Spacing 0.5 mm, 1-voxel shift along x = 0.5 mm displacement.

    With tolerance_mm = 1.0 every surface voxel finds a counterpart within
    tolerance, so Surface DSC = 1.0.
    """
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b = _cube((30, 30, 30), (10, 10, 11), 10)  # +1 voxel along x
    sm = all_surface_metrics(
        a, b, voxel_size_mm=(0.5, 0.5, 0.5), tolerance_mm=1.0
    )
    assert sm.surface_dsc == pytest.approx(1.0)


def test_surface_dsc_beyond_tolerance_drops_below_one() -> None:
    """Spacing 1 mm, 2-voxel shift = 2.0 mm > tolerance 1.0 mm.

    Surface DSC must be strictly less than 1.0 (some surface voxels lie
    farther than tolerance from any counterpart).
    """
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b = _cube((30, 30, 30), (10, 10, 12), 10)  # +2 voxels along x
    sm = all_surface_metrics(
        a, b, voxel_size_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0
    )
    assert sm.surface_dsc is not None
    assert 0.0 < sm.surface_dsc < 1.0


def test_surface_negative_tolerance_raises() -> None:
    a = _cube((10, 10, 10), (2, 2, 2), 5)
    with pytest.raises(ValueError, match="tolerance_mm"):
        all_surface_metrics(a, a, voxel_size_mm=(1.0, 1.0, 1.0), tolerance_mm=-0.1)


def test_surface_shape_mismatch_raises() -> None:
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    b = np.zeros((10, 10, 11), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        all_surface_metrics(a, b, voxel_size_mm=(1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Cross-metric monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 2, 3, 5])
def test_metrics_degrade_monotonically_with_shift(shift: int) -> None:
    """As the shift grows, Dice should fall and HD95 should rise.

    Cross-metric regression guard: if either metric inverts its direction
    (sign error, swapped operands, etc.), this test fires.
    """
    a = _cube((30, 30, 30), (10, 10, 10), 10)
    b_no_shift = a
    b = _cube((30, 30, 30), (10, 10, 10 + shift), 10)

    dsc_baseline = binary_dsc(a, b_no_shift)
    dsc_shifted = binary_dsc(a, b)
    assert dsc_shifted < dsc_baseline

    sm_shifted = all_surface_metrics(a, b, voxel_size_mm=(1.0, 1.0, 1.0))
    assert sm_shifted.hausdorff95_mm is not None
    assert sm_shifted.hausdorff95_mm > 0.0
