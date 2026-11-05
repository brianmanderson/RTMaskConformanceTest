"""Pure-numpy metric implementations used by the benchmark harness.

All metrics consume binary uint8 masks (or float arrays where applicable) plus
the voxel-volume scalar -- no DICOM / SimpleITK coupling. Surface metrics
(Hausdorff, mean surface distance) require ``voxel_size_mm`` because their
result is reported in physical millimeters.

Per :doc:`../../../VerificationMethodology.md` Section 5:

* §5.1 Volume metrics                    -> :func:`volume_metrics`
* §5.2 Surface metrics                   -> :func:`surface_dsc`,
                                             :func:`hausdorff_distance`,
                                             :func:`mean_surface_distance`
* §5.3 Length / arc-length metrics       -> :func:`curve_metrics`
* §5.4 Point metrics                     -> :func:`point_metrics`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from .common.coords import Geometry


# -----------------------------------------------------------------
#  SimpleITK helpers shared by the surface metrics
# -----------------------------------------------------------------


def _to_sitk_binary(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    """Wrap a numpy ``(z, y, x)`` mask as a uint8 SimpleITK image with the given spacing."""
    arr = (mask > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    return img


def _binary_contour(image: sitk.Image) -> sitk.Image:
    """Extract the boundary voxels of a binary mask using SimpleITK's
    ``BinaryContourImageFilter`` (faster + more accurate than scipy's
    ``binary_erosion`` followed by an XOR).
    """
    f = sitk.BinaryContourImageFilter()
    f.SetForegroundValue(1)
    f.SetBackgroundValue(0)
    f.SetFullyConnected(False)
    return f.Execute(image)


# -----------------------------------------------------------------
#  Volume metrics (Section 5.1)
# -----------------------------------------------------------------


@dataclass
class VolumeMetrics:
    """Volume-error report for one (tool, primitive) pair."""

    tool_volume_mm3: float
    reference_volume_mm3: float
    abs_error_mm3: float
    rel_error_pct: float | None    # None if reference_volume_mm3 == 0
    tool_voxel_count: int
    reference_voxel_count: int


def volume_metrics(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_volume_mm3: float,
    reference_volume_mm3: float | None = None,
) -> VolumeMetrics:
    """Compute absolute and relative volume error.

    ``tool_mask`` is the mask the comparator produced; ``reference_mask`` is the
    analytical ground-truth mask. Both must be the same shape and binary
    (any non-zero is treated as foreground).

    ``reference_volume_mm3`` is the volume the error is measured against. Pass
    the primitive's closed-form volume (or, failing that, the partial-volume
    sum) so the denominator is analytical truth. When it is omitted the
    reference falls back to counting voxels in ``reference_mask``, i.e. the
    binarized PV>=0.5 mask -- a discretization of truth, not truth itself, and
    on this registry it sits up to 0.48% away from the closed form (the straw),
    which is the same order as the tool errors being measured. That fallback
    exists only for callers with no analytical volume available; the synthetic
    benchmark always passes one. See ``build_validation_dataset`` for why the
    partial-volume estimate is the intended reference.

    ``reference_voxel_count`` continues to report the binarized mask's voxel
    count regardless, since DSC and the surface metrics are computed against
    that mask.
    """
    if tool_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: tool={tool_mask.shape}, ref={reference_mask.shape}."
        )
    tool_count = int(np.count_nonzero(tool_mask))
    ref_count = int(np.count_nonzero(reference_mask))
    tool_vol = tool_count * voxel_volume_mm3
    ref_vol = (
        float(reference_volume_mm3)
        if reference_volume_mm3 is not None
        else ref_count * voxel_volume_mm3
    )
    abs_err = abs(tool_vol - ref_vol)
    rel_err = (100.0 * abs_err / ref_vol) if ref_vol > 0 else None
    return VolumeMetrics(
        tool_volume_mm3=tool_vol,
        reference_volume_mm3=ref_vol,
        abs_error_mm3=abs_err,
        rel_error_pct=rel_err,
        tool_voxel_count=tool_count,
        reference_voxel_count=ref_count,
    )


# -----------------------------------------------------------------
#  Binary Dice Similarity Coefficient (volume-level, not surface-level)
# -----------------------------------------------------------------


def binary_dsc(tool_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    """Sorensen-Dice over the *volume* of two binary masks.

    DSC = 2 |A intersect B| / (|A| + |B|).

    Returns 1.0 if both masks are empty (a zero-volume primitive correctly
    rasterized as empty by the tool); 0.0 if exactly one is empty.
    """
    if tool_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: tool={tool_mask.shape}, ref={reference_mask.shape}."
        )
    a = tool_mask.astype(bool)
    b = reference_mask.astype(bool)
    inter = int(np.count_nonzero(a & b))
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    if a_count == 0 and b_count == 0:
        return 1.0
    if a_count + b_count == 0:
        return 0.0
    return float(2.0 * inter / (a_count + b_count))


# -----------------------------------------------------------------
#  Surface metrics (Section 5.2) -- SimpleITK-backed
#
#  Surface DSC, Hausdorff (full + 95th percentile), and Mean Surface Distance
#  share a single pair of signed Maurer distance maps computed once per
#  invocation. SimpleITK's filters are faster than the equivalent scipy
#  calls (no measured ratio is recorded) because they use the C++ Maurer's
#  algorithm and avoid the ``~mask`` inversion + double-pass overhead of
#  ``distance_transform_edt``.
# -----------------------------------------------------------------


@dataclass
class SurfaceMetricsBundle:
    surface_dsc: float | None
    hausdorff_mm: float | None
    hausdorff95_mm: float | None
    mean_surface_distance_mm: float | None


#: Default tolerance threshold for the Nikolov-style surface DSC, in mm.
#:
#: 0.5 mm -- half a voxel on the 1-mm-isotropic synthetic grid.
#:
#: This was 1.5 mm, justified on the grounds that it forgave the one-voxel
#: boundary shell between the two vertex-handling clusters while staying
#: "discriminating on small structures (Esophagus, SpinalCord on LCTSC; the
#: diagonal_needle on the synthetic dataset)". Both of
#: those examples were fictitious: `surface_dsc` is NaN for every OPEN_NONPLANAR
#: and POINT row, so the diagonal needle has no surface DSC, and the metric is
#: not computed on LCTSC at all. What 1.5 mm actually did was saturate -- it
#: returns exactly 1.0000 for every tool on every CLOSED_PLANAR primitive
#: except plastimatch, so the column carried no information.
#:
#: That is inherent to the quantity being measured: the difference between the
#: boundary rules is a one-voxel shell, so any tolerance of a voxel or more
#: forgives it by construction. Measured sweep over four tools on cube, sphere,
#: torus and straw:
#:
#:   tolerance   cluster B (dcmrtstruct2nii)   cluster A (platipy/dicomrttool/rt-utils)
#:   1.5 mm      1.0000                        1.0000          (no discrimination)
#:   1.0 mm      1.0000                        0.9986-1.0000   (marginal)
#:   0.5 mm      0.9268-1.0000                 0.6057-0.7914   (separates)
#:
#: Report the sweep, not just the chosen value -- the tolerance-dependence is
#: itself evidence for this project's thesis that metric selection can hide a
#: real difference between tools.
#:
#: Note this is deliberately NOT the conformance gate's tolerance. The gate
#: asserts `surface_dice_1mm >= 0.95` as a pass/fail check that a mask is in
#: roughly the right place; this constant governs a reported comparison metric
#: and is set where tools actually differ. Keep the two distinct and say so
#: wherever the numbers appear together.
DEFAULT_SURFACE_DSC_TOLERANCE_MM: float = 0.5


def all_surface_metrics(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    tolerance_mm: float = DEFAULT_SURFACE_DSC_TOLERANCE_MM,
) -> SurfaceMetricsBundle:
    """Compute surface-DSC, Hausdorff (full + 95), and MSD via SimpleITK.

    Pipeline:

    1. ``BinaryContourImageFilter`` extracts the boundary voxels of each mask.
    2. ``SignedMaurerDistanceMapImageFilter`` builds a signed distance map
       from each contour (negative inside, positive outside) -- one per mask.
    3. The boundary-to-boundary distances we need are looked up by sampling
       each contour image's voxels in the *other* mask's distance map.

    Surface DSC is the **Nikolov-style tolerance-bounded** definition (cf.
    Nikolov et al. 2018, arXiv:1809.04430): a tool-surface voxel "matches"
    the reference if its nearest-reference-surface distance is <= ``tolerance_mm``,
    and likewise for reference voxels with respect to the tool. The metric
    is then ``(matched_A + matched_B) / (|S_A| + |S_B|)`` where ``S_A``,
    ``S_B`` are the two surface voxel sets. (No factor of 2 -- both matched
    counts already appear in the numerator.) An ``argument`` of ``0.0`` mm
    reduces to the legacy strict-voxel-overlap behavior.

    The two distance maps + two contour images are the entire heavy lift,
    and they're reused across all four returned metrics.
    """
    if tool_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: tool={tool_mask.shape}, ref={reference_mask.shape}."
        )
    if tolerance_mm < 0:
        raise ValueError(f"tolerance_mm must be >= 0; got {tolerance_mm}.")

    tool_img = _to_sitk_binary(tool_mask, voxel_size_mm)
    ref_img = _to_sitk_binary(reference_mask, voxel_size_mm)

    # Boundary extractions and counts.
    tool_contour = _binary_contour(tool_img)
    ref_contour = _binary_contour(ref_img)
    tool_contour_arr = sitk.GetArrayViewFromImage(tool_contour)
    ref_contour_arr = sitk.GetArrayViewFromImage(ref_contour)
    a_count = int(np.count_nonzero(tool_contour_arr))
    b_count = int(np.count_nonzero(ref_contour_arr))

    if a_count == 0 and b_count == 0:
        return SurfaceMetricsBundle(
            surface_dsc=1.0,
            hausdorff_mm=None,
            hausdorff95_mm=None,
            mean_surface_distance_mm=None,
        )
    if a_count == 0 or b_count == 0:
        return SurfaceMetricsBundle(
            surface_dsc=0.0,
            hausdorff_mm=None,
            hausdorff95_mm=None,
            mean_surface_distance_mm=None,
        )

    # Distance maps from each binary contour image. SimpleITK's Maurer
    # implementation is C++-backed and only walks the foreground -> background
    # boundary, so it's much faster than scipy's ``distance_transform_edt``
    # which would require a full ``~mask`` pass on a 512x512x200 grid.
    dt = sitk.SignedMaurerDistanceMapImageFilter()
    dt.SetUseImageSpacing(True)
    dt.SetSquaredDistance(False)
    # Treat anything > 0 as "inside the contour" (the source surface).
    dt.SetInsideIsPositive(False)  # outside positive, inside negative
    dist_to_tool = dt.Execute(tool_contour)
    dist_to_ref = dt.Execute(ref_contour)
    dist_to_tool_arr = sitk.GetArrayViewFromImage(dist_to_tool)
    dist_to_ref_arr = sitk.GetArrayViewFromImage(dist_to_ref)

    # Distances from each surface's voxels to the other surface (mm).
    # We take absolute values: the signed map is negative inside / positive
    # outside, but for surface-to-surface distance we want unsigned.
    a_to_b = np.abs(dist_to_ref_arr[tool_contour_arr > 0])
    b_to_a = np.abs(dist_to_tool_arr[ref_contour_arr > 0])

    # Nikolov tolerance-bounded surface DSC. With tolerance_mm = 0 this
    # reduces to strict voxel-overlap (rounding-tolerant only via the
    # underlying distance map's discretisation).
    matched_a = int((a_to_b <= tolerance_mm).sum())
    matched_b = int((b_to_a <= tolerance_mm).sum())
    sd = float((matched_a + matched_b) / (a_count + b_count))

    h_full = float(max(a_to_b.max(), b_to_a.max()))
    h_95 = float(max(np.percentile(a_to_b, 95.0), np.percentile(b_to_a, 95.0)))
    msd = float((a_to_b.sum() + b_to_a.sum()) / (a_count + b_count))

    return SurfaceMetricsBundle(
        surface_dsc=sd,
        hausdorff_mm=h_full,
        hausdorff95_mm=h_95,
        mean_surface_distance_mm=msd,
    )


# Standalone single-metric wrappers -- thin shims over the bundle for callers
# that only need one number. Keep the same signatures we shipped before so the
# benchmark driver doesn't need to change.


def surface_dsc(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = DEFAULT_SURFACE_DSC_TOLERANCE_MM,
) -> float:
    """Nikolov-style tolerance-bounded surface DSC.

    Thin shim over :func:`all_surface_metrics` that returns just the
    surface-DSC scalar. Callers that need Hausdorff or MSD should call
    ``all_surface_metrics`` directly to avoid recomputing the Maurer
    distance maps.
    """
    return all_surface_metrics(
        tool_mask, reference_mask, voxel_size_mm, tolerance_mm
    ).surface_dsc


def hausdorff_distance(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    percentile: float = 100.0,
) -> float | None:
    """Symmetric Hausdorff distance (full or percentile, in mm) — SimpleITK-backed."""
    bundle = all_surface_metrics(tool_mask, reference_mask, voxel_size_mm)
    if percentile >= 100.0:
        return bundle.hausdorff_mm
    if abs(percentile - 95.0) < 1e-6:
        return bundle.hausdorff95_mm
    raise ValueError(
        f"hausdorff_distance only exposes percentile=100 or 95 via the bundle; "
        f"got {percentile}. Use all_surface_metrics() and access dist arrays directly."
    )


def mean_surface_distance(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
) -> float | None:
    """Symmetric mean surface distance (mm) — SimpleITK-backed."""
    return all_surface_metrics(tool_mask, reference_mask, voxel_size_mm).mean_surface_distance_mm


