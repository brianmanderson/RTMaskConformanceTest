"""Closed-shape metrics (vendored subset of rtmask_validation.benchmark.metrics).

Includes Dice, volume metrics, and SimpleITK-backed Surface DSC / Hausdorff /
mean surface distance. Point/curve/arc-length metrics are not vendored — the
v0.1 conformance suite covers closed-planar primitives only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


def _to_sitk_binary(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    arr = (mask > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xyz)
    return img


def _binary_contour(image: sitk.Image) -> sitk.Image:
    f = sitk.BinaryContourImageFilter()
    f.SetForegroundValue(1)
    f.SetBackgroundValue(0)
    f.SetFullyConnected(False)
    return f.Execute(image)


@dataclass
class VolumeMetrics:
    tool_volume_mm3: float
    reference_volume_mm3: float
    abs_error_mm3: float
    rel_error_pct: float | None
    tool_voxel_count: int
    reference_voxel_count: int


def volume_metrics(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_volume_mm3: float,
) -> VolumeMetrics:
    if tool_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: tool={tool_mask.shape}, ref={reference_mask.shape}."
        )
    tool_count = int(np.count_nonzero(tool_mask))
    ref_count = int(np.count_nonzero(reference_mask))
    tool_vol = tool_count * voxel_volume_mm3
    ref_vol = ref_count * voxel_volume_mm3
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


def binary_dsc(tool_mask: np.ndarray, reference_mask: np.ndarray) -> float:
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


@dataclass
class SurfaceMetricsBundle:
    surface_dsc: float | None
    hausdorff_mm: float | None
    hausdorff95_mm: float | None
    mean_surface_distance_mm: float | None


DEFAULT_SURFACE_DSC_TOLERANCE_MM: float = 1.0


def all_surface_metrics(
    tool_mask: np.ndarray,
    reference_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    tolerance_mm: float = DEFAULT_SURFACE_DSC_TOLERANCE_MM,
) -> SurfaceMetricsBundle:
    """Compute Surface DSC (Nikolov-style, tolerance-bounded), Hausdorff (full + 95),
    and mean surface distance via SimpleITK signed Maurer distance maps.
    """
    if tool_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: tool={tool_mask.shape}, ref={reference_mask.shape}."
        )
    if tolerance_mm < 0:
        raise ValueError(f"tolerance_mm must be >= 0; got {tolerance_mm}.")

    tool_img = _to_sitk_binary(tool_mask, voxel_size_mm)
    ref_img = _to_sitk_binary(reference_mask, voxel_size_mm)

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

    dt = sitk.SignedMaurerDistanceMapImageFilter()
    dt.SetUseImageSpacing(True)
    dt.SetSquaredDistance(False)
    dt.SetInsideIsPositive(False)
    dist_to_tool = dt.Execute(tool_contour)
    dist_to_ref = dt.Execute(ref_contour)
    dist_to_tool_arr = sitk.GetArrayViewFromImage(dist_to_tool)
    dist_to_ref_arr = sitk.GetArrayViewFromImage(dist_to_ref)

    a_to_b = np.abs(dist_to_ref_arr[tool_contour_arr > 0])
    b_to_a = np.abs(dist_to_tool_arr[ref_contour_arr > 0])

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
