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


# -----------------------------------------------------------------
#  Helpers shared by point/curve metrics
# -----------------------------------------------------------------


def _marked_voxel_centers_mm(mask: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Return the physical (mm) coordinates of every non-zero voxel center.

    SimpleITK / numpy store volumes as ``(slices, rows, cols) = (z, y, x)``;
    ``geometry.index_to_physical`` expects ``(x, y, z)`` order, so we re-shuffle.
    """
    marked = np.argwhere(mask > 0)  # (M, 3) in (z, y, x) order
    if marked.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xyz_indices = marked[:, [2, 1, 0]].astype(np.float64)
    return geometry.index_to_physical(xyz_indices)


def _voxel_diagonal_mm(geometry: Geometry) -> float:
    sx, sy, sz = geometry.spacing
    return float(np.sqrt(sx * sx + sy * sy + sz * sz))


def densify_polyline(points: np.ndarray, max_step_mm: float = 0.3) -> np.ndarray:
    """Sub-sample a 3D polyline so consecutive samples are at most ``max_step_mm`` apart.

    Used to give curve-coverage metrics sub-voxel-density analytical samples
    regardless of how the parametric primitive sampled the curve when emitted.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    out = [pts[0]]
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i - 1]
        seg_len = float(np.linalg.norm(seg))
        if seg_len == 0:
            continue
        n = max(1, int(np.ceil(seg_len / max_step_mm)))
        for k in range(1, n + 1):
            out.append(pts[i - 1] + seg * (k / n))
    return np.asarray(out, dtype=np.float64)


# -----------------------------------------------------------------
#  Point metrics (Section 5.4)
# -----------------------------------------------------------------


@dataclass
class PointMetrics:
    """Per-ROI point-cloud comparison report."""

    n_points: int                              # how many analytical points there are
    n_marked_voxels: int                       # how many voxels the tool marked
    mean_point_displacement_mm: float | None   # average distance: analytical point -> nearest marked voxel
    max_point_displacement_mm: float | None
    recall: float | None                       # fraction of analytical points covered (within voxel diagonal)
    fp_rate: float | None                      # fraction of marked voxels not near any analytical point


def point_metrics(
    tool_mask: np.ndarray,
    analytical_points_mm: np.ndarray,
    geometry: Geometry,
) -> PointMetrics:
    """Score a tool's POINT rasterization.

    For each analytical point we measure the distance to the nearest non-zero
    voxel in ``tool_mask``. ``recall`` counts analytical points whose nearest
    voxel is within one voxel diagonal (i.e., the rasterization could plausibly
    represent it). ``fp_rate`` flags marked voxels with no nearby analytical
    point -- catches over-aggressive kernel marking or stray voxels.
    """
    pts = np.asarray(analytical_points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"analytical_points_mm must be (N>=1, 3); got {pts.shape}.")
    n_points = int(pts.shape[0])
    marked_xyz = _marked_voxel_centers_mm(tool_mask, geometry)
    n_marked = int(marked_xyz.shape[0])

    if n_marked == 0:
        return PointMetrics(
            n_points=n_points,
            n_marked_voxels=0,
            mean_point_displacement_mm=None,
            max_point_displacement_mm=None,
            recall=0.0,
            fp_rate=None,
        )

    voxel_tree = cKDTree(marked_xyz)
    dist_to_voxel, _ = voxel_tree.query(pts, k=1)

    point_tree = cKDTree(pts)
    dist_to_point, _ = point_tree.query(marked_xyz, k=1)

    diag = _voxel_diagonal_mm(geometry)
    return PointMetrics(
        n_points=n_points,
        n_marked_voxels=n_marked,
        mean_point_displacement_mm=float(dist_to_voxel.mean()),
        max_point_displacement_mm=float(dist_to_voxel.max()),
        recall=float((dist_to_voxel <= diag).mean()),
        fp_rate=float((dist_to_point > diag).mean()),
    )


# -----------------------------------------------------------------
#  Curve metrics (Section 5.3)
# -----------------------------------------------------------------


@dataclass
class CurveMetrics:
    """Per-ROI curve comparison report.

    Voxel-marking-vs-curve metrics, *not* skeleton arc-length metrics. Tools
    rasterize curves to thin chains of voxels via DDA / Bresenham 3D; we score
    how close those voxels are to the analytical curve. True arc-length error
    via skeletonization is deferred to a later slice.
    """

    n_curve_samples: int
    n_marked_voxels: int
    coverage_recall: float | None              # fraction of analytical samples within one voxel diagonal of a marked voxel
    mean_curve_distance_mm: float | None       # analytical -> nearest marked voxel
    max_curve_distance_mm: float | None
    voxel_to_curve_mean_mm: float | None       # marked voxel -> nearest analytical sample
    voxel_to_curve_max_mm: float | None
    endpoint_displacement_max_mm: float | None  # only set for OPEN_* curves


def curve_metrics(
    tool_mask: np.ndarray,
    analytical_curve_mm: np.ndarray,
    geometry: Geometry,
    endpoints_mm: np.ndarray | None = None,
) -> CurveMetrics:
    """Score a tool's curve rasterization against an analytical polyline.

    Parameters
    ----------
    tool_mask : np.ndarray
        The tool's rasterized voxel mask (z, y, x).
    analytical_curve_mm : np.ndarray
        ``(N, 3)`` densely-sampled physical coordinates along the analytical
        curve. Should be at least sub-voxel density (use :func:`densify_polyline`).
    geometry : Geometry
        Reference grid geometry.
    endpoints_mm : np.ndarray | None
        ``(K, 3)`` endpoint coordinates for OPEN_* curves, or ``None`` for
        closed curves. The endpoint-displacement metric is only computed when
        these are supplied.
    """
    curve = np.asarray(analytical_curve_mm, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
        raise ValueError(f"analytical_curve_mm must be (N>=2, 3); got {curve.shape}.")

    marked_xyz = _marked_voxel_centers_mm(tool_mask, geometry)
    n_marked = int(marked_xyz.shape[0])
    n_samples = int(curve.shape[0])
    diag = _voxel_diagonal_mm(geometry)

    if n_marked == 0:
        return CurveMetrics(
            n_curve_samples=n_samples,
            n_marked_voxels=0,
            coverage_recall=0.0,
            mean_curve_distance_mm=None,
            max_curve_distance_mm=None,
            voxel_to_curve_mean_mm=None,
            voxel_to_curve_max_mm=None,
            endpoint_displacement_max_mm=None,
        )

    voxel_tree = cKDTree(marked_xyz)
    dist_curve_to_voxel, _ = voxel_tree.query(curve, k=1)

    point_tree = cKDTree(curve)
    dist_voxel_to_curve, _ = point_tree.query(marked_xyz, k=1)

    endpoint_max = None
    if endpoints_mm is not None and len(endpoints_mm) > 0:
        endpoint_dist, _ = voxel_tree.query(np.asarray(endpoints_mm, dtype=np.float64), k=1)
        endpoint_max = float(endpoint_dist.max())

    return CurveMetrics(
        n_curve_samples=n_samples,
        n_marked_voxels=n_marked,
        coverage_recall=float((dist_curve_to_voxel <= diag).mean()),
        mean_curve_distance_mm=float(dist_curve_to_voxel.mean()),
        max_curve_distance_mm=float(dist_curve_to_voxel.max()),
        voxel_to_curve_mean_mm=float(dist_voxel_to_curve.mean()),
        voxel_to_curve_max_mm=float(dist_voxel_to_curve.max()),
        endpoint_displacement_max_mm=endpoint_max,
    )


# -----------------------------------------------------------------
#  Arc-length metric (Section 5.3 -- the headline curve metric)
#
#  The methodology document defines this as "percentage difference between
#  rasterized polyline length and analytical arc length". For our binary
#  voxel masks the rasterized "polyline length" is computed as:
#
#    1. Skeletonize the rasterized mask down to a 1-voxel-wide chain via 3D
#       morphological thinning (skimage.morphology.skeletonize_3d).
#    2. Build the 26-connectivity neighbor graph over the skeleton voxels.
#    3. Sum every unique edge length (Euclidean distance in physical mm
#       between adjacent voxel centers, accounting for image spacing).
#
#  For multi-component primitives (e.g. MultiNeedleBundle) every connected
#  skeleton component contributes; we report the total measured length and
#  compare it to the sum of analytical arc lengths.
# -----------------------------------------------------------------


@dataclass
class ArcLengthMetrics:
    n_skeleton_voxels: int
    raster_arc_length_mm: float | None
    analytical_arc_length_mm: float
    arc_length_error_mm: float | None
    arc_length_error_pct: float | None


def _skeleton_arc_length_mm(
    skeleton: np.ndarray, voxel_size_mm: tuple[float, float, float]
) -> tuple[float, int]:
    """Estimate the rasterized polyline length by summing edge lengths along the
    *longest simple path* through each connected component of the skeleton's
    26-neighbor graph.

    Why longest-path rather than "sum every unique edge":

    A naive sum-all-edges estimator over-counts by edges that branch off the
    main curve (for instance, a 1-voxel side-spur left behind after
    skeletonization, or extra edges where a 3D-DDA trace produces locally
    thicker patches at slope changes). It also penalizes rasterizers that use
    diagonal voxel steps because each sqrt(3) edge contributes more than
    three orthogonal edges of length 1 -- even when both rasterize the same
    underlying analytical line. Longest-path tracing avoids both biases:
    only the path between the two extremal voxels is summed, regardless of
    how many incidental short branches exist along the way.

    For multi-component primitives (e.g. ``MultiNeedleBundle``) every
    connected component contributes its own longest path; the total is the
    sum of those lengths.

    The "longest path" is computed via the 2-BFS diameter trick: from any
    starting node, find the farthest node A; from A, find the farthest node
    B; the path A->B is the diameter (exact for tree-structured skeletons,
    a tight lower bound for the rare cases where skeletonize leaves small
    cycles). Distances are weighted by physical voxel spacing.

    Returns ``(total_length_mm, n_skeleton_voxels)``.
    """
    coords = np.argwhere(skeleton)  # (N, 3) in (z, y, x)
    n = int(coords.shape[0])
    if n < 2:
        return 0.0, n

    sx, sy, sz = voxel_size_mm
    spacing_zyx = (sz, sy, sx)
    idx = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(coords)}

    # Pre-compute the 26 nonzero offsets and their physical edge lengths.
    offsets: list[tuple[int, int, int, float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                length = float(
                    np.sqrt(
                        (dz * spacing_zyx[0]) ** 2
                        + (dy * spacing_zyx[1]) ** 2
                        + (dx * spacing_zyx[2]) ** 2
                    )
                )
                offsets.append((dz, dy, dx, length))

    # Adjacency list keyed by voxel index: list of (neighbor_idx, edge_length).
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, c in enumerate(coords):
        z, y, x = int(c[0]), int(c[1]), int(c[2])
        for dz, dy, dx, length in offsets:
            j = idx.get((z + dz, y + dy, x + dx))
            if j is None or j == i:
                continue
            adjacency[i].append((j, length))

    # 2-BFS diameter per connected component.
    import heapq

    def _farthest_node_and_distance(source: int) -> tuple[int, float]:
        """Dijkstra from ``source`` over edge weights; return farthest node and its distance."""
        dist: dict[int, float] = {source: 0.0}
        heap: list[tuple[float, int]] = [(0.0, source)]
        farthest = source
        max_dist = 0.0
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if d > max_dist:
                max_dist = d
                farthest = u
            for v, w in adjacency[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return farthest, max_dist

    visited = [False] * n
    total = 0.0
    for start in range(n):
        if visited[start]:
            continue
        # BFS to mark the component this node belongs to.
        component_nodes: list[int] = []
        stack = [start]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            component_nodes.append(u)
            for v, _ in adjacency[u]:
                if not visited[v]:
                    stack.append(v)

        if len(component_nodes) < 2:
            continue
        # 2-BFS: from any node in the component find A (farthest), then from A find B.
        a, _ = _farthest_node_and_distance(component_nodes[0])
        _, length = _farthest_node_and_distance(a)
        total += length

    return total, n


def arc_length_metrics(
    tool_mask: np.ndarray,
    analytical_arc_length_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> ArcLengthMetrics:
    """Score a tool's rasterization of a curve against the analytical arc length.

    Parameters
    ----------
    tool_mask : np.ndarray
        ``(z, y, x)`` binary mask containing the rasterized curve (or curves).
    analytical_arc_length_mm : float
        Closed-form (or numerically-integrated) arc length from the primitive's
        spec JSON. For multi-curve primitives like ``MultiNeedleBundle`` this is
        the sum across all sub-curves.
    voxel_size_mm : tuple
        Reference grid spacing.
    """
    # Lazy import: skimage's skeletonizer handles N-D inputs automatically as
    # of skimage 0.26 (the older ``skeletonize_3d`` was unified into the
    # generic ``skeletonize``).
    from skimage.morphology import skeletonize

    skeleton = skeletonize((tool_mask > 0).astype(np.uint8)).astype(bool)
    raster_length, n_skel = _skeleton_arc_length_mm(skeleton, voxel_size_mm)

    if n_skel == 0:
        return ArcLengthMetrics(
            n_skeleton_voxels=0,
            raster_arc_length_mm=None,
            analytical_arc_length_mm=float(analytical_arc_length_mm),
            arc_length_error_mm=None,
            arc_length_error_pct=None,
        )

    abs_err = abs(raster_length - float(analytical_arc_length_mm))
    rel_err = (
        100.0 * abs_err / float(analytical_arc_length_mm)
        if analytical_arc_length_mm > 0
        else None
    )
    return ArcLengthMetrics(
        n_skeleton_voxels=n_skel,
        raster_arc_length_mm=float(raster_length),
        analytical_arc_length_mm=float(analytical_arc_length_mm),
        arc_length_error_mm=float(abs_err),
        arc_length_error_pct=rel_err,
    )
