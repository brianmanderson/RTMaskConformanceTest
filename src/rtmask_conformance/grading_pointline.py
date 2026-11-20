"""Grading for ROIs whose ground truth is a curve or a point set.

``POINT`` and the open-curve Contour Geometric Types (``OPEN_PLANAR``,
``OPEN_NONPLANAR``) enclose no volume, so Dice and the surface metrics do not
apply. A tool's mask for such a ROI is scored against the analytic truth the
generator writes as JSON:

* ``groundtruth_points/<roi>.json`` — the point coordinates (truth kind
  ``points``), scored with the vendored :func:`point_metrics`;
* ``groundtruth_curves/<roi>.json`` — each of the ROI's curves densified to
  at most :data:`DENSIFY_MAX_STEP_MM` between samples, the open-curve
  endpoints and the analytic arc length (truth kind ``curve``), scored with
  the vendored :func:`curve_metrics` (against all curves' samples together)
  and :func:`arc_length_metrics`.

Both files also record the reference grid, which the prediction must match.

Thresholds live in a top-level ``pointline:`` block of the threshold YAML, one
sub-block per ROI name; the metrics a sub-block names are the ones graded::

    pointline:
      diagonal_needle_x100_y200_z40:
        arc_length_error_pct: 16.0
        curve_max_distance_mm: 1.5

A user YAML's ``pointline.<roi>`` block shallow-merges over the shipped one.
The block is loaded and validated with every other threshold block by
:func:`rtmask_conformance.thresholds.load_config`
(:attr:`~rtmask_conformance.thresholds.ConformanceConfig.pointline`).
``point_recall`` and ``curve_coverage_recall`` are higher-is-better; every
other metric is lower-is-better. Curve and point ROIs have no boundary
convention: ``claims.convention`` and ``verify --convention`` do not change
how they are graded.

Status rules (the order of :mod:`rtmask_conformance.verify` applies around
them): an unclaimed type with no prediction, or with an all-zero one, is
``UNSUPPORTED_TYPE``; a claimed type with an all-zero mask is ``FAIL``; a
metric beyond its threshold is ``FAIL``; otherwise ``PASS``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._vendor.common.coords import Geometry
from ._vendor.metrics import (
    arc_length_metrics,
    curve_metrics,
    densify_polyline,
    point_metrics,
)
from ._vendor.primitives.base import AnalyticalPrimitive
from .thresholds import (
    CURVE_METRIC_KEYS,
    POINT_METRIC_KEYS,
    load_config,
    pointline_thresholds_for,
)

#: Truth kinds this module grades.
POINTLINE_TRUTH_KINDS: tuple[str, ...] = ("points", "curve")

#: Fixture directory for each truth kind's JSON truth.
TRUTH_DIRS: dict[str, str] = {"points": "groundtruth_points", "curve": "groundtruth_curves"}

#: Maximum spacing (mm) between consecutive samples of the densified analytic
#: curve; the upstream harness uses the same value.
DENSIFY_MAX_STEP_MM: float = 0.3

#: Metrics reported (and gradable) per truth kind, in report order. The key
#: tuples are defined in thresholds.py, which validates the ``pointline:`` block.
METRIC_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "points": POINT_METRIC_KEYS,
    "curve": CURVE_METRIC_KEYS,
}

#: Metrics where a larger value is better; all others are lower-is-better.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"point_recall", "curve_coverage_recall"})

#: Spec-dict keys that carry a curve's analytic arc length, in the order the
#: upstream harness reads them (the first non-null one wins).
ARC_LENGTH_SPEC_KEYS: tuple[str, ...] = (
    "analytical_total_arc_length_mm",
    "analytical_arc_length_mm",
    "continuous_arc_length_mm",
    "polyline_arc_length_mm",
)

_OPEN_CURVE_TYPES = ("OPEN_PLANAR", "OPEN_NONPLANAR")


# -----------------------------------------------------------------
#  Truth: built by the generator, read by the grader
# -----------------------------------------------------------------


#: Significant digits of every coordinate and length in a truth JSON. Sine and
#: cosine (helix, needle track, sinusoid) can differ in the last bit between
#: platforms; 12 digits absorbs that, so the files hash the same on every OS,
#: while moving a value by at most half a unit in the 12th digit (~1e-10 mm at
#: 500 mm). The RTSTRUCT writer upstream uses the same precision for ContourData.
TRUTH_SIGNIFICANT_DIGITS: int = 12


def _sig(values: np.ndarray | float) -> Any:
    """``values`` rounded to :data:`TRUTH_SIGNIFICANT_DIGITS`, as nested lists / a float."""
    fmt = f".{TRUTH_SIGNIFICANT_DIGITS}g"
    if np.isscalar(values):
        return float(format(float(values), fmt))
    return np.vectorize(lambda v: float(format(float(v), fmt)), otypes=[np.float64])(
        np.asarray(values, dtype=np.float64)
    ).tolist()


def _geometry_dict(geometry: Geometry) -> dict[str, list]:
    return {
        "origin": [float(v) for v in geometry.origin],
        "spacing": [float(v) for v in geometry.spacing],
        "size": [int(v) for v in geometry.size],
        "direction": [float(v) for v in geometry.direction],
    }


def geometry_from_truth(truth: dict) -> Geometry:
    """The reference grid recorded in a truth JSON."""
    g = truth["geometry"]
    return Geometry(
        origin=tuple(g["origin"]),
        spacing=tuple(g["spacing"]),
        size=tuple(int(v) for v in g["size"]),
        direction=tuple(g["direction"]),
    )


def build_truth(
    primitive: AnalyticalPrimitive, truth_kind: str, geometry: Geometry
) -> dict[str, Any]:
    """The analytic truth record of a curve or point-set primitive.

    Points: the primitive's point list. Curves: every contour of
    ``primitive.get_contours`` densified with :func:`densify_polyline` (one
    polyline per contour), the first and last vertex of each open contour, and
    the analytic arc length from the spec dict (summed over the contours by
    the primitive itself). Coordinates and the arc length are rounded to
    :data:`TRUTH_SIGNIFICANT_DIGITS`.
    """
    record: dict[str, Any] = {
        "roi": primitive.name,
        "geometric_type": primitive.geometric_type.value,
        "truth_kind": truth_kind,
        "geometry": _geometry_dict(geometry),
    }
    if truth_kind == "points":
        points = np.asarray(primitive.points, dtype=np.float64)  # type: ignore[attr-defined]
        record["n_points"] = int(points.shape[0])
        record["points_mm"] = _sig(points)
        return record
    if truth_kind != "curve":
        raise ValueError(f"{primitive.name}: not a curve or point truth kind: {truth_kind!r}")

    chunks: list[np.ndarray] = []
    endpoints: list[list[float]] = []
    contours = primitive.get_contours(geometry)
    for c in contours:
        chunks.append(densify_polyline(c.points_xyz, max_step_mm=DENSIFY_MAX_STEP_MM))
        gt = c.geometric_type.value if hasattr(c.geometric_type, "value") else str(c.geometric_type)
        if gt in _OPEN_CURVE_TYPES:
            endpoints.append(_sig(c.points_xyz[0]))
            endpoints.append(_sig(c.points_xyz[-1]))
    spec = primitive.to_spec_dict()
    source = next((k for k in ARC_LENGTH_SPEC_KEYS if spec.get(k) is not None), None)
    if source is None:
        raise ValueError(f"{primitive.name}: spec carries no analytic arc length")
    record["n_curves"] = len(contours)
    record["densify_max_step_mm"] = DENSIFY_MAX_STEP_MM
    record["analytical_arc_length_mm"] = _sig(float(spec[source]))
    record["arc_length_source"] = source
    record["endpoints_mm"] = endpoints or None
    record["polylines_mm"] = [_sig(chunk) for chunk in chunks]
    return record


def curve_samples(truth: dict[str, Any]) -> np.ndarray:
    """All densified samples of a curve truth, every contour concatenated, ``(N, 3)``."""
    return np.concatenate(
        [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in truth["polylines_mm"]], axis=0
    )


def write_truth(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def read_truth(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        record = json.load(fh)
    if record.get("truth_kind") not in POINTLINE_TRUTH_KINDS:
        raise ValueError(f"{path}: not a curve or point truth file")
    return record


# -----------------------------------------------------------------
#  The suite's reference thin-path rasterisation
# -----------------------------------------------------------------

#: Resampling step (mm) of the reference curve rasteriser: fine enough that
#: every voxel the analytic curve crosses receives a sample.
REFERENCE_RASTER_STEP_MM: float = 0.05


def _mark_nearest_voxels(points_xyz: np.ndarray, geometry: Geometry) -> np.ndarray:
    nx, ny, nz = geometry.size
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    if points_xyz.size == 0:
        return mask
    idx = np.rint(geometry.physical_to_index(points_xyz)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array([nx, ny, nz])), axis=1)
    idx = idx[inside]
    mask[idx[:, 2], idx[:, 1], idx[:, 0]] = 1
    return mask


def reference_rasterisation(truth: dict[str, Any]) -> np.ndarray:
    """The suite's own ideal ``(z, y, x)`` uint8 mask for a curve or point truth.

    Points: the voxel nearest each point. Curves: the voxel containing each
    sample of each analytic polyline resampled every
    :data:`REFERENCE_RASTER_STEP_MM`, i.e. the voxels the curve passes through.
    Separate contours are never joined.
    """
    geometry = geometry_from_truth(truth)
    if truth["truth_kind"] == "points":
        return _mark_nearest_voxels(np.asarray(truth["points_mm"], dtype=np.float64), geometry)
    samples = [
        densify_polyline(np.asarray(p, dtype=np.float64), max_step_mm=REFERENCE_RASTER_STEP_MM)
        for p in truth["polylines_mm"]
    ]
    return _mark_nearest_voxels(np.concatenate(samples, axis=0), geometry)


# -----------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------


def _cropped(mask: np.ndarray, pad: int = 2) -> np.ndarray:
    """``mask`` cropped to its non-zero bounding box plus ``pad`` background voxels.

    Skeletonisation is local to the object, so the skeleton (and the arc length
    measured on it) is unchanged; the full 512 x 512 x 200 grid is just slower.
    """
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad + 1, mask.shape)
    return mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def compute_metrics(
    mask_zyx: np.ndarray, truth: dict[str, Any], geometry: Geometry
) -> dict[str, float | int | None]:
    """Score a non-empty ``(z, y, x)`` mask against a curve or point truth record.

    Returns the gradable metrics (:data:`POINT_METRIC_KEYS` or
    :data:`CURVE_METRIC_KEYS`) plus informational counts and lengths.
    """
    mask = (np.asarray(mask_zyx) > 0).astype(np.uint8)
    if truth["truth_kind"] == "points":
        pm = point_metrics(mask, np.asarray(truth["points_mm"], dtype=np.float64), geometry)
        return {
            "point_recall": pm.recall,
            "point_fp_rate": pm.fp_rate,
            "point_mean_displacement_mm": pm.mean_point_displacement_mm,
            "point_max_displacement_mm": pm.max_point_displacement_mm,
            "point_n_points": pm.n_points,
            "n_marked_voxels": pm.n_marked_voxels,
        }

    endpoints = truth.get("endpoints_mm")
    cm = curve_metrics(
        mask,
        curve_samples(truth),
        geometry,
        endpoints_mm=np.asarray(endpoints, dtype=np.float64) if endpoints else None,
    )
    if mask.any():
        am = arc_length_metrics(
            _cropped(mask), float(truth["analytical_arc_length_mm"]), tuple(geometry.spacing)
        )
        raster_len, err_pct = am.raster_arc_length_mm, am.arc_length_error_pct
    else:
        raster_len, err_pct = None, None
    return {
        "curve_coverage_recall": cm.coverage_recall,
        "curve_mean_distance_mm": cm.mean_curve_distance_mm,
        "curve_max_distance_mm": cm.max_curve_distance_mm,
        "curve_voxel_to_curve_mean_mm": cm.voxel_to_curve_mean_mm,
        "arc_length_error_pct": err_pct,
        "curve_voxel_to_curve_max_mm": cm.voxel_to_curve_max_mm,
        "curve_endpoint_max_displacement_mm": cm.endpoint_displacement_max_mm,
        "arc_length_raster_mm": raster_len,
        "arc_length_analytical_mm": float(truth["analytical_arc_length_mm"]),
        "n_marked_voxels": cm.n_marked_voxels,
    }


def violations(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Human-readable violations of ``thresholds`` (empty if every graded metric passes).

    A graded metric whose value is ``None`` is a violation.
    """
    out: list[str] = []
    for name, limit in thresholds.items():
        v = metrics.get(name)
        if v is None:
            out.append(f"{name} not computable")
        elif name in HIGHER_IS_BETTER and v < limit:
            out.append(f"{name} {v:.4f} < {limit}")
        elif name not in HIGHER_IS_BETTER and v > limit:
            out.append(f"{name} {v:.4f} > {limit}")
    return out


# -----------------------------------------------------------------
#  Thresholds: the top-level ``pointline:`` YAML block (thresholds.py)
# -----------------------------------------------------------------


def load_pointline_thresholds(config_source: str = "default") -> dict[str, dict[str, float]]:
    """The shipped ``pointline`` block, with a user YAML's merged over it per ROI.

    ``config_source`` is :attr:`ConformanceConfig.config_source`: ``"default"``
    or the path of the user YAML. Kept for callers that hold a path, not a
    config; :func:`rtmask_conformance.thresholds.load_config` does the work.
    """
    return load_config(None if config_source == "default" else config_source).pointline


def thresholds_for(config_source: str, roi: str, truth_kind: str) -> dict[str, float]:
    """The graded metrics and their thresholds for one curve or point ROI.

    See :func:`rtmask_conformance.thresholds.pointline_thresholds_for`, which
    takes a loaded config.
    """
    config = load_config(None if config_source == "default" else config_source)
    return pointline_thresholds_for(config, roi, truth_kind)
