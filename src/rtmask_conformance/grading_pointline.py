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
``point_recall`` and ``curve_coverage_recall`` are higher-is-better; every
other metric is lower-is-better.

Status rules (the order of :mod:`rtmask_conformance.verify` applies around
them): an unclaimed type with no prediction, or with an all-zero one, is
``UNSUPPORTED_TYPE``; a claimed type with an all-zero mask is ``FAIL``; a
metric beyond its threshold is ``FAIL``; otherwise ``PASS``.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ._vendor.common.coords import Geometry
from ._vendor.metrics import (
    arc_length_metrics,
    curve_metrics,
    densify_polyline,
    point_metrics,
)
from ._vendor.primitives.base import AnalyticalPrimitive

#: Truth kinds this module grades.
POINTLINE_TRUTH_KINDS: tuple[str, ...] = ("points", "curve")

#: Fixture directory for each truth kind's JSON truth.
TRUTH_DIRS: dict[str, str] = {"points": "groundtruth_points", "curve": "groundtruth_curves"}

#: Maximum spacing (mm) between consecutive samples of the densified analytic
#: curve; the upstream harness uses the same value.
DENSIFY_MAX_STEP_MM: float = 0.3

#: Metrics reported (and gradable) for a point set, in report order.
POINT_METRIC_KEYS: tuple[str, ...] = (
    "point_recall",
    "point_fp_rate",
    "point_mean_displacement_mm",
    "point_max_displacement_mm",
)

#: Metrics reported (and gradable) for a curve, in report order.
CURVE_METRIC_KEYS: tuple[str, ...] = (
    "curve_coverage_recall",
    "curve_mean_distance_mm",
    "curve_max_distance_mm",
    "curve_voxel_to_curve_mean_mm",
    "arc_length_error_pct",
)

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
    the primitive itself).
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
        record["points_mm"] = points.tolist()
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
            endpoints.append(np.asarray(c.points_xyz[0], dtype=np.float64).tolist())
            endpoints.append(np.asarray(c.points_xyz[-1], dtype=np.float64).tolist())
    spec = primitive.to_spec_dict()
    source = next((k for k in ARC_LENGTH_SPEC_KEYS if spec.get(k) is not None), None)
    if source is None:
        raise ValueError(f"{primitive.name}: spec carries no analytic arc length")
    record["n_curves"] = len(contours)
    record["densify_max_step_mm"] = DENSIFY_MAX_STEP_MM
    record["analytical_arc_length_mm"] = float(spec[source])
    record["arc_length_source"] = source
    record["endpoints_mm"] = endpoints or None
    record["polylines_mm"] = [chunk.tolist() for chunk in chunks]
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
#  Thresholds: the top-level ``pointline:`` YAML block
# -----------------------------------------------------------------

_ALL_KEYS: frozenset[str] = frozenset(POINT_METRIC_KEYS + CURVE_METRIC_KEYS)


def _validate_block(block: Any, context: str) -> dict[str, dict[str, float]]:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping of ROI name -> metrics")
    out: dict[str, dict[str, float]] = {}
    for roi, metrics in block.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"{context}.{roi}: expected a mapping of metric -> threshold")
        out[roi] = {}
        for k, v in metrics.items():
            if k not in _ALL_KEYS:
                raise ValueError(
                    f"{context}.{roi}: unknown metric {k!r}. Allowed: {sorted(_ALL_KEYS)}"
                )
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{context}.{roi}.{k}: expected a number, got {type(v).__name__}")
            out[roi][k] = float(v)
    return out


def _yaml_block(path: Path, context: str) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return _validate_block(data.get("pointline"), context)


def load_pointline_thresholds(config_source: str = "default") -> dict[str, dict[str, float]]:
    """The shipped ``pointline`` block, with a user YAML's merged over it per ROI.

    ``config_source`` is :attr:`ConformanceConfig.config_source`: ``"default"``
    or the path of the user YAML.
    """
    shipped = Path(str(files("rtmask_conformance").joinpath("data/default_thresholds.yaml")))
    merged = _yaml_block(shipped, "shipped.pointline")
    if config_source != "default":
        for roi, block in _yaml_block(Path(config_source), "user.pointline").items():
            merged.setdefault(roi, {}).update(block)
    return merged


def thresholds_for(config_source: str, roi: str, truth_kind: str) -> dict[str, float]:
    """The graded metrics and their thresholds for one curve or point ROI."""
    block = load_pointline_thresholds(config_source).get(roi)
    if block is None:
        raise ValueError(
            f"no thresholds for {truth_kind} ROI {roi!r}: add a `pointline: {{{roi}: ...}}` block"
        )
    allowed = set(METRIC_KEYS_BY_KIND[truth_kind])
    wrong = sorted(set(block) - allowed)
    if wrong:
        raise ValueError(
            f"pointline.{roi}: {wrong} do not apply to a {truth_kind} ROI. Allowed: {sorted(allowed)}"
        )
    # Report order, not YAML order.
    return {k: block[k] for k in METRIC_KEYS_BY_KIND[truth_kind] if k in block}
