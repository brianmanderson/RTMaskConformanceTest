"""Threshold configuration loading and per-primitive resolution.

This module validates every block of a threshold YAML. Schema 2::

    schema_version: 2
    claims:
        geometric_types: [CLOSED_PLANAR, CLOSEDPLANAR_XOR]   # default: all five
        convention: pv_ge_0.5          # or pv_gt_0, centre_in; default pv_ge_0.5
        end_cap: none                  # or half_slice; volume-only mode only
    report_only: [torus, OPEN_PLANAR]   # ROI names or geometric types; default: none
    defaults:                           # binary-mask ROIs, convention pv_ge_0.5
        dice: 0.95
        surface_dice_1mm: 0.95
        hd95_mm: 2.0
        msd_mm: 0.5
        volume_rel_err: 0.03
    primitives:
        torus: { dice: 0.90 }   # shallow-merge over defaults; any ROI name
    conventions:                # binary-mask ROIs under the other two conventions
        pv_gt_0:
            defaults: { hd95_mm: 2.0 }                      # every ROI under pv_gt_0
            primitives: { torus: { volume_rel_err: 0.055 } }
        centre_in: {}
    pointline:                  # curve and point ROIs, one sub-block per ROI
        diagonal_needle_x100_y200_z40: { arc_length_error_pct: 16.0 }
    volume_only:
        tolerance: 0.01         # relative volume error, `verify --volumes` only

``claims.geometric_types`` lists the Contour Geometric Type values the tool
under test supports. A ROI of a type that is not claimed is
``UNSUPPORTED_TYPE``, not a failure, when the tool writes no prediction for it
or an all-zero one. ``report_only`` lists ROI names or geometric types that
are scored and reported but never graded. ``claims.end_cap`` (``none`` or
``half_slice``) declares how the system closes a structure beyond its end
contours; only ``verify --volumes`` reads it, as it does
``volume_only.tolerance`` (see :mod:`rtmask_conformance.volume_only`).

``claims.convention`` declares the boundary convention the tool rasterises by,
one of :data:`CONVENTIONS`: ``pv_ge_0.5`` (a voxel is in when at least half of
it lies inside the structure; the default), ``pv_gt_0`` (any part of it) or
``centre_in`` (its centre). A binary-mask ROI is graded against the reference
of the declared convention, with that convention's thresholds; ``verify
--convention`` overrides the claim. Curve and point ROIs have no convention.

Threshold resolution for a binary-mask ROI ``r`` under convention ``c``, each
step a shallow merge over the one before (a key a later block names wins)::

    defaults  <-  primitives[r]                                  (c = pv_ge_0.5)
    defaults  <-  primitives[r]  <-  conventions[c].defaults
              <-  conventions[c].primitives[r]                   (any other c)

So the top-level ``defaults`` and ``primitives`` mean ``pv_ge_0.5``, and a
convention block names only what differs: whatever it leaves out is the
``pv_ge_0.5`` value. ``conventions`` may name ``pv_gt_0`` and ``centre_in``,
never ``pv_ge_0.5``.

Merging a user YAML over the shipped one: the user's ``defaults`` shallow-merge
over the shipped ``defaults``, the user's ``primitives[r]`` over the shipped
``primitives[r]``, and, inside ``conventions[c]``, the user's ``defaults`` and
``primitives[r]`` over the shipped ones the same way; the user's
``pointline[r]`` and ``volume_only`` blocks shallow-merge over the shipped
ones. A user ``claims`` block replaces the shipped one (a key it omits takes
its default, not the shipped value); a user ``report_only`` list adds to the
shipped one. Because a convention block wins over the top-level blocks, a user
who changes ``primitives[r]`` for a non-default convention must do it in
``conventions[c].primitives[r]`` wherever the shipped convention block names
that metric.

Schema 1 files (top level ``defaults``, ``primitives`` and ``pointline`` only)
still load: they claim every type, convention ``pv_ge_0.5``, and mark nothing
report-only. Unknown ``schema_version`` values are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCHEMA_VERSIONS: set[int] = {1, 2}

#: The five Contour Geometric Type (3006,0042) Enumerated Values, in the
#: order the standard lists them. ``claims.geometric_types`` may name only these.
STANDARD_GEOMETRIC_TYPES: tuple[str, ...] = (
    "CLOSED_PLANAR",
    "CLOSEDPLANAR_XOR",
    "OPEN_PLANAR",
    "OPEN_NONPLANAR",
    "POINT",
)

#: Keys accepted inside a ``claims:`` block.
CLAIMS_KEYS: set[str] = {"geometric_types", "convention", "end_cap"}

#: The boundary conventions a tool can declare (``claims.convention``), the
#: first the default: PV >= 0.5, PV > 0, voxel centre inside.
CONVENTIONS: tuple[str, ...] = ("pv_ge_0.5", "pv_gt_0", "centre_in")
DEFAULT_CONVENTION: str = CONVENTIONS[0]

#: Quadrature (sub-voxel samples per axis) of the fixture the ``conventions:``
#: thresholds were calibrated on, the generator's default. The ``pv_gt_0``
#: reference depends on it: a coarser quadrature misses small overlaps, so its
#: any-overlap mask is smaller (the other two references barely move).
CONVENTION_CALIBRATION_N_QUADRATURE: int = 8

#: Keys accepted inside one ``conventions.<name>:`` block.
CONVENTION_BLOCK_KEYS: set[str] = {"defaults", "primitives"}

#: Top-level keys only a schema-2 file may carry.
SCHEMA_2_ONLY_KEYS: tuple[str, ...] = ("claims", "report_only", "volume_only", "conventions")

#: ``claims.end_cap`` values, the first the default (volume-only mode).
END_CAP_VALUES: tuple[str, ...] = ("none", "half_slice")

#: Keys accepted inside a ``volume_only:`` block.
VOLUME_ONLY_KEYS: set[str] = {"tolerance"}

#: Default relative volume tolerance of ``verify --volumes``.
DEFAULT_VOLUME_TOLERANCE: float = 0.01

# Closed set of metric keys accepted in either `defaults` or per-primitive blocks.
METRIC_KEYS: set[str] = {
    "dice",
    "surface_dice_1mm",
    "hd95_mm",
    "msd_mm",
    "volume_rel_err",
}

#: Gradable metrics of a point-set ROI (``pointline:`` sub-blocks), in report order.
POINT_METRIC_KEYS: tuple[str, ...] = (
    "point_recall",
    "point_fp_rate",
    "point_mean_displacement_mm",
    "point_max_displacement_mm",
)

#: Gradable metrics of a curve ROI (``pointline:`` sub-blocks), in report order.
CURVE_METRIC_KEYS: tuple[str, ...] = (
    "curve_coverage_recall",
    "curve_mean_distance_mm",
    "curve_max_distance_mm",
    "curve_voxel_to_curve_mean_mm",
    "arc_length_error_pct",
)

POINTLINE_METRIC_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "points": POINT_METRIC_KEYS,
    "curve": CURVE_METRIC_KEYS,
}


@dataclass(frozen=True)
class Thresholds:
    """Resolved thresholds for one ROI."""

    dice: float
    surface_dice_1mm: float
    hd95_mm: float
    msd_mm: float
    volume_rel_err: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dice": self.dice,
            "surface_dice_1mm": self.surface_dice_1mm,
            "hd95_mm": self.hd95_mm,
            "msd_mm": self.msd_mm,
            "volume_rel_err": self.volume_rel_err,
        }


@dataclass
class ConformanceConfig:
    """Per-primitive threshold configuration backed by package defaults + optional YAML override."""

    defaults: dict[str, float]
    primitives: dict[str, dict[str, float]]
    config_source: str  # "default" or path to user YAML
    #: Geometric types the tool under test claims to support (standard order).
    claimed_types: tuple[str, ...] = STANDARD_GEOMETRIC_TYPES
    #: ROI names and geometric types that are scored but never graded.
    report_only: tuple[str, ...] = field(default_factory=tuple)
    #: End-cap convention claimed for volume-only grading (``END_CAP_VALUES``).
    end_cap: str = END_CAP_VALUES[0]
    #: Relative volume tolerance of volume-only grading.
    volume_tolerance: float = DEFAULT_VOLUME_TOLERANCE
    #: Boundary convention binary-mask ROIs are graded under (``CONVENTIONS``).
    convention: str = DEFAULT_CONVENTION
    #: Whether the convention was declared (``claims.convention`` or
    #: :meth:`with_convention`); only then does a report record it.
    convention_declared: bool = False
    #: ``conventions:`` blocks: name -> {"defaults": {...}, "primitives": {roi: {...}}}.
    conventions: dict[str, dict[str, dict]] = field(default_factory=dict)
    #: ``pointline:`` block: curve or point ROI name -> {metric: threshold}.
    pointline: dict[str, dict[str, float]] = field(default_factory=dict)

    def claims(self, geometric_type: str | None) -> bool:
        """Whether ``geometric_type`` is claimed.

        Only the five standard values can be declined: ``None`` (type not known)
        and any other string (``"unknown"`` from an upgraded v1 manifest, say)
        count as claimed.
        """
        if geometric_type not in STANDARD_GEOMETRIC_TYPES:
            return True
        return geometric_type in self.claimed_types

    def is_report_only(self, roi: str, geometric_type: str | None = None) -> bool:
        """Whether ``roi`` is listed in ``report_only`` by name or by its geometric type."""
        return roi in self.report_only or (
            geometric_type is not None and geometric_type in self.report_only
        )

    def claims_dict(self) -> dict[str, list[str] | str]:
        """The ``claims`` block as recorded in a mask report.

        ``convention`` is included only when declared, so a report that
        declares none is unchanged from before conventions existed.
        """
        d: dict[str, list[str] | str] = {"geometric_types": list(self.claimed_types)}
        if self.convention_declared:
            d["convention"] = self.convention
        return d

    def with_convention(self, convention: str) -> ConformanceConfig:
        """A copy of this config graded under ``convention`` (``verify --convention``)."""
        if convention not in CONVENTIONS:
            raise ValueError(f"convention {convention!r} not one of {list(CONVENTIONS)}")
        return replace(self, convention=convention, convention_declared=True)

    def thresholds_for(self, roi: str, convention: str | None = None) -> Thresholds:
        """Resolved thresholds of binary-mask ROI ``roi`` (order in the module docstring).

        ``convention`` defaults to the config's own.
        """
        convention = convention or self.convention
        merged: dict[str, float] = dict(self.defaults)
        merged.update(self.primitives.get(roi, {}))
        if convention != DEFAULT_CONVENTION:
            block = self.conventions.get(convention, {})
            merged.update(block.get("defaults", {}))
            merged.update(block.get("primitives", {}).get(roi, {}))
        try:
            return Thresholds(**merged)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError(
                f"Resolved threshold dict for ROI {roi!r} is missing required keys: {merged}"
            ) from e


def _validate_metric_block(block: dict[str, Any], context: str) -> dict[str, float]:
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(block).__name__}")
    out: dict[str, float] = {}
    for k, v in block.items():
        if k not in METRIC_KEYS:
            raise ValueError(
                f"{context}: unknown metric {k!r}. Allowed: {sorted(METRIC_KEYS)}"
            )
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{context}.{k}: expected a number, got {type(v).__name__}")
        out[k] = float(v)
    return out


def _validate_claims(block: Any, context: str) -> tuple[str, ...]:
    """Validate a ``claims:`` block; return the claimed types in the standard's order."""
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - CLAIMS_KEYS)
    if unknown:
        raise ValueError(f"{context}: unknown key(s) {unknown}. Allowed: {sorted(CLAIMS_KEYS)}")
    if "geometric_types" not in block:
        return STANDARD_GEOMETRIC_TYPES
    types = block["geometric_types"]
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
        raise ValueError(f"{context}.geometric_types: expected a list of strings")
    if not types:
        raise ValueError(
            f"{context}.geometric_types: claim at least one of {list(STANDARD_GEOMETRIC_TYPES)}"
        )
    bad = [t for t in types if t not in STANDARD_GEOMETRIC_TYPES]
    if bad:
        raise ValueError(
            f"{context}.geometric_types: {bad} not a Contour Geometric Type value. "
            f"Allowed: {list(STANDARD_GEOMETRIC_TYPES)}"
        )
    return tuple(t for t in STANDARD_GEOMETRIC_TYPES if t in types)


def _validate_convention(block: dict[str, Any], context: str) -> str | None:
    """``claims.convention`` from an already-validated claims mapping, or ``None`` if absent."""
    if "convention" not in block:
        return None
    value = block["convention"]
    if value not in CONVENTIONS:
        raise ValueError(f"{context}.convention: {value!r} not one of {list(CONVENTIONS)}")
    return value


def _validate_primitives(block: Any, context: str) -> dict[str, dict[str, float]]:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping of ROI name -> metrics")
    return {roi: _validate_metric_block(b, f"{context}.{roi}") for roi, b in block.items()}


def _validate_conventions(block: Any, context: str) -> dict[str, dict[str, dict]]:
    """Validate a ``conventions:`` block (keys: the non-default conventions)."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(block).__name__}")
    out: dict[str, dict[str, dict]] = {}
    for name, sub in block.items():
        if name == DEFAULT_CONVENTION:
            raise ValueError(
                f"{context}.{name}: the top-level defaults and primitives are the "
                f"{DEFAULT_CONVENTION} thresholds; set them there"
            )
        if name not in CONVENTIONS:
            raise ValueError(
                f"{context}: unknown convention {name!r}. Allowed: {list(CONVENTIONS[1:])}"
            )
        sub = sub or {}
        if not isinstance(sub, dict):
            raise ValueError(f"{context}.{name}: expected a mapping, got {type(sub).__name__}")
        unknown = sorted(set(sub) - CONVENTION_BLOCK_KEYS)
        if unknown:
            raise ValueError(
                f"{context}.{name}: unknown key(s) {unknown}. "
                f"Allowed: {sorted(CONVENTION_BLOCK_KEYS)}"
            )
        out[name] = {
            "defaults": _validate_metric_block(
                sub.get("defaults") or {}, f"{context}.{name}.defaults"
            ),
            "primitives": _validate_primitives(
                sub.get("primitives"), f"{context}.{name}.primitives"
            ),
        }
    return out


_POINTLINE_KEYS: frozenset[str] = frozenset(POINT_METRIC_KEYS + CURVE_METRIC_KEYS)


def _validate_pointline(block: Any, context: str) -> dict[str, dict[str, float]]:
    """Validate a ``pointline:`` block: curve or point ROI name -> {metric: threshold}.

    Whether a metric suits the ROI's truth kind is checked when the ROI is
    graded (:func:`pointline_thresholds_for`).
    """
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
            if k not in _POINTLINE_KEYS:
                raise ValueError(
                    f"{context}.{roi}: unknown metric {k!r}. Allowed: {sorted(_POINTLINE_KEYS)}"
                )
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"{context}.{roi}.{k}: expected a number, got {type(v).__name__}"
                )
            out[roi][k] = float(v)
    return out


def pointline_thresholds_for(
    config: ConformanceConfig, roi: str, truth_kind: str
) -> dict[str, float]:
    """The graded metrics and their thresholds of one curve or point ROI, in report order.

    A ROI without a ``pointline`` sub-block, or a sub-block naming a metric of
    the other truth kind, is a configuration error (``ValueError``).
    """
    block = config.pointline.get(roi)
    if block is None:
        raise ValueError(
            f"no thresholds for {truth_kind} ROI {roi!r}: add a `pointline: {{{roi}: ...}}` block"
        )
    allowed = POINTLINE_METRIC_KEYS_BY_KIND[truth_kind]
    wrong = sorted(set(block) - set(allowed))
    if wrong:
        raise ValueError(
            f"pointline.{roi}: {wrong} do not apply to a {truth_kind} ROI. "
            f"Allowed: {sorted(allowed)}"
        )
    return {k: block[k] for k in allowed if k in block}


def _validate_end_cap(block: dict[str, Any], context: str) -> str | None:
    """``claims.end_cap`` from an already-validated claims mapping, or ``None`` if absent."""
    if "end_cap" not in block:
        return None
    value = block["end_cap"]
    if value not in END_CAP_VALUES:
        raise ValueError(
            f"{context}.end_cap: {value!r} not one of {list(END_CAP_VALUES)}"
        )
    return value


def _validate_volume_only(block: Any, context: str) -> dict[str, float]:
    """Validate a ``volume_only:`` block."""
    if not isinstance(block, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - VOLUME_ONLY_KEYS)
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {unknown}. Allowed: {sorted(VOLUME_ONLY_KEYS)}"
        )
    out: dict[str, float] = {}
    if "tolerance" in block:
        tol = block["tolerance"]
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0 < tol < 1:
            raise ValueError(f"{context}.tolerance: expected a fraction in (0, 1), got {tol!r}")
        out["tolerance"] = float(tol)
    return out


def _validate_report_only(block: Any, context: str) -> tuple[str, ...]:
    """Validate a ``report_only:`` list of ROI names or geometric types (deduplicated)."""
    if not isinstance(block, list) or not all(isinstance(t, str) and t for t in block):
        raise ValueError(f"{context}: expected a list of ROI names or geometric types")
    return tuple(dict.fromkeys(block))


def _check_schema(data: dict[str, Any], context: str) -> int:
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{context}: unsupported schema_version {version!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if version == 1:
        v2_only = sorted(k for k in SCHEMA_2_ONLY_KEYS if k in data)
        if v2_only:
            raise ValueError(f"{context}: {v2_only} require schema_version: 2")
    return version


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _shipped_default_path() -> Path:
    return Path(str(files("rtmask_conformance").joinpath("data/default_thresholds.yaml")))


def load_config(user_yaml: str | Path | None = None) -> ConformanceConfig:
    """Load the package-shipped defaults and merge a user YAML over them.

    User YAMLs may omit ``defaults`` entirely or override individual metrics; any
    metric the user does not name falls back to the shipped value. The merge
    rules for every block are in the module docstring.
    """
    shipped = _load_yaml(_shipped_default_path())
    _check_schema(shipped, "Shipped default_thresholds.yaml")
    base_defaults = _validate_metric_block(shipped.get("defaults", {}), "shipped.defaults")
    base_primitives_raw = shipped.get("primitives", {}) or {}
    base_primitives = {
        roi: _validate_metric_block(block, f"shipped.primitives.{roi}")
        for roi, block in base_primitives_raw.items()
    }
    base_claims = _validate_claims(shipped.get("claims") or {}, "shipped.claims")
    base_end_cap = (
        _validate_end_cap(shipped.get("claims") or {}, "shipped.claims") or END_CAP_VALUES[0]
    )
    base_volume_only = _validate_volume_only(
        shipped.get("volume_only") or {}, "shipped.volume_only"
    )
    base_tolerance = base_volume_only.get("tolerance", DEFAULT_VOLUME_TOLERANCE)
    base_report_only = _validate_report_only(
        shipped.get("report_only") or [], "shipped.report_only"
    )
    base_convention = _validate_convention(shipped.get("claims") or {}, "shipped.claims")
    base_conventions = _validate_conventions(shipped.get("conventions"), "shipped.conventions")
    base_pointline = _validate_pointline(shipped.get("pointline"), "shipped.pointline")

    if user_yaml is None:
        return ConformanceConfig(
            defaults=base_defaults,
            primitives=base_primitives,
            config_source="default",
            claimed_types=base_claims,
            report_only=base_report_only,
            end_cap=base_end_cap,
            volume_tolerance=base_tolerance,
            convention=base_convention or DEFAULT_CONVENTION,
            convention_declared=base_convention is not None,
            conventions=base_conventions,
            pointline=base_pointline,
        )

    user_path = Path(user_yaml)
    if not user_path.exists():
        raise FileNotFoundError(f"User threshold YAML not found: {user_path}")
    user = _load_yaml(user_path)
    _check_schema(user, str(user_path))

    user_defaults = _validate_metric_block(user.get("defaults", {}), "user.defaults")
    merged_defaults = dict(base_defaults)
    merged_defaults.update(user_defaults)

    user_primitives_raw = user.get("primitives", {}) or {}
    merged_primitives: dict[str, dict[str, float]] = {
        roi: dict(block) for roi, block in base_primitives.items()
    }
    for roi, block in user_primitives_raw.items():
        validated = _validate_metric_block(block, f"user.primitives.{roi}")
        merged_primitives.setdefault(roi, {})
        merged_primitives[roi].update(validated)

    claims = (
        _validate_claims(user["claims"], "user.claims") if "claims" in user else base_claims
    )
    # A user claims block replaces the shipped one, end_cap included.
    end_cap = (
        (_validate_end_cap(user["claims"], "user.claims") or END_CAP_VALUES[0])
        if "claims" in user else base_end_cap
    )
    # A user claims block replaces the shipped one, convention included.
    user_convention = (
        _validate_convention(user["claims"], "user.claims") if "claims" in user
        else base_convention
    )

    merged_conventions = {
        name: {"defaults": dict(b["defaults"]),
               "primitives": {r: dict(m) for r, m in b["primitives"].items()}}
        for name, b in base_conventions.items()
    }
    for name, b in _validate_conventions(user.get("conventions"), "user.conventions").items():
        target = merged_conventions.setdefault(name, {"defaults": {}, "primitives": {}})
        target["defaults"].update(b["defaults"])
        for roi, m in b["primitives"].items():
            target["primitives"].setdefault(roi, {}).update(m)

    merged_pointline = {roi: dict(m) for roi, m in base_pointline.items()}
    for roi, m in _validate_pointline(user.get("pointline"), "user.pointline").items():
        merged_pointline.setdefault(roi, {}).update(m)

    user_volume_only = _validate_volume_only(user.get("volume_only") or {}, "user.volume_only")
    report_only = tuple(dict.fromkeys(
        base_report_only
        + _validate_report_only(user.get("report_only") or [], "user.report_only")
    ))

    return ConformanceConfig(
        defaults=merged_defaults,
        primitives=merged_primitives,
        config_source=str(user_path),
        claimed_types=claims,
        report_only=report_only,
        end_cap=end_cap,
        volume_tolerance=user_volume_only.get("tolerance", base_tolerance),
        convention=user_convention or DEFAULT_CONVENTION,
        convention_declared=user_convention is not None,
        conventions=merged_conventions,
        pointline=merged_pointline,
    )
