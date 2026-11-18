"""Threshold configuration loading and per-primitive resolution.

A threshold YAML follows this schema (schema_version: 2)::

    schema_version: 2
    claims:
        geometric_types: [CLOSED_PLANAR, CLOSEDPLANAR_XOR]   # default: all five
        end_cap: none                  # or half_slice; volume-only mode only
    report_only: [torus, OPEN_PLANAR]   # ROI names or geometric types; default: none
    defaults:
        dice: 0.95
        surface_dice_1mm: 0.95
        hd95_mm: 2.0
        msd_mm: 0.5
        volume_rel_err: 0.03
    primitives:
        torus: { dice: 0.90 }   # shallow-merge over defaults; any ROI name
    volume_only:
        tolerance: 0.01         # relative volume error, `verify --volumes` only

``claims.geometric_types`` lists the Contour Geometric Type values the tool
under test supports. A ROI whose type is not claimed and whose prediction is
absent is ``UNSUPPORTED_TYPE``, not a failure. ``report_only`` lists ROI names
or geometric types that are scored and reported but never graded.
``claims.end_cap`` (``none`` or ``half_slice``) declares how the system closes
a structure beyond its end contours; only ``verify --volumes`` reads it, as it
does ``volume_only.tolerance`` (see :mod:`rtmask_conformance.volume_only`).

Schema 1 files (no ``claims``, no ``report_only``, no ``volume_only``) still load: they claim
every type and mark nothing report-only. A user YAML's per-primitive
overrides shallow-merge over the user's own ``defaults``, which themselves
shallow-merge over the package-shipped defaults; a user ``claims`` block
replaces the shipped one, a user ``report_only`` list adds to the
shipped one, and a user ``volume_only`` block shallow-merges over the shipped one. Unknown ``schema_version`` values are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
CLAIMS_KEYS: set[str] = {"geometric_types", "end_cap"}

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

    def claims_dict(self) -> dict[str, list[str]]:
        """The ``claims`` block as recorded in a report."""
        return {"geometric_types": list(self.claimed_types)}

    def thresholds_for(self, roi: str) -> Thresholds:
        merged: dict[str, float] = dict(self.defaults)
        merged.update(self.primitives.get(roi, {}))
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
        v2_only = sorted(k for k in ("claims", "report_only", "volume_only") if k in data)
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
    metric the user does not name falls back to the shipped value. A user
    ``claims`` block replaces the shipped claims; a user ``report_only`` list
    is added to the shipped one.
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

    if user_yaml is None:
        return ConformanceConfig(
            defaults=base_defaults,
            primitives=base_primitives,
            config_source="default",
            claimed_types=base_claims,
            report_only=base_report_only,
            end_cap=base_end_cap,
            volume_tolerance=base_tolerance,
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
    )
