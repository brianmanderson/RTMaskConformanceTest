"""Threshold configuration loading and per-primitive resolution.

A threshold YAML follows this schema (schema_version: 1):

    schema_version: 1
    defaults:
        dice: 0.95
        surface_dice_1mm: 0.95
        hd95_mm: 2.0
        msd_mm: 0.5
        volume_rel_err: 0.03
    primitives:
        torus_R60_r20_x400_y400: { dice: 0.90 }   # shallow-merge over defaults

User-supplied YAML files use the same shape; per-primitive overrides
shallow-merge over the user's own ``defaults``, which themselves
shallow-merge over the package-shipped defaults. Unknown ``schema_version``
values are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCHEMA_VERSIONS: set[int] = {1}

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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _shipped_default_path() -> Path:
    return Path(str(files("rtmask_conformance").joinpath("data/default_thresholds.yaml")))


def load_config(user_yaml: str | Path | None = None) -> ConformanceConfig:
    """Load the package-shipped defaults and shallow-merge a user YAML over them.

    User YAMLs may omit ``defaults`` entirely or override individual metrics; any
    metric the user does not name falls back to the shipped value.
    """
    shipped = _load_yaml(_shipped_default_path())
    if shipped.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Shipped default_thresholds.yaml has unsupported schema_version "
            f"{shipped.get('schema_version')!r}; expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    base_defaults = _validate_metric_block(shipped.get("defaults", {}), "shipped.defaults")
    base_primitives_raw = shipped.get("primitives", {}) or {}
    base_primitives = {
        roi: _validate_metric_block(block, f"shipped.primitives.{roi}")
        for roi, block in base_primitives_raw.items()
    }

    if user_yaml is None:
        return ConformanceConfig(
            defaults=base_defaults,
            primitives=base_primitives,
            config_source="default",
        )

    user_path = Path(user_yaml)
    if not user_path.exists():
        raise FileNotFoundError(f"User threshold YAML not found: {user_path}")
    user = _load_yaml(user_path)
    if user.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{user_path}: unsupported schema_version {user.get('schema_version')!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

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

    return ConformanceConfig(
        defaults=merged_defaults,
        primitives=merged_primitives,
        config_source=str(user_path),
    )
