"""Manifest schema for a generated fixture.

A ``manifest.json`` written by :func:`rtmask_conformance.generate.generate_fixture`
records exactly what a fixture contains so the verifier (or a CI job) can
cross-check predictions filenames + ground-truth filenames + their hashes
without re-discovering the layout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_VERSION = "1"


@dataclass
class FixtureManifest:
    manifest_version: str
    package_version: str
    geometry: dict[str, list[float] | list[int]]  # origin, spacing, size, direction
    rois: list[str]
    expected_predictions: list[str]
    rtstruct_path: str            # relative path inside fixture dir
    rtstruct_sha256: str
    groundtruth_dir: str          # relative path inside fixture dir
    groundtruth_sha256: dict[str, str] = field(default_factory=dict)  # roi -> sha256
    refct_dir: str = "refct"


def write_manifest(manifest: FixtureManifest, out_dir: str | Path) -> Path:
    out = Path(out_dir) / "manifest.json"
    out.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return out


def read_manifest(fixture_dir: str | Path) -> FixtureManifest:
    p = Path(fixture_dir) / "manifest.json"
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if raw.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"{p}: manifest_version {raw.get('manifest_version')!r} not supported "
            f"(expected {MANIFEST_VERSION!r})"
        )
    return FixtureManifest(**raw)
