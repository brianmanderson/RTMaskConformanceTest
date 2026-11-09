"""Manifest schema for a generated fixture.

A ``manifest.json`` written by :func:`rtmask_conformance.generate.generate_fixture`
records exactly what a fixture contains, so the verifier (or a CI job) can
cross-check predictions filenames, ground-truth files and their hashes, and
the generator settings, without re-discovering the layout.

Version 2 layout (all paths relative to the fixture directory)::

    {
      "manifest_version": "2",
      "package_version": "0.1.0",
      "git_sha": "<40 hex chars, optionally +dirty>" | "unknown",
      "n_quadrature": 8,
      "pv_threshold": 0.5,
      "geometry": {"origin": [...], "spacing": [...], "size": [...], "direction": [...]},
      "refct_dir": "refct",
      "groundtruth_dir": "groundtruth",
      "bundles": {
        "<bundle>": {"rtstruct_path": "rtstruct/primitives_<bundle>.dcm",
                     "rtstruct_sha256": "...", "rois": ["<roi>", ...]}
      },
      "rois": {
        "<roi>": {"bundle": "<bundle>", "geometric_type": "CLOSED_PLANAR",
                  "truth_kind": "binary_mask", "graded": true,
                  "truth_path": "groundtruth/<roi>.nii.gz" | null,
                  "truth_sha256": "..." | null,
                  "spec_path": "specs/<roi>.json",
                  "expected_prediction": "<roi>.nii.gz" | null}
      }
    }

``bundles`` and ``rois`` keep canonical order. :func:`read_manifest` still
accepts a version-1 manifest (one RTSTRUCT, a flat ROI list) and upgrades it
in memory, with a :class:`ManifestDeprecationWarning`.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

MANIFEST_VERSION = "2"
SUPPORTED_MANIFEST_VERSIONS: tuple[str, ...] = ("1", "2")

UNKNOWN = "unknown"


class ManifestDeprecationWarning(FutureWarning):
    """A fixture was written with an older manifest version; regenerate it."""


@dataclass
class BundleEntry:
    rtstruct_path: str            # relative path inside fixture dir
    rtstruct_sha256: str
    rois: list[str] = field(default_factory=list)


@dataclass
class RoiEntry:
    bundle: str
    geometric_type: str
    truth_kind: str
    graded: bool
    truth_path: str | None        # relative path inside fixture dir; None if no truth file
    truth_sha256: str | None
    spec_path: str | None = None
    expected_prediction: str | None = None  # filename the tool must write; None if ungraded


@dataclass
class FixtureManifest:
    manifest_version: str
    package_version: str
    git_sha: str
    n_quadrature: int | None      # None only for an upgraded v1 manifest
    pv_threshold: float | None
    geometry: dict[str, list[float] | list[int]]  # origin, spacing, size, direction
    bundles: dict[str, BundleEntry] = field(default_factory=dict)
    rois: dict[str, RoiEntry] = field(default_factory=dict)
    groundtruth_dir: str = "groundtruth"
    refct_dir: str = "refct"

    @property
    def graded_rois(self) -> list[str]:
        """Graded ROI names in canonical order."""
        return [name for name, entry in self.rois.items() if entry.graded]

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> FixtureManifest:
        raw = dict(raw)
        raw["bundles"] = {k: BundleEntry(**v) for k, v in raw.get("bundles", {}).items()}
        raw["rois"] = {k: RoiEntry(**v) for k, v in raw.get("rois", {}).items()}
        return cls(**raw)


def write_manifest(manifest: FixtureManifest, out_dir: str | Path) -> Path:
    out = Path(out_dir) / "manifest.json"
    out.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    return out


def read_manifest(fixture_dir: str | Path) -> FixtureManifest:
    p = Path(fixture_dir) / "manifest.json"
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    version = raw.get("manifest_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(
            f"{p}: manifest_version {version!r} not supported "
            f"(expected one of {SUPPORTED_MANIFEST_VERSIONS})"
        )
    if version == "1":
        warnings.warn(
            f"{p}: manifest_version '1' is deprecated; regenerate the fixture with "
            "`rtmask-conformance generate` to record bundles, truth kinds and "
            "generator settings.",
            ManifestDeprecationWarning,
            stacklevel=2,
        )
        return _upgrade_v1(raw)
    return FixtureManifest.from_dict(raw)


def _upgrade_v1(raw: dict) -> FixtureManifest:
    """Map a v1 manifest (one planar RTSTRUCT, binary truth) onto the v2 schema."""
    from ._roi_set import BUNDLES

    known_types = {roi.name: roi.geometric_type.value for rois in BUNDLES.values() for roi in rois}
    gt_dir = raw.get("groundtruth_dir", "groundtruth")
    gt_sha = raw.get("groundtruth_sha256", {})
    rois: dict[str, RoiEntry] = {}
    for name in raw["rois"]:
        rois[name] = RoiEntry(
            bundle="planar",
            geometric_type=known_types.get(name, UNKNOWN),
            truth_kind="binary_mask",
            graded=True,
            truth_path=f"{gt_dir}/{name}.nii.gz",
            truth_sha256=gt_sha.get(name),
            spec_path=f"specs/{name}.json",
            expected_prediction=f"{name}.nii.gz",
        )
    return FixtureManifest(
        manifest_version="1",
        package_version=raw.get("package_version", UNKNOWN),
        git_sha=UNKNOWN,
        n_quadrature=None,
        pv_threshold=None,
        geometry=raw["geometry"],
        bundles={
            "planar": BundleEntry(
                rtstruct_path=raw["rtstruct_path"],
                rtstruct_sha256=raw["rtstruct_sha256"],
                rois=list(raw["rois"]),
            )
        },
        rois=rois,
        groundtruth_dir=gt_dir,
        refct_dir=raw.get("refct_dir", "refct"),
    )


def package_git_sha() -> str:
    """Best-effort git commit of the installed package, or ``"unknown"``.

    1. A ``pip install git+https://...`` records the commit in the
       distribution's ``direct_url.json`` (PEP 610).
    2. Otherwise, if the package is imported from a git checkout of this
       repository (editable install or ``PYTHONPATH=src``), ask git; a
       checkout with uncommitted changes gets a ``+dirty`` suffix.
    """
    try:
        direct_url = metadata.distribution("rtmask-conformance").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        try:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except (ValueError, AttributeError):
            commit = None
        if commit:
            return str(commit)

    pkg_dir = Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(pkg_dir), *args],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()

    try:
        toplevel = Path(_git("rev-parse", "--show-toplevel")).resolve()
        # Only trust git when the package sits in this repository's own layout,
        # not in some unrelated repository that happens to contain a venv.
        if toplevel / "src" / "rtmask_conformance" != pkg_dir:
            return UNKNOWN
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return f"{sha}+dirty" if dirty else sha
