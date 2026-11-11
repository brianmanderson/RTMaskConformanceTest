"""Manifest schema v2: write/read round trip, v1 compatibility, bundle invariants.

Pure in-memory and small-file tests; no fixture generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtmask_conformance import BUNDLES, CONFORMANCE_ROIS
from rtmask_conformance._roi_set import (
    BUNDLE_NAMES,
    TRUTH_KINDS,
    ConformanceROI,
    build_conformance_primitives,
)
from rtmask_conformance._vendor.primitives.closed_planar import Sphere
from rtmask_conformance.manifest import (
    MANIFEST_VERSION,
    BundleEntry,
    FixtureManifest,
    ManifestDeprecationWarning,
    RoiEntry,
    package_git_sha,
    read_manifest,
    write_manifest,
)

V01_ROIS = ["sphere", "cube", "cylinder", "ellipsoid", "torus", "hollow_sphere", "straw"]

GEOMETRY = {
    "origin": [0.0, 0.0, 0.0],
    "spacing": [1.0, 1.0, 1.0],
    "size": [512, 512, 200],
    "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}


def _sample_manifest() -> FixtureManifest:
    return FixtureManifest(
        manifest_version=MANIFEST_VERSION,
        package_version="0.1.0",
        git_sha="unknown",
        n_quadrature=2,
        pv_threshold=0.5,
        geometry=GEOMETRY,
        bundles={
            "planar": BundleEntry("rtstruct/primitives_planar.dcm", "a" * 64, ["sphere", "cube"]),
            "pointline": BundleEntry("rtstruct/primitives_pointline.dcm", "b" * 64, ["marker"]),
        },
        rois={
            "sphere": RoiEntry("planar", "CLOSED_PLANAR", "binary_mask", True,
                               "groundtruth/sphere.nii.gz", "c" * 64, "specs/sphere.json",
                               "sphere.nii.gz"),
            "cube": RoiEntry("planar", "CLOSED_PLANAR", "binary_mask", True,
                             "groundtruth/cube.nii.gz", "d" * 64, "specs/cube.json",
                             "cube.nii.gz"),
            "marker": RoiEntry("pointline", "POINT", "points", False, None, None,
                               "specs/marker.json", None),
        },
    )


def test_manifest_v2_round_trip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    write_manifest(manifest, tmp_path)
    back = read_manifest(tmp_path)
    assert back == manifest
    assert back.graded_rois == ["sphere", "cube"]
    assert list(back.bundles) == ["planar", "pointline"]


def test_manifest_v2_json_layout(tmp_path: Path) -> None:
    write_manifest(_sample_manifest(), tmp_path)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw["manifest_version"] == "2"
    for key in ("package_version", "git_sha", "n_quadrature", "pv_threshold", "geometry",
                "bundles", "rois", "groundtruth_dir", "refct_dir"):
        assert key in raw, key
    assert set(raw["bundles"]["planar"]) == {"rtstruct_path", "rtstruct_sha256", "rois"}
    assert set(raw["rois"]["sphere"]) == {
        "bundle", "geometric_type", "truth_kind", "graded", "truth_path", "truth_sha256",
        "spec_path", "expected_prediction",
    }


def test_manifest_v1_still_reads_with_warning(tmp_path: Path) -> None:
    v1 = {
        "manifest_version": "1",
        "package_version": "0.1.0",
        "geometry": GEOMETRY,
        "rois": V01_ROIS,
        "expected_predictions": [f"{r}.nii.gz" for r in V01_ROIS],
        "rtstruct_path": "rtstruct/primitives_planar.dcm",
        "rtstruct_sha256": "e" * 64,
        "groundtruth_dir": "groundtruth",
        "groundtruth_sha256": {r: f"{i:064x}" for i, r in enumerate(V01_ROIS)},
        "refct_dir": "refct",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(v1), encoding="utf-8")
    with pytest.warns(ManifestDeprecationWarning, match="deprecated"):
        m = read_manifest(tmp_path)
    assert m.manifest_version == "1"
    assert m.graded_rois == V01_ROIS
    assert list(m.bundles) == ["planar"]
    assert m.bundles["planar"].rtstruct_sha256 == "e" * 64
    assert m.rois["torus"].truth_path == "groundtruth/torus.nii.gz"
    assert m.rois["torus"].truth_sha256 == v1["groundtruth_sha256"]["torus"]
    assert m.rois["straw"].geometric_type == "CLOSED_PLANAR"
    assert m.n_quadrature is None and m.git_sha == "unknown"


def test_manifest_unknown_version_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_version": "99"}),
                                            encoding="utf-8")
    with pytest.raises(ValueError, match="not supported"):
        read_manifest(tmp_path)


def test_package_git_sha_is_a_string() -> None:
    sha = package_git_sha()
    assert sha == "unknown" or len(sha.split("+")[0]) == 40


def test_bundles_hold_the_v01_planar_set() -> None:
    assert list(BUNDLES) == ["planar", "pointline"]
    assert [roi.name for roi in BUNDLES["planar"]] == V01_ROIS
    assert CONFORMANCE_ROIS == V01_ROIS
    assert [p.name for p in build_conformance_primitives()] == V01_ROIS
    for name, rois in BUNDLES.items():
        assert name in BUNDLE_NAMES
        for roi in rois:
            assert roi.bundle == name
            assert roi.truth_kind in TRUTH_KINDS
            assert roi.graded
            if name == "planar":
                assert roi.truth_kind == "binary_mask"
                assert roi.geometric_type.value == "CLOSED_PLANAR"
            else:  # pointline: graded on its own metrics (test_pointline.py)
                assert roi.truth_kind in ("points", "curve")
                assert roi.geometric_type.value in ("POINT", "OPEN_PLANAR", "OPEN_NONPLANAR")


def test_conformance_roi_rejects_unknown_bundle_and_truth_kind() -> None:
    sphere = Sphere(name="s", center=(0.0, 0.0, 0.0), radius=1.0)
    with pytest.raises(ValueError, match="bundle"):
        ConformanceROI(primitive=sphere, bundle="nope", truth_kind="binary_mask", graded=True)
    with pytest.raises(ValueError, match="truth_kind"):
        ConformanceROI(primitive=sphere, bundle="planar", truth_kind="nope", graded=True)
