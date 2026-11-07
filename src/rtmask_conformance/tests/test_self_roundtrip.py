"""Self-roundtrip: rasterize the analytic primitives via the same vendored
voxelizer that built the ground truth, and confirm verify() reports Dice = 1.0.

This is the suite's own correctness test — if it ever fails, ``verify.py``
or the vendored metric/voxelizer code has a bug. It runs end-to-end
(``generate`` → re-rasterize → ``verify``) on a tiny grid so it stays under a
few seconds in CI.

The second half covers the v0.2 fixture layout on one shared fixture: the
``planar`` bundle is byte-identical to v0.1, the manifest is v2, ``verify
--fixture`` passes 7/7 on the ground truth, and tampered ground truth is
caught.

Skipped by default when ``RTMASK_CONFORMANCE_SKIP_E2E=1`` is set; otherwise
runs unconditionally.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtmask_conformance import CONFORMANCE_ROIS, load_config
from rtmask_conformance._roi_set import build_conformance_primitives
from rtmask_conformance._vendor.common.io import sitk_image_to_geometry, write_nifti
from rtmask_conformance._vendor.groundtruth.partial_volume import (
    binary_threshold,
    partial_volume_mask,
)
from rtmask_conformance._vendor.primitives.closed_planar import (
    Cube,
    Cylinder,
    Ellipsoid,
    Sphere,
    Torus,
)
from rtmask_conformance._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere
from rtmask_conformance._vendor.refimage.build_reference_ct import (
    ReferenceCTSpec,
    build_reference_ct,
)
from rtmask_conformance._vendor.rtstruct import pydicom_writer
from rtmask_conformance._vendor.rtstruct.pydicom_writer import build_rtstruct
from rtmask_conformance.cli import main as cli_main
from rtmask_conformance.generate import GenerateOptions, generate_fixture
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.verify import (
    REPORT_VERSION,
    FixtureIntegrityError,
    Status,
    verify_fixture,
    verify_predictions,
)

# SHA-256 of the v0.1 (0.1.0, commit 27c3c47) planar fixture at --n-quadrature 2
# on the default 512x512x200 1 mm grid, per platform. The RTSTRUCT writer stamps
# the wall clock into InstanceCreation/StructureSet Date and Time, so the RTSTRUCT
# hash is taken with that clock frozen at 2026-01-01 00:00:00; nothing else in
# the fixture depends on the clock. The bytes differ between Windows and Linux
# (the same on Python 3.10-3.12 within each), so there is one list per platform;
# the platform-independent proof is the in-process comparison against
# _generate_v01_reference below.
V01_PLANAR_SHA256_BY_PLATFORM: dict[str, dict[str, str]] = {
    "win32": {
        "rtstruct/primitives_planar.dcm":
            "3ac34957d99bd04ad59768b8c5d1a93db132876ff0c4d265ec976a2f7cf09e0c",
        "groundtruth/cube.nii.gz":
            "489e7c2de06989ff1410d9f809886c116bebba1a6a150f7ce6e993a99fe52c81",
        "groundtruth/cylinder.nii.gz":
            "971e6fa0378ca3f4ed62bcf619e6f8587d399442d3306fc32abe5172c07b3345",
        "groundtruth/ellipsoid.nii.gz":
            "ed54bd2993ccd4ff28dafa00a67a82c21307a3ef38c6511e31df206e1694e5b5",
        "groundtruth/hollow_sphere.nii.gz":
            "c01ebdfde7355963ba79021ab54b65dca5f762aff55031beb562b5ac97119d8b",
        "groundtruth/sphere.nii.gz":
            "437be70ae95b68d0e3c85ed384d183c3cc6864b6640b90261cf1c52ccb7840e2",
        "groundtruth/straw.nii.gz":
            "22e16ba1a518bc29672fc192abb0f0183ca88dc3c657e9896c6e2f9692885c20",
        "groundtruth/torus.nii.gz":
            "12d1f33dfed11d5fc4a6de0980f82634254018eb3ba0effc4dc2333d20ccc31f",
    },
    "linux": {
        "rtstruct/primitives_planar.dcm":
            "096d4e8b15b0b12a1b55d79f614945dd01b5724dd6e84b8f6b07a350f73634f9",
        "groundtruth/cube.nii.gz":
            "81fba83d77595c0264b30d07c7693475977be9e17cb9810592a6d1b10f33ae54",
        "groundtruth/cylinder.nii.gz":
            "77b4bfb24532851049392a9d3dbe5b5a770625cc4bca1e4b34eb60ed7eee5902",
        "groundtruth/ellipsoid.nii.gz":
            "5b4095a9c48cb1fab06ce70f7dde842cbc52f539c07d7d8fb77c033fce25a5f4",
        "groundtruth/hollow_sphere.nii.gz":
            "f6c198042ebc083f0233547692de08ce522add0b5768735ec02ab33eea5e6023",
        "groundtruth/sphere.nii.gz":
            "33824dd8a03dc897ef0c7335113ee688844bdffc360ab7cda490944685d49b23",
        "groundtruth/straw.nii.gz":
            "77d4af876687b5194bb8010caafbef5fd394312c247f04290e9c9c9cf1b04bdd",
        "groundtruth/torus.nii.gz":
            "2b6699643048ab53f82e1daf700d4dcf735f1454a4db1572f5fb83078dfbb1e7",
    },
}

V01_PLANAR_FILES: list[str] = sorted(V01_PLANAR_SHA256_BY_PLATFORM["win32"])


class _FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_clock() -> pytest.MonkeyPatch:
    mp = pytest.MonkeyPatch()
    mp.setattr(pydicom_writer, "datetime", types.SimpleNamespace(datetime=_FrozenDateTime))
    return mp


def _generate_v01_reference(out: Path, n_quadrature: int) -> None:
    """The v0.1 generator's RTSTRUCT and ground-truth steps, frozen as of 27c3c47.

    Kept independent of ``_roi_set`` and ``generate`` on purpose: the v0.2
    generator must reproduce these bytes on whatever platform the test runs.
    """
    primitives = [
        Sphere(name="sphere", center=(256.0, 256.0, 100.0), radius=40.0),
        Cube(name="cube", center=(100.5, 100.5, 100.5), side_length=60.0),
        Cylinder(name="cylinder", center=(400.0, 100.0, 100.5), radius=30.0, height=80.0),
        Ellipsoid(name="ellipsoid", center=(100.0, 400.0, 100.0), semi_axes=(30.0, 50.0, 60.0)),
        Torus(name="torus", center=(400.0, 400.0, 100.0), major_radius=60.0, minor_radius=20.0),
        HollowSphere(name="hollow_sphere", center=(256.0, 100.0, 100.0),
                     outer_radius=40.0, inner_radius=20.0),
        HollowCylinder(name="straw", center=(256.0, 400.0, 100.5),
                       outer_radius=40.0, inner_radius=20.0, height=120.0),
    ]
    refct_dir = out / "refct"
    for d in (refct_dir, out / "rtstruct", out / "groundtruth"):
        d.mkdir(parents=True, exist_ok=True)
    geometry = build_reference_ct(
        refct_dir, spec=ReferenceCTSpec(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200))
    )
    build_rtstruct(
        primitives=primitives,
        ref_image_folder=refct_dir,
        out_path=out / "rtstruct" / "primitives_planar.dcm",
        structure_set_label="RTMASK_CONFORM",
    )
    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        write_nifti(binary, geometry, out / "groundtruth" / f"{primitive.name}.nii.gz")


pytestmark = pytest.mark.skipif(
    os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set",
)


def test_self_roundtrip(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    # Primitive positions are calibrated to the canonical 512x512x200 grid; smaller
    # grids would clip several ROIs. Drop n_quadrature to keep this end-to-end
    # test under a minute on a typical CI runner.
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)

    # Rasterize each primitive on the same geometry as the GT and dump as a
    # prediction. By construction these masks are byte-identical to the GT.
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    primitives = build_conformance_primitives()
    gt_dir = fixture / "groundtruth"

    # Pull the geometry back from the GT NIfTI rather than reconstructing it,
    # since a real consumer reads it that way.
    sample = sitk.ReadImage(str(gt_dir / f"{primitives[0].name}.nii.gz"))
    geometry = sitk_image_to_geometry(sample)

    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=options.n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        write_nifti(binary, geometry, pred_dir / f"{primitive.name}.nii.gz")

    report = verify_predictions(pred_dir, gt_dir, load_config())
    assert report.overall_pass, [r.violations for r in report.results if r.status != Status.PASS]
    for record in report.results:
        assert record.status == Status.PASS, (record.roi, record.violations)
        assert record.metrics["dice"] == pytest.approx(1.0)


def test_all_seven_rois_present_in_groundtruth(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)
    gt_dir = fixture / "groundtruth"
    for roi in CONFORMANCE_ROIS:
        assert (gt_dir / f"{roi}.nii.gz").is_file(), roi


# --------------------------------------------------------------------------- #
#  v0.2 fixture layout, on one shared fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def v02_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture = tmp_path_factory.mktemp("v02") / "fixture"
    mp = _frozen_clock()
    try:
        generate_fixture(fixture, options=GenerateOptions(n_quadrature=2))
    finally:
        mp.undo()
    return fixture


@pytest.fixture(scope="module")
def v01_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v01") / "reference"
    mp = _frozen_clock()
    try:
        _generate_v01_reference(out, n_quadrature=2)
    finally:
        mp.undo()
    return out


def test_planar_bundle_byte_identical_to_v01(v02_fixture: Path, v01_reference: Path) -> None:
    expected = {rel: _sha256(v01_reference / rel) for rel in V01_PLANAR_FILES}
    actual = {rel: _sha256(v02_fixture / rel) for rel in V01_PLANAR_FILES}
    assert actual == expected
    assert sorted(p.name for p in (v02_fixture / "rtstruct").iterdir()) == [
        "primitives_planar.dcm"
    ]
    assert sorted(p.name for p in (v02_fixture / "groundtruth").iterdir()) == sorted(
        f"{roi}.nii.gz" for roi in CONFORMANCE_ROIS
    )


def test_planar_bundle_matches_pinned_v01_hashes(v02_fixture: Path) -> None:
    pinned = V01_PLANAR_SHA256_BY_PLATFORM.get(sys.platform)
    if pinned is None:
        pytest.skip(f"no pinned v0.1 hashes for platform {sys.platform!r}")
    actual = {rel: _sha256(v02_fixture / rel) for rel in pinned}
    assert actual == pinned, (
        "planar fixture bytes differ from the pinned v0.1 hashes for this platform; "
        "if test_planar_bundle_byte_identical_to_v01 passes, a dependency (numpy, "
        "SimpleITK, pydicom, zlib) changed the output, not this package"
    )


def test_manifest_v2_describes_the_fixture(v02_fixture: Path) -> None:
    m = read_manifest(v02_fixture)
    assert m.manifest_version == "2"
    assert m.n_quadrature == 2 and m.pv_threshold == 0.5
    assert m.git_sha
    assert list(m.bundles) == ["planar"]
    planar = m.bundles["planar"]
    assert planar.rtstruct_path == "rtstruct/primitives_planar.dcm"
    assert planar.rtstruct_sha256 == _sha256(v02_fixture / planar.rtstruct_path)
    assert planar.rois == list(CONFORMANCE_ROIS)
    assert m.graded_rois == list(CONFORMANCE_ROIS)
    for roi, entry in m.rois.items():
        assert entry.bundle == "planar"
        assert entry.geometric_type == "CLOSED_PLANAR"
        assert entry.truth_kind == "binary_mask" and entry.graded
        assert entry.truth_path == f"groundtruth/{roi}.nii.gz"
        assert entry.truth_sha256 == _sha256(v02_fixture / entry.truth_path)
        assert entry.expected_prediction == f"{roi}.nii.gz"
        spec = json.loads((v02_fixture / entry.spec_path).read_text(encoding="utf-8"))
        assert spec["bundle"] == "planar"
        assert spec["geometric_type"] == "CLOSED_PLANAR"
        assert spec["truth_kind"] == "binary_mask"
        assert spec["graded"] is True
        assert spec["pv_quadrature_per_axis"] == 2


def test_verify_fixture_ground_truth_passes(v02_fixture: Path) -> None:
    report = verify_fixture(v02_fixture / "groundtruth", v02_fixture, load_config())
    assert report.summary["total"] == 7
    assert report.summary["passed"] == 7, [r.violations for r in report.results]
    assert [r.roi for r in report.results] == list(CONFORMANCE_ROIS)
    d = report.as_dict()
    assert d["version"] == REPORT_VERSION == "2.0"
    assert d["fixture"]["manifest_version"] == "2"
    assert d["fixture"]["n_quadrature"] == 2
    assert d["fixture"]["bundles"] == ["planar"]
    assert d["fixture"]["package_version"]
    assert d["fixture"]["git_sha"]


@pytest.fixture
def tampered_fixture(v02_fixture: Path, tmp_path: Path) -> Path:
    """A copy of the manifest + ground truth with one voxel of ``cube`` flipped."""
    copy = tmp_path / "tampered"
    copy.mkdir()
    shutil.copyfile(v02_fixture / "manifest.json", copy / "manifest.json")
    shutil.copytree(v02_fixture / "groundtruth", copy / "groundtruth")
    target = copy / "groundtruth" / "cube.nii.gz"
    img = sitk.ReadImage(str(target))
    arr = sitk.GetArrayFromImage(img)
    arr[0, 0, 0] = 1 - arr[0, 0, 0]
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(img)
    sitk.WriteImage(out, str(target), True)
    return copy


def test_verify_fixture_detects_tampered_truth(tampered_fixture: Path) -> None:
    with pytest.raises(FixtureIntegrityError, match=r"cube\.nii\.gz.*[Rr]e-run"):
        verify_fixture(tampered_fixture / "groundtruth", tampered_fixture, load_config())


def test_verify_fixture_unknown_bundle_rejected(v02_fixture: Path) -> None:
    with pytest.raises(ValueError, match="not in this fixture"):
        verify_fixture(v02_fixture / "groundtruth", v02_fixture, bundles=["keyhole"])


def test_cli_verify_fixture_error_paths(
    v02_fixture: Path, tampered_fixture: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = str(v02_fixture / "groundtruth")
    assert cli_main(["verify", "--predictions", pred, "--fixture", str(tampered_fixture)]) == 2
    assert "does not match the manifest" in capsys.readouterr().err
    assert cli_main(["verify", "--predictions", pred, "--fixture", str(v02_fixture),
                     "--bundle", "keyhole"]) == 2
    assert cli_main(["verify", "--predictions", pred, "--groundtruth", pred,
                     "--bundle", "planar"]) == 2
    assert "--bundle requires --fixture" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # --fixture and --groundtruth are mutually exclusive
        cli_main(["verify", "--predictions", pred, "--fixture", str(v02_fixture),
                  "--groundtruth", pred])


def _load_sync_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "sync_from_upstream.py"
    if not path.is_file():
        pytest.skip("tools/ not present (installed package, not a checkout)")
    spec = importlib.util.spec_from_file_location("sync_from_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_tool_refuses_to_write_outside_vendor(tmp_path: Path) -> None:
    sync = _load_sync_tool()
    vendor = tmp_path / "src" / "rtmask_conformance" / "_vendor"
    version_file = tmp_path / "tools" / "UPSTREAM_VERSION.txt"
    assert sync._guarded(vendor / "common" / "io.py", vendor, version_file)
    assert sync._guarded(version_file, vendor, version_file)
    for bad in (vendor.parent / "primitives" / "__init__.py",
                vendor.parent / "verify.py",
                vendor / ".." / "generate.py"):
        with pytest.raises(sync.UnsafeDestinationError):
            sync._guarded(bad, vendor, version_file)
