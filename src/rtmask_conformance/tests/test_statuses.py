"""Verdict statuses, the ``claims:`` / ``report_only:`` YAML fields and report schema 2.0.

Most tests run on a small hand-built fixture (a 24^3 grid, four ROIs across two
bundles, one of them ``CLOSEDPLANAR_XOR``) so they take milliseconds. The last
test pins the status list of the seven v0.1 primitives under the default YAML
on a generated fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtmask_conformance import CONFORMANCE_ROIS, __version__, load_config
from rtmask_conformance.cli import main as cli_main
from rtmask_conformance.manifest import (
    MANIFEST_VERSION,
    BundleEntry,
    FixtureManifest,
    RoiEntry,
    write_manifest,
)
from rtmask_conformance.report import render_text
from rtmask_conformance.thresholds import STANDARD_GEOMETRIC_TYPES
from rtmask_conformance.verify import (
    FAILING_STATUSES,
    REPORT_VERSION,
    Status,
    evaluate_one,
    verify_fixture,
    verify_predictions,
)

SHAPE = (24, 24, 24)


def _block(lo: int, hi: int) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = 1
    return arr


def _ring() -> np.ndarray:
    arr = _block(4, 20)
    arr[8:16, 8:16, 8:16] = 0
    return arr


def _write(arr: np.ndarray, path: Path, origin=(0.0, 0.0, 0.0)) -> Path:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(path), True)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: name -> (bundle, geometric_type, truth_kind, graded, truth array or None)
ROIS = {
    "box": ("planar", "CLOSED_PLANAR", "binary_mask", True, lambda: _block(6, 18)),
    "probe": ("planar", "CLOSED_PLANAR", "binary_mask", False, lambda: _block(8, 16)),
    "ring_xor": ("xor", "CLOSEDPLANAR_XOR", "binary_mask", True, _ring),
    "marker": ("pointline", "POINT", "none", False, None),
}


def _make_fixture(root: Path, package_version: str = __version__) -> Path:
    gt = root / "groundtruth"
    gt.mkdir(parents=True)
    rois: dict[str, RoiEntry] = {}
    bundles: dict[str, BundleEntry] = {}
    for name, (bundle, gtype, kind, graded, make) in ROIS.items():
        truth_path = truth_sha = None
        if make is not None:
            p = _write(make(), gt / f"{name}.nii.gz")
            truth_path, truth_sha = f"groundtruth/{name}.nii.gz", _sha(p)
        rois[name] = RoiEntry(
            bundle=bundle, geometric_type=gtype, truth_kind=kind, graded=graded,
            truth_path=truth_path, truth_sha256=truth_sha,
            expected_prediction=f"{name}.nii.gz" if graded else None,
        )
        bundles.setdefault(bundle, BundleEntry(f"rtstruct/primitives_{bundle}.dcm", "0" * 64))
        bundles[bundle].rois.append(name)
    write_manifest(FixtureManifest(
        manifest_version=MANIFEST_VERSION, package_version=package_version, git_sha="unknown",
        n_quadrature=2, pv_threshold=0.5,
        geometry={"origin": [0.0] * 3, "spacing": [1.0] * 3, "size": list(SHAPE),
                  "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
        bundles=bundles, rois=rois,
    ), root)
    return root


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    return _make_fixture(tmp_path / "fixture")


def _predictions(fixture: Path, tmp_path: Path, rois: list[str]) -> Path:
    pred = tmp_path / "pred"
    pred.mkdir(exist_ok=True)
    for roi in rois:
        shutil.copyfile(fixture / "groundtruth" / f"{roi}.nii.gz", pred / f"{roi}.nii.gz")
    return pred


def _yaml(tmp_path: Path, text: str, name: str = "conformance.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _statuses(report) -> dict[str, str]:
    return {r.roi: r.status.value for r in report.results}


NO_XOR = "schema_version: 2\nclaims:\n  geometric_types: [CLOSED_PLANAR]\n"


# ---------------------------------------------------------------------------
# claims -> UNSUPPORTED_TYPE
# ---------------------------------------------------------------------------


def test_unclaimed_missing_type_is_unsupported_and_passes(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    config = load_config(_yaml(tmp_path, NO_XOR))
    report = verify_fixture(pred, fixture, config)

    assert _statuses(report) == {"box": "PASS", "probe": "REPORT_ONLY",
                                 "ring_xor": "UNSUPPORTED_TYPE"}
    assert report.summary == {"total": 3, "passed": 1, "failed": 0, "missing": 0,
                              "geometry_mismatch": 0, "unsupported_type": 1, "report_only": 1}
    assert report.overall_pass
    assert cli_main(["verify", "--predictions", str(pred), "--fixture", str(fixture),
                     "--config", str(tmp_path / "conformance.yaml")]) == 0


def test_claimed_missing_type_is_missing_and_fails(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    report = verify_fixture(pred, fixture, load_config())  # default claims all five

    assert _statuses(report)["ring_xor"] == "MISSING"
    assert not report.overall_pass
    assert cli_main(["verify", "--predictions", str(pred), "--fixture", str(fixture)]) == 1


def test_unclaimed_type_with_a_prediction_is_graded(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box"])
    _write(_block(4, 20), pred / "ring_xor.nii.gz")  # inner hole filled: the solid-fill bug
    report = verify_fixture(pred, fixture, load_config(_yaml(tmp_path, NO_XOR)))
    assert _statuses(report)["ring_xor"] == "FAIL"
    assert not report.overall_pass


# ---------------------------------------------------------------------------
# report_only -> REPORT_ONLY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["box", "CLOSED_PLANAR"])
def test_report_only_scores_but_never_fails(fixture: Path, tmp_path: Path, entry: str) -> None:
    pred = _predictions(fixture, tmp_path, ["ring_xor"])
    _write(np.roll(_block(6, 18), 4, axis=2), pred / "box.nii.gz")  # would FAIL if graded
    config = load_config(_yaml(tmp_path, f"schema_version: 2\nreport_only: [{entry}]\n"))
    report = verify_fixture(pred, fixture, config)

    box = next(r for r in report.results if r.roi == "box")
    assert box.status == Status.REPORT_ONLY
    assert box.metrics["dice"] is not None and box.metrics["dice"] < 0.95
    assert box.violations  # what grading would have said, for information
    assert report.summary["report_only"] == 2  # box by config, probe by graded: false
    assert report.overall_pass
    assert report.report_only == [entry]


def test_report_only_missing_or_mismatched_never_fails(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["ring_xor"])
    _write(_block(8, 16), pred / "probe.nii.gz", origin=(0.5, 0.0, 0.0))  # geometry mismatch
    config = load_config(_yaml(tmp_path, "schema_version: 2\nreport_only: [box]\n"))
    report = verify_fixture(pred, fixture, config)

    records = {r.roi: r for r in report.results}
    assert records["box"].status == Status.REPORT_ONLY  # absent
    assert any("not found" in v for v in records["box"].violations)
    assert records["probe"].status == Status.REPORT_ONLY  # graded: false, mismatched
    assert "origin" in records["probe"].geometry_diagnostic
    assert report.overall_pass


def test_ungraded_roi_without_binary_truth_is_skipped(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe", "ring_xor"])
    report = verify_fixture(pred, fixture, load_config())
    assert "marker" not in _statuses(report)
    assert _statuses(report) == {"box": "PASS", "probe": "REPORT_ONLY", "ring_xor": "PASS"}


def test_unsupported_takes_precedence_over_report_only(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    config = load_config(_yaml(tmp_path, NO_XOR + "report_only: [ring_xor]\n"))
    assert _statuses(verify_fixture(pred, fixture, config))["ring_xor"] == "UNSUPPORTED_TYPE"


def test_failing_statuses_are_exactly_three() -> None:
    assert FAILING_STATUSES == {Status.FAIL, Status.MISSING, Status.GEOMETRY_MISMATCH}
    assert [s.value for s in Status] == ["PASS", "FAIL", "MISSING", "GEOMETRY_MISMATCH",
                                         "UNSUPPORTED_TYPE", "REPORT_ONLY"]


def test_evaluate_one_looks_up_the_suite_geometric_type(tmp_path: Path) -> None:
    gt = _write(_block(6, 18), tmp_path / "sphere.nii.gz")
    config = load_config(_yaml(tmp_path, "schema_version: 2\nclaims:\n"
                                         "  geometric_types: [POINT]\n"))
    rec = evaluate_one("sphere", None, gt, config)
    assert rec.status == Status.UNSUPPORTED_TYPE
    assert rec.geometric_type == "CLOSED_PLANAR"
    # A ROI the suite does not know has no type, and counts as claimed; so does a
    # type outside the five standard values ("unknown" from an upgraded v1 manifest).
    assert evaluate_one("not_a_roi", None, gt, config).status == Status.MISSING
    rec = evaluate_one("sphere", None, gt, config, geometric_type="unknown")
    assert rec.status == Status.MISSING


# ---------------------------------------------------------------------------
# The YAML schema
# ---------------------------------------------------------------------------


def test_shipped_default_claims_everything_and_reports_nothing() -> None:
    config = load_config()
    assert config.claimed_types == STANDARD_GEOMETRIC_TYPES
    assert len(STANDARD_GEOMETRIC_TYPES) == 5
    assert config.report_only == ()


def test_schema_1_still_loads_and_claims_everything(tmp_path: Path) -> None:
    config = load_config(_yaml(tmp_path, "schema_version: 1\nprimitives:\n"
                                         "  torus:\n    dice: 0.90\n"))
    assert config.claimed_types == STANDARD_GEOMETRIC_TYPES
    assert config.report_only == ()
    assert config.thresholds_for("torus").dice == 0.90


def test_user_claims_replace_and_report_only_adds(tmp_path: Path) -> None:
    config = load_config(_yaml(tmp_path, "schema_version: 2\nclaims:\n"
                                         "  geometric_types: [POINT, CLOSED_PLANAR, POINT]\n"
                                         "report_only: [torus, OPEN_PLANAR, torus]\n"))
    assert config.claimed_types == ("CLOSED_PLANAR", "POINT")  # standard order, deduplicated
    assert config.report_only == ("torus", "OPEN_PLANAR")
    assert config.claims("POINT") and not config.claims("CLOSEDPLANAR_XOR")
    assert config.is_report_only("torus") and config.is_report_only("x", "OPEN_PLANAR")
    assert not config.is_report_only("sphere", "CLOSED_PLANAR")


def test_claims_without_geometric_types_claims_everything(tmp_path: Path) -> None:
    config = load_config(_yaml(tmp_path, "schema_version: 2\nclaims: {}\n"))
    assert config.claimed_types == STANDARD_GEOMETRIC_TYPES


def test_per_primitive_block_may_name_any_roi(tmp_path: Path) -> None:
    config = load_config(_yaml(tmp_path, "schema_version: 2\nprimitives:\n"
                                         "  not_yet_shipped:\n    hd95_mm: 3.0\n"))
    assert config.thresholds_for("not_yet_shipped").hd95_mm == 3.0


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("schema_version: 1\nclaims:\n  geometric_types: [POINT]\n", "require schema_version: 2"),
        ("schema_version: 1\nreport_only: [torus]\n", "require schema_version: 2"),
        ("schema_version: 3\n", "unsupported schema_version"),
        ("schema_version: 2\nclaims:\n  geometric_types: [CLOSED_NONPLANAR]\n",
         "not a Contour Geometric Type"),
        ("schema_version: 2\nclaims:\n  geometric_types: []\n", "at least one"),
        ("schema_version: 2\nclaims:\n  geometric_types: POINT\n", "list of strings"),
        ("schema_version: 2\nclaims:\n  types: [POINT]\n", "unknown key"),
        ("schema_version: 2\nclaims: [POINT]\n", "expected a mapping"),
        ("schema_version: 2\nreport_only: torus\n", "list of ROI names"),
    ],
)
def test_bad_schema_2_fields_rejected(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(_yaml(tmp_path, text))


def test_cli_rejects_bad_claims_with_exit_2(fixture: Path, tmp_path: Path, capsys) -> None:
    bad = _yaml(tmp_path, "schema_version: 2\nclaims:\n  geometric_types: [XOR]\n")
    assert cli_main(["verify", "--predictions", str(fixture / "groundtruth"),
                     "--fixture", str(fixture), "--config", str(bad)]) == 2
    assert "not a Contour Geometric Type" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_json_report_schema_2(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    out = tmp_path / "report.json"
    assert cli_main(["verify", "--predictions", str(pred), "--fixture", str(fixture),
                     "--config", str(_yaml(tmp_path, NO_XOR)), "--report-json", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["version"] == REPORT_VERSION == "2.0"
    assert d["claims"] == {"geometric_types": ["CLOSED_PLANAR"]}
    assert d["report_only"] == []
    assert d["fixture"]["bundles"] == ["planar", "xor", "pointline"]
    assert set(d["summary"]) == {"total", "passed", "failed", "missing", "geometry_mismatch",
                                 "unsupported_type", "report_only"}
    by_roi = {r["roi"]: r for r in d["results"]}
    assert by_roi["ring_xor"]["status"] == "UNSUPPORTED_TYPE"
    assert by_roi["ring_xor"]["geometric_type"] == "CLOSEDPLANAR_XOR"
    assert all(r["truth_kind"] == "binary_mask" for r in d["results"])


def test_text_report_badges(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    text = render_text(verify_fixture(pred, fixture, load_config(_yaml(tmp_path, NO_XOR))))
    assert "UNSUP " in text and "REPORT " in text
    assert "UNSUPPORTED_TYPE: 1  (type not claimed; not a failure)" in text
    assert "REPORT_ONLY: 1  (scored, not graded; not a failure)" in text
    assert "probe [REPORT_ONLY]  (for information; not graded)" in text


def test_groundtruth_form_reports_claims(fixture: Path, tmp_path: Path) -> None:
    report = verify_predictions(fixture / "groundtruth", fixture / "groundtruth",
                                load_config(), rois=["box"])
    assert report.fixture is None
    assert report.claims == {"geometric_types": list(STANDARD_GEOMETRIC_TYPES)}
    assert _statuses(report) == {"box": "PASS"}


# ---------------------------------------------------------------------------
# The pytest harness (test_conformance.py) maps statuses onto pytest outcomes
# ---------------------------------------------------------------------------


def _run_harness(fixture: Path, pred: Path, config: Path | None) -> str:
    env = dict(os.environ, RTMASK_CONFORMANCE_PREDICTIONS=str(pred),
               RTMASK_CONFORMANCE_FIXTURE=str(fixture))
    env.pop("RTMASK_CONFORMANCE_GROUNDTRUTH", None)
    env.pop("RTMASK_CONFORMANCE_CONFIG", None)
    if config is not None:
        env["RTMASK_CONFORMANCE_CONFIG"] = str(config)
    harness = Path(__file__).with_name("test_conformance.py")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(harness), "-rA", "-p", "no:cacheprovider"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    return proc.stdout + proc.stderr


def test_harness_skips_unsupported_and_report_only(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    out = _run_harness(fixture, pred, _yaml(tmp_path, NO_XOR))
    assert "1 passed, 2 skipped" in out, out
    assert "ring_xor: geometric type CLOSEDPLANAR_XOR not claimed" in out, out
    assert "probe: report-only, not graded" in out, out


def test_harness_fails_a_claimed_missing_type(fixture: Path, tmp_path: Path) -> None:
    pred = _predictions(fixture, tmp_path, ["box", "probe"])
    out = _run_harness(fixture, pred, None)
    assert "1 failed, 1 passed, 1 skipped" in out, out
    assert "FAILED" in out and "test_conformance[ring_xor]" in out, out


# ---------------------------------------------------------------------------
# Version string and the fixture-version warning
# ---------------------------------------------------------------------------


def test_version_is_a_pre_release_of_0_2_0() -> None:
    assert __version__ == "0.2.0.dev0"


def test_same_pre_release_fixture_does_not_warn(fixture: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        verify_fixture(fixture / "groundtruth", fixture, load_config(), bundles=["planar"])


@pytest.mark.parametrize("old", ["0.1.0", "0.2.0.dev1", "unknown"])
def test_other_version_fixture_warns(tmp_path: Path, old: str) -> None:
    fx = _make_fixture(tmp_path / "old", package_version=old)
    with pytest.warns(UserWarning, match=f"generated by rtmask-conformance {old}; "
                                         f"verifying with {__version__}"):
        report = verify_fixture(fx / "groundtruth", fx, load_config(), bundles=["planar"])
    assert report.overall_pass


# ---------------------------------------------------------------------------
# The seven v0.1 primitives: status list pinned under the default YAML
# ---------------------------------------------------------------------------


#: The v0.1 status of each primitive when predictions are: ground truth for all
#: but the first four, zeroed, shifted 5 voxels in x, shifted 0.5 mm in origin,
#: and absent. Recorded from v0.1 (commit 70b8444) and unchanged since.
V01_NEGATIVE_STATUSES = [
    ("sphere", "FAIL"),
    ("cube", "FAIL"),
    ("cylinder", "GEOMETRY_MISMATCH"),
    ("ellipsoid", "MISSING"),
    ("torus", "PASS"),
    ("hollow_sphere", "PASS"),
    ("straw", "PASS"),
]


@pytest.mark.skipif(os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
                    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set")
def test_v01_primitives_status_list_unchanged(tmp_path: Path) -> None:
    from rtmask_conformance.generate import GenerateOptions, generate_fixture

    fx = tmp_path / "fixture"
    generate_fixture(fx, options=GenerateOptions(n_quadrature=2))
    gt = fx / "groundtruth"
    pred = tmp_path / "pred"
    pred.mkdir()
    rois = list(CONFORMANCE_ROIS)
    for roi in rois[4:]:
        shutil.copyfile(gt / f"{roi}.nii.gz", pred / f"{roi}.nii.gz")
    img = sitk.ReadImage(str(gt / f"{rois[0]}.nii.gz"))
    zero = sitk.GetImageFromArray(np.zeros_like(sitk.GetArrayFromImage(img)))
    zero.CopyInformation(img)
    sitk.WriteImage(zero, str(pred / f"{rois[0]}.nii.gz"), True)
    img = sitk.ReadImage(str(gt / f"{rois[1]}.nii.gz"))
    shifted = sitk.GetImageFromArray(np.roll(sitk.GetArrayFromImage(img), 5, axis=2))
    shifted.CopyInformation(img)
    sitk.WriteImage(shifted, str(pred / f"{rois[1]}.nii.gz"), True)
    img = sitk.ReadImage(str(gt / f"{rois[2]}.nii.gz"))
    origin = list(img.GetOrigin())
    origin[0] += 0.5
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(pred / f"{rois[2]}.nii.gz"), True)

    for report in (verify_fixture(pred, fx, load_config()),
                   verify_predictions(pred, gt, load_config())):
        assert [(r.roi, r.status.value) for r in report.results] == V01_NEGATIVE_STATUSES
        assert report.summary["unsupported_type"] == report.summary["report_only"] == 0
        assert not report.overall_pass
