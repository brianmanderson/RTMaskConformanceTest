"""The ``pointline`` bundle: POINT and open-curve ROIs, graded on point and curve metrics.

The OPEN_PLANAR curves' own geometry and ceilings are tested in
``test_open_planar.py``; here they ride along in the bundle-wide checks.

The first half needs no fixture: the analytic truth records, the suite's
reference rasterisation, the ``pointline:`` threshold block and the grading
rules on hand-made masks. The second half generates one full fixture and runs
``verify --fixture`` on predictions from an independent rasteriser written
here: a 3D Bresenham trace of each contour read back from
``primitives_pointline.dcm``, i.e. a thin-path rasteriser working from the DICOM
file a tool would read. It also checks that a curve filled into a polygon
fails on arc length and that a seed implant with one seed missing fails on
recall.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from rtmask_conformance import grading_pointline as gp
from rtmask_conformance import load_config
from rtmask_conformance._roi_set import BUNDLES, CONFORMANCE_ROIS, _open_planar_rois
from rtmask_conformance._vendor.common.coords import Geometry
from rtmask_conformance._vendor.common.io import write_nifti
from rtmask_conformance.generate import GenerateOptions, generate_fixture
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.report import render_text
from rtmask_conformance.verify import (
    FixtureIntegrityError,
    Status,
    evaluate_one,
    fixture_targets,
    verify_fixture,
)

GEOMETRY = Geometry(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), size=(512, 512, 200))
POINTLINE = {roi.name: roi for roi in BUNDLES["pointline"]}
POINT_ROIS = [n for n, r in POINTLINE.items() if r.truth_kind == "points"]
CURVE_ROIS = [n for n, r in POINTLINE.items() if r.truth_kind == "curve"]
HELIX = "helix_r12_pitch6_4turns_x300_y200"
SEEDS = "ldr_seed_implant_x350_y350"


def _truth(name: str) -> dict:
    roi = POINTLINE[name]
    return gp.build_truth(roi.primitive, roi.truth_kind, GEOMETRY)


def _grade(mask: np.ndarray, name: str, config_source: str = "default") -> list[str]:
    truth = _truth(name)
    metrics = gp.compute_metrics(mask, truth, GEOMETRY)
    return gp.violations(metrics, gp.thresholds_for(config_source, name, truth["truth_kind"]))


def _filled_per_slice(truth: dict) -> np.ndarray:
    """Each slice the curve crosses, filled inside the polygon spanned by ALL its vertices.

    Projecting the whole contour onto every slice it touches and filling it is the
    failure mode this bundle exists to catch: an open curve rasterised as a closed
    polygon. For the helix that fills the disc under it on each of its slices.
    """
    from skimage.draw import polygon

    mask = np.zeros(GEOMETRY.shape_zyx, dtype=np.uint8)
    for poly in truth["polylines_mm"]:
        idx = GEOMETRY.physical_to_index(np.asarray(poly))
        rr, cc = polygon(idx[:, 1], idx[:, 0], shape=mask.shape[1:])
        for z in np.unique(np.rint(idx[:, 2]).astype(int)):
            mask[z, rr, cc] = 1
    return mask


# --------------------------------------------------------------------------- #
#  Registration and analytic truth
# --------------------------------------------------------------------------- #


def test_pointline_bundle_registration() -> None:
    assert POINT_ROIS == [
        "single_point_x256_y256_z150",
        "needle_track_10pts_x150_y150",
        "ldr_seed_implant_x350_y350",
    ]
    open_nonplanar = [
        "diagonal_needle_x100_y200_z40",
        "helix_r12_pitch6_4turns_x300_y200",
        "sinusoid_x150_y350_z20to80",
        "syed_bundle_5needles_x400_y100_z50",
    ]
    # The two OPEN_PLANAR curves (test_open_planar.py) follow the upstream seven.
    open_planar = [roi.name for roi in _open_planar_rois()]
    assert CURVE_ROIS == open_nonplanar + open_planar
    assert list(POINTLINE) == POINT_ROIS + open_nonplanar + open_planar
    for name, roi in POINTLINE.items():
        assert roi.graded and roi.bundle == "pointline"
        if roi.truth_kind == "points":
            expected = "POINT"
        else:
            expected = "OPEN_PLANAR" if name in open_planar else "OPEN_NONPLANAR"
        assert roi.geometric_type.value == expected, name
        # The v0.1 bare-groundtruth API lists binary-mask ROIs only.
        assert name not in CONFORMANCE_ROIS


def test_point_truth_matches_the_primitive() -> None:
    counts = {"single_point_x256_y256_z150": 1, "needle_track_10pts_x150_y150": 10, SEEDS: 147}
    for name, n in counts.items():
        truth = _truth(name)
        assert truth["n_points"] == n == len(truth["points_mm"])
        np.testing.assert_array_equal(truth["points_mm"], POINTLINE[name].primitive.points)
    assert _truth("single_point_x256_y256_z150")["points_mm"] == [[256.0, 256.0, 150.0]]
    track = np.asarray(_truth("needle_track_10pts_x150_y150")["points_mm"])
    np.testing.assert_allclose(track[0], (140.0, 140.0, 130.0))
    np.testing.assert_allclose(track[-1], (170.0, 170.0, 175.0))


def test_curve_truth_arc_lengths_and_sampling() -> None:
    expected_length = {
        "diagonal_needle_x100_y200_z40": math.dist((80, 180, 30), (160, 240, 70)),
        HELIX: 4 * math.hypot(2 * math.pi * 12, 6),
        "syed_bundle_5needles_x400_y100_z50": 5 * 60.0,
    }
    for name in CURVE_ROIS:
        truth = _truth(name)
        if name in expected_length:
            assert truth["analytical_arc_length_mm"] == pytest.approx(expected_length[name], 1e-12)
        n_curves = 5 if name.startswith("syed") else 1
        assert truth["n_curves"] == len(truth["polylines_mm"]) == n_curves
        assert len(truth["endpoints_mm"]) == 2 * n_curves
        for poly in truth["polylines_mm"]:
            steps = np.linalg.norm(np.diff(np.asarray(poly), axis=0), axis=1)
            assert steps.max() <= gp.DENSIFY_MAX_STEP_MM + 1e-12
    # The sinusoid's arc length is the integral of the continuous curve, not its polyline.
    sin = _truth("sinusoid_x150_y350_z20to80")
    assert sin["arc_length_source"] == "continuous_arc_length_mm"
    assert sin["analytical_arc_length_mm"] == pytest.approx(110.5495, abs=1e-3)


def test_truth_json_round_trips(tmp_path: Path) -> None:
    for name in (SEEDS, HELIX):
        truth = _truth(name)
        path = tmp_path / f"{name}.json"
        gp.write_truth(truth, path)
        assert gp.read_truth(path) == truth
    (tmp_path / "bad.json").write_text(json.dumps({"truth_kind": "binary_mask"}))
    with pytest.raises(ValueError, match="not a curve or point truth"):
        gp.read_truth(tmp_path / "bad.json")


# --------------------------------------------------------------------------- #
#  Thresholds
# --------------------------------------------------------------------------- #


def test_shipped_block_covers_every_pointline_roi() -> None:
    shipped = gp.load_pointline_thresholds()
    for name, roi in POINTLINE.items():
        block = gp.thresholds_for("default", name, roi.truth_kind)
        assert block, name
        assert set(block) <= set(gp.METRIC_KEYS_BY_KIND[roi.truth_kind])
        assert shipped[name] == block
    for name in POINT_ROIS:
        assert gp.thresholds_for("default", name, "points")["point_recall"] == 1.0
        assert "point_fp_rate" not in gp.thresholds_for("default", name, "points")
    for name in CURVE_ROIS:
        assert "arc_length_error_pct" in gp.thresholds_for("default", name, "curve")
    # The pointline block never reaches the planar resolver.
    assert load_config().primitives.keys().isdisjoint(POINTLINE)


def _user_yaml(tmp_path: Path, body: str) -> str:
    path = tmp_path / "user.yaml"
    path.write_text("schema_version: 2\n" + body, encoding="utf-8")
    return str(path)


def test_user_pointline_block_merges_per_roi(tmp_path: Path) -> None:
    src = _user_yaml(tmp_path, f"pointline:\n  {HELIX}:\n    arc_length_error_pct: 99\n"
                               f"    curve_coverage_recall: 0.5\n")
    block = gp.thresholds_for(src, HELIX, "curve")
    assert block["arc_length_error_pct"] == 99.0
    assert block["curve_coverage_recall"] == 0.5
    assert block["curve_max_distance_mm"] == gp.thresholds_for("default", HELIX, "curve")[
        "curve_max_distance_mm"]
    assert load_config(src).config_source == src  # thresholds.py accepts the extra block


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (f"pointline:\n  {HELIX}:\n    dice: 0.9\n", "unknown metric 'dice'"),
        (f"pointline:\n  {HELIX}:\n    arc_length_error_pct: fast\n", "expected a number"),
        ("pointline: [1, 2]\n", "expected a mapping"),
        (f"pointline:\n  {HELIX}:\n    point_recall: 1.0\n", "do not apply to a curve ROI"),
    ],
)
def test_bad_user_pointline_block_rejected(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gp.thresholds_for(_user_yaml(tmp_path, body), HELIX, "curve")


def test_roi_without_thresholds_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="no thresholds for curve ROI 'not_a_roi'"):
        gp.thresholds_for("default", "not_a_roi", "curve")


# --------------------------------------------------------------------------- #
#  Grading on hand-made masks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(POINTLINE))
def test_reference_rasterisation_passes(name: str) -> None:
    assert _grade(gp.reference_rasterisation(_truth(name)), name) == []


def test_reference_rasterisation_points_land_on_nearest_voxels() -> None:
    mask = gp.reference_rasterisation(_truth("single_point_x256_y256_z150"))
    assert np.argwhere(mask).tolist() == [[150, 256, 256]]
    assert int(gp.reference_rasterisation(_truth(SEEDS)).sum()) == 147


def test_reference_rasterisation_keeps_needles_apart() -> None:
    mask = gp.reference_rasterisation(_truth("syed_bundle_5needles_x400_y100_z50"))
    # Five vertical needles, 61 voxels each, nothing in between them.
    assert int(mask.sum()) == 5 * 61
    assert sorted(set(np.argwhere(mask)[:, 2].tolist())) == [386, 393, 400, 407, 414]


def test_curve_filled_into_a_polygon_fails_on_arc_length() -> None:
    truth = _truth(HELIX)
    filled = _filled_per_slice(truth)
    metrics = gp.compute_metrics(filled, truth, GEOMETRY)
    assert metrics["arc_length_error_pct"] > 50.0
    problems = _grade(filled, HELIX)
    assert any(p.startswith("arc_length_error_pct") for p in problems), problems


def test_missing_seed_fails_recall() -> None:
    truth = _truth(SEEDS)
    mask = gp.reference_rasterisation(truth)
    z, y, x = np.argwhere(mask)[73]
    mask[z, y, x] = 0
    metrics = gp.compute_metrics(mask, truth, GEOMETRY)
    assert metrics["point_recall"] == pytest.approx(146 / 147)
    assert metrics["point_max_displacement_mm"] > 4.0  # the nearest other seed is ~5 mm away
    problems = _grade(mask, SEEDS)
    assert any(p.startswith("point_recall") for p in problems), problems


def test_dilated_seeds_pass_and_show_in_fp_rate() -> None:
    """point_fp_rate is reported, not graded: a seed drawn as a 5x5x5 block still
    passes; its fp rate shows the extra marking."""
    from scipy.ndimage import binary_dilation

    truth = _truth(SEEDS)
    mask = binary_dilation(gp.reference_rasterisation(truth), iterations=2).astype(np.uint8)
    metrics = gp.compute_metrics(mask, truth, GEOMETRY)
    assert metrics["point_fp_rate"] > 0.0
    assert _grade(mask, SEEDS) == []


def test_skeleton_crop_does_not_change_arc_length() -> None:
    from rtmask_conformance._vendor.metrics import arc_length_metrics

    truth = _truth(HELIX)
    mask = gp.reference_rasterisation(truth)
    full = arc_length_metrics(mask, truth["analytical_arc_length_mm"], GEOMETRY.spacing)
    assert gp.compute_metrics(mask, truth, GEOMETRY)["arc_length_error_pct"] == (
        full.arc_length_error_pct
    )


# --------------------------------------------------------------------------- #
#  End to end on a generated fixture
# --------------------------------------------------------------------------- #

e2e = pytest.mark.skipif(
    os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set",
)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pointline") / "fixture"
    generate_fixture(out, options=GenerateOptions(n_quadrature=2))
    return out


def _bresenham_3d(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int, int]]:
    """Voxels of a 26-connected 3D Bresenham line between two integer (x, y, z) indices."""
    a = [int(v) for v in a]
    b = [int(v) for v in b]
    d = [abs(b[i] - a[i]) for i in range(3)]
    s = [1 if b[i] > a[i] else -1 for i in range(3)]
    major = int(np.argmax(d))
    others = [i for i in range(3) if i != major]
    err = {i: 2 * d[i] - d[major] for i in others}
    p = list(a)
    out = [tuple(p)]
    for _ in range(d[major]):
        p[major] += s[major]
        for i in others:
            if err[i] > 0:
                p[i] += s[i]
                err[i] -= 2 * d[major]
            err[i] += 2 * d[i]
        out.append(tuple(p))
    return out


def _thin_path_predictions(fixture: Path, out: Path) -> None:
    """Rasterise ``primitives_pointline.dcm`` as a thin-path converter would."""
    ds = pydicom.dcmread(str(fixture / "rtstruct" / "primitives_pointline.dcm"))
    names = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    out.mkdir()
    for rc in ds.ROIContourSequence:
        name = names[rc.ReferencedROINumber]
        mask = np.zeros(GEOMETRY.shape_zyx, dtype=np.uint8)
        for c in rc.ContourSequence:
            pts = np.asarray([float(v) for v in c.ContourData]).reshape(-1, 3)
            idx = np.rint(GEOMETRY.physical_to_index(pts)).astype(int)
            if c.ContourGeometricType == "POINT":
                voxels = [tuple(i) for i in idx]
            else:
                voxels = [v for a, b in zip(idx[:-1], idx[1:], strict=True) for v in _bresenham_3d(a, b)]
            for x, y, z in voxels:
                mask[z, y, x] = 1
        write_nifti(mask, GEOMETRY, out / f"{name}.nii.gz")


@e2e
def test_fixture_writes_curve_and_point_truth(fixture: Path) -> None:
    manifest = read_manifest(fixture)
    assert manifest.bundles["pointline"].rois == list(POINTLINE)
    assert manifest.bundles["pointline"].rtstruct_path == "rtstruct/primitives_pointline.dcm"
    for name, roi in POINTLINE.items():
        entry = manifest.rois[name]
        assert entry.truth_kind == roi.truth_kind and entry.graded
        assert entry.truth_path == f"{gp.TRUTH_DIRS[roi.truth_kind]}/{name}.json"
        assert entry.expected_prediction == f"{name}.nii.gz"
        truth = gp.read_truth(fixture / entry.truth_path)
        assert truth["geometry"]["size"] == [512, 512, 200]
        spec = json.loads((fixture / entry.spec_path).read_text(encoding="utf-8"))
        assert spec["truth_kind"] == roi.truth_kind
        assert "pv_threshold" not in spec
        # The reference rasterisation ships next to the binary truths, as an example.
        assert (fixture / "groundtruth" / f"{name}.nii.gz").is_file()
    assert [t.roi for t in fixture_targets(fixture, manifest, bundles=["pointline"])] == list(
        POINTLINE)


@e2e
def test_rtstruct_carries_point_and_open_nonplanar_contours(fixture: Path) -> None:
    ds = pydicom.dcmread(str(fixture / "rtstruct" / "primitives_pointline.dcm"))
    names = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    contours = {names[rc.ReferencedROINumber]: rc.ContourSequence for rc in ds.ROIContourSequence}
    assert list(contours) == list(POINTLINE)
    for name in POINT_ROIS:
        assert {c.ContourGeometricType for c in contours[name]} == {"POINT"}
        assert all(int(c.NumberOfContourPoints) == 1 for c in contours[name])
        assert len(contours[name]) == len(_truth(name)["points_mm"])
    for name in CURVE_ROIS:
        expected = POINTLINE[name].geometric_type.value
        assert {c.ContourGeometricType for c in contours[name]} == {expected}
    assert len(contours["syed_bundle_5needles_x400_y100_z50"]) == 5


@e2e
def test_thin_path_predictions_pass(fixture: Path, tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    _thin_path_predictions(fixture, pred)
    report = verify_fixture(pred, fixture, load_config(), bundles=["pointline"])
    assert report.summary["total"] == len(POINTLINE)
    failed = {r.roi: r.violations for r in report.results if r.status != Status.PASS}
    assert not failed, failed
    for r in report.results:
        keys = gp.METRIC_KEYS_BY_KIND[r.truth_kind]
        assert set(keys) <= set(r.metrics), r.roi
        assert r.thresholds == gp.thresholds_for("default", r.roi, r.truth_kind)


@e2e
def test_reference_rasterisation_as_predictions_passes_cli(fixture: Path) -> None:
    from rtmask_conformance.cli import main as cli_main

    assert cli_main(["verify", "--predictions", str(fixture / "groundtruth"),
                     "--fixture", str(fixture)]) == 0


@e2e
def test_statuses_for_curve_and_point_rois(fixture: Path, tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    gt = fixture / "groundtruth"
    # A filled helix (FAIL), an empty seed implant (FAIL for a claimed type),
    # a shifted-grid needle (GEOMETRY_MISMATCH), the single point absent (MISSING).
    write_nifti(_filled_per_slice(_truth(HELIX)), GEOMETRY, pred / f"{HELIX}.nii.gz")
    write_nifti(np.zeros(GEOMETRY.shape_zyx, np.uint8), GEOMETRY, pred / f"{SEEDS}.nii.gz")
    img = sitk.ReadImage(str(gt / "diagonal_needle_x100_y200_z40.nii.gz"))
    img.SetOrigin((0.5, 0.0, 0.0))
    sitk.WriteImage(img, str(pred / "diagonal_needle_x100_y200_z40.nii.gz"), True)
    for name in ("needle_track_10pts_x150_y150", "sinusoid_x150_y350_z20to80",
                 "syed_bundle_5needles_x400_y100_z50"):
        shutil.copyfile(gt / f"{name}.nii.gz", pred / f"{name}.nii.gz")

    by_roi = {r.roi: r for r in verify_fixture(
        pred, fixture, load_config(), bundles=["pointline"]).results}
    assert by_roi[HELIX].status == Status.FAIL
    assert by_roi[SEEDS].status == Status.FAIL
    assert by_roi[SEEDS].violations == ["empty mask: no voxels marked for a claimed geometric type"]
    assert by_roi["diagonal_needle_x100_y200_z40"].status == Status.GEOMETRY_MISMATCH
    assert by_roi["single_point_x256_y256_z150"].status == Status.MISSING
    assert by_roi["sinusoid_x150_y350_z20to80"].status == Status.PASS

    text = render_text(verify_fixture(pred, fixture, load_config(), bundles=["pointline"]))
    assert "arcL%" in text and "recall" in text and "empty mask" in text

    # Declining POINT: the absent single point and the empty seed implant are
    # UNSUPPORTED_TYPE; a non-empty prediction for the unclaimed type is still graded.
    no_point = tmp_path / "no_point.yaml"
    no_point.write_text("schema_version: 2\nclaims:\n  geometric_types: "
                        "[CLOSED_PLANAR, CLOSEDPLANAR_XOR, OPEN_PLANAR, OPEN_NONPLANAR]\n")
    by_roi = {r.roi: r for r in verify_fixture(
        pred, fixture, load_config(no_point), bundles=["pointline"]).results}
    assert by_roi["single_point_x256_y256_z150"].status == Status.UNSUPPORTED_TYPE
    assert by_roi[SEEDS].status == Status.UNSUPPORTED_TYPE
    assert by_roi["needle_track_10pts_x150_y150"].status == Status.PASS
    assert by_roi[HELIX].status == Status.FAIL

    # Report-only by type: everything is scored, nothing fails.
    ro = tmp_path / "ro.yaml"
    ro.write_text("schema_version: 2\nreport_only: [OPEN_NONPLANAR]\n")
    record = evaluate_one(HELIX, pred / f"{HELIX}.nii.gz",
                          fixture / "groundtruth_curves" / f"{HELIX}.json", load_config(ro),
                          geometric_type="OPEN_NONPLANAR", truth_kind="curve")
    assert record.status == Status.REPORT_ONLY
    assert any(v.startswith("arc_length_error_pct") for v in record.violations)


@e2e
def test_tampered_curve_truth_is_caught(fixture: Path, tmp_path: Path) -> None:
    copy = tmp_path / "tampered"
    shutil.copytree(fixture, copy, ignore=shutil.ignore_patterns("refct"))
    path = copy / "groundtruth_curves" / f"{HELIX}.json"
    truth = json.loads(path.read_text(encoding="utf-8"))
    truth["analytical_arc_length_mm"] *= 2
    path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(FixtureIntegrityError, match=HELIX):
        verify_fixture(copy / "groundtruth", copy, load_config(), bundles=["pointline"])
