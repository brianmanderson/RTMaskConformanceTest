"""Boundary conventions: the three references, ``claims.convention``,
``verify --convention`` and the ``conventions:`` threshold blocks.

A binary-mask ROI ships one reference per convention:

* ``groundtruth/`` -- PV >= 0.5 (``pv_ge_0.5``, the default; unchanged bytes);
* ``groundtruth_pv_gt0/`` -- PV > 0 (``pv_gt_0``), from the same float field;
* ``groundtruth_centre_in/`` -- the primitive's inside test at each voxel
  centre (``centre_in``);
* ``groundtruth_pv/`` -- the float partial-volume field itself.

The end-to-end tests share one fixture generated at ``n_quadrature=2``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtmask_conformance._roi_set import BUNDLES
from rtmask_conformance._vendor.common.coords import Geometry
from rtmask_conformance._vendor.groundtruth.partial_volume import (
    any_overlap_mask,
    binary_threshold,
    partial_volume_mask,
)
from rtmask_conformance.centre_in import centre_in_mask
from rtmask_conformance.cli import main as cli_main
from rtmask_conformance.generate import (
    CONVENTION_DIRS,
    PV_FIELD_DIR,
    GenerateOptions,
    generate_fixture,
)
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.report import render_text
from rtmask_conformance.thresholds import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
    load_config,
)
from rtmask_conformance.verify import Status, verify_fixture, verify_predictions

GEOMETRY = Geometry(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), size=(512, 512, 200))
PLANAR = {roi.name: roi for roi in BUNDLES["planar"]}
BINARY_ROIS = [
    roi.name for rois in BUNDLES.values() for roi in rois if roi.truth_kind == "binary_mask"
]
TWINS = {
    roi.name: roi.name.rsplit("_", 1)[0]
    for enc in ("xor", "keyhole") for roi in BUNDLES[enc]
}

#: Quadrature at which the float-sum bound is asserted: the shipped default
#: (``generate`` without ``--n-quadrature``). Coarser quadratures miss the bound:
#: at n = 2 (the CI fixtures') the cylinder and the torus are 0.05 % off, at
#: n = 4 the straw is 0.04 % off. The slowest tests in the suite (the inside
#: test runs 512 times per voxel of each bounding box).
FLOAT_SUM_N_QUADRATURE: int = GenerateOptions().n_quadrature
FLOAT_SUM_BOUND_PCT: float = 0.02

#: centre_in voxel count x voxel volume against the closed form, signed % (the
#: count is exact: one inside test per voxel centre, no quadrature). Recorded
#: 2026-11 on the 1 mm fixture grid.
CENTRE_IN_RESIDUAL_PCT: dict[str, float] = {
    "sphere": -0.1200,
    "cube": 0.0,
    "cylinder": -0.2275,
    "ellipsoid": -0.0738,
    "torus": -0.4511,
    "hollow_sphere": -0.0905,
    "straw": -0.0507,
}


def _read(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("conventions") / "fixture"
    generate_fixture(out, options=GenerateOptions(n_quadrature=2))
    return out


def _ideal_predictions(fixture: Path, convention: str, dest: Path) -> Path:
    """Predictions = ``convention``'s reference for every binary-mask ROI, plus the
    suite's reference masks for the curve and point ROIs."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in (fixture / CONVENTION_DIRS[convention]).glob("*.nii.gz"):
        shutil.copyfile(f, dest / f.name)
    for f in (fixture / "reference_masks").glob("*.nii.gz"):
        shutil.copyfile(f, dest / f.name)
    return dest


# --------------------------------------------------------------------------- #
#  The references on disk
# --------------------------------------------------------------------------- #


def test_manifest_records_every_reference_with_its_hash(fixture: Path) -> None:
    manifest = read_manifest(fixture)
    for name, entry in manifest.rois.items():
        if entry.truth_kind != "binary_mask":
            assert entry.convention_truths is None and entry.pv_field is None, name
            continue
        assert set(entry.convention_truths) == set(CONVENTIONS), name
        # The default convention's entry is the v0.1 ground truth, unchanged.
        assert entry.convention_truths[DEFAULT_CONVENTION] == {
            "path": entry.truth_path, "sha256": entry.truth_sha256}
        assert entry.truth_path == f"groundtruth/{name}.nii.gz"
        for convention, rec in entry.convention_truths.items():
            assert rec["path"] == f"{CONVENTION_DIRS[convention]}/{name}.nii.gz"
            assert _sha(fixture / rec["path"]) == rec["sha256"]
            assert entry.truth_for(convention) == (rec["path"], rec["sha256"])
        assert entry.pv_field["path"] == f"{PV_FIELD_DIR}/{name}.nii.gz"
        assert _sha(fixture / entry.pv_field["path"]) == entry.pv_field["sha256"]


@pytest.mark.parametrize("roi", BINARY_ROIS)
def test_references_are_the_field_binarised_and_nested(fixture: Path, roi: str) -> None:
    """ge0.5 and gt0 are the float field's two binarisations; gt0 contains the other two."""
    field = _read(fixture / PV_FIELD_DIR / f"{roi}.nii.gz")
    assert field.dtype == np.float32
    ge = _read(fixture / "groundtruth" / f"{roi}.nii.gz")
    gt0 = _read(fixture / "groundtruth_pv_gt0" / f"{roi}.nii.gz")
    centre = _read(fixture / "groundtruth_centre_in" / f"{roi}.nii.gz")
    for mask in (ge, gt0, centre):
        assert mask.dtype == np.uint8
    np.testing.assert_array_equal(ge, binary_threshold(field, 0.5))
    np.testing.assert_array_equal(gt0, any_overlap_mask(field))
    assert not np.any(ge & ~gt0), "pv_gt_0 must contain pv_ge_0.5"
    assert not np.any(centre & ~gt0), "pv_gt_0 must contain centre_in"


def test_groundtruth_bytes_match_the_default_binarisation(fixture: Path) -> None:
    """``groundtruth/`` is written exactly as before conventions existed:
    ``write_nifti(binary_threshold(partial_volume_mask(...), 0.5))``."""
    from rtmask_conformance._vendor.common.io import write_nifti

    scratch = fixture.parent / "rewrite"
    for name in ("sphere", "torus"):
        prim = PLANAR[name].primitive
        binary = binary_threshold(partial_volume_mask(prim, GEOMETRY, n_quadrature=2), 0.5)
        write_nifti(binary, GEOMETRY, scratch / f"{name}.nii.gz")
        assert _sha(scratch / f"{name}.nii.gz") == _sha(fixture / "groundtruth" / f"{name}.nii.gz")


@pytest.mark.parametrize("roi", list(PLANAR))
def test_centre_in_is_the_inside_test_at_voxel_centres(fixture: Path, roi: str) -> None:
    """centre_in = one quadrature sample per voxel, at its centre; its volume residual
    against the closed form is the recorded one."""
    prim = PLANAR[roi].primitive
    centre = _read(fixture / "groundtruth_centre_in" / f"{roi}.nii.gz")
    np.testing.assert_array_equal(
        centre, binary_threshold(partial_volume_mask(prim, GEOMETRY, n_quadrature=1), 0.5)
    )
    v_true = prim.analytical_volume()
    residual = 100.0 * (int(centre.sum()) * GEOMETRY.voxel_volume_mm3 - v_true) / v_true
    print(f"centre_in residual {roi}: {residual:+.4f} %")
    assert residual == pytest.approx(CENTRE_IN_RESIDUAL_PCT[roi], abs=5e-4)


def test_centre_in_mask_matches_the_writer() -> None:
    prim = PLANAR["sphere"].primitive
    np.testing.assert_array_equal(
        centre_in_mask(prim, GEOMETRY),
        binary_threshold(partial_volume_mask(prim, GEOMETRY, n_quadrature=1), 0.5),
    )


@functools.cache
def _shipped_default_truths(roi: str) -> dict:
    """The float field's sum and the three references' voxel counts of ``roi`` at the
    shipped default quadrature, computed once per primitive (the slow part)."""
    prim = PLANAR[roi].primitive
    field = partial_volume_mask(prim, GEOMETRY, n_quadrature=FLOAT_SUM_N_QUADRATURE)
    counts = {
        "pv_ge_0.5": int(binary_threshold(field, 0.5).sum()),
        "pv_gt_0": int(any_overlap_mask(field).sum()),
        "centre_in": int(centre_in_mask(prim, GEOMETRY).sum()),
    }
    return {"field_sum": float(field.sum(dtype=np.float64)), "counts": counts}


@pytest.mark.parametrize("roi", list(PLANAR))
def test_float_field_sum_matches_the_closed_form(roi: str) -> None:
    v_true = PLANAR[roi].primitive.analytical_volume()
    field_sum = _shipped_default_truths(roi)["field_sum"]
    residual = 100.0 * (field_sum * GEOMETRY.voxel_volume_mm3 - v_true) / v_true
    print(f"float-sum residual {roi} (n={FLOAT_SUM_N_QUADRATURE}): {residual:+.5f} %")
    assert abs(residual) <= FLOAT_SUM_BOUND_PCT


#: Pairs (ideal mask's convention, reference's convention) whose volumes differ by
#: more than the reference convention's volume ceiling at the shipped default
#: quadrature, per primitive. pv_gt_0 is told apart from the other two on every
#: curved primitive; pv_ge_0.5 and centre_in never are (their masks differ by at
#: most 0.53 % in volume), and on the cube no pair is.
_GT0_PAIRS = {("pv_ge_0.5", "pv_gt_0"), ("centre_in", "pv_gt_0"),
              ("pv_gt_0", "pv_ge_0.5"), ("pv_gt_0", "centre_in")}
VOLUME_SEPARATED: dict[str, set[tuple[str, str]]] = {
    "sphere": _GT0_PAIRS,
    "cube": set(),
    "cylinder": _GT0_PAIRS,
    "ellipsoid": _GT0_PAIRS,
    "torus": _GT0_PAIRS,
    "hollow_sphere": _GT0_PAIRS,
    "straw": _GT0_PAIRS,
}


@pytest.mark.parametrize("roi", list(PLANAR))
def test_ideal_masks_fail_volume_under_another_convention(roi: str) -> None:
    """An ideal mask of each convention, graded on volume under each other convention
    at the shipped default quadrature: exactly the recorded pairs fail."""
    config = load_config()
    counts = _shipped_default_truths(roi)["counts"]
    separated = set()
    for pred in CONVENTIONS:
        for ref in CONVENTIONS:
            err = abs(counts[pred] - counts[ref]) / counts[ref]
            print(f"{roi}: {pred} mask under {ref}: volume error {100 * err:.2f} %")
            if err > config.thresholds_for(roi, ref).volume_rel_err:
                separated.add((pred, ref))
    assert separated == VOLUME_SEPARATED[roi]


@pytest.mark.parametrize("twin", list(TWINS))
def test_hole_encoding_twins_share_their_parents_references(fixture: Path, twin: str) -> None:
    parent = TWINS[twin]
    for d in (*CONVENTION_DIRS.values(), PV_FIELD_DIR):
        np.testing.assert_array_equal(
            _read(fixture / d / f"{twin}.nii.gz"), _read(fixture / d / f"{parent}.nii.gz")
        )


def test_cube_references_coincide(fixture: Path) -> None:
    """The cube's faces lie on voxel boundaries, so every voxel is wholly in or out:
    the three conventions give the same mask and cannot be told apart on it."""
    ref = _read(fixture / "groundtruth" / "cube.nii.gz")
    for convention in CONVENTIONS[1:]:
        np.testing.assert_array_equal(
            _read(fixture / CONVENTION_DIRS[convention] / "cube.nii.gz"), ref
        )


# --------------------------------------------------------------------------- #
#  Grading under a convention
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ideal_reports(fixture: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Per convention: (its ideal predictions, the full-fixture report under it, the
    warnings verify raised). Scoring every ROI on the full grid is slow, so the
    three reports are computed once and shared."""
    root = tmp_path_factory.mktemp("ideal")
    out = {}
    for convention in CONVENTIONS:
        pred = _ideal_predictions(fixture, convention, root / convention)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = verify_fixture(pred, fixture, load_config(), convention=convention)
        out[convention] = (pred, report, [str(w.message) for w in caught])
    return out


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_ideal_rasterisation_passes_under_its_own_convention(
    ideal_reports: dict, convention: str
) -> None:
    _, report, _ = ideal_reports[convention]
    assert report.overall_pass, [(r.roi, r.violations) for r in report.results if r.violations]
    assert report.claims["convention"] == convention
    for r in report.results:
        if r.truth_kind == "binary_mask":
            assert r.groundtruth_path.endswith(
                str(Path(CONVENTION_DIRS[convention]) / f"{r.roi}.nii.gz"))
            assert r.metrics["dice"] == 1.0


def test_pv_gt_0_on_a_coarse_fixture_warns(ideal_reports: dict) -> None:
    assert any("n_quadrature=2" in w for w in ideal_reports["pv_gt_0"][2])
    for convention in ("pv_ge_0.5", "centre_in"):
        assert not any("n_quadrature" in w for w in ideal_reports[convention][2])


def test_any_overlap_masks_fail_under_centre_in_end_to_end(
    fixture: Path, ideal_reports: dict
) -> None:
    """The cross-convention verdicts through verify_fixture on the CI fixture: its
    n_quadrature=2 pv_gt_0 reference is smaller than the default's (fewer samples
    see a small overlap), but the torus still fails volume; the cube never fails."""
    pred = ideal_reports["pv_gt_0"][0]
    report = verify_fixture(pred, fixture, load_config(), bundles=["planar"],
                            convention="centre_in")
    failed = {r.roi: r.violations for r in report.results if r.status == Status.FAIL}
    assert any(v.startswith("volume_rel_err") for v in failed["torus"])
    assert "cube" not in failed


def test_curve_and_point_rois_are_graded_the_same_under_every_convention(
    ideal_reports: dict,
) -> None:
    """Every convention's predictions carry the same reference masks for these ROIs."""
    by_convention = {
        c: {r.roi: r for r in ideal_reports[c][1].results if r.truth_kind != "binary_mask"}
        for c in CONVENTIONS
    }
    base = by_convention[DEFAULT_CONVENTION]
    assert base
    for c in CONVENTIONS[1:]:
        for roi, r in by_convention[c].items():
            assert (r.status, r.metrics, r.thresholds, r.groundtruth_path) == (
                base[roi].status, base[roi].metrics, base[roi].thresholds,
                base[roi].groundtruth_path)


def test_yaml_claim_selects_the_reference(fixture: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("schema_version: 2\nclaims:\n  convention: centre_in\n", encoding="utf-8")
    pred = _ideal_predictions(fixture, "centre_in", tmp_path / "ci")
    report = verify_fixture(pred, fixture, load_config(cfg), bundles=["xor"])
    assert report.overall_pass
    assert report.claims["convention"] == "centre_in"
    roi = "hollow_sphere_xor"
    ci = {r.roi: r for r in report.results}[roi]
    assert ci.groundtruth_path.endswith(str(Path("groundtruth_centre_in") / f"{roi}.nii.gz"))
    # --convention overrides the claim.
    report = verify_fixture(pred, fixture, load_config(cfg), bundles=["xor"],
                            convention="pv_gt_0")
    assert report.claims["convention"] == "pv_gt_0"
    gt0 = {r.roi: r for r in report.results}[roi]
    assert gt0.groundtruth_path.endswith(str(Path("groundtruth_pv_gt0") / f"{roi}.nii.gz"))


def test_undeclared_convention_leaves_the_report_unchanged(fixture: Path, tmp_path: Path) -> None:
    """No claim, no flag: the claims block and the header are as before conventions."""
    pred = _ideal_predictions(fixture, DEFAULT_CONVENTION, tmp_path / "p")
    report = verify_fixture(pred, fixture, load_config(), bundles=["xor"])
    assert report.as_dict()["claims"] == {"geometric_types": list(load_config().claimed_types)}
    assert "convention=" not in render_text(report).splitlines()[0]
    declared = verify_fixture(pred, fixture, load_config(), bundles=["xor"],
                              convention=DEFAULT_CONVENTION)
    assert declared.as_dict()["claims"]["convention"] == DEFAULT_CONVENTION
    assert "convention=pv_ge_0.5" in render_text(declared).splitlines()[0]
    # Declaring the default changes only the claims block and the header.
    a, b = report.as_dict(), declared.as_dict()
    a.pop("claims"), b.pop("claims")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_old_fixture_without_the_reference_is_a_clear_error(fixture: Path, tmp_path: Path) -> None:
    old = tmp_path / "old"
    shutil.copytree(fixture, old, ignore=shutil.ignore_patterns("refct", "groundtruth_pv*",
                                                                "groundtruth_centre_in"))
    raw = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    for entry in raw["rois"].values():
        entry.pop("convention_truths", None)
        entry.pop("pv_field", None)
    (old / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    pred = _ideal_predictions(fixture, DEFAULT_CONVENTION, tmp_path / "p")
    assert verify_fixture(pred, old, load_config(), bundles=["xor"]).overall_pass  # default
    with pytest.raises(ValueError, match="no 'centre_in' reference for 'sphere'"):
        verify_fixture(pred, old, load_config(), convention="centre_in")
    assert cli_main(["verify", "--predictions", str(pred), "--fixture", str(old),
                     "--convention", "pv_gt_0"]) == 2


def test_tampered_convention_reference_fails_the_hash_check(fixture: Path, tmp_path: Path) -> None:
    from rtmask_conformance.verify import FixtureIntegrityError

    copy = tmp_path / "copy"
    shutil.copytree(fixture, copy, ignore=shutil.ignore_patterns("refct"))
    roi = "hollow_sphere_xor"
    target = copy / "groundtruth_centre_in" / f"{roi}.nii.gz"
    shutil.copyfile(copy / "groundtruth_pv_gt0" / f"{roi}.nii.gz", target)
    pred = _ideal_predictions(fixture, DEFAULT_CONVENTION, tmp_path / "p")
    # Only the graded convention's reference is checked.
    assert verify_fixture(pred, copy, load_config(), bundles=["xor"]).overall_pass
    with pytest.raises(FixtureIntegrityError):
        verify_fixture(pred, copy, load_config(), bundles=["xor"], convention="centre_in")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_cli_convention_flag(fixture: Path, tmp_path: Path, convention: str, capsys) -> None:
    pred = _ideal_predictions(fixture, convention, tmp_path / "p")
    out_json = tmp_path / "r.json"
    assert cli_main(["verify", "--predictions", str(pred), "--fixture", str(fixture),
                     "--bundle", "xor", "--convention", convention,
                     "--report-json", str(out_json)]) == 0
    assert f"convention={convention}" in capsys.readouterr().out
    assert json.loads(out_json.read_text(encoding="utf-8"))["claims"]["convention"] == convention


def test_cli_convention_is_rejected_where_it_does_not_apply(fixture: Path, tmp_path: Path) -> None:
    pred = _ideal_predictions(fixture, "centre_in", tmp_path / "p")
    gt = fixture / "groundtruth"
    assert cli_main(["verify", "--predictions", str(pred), "--groundtruth", str(gt),
                     "--convention", "centre_in"]) == 2
    cfg = tmp_path / "c.yaml"
    cfg.write_text("schema_version: 2\nclaims:\n  convention: pv_gt_0\n", encoding="utf-8")
    assert cli_main(["verify", "--predictions", str(pred), "--groundtruth", str(gt),
                     "--config", str(cfg)]) == 2
    with pytest.raises(ValueError, match="bare ground-truth directory"):
        verify_predictions(pred, gt, load_config(cfg))
    csv = tmp_path / "v.csv"
    csv.write_text("roi,volume_cc\n", encoding="utf-8")
    assert cli_main(["verify", "--volumes", str(csv), "--fixture", str(fixture),
                     "--convention", "pv_gt_0"]) == 2
    with pytest.raises(SystemExit):
        cli_main(["verify", "--predictions", str(pred), "--fixture", str(fixture),
                  "--convention", "pv_ge_0"])


# --------------------------------------------------------------------------- #
#  Threshold YAML: claims.convention and the conventions: block
# --------------------------------------------------------------------------- #


def _yaml(tmp_path: Path, body: str, version: int = 2) -> Path:
    p = tmp_path / "t.yaml"
    p.write_text(f"schema_version: {version}\n{body}", encoding="utf-8")
    return p


def test_shipped_default_declares_no_convention() -> None:
    config = load_config()
    assert config.convention == DEFAULT_CONVENTION and not config.convention_declared
    assert "convention" not in config.claims_dict()
    assert set(config.conventions) == {"pv_gt_0", "centre_in"}


def test_pv_ge_05_thresholds_are_the_top_level_blocks() -> None:
    config = load_config()
    for roi in BINARY_ROIS:
        assert config.thresholds_for(roi, DEFAULT_CONVENTION) == config.thresholds_for(roi)


def test_merge_order(tmp_path: Path) -> None:
    """defaults <- primitives[r] <- conventions[c].defaults <- conventions[c].primitives[r]."""
    config = load_config(_yaml(tmp_path, """
defaults: {dice: 0.90, hd95_mm: 9.0}
primitives:
  sphere: {dice: 0.91, msd_mm: 0.7}
conventions:
  centre_in:
    defaults: {dice: 0.80, volume_rel_err: 0.07}
    primitives:
      sphere: {volume_rel_err: 0.08}
"""))
    ge = config.thresholds_for("sphere", DEFAULT_CONVENTION)
    # The shipped primitives.sphere.hd95_mm (1.5) wins over the user's defaults.
    assert (ge.dice, ge.hd95_mm, ge.msd_mm, ge.surface_dice_1mm) == (0.91, 1.5, 0.7, 0.95)
    assert config.thresholds_for("straw", DEFAULT_CONVENTION).hd95_mm == 2.5
    assert config.thresholds_for("not_shipped", DEFAULT_CONVENTION).hd95_mm == 9.0
    ci = config.thresholds_for("sphere", "centre_in")
    # The convention's defaults win over primitives[r]; its primitives[r] wins over all.
    assert (ci.dice, ci.hd95_mm, ci.msd_mm, ci.volume_rel_err) == (0.80, 1.5, 0.7, 0.08)
    assert config.thresholds_for("cube", "centre_in").volume_rel_err == 0.07
    # A convention the user names nothing for falls back to the shipped block.
    shipped = load_config()
    assert config.conventions["pv_gt_0"] == shipped.conventions["pv_gt_0"]


def test_user_convention_block_merges_over_the_shipped_one(tmp_path: Path) -> None:
    shipped = load_config()
    config = load_config(_yaml(tmp_path, """
conventions:
  pv_gt_0:
    primitives:
      torus: {hd95_mm: 4.0}
"""))
    gt0 = config.conventions["pv_gt_0"]
    assert gt0["primitives"]["torus"]["hd95_mm"] == 4.0
    for key, value in shipped.conventions["pv_gt_0"]["primitives"].get("torus", {}).items():
        if key != "hd95_mm":
            assert gt0["primitives"]["torus"][key] == value
    assert gt0["defaults"] == shipped.conventions["pv_gt_0"]["defaults"]
    assert config.thresholds_for("torus", "pv_gt_0").hd95_mm == 4.0


def test_claims_convention_and_replacement(tmp_path: Path) -> None:
    config = load_config(_yaml(tmp_path, "claims:\n  convention: pv_gt_0\n"))
    assert config.convention == "pv_gt_0" and config.convention_declared
    assert config.claims_dict()["convention"] == "pv_gt_0"
    # A user claims block without `convention` takes the default, undeclared.
    config = load_config(_yaml(tmp_path, "claims:\n  geometric_types: [CLOSED_PLANAR]\n"))
    assert config.convention == DEFAULT_CONVENTION and not config.convention_declared
    over = config.with_convention("centre_in")
    assert over.convention == "centre_in" and over.claims_dict()["convention"] == "centre_in"
    assert config.convention == DEFAULT_CONVENTION  # with_convention copies


@pytest.mark.parametrize(
    ("body", "version", "match"),
    [
        ("claims:\n  convention: pv_ge_05\n", 2, "not one of"),
        ("conventions:\n  pv_ge_0.5: {defaults: {dice: 0.9}}\n", 2, "set them there"),
        ("conventions:\n  any_overlap: {}\n", 2, "unknown convention"),
        ("conventions:\n  pv_gt_0: {thresholds: {}}\n", 2, "unknown key"),
        ("conventions:\n  pv_gt_0: {defaults: {arc_length_error_pct: 3}}\n", 2, "unknown metric"),
        ("conventions:\n  pv_gt_0: {primitives: {sphere: {dice: high}}}\n", 2, "expected a number"),
        ("conventions:\n  pv_gt_0: {}\n", 1, "require schema_version: 2"),
        ("claims:\n  convention: centre_in\n", 1, "require schema_version: 2"),
    ],
)
def test_bad_convention_yaml_rejected(tmp_path: Path, body: str, version: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(_yaml(tmp_path, body, version))


def test_pointline_block_is_validated_at_load(tmp_path: Path) -> None:
    """One module validates every threshold block: a malformed pointline block fails
    load_config, before any ROI is graded."""
    with pytest.raises(ValueError, match="user.pointline.x: unknown metric 'dice'"):
        load_config(_yaml(tmp_path, "pointline:\n  x: {dice: 0.9}\n"))
    assert load_config().pointline  # the shipped block
