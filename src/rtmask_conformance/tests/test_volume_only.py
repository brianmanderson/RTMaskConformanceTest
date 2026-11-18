"""Volume-only mode (``verify --volumes``): CSV reading, end caps, grading, CLI.

Most tests grade hand-built :class:`VolumeTarget` lists, so they need no
fixture. The end-to-end tests build a fixture without ground-truth masks (the
mode reads only ``manifest.json``, ``specs/`` and the RTSTRUCTs): the suite's
own primitives over a reference CT of 16 x 16 x 200 voxels, which is enough
for the writer to place every contour on its slice.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from rtmask_conformance import __version__, load_config
from rtmask_conformance._roi_set import BUNDLES
from rtmask_conformance._vendor.common.io import sha256_file
from rtmask_conformance._vendor.refimage.build_reference_ct import (
    ReferenceCTSpec,
    build_reference_ct,
)
from rtmask_conformance._vendor.rtstruct.pydicom_writer import build_rtstruct
from rtmask_conformance.cli import main as cli_main
from rtmask_conformance.generate import rtstruct_relpath
from rtmask_conformance.manifest import (
    MANIFEST_VERSION,
    BundleEntry,
    FixtureManifest,
    RoiEntry,
    read_manifest,
    write_manifest,
)
from rtmask_conformance.report import render_text
from rtmask_conformance.thresholds import ConformanceConfig
from rtmask_conformance.verify import Status, verify_volumes
from rtmask_conformance.volume_only import (
    TEMPLATE_COLUMNS,
    VolumeTable,
    VolumeTarget,
    end_cap_volume_mm3,
    grade_rows,
    read_volumes_csv,
    reference_volume_mm3,
    slice_area_even_odd,
    write_volumes_template,
)

CYL_R, CYL_H = 30.0, 80.0           # the shipped `cylinder`
POLY = 0.5 * 64 * math.sin(2 * math.pi / 64)  # area of the writer's 64-gon / r^2


def _config(tmp_path: Path, body: str) -> ConformanceConfig:
    p = tmp_path / "claims.yaml"
    p.write_text("schema_version: 2\n" + body, encoding="utf-8")
    return load_config(p)


def _csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows([["" if c is None else c for c in r] for r in rows])
    return path


def _by_roi(results) -> dict:
    return {r.roi: r for r in results}


# --------------------------------------------------------------------------- end-cap formula


def test_end_cap_formula_on_the_shipped_cylinder():
    """Pin the formula with hand-computed numbers.

    The cylinder spans z 60.5..140.5 mm; the CT has slices at integer z, so its
    80 contours sit at z 61..140, the end ones half a slice inside the caps.
    End-contour area A = 64-gon of r 30 = 2822.89 mm^2 (pi r^2 = 2827.43).
      half_slice: V_ref = pi r^2 h                       = 226194.67 mm^3
      none:       V_ref = pi r^2 h - (A + A) * 1 mm / 2  = 223371.78 mm^3
    The analytic volume is +1.26 % on the `none` reference; a slab system
    (80 contours x 1 mm x A = 225831.18 mm^3) is +1.10 %: both fail the default.
    """
    analytic = math.pi * CYL_R**2 * CYL_H
    area = POLY * CYL_R**2
    cap = end_cap_volume_mm3((area, area), 1.0)
    assert cap == pytest.approx(2822.89, abs=0.01)
    assert reference_volume_mm3(analytic, cap, "half_slice") == pytest.approx(226194.67, abs=0.01)
    assert reference_volume_mm3(analytic, cap, "none") == pytest.approx(223371.78, abs=0.01)
    slab = 80 * area * 1.0
    assert (analytic - 223371.78) / 223371.78 == pytest.approx(0.012638, abs=1e-5)
    assert (slab - 223371.78) / 223371.78 == pytest.approx(0.011012, abs=1e-5)
    with pytest.raises(ValueError, match="unknown end_cap"):
        reference_volume_mm3(analytic, cap, "quarter")


def test_even_odd_area_of_an_annulus_and_a_keyhole():
    def circle(r, n=256, reverse=False):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if reverse:
            t = t[::-1]
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

    ring = slice_area_even_odd([circle(40), circle(20)])
    assert ring == pytest.approx(math.pi * (1600 - 400), rel=1e-3)
    keyhole = np.concatenate([circle(40), circle(40)[:1], circle(20, reverse=True),
                              circle(20, reverse=True)[:1]])
    assert slice_area_even_odd([keyhole]) == pytest.approx(math.pi * 1200, rel=1e-3)


# --------------------------------------------------------------------------- grading (no I/O)


def _table(rows: list[tuple], unit="cc", api=False) -> VolumeTable:
    """rows: (roi, volume, decimals, outcome[, api_volume])."""
    from rtmask_conformance.volume_only import VolumeRow

    return VolumeTable(
        path="mem.csv",
        rows=[VolumeRow(roi=r[0], volume=r[1], api_volume=r[4] if len(r) > 4 else None,
                        volume_decimals=r[2], import_outcome=r[3], message=None, line=i)
              for i, r in enumerate(rows, start=2)],
        volume_column=f"volume_{unit}", unit=unit,
        api_column="x_volume_cc_api" if api else None, api_unit="cc" if api else None,
    )


def _box(name="box", mm3=1000.0, gtype="CLOSED_PLANAR", graded=True) -> VolumeTarget:
    return VolumeTarget(roi=name, geometric_type=gtype, graded=graded,
                        analytic_mm3=mm3, end_areas_mm2=None)


def test_exact_volume_passes_and_one_and_a_half_percent_fails_at_the_default():
    cfg = load_config()
    assert cfg.volume_tolerance == 0.01 and cfg.end_cap == "none"
    res, _ = grade_rows(_table([("a", 1.000, None, None), ("b", 1.015, None, None)]),
                        [_box("a"), _box("b")], cfg)
    a, b = res
    assert a.status == Status.PASS and a.error_pct == pytest.approx(0.0)
    assert b.status == Status.FAIL and b.error_pct == pytest.approx(1.5)


def test_missing_row_is_missing_and_rejected_unclaimed_xor_is_unsupported(tmp_path):
    cfg = _config(tmp_path, "claims:\n  geometric_types: [CLOSED_PLANAR]\n")
    targets = [_box("solid"), _box("ring_xor", gtype="CLOSEDPLANAR_XOR"),
               _box("ring2_xor", gtype="CLOSEDPLANAR_XOR")]
    res, _ = grade_rows(_table([("ring_xor", None, None, "rejected"),
                                ("solid", None, None, "rejected_file")]), targets, cfg)
    r = _by_roi(res)
    assert r["solid"].status == Status.MISSING
    assert r["solid"].absent_reason == "rejected_file"
    assert r["ring_xor"].status == Status.UNSUPPORTED_TYPE
    assert r["ring_xor"].absent_reason == "rejected"
    assert r["ring2_xor"].status == Status.UNSUPPORTED_TYPE  # no row, unclaimed

    # Claimed: a rejected ROI is MISSING, and the report says which outcome.
    res, _ = grade_rows(_table([("ring_xor", None, None, "rejected")]),
                        [_box("ring_xor", gtype="CLOSEDPLANAR_XOR"), _box("solid")], load_config())
    r = _by_roi(res)
    assert r["ring_xor"].status == Status.MISSING
    assert "rejected" in r["ring_xor"].violations[0]
    assert r["solid"].status == Status.MISSING and r["solid"].absent_reason == "no_row"


def test_curve_and_point_rois_are_report_only_with_the_volume_echoed():
    targets = [_box("pt", mm3=None, gtype="POINT"), _box("arc", mm3=None, gtype="OPEN_PLANAR")]
    res, _ = grade_rows(_table([("pt", 0.02, 2, "warned"), ("arc", None, None, "accepted")]),
                        targets, load_config())
    r = _by_roi(res)
    assert r["pt"].status == Status.REPORT_ONLY
    assert r["pt"].reported_volume_cc == pytest.approx(0.02)
    assert r["pt"].analytic_volume_cc == 0.0
    assert r["arc"].status == Status.REPORT_ONLY


def test_report_only_by_config_and_ungraded_manifest_rois(tmp_path):
    cfg = _config(tmp_path, "report_only: [a]\n")
    res, _ = grade_rows(_table([("a", 2.0, None, None), ("b", 2.0, None, None)]),
                        [_box("a"), _box("b", graded=False)], cfg)
    for r in res:
        assert r.status == Status.REPORT_ONLY
        assert r.error_pct == pytest.approx(100.0)  # scored for information


def test_display_quantisation_api_volume_and_display_limited_marking():
    """A 0.30 cc structure displayed at two decimals: q = 0.005 / 0.30 = 1.67 %."""
    small = _box("small", mm3=300.0)
    reported = 0.30 * 1.015  # +1.5 %
    res, _ = grade_rows(_table([("small", reported, 2, "accepted")]), [small], load_config())
    (r,) = res
    assert r.status == Status.PASS
    assert r.display_limited is True
    assert r.display_quantisation_pct == pytest.approx(100 * 0.005 / 0.30)
    assert r.effective_tolerance_pct == pytest.approx(100 * 0.005 / 0.30)
    assert r.graded_volume == "displayed"

    res, _ = grade_rows(_table([("small", reported, None, "accepted")]), [small], load_config())
    assert res[0].status == Status.FAIL and res[0].display_limited is False

    # A large structure at two decimals is not display-limited.
    res, _ = grade_rows(_table([("big", 226.19, 2, None)]), [_box("big", mm3=226194.67)],
                        load_config())
    assert res[0].status == Status.PASS and res[0].display_limited is False

    # The API volume is graded instead, at the tolerance alone.
    tab = _table([("small", reported, 2, "accepted", reported)], api=True)
    res, _ = grade_rows(tab, [small], load_config())
    assert res[0].graded_volume == "api"
    assert res[0].status == Status.FAIL
    assert res[0].effective_tolerance_pct == pytest.approx(1.0)
    tab = _table([("small", reported, 2, "accepted", 0.30 * 1.009)], api=True)
    assert grade_rows(tab, [small], load_config())[0][0].status == Status.PASS


def test_tolerance_override_and_yaml_tolerance(tmp_path):
    res, _ = grade_rows(_table([("a", 1.015, None, None)]), [_box("a")], load_config(),
                        tolerance=0.02)
    assert res[0].status == Status.PASS
    cfg = _config(tmp_path, "volume_only:\n  tolerance: 0.02\n")
    assert cfg.volume_tolerance == 0.02
    assert grade_rows(_table([("a", 1.015, None, None)]), [_box("a")], cfg)[0][0].status \
        == Status.PASS


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("body, match", [
    ("claims:\n  end_cap: quarter\n", "end_cap"),
    ("claims:\n  end_cap: 0.5\n", "end_cap"),
    ("volume_only:\n  tolerance: 1.5\n", "tolerance"),
    ("volume_only:\n  tolerance: true\n", "tolerance"),
    ("volume_only:\n  tol: 0.01\n", "unknown key"),
    ("volume_only: [0.01]\n", "mapping"),
])
def test_config_rejects_bad_volume_only_settings(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, body)


def test_volume_only_requires_schema_2_and_claims_replace_end_cap(tmp_path):
    p = tmp_path / "v1.yaml"
    p.write_text("schema_version: 1\nvolume_only:\n  tolerance: 0.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version: 2"):
        load_config(p)
    cfg = _config(tmp_path, "claims:\n  end_cap: half_slice\n")
    assert cfg.end_cap == "half_slice"
    assert len(cfg.claimed_types) == 5  # geometric_types defaults to all five
    assert cfg.claims_dict() == {"geometric_types": list(cfg.claimed_types)}  # mask path unchanged
    cfg = _config(tmp_path, "claims:\n  geometric_types: [CLOSED_PLANAR]\n")
    assert cfg.end_cap == "none"


# --------------------------------------------------------------------------- CSV reader


def test_reader_header_variants_and_errors(tmp_path):
    t = read_volumes_csv(_csv(tmp_path / "a.csv", ["ROI", "Volume_mm3"], [["box", 1000]]))
    assert t.unit == "mm3" and t.rows[0].volume == 1000.0
    t = read_volumes_csv(_csv(tmp_path / "b.csv", ["roi", "eclipse_volume_cc", "notes"],
                              [["box", "1.0", "x"]]))
    assert t.volume_column == "eclipse_volume_cc" and t.unit == "cc"
    bom = tmp_path / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbfroi,volume_cc\nbox,1.0\n")
    assert read_volumes_csv(bom).rows[0].roi == "box"

    bad = {
        "no_roi.csv": (["name", "volume_cc"], [["box", 1]], "no 'roi'"),
        "no_vol.csv": (["roi", "notes"], [["box", 1]], "no volume column"),
        "both.csv": (["roi", "volume_cc", "volume_mm3"], [["box", 1, 1000]], "both"),
        "two_sys.csv": (["roi", "a_volume_cc", "b_volume_cc"], [["box", 1, 1]], "several"),
        "dup.csv": (["roi", "volume_cc"], [["box", 1], ["box", 2]], "more than one row"),
        "outcome.csv": (["roi", "volume_cc", "import_outcome"], [["box", 1, "ok"]], "import_outcome"),
        "nan.csv": (["roi", "volume_cc"], [["box", "1,0"]], "not a number"),
        "neg.csv": (["roi", "volume_cc"], [["box", -1]], "non-negative"),
        "dec.csv": (["roi", "volume_cc", "volume_decimals"], [["box", 1, "two"]], "volume_decimals"),
    }
    for name, (header, rows, match) in bad.items():
        with pytest.raises(ValueError, match=match):
            read_volumes_csv(_csv(tmp_path / name, header, rows))


def test_case_insensitive_match_warns_and_unknown_rows_are_reported():
    with pytest.warns(UserWarning) as rec:
        res, unmatched = grade_rows(_table([("BOX", 1.0, None, None), ("other", 1.0, None, None)]),
                                    [_box("box")], load_config())
    assert res[0].status == Status.PASS
    assert unmatched == ["other"]
    text = " ".join(str(w.message) for w in rec)
    assert "case-insensitively" in text and "not in the fixture" in text


# --------------------------------------------------------------------------- end to end


def _volume_fixture(root: Path, bundles: dict) -> Path:
    """A fixture with RTSTRUCTs, specs and manifest but no ground-truth masks."""
    root.mkdir(parents=True)
    geometry = build_reference_ct(root / "refct", ReferenceCTSpec(size=(16, 16, 200)))
    bundle_entries, roi_entries = {}, {}
    (root / "specs").mkdir()
    for bundle, rois in bundles.items():
        rel = rtstruct_relpath(bundle)
        build_rtstruct(primitives=[r.primitive for r in rois], ref_image_folder=root / "refct",
                       out_path=root / rel, structure_set_label="RTMASK_CONFORM")
        bundle_entries[bundle] = BundleEntry(rel, sha256_file(root / rel), [r.name for r in rois])
        for r in rois:
            spec = r.primitive.to_spec_dict()
            (root / f"specs/{r.name}.json").write_text(json.dumps(spec), encoding="utf-8")
            roi_entries[r.name] = RoiEntry(
                bundle=bundle, geometric_type=r.geometric_type.value, truth_kind=r.truth_kind,
                graded=r.graded, truth_path=None, truth_sha256=None,
                spec_path=f"specs/{r.name}.json",
            )
    write_manifest(FixtureManifest(
        manifest_version=MANIFEST_VERSION, package_version=__version__, git_sha="unknown",
        n_quadrature=None, pv_threshold=None,
        geometry={"origin": list(geometry.origin), "spacing": list(geometry.spacing),
                  "size": list(geometry.size), "direction": list(geometry.direction)},
        bundles=bundle_entries, rois=roi_entries,
    ), root)
    write_volumes_template(root, read_manifest(root))
    return root


@pytest.fixture(scope="module")
def planar_fixture(tmp_path_factory) -> Path:
    return _volume_fixture(tmp_path_factory.mktemp("vol") / "fixture",
                           {"planar": BUNDLES["planar"]})


def _analytic_cc(fixture: Path) -> dict[str, float]:
    m = read_manifest(fixture)
    return {roi: json.loads((fixture / e.spec_path).read_text())["analytical_volume_mm3"] / 1000
            for roi, e in m.rois.items()}


def test_exact_analytic_volumes_pass_under_half_slice_and_cylinder_flips(planar_fixture, tmp_path):
    analytic = _analytic_cc(planar_fixture)
    path = _csv(tmp_path / "v.csv", ["roi", "volume_cc"], [[k, repr(v)] for k, v in analytic.items()])
    half = _config(tmp_path, "claims:\n  end_cap: half_slice\n")
    rep = verify_volumes(path, planar_fixture, half)
    assert rep.overall_pass, render_text(rep)
    assert all(r.status == Status.PASS and abs(r.error_pct) < 1e-9 for r in rep.results)

    # The same slab-system volumes under the default `none`: the flat-capped
    # prisms (cube, cylinder, straw) are a half slice per end too large.
    rep = verify_volumes(path, planar_fixture)
    r = _by_roi(rep.results)
    assert r["cylinder"].status == Status.FAIL
    assert r["cylinder"].error_pct == pytest.approx(1.2638, abs=1e-3)
    assert r["cylinder"].end_cap_volume_cc == pytest.approx(2.82289, abs=1e-4)
    assert r["sphere"].status == Status.PASS and abs(r["sphere"].error_pct) < 0.1

    # Volumes that stop at the end contours pass the default.
    none = [[k, repr(r[k].reference_volume_cc)] for k in analytic]
    rep = verify_volumes(_csv(tmp_path / "n.csv", ["roi", "volume_cc"], none), planar_fixture)
    assert rep.overall_pass, render_text(rep)


def _contour_volumes_cc(fixture: Path) -> dict[str, tuple[float, float]]:
    """``{roi: (slab volume, contour-to-contour volume)}`` summed from the RTSTRUCT."""
    import pydicom

    ds = pydicom.dcmread(str(fixture / rtstruct_relpath("planar")))
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    out = {}
    for rc in ds.ROIContourSequence:
        by_z: dict[float, list] = {}
        for c in rc.ContourSequence:
            pts = np.asarray([float(v) for v in c.ContourData]).reshape(-1, 3)
            by_z.setdefault(round(pts[0, 2], 6), []).append(pts[:, :2])
        areas = [slice_area_even_odd(by_z[z]) for z in sorted(by_z)]
        slab = sum(areas) * 1.0  # each contour a 1 mm slab: half a slice beyond each end
        out[names[int(rc.ReferencedROINumber)]] = (slab / 1000, (slab - (areas[0] + areas[-1]) / 2) / 1000)
    return out


def test_each_end_cap_convention_passes_its_own_claim(planar_fixture, tmp_path):
    """Volumes a system would compute from the shipped contours, under each convention.

    Slab (half_slice) and contour-to-contour (none) volumes each pass their own
    claim within 0.6 % and fail the other on the flat-capped cube and cylinder.
    """
    vols = _contour_volumes_cc(planar_fixture)
    half = _config(tmp_path, "claims:\n  end_cap: half_slice\n")
    for i, (cfg, other) in enumerate(((half, load_config()), (load_config(), half))):
        rows = [[roi, repr(v[i])] for roi, v in vols.items()]
        path = _csv(tmp_path / f"conv{i}.csv", ["roi", "volume_cc"], rows)
        own = verify_volumes(path, planar_fixture, cfg)
        assert own.overall_pass, render_text(own)
        assert max(abs(r.error_pct) for r in own.results) < 0.6
        crossed = _by_roi(verify_volumes(path, planar_fixture, other).results)
        assert crossed["cube"].status == Status.FAIL
        assert crossed["cylinder"].status == Status.FAIL


def test_console_record_grades_like_its_projection(planar_fixture, tmp_path):
    """A console import record (one row per ROI attempted, extra columns), read as is."""
    analytic = _analytic_cc(planar_fixture)
    header = ["bundle", "roi", "contour_geometric_type", "import_outcome", "message_verbatim",
              "raystation_volume_cc", "raystation_version", "date", "source", "rtstruct_sha256",
              "fixture_version", "file_outcome", "appears", "displayed_name", "displayed_type",
              "volume_decimals", "raystation_volume_cc_api", "notes"]
    record, projection = [], []
    for i, (roi, v) in enumerate(analytic.items()):
        shown = round(v * (1.004 if i % 2 else 0.97), 2)
        outcome, msg = ("warned", "Contour \"x\", closed") if i == 1 else ("accepted", "")
        if roi == "torus":
            outcome, shown, msg = "rejected_file", None, "File <redacted>, refused | twice"
        record.append(["planar", roi, "CLOSED_PLANAR", outcome, msg, shown, "RS 1.0", "2026-11-18",
                       "suite", "ab" * 32, __version__, outcome if roi == "torus" else "warned",
                       "no" if roi == "torus" else "yes", "", "ROI/Contours", 2, "", "observed"])
        projection.append([roi, shown, "rejected" if roi == "torus" else outcome, msg])
    # A second fixture's rows in the same file, one with a colliding ROI name.
    record.append(["planar", "sphere", "CLOSED_PLANAR", "accepted", "", 1.0, "RS 1.0", "2026-11-18",
                   "research_v4", "cd" * 32, "", "accepted", "yes", "", "ROI/Contours", 2, "", ""])
    rec = verify_volumes(_csv(tmp_path / "raystation_import.csv", header, record), planar_fixture)
    proj = verify_volumes(_csv(tmp_path / "proj.csv", ["roi", "volume_cc", "import_outcome",
                                                       "message"], projection), planar_fixture)
    strip = ("import_outcome", "volume_decimals", "display_quantisation_pct",
             "effective_tolerance_pct", "display_limited", "absent_reason", "violations")
    for a, b in zip(rec.results, proj.results, strict=True):
        da, db = a.as_dict(), b.as_dict()
        assert {k: v for k, v in da.items() if k not in strip} == \
               {k: v for k, v in db.items() if k not in strip}
    # Same verdicts; the record also carries the decimals, so its tolerance can only widen.
    assert [r.status for r in rec.results] == [r.status for r in proj.results]
    t = _by_roi(rec.results)["torus"]
    assert t.status == Status.MISSING and t.absent_reason == "rejected_file"
    assert rec.settings["skipped_rows_other_source"] == 1
    assert rec.settings["volume_column"] == "raystation_volume_cc"


def test_cli_volume_mode_exit_codes_and_json(planar_fixture, tmp_path, capsys):
    analytic = _analytic_cc(planar_fixture)
    good = _csv(tmp_path / "g.csv", ["roi", "volume_cc"], [[k, repr(v)] for k, v in analytic.items()])
    claims = tmp_path / "half.yaml"
    claims.write_text("schema_version: 2\nclaims:\n  end_cap: half_slice\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert cli_main(["verify", "--volumes", str(good), "--fixture", str(planar_fixture),
                     "--config", str(claims), "--report-json", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["mode"] == "volume_only"
    assert d["volume_only"]["end_cap"] == "half_slice"
    assert d["claims"]["end_cap"] == "half_slice"
    assert d["summary"]["passed"] == len(analytic)
    assert {"analytic_volume_cc", "reference_volume_cc", "reported_volume_cc", "error_pct",
            "display_limited", "graded_volume"} <= set(d["results"][0])
    assert "volume only" in capsys.readouterr().out

    assert cli_main(["verify", "--volumes", str(good), "--fixture", str(planar_fixture)]) == 1
    assert cli_main(["verify", "--volumes", str(good), "--fixture", str(planar_fixture),
                     "--volume-tolerance", "0.02"]) == 0
    assert cli_main(["verify", "--volumes", str(good), "--groundtruth", str(tmp_path)]) == 2
    assert cli_main(["verify", "--volumes", str(good), "--predictions", str(tmp_path),
                     "--fixture", str(planar_fixture)]) == 2
    assert cli_main(["verify", "--fixture", str(planar_fixture)]) == 2
    assert cli_main(["verify", "--predictions", str(tmp_path), "--fixture", str(planar_fixture),
                     "--volume-tolerance", "0.02"]) == 2
    assert cli_main(["verify", "--volumes", str(tmp_path / "nope.csv"),
                     "--fixture", str(planar_fixture)]) == 2


def test_fixture_template_lists_every_roi_with_empty_cells(planar_fixture):
    with (planar_fixture / "volumes_template.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == TEMPLATE_COLUMNS
    assert [r[0] for r in rows[1:]] == list(read_manifest(planar_fixture).rois)
    assert all(not any(r[1:]) for r in rows[1:])
    # The blank template is a valid input: every ROI is MISSING.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = verify_volumes(planar_fixture / "volumes_template.csv", planar_fixture)
    assert rep.summary["missing"] == rep.summary["total"]


def test_shipped_template_matches_the_suite_rois():
    """``data/volumes_template.csv`` is what ``generate`` writes for the shipped bundles."""
    path = Path(str(files("rtmask_conformance").joinpath("data/volumes_template.csv")))
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == TEMPLATE_COLUMNS
    assert [r[0] for r in rows[1:]] == [roi.name for rois in BUNDLES.values() for roi in rois]
    assert all(not any(r[1:]) for r in rows[1:])
