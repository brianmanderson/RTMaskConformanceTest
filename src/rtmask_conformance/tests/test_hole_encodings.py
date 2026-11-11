"""The ``xor`` and ``keyhole`` bundles: the hollow sphere and the straw in the
standard's two hole encodings (PS3.3 C.8.8.6.3).

The first half runs on small grids in milliseconds: the twins keep the parents'
inside test, the XOR twins re-type the parents' contours, and the keyhole contour
has the documented shape. The second half generates one fixture and checks what a
tool sees: twin ground truth byte-identical to the originals, the XOR value on every
contour, one keyhole contour per slice whose fill reproduces the cavity, and DS
lengths within the writer's limits.

The rasteriser here is a reference polygon fill at pixel centres (crossing number,
half-open in y), with both the even-odd and the non-zero winding rule. It is a test
oracle only: the suite's own ground truth never reads contours.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from rtmask_conformance import BUNDLES, load_config
from rtmask_conformance._vendor.common.coords import Geometry
from rtmask_conformance._vendor.common.io import sitk_image_to_geometry
from rtmask_conformance._vendor.primitives.base import GeometricType
from rtmask_conformance._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere
from rtmask_conformance.primitives.hole_encodings import (
    HollowCylinderKeyhole,
    HollowCylinderXor,
    HollowSphereKeyhole,
    HollowSphereXor,
    keyhole_points,
)
from rtmask_conformance.verify import Status, verify_fixture

#: twin ROI -> the planar ROI whose geometry and ground truth it shares.
TWINS: dict[str, str] = {
    "hollow_sphere_xor": "hollow_sphere",
    "straw_xor": "straw",
    "hollow_sphere_keyhole": "hollow_sphere",
    "straw_keyhole": "straw",
}

#: Explicit VR Little Endian gives DS a 16-bit value length.
DS_MAX_VALUE_LENGTH = 0xFFFE
DS_MAX_CHARS = 16

SMALL = Geometry(
    origin=(-0.5, -0.5, -0.5), spacing=(1.0, 1.0, 1.0), size=(64, 64, 64),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
SMALL_PARAMS = {
    HollowSphere: dict(center=(31.5, 31.0, 31.0), outer_radius=20.0, inner_radius=10.0),
    HollowCylinder: dict(center=(31.0, 31.5, 31.5), outer_radius=20.0, inner_radius=10.0,
                         height=30.0),
}
TWIN_CLASSES = [
    (HollowSphere, HollowSphereXor, HollowSphereKeyhole),
    (HollowCylinder, HollowCylinderXor, HollowCylinderKeyhole),
]


# --------------------------------------------------------------------------- #
#  Reference polygon fill
# --------------------------------------------------------------------------- #


def _fill(polygons: list[np.ndarray], xs: np.ndarray, ys: np.ndarray, rule: str) -> np.ndarray:
    """Fill closed polygons (each ``(N, 2)``, implicitly closed) at the pixel centres
    ``xs`` x ``ys``; returns a ``(len(ys), len(xs))`` bool mask. All polygons are
    filled together, so two rings under ``evenodd`` give their XOR."""
    X, Y = np.meshgrid(xs, ys)
    count = np.zeros(X.shape, dtype=np.int64)
    for pts in polygons:
        x0, y0 = pts[:, 0], pts[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for a, b, c, d in zip(x0, y0, x1, y1, strict=True):
            if b == d:
                continue  # horizontal edges never cross a half-open scanline
            straddles = (b > Y) != (d > Y)
            x_at = a + (Y - b) * (c - a) / (d - b)
            crosses = straddles & (X < x_at)
            count += np.where(crosses, 1 if rule == "evenodd" else (1 if d > b else -1), 0)
    return (count % 2 == 1) if rule == "evenodd" else (count != 0)


def _pixel_centres(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    xs = geometry.origin[0] + geometry.spacing[0] * np.arange(geometry.size[0])
    ys = geometry.origin[1] + geometry.spacing[1] * np.arange(geometry.size[1])
    return xs, ys


# --------------------------------------------------------------------------- #
#  Unit level: classes and the keyhole shape
# --------------------------------------------------------------------------- #


def test_keyhole_points_shape_start_and_winding() -> None:
    t = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    # Start the input rings away from +x to check the rotation to the +x vertex.
    outer = np.stack([10 * np.cos(t), 10 * np.sin(t), np.zeros(8)], axis=1)[[3, 4, 5, 6, 7, 0, 1, 2]]
    inner = np.stack([4 * np.cos(t), 4 * np.sin(t), np.zeros(8)], axis=1)[[5, 6, 7, 0, 1, 2, 3, 4]]
    pts = keyhole_points(outer, inner)
    assert pts.shape == (8 + 8 + 2, 3)
    np.testing.assert_allclose(pts[0], [10, 0, 0], atol=1e-12)    # outer start: +x vertex
    np.testing.assert_allclose(pts[8], pts[0])                     # outer ring closed
    np.testing.assert_allclose(pts[9], [4, 0, 0], atol=1e-12)     # channel: same angle
    np.testing.assert_allclose(pts[-1], pts[9])                    # inner ring closed
    # Implicit closing edge pts[-1] -> pts[0] is the channel back: the two channel
    # edges coincide (zero width), and lie on y = 0.
    assert pts[8][1] == pts[9][1] == pts[-1][1] == pts[0][1] == 0.0

    def signed_area(ring: np.ndarray) -> float:
        x, y = ring[:, 0], ring[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    assert signed_area(pts[:8]) > 0          # outer: increasing angle
    assert signed_area(pts[9:17]) < 0        # inner: decreasing angle


@pytest.mark.parametrize(("parent", "xor_cls", "keyhole_cls"), TWIN_CLASSES)
def test_twins_share_parameters_inside_test_and_spec(parent, xor_cls, keyhole_cls) -> None:
    params = SMALL_PARAMS[parent]
    p = parent(name="p", **params)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 63.0, size=(20000, 3))
    for cls in (xor_cls, keyhole_cls):
        twin = cls(name="t", **params)
        assert isinstance(twin, parent)
        assert [f.name for f in fields(twin) if f.init] == [f.name for f in fields(p) if f.init]
        np.testing.assert_array_equal(twin.is_inside(pts), p.is_inside(pts))
        assert twin.analytical_volume() == p.analytical_volume()
        spec = twin.to_spec_dict()
        assert spec["parameters"] == p.to_spec_dict()["parameters"]
        assert spec["geometric_type"] == twin.geometric_type.value
    assert xor_cls(name="t", **params).geometric_type is GeometricType.CLOSEDPLANAR_XOR
    assert keyhole_cls(name="t", **params).geometric_type is GeometricType.CLOSED_PLANAR


@pytest.mark.parametrize(("parent", "xor_cls", "keyhole_cls"), TWIN_CLASSES)
def test_xor_twin_retypes_every_parent_contour(parent, xor_cls, keyhole_cls) -> None:
    params = SMALL_PARAMS[parent]
    base = parent(name="p", **params).get_contours(SMALL)
    xor = xor_cls(name="t", **params).get_contours(SMALL)
    assert len(xor) == len(base)
    for a, b in zip(base, xor, strict=True):
        assert a.geometric_type is GeometricType.CLOSED_PLANAR
        assert b.geometric_type is GeometricType.CLOSEDPLANAR_XOR
        assert b.slice_indices == a.slice_indices
        np.testing.assert_array_equal(b.points_xyz, a.points_xyz)


@pytest.mark.parametrize(("parent", "xor_cls", "keyhole_cls"), TWIN_CLASSES)
def test_keyhole_twin_one_contour_per_slice_and_same_fill(parent, xor_cls, keyhole_cls) -> None:
    params = SMALL_PARAMS[parent]
    base = parent(name="p", **params).get_contours(SMALL)
    keyhole = keyhole_cls(name="t", **params).get_contours(SMALL)
    rings_by_slice: dict[int, list[np.ndarray]] = {}
    for c in base:
        rings_by_slice.setdefault(c.slice_indices[0], []).append(c.points_xyz)
    assert [c.slice_indices[0] for c in keyhole] == list(rings_by_slice)
    assert any(len(r) == 2 for r in rings_by_slice.values())
    xs, ys = _pixel_centres(SMALL)
    for c in keyhole:
        rings = rings_by_slice[c.slice_indices[0]]
        assert c.geometric_type is GeometricType.CLOSED_PLANAR
        assert len(c.points_xyz) == sum(len(r) for r in rings) + (2 if len(rings) == 2 else 0)
        expected = _fill([r[:, :2] for r in rings], xs, ys, "evenodd")
        for rule in ("evenodd", "nonzero"):
            np.testing.assert_array_equal(_fill([c.points_xyz[:, :2]], xs, ys, rule), expected)


# --------------------------------------------------------------------------- #
#  Fixture level
# --------------------------------------------------------------------------- #

e2e = pytest.mark.skipif(
    os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set",
)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rtmask_conformance.generate import GenerateOptions, generate_fixture

    out = tmp_path_factory.mktemp("hole_encodings") / "fixture"
    generate_fixture(out, options=GenerateOptions(n_quadrature=2))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contours_by_roi(rtstruct_path: Path) -> dict[str, list[pydicom.Dataset]]:
    ds = pydicom.dcmread(rtstruct_path)
    names = {item.ROINumber: item.ROIName for item in ds.StructureSetROISequence}
    return {names[rc.ReferencedROINumber]: list(rc.ContourSequence)
            for rc in ds.ROIContourSequence}


def _points(contour: pydicom.Dataset) -> np.ndarray:
    return np.asarray([float(v) for v in contour.ContourData]).reshape(-1, 3)


def _slice_key(contour: pydicom.Dataset) -> str:
    return contour.ContourImageSequence[0].ReferencedSOPInstanceUID


@e2e
def test_bundles_and_manifest(fixture: Path) -> None:
    from rtmask_conformance.manifest import read_manifest

    assert [r.name for r in BUNDLES["xor"]] == ["hollow_sphere_xor", "straw_xor"]
    assert [r.name for r in BUNDLES["keyhole"]] == ["hollow_sphere_keyhole", "straw_keyhole"]
    m = read_manifest(fixture)
    assert list(m.bundles) == ["planar", "xor", "keyhole"]
    for bundle in ("xor", "keyhole"):
        entry = m.bundles[bundle]
        assert entry.rtstruct_path == f"rtstruct/primitives_{bundle}.dcm"
        assert entry.rtstruct_sha256 == _sha256(fixture / entry.rtstruct_path)
        for roi in entry.rois:
            r = m.rois[roi]
            assert r.bundle == bundle and r.graded and r.truth_kind == "binary_mask"
    assert m.rois["hollow_sphere_xor"].geometric_type == "CLOSEDPLANAR_XOR"
    assert m.rois["straw_xor"].geometric_type == "CLOSEDPLANAR_XOR"
    assert m.rois["hollow_sphere_keyhole"].geometric_type == "CLOSED_PLANAR"
    assert m.rois["straw_keyhole"].geometric_type == "CLOSED_PLANAR"


@e2e
def test_twin_ground_truth_byte_identical_to_original(fixture: Path) -> None:
    gt = fixture / "groundtruth"
    table = {twin: (_sha256(gt / f"{twin}.nii.gz"), _sha256(gt / f"{orig}.nii.gz"))
             for twin, orig in TWINS.items()}
    print("\ntwin                    original        sha256 (twin == original)")
    for twin, (a, b) in table.items():
        print(f"{twin:<23} {TWINS[twin]:<15} {a[:16]}... {'==' if a == b else '!='} {b[:16]}...")
    assert all(a == b for a, b in table.values()), table


@e2e
def test_xor_rtstruct_carries_the_standard_value_on_every_contour(fixture: Path) -> None:
    path = fixture / "rtstruct" / "primitives_xor.dcm"
    by_roi = _contours_by_roi(path)
    assert sorted(by_roi) == ["hollow_sphere_xor", "straw_xor"]
    planar = _contours_by_roi(fixture / "rtstruct" / "primitives_planar.dcm")
    raw = pydicom.dcmread(path)
    for roi, contours in by_roi.items():
        assert len(contours) == len(planar[TWINS[roi]])
        for c in contours:
            assert c.ContourGeometricType == "CLOSEDPLANAR_XOR"
            assert len(c.ContourGeometricType) == 16
    for rc in raw.ROIContourSequence:  # the raw bytes, not pydicom's decoded value
        for c in rc.ContourSequence:
            assert c.get_item(0x30060042).value == b"CLOSEDPLANAR_XOR"


@e2e
def test_keyhole_rtstruct_one_contour_per_slice_and_cavity_reproduced(fixture: Path) -> None:
    keyhole = _contours_by_roi(fixture / "rtstruct" / "primitives_keyhole.dcm")
    planar = _contours_by_roi(fixture / "rtstruct" / "primitives_planar.dcm")
    geometry = sitk_image_to_geometry(sitk.ReadImage(str(fixture / "groundtruth" / "straw.nii.gz")))
    xs, ys = _pixel_centres(geometry)
    for roi in ("hollow_sphere_keyhole", "straw_keyhole"):
        prim = next(r.primitive for r in BUNDLES["keyhole"] if r.name == roi)
        cx, cy, _ = prim.center
        contours = keyhole[roi]
        keys = [_slice_key(c) for c in contours]
        assert len(keys) == len(set(keys)), f"{roi}: more than one contour on a slice"
        rings: dict[str, list[np.ndarray]] = {}
        for c in planar[TWINS[roi]]:
            rings.setdefault(_slice_key(c), []).append(_points(c))
        assert set(keys) == set(rings)
        sel_x = np.abs(xs - cx) <= prim.outer_radius + 2
        sel_y = np.abs(ys - cy) <= prim.outer_radius + 2
        X, Y = np.meshgrid(xs[sel_x], ys[sel_y])
        n_cavity = 0
        for c in contours:
            assert c.ContourGeometricType == "CLOSED_PLANAR"
            pts = _points(c)
            ring_pts = rings[_slice_key(c)]
            assert len(np.unique(pts[:, 2])) == 1
            mask = _fill([pts[:, :2]], xs[sel_x], ys[sel_y], "evenodd")
            expected = _fill([r[:, :2] for r in ring_pts], xs[sel_x], ys[sel_y], "evenodd")
            np.testing.assert_array_equal(mask, expected)
            np.testing.assert_array_equal(
                _fill([pts[:, :2]], xs[sel_x], ys[sel_y], "nonzero"), expected)
            if len(ring_pts) == 2:
                n_cavity += 1
                assert len(pts) == len(ring_pts[0]) + len(ring_pts[1]) + 2
                inner_r = float(np.max(np.hypot(ring_pts[1][:, 0] - cx, ring_pts[1][:, 1] - cy)))
                outer_r = float(np.min(np.hypot(ring_pts[0][:, 0] - cx, ring_pts[0][:, 1] - cy)))
                rho = np.hypot(X - cx, Y - cy)
                cavity = rho < inner_r * np.cos(np.pi / 64) - 0.01  # inside the inner polygon
                wall = (rho > inner_r + 0.01) & (rho < outer_r * np.cos(np.pi / 64) - 0.01)
                assert cavity.any() and not mask[cavity].any(), f"{roi}: cavity filled"
                assert mask[wall].all(), f"{roi}: wall not filled"
        assert n_cavity > 0


@e2e
@pytest.mark.parametrize("bundle", ["planar", "xor", "keyhole"])
def test_contour_data_within_ds_limits(fixture: Path, bundle: str) -> None:
    ds = pydicom.dcmread(fixture / "rtstruct" / f"primitives_{bundle}.dcm")
    longest = 0
    for rc in ds.ROIContourSequence:
        for c in rc.ContourSequence:
            raw = c.get_item(0x30060050).value
            assert len(raw) <= DS_MAX_VALUE_LENGTH
            assert max(len(v) for v in raw.decode("ascii").strip().split("\\")) <= DS_MAX_CHARS
            assert int(c.NumberOfContourPoints) * 3 == len(c.ContourData)
            longest = max(longest, len(raw))
    print(f"\n{bundle}: longest ContourData {longest} bytes (limit {DS_MAX_VALUE_LENGTH})")


@e2e
def test_verify_fixture_grades_twins_like_their_originals(fixture: Path, tmp_path: Path) -> None:
    report = verify_fixture(fixture / "groundtruth", fixture, load_config())
    statuses = {r.roi: r.status for r in report.results}
    assert all(statuses[t] == Status.PASS for t in TWINS), statuses
    assert report.overall_pass

    # A tool that does not claim CLOSEDPLANAR_XOR and writes nothing for the xor
    # bundle: UNSUPPORTED_TYPE there, and the run still passes.
    pred = tmp_path / "pred"
    pred.mkdir()
    for roi in statuses:
        if roi not in ("hollow_sphere_xor", "straw_xor"):
            (pred / f"{roi}.nii.gz").write_bytes((fixture / "groundtruth" / f"{roi}.nii.gz")
                                                 .read_bytes())
    cfg_path = tmp_path / "no_xor.yaml"
    cfg_path.write_text("schema_version: 2\nclaims:\n  geometric_types: [CLOSED_PLANAR]\n",
                        encoding="utf-8")
    report = verify_fixture(pred, fixture, load_config(cfg_path))
    statuses = {r.roi: r.status for r in report.results}
    assert statuses["hollow_sphere_xor"] == statuses["straw_xor"] == Status.UNSUPPORTED_TYPE
    assert statuses["hollow_sphere_keyhole"] == statuses["straw_keyhole"] == Status.PASS
    assert report.overall_pass
