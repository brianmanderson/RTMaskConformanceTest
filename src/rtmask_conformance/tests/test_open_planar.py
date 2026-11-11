"""OPEN_PLANAR: the line segment and the open arc of the ``pointline`` bundle.

Three groups of tests:

* the analytic primitives: closed-form arc length, open-arc validation, the
  spec record the curve metrics read;
* one slice each: every vertex of each ROI lies on a single CT slice, both in
  the primitive's contours and in the RTSTRUCT the vendored writer emits;
* the ``arc_length_error_pct`` ceilings in ``default_thresholds.yaml``. The
  suite's ideal rasteriser below (a one-voxel-wide, 8-connected Bresenham
  trace of the emitted polyline on its slice) must pass them, each ceiling
  must be twice the error that rasteriser measures, and the arc closed into a
  filled polygon must fail.
"""

from __future__ import annotations

import math
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pydicom
import pytest
import yaml
from scipy.ndimage import binary_fill_holes

from rtmask_conformance._roi_set import BUNDLES, ConformanceROI, _open_planar_rois
from rtmask_conformance._vendor.common.coords import Geometry
from rtmask_conformance._vendor.primitives.base import AnalyticalPrimitive, GeometricType
from rtmask_conformance._vendor.refimage.build_reference_ct import (
    ReferenceCTSpec,
    build_reference_ct,
)
from rtmask_conformance._vendor.rtstruct.pydicom_writer import build_rtstruct
from rtmask_conformance.primitives.open_arc import OpenArc2D

LINE = "line_segment_x200_y200_to_x300_y280_z175"
ARC = "open_arc_r40_90deg_x380_y300_z180"

_SPEC = ReferenceCTSpec()
GEOMETRY = Geometry(origin=_SPEC.origin, spacing=_SPEC.voxel_size, size=_SPEC.size)

def _rois() -> dict[str, ConformanceROI]:
    return {roi.name: roi for roi in _open_planar_rois()}


# --------------------------------------------------------------------------
#  The suite's ideal rasteriser for a planar polyline
# --------------------------------------------------------------------------


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer 8-connected line from (x0, y0) to (x1, y1), both ends included."""
    pts: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return pts
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _polyline_xyz(primitive: AnalyticalPrimitive) -> np.ndarray:
    (contour,) = primitive.get_contours(GEOMETRY)
    return np.asarray(contour.points_xyz, dtype=np.float64)


def _trace(points_xyz: np.ndarray) -> np.ndarray:
    """``(z, y, x)`` mask of the Bresenham trace through ``points_xyz`` on its slice."""
    origin = np.asarray(GEOMETRY.origin)
    spacing = np.asarray(GEOMETRY.spacing)
    ijk = np.rint((points_xyz - origin) / spacing).astype(int)
    (k,) = set(ijk[:, 2].tolist())
    nx, ny, nz = GEOMETRY.size
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    for a, b in zip(ijk[:-1], ijk[1:], strict=True):
        for x, y in _bresenham(int(a[0]), int(a[1]), int(b[0]), int(b[1])):
            mask[k, y, x] = 1
    return mask


def ideal_mask(primitive: AnalyticalPrimitive) -> np.ndarray:
    """One-voxel-wide rasterisation of an OPEN_PLANAR primitive's polyline."""
    return _trace(_polyline_xyz(primitive))


def closed_filled_mask(primitive: AnalyticalPrimitive) -> np.ndarray:
    """The polyline closed back to its first vertex and filled: a tool that
    reads OPEN_PLANAR as CLOSED_PLANAR."""
    pts = _polyline_xyz(primitive)
    outline = _trace(np.vstack([pts, pts[:1]]))
    return np.stack([binary_fill_holes(s) for s in outline]).astype(np.uint8)


def _ceiling(roi: str) -> float:
    path = Path(str(files("rtmask_conformance").joinpath("data/default_thresholds.yaml")))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return float(data["pointline"][roi]["arc_length_error_pct"])


def _arc_length_error_pct(mask: np.ndarray, primitive: AnalyticalPrimitive) -> float:
    from rtmask_conformance._vendor.metrics import arc_length_metrics

    result = arc_length_metrics(mask, primitive.arc_length_mm, GEOMETRY.spacing)
    assert result.arc_length_error_pct is not None
    return float(result.arc_length_error_pct)


# --------------------------------------------------------------------------
#  Analytic primitives
# --------------------------------------------------------------------------


def test_open_arc_closed_form() -> None:
    arc = _rois()[ARC].primitive
    assert isinstance(arc, OpenArc2D)
    assert arc.arc_length_mm == pytest.approx(40.0 * math.pi / 2, rel=1e-12)

    pts = arc.polyline()
    assert pts.shape == (65, 3)
    radii = np.hypot(pts[:, 0] - 380.0, pts[:, 1] - 300.0)
    np.testing.assert_allclose(radii, 40.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(pts[0], (420.0, 300.0, 180.0), atol=1e-9)
    np.testing.assert_allclose(pts[-1], (380.0, 340.0, 180.0), atol=1e-9)
    chords = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    assert chords < arc.arc_length_mm
    assert chords == pytest.approx(arc.arc_length_mm, rel=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"sweep_deg": 0.0},
        {"sweep_deg": 360.0},
        {"n_samples": 2},
    ],
)
def test_open_arc_rejects_closed_or_degenerate(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        OpenArc2D(name="bad", **{"radius": 10.0, **kwargs})


@pytest.mark.parametrize("name", [LINE, ARC])
def test_roi_metadata_and_spec(name: str) -> None:
    roi = _rois()[name]
    assert roi.bundle == "pointline"
    assert roi.truth_kind == "curve"
    assert roi.graded is True
    assert roi.geometric_type is GeometricType.OPEN_PLANAR

    spec = roi.primitive.to_spec_dict()
    assert spec["geometric_type"] == "OPEN_PLANAR"
    assert spec["analytical_volume_mm3"] is None
    assert spec["analytical_arc_length_mm"] == pytest.approx(roi.primitive.arc_length_mm)
    assert roi.primitive.analytical_surface_mesh() is None
    assert not roi.primitive.is_inside(np.zeros((4, 3))).any()


def test_line_segment_arc_length() -> None:
    line = _rois()[LINE].primitive
    assert line.arc_length_mm == pytest.approx(math.hypot(100.0, 80.0), rel=1e-12)


# --------------------------------------------------------------------------
#  One slice each
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [LINE, ARC])
def test_contour_lies_on_one_slice(name: str) -> None:
    contours = _rois()[name].primitive.get_contours(GEOMETRY)
    assert len(contours) == 1
    (c,) = contours
    assert c.geometric_type is GeometricType.OPEN_PLANAR
    z = np.unique(c.points_xyz[:, 2])
    assert z.size == 1, f"{name}: vertices span z = {z}"
    k = (float(z[0]) - GEOMETRY.origin[2]) / GEOMETRY.spacing[2]
    assert k == pytest.approx(round(k), abs=1e-9), f"{name}: z = {z[0]} is not a slice position"
    assert 0 <= round(k) < GEOMETRY.size[2]
    assert c.slice_indices == [round(k)]


@pytest.fixture(scope="module")
def open_planar_rtstruct(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    if os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1":
        pytest.skip("RTMASK_CONFORMANCE_SKIP_E2E=1 set")
    root = tmp_path_factory.mktemp("open_planar")
    refct = root / "refct"
    build_reference_ct(refct, spec=_SPEC)
    rtss = root / "open_planar.dcm"
    build_rtstruct(
        primitives=[roi.primitive for roi in _open_planar_rois()],
        ref_image_folder=refct,
        out_path=rtss,
        structure_set_label="OPEN_PLANAR",
    )
    return refct, rtss


@pytest.mark.parametrize("name", [LINE, ARC])
def test_rtstruct_contour_lies_on_one_slice(
    name: str, open_planar_rtstruct: tuple[Path, Path]
) -> None:
    refct, rtss_path = open_planar_rtstruct
    slice_z = {}
    for f in refct.glob("*.dcm"):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        slice_z[ds.SOPInstanceUID] = float(ds.ImagePositionPatient[2])

    rtss = pydicom.dcmread(rtss_path)
    number = {r.ROIName: r.ROINumber for r in rtss.StructureSetROISequence}[name]
    (roi_contour,) = [
        rc for rc in rtss.ROIContourSequence if rc.ReferencedROINumber == number
    ]
    assert len(roi_contour.ContourSequence) == 1
    (item,) = roi_contour.ContourSequence
    assert item.ContourGeometricType == "OPEN_PLANAR"
    xyz = np.asarray(item.ContourData, dtype=np.float64).reshape(-1, 3)
    assert int(item.NumberOfContourPoints) == xyz.shape[0]
    assert np.unique(xyz[:, 2]).size == 1

    (ref,) = item.ContourImageSequence
    assert ref.ReferencedSOPInstanceUID in slice_z
    assert slice_z[ref.ReferencedSOPInstanceUID] == pytest.approx(xyz[0, 2], abs=1e-6)


# --------------------------------------------------------------------------
#  Thresholds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [LINE, ARC])
def test_ceiling_is_twice_the_ideal_discretisation_error(name: str) -> None:
    primitive = _rois()[name].primitive
    measured = _arc_length_error_pct(ideal_mask(primitive), primitive)
    ceiling = _ceiling(name)
    assert measured > 0.0
    assert measured <= ceiling
    assert ceiling == pytest.approx(2.0 * measured, abs=0.01)


@pytest.mark.parametrize("name", [LINE, ARC])
def test_ideal_rasterisation_traces_the_curve(name: str) -> None:
    from rtmask_conformance._vendor.metrics import curve_metrics, densify_polyline

    primitive = _rois()[name].primitive
    pts = _polyline_xyz(primitive)
    cm = curve_metrics(
        ideal_mask(primitive), densify_polyline(pts), GEOMETRY, endpoints_mm=pts[[0, -1]]
    )
    assert cm.coverage_recall == 1.0
    assert cm.max_curve_distance_mm < 1.0
    assert cm.voxel_to_curve_max_mm < 1.0


def test_closed_filled_arc_fails_on_arc_length() -> None:
    arc = _rois()[ARC].primitive
    filled = closed_filled_mask(arc)
    assert int(filled.sum()) > 5 * int(ideal_mask(arc).sum())
    assert _arc_length_error_pct(filled, arc) > _ceiling(ARC)


def test_registered_in_pointline_bundle() -> None:
    registered = {roi.name: roi for roi in BUNDLES["pointline"]}
    for name in (LINE, ARC):
        roi = registered[name]
        assert roi.truth_kind == "curve"
        assert roi.graded is True
        assert roi.geometric_type is GeometricType.OPEN_PLANAR
