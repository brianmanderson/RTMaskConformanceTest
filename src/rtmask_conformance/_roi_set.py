"""The ROI bundles that make up the conformance suite.

Single source of truth: every other module imports ``BUNDLES``,
``CONFORMANCE_ROIS`` and ``build_conformance_primitives()`` from here.

A *bundle* is a group of ROIs shipped in one RTSTRUCT
(``rtstruct/primitives_<bundle>.dcm``). Every ROI is a
:class:`ConformanceROI`: the analytical primitive plus the suite metadata
the generator, the manifest and the verifier need:

* ``name`` and ``geometric_type`` — taken from the primitive;
* ``bundle`` — the bundle the ROI ships in;
* ``truth_kind`` — what ground truth the generator writes for it (one of
  :data:`TRUTH_KINDS`);
* ``graded`` — whether ``verify`` scores it against thresholds. Ungraded ROIs
  are shipped (and recorded in the manifest) but never affect the verdict.

The bundle names in :data:`BUNDLE_NAMES` are reserved. ``planar`` holds the
seven closed-planar primitives of v0.1, unchanged; ``xor`` and ``keyhole`` hold
the hollow sphere and the straw again, in the standard's two hole encodings
(:mod:`rtmask_conformance.primitives.hole_encodings`), with the same geometry
and therefore the same ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ._vendor.primitives.base import AnalyticalPrimitive, GeometricType
from ._vendor.primitives.closed_planar import Cube, Cylinder, Ellipsoid, Sphere, Torus
from ._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere
from .primitives.hole_encodings import (
    HollowCylinderKeyhole,
    HollowCylinderXor,
    HollowSphereKeyhole,
    HollowSphereXor,
)

#: Reserved bundle names, in canonical order.
BUNDLE_NAMES: tuple[str, ...] = ("planar", "xor", "keyhole", "pointline", "negative")

#: Ground-truth kinds a ROI can carry. Only ``binary_mask`` has a generator and
#: a grader today; the others are reserved for later bundles.
TRUTH_KINDS: tuple[str, ...] = ("binary_mask", "float_field", "curve", "points", "none")


@dataclass(frozen=True)
class ConformanceROI:
    """One ROI of the suite: an analytical primitive plus its bundle metadata."""

    primitive: AnalyticalPrimitive
    bundle: str
    truth_kind: str
    graded: bool

    def __post_init__(self) -> None:
        if self.bundle not in BUNDLE_NAMES:
            raise ValueError(f"{self.name}: unknown bundle {self.bundle!r}")
        if self.truth_kind not in TRUTH_KINDS:
            raise ValueError(f"{self.name}: unknown truth_kind {self.truth_kind!r}")

    @property
    def name(self) -> str:
        return self.primitive.name

    @property
    def geometric_type(self) -> GeometricType:
        return self.primitive.geometric_type


def _planar_primitives() -> list[AnalyticalPrimitive]:
    """The seven v0.1 analytical primitives in canonical order.

    Geometry is locked: the centers and dimensions match the upstream
    PRIMITIVE_REGISTRY entries, so the partial-volume ground-truth NIfTIs
    a tool sees in `<fixture>/groundtruth/` are bit-identical regardless of
    whether they were generated here or via upstream rtmask_validation.
    """
    return [
        Sphere(
            name="sphere",
            center=(256.0, 256.0, 100.0),
            radius=40.0,
        ),
        Cube(
            name="cube",
            center=(100.5, 100.5, 100.5),
            side_length=60.0,
        ),
        Cylinder(
            name="cylinder",
            center=(400.0, 100.0, 100.5),
            radius=30.0,
            height=80.0,
        ),
        Ellipsoid(
            name="ellipsoid",
            center=(100.0, 400.0, 100.0),
            semi_axes=(30.0, 50.0, 60.0),
        ),
        Torus(
            name="torus",
            center=(400.0, 400.0, 100.0),
            major_radius=60.0,
            minor_radius=20.0,
        ),
        HollowSphere(
            name="hollow_sphere",
            center=(256.0, 100.0, 100.0),
            outer_radius=40.0,
            inner_radius=20.0,
        ),
        HollowCylinder(
            name="straw",
            center=(256.0, 400.0, 100.5),
            outer_radius=40.0,
            inner_radius=20.0,
            height=120.0,
        ),
    ]


def _hole_encoding_twins(encoding: str) -> list[AnalyticalPrimitive]:
    """``hollow_sphere`` and ``straw`` re-encoded, named ``<name>_<encoding>``.

    Built from the planar primitives' own parameters, so the twins cannot drift
    from them: same centre, radii and height, same inside test, same ground truth.
    """
    classes = {
        "xor": {HollowSphere: HollowSphereXor, HollowCylinder: HollowCylinderXor},
        "keyhole": {HollowSphere: HollowSphereKeyhole, HollowCylinder: HollowCylinderKeyhole},
    }[encoding]
    twins: list[AnalyticalPrimitive] = []
    for p in _planar_primitives():
        if type(p) in classes:
            params = {f.name: getattr(p, f.name) for f in fields(p) if f.init}
            params["name"] = f"{p.name}_{encoding}"
            twins.append(classes[type(p)](**params))
    return twins


def build_bundles() -> dict[str, list[ConformanceROI]]:
    """Construct every populated bundle, keyed by bundle name in canonical order.

    Returns fresh primitive instances on each call. Empty bundles are omitted,
    so no RTSTRUCT is written for them.
    """
    bundles: dict[str, list[ConformanceROI]] = {
        "planar": [
            ConformanceROI(primitive=p, bundle="planar", truth_kind="binary_mask", graded=True)
            for p in _planar_primitives()
        ],
        **{
            enc: [
                ConformanceROI(primitive=p, bundle=enc, truth_kind="binary_mask", graded=True)
                for p in _hole_encoding_twins(enc)
            ]
            for enc in ("xor", "keyhole")
        },
    }
    names = [roi.name for rois in bundles.values() for roi in rois]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"ROI names must be unique across bundles: {duplicates}")
    return {name: bundles[name] for name in BUNDLE_NAMES if bundles.get(name)}


BUNDLES: dict[str, list[ConformanceROI]] = build_bundles()

#: Graded ROI names across all bundles, in canonical order. Kept under the v0.1
#: name so existing imports keep working.
CONFORMANCE_ROIS: list[str] = [
    roi.name for rois in BUNDLES.values() for roi in rois if roi.graded
]


def build_conformance_primitives() -> list[AnalyticalPrimitive]:
    """Construct the primitives of every graded ROI, in ``CONFORMANCE_ROIS`` order."""
    return [roi.primitive for rois in build_bundles().values() for roi in rois if roi.graded]
