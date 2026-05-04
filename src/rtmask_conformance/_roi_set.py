"""The seven ROIs that make up the v0.1 conformance suite.

Single source of truth: every other module imports ``CONFORMANCE_ROIS`` and
``build_conformance_primitives()`` from here.

These are closed-planar primitives only. Tools that ship support for
POINT or OPEN_NONPLANAR ROI types are encouraged but not required by v0.1.
"""

from __future__ import annotations

from ._vendor.primitives.base import AnalyticalPrimitive
from ._vendor.primitives.closed_planar import Cube, Cylinder, Ellipsoid, Sphere, Torus
from ._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere

CONFORMANCE_ROIS: list[str] = [
    "sphere",
    "cube",
    "cylinder",
    "ellipsoid",
    "torus",
    "hollow_sphere",
    "straw",
]


def build_conformance_primitives() -> list[AnalyticalPrimitive]:
    """Construct the seven analytical primitives in canonical order.

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
