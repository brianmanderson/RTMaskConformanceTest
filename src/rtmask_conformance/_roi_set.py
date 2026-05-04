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
    "sphere_r40_center",
    "cube_s60_x100_y100",
    "cylinder_r30_h80_x400_y100",
    "ellipsoid_30_50_60_x100_y400",
    "torus_R60_r20_x400_y400",
    "hollow_sphere_R40_r20_x256_y100",
    "straw_R40_r20_h120_x256_y400",
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
            name="sphere_r40_center",
            center=(256.0, 256.0, 100.0),
            radius=40.0,
        ),
        Cube(
            name="cube_s60_x100_y100",
            center=(100.5, 100.5, 100.5),
            side_length=60.0,
        ),
        Cylinder(
            name="cylinder_r30_h80_x400_y100",
            center=(400.0, 100.0, 100.5),
            radius=30.0,
            height=80.0,
        ),
        Ellipsoid(
            name="ellipsoid_30_50_60_x100_y400",
            center=(100.0, 400.0, 100.0),
            semi_axes=(30.0, 50.0, 60.0),
        ),
        Torus(
            name="torus_R60_r20_x400_y400",
            center=(400.0, 400.0, 100.0),
            major_radius=60.0,
            minor_radius=20.0,
        ),
        HollowSphere(
            name="hollow_sphere_R40_r20_x256_y100",
            center=(256.0, 100.0, 100.0),
            outer_radius=40.0,
            inner_radius=20.0,
        ),
        HollowCylinder(
            name="straw_R40_r20_h120_x256_y400",
            center=(256.0, 400.0, 100.5),
            outer_radius=40.0,
            inner_radius=20.0,
            height=120.0,
        ),
    ]
