"""The hollow sphere and the straw in the two hole encodings the standard defines.

DICOM PS3.3 C.8.8.6.3 defines two ways to encode a contour with a hole:

* **CLOSEDPLANAR_XOR**: coplanar contours whose Contour Geometric Type is
  ``CLOSEDPLANAR_XOR`` are combined by exclusive or. Every contour of the ROI
  carries the value.
* **Keyhole**: one ``CLOSED_PLANAR`` contour that runs round the outer
  boundary, along an arbitrarily narrow channel to the inner boundary, round
  it, and back along the channel.

The vendored :class:`HollowSphere` and :class:`HollowCylinder` (the ``planar``
bundle's ``hollow_sphere`` and ``straw``) use neither: they emit two coplanar
``CLOSED_PLANAR`` contours per slice, read even-odd by convention. The classes
here re-encode the same contours. They inherit the parent's parameters and its
:meth:`is_inside`, so their analytic ground truth is the parent's, voxel for
voxel; only :meth:`get_contours` (and the type they report) differs.

Keyhole contour, per slice that has both rings (PS3.3 says nothing on winding
or start point, so this module fixes both):

* start at the outer-ring vertex with the largest x (the vertex at angle 0,
  i.e. on the +x side of the centre);
* traverse the outer ring once in the parent's order (increasing angle
  ``atan2(y - cy, x - cx)``) and repeat the start vertex to close it;
* step along the channel to the inner-ring vertex with the largest x (same
  angle as the outer start, so the channel is radial and lies on ``y = cy``);
* traverse the inner ring once in the **opposite** order (decreasing angle)
  and repeat its start vertex;
* the implicit closing edge of the contour, inner start to outer start, is the
  channel back.

The channel has zero width: its two edges coincide. The opposite traversal
makes the cavity empty under both the even-odd and the non-zero winding fill
rules. A slice where the cavity is absent (or below one pixel in radius, which
the parent drops) keeps the parent's single outer contour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._vendor.primitives.base import ContourItem, GeometricType
from .._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere

__all__ = [
    "HollowCylinderKeyhole",
    "HollowCylinderXor",
    "HollowSphereKeyhole",
    "HollowSphereXor",
    "keyhole_points",
]


class _XorTagged:
    """Mixin: the parent's contours, every one typed ``CLOSEDPLANAR_XOR``."""

    def get_contours(self, geometry) -> list[ContourItem]:
        return [
            ContourItem(
                points_xyz=c.points_xyz,
                geometric_type=GeometricType.CLOSEDPLANAR_XOR,
                slice_indices=list(c.slice_indices),
            )
            for c in super().get_contours(geometry)  # type: ignore[misc]
        ]

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()  # type: ignore[misc]
        d["encoding"] = "closedplanar_xor"
        return d


def keyhole_points(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Join an outer and an inner ring into one keyhole contour.

    Both rings are ``(N, 3)`` arrays of coplanar points in traversal order,
    without a repeated closing vertex. Each ring is rotated to start at its
    vertex of largest x (the first such vertex on a tie); the inner ring is then
    reversed. The result is ``outer..., outer[0], inner[0], inner reversed...,
    inner[0]``: ``len(outer) + len(inner) + 2`` points, whose implicit closing
    edge (``inner[0]`` to ``outer[0]``) is the return leg of the channel.
    """
    outer = np.asarray(outer, dtype=np.float64)
    inner = np.asarray(inner, dtype=np.float64)
    outer = np.roll(outer, -int(np.argmax(outer[:, 0])), axis=0)
    inner = np.roll(inner, -int(np.argmax(inner[:, 0])), axis=0)
    inner_reversed = np.concatenate([inner[:1], inner[:0:-1]], axis=0)
    return np.concatenate([outer, outer[:1], inner_reversed, inner[:1]], axis=0)


class _Keyhole:
    """Mixin: one keyhole ``CLOSED_PLANAR`` contour per slice with a cavity."""

    def get_contours(self, geometry) -> list[ContourItem]:
        parent = super().get_contours(geometry)  # type: ignore[misc]
        by_slice: dict[tuple[int, ...], list[ContourItem]] = {}
        for c in parent:
            by_slice.setdefault(tuple(c.slice_indices), []).append(c)
        contours: list[ContourItem] = []
        for slice_indices, rings in by_slice.items():
            if len(rings) == 1:
                contours.append(rings[0])
                continue
            if len(rings) != 2:  # pragma: no cover - the parents emit at most two
                raise ValueError(f"{len(rings)} contours on slice {slice_indices}")
            outer, inner = rings  # the parents emit the outer ring first
            contours.append(
                ContourItem(
                    points_xyz=keyhole_points(outer.points_xyz, inner.points_xyz),
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=list(slice_indices),
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()  # type: ignore[misc]
        d["encoding"] = "keyhole_zero_width_channel"
        return d


@dataclass
class HollowSphereXor(_XorTagged, HollowSphere):
    """:class:`HollowSphere` with every contour typed ``CLOSEDPLANAR_XOR``."""

    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSEDPLANAR_XOR)


@dataclass
class HollowCylinderXor(_XorTagged, HollowCylinder):
    """:class:`HollowCylinder` (the straw) with every contour typed ``CLOSEDPLANAR_XOR``."""

    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSEDPLANAR_XOR)


@dataclass
class HollowSphereKeyhole(_Keyhole, HollowSphere):
    """:class:`HollowSphere` with one keyhole contour per slice through the cavity."""


@dataclass
class HollowCylinderKeyhole(_Keyhole, HollowCylinder):
    """:class:`HollowCylinder` (the straw) with one keyhole contour per slice."""
