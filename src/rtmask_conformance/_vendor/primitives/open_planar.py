"""OPEN_PLANAR primitives.

DICOM ``ContourGeometricType = OPEN_PLANAR`` -- an open polyline whose vertices
are all coplanar (single z). Last vertex does **not** connect back to the
first. Less common clinically than CLOSED_PLANAR or OPEN_NONPLANAR but
explicitly defined in the standard. Used for single-slice fiducial line
segments, partial outlines, and atlas-construction reference geometry.

Each primitive emits **one** ContourSequence item containing all polyline
vertices on a single z slice. ``slice_indices`` is a single-element list.

Properties:

* Volume = 0 (1D curve in 2D plane embedded in 3D space).
* No surface mesh.
* Has well-defined arc length (closed-form for a straight segment).

Primitives in this module:

* :class:`LineSegment2D` -- straight segment between two coplanar endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore[assignment]

from ..common.coords import Geometry
from .base import AnalyticalPrimitive, ContourItem, GeometricType
from .open_nonplanar import _slice_indices_for_polyline, _zero_volume_no_inside


@dataclass
class LineSegment2D(AnalyticalPrimitive):
    """A straight 2D line segment lying on a single z slice.

    Endpoints are ``(x, y)`` coordinates with a shared ``z_mm``. Densified to
    ``n_samples`` evenly-spaced points along the segment, matching the
    DiagonalNeedle (OPEN_NONPLANAR) sampling convention so the curve-metric
    skeletonization picks up a single connected component.

    Arc length = ``||p2_xy - p1_xy||``.
    """

    p1_xy: tuple[float, float] = (0.0, 0.0)
    p2_xy: tuple[float, float] = (1.0, 0.0)
    z_mm: float = 0.0
    n_samples: int = 32
    geometric_type: GeometricType = field(
        init=False, default=GeometricType.OPEN_PLANAR
    )

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2 (got {self.n_samples}).")
        if np.allclose(self.p1_xy, self.p2_xy):
            raise ValueError("p1_xy and p2_xy must differ.")

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        ts = np.linspace(0.0, 1.0, self.n_samples)
        p1 = np.asarray([self.p1_xy[0], self.p1_xy[1], self.z_mm], dtype=np.float64)
        p2 = np.asarray([self.p2_xy[0], self.p2_xy[1], self.z_mm], dtype=np.float64)
        polyline = p1 + (p2 - p1) * ts[:, None]
        return [
            ContourItem(
                points_xyz=polyline,
                geometric_type=GeometricType.OPEN_PLANAR,
                slice_indices=_slice_indices_for_polyline(polyline, geometry),
            )
        ]

    @property
    def arc_length_mm(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p2_xy, self.p1_xy)))

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        d.update(
            {
                "parameters": {
                    "p1_xy_mm": list(self.p1_xy),
                    "p2_xy_mm": list(self.p2_xy),
                    "z_mm": float(self.z_mm),
                    "n_samples": int(self.n_samples),
                },
                "analytical_arc_length_mm": self.arc_length_mm,
            }
        )
        return d
