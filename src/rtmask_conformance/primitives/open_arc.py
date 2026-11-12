"""``OpenArc2D``: an open circular arc lying on one axial slice (OPEN_PLANAR).

Suite-native companion to the vendored
:class:`rtmask_conformance._vendor.primitives.open_planar.LineSegment2D`. It
follows the same contract: one ContourSequence item of type ``OPEN_PLANAR``
holding every vertex on a single z, no enclosed volume, no surface mesh, and
a closed-form arc length (``r * theta``) published in the spec as
``analytical_arc_length_mm``, the key the curve metrics read.

The vendored module has no shared open-planar base class, so this one
subclasses :class:`AnalyticalPrimitive` directly and reuses the vendored
OPEN_* helpers for slice indexing and the empty inside-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._vendor.common.coords import Geometry
from .._vendor.primitives.base import AnalyticalPrimitive, ContourItem, GeometricType
from .._vendor.primitives.open_nonplanar import (
    _slice_indices_for_polyline,
    _zero_volume_no_inside,
)


@dataclass
class OpenArc2D(AnalyticalPrimitive):
    """A circular arc of ``radius`` mm about ``center_xy`` on the plane ``z = z_mm``.

    The arc starts at ``start_deg`` (measured from +x towards +y) and sweeps
    ``sweep_deg`` counter-clockwise; ``0 < sweep_deg < 360`` keeps it open.
    It is emitted as ``n_samples`` vertices evenly spaced in angle, endpoints
    included.

    Arc length = ``radius * radians(sweep_deg)``. The emitted chords fall
    short of it by about ``(dtheta**2 / 24)`` relative, under 3e-5 for the
    default sampling of a 90 degree arc.
    """

    center_xy: tuple[float, float] = (0.0, 0.0)
    z_mm: float = 0.0
    radius: float = 1.0
    start_deg: float = 0.0
    sweep_deg: float = 90.0
    n_samples: int = 65
    geometric_type: GeometricType = field(init=False, default=GeometricType.OPEN_PLANAR)

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0 (got {self.radius}).")
        if not 0.0 < self.sweep_deg < 360.0:
            raise ValueError(
                f"sweep_deg must be in (0, 360) for an open arc (got {self.sweep_deg})."
            )
        if self.n_samples < 3:
            raise ValueError(f"n_samples must be >= 3 (got {self.n_samples}).")

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> None:
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    def polyline(self) -> np.ndarray:
        """The ``(n_samples, 3)`` vertices, in mm, from the start angle to the end."""
        theta = np.deg2rad(self.start_deg + np.linspace(0.0, self.sweep_deg, self.n_samples))
        cx, cy = self.center_xy
        return np.column_stack([
            cx + self.radius * np.cos(theta),
            cy + self.radius * np.sin(theta),
            np.full(self.n_samples, float(self.z_mm)),
        ])

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        points = self.polyline()
        return [
            ContourItem(
                points_xyz=points,
                geometric_type=GeometricType.OPEN_PLANAR,
                slice_indices=_slice_indices_for_polyline(points, geometry),
            )
        ]

    @property
    def arc_length_mm(self) -> float:
        return float(self.radius * math.radians(self.sweep_deg))

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        d.update(
            {
                "parameters": {
                    "center_xy_mm": list(self.center_xy),
                    "z_mm": float(self.z_mm),
                    "radius_mm": float(self.radius),
                    "start_deg": float(self.start_deg),
                    "sweep_deg": float(self.sweep_deg),
                    "n_samples": int(self.n_samples),
                },
                "analytical_arc_length_mm": self.arc_length_mm,
            }
        )
        return d
