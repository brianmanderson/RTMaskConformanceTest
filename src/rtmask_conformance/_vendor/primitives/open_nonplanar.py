"""OPEN_NONPLANAR primitives.

DICOM ``ContourGeometricType = OPEN_NONPLANAR`` -- a single open polyline whose
vertices span multiple z-slices. Last vertex does **not** connect back to the first.
The DICOM standard explicitly cites brachytherapy applicators (catheters, needles)
as the canonical use case.

Each primitive emits **one** ContourSequence item containing all of its sampled
vertices. ``slice_indices`` lists every reference-CT slice the polyline passes
through so ContourImageSequence references them all.

Properties:

* Volume = 0 (1D curve in 3D space).
* No surface mesh.
* Has well-defined arc length: closed-form for line segment and helix, numerical
  for arbitrary parametric paths.

Primitives in this module:

* :class:`DiagonalNeedle`        -- straight line between two 3D endpoints.
* :class:`Helix`                 -- helix around a vertical axis.
* :class:`Sinusoidal3DPath`      -- wavy path in (x, y) as z advances.
* :class:`MultiNeedleBundle`     -- N parallel diagonal needles in a regular pattern;
  emits N separate ContourItem entries (each a single open polyline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore[assignment]

from ..common.coords import Geometry
from .base import AnalyticalPrimitive, ContourItem, GeometricType


def _slice_indices_for_polyline(points: np.ndarray, geometry: Geometry) -> list[int]:
    """Compute the unique sorted reference-CT slice indices a polyline passes through."""
    sz = geometry.spacing[2]
    n_slices = geometry.size[2]
    rel_z = (points[:, 2] - geometry.origin[2]) / sz
    rounded = np.round(rel_z).astype(int)
    rounded = np.clip(rounded, 0, n_slices - 1)
    return sorted(set(int(i) for i in rounded))


def _zero_volume_no_inside(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    return np.zeros(pts.shape[:-1], dtype=bool)


@dataclass
class DiagonalNeedle(AnalyticalPrimitive):
    """A straight 3D line from ``p1`` to ``p2``, sampled at ``n_samples`` points
    (default 32).

    Arc length = ``||p2 - p1||``. Models a clinical brachytherapy needle inserted
    diagonally through tissue.
    """

    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_samples: int = 32
    geometric_type: GeometricType = field(init=False, default=GeometricType.OPEN_NONPLANAR)

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2 (got {self.n_samples}).")
        if np.allclose(self.p1, self.p2):
            raise ValueError("p1 and p2 must differ.")

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        ts = np.linspace(0.0, 1.0, self.n_samples)
        p1 = np.asarray(self.p1, dtype=np.float64)
        p2 = np.asarray(self.p2, dtype=np.float64)
        polyline = p1 + (p2 - p1) * ts[:, None]
        return [
            ContourItem(
                points_xyz=polyline,
                geometric_type=GeometricType.OPEN_NONPLANAR,
                slice_indices=_slice_indices_for_polyline(polyline, geometry),
            )
        ]

    @property
    def arc_length_mm(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p2, self.p1)))

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        d.update(
            {
                "parameters": {
                    "p1_mm": list(self.p1),
                    "p2_mm": list(self.p2),
                    "n_samples": int(self.n_samples),
                },
                "analytical_arc_length_mm": self.arc_length_mm,
            }
        )
        return d


@dataclass
class Helix(AnalyticalPrimitive):
    """A right-handed helix around a vertical axis.

    Parametric form (with axis through ``axis_center``)::

        x(t) = axis_center.x + radius * cos(2 pi t / pitch)
        y(t) = axis_center.y + radius * sin(2 pi t / pitch)
        z(t) = axis_center.z - height/2 + t

    where ``t`` runs from 0 to ``height``. Closed-form arc length per turn is
    ``sqrt((2 pi r)^2 + p^2)`` where ``p`` is the pitch (z-rise per turn); total
    arc length scales linearly with ``n_turns``.

    The helix wraps clockwise as you look down the +z axis when ``handedness=+1``.
    """

    axis_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 5.0
    pitch_mm_per_turn: float = 5.0
    n_turns: float = 4.0
    handedness: int = 1
    n_samples_per_turn: int = 64
    geometric_type: GeometricType = field(init=False, default=GeometricType.OPEN_NONPLANAR)

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive (got {self.radius}).")
        if self.pitch_mm_per_turn <= 0:
            raise ValueError(f"pitch_mm_per_turn must be positive (got {self.pitch_mm_per_turn}).")
        if self.n_turns <= 0:
            raise ValueError(f"n_turns must be positive (got {self.n_turns}).")
        if self.handedness not in (-1, 1):
            raise ValueError(f"handedness must be +1 or -1 (got {self.handedness}).")

    @property
    def height_mm(self) -> float:
        return self.pitch_mm_per_turn * self.n_turns

    @property
    def arc_length_mm(self) -> float:
        return self.n_turns * math.sqrt((2.0 * math.pi * self.radius) ** 2 + self.pitch_mm_per_turn**2)

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.axis_center
        n_total = max(2, int(round(self.n_turns * self.n_samples_per_turn)))
        ts = np.linspace(0.0, self.n_turns, n_total)
        thetas = self.handedness * 2.0 * math.pi * ts
        xs = cx + self.radius * np.cos(thetas)
        ys = cy + self.radius * np.sin(thetas)
        zs = cz - self.height_mm / 2.0 + self.pitch_mm_per_turn * ts
        polyline = np.stack([xs, ys, zs], axis=1)
        return [
            ContourItem(
                points_xyz=polyline,
                geometric_type=GeometricType.OPEN_NONPLANAR,
                slice_indices=_slice_indices_for_polyline(polyline, geometry),
            )
        ]

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        d.update(
            {
                "parameters": {
                    "axis_center_mm": list(self.axis_center),
                    "radius_mm": float(self.radius),
                    "pitch_mm_per_turn": float(self.pitch_mm_per_turn),
                    "n_turns": float(self.n_turns),
                    "handedness": int(self.handedness),
                    "n_samples_per_turn": int(self.n_samples_per_turn),
                },
                "analytical_height_mm": self.height_mm,
                "analytical_arc_length_mm": self.arc_length_mm,
            }
        )
        return d


@dataclass
class Sinusoidal3DPath(AnalyticalPrimitive):
    """A wavy path in (x, y) as z advances linearly.

    Parametrically, with ``z`` running from ``z_start`` to ``z_end``::

        x(z) = base_xy.x + amplitude_x * sin(2 pi (z - z_start) / wavelength_z)
        y(z) = base_xy.y + amplitude_y * cos(2 pi (z - z_start) / wavelength_z)

    Arc length is computed numerically via the discrete polyline (sum of segment
    lengths). The continuous-curve length is also reported in the spec dict for
    reference, integrated with high-order Simpson quadrature.
    """

    base_xy: tuple[float, float] = (0.0, 0.0)
    z_start: float = 0.0
    z_end: float = 50.0
    amplitude_x: float = 5.0
    amplitude_y: float = 5.0
    wavelength_z: float = 25.0
    n_samples: int = 80
    geometric_type: GeometricType = field(init=False, default=GeometricType.OPEN_NONPLANAR)

    def __post_init__(self) -> None:
        if self.z_end <= self.z_start:
            raise ValueError(
                f"z_end must be greater than z_start (got {self.z_start} -> {self.z_end})."
            )
        if self.wavelength_z <= 0:
            raise ValueError(f"wavelength_z must be positive (got {self.wavelength_z}).")
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2 (got {self.n_samples}).")

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    def _sample_curve(self, n: int) -> np.ndarray:
        zs = np.linspace(self.z_start, self.z_end, n)
        bx, by = self.base_xy
        u = 2.0 * math.pi * (zs - self.z_start) / self.wavelength_z
        xs = bx + self.amplitude_x * np.sin(u)
        ys = by + self.amplitude_y * np.cos(u)
        return np.stack([xs, ys, zs], axis=1)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        polyline = self._sample_curve(self.n_samples)
        return [
            ContourItem(
                points_xyz=polyline,
                geometric_type=GeometricType.OPEN_NONPLANAR,
                slice_indices=_slice_indices_for_polyline(polyline, geometry),
            )
        ]

    def _continuous_arc_length(self, n_quad: int = 1024) -> float:
        # Simpson's-rule integration of |dr/dz| dz.
        zs = np.linspace(self.z_start, self.z_end, n_quad)
        u = 2.0 * math.pi * (zs - self.z_start) / self.wavelength_z
        ux = 2.0 * math.pi / self.wavelength_z
        dxdz = self.amplitude_x * np.cos(u) * ux
        dydz = -self.amplitude_y * np.sin(u) * ux
        speed = np.sqrt(1.0 + dxdz**2 + dydz**2)
        return float(np.trapezoid(speed, zs))

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        polyline = self._sample_curve(self.n_samples)
        seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        d.update(
            {
                "parameters": {
                    "base_xy_mm": list(self.base_xy),
                    "z_start_mm": float(self.z_start),
                    "z_end_mm": float(self.z_end),
                    "amplitude_x_mm": float(self.amplitude_x),
                    "amplitude_y_mm": float(self.amplitude_y),
                    "wavelength_z_mm": float(self.wavelength_z),
                    "n_samples": int(self.n_samples),
                },
                "polyline_arc_length_mm": float(seg_lengths.sum()),
                "continuous_arc_length_mm": self._continuous_arc_length(),
            }
        )
        return d


@dataclass
class MultiNeedleBundle(AnalyticalPrimitive):
    """A bundle of N parallel needles arranged in a regular pattern.

    Models a Syed-Neblett-style gynecologic interstitial implant: several parallel
    needles inserted through a template grid, each at a fixed (x, y) offset from a
    common bundle center, all with the same axis direction.

    Emits one :class:`ContourItem` per needle (i.e., N OPEN_NONPLANAR contours
    inside one ROI).
    """

    bundle_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    needle_length_mm: float = 60.0
    pattern_offsets_xy: Sequence[tuple[float, float]] = field(
        default_factory=lambda: ((-5.0, 0.0), (0.0, 0.0), (5.0, 0.0))
    )
    n_samples_per_needle: int = 24
    geometric_type: GeometricType = field(init=False, default=GeometricType.OPEN_NONPLANAR)

    def __post_init__(self) -> None:
        if self.needle_length_mm <= 0:
            raise ValueError(f"needle_length_mm must be positive (got {self.needle_length_mm}).")
        if len(self.pattern_offsets_xy) == 0:
            raise ValueError("pattern_offsets_xy must contain at least one (dx, dy) entry.")
        d = np.asarray(self.direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm == 0:
            raise ValueError("direction must be a non-zero vector.")
        self._unit = d / norm
        if self.n_samples_per_needle < 2:
            raise ValueError("n_samples_per_needle must be >= 2.")

    def analytical_volume(self) -> float | None:
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        return _zero_volume_no_inside(points_xyz)

    @property
    def arc_length_per_needle_mm(self) -> float:
        return float(self.needle_length_mm)

    @property
    def total_arc_length_mm(self) -> float:
        return self.arc_length_per_needle_mm * len(self.pattern_offsets_xy)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.bundle_center
        unit = self._unit
        ts = np.linspace(-self.needle_length_mm / 2.0, self.needle_length_mm / 2.0, self.n_samples_per_needle)
        contours: list[ContourItem] = []
        for dx, dy in self.pattern_offsets_xy:
            anchor = np.array([cx + dx, cy + dy, cz], dtype=np.float64)
            polyline = anchor[None, :] + ts[:, None] * unit[None, :]
            contours.append(
                ContourItem(
                    points_xyz=polyline,
                    geometric_type=GeometricType.OPEN_NONPLANAR,
                    slice_indices=_slice_indices_for_polyline(polyline, geometry),
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d["analytical_volume_mm3"] = None
        d.update(
            {
                "parameters": {
                    "bundle_center_mm": list(self.bundle_center),
                    "direction": list(self.direction),
                    "needle_length_mm": float(self.needle_length_mm),
                    "pattern_offsets_xy_mm": [list(o) for o in self.pattern_offsets_xy],
                    "n_samples_per_needle": int(self.n_samples_per_needle),
                },
                "n_needles": len(self.pattern_offsets_xy),
                "analytical_arc_length_per_needle_mm": self.arc_length_per_needle_mm,
                "analytical_total_arc_length_mm": self.total_arc_length_mm,
            }
        )
        return d
