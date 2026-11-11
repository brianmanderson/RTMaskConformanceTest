"""POINT primitives.

DICOM ``ContourGeometricType = POINT`` -- each contour is a single point
(``NumberOfContourPoints = 1``). An ROI may contain many such contours; we emit
one :class:`ContourItem` per point.

Clinical analogues:

* LDR brachytherapy seed positions (canonical use case).
* Fiducial markers (gold seeds for IGRT).
* Surgical clips, isocenters, planning reference points.

Properties:

* Volume = 0 (zero-dimensional).
* No surface mesh.
* "Ground truth" is the list of voxel coordinates each point falls into under each
  tool's quantization rule.

Primitives in this module:

* :class:`SinglePoint`        -- one fiducial.
* :class:`SparsePointCloud`   -- ~10 points along a clinically-shaped curved
  needle track.
* :class:`DensePointCloud`    -- 147 seeds (default pattern) in a TG-186-like LDR-prostate seed
  implant pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore[assignment]

from ..common.coords import Geometry
from .base import AnalyticalPrimitive, ContourItem, GeometricType


def _slice_index_for_z(z_phys: float, geometry: Geometry) -> int:
    """Return the reference-CT slice index whose physical z is closest to ``z_phys``."""
    sz = geometry.spacing[2]
    n_slices = geometry.size[2]
    rel = (z_phys - geometry.origin[2]) / sz
    return max(0, min(n_slices - 1, int(round(rel))))


@dataclass
class _PointSet(AnalyticalPrimitive):
    """Shared base for any POINT primitive: a list of 3D points + per-point ROI emit."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    geometric_type: GeometricType = field(init=False, default=GeometricType.POINT)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"points must have shape (N, 3); got {self.points.shape}."
            )
        if self.points.shape[0] == 0:
            raise ValueError("points must contain at least one (x, y, z) entry.")

    def analytical_volume(self) -> float | None:
        # Points have no volumetric ground truth; signal this with None so the
        # build pipeline skips the (very slow) full-grid partial-volume pass.
        # Tools' POINT outputs are scored against ``points_mm`` in the spec dict
        # rather than against a binary volume mask.
        return None

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        return None

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        # Points have no enclosed volume.
        pts = np.asarray(points_xyz, dtype=np.float64)
        return np.zeros(pts.shape[:-1], dtype=bool)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        contours: list[ContourItem] = []
        for p in self.points:
            z_idx = _slice_index_for_z(float(p[2]), geometry)
            contours.append(
                ContourItem(
                    points_xyz=p.reshape(1, 3),
                    geometric_type=GeometricType.POINT,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d.update(
            {
                "n_points": int(self.points.shape[0]),
                "points_mm": self.points.tolist(),
                "analytical_volume_mm3": None,
            }
        )
        return d


@dataclass
class SinglePoint(_PointSet):
    """A single fiducial point.

    Constructor convenience: takes ``position`` instead of an N=1 ``points`` array.
    """

    position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.position is not None:
            self.points = np.asarray([list(self.position)], dtype=np.float64)
        super().__post_init__()
        if self.points.shape[0] != 1:
            raise ValueError(
                f"SinglePoint must contain exactly 1 point (got {self.points.shape[0]})."
            )


@dataclass
class SparsePointCloud(_PointSet):
    """A small set (~10) of points laid out along a slightly curved needle track.

    The pattern mimics a clinical scenario: a single brachytherapy needle pushed
    in along a roughly-straight trajectory with mild curvature, with seed
    positions spaced at ~5 mm intervals along its length.

    Either pass ``points`` directly, or use the convenience args ``track_start``,
    ``track_end``, ``curvature_amplitude``, ``n_points`` to generate a curved track
    automatically.
    """

    track_start: tuple[float, float, float] | None = None
    track_end: tuple[float, float, float] | None = None
    curvature_amplitude_mm: float = 3.0
    n_points: int = 10

    def __post_init__(self) -> None:
        if self.points.size == 0 and self.track_start is not None and self.track_end is not None:
            self.points = self._sample_curved_track(
                np.asarray(self.track_start, dtype=np.float64),
                np.asarray(self.track_end, dtype=np.float64),
                self.curvature_amplitude_mm,
                self.n_points,
            )
        super().__post_init__()

    @staticmethod
    def _sample_curved_track(
        p1: np.ndarray, p2: np.ndarray, amplitude: float, n: int
    ) -> np.ndarray:
        """Sample ``n`` points along a slightly-curved track from ``p1`` to ``p2``.

        The curvature is a single half-sinusoid bowing the track perpendicular to its
        primary direction. With ``amplitude=0`` the track is exactly straight.
        """
        ts = np.linspace(0.0, 1.0, n)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("track_start and track_end must differ.")
        unit = direction / length
        # Pick a perpendicular bowing direction (any unit vector orthogonal to unit).
        helper = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        perp = np.cross(unit, helper)
        perp /= np.linalg.norm(perp)
        # bow profile: half-sine, peaks at t=0.5
        bow = amplitude * np.sin(np.pi * ts)
        return p1[None, :] + ts[:, None] * direction[None, :] + bow[:, None] * perp[None, :]


@dataclass
class DensePointCloud(_PointSet):
    """A dense seed-implant pattern modelled on TG-186 / LDR-prostate cases
    (147 seeds at the defaults below).

    The default pattern lays out seeds in roughly axial planes spaced at ~5 mm,
    each plane filled with a hexagonal close-packed seed grid bounded by an
    ellipse approximating the prostate cross-section. Pass ``points`` directly to
    override.
    """

    centroid: tuple[float, float, float] | None = None
    semi_axes_xy: tuple[float, float] = (15.0, 10.0)
    plane_spacing_mm: float = 5.0
    n_planes: int = 7
    seed_spacing_mm: float = 5.0

    def __post_init__(self) -> None:
        if self.points.size == 0 and self.centroid is not None:
            self.points = self._build_default_pattern()
        super().__post_init__()

    def _build_default_pattern(self) -> np.ndarray:
        cx, cy, cz = self.centroid  # type: ignore[misc]
        a, b = self.semi_axes_xy
        plane_zs = cz + (np.arange(self.n_planes) - (self.n_planes - 1) / 2.0) * self.plane_spacing_mm
        seeds: list[list[float]] = []
        # Hexagonal close-packing within an ellipse.
        dx = self.seed_spacing_mm
        dy = self.seed_spacing_mm * math.sqrt(3.0) / 2.0
        for plane_idx, z in enumerate(plane_zs):
            row_idx = 0
            y = cy - b
            while y <= cy + b + 1e-6:
                x_offset = dx / 2.0 if row_idx % 2 == 1 else 0.0
                x = cx - a + x_offset
                while x <= cx + a + 1e-6:
                    if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0:
                        seeds.append([x, y, float(z)])
                    x += dx
                y += dy
                row_idx += 1
        return np.array(seeds, dtype=np.float64) if seeds else np.zeros((0, 3))
