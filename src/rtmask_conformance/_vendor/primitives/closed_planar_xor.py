"""Hollow primitives via multi-contour CLOSED_PLANAR (vendored from rtmask_validation)."""

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
from .closed_planar import _slice_z_coords

_CIRCLE_ANGLES = 64


@dataclass
class HollowSphere(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outer_radius: float = 2.0
    inner_radius: float = 1.0
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if self.outer_radius <= 0:
            raise ValueError(
                f"HollowSphere outer_radius must be positive (got {self.outer_radius})."
            )
        if self.inner_radius <= 0:
            raise ValueError(
                f"HollowSphere inner_radius must be positive (got {self.inner_radius})."
            )
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                "HollowSphere inner_radius must be smaller than outer_radius "
                f"(got inner={self.inner_radius}, outer={self.outer_radius})."
            )

    def analytical_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * (self.outer_radius**3 - self.inner_radius**3)

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=self.outer_radius)
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        dx = pts[..., 0] - cx
        dy = pts[..., 1] - cy
        dz = pts[..., 2] - cz
        d2 = dx * dx + dy * dy + dz * dz
        return (d2 <= self.outer_radius**2) & (d2 > self.inner_radius**2)

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        R = self.outer_radius
        r = self.inner_radius
        slice_zs = _slice_z_coords(geometry)
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        min_radius_mm = float(min(geometry.spacing[0], geometry.spacing[1]))
        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            dz = z - cz
            if abs(dz) >= R:
                continue
            outer_r = math.sqrt(R * R - dz * dz)
            if outer_r < min_radius_mm:
                continue
            zs = np.full_like(cos_t, z)
            outer = np.stack([cx + outer_r * cos_t, cy + outer_r * sin_t, zs], axis=1)
            contours.append(
                ContourItem(
                    points_xyz=outer,
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
            if abs(dz) < r:
                inner_r = math.sqrt(r * r - dz * dz)
                if inner_r >= min_radius_mm:
                    inner = np.stack(
                        [cx + inner_r * cos_t, cy + inner_r * sin_t, zs], axis=1
                    )
                    contours.append(
                        ContourItem(
                            points_xyz=inner,
                            geometric_type=GeometricType.CLOSED_PLANAR,
                            slice_indices=[z_idx],
                        )
                    )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        outer_area = 4.0 * math.pi * self.outer_radius**2
        inner_area = 4.0 * math.pi * self.inner_radius**2
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "outer_radius_mm": float(self.outer_radius),
                    "inner_radius_mm": float(self.inner_radius),
                },
                "analytical_outer_surface_area_mm2": outer_area,
                "analytical_inner_surface_area_mm2": inner_area,
                "encoding": "multi_contour_closed_planar_xor",
            }
        )
        return d


@dataclass
class HollowCylinder(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outer_radius: float = 2.0
    inner_radius: float = 1.0
    height: float = 1.0
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if self.outer_radius <= 0:
            raise ValueError(
                f"HollowCylinder outer_radius must be positive (got {self.outer_radius})."
            )
        if self.inner_radius <= 0:
            raise ValueError(
                f"HollowCylinder inner_radius must be positive (got {self.inner_radius})."
            )
        if self.height <= 0:
            raise ValueError(f"HollowCylinder height must be positive (got {self.height}).")
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                "HollowCylinder inner_radius must be smaller than outer_radius "
                f"(got inner={self.inner_radius}, outer={self.outer_radius})."
            )

    def analytical_volume(self) -> float:
        return math.pi * (self.outer_radius**2 - self.inner_radius**2) * self.height

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.cylinder(
            radius=self.outer_radius, height=self.height, sections=64
        )
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        dx = pts[..., 0] - cx
        dy = pts[..., 1] - cy
        dz = pts[..., 2] - cz
        rho2 = dx * dx + dy * dy
        return (
            (rho2 <= self.outer_radius**2)
            & (rho2 > self.inner_radius**2)
            & (np.abs(dz) <= self.height / 2.0)
        )

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        slice_zs = _slice_z_coords(geometry)
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        half_h = self.height / 2.0
        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            if abs(z - cz) > half_h:
                continue
            zs = np.full_like(cos_t, z)
            outer = np.stack(
                [cx + self.outer_radius * cos_t, cy + self.outer_radius * sin_t, zs],
                axis=1,
            )
            inner = np.stack(
                [cx + self.inner_radius * cos_t, cy + self.inner_radius * sin_t, zs],
                axis=1,
            )
            contours.append(
                ContourItem(
                    points_xyz=outer,
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
            contours.append(
                ContourItem(
                    points_xyz=inner,
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        outer_lateral = 2.0 * math.pi * self.outer_radius * self.height
        inner_lateral = 2.0 * math.pi * self.inner_radius * self.height
        annulus_caps = 2.0 * math.pi * (self.outer_radius**2 - self.inner_radius**2)
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "outer_radius_mm": float(self.outer_radius),
                    "inner_radius_mm": float(self.inner_radius),
                    "height_mm": float(self.height),
                },
                "analytical_outer_lateral_area_mm2": outer_lateral,
                "analytical_inner_lateral_area_mm2": inner_lateral,
                "analytical_annulus_cap_area_mm2": annulus_caps,
                "encoding": "multi_contour_closed_planar_xor",
            }
        )
        return d
