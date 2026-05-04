"""CLOSED_PLANAR primitives (vendored from rtmask_validation)."""

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

_CIRCLE_ANGLES = 64


def _slice_z_coords(geometry: Geometry) -> np.ndarray:
    sz = geometry.spacing[2]
    n_slices = geometry.size[2]
    return np.array(
        [geometry.origin[2] + i * sz for i in range(n_slices)], dtype=np.float64
    )


@dataclass
class Sphere(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Sphere radius must be positive (got {self.radius}).")

    def analytical_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.radius**3

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=self.radius)
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        dx = pts[..., 0] - cx
        dy = pts[..., 1] - cy
        dz = pts[..., 2] - cz
        return (dx * dx + dy * dy + dz * dz) <= self.radius * self.radius

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        r = self.radius
        slice_zs = _slice_z_coords(geometry)
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        min_radius_mm = float(min(geometry.spacing[0], geometry.spacing[1]))

        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            dz = z - cz
            if abs(dz) >= r:
                continue
            slice_radius = math.sqrt(r * r - dz * dz)
            if slice_radius < min_radius_mm:
                continue
            xs = cx + slice_radius * cos_t
            ys = cy + slice_radius * sin_t
            zs = np.full_like(xs, z)
            polygon = np.stack([xs, ys, zs], axis=1)
            contours.append(
                ContourItem(
                    points_xyz=polygon,
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "radius_mm": float(self.radius),
                },
                "analytical_surface_area_mm2": 4.0 * math.pi * self.radius**2,
            }
        )
        return d


@dataclass
class Box(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.size):
            raise ValueError(f"Box dimensions must be positive (got {self.size}).")

    def analytical_volume(self) -> float:
        wx, wy, wz = self.size
        return float(wx * wy * wz)

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.box(extents=self.size)
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        wx, wy, wz = self.size
        hx, hy, hz = wx / 2.0, wy / 2.0, wz / 2.0
        inside_x = np.abs(pts[..., 0] - cx) <= hx
        inside_y = np.abs(pts[..., 1] - cy) <= hy
        inside_z = np.abs(pts[..., 2] - cz) <= hz
        return inside_x & inside_y & inside_z

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        wx, wy, wz = self.size
        hx, hy, hz = wx / 2.0, wy / 2.0, wz / 2.0
        slice_zs = _slice_z_coords(geometry)
        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            if abs(z - cz) > hz:
                continue
            corners = np.array(
                [
                    [cx - hx, cy - hy, z],
                    [cx + hx, cy - hy, z],
                    [cx + hx, cy + hy, z],
                    [cx - hx, cy + hy, z],
                ],
                dtype=np.float64,
            )
            contours.append(
                ContourItem(
                    points_xyz=corners,
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        wx, wy, wz = self.size
        surface_area = 2.0 * (wx * wy + wy * wz + wx * wz)
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "size_mm": list(self.size),
                },
                "analytical_surface_area_mm2": surface_area,
            }
        )
        return d


def Cube(name: str, center: tuple[float, float, float], side_length: float) -> Box:
    return Box(name=name, center=center, size=(side_length, side_length, side_length))


@dataclass
class Cylinder(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    height: float = 1.0
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Cylinder radius must be positive (got {self.radius}).")
        if self.height <= 0:
            raise ValueError(f"Cylinder height must be positive (got {self.height}).")

    def analytical_volume(self) -> float:
        return math.pi * self.radius**2 * self.height

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.height, sections=64)
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        dx = pts[..., 0] - cx
        dy = pts[..., 1] - cy
        dz = pts[..., 2] - cz
        return (dx * dx + dy * dy <= self.radius * self.radius) & (
            np.abs(dz) <= self.height / 2.0
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
            xs = cx + self.radius * cos_t
            ys = cy + self.radius * sin_t
            zs = np.full_like(xs, z)
            contours.append(
                ContourItem(
                    points_xyz=np.stack([xs, ys, zs], axis=1),
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        lateral = 2.0 * math.pi * self.radius * self.height
        caps = 2.0 * math.pi * self.radius**2
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "radius_mm": float(self.radius),
                    "height_mm": float(self.height),
                },
                "analytical_surface_area_mm2": lateral + caps,
            }
        )
        return d


@dataclass
class Ellipsoid(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.semi_axes):
            raise ValueError(f"Ellipsoid semi-axes must be positive (got {self.semi_axes}).")

    def analytical_volume(self) -> float:
        a, b, c = self.semi_axes
        return (4.0 / 3.0) * math.pi * a * b * c

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        a, b, c = self.semi_axes
        mesh.apply_scale([a, b, c])
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        a, b, c = self.semi_axes
        dx = (pts[..., 0] - cx) / a
        dy = (pts[..., 1] - cy) / b
        dz = (pts[..., 2] - cz) / c
        return dx * dx + dy * dy + dz * dz <= 1.0

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        a, b, c = self.semi_axes
        slice_zs = _slice_z_coords(geometry)
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        min_radius_mm = float(min(geometry.spacing[0], geometry.spacing[1]))
        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            t = (z - cz) / c
            if abs(t) >= 1.0:
                continue
            scale = math.sqrt(1.0 - t * t)
            slice_a = a * scale
            slice_b = b * scale
            if max(slice_a, slice_b) < min_radius_mm:
                continue
            xs = cx + slice_a * cos_t
            ys = cy + slice_b * sin_t
            zs = np.full_like(xs, z)
            contours.append(
                ContourItem(
                    points_xyz=np.stack([xs, ys, zs], axis=1),
                    geometric_type=GeometricType.CLOSED_PLANAR,
                    slice_indices=[z_idx],
                )
            )
        return contours

    def to_spec_dict(self) -> dict:
        d = super().to_spec_dict()
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "semi_axes_mm": list(self.semi_axes),
                },
            }
        )
        return d


@dataclass
class Torus(AnalyticalPrimitive):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    major_radius: float = 1.0
    minor_radius: float = 0.5
    geometric_type: GeometricType = field(init=False, default=GeometricType.CLOSED_PLANAR)

    def __post_init__(self) -> None:
        if self.major_radius <= 0:
            raise ValueError(f"Torus major_radius must be positive (got {self.major_radius}).")
        if self.minor_radius <= 0:
            raise ValueError(f"Torus minor_radius must be positive (got {self.minor_radius}).")
        if self.minor_radius >= self.major_radius:
            raise ValueError(
                "Torus minor_radius must be smaller than major_radius "
                f"(got minor={self.minor_radius}, major={self.major_radius})."
            )

    def analytical_volume(self) -> float:
        return 2.0 * math.pi**2 * self.major_radius * self.minor_radius**2

    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        if trimesh is None:
            return None
        mesh = trimesh.creation.torus(
            major_radius=self.major_radius,
            minor_radius=self.minor_radius,
            major_sections=64,
            minor_sections=32,
        )
        mesh.apply_translation(self.center)
        return mesh

    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        cx, cy, cz = self.center
        dx = pts[..., 0] - cx
        dy = pts[..., 1] - cy
        dz = pts[..., 2] - cz
        rho = np.sqrt(dx * dx + dy * dy)
        return (rho - self.major_radius) ** 2 + dz * dz <= self.minor_radius**2

    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        cx, cy, cz = self.center
        slice_zs = _slice_z_coords(geometry)
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        min_radius_mm = float(min(geometry.spacing[0], geometry.spacing[1]))
        contours: list[ContourItem] = []
        for z_idx, z in enumerate(slice_zs):
            dz = z - cz
            if abs(dz) >= self.minor_radius:
                continue
            offset = math.sqrt(self.minor_radius**2 - dz * dz)
            outer_r = self.major_radius + offset
            inner_r = self.major_radius - offset
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
            if inner_r >= min_radius_mm:
                inner = np.stack([cx + inner_r * cos_t, cy + inner_r * sin_t, zs], axis=1)
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
        surface_area = 4.0 * math.pi**2 * self.major_radius * self.minor_radius
        d.update(
            {
                "parameters": {
                    "center_mm": list(self.center),
                    "major_radius_mm": float(self.major_radius),
                    "minor_radius_mm": float(self.minor_radius),
                },
                "analytical_surface_area_mm2": surface_area,
            }
        )
        return d
