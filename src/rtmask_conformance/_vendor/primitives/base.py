"""Abstract base class for analytical primitives (vendored from rtmask_validation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore[assignment]

from ..common.coords import Geometry


class GeometricType(str, Enum):
    """The five DICOM ContourGeometricType values plus the XOR variant."""

    CLOSED_PLANAR = "CLOSED_PLANAR"
    CLOSED_PLANAR_XOR = "CLOSED_PLANAR_XOR"
    OPEN_PLANAR = "OPEN_PLANAR"
    OPEN_NONPLANAR = "OPEN_NONPLANAR"
    CLOSED_NONPLANAR = "CLOSED_NONPLANAR"
    POINT = "POINT"


@dataclass
class ContourItem:
    points_xyz: np.ndarray
    geometric_type: GeometricType
    slice_indices: list[int] = field(default_factory=list)


@dataclass
class AnalyticalPrimitive(ABC):
    name: str
    geometric_type: GeometricType = field(init=False)

    @abstractmethod
    def analytical_volume(self) -> float | None:
        ...

    @abstractmethod
    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        ...

    @abstractmethod
    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        ...

    def to_spec_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "geometric_type": self.geometric_type.value,
            "analytical_volume_mm3": self.analytical_volume(),
        }
