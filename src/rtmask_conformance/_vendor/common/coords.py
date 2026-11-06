"""Physical <-> voxel coordinate transforms.

The Verification Methodology assumes a standard DICOM patient coordinate system (LPS),
identity image orientation, and origin at (0, 0, 0). For the v0.1 reference CT we keep
that simple geometry and provide thin helpers so primitive code does not need to fish
into SimpleITK image handles.

Adding oblique image orientations later is a localized change here and in
:mod:`rtmask_validation.refimage.build_reference_ct`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Reference-image geometry: enough to convert between physical (mm) and
    continuous voxel index coordinates without depending on SimpleITK.

    Attributes
    ----------
    origin : tuple[float, float, float]
        Physical (mm) coordinates of voxel index (0, 0, 0).
    spacing : tuple[float, float, float]
        Voxel size in mm along (x, y, z).
    size : tuple[int, int, int]
        Number of voxels along (x, y, z), i.e. (cols, rows, slices).
    direction : tuple[float, ...] = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        3x3 row-major direction cosines. Identity by default.
    """

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    size: tuple[int, int, int]
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # ---- forward & reverse transforms ----

    def index_to_physical(self, idx_xyz: np.ndarray) -> np.ndarray:
        """Continuous voxel index (x, y, z) -> physical (mm).

        Accepts a single index ``(3,)`` or a batch ``(N, 3)``. Returns the same shape.
        """
        idx = np.asarray(idx_xyz, dtype=np.float64)
        d = np.asarray(self.direction, dtype=np.float64).reshape(3, 3)
        spacing = np.asarray(self.spacing, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        return origin + (idx * spacing) @ d.T

    def physical_to_index(self, phys_xyz: np.ndarray) -> np.ndarray:
        """Physical (mm) -> continuous voxel index (x, y, z)."""
        phys = np.asarray(phys_xyz, dtype=np.float64)
        d = np.asarray(self.direction, dtype=np.float64).reshape(3, 3)
        spacing = np.asarray(self.spacing, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        return ((phys - origin) @ np.linalg.inv(d.T)) / spacing

    # ---- convenience ----

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx * sy * sz)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Numpy-style shape: (slices, rows, cols)."""
        cols, rows, slices = self.size
        return slices, rows, cols
