"""Abstract base class for analytical primitives.

Every primitive (Sphere, Helix, TiltedClosedLoop, ...) subclasses
:class:`AnalyticalPrimitive` and implements:

* :meth:`analytical_volume` -- exact closed-form volume in mm^3 (or ``None`` if the
  geometric type does not have an enclosed volume; e.g. OPEN_PLANAR / OPEN_NONPLANAR /
  POINT / CLOSED_NONPLANAR under the surface interpretation).
* :meth:`analytical_surface_mesh` -- a watertight ``trimesh.Trimesh`` for surface DSC /
  Hausdorff metrics, or ``None`` for non-surface primitives.
* :meth:`is_inside` -- vectorized point-in-volume predicate, used by the partial-volume
  rasterizer to produce ground-truth masks. For non-volume primitives this should
  return an all-False array (the ground-truth mask is then either empty -- POINT --
  or built by a different mechanism -- arc-length sampling for the OPEN_*/NONPLANAR
  curves; that lives in ``rtmask_validation.groundtruth``).
* :meth:`get_contours` -- produce the list of :class:`ContourItem` records that
  the writer turns into ContourSequence items in the RTSTRUCT. This is the unified
  emission API used by *every* primitive, regardless of geometric type:

  - **CLOSED_PLANAR** primitives return one ContourItem per slice they intersect.
  - **POINT** primitives return one ContourItem per point (NumberOfContourPoints = 1).
  - **OPEN_NONPLANAR / CLOSED_NONPLANAR** primitives return one ContourItem holding
    the entire 3D polyline; ``slice_indices`` lists every slice the polyline passes
    through so ContourImageSequence references them all.
* :meth:`to_spec_dict` -- JSON-serializable record of parameters + analytical truth,
  for the per-primitive ``spec.json`` artifact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover -- only used for surface metrics
    trimesh = None  # type: ignore[assignment]

from ..common.coords import Geometry


class GeometricType(str, Enum):
    """Contour Geometric Type (3006,0042) values a primitive can carry.

    Five are the standard's Enumerated Values (PS3.3 C.8.8.6): CLOSED_PLANAR,
    CLOSEDPLANAR_XOR, OPEN_PLANAR, OPEN_NONPLANAR and POINT. The XOR value has no
    underscore between CLOSED and PLANAR (16 characters, the CS maximum).

    ``CLOSED_NONPLANAR`` is **not** a DICOM value, and an RTSTRUCT carrying it is
    invalid. It stays so the off-registry probes in ``closed_nonplanar.py`` can still
    be built to see how tools handle it.
    """

    CLOSED_PLANAR = "CLOSED_PLANAR"
    CLOSEDPLANAR_XOR = "CLOSEDPLANAR_XOR"
    OPEN_PLANAR = "OPEN_PLANAR"
    OPEN_NONPLANAR = "OPEN_NONPLANAR"
    CLOSED_NONPLANAR = "CLOSED_NONPLANAR"  # non-standard; see the class docstring
    POINT = "POINT"

    @classmethod
    def _missing_(cls, value):
        # Parsing tolerance only: this package spelled the XOR value
        # CLOSED_PLANAR_XOR until 2026-11, and readers such as dcmrtstruct2nii
        # accept that spelling too. Writers never emit it, because the value of
        # CLOSEDPLANAR_XOR is the standard spelling.
        if isinstance(value, str) and value.strip().upper() == "CLOSED_PLANAR_XOR":
            return cls.CLOSEDPLANAR_XOR
        return None


@dataclass
class ContourItem:
    """One ContourSequence item to be emitted into the RTSTRUCT.

    Attributes
    ----------
    points_xyz : np.ndarray
        ``(N, 3)`` array of physical (mm) coordinates. ``N == 1`` for POINT,
        ``N >= 3`` for CLOSED_*, ``N >= 2`` for OPEN_*.
    geometric_type : GeometricType
        Sets the ``ContourGeometricType`` DICOM tag on this ContourSequence item.
    slice_indices : list[int]
        Z-indices of every reference-image slice this contour touches. The writer
        emits a ContourImageSequence with one item per index. Empty list -> omit
        ContourImageSequence (DICOM-permitted for OPEN_NONPLANAR, and used for the
        non-standard CLOSED_NONPLANAR too, but most TPSes prefer at least one entry).
    """

    points_xyz: np.ndarray
    geometric_type: GeometricType
    slice_indices: list[int] = field(default_factory=list)


@dataclass
class AnalyticalPrimitive(ABC):
    """Common base for every analytical primitive.

    Subclasses add their own parameters as additional dataclass fields and override
    the abstract methods.
    """

    name: str
    geometric_type: GeometricType = field(init=False)

    # ---- abstract methods ----

    @abstractmethod
    def analytical_volume(self) -> float | None:
        """Closed-form volume in mm^3, or ``None`` if not applicable."""

    @abstractmethod
    def analytical_surface_mesh(self) -> "trimesh.Trimesh | None":
        """Watertight surface mesh for DSC/Hausdorff, or ``None``."""

    @abstractmethod
    def is_inside(self, points_xyz: np.ndarray) -> np.ndarray:
        """Vectorized point-in-volume predicate.

        Parameters
        ----------
        points_xyz : np.ndarray
            Array of shape ``(..., 3)`` of physical (mm) coordinates.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``points_xyz.shape[:-1]`` -- True where the point
            is inside the analytical volume. For non-volume primitives, returns
            an all-False array of the appropriate shape.
        """

    @abstractmethod
    def get_contours(self, geometry: Geometry) -> list[ContourItem]:
        """Produce the list of contours the writer should emit for this primitive.

        Each :class:`ContourItem` becomes one ContourSequence entry inside the
        primitive's ROIContourSequence. The unified shape lets every geometric type
        coexist in one writer code path.
        """

    def to_spec_dict(self) -> dict:
        """JSON-serializable record of parameters + analytical truth."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "geometric_type": self.geometric_type.value,
            "analytical_volume_mm3": self.analytical_volume(),
        }
