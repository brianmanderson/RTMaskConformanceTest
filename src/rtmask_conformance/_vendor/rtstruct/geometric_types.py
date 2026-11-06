"""DICOM ContourGeometricType constants and a small dispatch helper.

Contour Geometric Type (3006,0042) has five Enumerated Values (PS3.3 C.8.8.6):

* ``CLOSED_PLANAR``    -- a closed coplanar polyline (single z).
* ``CLOSEDPLANAR_XOR`` -- a closed coplanar polyline combined with the ROI's other
  inner or outer contours by XOR (the standard's explicit hole encoding). There is
  no underscore between CLOSED and PLANAR (16 characters, the CS maximum). Added by
  CP-2037, first published in PS3.3 2021a.
* ``OPEN_PLANAR``      -- an open coplanar polyline (single z).
* ``OPEN_NONPLANAR``   -- an open polyline whose vertices are not all coplanar.
* ``POINT``            -- a single point.

``CLOSED_NONPLANAR`` (a closed polyline whose vertices are not all coplanar) is
**not** a DICOM value. An RTSTRUCT carrying it is invalid, and a reader that rejects
it, as RayStation does, is behaving correctly. It is mapped here only so the
off-registry probe primitives can be written.

The registry's hollow primitives use neither hole encoding the standard defines
(a keyhole contour or ``CLOSEDPLANAR_XOR``). They use several coplanar
``CLOSED_PLANAR`` contours read even-odd, the common vendor convention.
"""

from __future__ import annotations

from ..primitives.base import GeometricType

#: Map from our internal :class:`GeometricType` enum value to the DICOM string.
DICOM_TYPE_VALUES: dict[GeometricType, str] = {
    GeometricType.CLOSED_PLANAR: "CLOSED_PLANAR",
    GeometricType.CLOSEDPLANAR_XOR: "CLOSEDPLANAR_XOR",
    GeometricType.OPEN_PLANAR: "OPEN_PLANAR",
    GeometricType.OPEN_NONPLANAR: "OPEN_NONPLANAR",
    GeometricType.CLOSED_NONPLANAR: "CLOSED_NONPLANAR",  # non-standard probe value
    GeometricType.POINT: "POINT",
}


def dicom_string(geom: GeometricType) -> str:
    """Return the DICOM ``ContourGeometricType`` string for an enum value."""
    return DICOM_TYPE_VALUES[geom]
