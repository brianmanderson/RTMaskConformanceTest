"""DICOM ContourGeometricType strings (vendored from rtmask_validation)."""

from __future__ import annotations

from ..primitives.base import GeometricType

DICOM_TYPE_VALUES: dict[GeometricType, str] = {
    GeometricType.CLOSED_PLANAR: "CLOSED_PLANAR",
    GeometricType.CLOSED_PLANAR_XOR: "CLOSED_PLANAR_XOR",
    GeometricType.OPEN_PLANAR: "OPEN_PLANAR",
    GeometricType.OPEN_NONPLANAR: "OPEN_NONPLANAR",
    GeometricType.CLOSED_NONPLANAR: "CLOSED_NONPLANAR",
    GeometricType.POINT: "POINT",
}


def dicom_string(geom: GeometricType) -> str:
    return DICOM_TYPE_VALUES[geom]
