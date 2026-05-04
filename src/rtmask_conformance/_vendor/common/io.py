"""NIfTI / DICOM-series I/O helpers (vendored from rtmask_validation)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .coords import Geometry


def read_dicom_series(folder: str | Path) -> sitk.Image:
    folder = str(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder))
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def sitk_image_to_geometry(image: sitk.Image) -> Geometry:
    sx, sy, sz = image.GetSpacing()
    cols, rows, slices = image.GetSize()
    return Geometry(
        origin=tuple(image.GetOrigin()),
        spacing=(float(sx), float(sy), float(sz)),
        size=(int(cols), int(rows), int(slices)),
        direction=tuple(image.GetDirection()),
    )


def write_nifti(array_zyx: np.ndarray, geometry: Geometry, out_path: str | Path) -> None:
    img = sitk.GetImageFromArray(array_zyx)
    img.SetSpacing(geometry.spacing)
    img.SetOrigin(geometry.origin)
    img.SetDirection(geometry.direction)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
