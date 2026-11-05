"""Thin I/O wrappers used across the package.

We deliberately keep the surface area small. SimpleITK is used directly elsewhere
where finer control is needed; these helpers exist so primitive / ground-truth code
can stay framework-agnostic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .coords import Geometry


def read_dicom_series(folder: str | Path) -> sitk.Image:
    """Load a DICOM image series into a single SimpleITK Image.

    Parameters
    ----------
    folder : str | Path
        Directory containing the DICOM image slices (one .dcm per slice).

    Returns
    -------
    sitk.Image
        3D image in the order returned by SimpleITK ``ImageSeriesReader``
        (sorted by slice position).
    """
    folder = str(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder))
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def sitk_image_to_geometry(image: sitk.Image) -> Geometry:
    """Project a SimpleITK Image's geometry into a serializable Geometry dataclass."""
    sx, sy, sz = image.GetSpacing()
    cols, rows, slices = image.GetSize()
    return Geometry(
        origin=tuple(image.GetOrigin()),
        spacing=(float(sx), float(sy), float(sz)),
        size=(int(cols), int(rows), int(slices)),
        direction=tuple(image.GetDirection()),
    )


def write_nifti(array_zyx: np.ndarray, geometry: Geometry, out_path: str | Path) -> None:
    """Write a (slices, rows, cols) numpy array as a NIfTI file with the given geometry.

    Notes
    -----
    The array layout follows the SimpleITK convention where the first axis is the
    slowest-varying (slice index, z). Spacing/origin/direction map to the SimpleITK
    image as one would expect.
    """
    img = sitk.GetImageFromArray(array_zyx)
    img.SetSpacing(geometry.spacing)
    img.SetOrigin(geometry.origin)
    img.SetDirection(geometry.direction)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def sha256_file(path: str | Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
