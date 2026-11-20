"""The ``centre_in`` reference: a voxel belongs to the structure when its centre does.

One of the three boundary conventions a tool can declare
(:data:`rtmask_conformance.thresholds.CONVENTIONS`). A voxel is foreground when its centre
satisfies the primitive's analytic inside test, evaluated directly with no
sub-voxel quadrature. This is the rule a scan-line rasteriser implements when it
fills the voxels whose centres fall inside each contour.

The inside test is the vendored primitive's own :meth:`is_inside`, called here,
never copied, and the evaluation is restricted to the same padded bounding box
:func:`partial_volume_mask` uses, so the result is the partial-volume
quadrature with one sample per voxel placed at its centre.
"""

from __future__ import annotations

import numpy as np

from ._vendor.common.coords import Geometry
from ._vendor.groundtruth.partial_volume import _bounding_box_voxel_indices
from ._vendor.primitives.base import AnalyticalPrimitive


def centre_in_mask(primitive: AnalyticalPrimitive, geometry: Geometry) -> np.ndarray:
    """``(z, y, x)`` uint8 mask: 1 where the voxel centre is inside ``primitive``."""
    cols, rows, slices = geometry.size
    mask = np.zeros((slices, rows, cols), dtype=np.uint8)
    bbox = _bounding_box_voxel_indices(primitive, geometry)
    if bbox is None:
        return mask
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = bbox

    origin = np.asarray(geometry.origin, dtype=np.float64)
    spacing = np.asarray(geometry.spacing, dtype=np.float64)
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    grid_x, grid_y = np.meshgrid(
        np.arange(x_lo, x_hi, dtype=np.float64),
        np.arange(y_lo, y_hi, dtype=np.float64),
        indexing="xy",
    )  # (ny, nx)
    # One slice at a time, as partial_volume_mask does, to keep memory bounded.
    for z in range(z_lo, z_hi):
        idx = np.stack([grid_x, grid_y, np.full_like(grid_x, z)], axis=-1).reshape(-1, 3)
        phys = origin + (idx * spacing) @ direction.T
        inside = np.asarray(primitive.is_inside(phys), dtype=bool).reshape(grid_x.shape)
        mask[z, y_lo:y_hi, x_lo:x_hi] = inside
    return mask
