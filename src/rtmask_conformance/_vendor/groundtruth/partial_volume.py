"""Partial-volume ground-truth rasterization (vendored from rtmask_validation)."""

from __future__ import annotations

import numpy as np

from ..common.coords import Geometry
from ..primitives.base import AnalyticalPrimitive


def _bounding_box_voxel_indices(
    primitive: AnalyticalPrimitive, geometry: Geometry, pad_voxels: int = 1
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    mesh = primitive.analytical_surface_mesh()
    cols, rows, slices = geometry.size
    if mesh is None:
        return ((0, cols), (0, rows), (0, slices))

    bbox_min, bbox_max = mesh.bounds
    idx_min = geometry.physical_to_index(bbox_min)
    idx_max = geometry.physical_to_index(bbox_max)
    x_lo = max(0, int(np.floor(min(idx_min[0], idx_max[0])) - pad_voxels))
    x_hi = min(cols, int(np.ceil(max(idx_min[0], idx_max[0])) + 1 + pad_voxels))
    y_lo = max(0, int(np.floor(min(idx_min[1], idx_max[1])) - pad_voxels))
    y_hi = min(rows, int(np.ceil(max(idx_min[1], idx_max[1])) + 1 + pad_voxels))
    z_lo = max(0, int(np.floor(min(idx_min[2], idx_max[2])) - pad_voxels))
    z_hi = min(slices, int(np.ceil(max(idx_min[2], idx_max[2])) + 1 + pad_voxels))

    if x_lo >= x_hi or y_lo >= y_hi or z_lo >= z_hi:
        return None
    return ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))


def partial_volume_mask(
    primitive: AnalyticalPrimitive,
    geometry: Geometry,
    n_quadrature: int = 8,
) -> np.ndarray:
    """Compute the partial-volume fraction mask for ``primitive`` on ``geometry``.

    Returns a float32 (slices, rows, cols) array with values in [0, 1].
    """
    if n_quadrature < 1:
        raise ValueError(f"n_quadrature must be >= 1 (got {n_quadrature}).")

    cols, rows, slices = geometry.size
    fractions = np.zeros((slices, rows, cols), dtype=np.float32)

    bbox = _bounding_box_voxel_indices(primitive, geometry)
    if bbox is None:
        return fractions

    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = bbox

    offsets_1d = (np.arange(n_quadrature, dtype=np.float64) + 0.5) / n_quadrature - 0.5
    dxx, dyy, dzz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    sub_offsets = np.stack(
        [dxx.ravel(), dyy.ravel(), dzz.ravel()], axis=1
    )
    n_samples = sub_offsets.shape[0]

    voxel_x = np.arange(x_lo, x_hi, dtype=np.float64)
    voxel_y = np.arange(y_lo, y_hi, dtype=np.float64)
    nx_box = voxel_x.size
    ny_box = voxel_y.size

    spacing = np.asarray(geometry.spacing, dtype=np.float64)
    origin = np.asarray(geometry.origin, dtype=np.float64)
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)

    grid_x, grid_y = np.meshgrid(voxel_x, voxel_y, indexing="xy")

    for z in range(z_lo, z_hi):
        idx_grid = np.stack(
            [grid_x, grid_y, np.full_like(grid_x, z)], axis=-1
        )
        idx_with_subs = idx_grid[:, :, None, :] + sub_offsets[None, None, :, :]
        idx_flat = idx_with_subs.reshape(-1, 3)
        phys_flat = origin + (idx_flat * spacing) @ direction.T
        inside = primitive.is_inside(phys_flat)
        inside = inside.reshape(ny_box, nx_box, n_samples)
        fractions[z, y_lo:y_hi, x_lo:x_hi] = inside.mean(axis=-1).astype(np.float32)

    return fractions


def binary_threshold(fractions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a partial-volume fraction array into a binary uint8 mask."""
    return (fractions >= threshold).astype(np.uint8)
