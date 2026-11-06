"""Partial-volume ground-truth rasterization.

For each voxel in the reference grid, we evaluate an N x N x N sub-grid of points and
count how many fall inside the primitive's analytical volume. The result is a float32
array of voxel-fractions in [0, 1], where ``sum(fractions) * voxel_volume`` converges
to the analytical volume as ``n_quadrature`` grows.

Implementation notes:

* The naive bounding box for a primitive is its analytical surface mesh's AABB,
  expanded by one voxel in each direction. Voxels outside the box are guaranteed
  zero. This matters for the 512x512x200 production grid -- evaluating every voxel
  against the predicate is fine for spheres but expensive for sub-voxel structures
  where most of the grid is empty.
* Within the bounding box, we vectorize the sub-voxel sampling: build a
  ``(M, N, N, N, 3)`` array of physical coordinates, call ``primitive.is_inside(...)``
  once, then mean-reduce along the quadrature axes.
"""

from __future__ import annotations

import numpy as np

from ..common.coords import Geometry
from ..primitives.base import AnalyticalPrimitive


def _bounding_box_voxel_indices(
    primitive: AnalyticalPrimitive, geometry: Geometry, pad_voxels: int = 1
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Compute a voxel-index bounding box around the primitive's analytical extent.

    Uses the surface mesh AABB if available; otherwise falls back to the entire grid.
    Returns ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` (half-open ranges) clamped
    to the grid, or ``None`` if the primitive is entirely outside the grid.
    """
    mesh = primitive.analytical_surface_mesh()
    cols, rows, slices = geometry.size
    if mesh is None:
        return ((0, cols), (0, rows), (0, slices))

    bbox_min, bbox_max = mesh.bounds  # shape (3,) each, in physical mm
    # Convert to continuous voxel indices.
    idx_min = geometry.physical_to_index(bbox_min)
    idx_max = geometry.physical_to_index(bbox_max)
    # Pad and clamp.
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

    Parameters
    ----------
    primitive : AnalyticalPrimitive
        Must implement ``is_inside(points_xyz)``.
    geometry : Geometry
        Reference-image geometry (origin, spacing, size, direction).
    n_quadrature : int
        Number of sub-voxel sample points per axis. Total samples per voxel is
        ``n_quadrature ** 3``. Default 8 -> 512 samples per voxel.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(slices, rows, cols)`` (numpy/sitk convention),
        with values in ``[0, 1]``.
    """
    if n_quadrature < 1:
        raise ValueError(f"n_quadrature must be >= 1 (got {n_quadrature}).")

    cols, rows, slices = geometry.size
    fractions = np.zeros((slices, rows, cols), dtype=np.float32)

    bbox = _bounding_box_voxel_indices(primitive, geometry)
    if bbox is None:
        return fractions

    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = bbox

    # Sub-voxel offsets in [-0.5, +0.5] (voxel-index units).
    offsets_1d = (np.arange(n_quadrature, dtype=np.float64) + 0.5) / n_quadrature - 0.5
    # Cartesian product -> shape (n^3, 3): (dx, dy, dz).
    dxx, dyy, dzz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    sub_offsets = np.stack(
        [dxx.ravel(), dyy.ravel(), dzz.ravel()], axis=1
    )  # (n^3, 3)
    n_samples = sub_offsets.shape[0]

    # Process the bounding box one slice at a time to keep memory bounded.
    voxel_x = np.arange(x_lo, x_hi, dtype=np.float64)
    voxel_y = np.arange(y_lo, y_hi, dtype=np.float64)
    nx_box = voxel_x.size
    ny_box = voxel_y.size

    spacing = np.asarray(geometry.spacing, dtype=np.float64)
    origin = np.asarray(geometry.origin, dtype=np.float64)
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)

    # Pre-compute the 2D voxel-center index grid for the slice.
    grid_x, grid_y = np.meshgrid(voxel_x, voxel_y, indexing="xy")  # (ny, nx)

    for z in range(z_lo, z_hi):
        # (ny, nx, 3) voxel-index grid at z.
        idx_grid = np.stack(
            [grid_x, grid_y, np.full_like(grid_x, z)], axis=-1
        )  # (ny_box, nx_box, 3)
        # Add sub-voxel offsets: (ny, nx, n_samples, 3).
        idx_with_subs = idx_grid[:, :, None, :] + sub_offsets[None, None, :, :]
        # Convert all (ny * nx * n_samples, 3) indices to physical coords.
        idx_flat = idx_with_subs.reshape(-1, 3)
        phys_flat = origin + (idx_flat * spacing) @ direction.T
        inside = primitive.is_inside(phys_flat)
        # Reshape to (ny, nx, n_samples) and mean over the quadrature axis.
        inside = inside.reshape(ny_box, nx_box, n_samples)
        fractions[z, y_lo:y_hi, x_lo:x_hi] = inside.mean(axis=-1).astype(np.float32)

    return fractions


def binary_threshold(fractions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a partial-volume fraction array into a binary uint8 mask.

    The comparison is ``>=``, so ``threshold=0.5`` is the "strict-center"
    (majority-of-voxel) rule: a voxel is foreground iff at least half of its
    volume lies inside the primitive.
    """
    return (fractions >= threshold).astype(np.uint8)


def any_overlap_mask(fractions: np.ndarray) -> np.ndarray:
    """Binarize a partial-volume fraction array under the ``PV > 0`` rule.

    This is the "any-overlap" / polygon-touch oracle: a voxel is foreground
    iff *any* part of it lies inside the primitive. It is the natural
    counterpart to :func:`binary_threshold` at 0.5 and brackets the truth from
    the other side -- PV>=0.5 under-counts boundary voxels, PV>0 over-counts
    them. Scoring the same tool masks against both is how the benchmark
    separates "distance from analytical truth" from "distance from the
    strict-center convention" (which several comparators implement by
    construction).

    Note this is NOT ``binary_threshold(fractions, threshold=0.0)`` -- that
    would use ``>= 0`` and mark the entire grid.
    """
    return (fractions > 0.0).astype(np.uint8)
