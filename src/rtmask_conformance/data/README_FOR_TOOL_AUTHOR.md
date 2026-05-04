# How to use this conformance fixture

This directory contains a self-contained test fixture for any tool that converts
DICOM RTSTRUCT contours into per-ROI NIfTI masks. The masks under `groundtruth/`
were generated analytically (sub-voxel quadrature against the closed-form shape
definitions); they do not depend on any third-party converter.

## Inputs to your tool

| Path | What it is |
|---|---|
| `refct/` | 200 DICOM CT slices (1 mm isotropic, 512×512×200 mm). Use as the reference image series. |
| `rtstruct/primitives_planar.dcm` | RTSTRUCT containing the seven ROIs, all `CLOSED_PLANAR` (multi-contour XOR for the two hollow shapes). |

## Required outputs from your tool

For each ROI in the manifest, your tool must produce a binary NIfTI at:

```
<predictions>/<roi_name>.nii.gz
```

with **the same geometry as the corresponding ground-truth file** (origin,
spacing, size, direction). The verifier rejects predictions whose geometry
differs by more than 1e-4 mm before scoring — most tool bugs are geometry
mismatches, and surfacing them separately is more diagnostic than a low Dice.

The seven ROI filenames are:

- `sphere.nii.gz`
- `cube.nii.gz`
- `cylinder.nii.gz`
- `ellipsoid.nii.gz`
- `torus.nii.gz`
- `hollow_sphere.nii.gz`
- `straw.nii.gz`

The ROI names match the `ROIName` field of each ROI in the RTSTRUCT, so a tool
that converts RTSTRUCT to per-ROI NIfTI in the obvious way (using `ROIName`
verbatim as the output filename) will produce these names automatically.

## Verifying

```
rtmask-conformance verify --predictions <your_predictions_dir> --groundtruth ./groundtruth
```

The verifier scores each ROI on Dice, Surface DSC @ 1 mm, Hausdorff 95, mean
surface distance, and relative volume error, and exits 0 if every ROI passes
its thresholds (1 otherwise). Pass `--config conformance.yaml` to override the
default thresholds.

## Notes

- The two hollow shapes (`hollow_sphere_*` and `straw_*`) test multi-contour
  even-odd fill. Tools that fill the inner contour as solid (a common bug)
  will produce a Dice around 0.6 on those ROIs and fail conformance.
- All ROIs sit ≥ 40 mm clear of the volume edges, so boundary-clipping
  shouldn't be a factor.
- Ground-truth masks are binarized at the 0.5 partial-volume level (8³
  sub-voxel samples per voxel).
