# How to use this conformance fixture

This directory contains a self-contained test fixture for any tool that converts
DICOM RTSTRUCT contours into per-ROI NIfTI masks. The masks under `groundtruth/`
were generated analytically (sub-voxel quadrature against the closed-form shape
definitions); they do not depend on any third-party converter.

## Inputs to your tool

| Path | What it is |
|---|---|
| `refct/` | 200 DICOM CT slices (1 mm isotropic, 512×512×200 mm). Use as the reference image series. |
| `rtstruct/primitives_planar.dcm` | RTSTRUCT containing the seven v0.1 ROIs, all `CLOSED_PLANAR`. The two hollow shapes use two coplanar contours per slice, read even-odd by convention (a vendor convention, not defined by the standard). |
| `rtstruct/primitives_xor.dcm` | `hollow_sphere_xor`, `straw_xor`: the same two hollow shapes in the standard's `CLOSEDPLANAR_XOR` encoding (every contour typed `CLOSEDPLANAR_XOR`). |
| `rtstruct/primitives_keyhole.dcm` | `hollow_sphere_keyhole`, `straw_keyhole`: the same two hollow shapes in the standard's single-contour keyhole encoding (one `CLOSED_PLANAR` contour per slice, a zero-width channel to the cavity). |

`manifest.json` lists every RTSTRUCT and the ROIs it holds. Convert each RTSTRUCT
against the same `refct/` series.

## Required outputs from your tool

For each ROI in the manifest, your tool must produce a binary NIfTI at:

```
<predictions>/<roi_name>.nii.gz
```

with **the same geometry as the corresponding ground-truth file** (origin,
spacing, size, direction). The verifier rejects predictions whose geometry
differs by more than 1e-4 mm before scoring — most tool bugs are geometry
mismatches, and surfacing them separately is more diagnostic than a low Dice.

The ROI filenames are:

- `sphere.nii.gz`
- `cube.nii.gz`
- `cylinder.nii.gz`
- `ellipsoid.nii.gz`
- `torus.nii.gz`
- `hollow_sphere.nii.gz`
- `straw.nii.gz`
- `hollow_sphere_xor.nii.gz`, `straw_xor.nii.gz`
- `hollow_sphere_keyhole.nii.gz`, `straw_keyhole.nii.gz`

The ROI names match the `ROIName` field of each ROI in the RTSTRUCT, so a tool
that converts RTSTRUCT to per-ROI NIfTI in the obvious way (using `ROIName`
verbatim as the output filename) will produce these names automatically.

## Verifying

```
rtmask-conformance verify --predictions <your_predictions_dir> --fixture .
```

`--fixture` points at this directory: the verifier reads `manifest.json` and checks every
ground-truth file against its recorded SHA-256 before scoring the ROIs it lists. The
older form, `--groundtruth ./groundtruth`, still works but skips the manifest and the hash check.

The verifier scores each ROI on Dice, Surface DSC @ 1 mm, Hausdorff 95, mean
surface distance, and relative volume error, and exits 0 if every ROI passes
its thresholds (1 otherwise). Pass `--config conformance.yaml` to override the
default thresholds.

## Notes

- The hollow sphere and the straw test hole handling, in three encodings: the
  planar bundle's multi-contour even-odd convention, the standard's
  `CLOSEDPLANAR_XOR` (`*_xor`) and the standard's keyhole contour (`*_keyhole`).
  All three have the same ground truth. Tools that fill the cavity as solid (a
  common bug) will produce a Dice around 0.6 on those ROIs and fail conformance.
- A tool that does not support `CLOSEDPLANAR_XOR` should leave it out of
  `claims: geometric_types:` in its threshold YAML and write nothing for the
  `xor` ROIs; they are then reported `UNSUPPORTED_TYPE`, which does not fail the run.
- All ROIs sit ≥ 40 mm clear of the volume edges, so boundary-clipping
  shouldn't be a factor.
- Ground-truth masks are binarized at the 0.5 partial-volume level (8³
  sub-voxel samples per voxel).
