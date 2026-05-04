# RTMaskConformanceTest

A universal conformance test suite for DICOM-RT-to-NIfTI converters.

This tool generates a deterministic synthetic CT volume plus an RTSTRUCT containing
seven analytically-defined ROIs (sphere, cube, cylinder, ellipsoid, torus, hollow
sphere, hollow cylinder), and provides analytic ground-truth NIfTI masks for each.
Any tool that claims to convert RTSTRUCT contours into per-ROI NIfTI masks can be
checked against these references — language-agnostic, two-step:

1. `rtmask-conformance generate <fixture_dir>` writes the fixture (CT + RTSTRUCT + GT).
2. Run your converter against the fixture; have it write `<roi>.nii.gz` files into a predictions directory.
3. `rtmask-conformance verify --predictions <pred_dir> --groundtruth <fixture_dir>/groundtruth` scores each ROI and exits 0 (pass) or 1 (fail).

Because the ground-truth masks are computed analytically (not by a competing
converter), the measurement is independent of the tool under test.

## Install

```
pip install git+https://github.com/brianmanderson/RTMaskConformanceTest
```

Requires Python ≥ 3.10. Runtime deps: `pydicom`, `SimpleITK`, `numpy`, `scipy`, `pyyaml`.

## Quick start

```bash
# 1. Generate fixture (DICOM CT series + RTSTRUCT + ground-truth NIfTIs)
rtmask-conformance generate ./fixture

# 2. Run YOUR tool. It must produce one binary NIfTI per ROI:
#    ./predictions/sphere_r40_center.nii.gz
#    ./predictions/cube_s60_x100_y100.nii.gz
#    ... etc
#
#    Inputs to your tool:
#      DICOM CT series : ./fixture/refct/
#      RTSTRUCT        : ./fixture/rtstruct/primitives_planar.dcm

# 3. Score the predictions
rtmask-conformance verify --predictions ./predictions --groundtruth ./fixture/groundtruth
```

Exit codes: `0` all ROIs PASS, `1` any FAIL/MISSING/GEOMETRY_MISMATCH, `2` usage error.

See the file `README_FOR_TOOL_AUTHOR.md` written into the fixture directory for the
complete contract a tool author must satisfy.

## ROIs (v0.1)

Seven closed-planar primitives, each centered in a different region of a 512×512×200
mm volume to avoid overlap:

| ROI name | Shape | Note |
|---|---|---|
| `sphere_r40_center` | sphere, r=40 mm | smooth, convex |
| `cube_s60_x100_y100` | cube, side 60 mm | axis-aligned |
| `cylinder_r30_h80_x400_y100` | z-axis cylinder | curved + flat caps |
| `ellipsoid_30_50_60_x100_y400` | ellipsoid | anisotropic |
| `torus_R60_r20_x400_y400` | z-axis torus | annular cross-sections |
| `hollow_sphere_R40_r20_x256_y100` | hollow sphere | XOR (multi-contour) |
| `straw_R40_r20_h120_x256_y400` | hollow cylinder | XOR (multi-contour) |

Tools that mishandle multi-contour even-odd fill produce a solid (Dice ≈ 0.6) on the
two XOR primitives and will fail conformance loudly — that is a feature, not a bug.

## Metrics

Each ROI is scored on:

- **Dice** (volumetric)
- **Surface DSC @ 1 mm** (Nikolov-style, tolerance-bounded)
- **Hausdorff 95** (mm)
- **Mean surface distance** (mm)
- **Relative volume error**

A geometry precheck runs first: if a prediction's `(origin, spacing, size, direction)`
differs from the ground-truth NIfTI by more than 1e-4, the ROI is flagged
`GEOMETRY_MISMATCH` rather than scored — most third-party tool bugs are geometry, not
voxel labeling, and surfacing them separately is more diagnostic.

## Custom thresholds

Defaults ship in `src/rtmask_conformance/data/default_thresholds.yaml`. Override with
your own YAML and pass `--config conformance.yaml`:

```yaml
schema_version: 1
defaults:
  dice: 0.95
  surface_dice_1mm: 0.95
  hd95_mm: 2.0
  msd_mm: 0.5
  volume_rel_err: 0.03
primitives:
  torus_R60_r20_x400_y400:
    dice: 0.90        # relax for tools known to struggle with toroidal cross-sections
```

Per-primitive overrides shallow-merge over `defaults`. Unknown `schema_version` is
rejected.

## Use as a pytest module

```bash
RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
RTMASK_CONFORMANCE_GROUNDTRUTH=./fixture/groundtruth \
pytest --pyargs rtmask_conformance.tests
```

This produces one parametrized test per ROI with the same pass/fail semantics as the
CLI.

## Provenance and ground-truth code

Ground-truth is computed by partial-volume sub-voxel quadrature against the analytic
shape definition (default: 8³ samples per voxel, thresholded at 0.5). The primitive
classes, voxelizer, RTSTRUCT writer, and metric implementations are vendored from
the upstream `rtmask_validation` project — see `tools/UPSTREAM_VERSION.txt` for the
exact source commit, and `tools/sync_from_upstream.py` for the re-vendor script.

## License

Apache-2.0
