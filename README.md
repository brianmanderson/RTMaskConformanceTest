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
#    ./predictions/sphere.nii.gz
#    ./predictions/cube.nii.gz
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

| ROI name | Shape | Dimensions | Note |
|---|---|---|---|
| `sphere` | sphere | r = 40 mm | smooth, convex |
| `cube` | cube | side 60 mm | axis-aligned |
| `cylinder` | z-axis cylinder | r = 30, h = 80 mm | curved + flat caps |
| `ellipsoid` | ellipsoid | semi-axes (30, 50, 60) mm | anisotropic |
| `torus` | z-axis torus | R = 60, r = 20 mm | annular cross-sections |
| `hollow_sphere` | hollow sphere | R = 40, r = 20 mm | XOR (multi-contour) |
| `straw` | hollow cylinder | R = 40, r = 20, h = 120 mm | XOR (multi-contour) |

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
  torus:
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
