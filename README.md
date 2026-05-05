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

## Real-world integration example: DicomRTTool

The [DicomRTTool](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask)
package wires this conformance suite in as a separate CI check. It's the
recommended pattern if your tool is a Python package with a `pyproject.toml`
and existing pytest suite — copy these four pieces and adapt the converter
call. Live files:

- [pyproject.toml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/pyproject.toml) — opt-in extra
- [tests/test_conformance.py](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/tests/test_conformance.py) — fixture + per-ROI assertions
- [tests/conformance.yaml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/tests/conformance.yaml) — calibrated thresholds
- [.github/workflows/conformance.yml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/.github/workflows/conformance.yml) — separate "Conformance" CI check

### 1. pyproject.toml — opt-in extra

Keep the conformance dependency out of the default install so users who only
want the package don't pull `pyyaml`/`trimesh`/etc.:

```toml
[project.optional-dependencies]
conformance = [
    "rtmask-conformance @ git+https://github.com/brianmanderson/RTMaskConformanceTest",
]
```

Developers and CI install with `pip install -e .[conformance]`.

### 2. tests/test_conformance.py — pytest fixture + assertions

```python
"""Conformance test: <YourTool> vs RTMaskConformanceTest analytic ground truth."""
from __future__ import annotations
import os
from pathlib import Path

import pytest
import SimpleITK as sitk

# Skips the entire module if the conformance extra isn't installed,
# so the default `pytest` run is unaffected.
rtmask_conformance = pytest.importorskip(
    "rtmask_conformance",
    reason="install the `conformance` extra: pip install -e .[conformance]",
)

from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config  # noqa: E402
from rtmask_conformance.generate import GenerateOptions  # noqa: E402
from rtmask_conformance.verify import Status, evaluate_one  # noqa: E402

# >>> Replace this import with your tool's converter API <<<
from YourTool import RTStructToMaskConverter  # noqa: E402


_CONFIG_YAML = Path(__file__).with_name("conformance.yaml")


@pytest.fixture(scope="session")
def conformance_fixture(tmp_path_factory):
    """Synthetic CT + RTSTRUCT + analytic GT NIfTIs (one per ROI)."""
    out = tmp_path_factory.mktemp("conformance_fixture")
    # n_quadrature=2 keeps fixture build under ~30 s; n=8 is the published default.
    generate_fixture(out, options=GenerateOptions(n_quadrature=2))
    return out


@pytest.fixture(scope="session")
def predictions(conformance_fixture, tmp_path_factory):
    """Run YOUR tool against the fixture; emit one binary <roi>.nii.gz per ROI."""
    pred_dir = tmp_path_factory.mktemp("preds")

    # >>> Adapt this block to your tool's API <<<
    converter = RTStructToMaskConverter(roi_names=list(CONFORMANCE_ROIS))
    converter.load_dicom_series(conformance_fixture / "refct")
    converter.load_rtstruct(conformance_fixture / "rtstruct" / "primitives_planar.dcm")

    # The verifier expects <pred_dir>/<roi>.nii.gz per ROI. If your tool emits
    # a single labeled mask, split it into per-ROI binaries here:
    for roi in CONFORMANCE_ROIS:
        binary_mask = converter.get_roi_mask(roi)        # <-- your API
        img = sitk.GetImageFromArray(binary_mask.astype("uint8"))
        img.CopyInformation(converter.reference_image)   # <-- your API
        sitk.WriteImage(img, str(pred_dir / f"{roi}.nii.gz"))

    return pred_dir


@pytest.fixture(scope="session")
def conformance_config():
    """Resolution: env var > tests/conformance.yaml > package defaults."""
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    if config_path is None and _CONFIG_YAML.is_file():
        config_path = str(_CONFIG_YAML)
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_conformance(roi, conformance_fixture, predictions, conformance_config):
    pred = predictions / f"{roi}.nii.gz"
    gt = conformance_fixture / "groundtruth" / f"{roi}.nii.gz"
    result = evaluate_one(roi, pred, gt, conformance_config)
    if result.status != Status.PASS:
        pytest.fail(
            f"{roi}: {result.status.value}\n"
            f"  violations: {result.violations}\n"
            f"  metrics:    {result.metrics}\n"
            f"  thresholds: {result.thresholds}"
        )
```

The only places you adapt are the marked `>>> ... <<<` blocks: the import and
the converter-driving block inside the `predictions` fixture. Everything
else (fixture wiring, geometry handling, parametrization, threshold
resolution) is identical across consumers.

### 3. tests/conformance.yaml — document any threshold relaxations

The first time you run the test, expect one or two ROIs to land just under
the published defaults — most rasterizers carry a half-voxel boundary bias.
Rather than baking that into the package, document it locally so a future
voxelizer fix can tighten it:

```yaml
schema_version: 1
primitives:
  cube:
    # cv2.fillPoly is boundary-inclusive: every voxel touched by the polygon
    # is filled. For an axis-aligned 60 mm cube on 1 mm voxels the rasterized
    # mask gains ~3.4% volume from boundary pixels along each face. Surface
    # metrics (sDSC=0.999, HD95=1.0 mm, MSD=0.33 mm) confirm the geometry
    # is right; the volume gap is purely the boundary convention. Tighten
    # back to defaults once the rasterizer honours a half-voxel-shrink.
    dice: 0.98
    volume_rel_err: 0.04
```

The header of [tests/conformance.yaml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/tests/conformance.yaml) in DicomRTTool is the
canonical example: every relaxation is dated, attributed to a specific
behavior, and ends with the path back to the published default. That way
the YAML stays self-explanatory as the rasterizer evolves.

### 4. .github/workflows/conformance.yml — separate CI check

A standalone job means "Conformance" appears as its own status check on PRs,
distinct from the existing `Tests` matrix:

```yaml
name: Conformance

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]
  workflow_dispatch:  # manual run from any branch

jobs:
  conformance:
    name: RTSTRUCT->mask conformance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml
      - run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,conformance]"
      - run: pytest tests/test_conformance.py -v
```

Conformance is an accuracy property — Python/OS portability is already
covered by your main test matrix, so a single `ubuntu-latest × py3.12`
job here is plenty. `workflow_dispatch` lets you re-run manually from
the Actions tab after an upstream `rtmask-conformance` change without
needing a code push.

### What you should expect on first run

- `sphere`, `cylinder`, `ellipsoid`, `torus`, `hollow_sphere`, `straw`
  typically pass on defaults if the converter is correct.
- `cube` is the most common near-miss for boundary-inclusive rasterizers
  (cv2.fillPoly, naïve scanline fill); document the relaxation per above.
- `hollow_sphere` and `straw` are the strongest signal — a ~0.6 Dice on
  these means even-odd / multi-contour XOR is broken, which is a real
  bug in the converter, not a threshold issue.

## Provenance and ground-truth code

Ground-truth is computed by partial-volume sub-voxel quadrature against the analytic
shape definition (default: 8³ samples per voxel, thresholded at 0.5). The primitive
classes, voxelizer, RTSTRUCT writer, and metric implementations are vendored from
the upstream `rtmask_validation` project — see `tools/UPSTREAM_VERSION.txt` for the
exact source commit, and `tools/sync_from_upstream.py` for the re-vendor script.

## License

Apache-2.0
