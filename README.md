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

## Real-world integration examples

Three end-to-end integrations live in sister projects. Each is the recommended
template for its language / convention: copy the four pieces, adapt the
converter call.

| Tool | Language / runtime | Pattern | Recommended for |
|---|---|---|---|
| [DicomRTTool](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask) | Python / pip | pyproject extra + pytest + CI job | Python packages with an existing pytest suite |
| [PyRaDiSe](https://github.com/brianmanderson/pyradise) | Python / pip | same as DicomRTTool, with single-folder staging | Python packages whose API takes a single root directory |
| [Dicom_RT_Images_Csharp](https://github.com/brianmanderson/Dicom_RT_Images_Csharp) | C# / .NET Framework 4.8 | CI-only: build → headless CLI → verify | Compiled tools with a CLI / headless mode |

The two Python integrations drive the suite from inside pytest; the C#
integration is purely CLI-driven inside a GitHub Actions job. The accuracy
gate is the same — only the surrounding plumbing differs.

### Python: DicomRTTool

The [DicomRTTool](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask)
package wires this conformance suite in as a separate CI check. It's the
recommended pattern if your tool is a Python package with a `pyproject.toml`
and existing pytest suite — copy these four pieces and adapt the converter
call. Live files:

- [pyproject.toml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/pyproject.toml) — opt-in extra
- [tests/test_conformance.py](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/tests/test_conformance.py) — fixture + per-ROI assertions
- [tests/conformance.yaml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/tests/conformance.yaml) — calibrated thresholds
- [.github/workflows/conformance.yml](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/.github/workflows/conformance.yml) — separate "Conformance" CI check

#### 1. pyproject.toml — opt-in extra

Keep the conformance dependency out of the default install so users who only
want the package don't pull `pyyaml`/`trimesh`/etc.:

```toml
[project.optional-dependencies]
conformance = [
    "rtmask-conformance @ git+https://github.com/brianmanderson/RTMaskConformanceTest",
]
```

Developers and CI install with `pip install -e .[conformance]`.

#### 2. tests/test_conformance.py — pytest fixture + assertions

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

#### 3. tests/conformance.yaml — document any threshold relaxations

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

#### 4. .github/workflows/conformance.yml — separate CI check

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

#### What you should expect on first run

- `sphere`, `cylinder`, `ellipsoid`, `torus`, `hollow_sphere`, `straw`
  typically pass on defaults if the converter is correct.
- `cube` is the most common near-miss for boundary-inclusive rasterizers
  (cv2.fillPoly, naïve scanline fill); document the relaxation per above.
- `hollow_sphere` and `straw` are the strongest signal — a ~0.6 Dice on
  these means even-odd / multi-contour XOR is broken, which is a real
  bug in the converter, not a threshold issue.

### Python (variant): PyRaDiSe

[PyRaDiSe](https://github.com/brianmanderson/pyradise) wires the suite in
the same four-piece shape as DicomRTTool — see those subsections above for
the canonical walkthrough. The only differences worth calling out are
PyRaDiSe-specific:

- [pyproject.toml](https://github.com/brianmanderson/pyradise/blob/main/pyproject.toml) — opt-in `conformance` extra
- [tests/test_conformance.py](https://github.com/brianmanderson/pyradise/blob/main/tests/test_conformance.py) — fixture + per-ROI assertions
- [tests/conformance.yaml](https://github.com/brianmanderson/pyradise/blob/main/tests/conformance.yaml) — calibrated thresholds
- [.github/workflows/conformance.yml](https://github.com/brianmanderson/pyradise/blob/main/.github/workflows/conformance.yml) — separate "Conformance" CI check

Three adaptations for PyRaDiSe specifically — these will apply to most
crawler-style packages:

**1. Single-folder staging.** PyRaDiSe's `SubjectDicomCrawler` walks one
root and groups by study/series — it can't take a CT folder and an
RTSTRUCT path as separate arguments. The predictions fixture hard-links
both into one temp dir before invoking the crawler:

```python
def _stage_dicom_inputs(rtstruct, image_folder, stage):
    stage.mkdir(parents=True, exist_ok=True)
    for src in image_folder.glob("*.dcm"):
        try:    os.link(src, stage / src.name)
        except OSError: shutil.copy2(src, stage / src.name)
    try:    os.link(rtstruct, stage / rtstruct.name)
    except OSError: shutil.copy2(rtstruct, stage / rtstruct.name)
```

This pattern generalizes: any tool that takes "a directory with both the
CT and the RTSTRUCT" needs the same staging step.

**2. Defensive image extraction.** Different PyRaDiSe point releases
expose the underlying SimpleITK image under different attribute names
(`get_image_data()` vs `get_image()`). The fixture tries both:

```python
def _extract_sitk_image(seg):
    for attr in ("get_image_data", "get_image"):
        if hasattr(seg, attr):
            try:
                v = getattr(seg, attr)()
                if v is not None: return v
            except Exception:
                continue
    return None
```

If your tool's API has shifted across releases, the same try-multiple-names
pattern keeps the test resilient without pinning a specific version.

**3. Python version mismatch handled by pip.** PyRaDiSe declares Python
≥ 3.8, but `rtmask-conformance` requires ≥ 3.10. The opt-in extra works
out automatically: pip simply refuses to install the extra on 3.8 / 3.9,
so users on older interpreters get PyRaDiSe minus the conformance gate
(the intended behavior — only CI / dev users on 3.10+ run the gate).

The CI workflow also pre-installs `setuptools` to provide the `distutils`
shim PyRaDiSe imports (removed from stdlib in Python 3.12). Same trick
applies to any package that hasn't yet migrated off the standard-library
distutils.

#### Cross-Python rasterizer fingerprinting

PyRaDiSe's first-run cube metrics came in **bit-identical** to DicomRTTool's:

| Metric on `cube` | DicomRTTool | PyRaDiSe |
|---|---|---|
| Dice                       | 0.9835 | 0.9835 |
| Surface DSC @ 1 mm         | 0.999  | 0.999  |
| HD95 (mm)                  | 1.0    | 1.0    |
| MSD (mm)                   | 0.33   | 0.33   |
| Volume relative error      | +3.36% | +3.36% |

Two ostensibly-different Python wrappers around contour-to-mask conversion
producing identical numbers down to four decimal places means they share
an underlying rasterizer (in this case, both call into `cv2.fillPoly`).
That's the suite functioning as a *fingerprint*, not just a pass/fail
gate — useful for reasoning about provenance when an upstream
implementation changes, or when validating that a new wrapper hasn't
introduced incidental drift on top of a shared dependency.

### C# / .NET Framework: Dicom_RT_Images_Csharp

The [Dicom_RT_Images_Csharp](https://github.com/brianmanderson/Dicom_RT_Images_Csharp)
project wires the suite in as a CI-only gate. It's the recommended pattern
when the tool under test is a compiled binary with a CLI or headless mode —
no Python package wrapping, no test runner needed. The pieces are:

- [conformance.yaml](https://github.com/brianmanderson/Dicom_RT_Images_Csharp/blob/main/conformance.yaml) at the repo root — calibrated thresholds (same schema as the Python case).
- [.github/workflows/conformance.yml](https://github.com/brianmanderson/Dicom_RT_Images_Csharp/blob/main/.github/workflows/conformance.yml) — single workflow that builds the C# project, runs its headless converter against the fixture, and verifies.

The C# tool's headless CLI surface (the bit you'll need an equivalent of in
your tool) takes the fixture's RTSTRUCT + reference CT folder and writes one
binary `<roi>.nii.gz` per ROI into an output directory:

```
Dicom_RT_images_Csharp.exe --headless --forward \
    --rtstruct PATH \
    --image-folder PATH \
    --output-folder PATH
```

Sourced in [Cli/HeadlessRunner.cs](https://github.com/brianmanderson/Dicom_RT_Images_Csharp/blob/main/Dicom_RT_images_Csharp/Cli/HeadlessRunner.cs)
— it returns 0/non-zero and emits per-ROI volume rows on stdout. Matching that
contract from your tool (whatever it's written in) is the only language-side
work; everything else is YAML.

#### Workflow walkthrough

The CI job runs the same generate → run-tool → verify chain, just shelling out
to the C# binary in the middle:

```yaml
name: Conformance

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

# Bump these to pin a different SimpleITK release. Required because the
# project's .csproj references SimpleITK C# binaries via a relative
# HintPath, and they're not on NuGet.
env:
  SIMPLEITK_VERSION: "2.5.0"
  SIMPLEITK_ZIP_URL: "https://github.com/SimpleITK/SimpleITK/releases/download/v2.5.0/SimpleITK-2.5.0-CSharp-win64-x64.zip"
  # Opt the JS actions into the upcoming Node.js 24 runtime ahead of
  # GitHub's 2026-06-02 forced switch.
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"

jobs:
  conformance:
    name: RTSTRUCT->mask conformance (C# headless)
    runs-on: windows-latest    # WPF / .NET Framework 4.8 means Windows-only.

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install rtmask-conformance
        # `python -m pip` rather than `pip` directly: on Windows runners the
        # bare pip.exe console script can't be overwritten if pip ever tries
        # to self-upgrade.
        run: python -m pip install git+https://github.com/brianmanderson/RTMaskConformanceTest

      - uses: microsoft/setup-msbuild@v2
      - uses: NuGet/setup-nuget@v2

      - name: Stage SimpleITK C# binaries one level above the repo
        # The .csproj's HintPath resolves to <repo_parent>/SimpleITK/, so
        # that's where the DLLs need to land. The release ZIP layout has
        # an inner directory; flatten it.
        shell: pwsh
        run: |
          Invoke-WebRequest -Uri "${{ env.SIMPLEITK_ZIP_URL }}" -OutFile sitk.zip -UseBasicParsing
          Expand-Archive -Path sitk.zip -DestinationPath sitk-extracted -Force
          $inner = Get-ChildItem sitk-extracted -Directory | Select-Object -First 1
          $sourcePath = if ($null -eq $inner) { "sitk-extracted" } else { $inner.FullName }
          $target = Join-Path (Split-Path -Parent $env:GITHUB_WORKSPACE) "SimpleITK"
          New-Item -ItemType Directory -Path $target -Force | Out-Null
          Copy-Item -Path "$sourcePath\*" -Destination $target -Recurse -Force

      - run: nuget restore Dicom_RT_images_Csharp.sln

      - name: Build C# project (Release|x64)
        # x64 (not AnyCPU) makes the 64-bit native SimpleITK requirement
        # explicit and matches how the project is built locally.
        run: msbuild Dicom_RT_images_Csharp.sln /p:Configuration=Release /p:Platform=x64 /m

      - name: Inspect build output
        # Asserts the three runtime artifacts (managed exe + managed
        # SimpleITK + native SimpleITK) are present before invoking the
        # binary. Catches the most common silent failure: a missing
        # native DLL produces a DllNotFoundException that AttachConsole
        # can swallow.
        shell: pwsh
        run: |
          $bin = "Dicom_RT_images_Csharp\bin\x64\Release"
          Get-ChildItem $bin
          foreach ($f in @("Dicom_RT_images_Csharp.exe", "SimpleITKCSharpManaged.dll", "SimpleITKCSharpNative.dll")) {
              if (-not (Test-Path (Join-Path $bin $f))) { throw "Missing artifact: $f" }
          }

      - run: rtmask-conformance generate ./fixture --n-quadrature 2

      - name: Run C# headless forward conversion
        # AttachConsole(ATTACH_PARENT_PROCESS) inside HeadlessRunner is
        # unreliable on hosted runners — pwsh's `&` invocation against a
        # WinExe doesn't always plumb stdout/stderr back to the GitHub
        # Actions log. Use Start-Process -Wait with explicit redirection
        # to files, then dump them unconditionally so any failure in the
        # binary is debuggable from the run log alone.
        shell: pwsh
        run: |
          $exe = (Resolve-Path "Dicom_RT_images_Csharp\bin\x64\Release\Dicom_RT_images_Csharp.exe").Path
          New-Item -ItemType Directory -Path predictions -Force | Out-Null
          $stdoutPath = Join-Path $env:RUNNER_TEMP "csharp.stdout.log"
          $stderrPath = Join-Path $env:RUNNER_TEMP "csharp.stderr.log"
          $proc = Start-Process -FilePath $exe -NoNewWindow -Wait -PassThru `
              -ArgumentList @(
                  "--headless", "--forward",
                  "--rtstruct",      (Resolve-Path fixture/rtstruct/primitives_planar.dcm).Path,
                  "--image-folder",  (Resolve-Path fixture/refct).Path,
                  "--output-folder", (Resolve-Path predictions).Path
              ) `
              -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
          Write-Host "----- C# stdout -----"; Get-Content $stdoutPath
          Write-Host "----- C# stderr -----"; Get-Content $stderrPath
          if ($proc.ExitCode -ne 0) { throw "Headless conversion failed: $($proc.ExitCode)" }

      - name: Verify
        run: |
          rtmask-conformance verify `
              --predictions ./predictions `
              --groundtruth ./fixture/groundtruth `
              --config ./conformance.yaml `
              --report-json conformance-report.json

      - name: Upload report + artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: conformance-report
          path: |
            conformance-report.json
            predictions/
            fixture/groundtruth/
            fixture/manifest.json
          retention-days: 30
```

#### Cross-language pitfalls worth pre-empting

Some of the steps above look like over-engineering until you hit the failure
mode. The four that bit us during this integration:

1. **`pip install --upgrade pip` fails on Windows runners.** Pip can't
   overwrite its own running `pip.exe`. Either use `python -m pip install`
   (the python.exe entry point isn't the locked file) or skip the upgrade —
   the runner's bundled pip is fresh enough.

2. **`& exe.exe ...` against a WinExe doesn't always surface stdout.** A
   `WinExe` (any GUI-subsystem .NET executable, even one with a CLI mode) has
   no console attached by default. `AttachConsole(ATTACH_PARENT_PROCESS)`
   inside the binary tries to attach to pwsh's console, but the timing and
   redirection on hosted runners is flaky. Use `Start-Process -Wait`
   with `-RedirectStandardOutput`/`-RedirectStandardError` to files and
   dump them with `Get-Content` regardless of exit code. The exit code
   propagates correctly even when the streams don't.

3. **External native DLLs need explicit staging.** SimpleITK's C# wrapper
   isn't on NuGet; the binaries ship as a separate ZIP from
   [github.com/SimpleITK/SimpleITK/releases](https://github.com/SimpleITK/SimpleITK/releases).
   Match whatever path-relative HintPath your `.csproj` uses (this project
   uses `..\..\SimpleITK\`, resolving to `<repo_parent>/SimpleITK/`).

4. **`Release|x64`, not AnyCPU**, when you have native deps. The native
   SimpleITK DLL is x86-64; building AnyCPU on a 64-bit runner technically
   works because the framework picks 64-bit at runtime, but pinning the
   platform makes the requirement explicit and unambiguous in the artifact
   path.

Mention these in your own workflow's comments — future-you will thank past-you.

#### What the run looks like

The verifier produces the same plain-text table the Python case does:

```
rtmask-conformance verify  config=conformance.yaml  7/7 passed

status ROI                                  dice  sDSC1  HD95mm MSD mm dV%
-----------------------------------------------------------------------------
PASS   sphere                              0.9898 1.0000  1.000  0.326   0.75
PASS   cube                                0.9833 1.0000  1.000  0.333   0.00
PASS   cylinder                            0.9872 1.0000  1.000  0.304   1.71
PASS   ellipsoid                           0.9912 1.0000  1.000  0.284   0.84
PASS   torus                               0.9850 1.0000  1.000  0.350   1.63
PASS   hollow_sphere                       0.9857 1.0000  1.000  0.323   0.99
PASS   straw                               0.9821 1.0000  1.000  0.339   1.88
```

The cube's relaxation in [conformance.yaml](https://github.com/brianmanderson/Dicom_RT_Images_Csharp/blob/main/conformance.yaml)
is documented in the file's header — every override should be.

#### The conformance suite as a cross-implementation diff

What makes this gate worth shipping across all three projects is that the
same fixture surfaces qualitatively different rasterizer behaviors:

| Metric on `cube` | DicomRTTool (cv2.fillPoly) | PyRaDiSe (cv2.fillPoly) | This repo (C# scanline) |
|---|---|---|---|
| Dice                       | 0.9835 | 0.9835 | 0.9833 |
| Surface DSC @ 1 mm         | 0.999  | 0.999  | 1.000  |
| HD95 (mm)                  | 1.0    | 1.0    | 1.0    |
| MSD (mm)                   | 0.33   | 0.33   | 0.33   |
| **Volume relative error**  | **+3.36%** | **+3.36%** | **0.00%** |

cv2.fillPoly is biased: it counts ~3.4% more voxels than ground truth, and
both Python wrappers around it inherit the bias bit-for-bit. The C# scanline
implementation gets the volume *exactly* right but disagrees with the GT on
which ~3500 voxels along the boundary belong to the cube — symmetric error,
not systematic over-fill. The two cv2 wrappers and the C# scanline land at
near-identical Dice on the cube but for completely different reasons.
Without the analytic ground truth this distinction would be invisible; with
it, all three implementations get a sharper picture of where they actually
sit, and you can tell at a glance which converters share a rasterizer.

## Provenance and ground-truth code

Ground-truth is computed by partial-volume sub-voxel quadrature against the analytic
shape definition (default: 8³ samples per voxel, thresholded at 0.5). The primitive
classes, voxelizer, RTSTRUCT writer, and metric implementations are vendored from
the upstream `rtmask_validation` project — see `tools/UPSTREAM_VERSION.txt` for the
exact source commit, and `tools/sync_from_upstream.py` for the re-vendor script.

## License

Apache-2.0
