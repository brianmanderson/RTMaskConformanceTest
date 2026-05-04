"""Generate a conformance fixture: synthetic CT + RTSTRUCT + analytic GT NIfTIs.

The output directory layout matches what ``verify`` (and the consuming tool)
expects:

    <out_dir>/
      refct/                              200 DICOM CT slices
      rtstruct/primitives_planar.dcm      single RTSTRUCT with all 7 ROIs
      groundtruth/<roi>.nii.gz            7 binary masks (uint8, PV >= 0.5)
      specs/<roi>.json                    analytic spec metadata
      manifest.json                       version + ROI list + sha256s
      README_FOR_TOOL_AUTHOR.md           drop-in instructions for converters
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from . import __version__
from ._roi_set import CONFORMANCE_ROIS, build_conformance_primitives
from ._vendor.common.io import sha256_file, write_nifti
from ._vendor.groundtruth.partial_volume import binary_threshold, partial_volume_mask
from ._vendor.refimage.build_reference_ct import ReferenceCTSpec, build_reference_ct
from ._vendor.rtstruct.pydicom_writer import build_rtstruct
from .manifest import MANIFEST_VERSION, FixtureManifest, write_manifest


@dataclass(frozen=True)
class GenerateOptions:
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[int, int, int] = (512, 512, 200)
    n_quadrature: int = 8


def generate_fixture(out_dir: str | Path, options: GenerateOptions | None = None) -> Path:
    """Build a complete fixture under ``out_dir`` and return the directory path.

    Idempotent: re-running with the same options overwrites existing files
    deterministically (UIDs are derived from a fixed salt, voxel content is a
    pure function of geometry).
    """
    options = options or GenerateOptions()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    refct_dir = out / "refct"
    rtstruct_dir = out / "rtstruct"
    gt_dir = out / "groundtruth"
    specs_dir = out / "specs"
    for d in (refct_dir, rtstruct_dir, gt_dir, specs_dir):
        d.mkdir(parents=True, exist_ok=True)

    spec = ReferenceCTSpec(voxel_size=options.voxel_size, size=options.size)
    geometry = build_reference_ct(refct_dir, spec=spec)

    primitives = build_conformance_primitives()

    rtss_path = rtstruct_dir / "primitives_planar.dcm"
    build_rtstruct(
        primitives=primitives,
        ref_image_folder=refct_dir,
        out_path=rtss_path,
        structure_set_label="RTMASK_CONFORM",
    )

    groundtruth_sha256: dict[str, str] = {}
    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=options.n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        gt_path = gt_dir / f"{primitive.name}.nii.gz"
        write_nifti(binary, geometry, gt_path)
        groundtruth_sha256[primitive.name] = sha256_file(gt_path)

        spec_dict = primitive.to_spec_dict()
        spec_dict["pv_quadrature_per_axis"] = options.n_quadrature
        spec_dict["pv_threshold"] = 0.5
        (specs_dir / f"{primitive.name}.json").write_text(
            json.dumps(spec_dict, indent=2), encoding="utf-8"
        )

    manifest = FixtureManifest(
        manifest_version=MANIFEST_VERSION,
        package_version=__version__,
        geometry={
            "origin": list(geometry.origin),
            "spacing": list(geometry.spacing),
            "size": list(geometry.size),
            "direction": list(geometry.direction),
        },
        rois=list(CONFORMANCE_ROIS),
        expected_predictions=[f"{name}.nii.gz" for name in CONFORMANCE_ROIS],
        rtstruct_path="rtstruct/primitives_planar.dcm",
        rtstruct_sha256=sha256_file(rtss_path),
        groundtruth_dir="groundtruth",
        groundtruth_sha256=groundtruth_sha256,
        refct_dir="refct",
    )
    write_manifest(manifest, out)

    readme_src = Path(str(files("rtmask_conformance").joinpath("data/README_FOR_TOOL_AUTHOR.md")))
    shutil.copyfile(readme_src, out / "README_FOR_TOOL_AUTHOR.md")

    return out
