"""Generate a conformance fixture: synthetic CT + RTSTRUCTs + analytic GT NIfTIs.

The output directory layout matches what ``verify`` (and the consuming tool)
expects:

    <out_dir>/
      refct/                              200 DICOM CT slices
      rtstruct/primitives_<bundle>.dcm    one RTSTRUCT per populated bundle
      groundtruth/<roi>.nii.gz            binary masks (uint8, PV >= 0.5) for
                                          every ROI whose truth_kind is binary_mask:
                                          the reference of the default convention
                                          ``pv_ge_0.5``
      groundtruth_pv/<roi>.nii.gz         the partial-volume field itself (float32,
                                          fraction of each voxel inside the solid)
      groundtruth_pv_gt0/<roi>.nii.gz     uint8, PV > 0: the ``pv_gt_0`` reference
      groundtruth_centre_in/<roi>.nii.gz  uint8, voxel centre inside the solid: the
                                          ``centre_in`` reference
      groundtruth_curves/<roi>.json       analytic truth of each curve ROI: densified
                                          polylines, endpoints, analytic arc length
      groundtruth_points/<roi>.json       analytic truth of each point ROI
      reference_masks/<roi>.nii.gz        for curve and point ROIs, the suite's own
                                          thin-path rasterisation: an example of a
                                          passing mask, not truth, never scored against
      specs/<roi>.json                    analytic spec metadata + bundle fields
      manifest.json                       bundles, ROIs, sha256s, generator settings
      volumes_template.csv                blank input for ``verify --volumes``: one
                                          row per ROI, volume cells empty
      README_FOR_TOOL_AUTHOR.md           drop-in instructions for converters
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from . import __version__
from ._roi_set import BUNDLES
from ._vendor.common.io import sha256_file, write_nifti
from ._vendor.groundtruth.partial_volume import (
    any_overlap_mask,
    binary_threshold,
    partial_volume_mask,
)
from ._vendor.refimage.build_reference_ct import ReferenceCTSpec, build_reference_ct
from ._vendor.rtstruct.pydicom_writer import build_rtstruct
from .centre_in import centre_in_mask
from .grading_pointline import (
    POINTLINE_TRUTH_KINDS,
    TRUTH_DIRS,
    build_truth,
    reference_rasterisation,
    write_truth,
)
from .manifest import (
    MANIFEST_VERSION,
    BundleEntry,
    FixtureManifest,
    RoiEntry,
    package_git_sha,
    write_manifest,
)
from .thresholds import DEFAULT_CONVENTION
from .volume_only import write_volumes_template

PV_THRESHOLD: float = 0.5

#: Fixture directory of the reference masks of curve and point ROIs.
REFERENCE_MASKS_DIR: str = "reference_masks"

#: Fixture directory of the float partial-volume field of each binary-mask ROI.
PV_FIELD_DIR: str = "groundtruth_pv"

#: Fixture directory of each boundary convention's binary reference.
CONVENTION_DIRS: dict[str, str] = {
    DEFAULT_CONVENTION: "groundtruth",
    "pv_gt_0": "groundtruth_pv_gt0",
    "centre_in": "groundtruth_centre_in",
}


@dataclass(frozen=True)
class GenerateOptions:
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[int, int, int] = (512, 512, 200)
    n_quadrature: int = 8


def rtstruct_relpath(bundle: str) -> str:
    """Fixture-relative path of a bundle's RTSTRUCT."""
    return f"rtstruct/primitives_{bundle}.dcm"


def _write_mask(array, geometry, out: Path, rel: str) -> dict[str, str]:
    write_nifti(array, geometry, out / rel)
    return {"path": rel, "sha256": sha256_file(out / rel)}


def generate_fixture(out_dir: str | Path, options: GenerateOptions | None = None) -> Path:
    """Build a complete fixture under ``out_dir`` and return the directory path.

    Idempotent: re-running with the same options overwrites existing files
    deterministically (UIDs are derived from a fixed salt, voxel content is a
    pure function of geometry). The RTSTRUCT's creation date and time are the
    only wall-clock values.
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

    bundle_entries: dict[str, BundleEntry] = {}
    roi_entries: dict[str, RoiEntry] = {}
    for bundle, rois in BUNDLES.items():
        rtss_rel = rtstruct_relpath(bundle)
        rtss_path = out / rtss_rel
        build_rtstruct(
            primitives=[roi.primitive for roi in rois],
            ref_image_folder=refct_dir,
            out_path=rtss_path,
            structure_set_label="RTMASK_CONFORM",
        )
        bundle_entries[bundle] = BundleEntry(
            rtstruct_path=rtss_rel,
            rtstruct_sha256=sha256_file(rtss_path),
            rois=[roi.name for roi in rois],
        )

        for roi in rois:
            truth_rel: str | None = None
            truth_sha: str | None = None
            reference_rel: str | None = None
            pv_field: dict[str, str] | None = None
            convention_truths: dict[str, dict[str, str]] | None = None
            if roi.truth_kind == "binary_mask":
                # One partial-volume field per ROI; the pv_ge_0.5 and pv_gt_0
                # references are its two binarisations, centre_in is the direct
                # inside test at the voxel centres.
                fractions = partial_volume_mask(
                    roi.primitive, geometry, n_quadrature=options.n_quadrature
                )
                name = f"{roi.name}.nii.gz"
                convention_truths = {
                    DEFAULT_CONVENTION: _write_mask(
                        binary_threshold(fractions, threshold=PV_THRESHOLD), geometry, out,
                        f"{CONVENTION_DIRS[DEFAULT_CONVENTION]}/{name}",
                    ),
                    "pv_gt_0": _write_mask(
                        any_overlap_mask(fractions), geometry, out,
                        f"{CONVENTION_DIRS['pv_gt_0']}/{name}",
                    ),
                    "centre_in": _write_mask(
                        centre_in_mask(roi.primitive, geometry), geometry, out,
                        f"{CONVENTION_DIRS['centre_in']}/{name}",
                    ),
                }
                pv_field = _write_mask(
                    fractions.astype("float32"), geometry, out, f"{PV_FIELD_DIR}/{name}"
                )
                truth_rel = convention_truths[DEFAULT_CONVENTION]["path"]
                truth_sha = convention_truths[DEFAULT_CONVENTION]["sha256"]
            elif roi.truth_kind in POINTLINE_TRUTH_KINDS:
                truth = build_truth(roi.primitive, roi.truth_kind, geometry)
                truth_rel = f"{TRUTH_DIRS[roi.truth_kind]}/{roi.name}.json"
                write_truth(truth, out / truth_rel)
                truth_sha = sha256_file(out / truth_rel)
                reference_rel = f"{REFERENCE_MASKS_DIR}/{roi.name}.nii.gz"
                write_nifti(reference_rasterisation(truth), geometry, out / reference_rel)
            elif roi.truth_kind != "none":
                raise NotImplementedError(
                    f"{roi.name}: no ground-truth writer for truth_kind {roi.truth_kind!r}"
                )

            spec_dict = roi.primitive.to_spec_dict()
            if roi.truth_kind == "binary_mask":
                spec_dict["pv_quadrature_per_axis"] = options.n_quadrature
                spec_dict["pv_threshold"] = PV_THRESHOLD
            spec_dict["bundle"] = bundle
            spec_dict["geometric_type"] = roi.geometric_type.value
            spec_dict["truth_kind"] = roi.truth_kind
            spec_dict["graded"] = roi.graded
            spec_rel = f"specs/{roi.name}.json"
            (out / spec_rel).write_text(json.dumps(spec_dict, indent=2), encoding="utf-8")

            roi_entries[roi.name] = RoiEntry(
                bundle=bundle,
                geometric_type=roi.geometric_type.value,
                truth_kind=roi.truth_kind,
                graded=roi.graded,
                truth_path=truth_rel,
                truth_sha256=truth_sha,
                spec_path=spec_rel,
                expected_prediction=f"{roi.name}.nii.gz" if roi.graded else None,
                reference_mask_path=reference_rel,
                pv_field=pv_field,
                convention_truths=convention_truths,
            )

    manifest = FixtureManifest(
        manifest_version=MANIFEST_VERSION,
        package_version=__version__,
        git_sha=package_git_sha(),
        n_quadrature=options.n_quadrature,
        pv_threshold=PV_THRESHOLD,
        geometry={
            "origin": list(geometry.origin),
            "spacing": list(geometry.spacing),
            "size": list(geometry.size),
            "direction": list(geometry.direction),
        },
        bundles=bundle_entries,
        rois=roi_entries,
        groundtruth_dir="groundtruth",
        refct_dir="refct",
    )
    write_manifest(manifest, out)
    write_volumes_template(out, manifest)

    readme_src = Path(str(files("rtmask_conformance").joinpath("data/README_FOR_TOOL_AUTHOR.md")))
    shutil.copyfile(readme_src, out / "README_FOR_TOOL_AUTHOR.md")

    return out
