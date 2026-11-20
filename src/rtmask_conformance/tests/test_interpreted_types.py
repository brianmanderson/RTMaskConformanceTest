"""RT ROI Interpreted Type (3006,00A4) and ROI Description (3006,0028) per ROI.

The mapping covers every ROI of every bundle with a Defined Term and a
description of 16 characters or fewer. The post-write pass changes only those
two attributes: contour data, ROI names and numbers, UIDs and the transfer
syntax are untouched, and undoing the two attributes restores the writer's
bytes exactly. The fixture stays deterministic: two generations are
byte-identical.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shutil
import types
from pathlib import Path

import pydicom
import pytest

from rtmask_conformance import BUNDLES
from rtmask_conformance._vendor.primitives.closed_planar import Sphere
from rtmask_conformance._vendor.refimage.build_reference_ct import (
    ReferenceCTSpec,
    build_reference_ct,
)
from rtmask_conformance._vendor.rtstruct import pydicom_writer
from rtmask_conformance._vendor.rtstruct.pydicom_writer import build_rtstruct
from rtmask_conformance.generate import GenerateOptions, generate_fixture, rtstruct_relpath
from rtmask_conformance.interpreted_types import (
    DEFINED_TERMS,
    MAX_DESCRIPTION_LENGTH,
    ROI_LABELS,
    RoiLabel,
    UnmappedRoiError,
    apply_interpreted_types,
    label_for,
)
from rtmask_conformance.manifest import read_manifest

ALL_ROIS: list[str] = [roi.name for rois in BUNDLES.values() for roi in rois]

#: The two attributes the pass writes; nothing else may change.
INTERPRETED_TYPE = pydicom.tag.Tag(0x3006, 0x00A4)
ROI_DESCRIPTION = pydicom.tag.Tag(0x3006, 0x0028)

_SKIP_E2E = os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1"


class _FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
#  The mapping
# --------------------------------------------------------------------------- #


def test_defined_terms_are_the_25_of_ps3_3() -> None:
    assert len(DEFINED_TERMS) == 25 == len(set(DEFINED_TERMS))


def test_every_roi_of_every_bundle_is_mapped() -> None:
    assert sorted(ROI_LABELS) == sorted(ALL_ROIS), "mapping and bundles disagree"
    assert list(ROI_LABELS) == ALL_ROIS, "keep ROI_LABELS in canonical bundle order"


@pytest.mark.parametrize("roi", ALL_ROIS)
def test_label_is_a_defined_term_with_a_short_description(roi: str) -> None:
    label = label_for(roi)
    assert label.interpreted_type in DEFINED_TERMS
    assert 0 < len(label.description) <= MAX_DESCRIPTION_LENGTH
    assert label.description.isascii() and " " not in label.description


def test_descriptions_are_unique() -> None:
    descriptions = [label.description for label in ROI_LABELS.values()]
    assert len(descriptions) == len(set(descriptions))


def test_brief_classes() -> None:
    """The primitive-to-class rules: solids ORGAN, cube PTV, torus AVOIDANCE,
    points MARKER or BRACHY_SRC_APP, curves BRACHY_CHANNEL, open-planar MARKER."""
    expected = {
        "sphere": "ORGAN", "cube": "PTV", "cylinder": "ORGAN", "ellipsoid": "ORGAN",
        "torus": "AVOIDANCE", "hollow_sphere": "ORGAN", "straw": "ORGAN",
        "hollow_sphere_xor": "ORGAN", "straw_xor": "ORGAN",
        "hollow_sphere_keyhole": "ORGAN", "straw_keyhole": "ORGAN",
        "single_point_x256_y256_z150": "MARKER",
        "needle_track_10pts_x150_y150": "MARKER",
        "ldr_seed_implant_x350_y350": "BRACHY_SRC_APP",
        "diagonal_needle_x100_y200_z40": "BRACHY_CHANNEL",
        "helix_r12_pitch6_4turns_x300_y200": "BRACHY_CHANNEL",
        "sinusoid_x150_y350_z20to80": "BRACHY_CHANNEL",
        "syed_bundle_5needles_x400_y100_z50": "BRACHY_CHANNEL",
        "line_segment_x200_y200_to_x300_y280_z175": "MARKER",
        "open_arc_r40_90deg_x380_y300_z180": "MARKER",
    }
    assert {roi: label.interpreted_type for roi, label in ROI_LABELS.items()} == expected


def test_label_validation() -> None:
    with pytest.raises(ValueError, match="Defined Term"):
        RoiLabel("TARGET", "Sphere_r40")
    with pytest.raises(ValueError, match="characters"):
        RoiLabel("ORGAN", "A" * (MAX_DESCRIPTION_LENGTH + 1))
    with pytest.raises(ValueError, match="characters"):
        RoiLabel("ORGAN", "")
    with pytest.raises(UnmappedRoiError, match="no_such_roi"):
        label_for("no_such_roi")


# --------------------------------------------------------------------------- #
#  The pass, on each bundle's RTSTRUCT as the vendored writer leaves it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Path]]:
    """Per bundle: (the writer's RTSTRUCT, a copy with the pass applied)."""
    if _SKIP_E2E:
        pytest.skip("RTMASK_CONFORMANCE_SKIP_E2E=1 set")
    root = tmp_path_factory.mktemp("interpreted_types")
    refct = root / "refct"
    build_reference_ct(refct, spec=ReferenceCTSpec(voxel_size=(1.0, 1.0, 1.0),
                                                   size=(512, 512, 200)))
    out: dict[str, tuple[Path, Path]] = {}
    for bundle, rois in BUNDLES.items():
        raw = root / "raw" / f"primitives_{bundle}.dcm"
        build_rtstruct(primitives=[r.primitive for r in rois], ref_image_folder=refct,
                       out_path=raw, structure_set_label="RTMASK_CONFORM")
        labelled = root / "labelled" / raw.name
        labelled.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw, labelled)
        apply_interpreted_types(labelled)
        out[bundle] = (raw, labelled)
    return out


def _without_labels(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Undo the pass in memory: empty Interpreted Type, no ROI Description."""
    for item in ds.StructureSetROISequence:
        if ROI_DESCRIPTION in item:
            del item[ROI_DESCRIPTION]
    for obs in ds.RTROIObservationsSequence:
        obs.RTROIInterpretedType = ""
    return ds


@pytest.mark.parametrize("bundle", list(BUNDLES))
def test_pass_writes_both_attributes_for_every_roi(
    written: dict[str, tuple[Path, Path]], bundle: str
) -> None:
    raw, labelled = written[bundle]
    before = pydicom.dcmread(raw)
    assert all(obs.RTROIInterpretedType == "" for obs in before.RTROIObservationsSequence)
    assert all(ROI_DESCRIPTION not in item for item in before.StructureSetROISequence)

    ds = pydicom.dcmread(labelled)
    names = {item.ROINumber: item.ROIName for item in ds.StructureSetROISequence}
    assert list(names.values()) == [roi.name for roi in BUNDLES[bundle]], "ROI names changed"
    for item in ds.StructureSetROISequence:
        assert item.ROIDescription == ROI_LABELS[item.ROIName].description
    assert len(ds.RTROIObservationsSequence) == len(names)
    for obs in ds.RTROIObservationsSequence:
        assert obs.RTROIInterpretedType == ROI_LABELS[names[obs.ReferencedROINumber]].interpreted_type
        assert obs.RTROIInterpretedType in DEFINED_TERMS


@pytest.mark.parametrize("bundle", list(BUNDLES))
def test_pass_leaves_contours_uids_and_transfer_syntax_unchanged(
    written: dict[str, tuple[Path, Path]], bundle: str
) -> None:
    raw, labelled = written[bundle]
    before = pydicom.dcmread(raw)
    after = pydicom.dcmread(labelled)
    for keyword in ("SOPInstanceUID", "SOPClassUID", "SeriesInstanceUID", "StudyInstanceUID",
                    "FrameOfReferenceUID"):
        assert after[keyword].value == before[keyword].value, keyword
    assert after.file_meta == before.file_meta
    assert after.file_meta.TransferSyntaxUID == before.file_meta.TransferSyntaxUID
    assert after.ROIContourSequence == before.ROIContourSequence
    assert after.ReferencedFrameOfReferenceSequence == before.ReferencedFrameOfReferenceSequence
    # Contour data byte for byte, as stored.
    for b_roi, a_roi in zip(before.ROIContourSequence, after.ROIContourSequence, strict=True):
        for b, a in zip(b_roi.ContourSequence, a_roi.ContourSequence, strict=True):
            assert a.get_item(0x30060050).value == b.get_item(0x30060050).value


@pytest.mark.parametrize("bundle", list(BUNDLES))
def test_pass_changes_only_the_two_attributes(
    written: dict[str, tuple[Path, Path]], bundle: str, tmp_path: Path
) -> None:
    """Undoing the two attributes and re-saving gives the writer's bytes exactly."""
    raw, labelled = written[bundle]
    assert labelled.read_bytes() != raw.read_bytes()
    undone = tmp_path / "undone.dcm"
    _without_labels(pydicom.dcmread(labelled)).save_as(undone, enforce_file_format=True)
    assert undone.read_bytes() == raw.read_bytes()


def test_pass_is_idempotent(written: dict[str, tuple[Path, Path]], tmp_path: Path) -> None:
    _, labelled = written["planar"]
    again = tmp_path / labelled.name
    shutil.copyfile(labelled, again)
    apply_interpreted_types(again)
    assert again.read_bytes() == labelled.read_bytes()


def test_unmapped_roi_is_an_error_and_leaves_the_file(
    written: dict[str, tuple[Path, Path]], tmp_path: Path
) -> None:
    refct = written["planar"][0].parents[1] / "refct"
    path = tmp_path / "unmapped.dcm"
    build_rtstruct(
        primitives=[BUNDLES["planar"][0].primitive,
                    Sphere(name="new_sphere", center=(256.0, 256.0, 100.0), radius=10.0)],
        ref_image_folder=refct, out_path=path,
    )
    before = path.read_bytes()
    with pytest.raises(UnmappedRoiError, match="new_sphere"):
        apply_interpreted_types(path)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------- #
#  The generated fixture
# --------------------------------------------------------------------------- #


def _generate_frozen(out: Path) -> Path:
    mp = pytest.MonkeyPatch()
    mp.setattr(pydicom_writer, "datetime", types.SimpleNamespace(datetime=_FrozenDateTime))
    try:
        # Determinism does not depend on the quadrature; one sample per axis keeps it quick.
        return generate_fixture(out, options=GenerateOptions(n_quadrature=1))
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def two_generations(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    if _SKIP_E2E:
        pytest.skip("RTMASK_CONFORMANCE_SKIP_E2E=1 set")
    root = tmp_path_factory.mktemp("determinism")
    return _generate_frozen(root / "a"), _generate_frozen(root / "b")


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_two_generations_are_byte_identical(two_generations: tuple[Path, Path]) -> None:
    a, b = two_generations
    tree_a, tree_b = _tree(a), _tree(b)
    assert sorted(tree_a) == sorted(tree_b)
    differing = sorted(rel for rel in tree_a if tree_a[rel] != tree_b[rel])
    assert not differing, f"files differ between two generations: {differing}"


def test_fixture_rtstructs_and_manifest_carry_the_labels(
    two_generations: tuple[Path, Path],
) -> None:
    fixture = two_generations[0]
    manifest = read_manifest(fixture)
    assert list(manifest.rois) == ALL_ROIS
    for roi, entry in manifest.rois.items():
        assert entry.interpreted_type == ROI_LABELS[roi].interpreted_type, roi
        assert entry.description == ROI_LABELS[roi].description, roi
    for bundle in BUNDLES:
        ds = pydicom.dcmread(fixture / rtstruct_relpath(bundle))
        names = {item.ROINumber: item.ROIName for item in ds.StructureSetROISequence}
        for item in ds.StructureSetROISequence:
            assert item.ROIDescription == manifest.rois[item.ROIName].description
        for obs in ds.RTROIObservationsSequence:
            entry = manifest.rois[names[obs.ReferencedROINumber]]
            assert obs.RTROIInterpretedType == entry.interpreted_type
        # The manifest hash is of the labelled file.
        assert manifest.bundles[bundle].rtstruct_sha256 == _sha256(
            fixture / rtstruct_relpath(bundle))
