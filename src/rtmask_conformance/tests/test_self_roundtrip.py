"""Self-roundtrip: rasterize the analytic primitives via the same vendored
voxelizer that built the ground truth, and confirm verify() reports Dice = 1.0.

This is the suite's own correctness test — if it ever fails, ``verify.py``
or the vendored metric/voxelizer code has a bug. It runs end-to-end
(``generate`` → re-rasterize → ``verify``) on a tiny grid so it stays under a
few seconds in CI.

The second half covers the v0.2 fixture layout on one shared fixture: the
``planar`` bundle is byte-identical to v0.1, the manifest is v2, ``verify
--fixture`` passes 7/7 on the ground truth, and tampered ground truth is
caught. Its content (voxels, geometry, contour structure exactly; contour
coordinates within a stated tolerance) is also checked against one
Windows-taken reference on every platform; ``data/planar_content_reference.npz``
holds the data that lets a mismatch name the differing voxels or contours.

Skipped by default when ``RTMASK_CONFORMANCE_SKIP_E2E=1`` is set; otherwise
runs unconditionally.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from rtmask_conformance import BUNDLES, CONFORMANCE_ROIS, load_config
from rtmask_conformance._roi_set import build_conformance_primitives
from rtmask_conformance._vendor.common.io import sitk_image_to_geometry, write_nifti
from rtmask_conformance._vendor.groundtruth.partial_volume import (
    binary_threshold,
    partial_volume_mask,
)
from rtmask_conformance._vendor.primitives.closed_planar import (
    Cube,
    Cylinder,
    Ellipsoid,
    Sphere,
    Torus,
)
from rtmask_conformance._vendor.primitives.closed_planar_xor import HollowCylinder, HollowSphere
from rtmask_conformance._vendor.refimage.build_reference_ct import (
    ReferenceCTSpec,
    build_reference_ct,
)
from rtmask_conformance._vendor.rtstruct import pydicom_writer
from rtmask_conformance._vendor.rtstruct.pydicom_writer import build_rtstruct
from rtmask_conformance.cli import main as cli_main
from rtmask_conformance.generate import GenerateOptions, generate_fixture
from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.verify import (
    REPORT_VERSION,
    FixtureIntegrityError,
    Status,
    verify_fixture,
    verify_predictions,
)

# SHA-256 of the v0.1 (0.1.0, commit 27c3c47) planar fixture at --n-quadrature 2
# on the default 512x512x200 1 mm grid, per platform. The RTSTRUCT writer stamps
# the wall clock into InstanceCreation/StructureSet Date and Time, so the RTSTRUCT
# hash is taken with that clock frozen at 2026-01-01 00:00:00; nothing else in
# the fixture depends on the clock. The bytes differ between Windows and Linux
# (the same on Python 3.10-3.12 within each), so there is one list per platform;
# the platform-independent proof is the in-process comparison against
# _generate_v01_reference below.
V01_PLANAR_SHA256_BY_PLATFORM: dict[str, dict[str, str]] = {
    "win32": {
        "rtstruct/primitives_planar.dcm":
            "3ac34957d99bd04ad59768b8c5d1a93db132876ff0c4d265ec976a2f7cf09e0c",
        "groundtruth/cube.nii.gz":
            "489e7c2de06989ff1410d9f809886c116bebba1a6a150f7ce6e993a99fe52c81",
        "groundtruth/cylinder.nii.gz":
            "971e6fa0378ca3f4ed62bcf619e6f8587d399442d3306fc32abe5172c07b3345",
        "groundtruth/ellipsoid.nii.gz":
            "ed54bd2993ccd4ff28dafa00a67a82c21307a3ef38c6511e31df206e1694e5b5",
        "groundtruth/hollow_sphere.nii.gz":
            "c01ebdfde7355963ba79021ab54b65dca5f762aff55031beb562b5ac97119d8b",
        "groundtruth/sphere.nii.gz":
            "437be70ae95b68d0e3c85ed384d183c3cc6864b6640b90261cf1c52ccb7840e2",
        "groundtruth/straw.nii.gz":
            "22e16ba1a518bc29672fc192abb0f0183ca88dc3c657e9896c6e2f9692885c20",
        "groundtruth/torus.nii.gz":
            "12d1f33dfed11d5fc4a6de0980f82634254018eb3ba0effc4dc2333d20ccc31f",
    },
    "linux": {
        "rtstruct/primitives_planar.dcm":
            "096d4e8b15b0b12a1b55d79f614945dd01b5724dd6e84b8f6b07a350f73634f9",
        "groundtruth/cube.nii.gz":
            "81fba83d77595c0264b30d07c7693475977be9e17cb9810592a6d1b10f33ae54",
        "groundtruth/cylinder.nii.gz":
            "77b4bfb24532851049392a9d3dbe5b5a770625cc4bca1e4b34eb60ed7eee5902",
        "groundtruth/ellipsoid.nii.gz":
            "5b4095a9c48cb1fab06ce70f7dde842cbc52f539c07d7d8fb77c033fce25a5f4",
        "groundtruth/hollow_sphere.nii.gz":
            "f6c198042ebc083f0233547692de08ce522add0b5768735ec02ab33eea5e6023",
        "groundtruth/sphere.nii.gz":
            "33824dd8a03dc897ef0c7335113ee688844bdffc360ab7cda490944685d49b23",
        "groundtruth/straw.nii.gz":
            "77d4af876687b5194bb8010caafbef5fd394312c247f04290e9c9c9cf1b04bdd",
        "groundtruth/torus.nii.gz":
            "2b6699643048ab53f82e1daf700d4dcf735f1454a4db1572f5fb83078dfbb1e7",
    },
}

V01_PLANAR_FILES: list[str] = sorted(V01_PLANAR_SHA256_BY_PLATFORM["win32"])

# SHA-256 of the planar bundle's CONTENT (same fixture as above), one list for
# every platform, taken on Windows. "mask" hashes the decoded voxel array as
# uint8 plus the image geometry and pixel type; "contour_structure" hashes the
# ROI name and, per contour, its geometric type and point count. The contour
# coordinates are compared point by point against the reference file below,
# within CONTOUR_TOLERANCE_MM. See _planar_content_fingerprint.
PLANAR_CONTENT_SHA256: dict[str, dict[str, str]] = {
    "sphere": {
        "mask": "cb55242f02294121b7e169d9676464681676906b395af2b528ee5e09ad69b57a",
        "contour_structure":
            "6fa4decd0c8b9da455b4773939b74c4aa99125b123924a0aeaacde246a46ecf7",
    },
    "cube": {
        "mask": "8b5f4a4721b308211dee04a0165e0925ce58914a39aa6bfcbfa42e0830699fdc",
        "contour_structure":
            "1f713e6cdaa3c1ef5815b4607c307c3c34708dde37b074010e885fff95912408",
    },
    "cylinder": {
        "mask": "94bddea794bb910a6bea5bf46b4d54aafc349044e1635cc19ff43b25dfb6b177",
        "contour_structure":
            "c6f5a0643fdc9cca416a84765c04d8ebe6c399964b990b6d1f9fe721fc81ae35",
    },
    "ellipsoid": {
        "mask": "25dc408ad08b06091c9eaf2b45b58941051913a6534375be7f4a234f10ffd22a",
        "contour_structure":
            "b174dc34a66105778119d4f90371ab243af8c948c20e36df106c8614e62c7e74",
    },
    "torus": {
        "mask": "33b834c8f9ca26c8effe9e2026ad8c72260fafbcfaff7c90ce45237dd6aa4c50",
        "contour_structure":
            "6644847e9b6f9b585814c434e424b9c8edbe9c3975dfee934bcf9b05b7b5017a",
    },
    "hollow_sphere": {
        "mask": "4e4a92e170784c71b5e8130929c62b822c7a4e90e300dcae4f1db7798cddb00f",
        "contour_structure":
            "4669ba07f404081d2c0dd7695ce9a51eb210832596327c8560d2c8f619ff553e",
    },
    "straw": {
        "mask": "bc342d072eb48c006a374c536910912562c0e09609e8eaa9425893c734287123",
        "contour_structure":
            "c05eed9c46f7039b0ca3900314d3b3718b5bedc531e8352be7a8a209a7b82359",
    },
}

# Windows-side diagnostic data for test_planar_bundle_content_identical_across_platforms:
# run-length-encoded masks (so a mismatch can name the differing voxels),
# per-contour digests, and serialisation-layer digests (gzip header, decompressed
# NIfTI bytes, raw ContourData strings, top-level DICOM elements).
PLANAR_CONTENT_REFERENCE = Path(__file__).parent / "data" / "planar_content_reference.npz"

# Absolute tolerance for contour coordinates across platforms. The vendored
# RTSTRUCT writer emits ContourData with up to 16 significant digits, and libm
# differences between platforms reach the file at the 1e-12 mm level (the last
# printed digit of a few coordinates). Once the writer is re-vendored with
# 12-significant-digit ContourData, this comparison returns to exact.
CONTOUR_TOLERANCE_MM = 1e-9


class _FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_clock() -> pytest.MonkeyPatch:
    mp = pytest.MonkeyPatch()
    mp.setattr(pydicom_writer, "datetime", types.SimpleNamespace(datetime=_FrozenDateTime))
    return mp


def _generate_v01_reference(out: Path, n_quadrature: int) -> None:
    """The v0.1 generator's RTSTRUCT and ground-truth steps, frozen as of 27c3c47.

    Kept independent of ``_roi_set`` and ``generate`` on purpose: the v0.2
    generator must reproduce these bytes on whatever platform the test runs.
    """
    primitives = [
        Sphere(name="sphere", center=(256.0, 256.0, 100.0), radius=40.0),
        Cube(name="cube", center=(100.5, 100.5, 100.5), side_length=60.0),
        Cylinder(name="cylinder", center=(400.0, 100.0, 100.5), radius=30.0, height=80.0),
        Ellipsoid(name="ellipsoid", center=(100.0, 400.0, 100.0), semi_axes=(30.0, 50.0, 60.0)),
        Torus(name="torus", center=(400.0, 400.0, 100.0), major_radius=60.0, minor_radius=20.0),
        HollowSphere(name="hollow_sphere", center=(256.0, 100.0, 100.0),
                     outer_radius=40.0, inner_radius=20.0),
        HollowCylinder(name="straw", center=(256.0, 400.0, 100.5),
                       outer_radius=40.0, inner_radius=20.0, height=120.0),
    ]
    refct_dir = out / "refct"
    for d in (refct_dir, out / "rtstruct", out / "groundtruth"):
        d.mkdir(parents=True, exist_ok=True)
    geometry = build_reference_ct(
        refct_dir, spec=ReferenceCTSpec(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200))
    )
    build_rtstruct(
        primitives=primitives,
        ref_image_folder=refct_dir,
        out_path=out / "rtstruct" / "primitives_planar.dcm",
        structure_set_label="RTMASK_CONFORM",
    )
    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        write_nifti(binary, geometry, out / "groundtruth" / f"{primitive.name}.nii.gz")


def _mask_runs(flat: np.ndarray) -> np.ndarray:
    """(start, stop) pairs of the foreground runs of a flattened binary mask."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flat.astype(np.int8), [0]))))
    return edges.astype(np.int32).reshape(-1, 2)


def _mask_from_runs(runs: np.ndarray, n: int) -> np.ndarray:
    flat = np.zeros(n, dtype=np.uint8)
    for start, stop in runs:
        flat[start:stop] = 1
    return flat


def _contour_lines(rtstruct: pydicom.Dataset) -> dict[str, list[str]]:
    """Per ROI name, one line per contour: type, point count, ContourData floats (repr)."""
    names = {item.ROINumber: item.ROIName for item in rtstruct.StructureSetROISequence}
    lines: dict[str, list[str]] = {}
    for roi_contour in rtstruct.ROIContourSequence:
        out = lines.setdefault(names[roi_contour.ReferencedROINumber], [])
        for contour in roi_contour.ContourSequence:
            floats = " ".join(repr(float(v)) for v in contour.ContourData)
            out.append(f"{contour.ContourGeometricType}|{contour.NumberOfContourPoints}|{floats}")
    return lines


def _planar_content_fingerprint(fixture: Path) -> dict:
    """Content hashes of the planar bundle plus the data to diagnose a mismatch."""
    rtstruct_path = fixture / "rtstruct" / "primitives_planar.dcm"
    lines = _contour_lines(pydicom.dcmread(rtstruct_path))
    content: dict[str, dict[str, str]] = {}
    masks: dict[str, np.ndarray] = {}
    geometry: dict[str, str] = {}
    serialisation: dict[str, str] = {}
    for roi in CONFORMANCE_ROIS:
        path = fixture / "groundtruth" / f"{roi}.nii.gz"
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        geometry[roi] = repr((img.GetOrigin(), img.GetSpacing(), img.GetDirection(),
                              img.GetSize(), img.GetPixelIDTypeAsString()))
        masks[roi] = arr
        h = hashlib.sha256(arr.tobytes())
        h.update(geometry[roi].encode())
        structure = "\n".join([roi, *(line.rsplit("|", 1)[0] for line in lines.get(roi, []))])
        content[roi] = {
            "mask": h.hexdigest(),
            "contour_structure": hashlib.sha256(structure.encode()).hexdigest(),
        }
        raw = path.read_bytes()
        serialisation[f"gzip_header/{roi}"] = raw[:10].hex()
        serialisation[f"nifti_decompressed/{roi}"] = hashlib.sha256(
            gzip.decompress(raw)).hexdigest()
    # Serialisation layer of the RTSTRUCT, on a fresh read so elements are still raw.
    ds = pydicom.dcmread(rtstruct_path)
    for name, dataset in (("meta", ds.file_meta), ("dataset", ds)):
        for tag in dataset.keys():
            elem = dataset.get_item(tag)
            value = elem.value if isinstance(elem.value, bytes) else repr(elem.value).encode()
            serialisation[f"dicom_{name}/{tag}"] = hashlib.sha256(value).hexdigest()
    names = {item.ROINumber: item.ROIName for item in ds.StructureSetROISequence}
    contour_ds: dict[str, list[str]] = {}
    for roi_contour in ds.ROIContourSequence:
        roi = names[roi_contour.ReferencedROINumber]
        h = hashlib.sha256()
        for contour in roi_contour.ContourSequence:
            raw = contour.get_item(0x30060050).value
            h.update(raw)
            contour_ds.setdefault(roi, []).append(raw.decode("ascii").strip())
        serialisation[f"contour_ds_strings/{roi}"] = h.hexdigest()
    return {
        "content": content,
        "masks": masks,
        "geometry": geometry,
        "contour_digests": {
            roi: [hashlib.sha256(line.encode()).hexdigest()[:16] for line in lines.get(roi, [])]
            for roi in CONFORMANCE_ROIS
        },
        "contour_ds": contour_ds,
        "serialisation": serialisation,
    }


def _write_content_reference(fingerprint: dict, out: Path) -> None:
    """Write the Windows-side diagnostic data (run once by hand, on Windows)."""
    arrays = {}
    for roi, arr in fingerprint["masks"].items():
        arrays[f"mask_runs/{roi}"] = _mask_runs(arr.ravel())
        arrays[f"mask_shape/{roi}"] = np.array(arr.shape, dtype=np.int64)
        arrays[f"geometry/{roi}"] = np.array(fingerprint["geometry"][roi])
        arrays[f"contour_digests/{roi}"] = np.array(fingerprint["contour_digests"][roi])
    for key, value in fingerprint["serialisation"].items():
        arrays[f"serialisation/{key}"] = np.array(value)
    # Raw ContourData strings of every contour, xz-compressed (about 130 KB).
    blob = "\n".join(
        f"{roi}\t{i}\t{raw}"
        for roi, contours in fingerprint["contour_ds"].items()
        for i, raw in enumerate(contours)
    )
    arrays["contour_ds_xz"] = np.frombuffer(lzma.compress(blob.encode("ascii")), dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)


def _diagnose_content_mismatch(roi: str, kind: str, fingerprint: dict, ref) -> str:
    """Name what differs for one ROI, using the Windows-side reference data."""
    if ref is None:
        return f"{roi}: {kind} content differs (no reference data to localise it)"
    if kind == "mask":
        expected_geometry = str(ref[f"geometry/{roi}"])
        if fingerprint["geometry"][roi] != expected_geometry:
            return (f"{roi}: geometry/pixel type differs: {fingerprint['geometry'][roi]} "
                    f"!= {expected_geometry}")
        arr = fingerprint["masks"][roi]
        expected = _mask_from_runs(ref[f"mask_runs/{roi}"], arr.size).reshape(
            tuple(ref[f"mask_shape/{roi}"]))
        diff = np.argwhere(arr != expected)
        first = [(tuple(int(i) for i in idx), int(expected[tuple(idx)]), int(arr[tuple(idx)]))
                 for idx in diff[:5]]
        return (f"{roi}: {len(diff)} voxel(s) differ (expected {int(expected.sum())} "
                f"foreground, got {int(arr.sum())}); first (z, y, x), expected, got: {first}")
    expected_ds = _reference_contour_ds(ref).get(roi, [])
    actual_ds = fingerprint["contour_ds"].get(roi, [])
    counts = [(i, len(a.split("\\")) // 3, len(e.split("\\")) // 3)
              for i, (a, e) in enumerate(zip(actual_ds, expected_ds, strict=False))]
    differing = [(i, a, e) for i, a, e in counts if a != e]
    return (f"{roi}: contour structure differs: {len(actual_ds)} contour(s), expected "
            f"{len(expected_ds)}; point counts (contour, got, expected) differ on "
            f"{differing[:5]}{' ...' if len(differing) > 5 else ''} (or a geometric type differs)")


def _compare_contour_coordinates(fingerprint: dict, ref) -> tuple[list[str], dict]:
    """Compare every contour's coordinates with the reference, per point.

    Returns the contours whose coordinates differ by more than CONTOUR_TOLERANCE_MM
    (described in full) and statistics on the exact DS-string differences.
    """
    problems: list[str] = []
    stats = {"strings": 0, "contours": 0, "max_mm": 0.0}
    expected_by_roi = _reference_contour_ds(ref)
    for roi in CONFORMANCE_ROIS:
        expected_ds = expected_by_roi.get(roi, [])
        for i, (actual_raw, expected_raw) in enumerate(
                zip(fingerprint["contour_ds"].get(roi, []), expected_ds, strict=False)):
            if actual_raw == expected_raw:
                continue
            actual = actual_raw.split("\\")
            expected = expected_raw.split("\\")
            if len(actual) != len(expected):
                continue  # a point-count difference; the structure hash reports it
            strings = sum(a != e for a, e in zip(actual, expected, strict=True))
            max_mm = max(abs(float(a) - float(e)) for a, e in zip(actual, expected, strict=True))
            stats["strings"] += strings
            stats["contours"] += 1
            stats["max_mm"] = max(stats["max_mm"], max_mm)
            if max_mm > CONTOUR_TOLERANCE_MM:
                problems.append(f"{roi}: " + _describe_contour_difference(
                    i, expected_raw, actual_raw) + f" (tolerance {CONTOUR_TOLERANCE_MM!r} mm)")
    return problems, stats


def _reference_contour_ds(ref) -> dict[str, list[str]]:
    if "contour_ds_xz" not in ref.files:
        return {}
    out: dict[str, list[str]] = {}
    for line in lzma.decompress(ref["contour_ds_xz"].tobytes()).decode("ascii").splitlines():
        roi, _, raw = line.split("\t")
        out.setdefault(roi, []).append(raw)
    return out


def _describe_contour_difference(index: int, expected_raw: str, actual_raw: str) -> str:
    """Point counts, differing DS strings, max |difference| in mm, first differing point."""
    expected = expected_raw.split("\\")
    actual = actual_raw.split("\\")
    n = min(len(expected), len(actual))
    differing = [k for k in range(n) if expected[k] != actual[k]]
    head = (f"contour {index}: {len(actual) // 3} point(s), expected {len(expected) // 3}; "
            f"{len(differing)} of {n} DS string(s) differ")
    if not differing:
        return head
    max_diff = max(abs(float(expected[k]) - float(actual[k])) for k in differing)
    p = differing[0] // 3
    first_expected = expected[3 * p:3 * p + 3]
    first_actual = actual[3 * p:3 * p + 3]
    return (f"{head}; max |difference| {max_diff!r} mm; first differing point {p}: "
            f"expected {tuple(float(v) for v in first_expected)!r} "
            f"(DS {'/'.join(first_expected)}), "
            f"got {tuple(float(v) for v in first_actual)!r} (DS {'/'.join(first_actual)})")


pytestmark = pytest.mark.skipif(
    os.environ.get("RTMASK_CONFORMANCE_SKIP_E2E") == "1",
    reason="RTMASK_CONFORMANCE_SKIP_E2E=1 set",
)


def test_self_roundtrip(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    # Primitive positions are calibrated to the canonical 512x512x200 grid; smaller
    # grids would clip several ROIs. Drop n_quadrature to keep this end-to-end
    # test under a minute on a typical CI runner.
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)

    # Rasterize each primitive on the same geometry as the GT and dump as a
    # prediction. By construction these masks are byte-identical to the GT.
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    primitives = build_conformance_primitives()
    gt_dir = fixture / "groundtruth"

    # Pull the geometry back from the GT NIfTI rather than reconstructing it,
    # since a real consumer reads it that way.
    sample = sitk.ReadImage(str(gt_dir / f"{primitives[0].name}.nii.gz"))
    geometry = sitk_image_to_geometry(sample)

    for primitive in primitives:
        fractions = partial_volume_mask(primitive, geometry, n_quadrature=options.n_quadrature)
        binary = binary_threshold(fractions, threshold=0.5)
        write_nifti(binary, geometry, pred_dir / f"{primitive.name}.nii.gz")

    report = verify_predictions(pred_dir, gt_dir, load_config())
    assert report.overall_pass, [r.violations for r in report.results if r.status != Status.PASS]
    for record in report.results:
        assert record.status == Status.PASS, (record.roi, record.violations)
        assert record.metrics["dice"] == pytest.approx(1.0)


def test_all_seven_rois_present_in_groundtruth(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    options = GenerateOptions(voxel_size=(1.0, 1.0, 1.0), size=(512, 512, 200), n_quadrature=2)
    generate_fixture(fixture, options=options)
    gt_dir = fixture / "groundtruth"
    for roi in CONFORMANCE_ROIS:
        assert (gt_dir / f"{roi}.nii.gz").is_file(), roi


# --------------------------------------------------------------------------- #
#  v0.2 fixture layout, on one shared fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def v02_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture = tmp_path_factory.mktemp("v02") / "fixture"
    mp = _frozen_clock()
    try:
        generate_fixture(fixture, options=GenerateOptions(n_quadrature=2))
    finally:
        mp.undo()
    return fixture


@pytest.fixture(scope="module")
def v01_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v01") / "reference"
    mp = _frozen_clock()
    try:
        _generate_v01_reference(out, n_quadrature=2)
    finally:
        mp.undo()
    return out


def test_planar_bundle_byte_identical_to_v01(v02_fixture: Path, v01_reference: Path) -> None:
    expected = {rel: _sha256(v01_reference / rel) for rel in V01_PLANAR_FILES}
    actual = {rel: _sha256(v02_fixture / rel) for rel in V01_PLANAR_FILES}
    assert actual == expected
    assert sorted(p.name for p in (v02_fixture / "rtstruct").iterdir()) == sorted(
        f"primitives_{bundle}.dcm" for bundle in BUNDLES
    )
    # Binary-mask truth, plus the reference rasterisation of every curve and point ROI.
    assert sorted(p.name for p in (v02_fixture / "groundtruth").iterdir()) == sorted(
        f"{roi.name}.nii.gz" for rois in BUNDLES.values() for roi in rois
    )


def test_planar_bundle_matches_pinned_v01_hashes(v02_fixture: Path) -> None:
    pinned = V01_PLANAR_SHA256_BY_PLATFORM.get(sys.platform)
    if pinned is None:
        pytest.skip(f"no pinned v0.1 hashes for platform {sys.platform!r}")
    actual = {rel: _sha256(v02_fixture / rel) for rel in pinned}
    assert actual == pinned, (
        "planar fixture bytes differ from the pinned v0.1 hashes for this platform; "
        "if test_planar_bundle_byte_identical_to_v01 passes, a dependency (numpy, "
        "SimpleITK, pydicom, zlib) changed the output, not this package"
    )


def _serialisation_differences(fingerprint: dict, ref) -> list[str]:
    """Serialisation layers that differ from the Windows reference (gzip headers in full)."""
    if ref is None:
        return []
    expected = {
        key.removeprefix("serialisation/"): str(ref[key])
        for key in ref.files if key.startswith("serialisation/")
    }
    actual = fingerprint["serialisation"]
    return [
        f"{key}: {expected.get(key)} -> {actual.get(key)}"
        if key.startswith("gzip_header/") else key
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def test_planar_bundle_content_identical_across_platforms(v02_fixture: Path) -> None:
    """The planar bundle's content is the same on every platform, whatever the file
    bytes: voxels, geometry, pixel type and contour structure exactly, contour
    coordinates within CONTOUR_TOLERANCE_MM. A failure names the ROI and what differs;
    a pass with differing file bytes warns with the serialisation layers that differ.

    To regenerate the reference (on Windows only, after a deliberate content change),
    generate the fixture as ``v02_fixture`` does (frozen clock, ``n_quadrature=2``),
    call ``_write_content_reference(_planar_content_fingerprint(fixture),
    PLANAR_CONTENT_REFERENCE)`` and copy the fingerprint's ``"content"`` hashes into
    PLANAR_CONTENT_SHA256.
    """
    fingerprint = _planar_content_fingerprint(v02_fixture)
    ref = np.load(PLANAR_CONTENT_REFERENCE) if PLANAR_CONTENT_REFERENCE.is_file() else None
    problems = [
        _diagnose_content_mismatch(roi, kind, fingerprint, ref)
        for roi, expected in PLANAR_CONTENT_SHA256.items()
        for kind in ("mask", "contour_structure")
        if fingerprint["content"][roi][kind] != expected[kind]
    ]
    serialisation = _serialisation_differences(fingerprint, ref)
    if ref is not None:
        coordinate_problems, ds = _compare_contour_coordinates(fingerprint, ref)
        problems += coordinate_problems
        if ds["strings"]:
            serialisation.insert(0, (
                f"ContourData strings differ in {ds['strings']} DS value(s) on "
                f"{ds['contours']} contour(s); max |difference| {ds['max_mm']!r} mm"))
    else:
        warnings.warn(f"{PLANAR_CONTENT_REFERENCE.name} not found; contour coordinates "
                      "were not compared", stacklevel=1)
    assert sorted(PLANAR_CONTENT_SHA256) == sorted(CONFORMANCE_ROIS)
    assert not problems, (
        "planar bundle content differs from the Windows reference:\n" + "\n".join(problems)
        + "\nserialisation layers that differ:\n" + ("\n".join(serialisation) or "none")
    )
    if serialisation:
        warnings.warn(
            "planar bundle content matches the Windows reference but these serialisation "
            "layers differ:\n" + "\n".join(serialisation),
            stacklevel=1,
        )


def test_manifest_v2_describes_the_fixture(v02_fixture: Path) -> None:
    m = read_manifest(v02_fixture)
    assert m.manifest_version == "2"
    assert m.n_quadrature == 2 and m.pv_threshold == 0.5
    assert m.git_sha
    assert list(m.bundles) == list(BUNDLES)
    for name, bundle in m.bundles.items():
        assert bundle.rtstruct_path == f"rtstruct/primitives_{name}.dcm"
        assert bundle.rtstruct_sha256 == _sha256(v02_fixture / bundle.rtstruct_path)
        assert bundle.rois == [roi.name for roi in BUNDLES[name]]
    assert m.graded_rois == [roi.name for rois in BUNDLES.values() for roi in rois]
    # The planar bundle in detail; the pointline bundle is covered by test_pointline.py.
    for roi in m.bundles["planar"].rois:
        entry = m.rois[roi]
        assert entry.bundle == "planar"
        assert entry.geometric_type == "CLOSED_PLANAR"
        assert entry.truth_kind == "binary_mask" and entry.graded
        assert entry.truth_path == f"groundtruth/{roi}.nii.gz"
        assert entry.truth_sha256 == _sha256(v02_fixture / entry.truth_path)
        assert entry.expected_prediction == f"{roi}.nii.gz"
        spec = json.loads((v02_fixture / entry.spec_path).read_text(encoding="utf-8"))
        assert spec["bundle"] == "planar"
        assert spec["geometric_type"] == "CLOSED_PLANAR"
        assert spec["truth_kind"] == "binary_mask"
        assert spec["graded"] is True
        assert spec["pv_quadrature_per_axis"] == 2


def test_verify_fixture_ground_truth_passes(v02_fixture: Path) -> None:
    report = verify_fixture(v02_fixture / "groundtruth", v02_fixture, load_config())
    every_roi = [roi.name for rois in BUNDLES.values() for roi in rois]
    assert report.summary["total"] == len(every_roi)
    assert report.summary["passed"] == len(every_roi), [r.violations for r in report.results]
    assert [r.roi for r in report.results] == every_roi
    d = report.as_dict()
    assert d["version"] == REPORT_VERSION == "2.0"
    assert d["fixture"]["manifest_version"] == "2"
    assert d["fixture"]["n_quadrature"] == 2
    assert d["fixture"]["bundles"] == list(BUNDLES)
    assert d["fixture"]["package_version"]
    assert d["fixture"]["git_sha"]


@pytest.fixture
def tampered_fixture(v02_fixture: Path, tmp_path: Path) -> Path:
    """A copy of the manifest + ground truth with one voxel of ``cube`` flipped."""
    copy = tmp_path / "tampered"
    copy.mkdir()
    shutil.copyfile(v02_fixture / "manifest.json", copy / "manifest.json")
    shutil.copytree(v02_fixture / "groundtruth", copy / "groundtruth")
    target = copy / "groundtruth" / "cube.nii.gz"
    img = sitk.ReadImage(str(target))
    arr = sitk.GetArrayFromImage(img)
    arr[0, 0, 0] = 1 - arr[0, 0, 0]
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(img)
    sitk.WriteImage(out, str(target), True)
    return copy


def test_verify_fixture_detects_tampered_truth(tampered_fixture: Path) -> None:
    with pytest.raises(FixtureIntegrityError, match=r"cube\.nii\.gz.*[Rr]e-run"):
        verify_fixture(tampered_fixture / "groundtruth", tampered_fixture, load_config())


def test_verify_fixture_unknown_bundle_rejected(v02_fixture: Path) -> None:
    with pytest.raises(ValueError, match="not in this fixture"):
        verify_fixture(v02_fixture / "groundtruth", v02_fixture, bundles=["keyhole"])


def test_cli_verify_fixture_error_paths(
    v02_fixture: Path, tampered_fixture: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = str(v02_fixture / "groundtruth")
    assert cli_main(["verify", "--predictions", pred, "--fixture", str(tampered_fixture)]) == 2
    assert "does not match the manifest" in capsys.readouterr().err
    assert cli_main(["verify", "--predictions", pred, "--fixture", str(v02_fixture),
                     "--bundle", "keyhole"]) == 2
    assert cli_main(["verify", "--predictions", pred, "--groundtruth", pred,
                     "--bundle", "planar"]) == 2
    assert "--bundle requires --fixture" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # --fixture and --groundtruth are mutually exclusive
        cli_main(["verify", "--predictions", pred, "--fixture", str(v02_fixture),
                  "--groundtruth", pred])


def _load_sync_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "sync_from_upstream.py"
    if not path.is_file():
        pytest.skip("tools/ not present (installed package, not a checkout)")
    spec = importlib.util.spec_from_file_location("sync_from_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_tool_refuses_to_write_outside_vendor(tmp_path: Path) -> None:
    sync = _load_sync_tool()
    vendor = tmp_path / "src" / "rtmask_conformance" / "_vendor"
    version_file = tmp_path / "tools" / "UPSTREAM_VERSION.txt"
    assert sync._guarded(vendor / "common" / "io.py", vendor, version_file)
    assert sync._guarded(version_file, vendor, version_file)
    for bad in (vendor.parent / "primitives" / "__init__.py",
                vendor.parent / "verify.py",
                vendor / ".." / "generate.py"):
        with pytest.raises(sync.UnsafeDestinationError):
            sync._guarded(bad, vendor, version_file)
