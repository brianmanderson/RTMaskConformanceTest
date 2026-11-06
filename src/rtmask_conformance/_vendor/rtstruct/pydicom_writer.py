"""Build a DICOM RT Structure Set (RTSTRUCT) from one or more analytical primitives.

This is the Python counterpart to the C# tool's ``RtStructWriterService.cs``. The two
implementations should produce semantically equivalent output for the same primitives,
which the benchmark harness exploits to cross-validate the C# writer.

The writer consumes a unified contour stream from each primitive
(:meth:`AnalyticalPrimitive.get_contours`) -- one :class:`ContourItem` per
ContourSequence entry. This single code path handles every geometric type:

* CLOSED_PLANAR / CLOSEDPLANAR_XOR  -- one ContourItem per slice the shape touches.
* POINT                              -- one ContourItem per point (NumberOfContourPoints = 1).
* OPEN_NONPLANAR / CLOSED_NONPLANAR  -- one ContourItem holding the entire 3D
  polyline. ``slice_indices`` lists every slice the polyline passes through so
  ContourImageSequence references them all.

ContourGeometricType comes from :func:`dicom_string`, which emits the standard
spelling ``CLOSEDPLANAR_XOR``. ``CLOSED_NONPLANAR`` is not a DICOM value: only the
off-registry probe primitives carry it, and an RTSTRUCT that contains it is
deliberately non-conformant.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from pydicom import Dataset, Sequence as DicomSequence, dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from ..common.coords import Geometry
from ..primitives.base import AnalyticalPrimitive, ContourItem
from ..refimage.uid_hierarchy import PROJECT_SALT, derived_uid
from .geometric_types import dicom_string

#: SOP Class UID for RT Structure Set Storage (DICOM PS3.4).
RT_STRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"

#: ROI display color palette (cycled as new ROIs are added).
_COLOR_PALETTE: list[list[int]] = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 128, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 165, 0],
    [160, 32, 240],
]


def _format_ds(value: float) -> str:
    """Format a float for the DICOM ``DS`` (Decimal String) VR.

    The DS VR caps each value at 16 characters. We pick the highest-precision ``G``
    format that fits, falling back to integer rounding for pathological inputs.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return "0"
    for prec in range(16, 0, -1):
        s = f"{value:.{prec}g}"
        if len(s) <= 16:
            return s
    return str(int(round(value)))


def _format_ds_list(values: Sequence[float]) -> list[str]:
    return [_format_ds(v) for v in values]


def _read_ref_series_metadata(ref_image_folder: Path) -> tuple[Dataset, list[Dataset], Geometry]:
    """Read every CT .dcm in the folder. Return (first_slice_dataset, sorted_slices, geometry).

    Slices are sorted by ImagePositionPatient z so SOPInstanceUIDs map to the same
    z-indices the primitive's :meth:`get_contours` produced.
    """
    files = sorted(p for p in ref_image_folder.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        raise FileNotFoundError(f"No .dcm files in {ref_image_folder}")

    slices = [dcmread(str(p)) for p in files]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    first = slices[0]
    sx, sy = float(first.PixelSpacing[1]), float(first.PixelSpacing[0])
    sz = float(first.SliceThickness)
    cols = int(first.Columns)
    rows = int(first.Rows)
    n_slices = len(slices)
    origin = tuple(float(x) for x in first.ImagePositionPatient)
    direction = tuple(
        list(first.ImageOrientationPatient[0:3])
        + list(first.ImageOrientationPatient[3:6])
        + [0.0, 0.0, 1.0]
    )
    geometry = Geometry(
        origin=origin,
        spacing=(sx, sy, sz),
        size=(cols, rows, n_slices),
        direction=tuple(float(v) for v in direction),
    )
    return first, slices, geometry


def _build_contour_image_sequence(
    slices: list[Dataset], slice_indices: Sequence[int]
) -> DicomSequence:
    """Build a ContourImageSequence referencing the given reference-CT slice indices."""
    items: list[Dataset] = []
    for z_idx in slice_indices:
        if z_idx < 0 or z_idx >= len(slices):
            continue
        s = slices[z_idx]
        item = Dataset()
        item.ReferencedSOPClassUID = s.SOPClassUID
        item.ReferencedSOPInstanceUID = s.SOPInstanceUID
        items.append(item)
    return DicomSequence(items)


def _build_full_contour_image_sequence(slices: list[Dataset]) -> DicomSequence:
    """Build a ContourImageSequence covering every slice in the reference series.

    Used in ReferencedFrameOfReferenceSequence to declare which CT slices the
    structure set is registered against.
    """
    items: list[Dataset] = []
    for s in slices:
        item = Dataset()
        item.ReferencedSOPClassUID = s.SOPClassUID
        item.ReferencedSOPInstanceUID = s.SOPInstanceUID
        items.append(item)
    return DicomSequence(items)


def _build_referenced_frame_of_reference_sequence(
    *, frame_of_reference_uid: str, study_uid: str, series_uid: str, slices: list[Dataset]
) -> DicomSequence:
    contour_image_seq = _build_full_contour_image_sequence(slices)

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = series_uid
    ref_series.ContourImageSequence = contour_image_seq

    ref_study = Dataset()
    # OID for "Detached Study Management" (legacy but DICOM-compliant placeholder).
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    ref_study.ReferencedSOPInstanceUID = study_uid
    ref_study.RTReferencedSeriesSequence = DicomSequence([ref_series])

    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = frame_of_reference_uid
    ref_frame.RTReferencedStudySequence = DicomSequence([ref_study])

    return DicomSequence([ref_frame])


def _build_one_contour_item(
    *, contour: ContourItem, slices: list[Dataset]
) -> Dataset:
    """Build one ContourSequence entry from a :class:`ContourItem`."""
    points = np.asarray(contour.points_xyz, dtype=np.float64)
    flat = np.ravel(points).tolist()

    item = Dataset()
    item.ContourGeometricType = dicom_string(contour.geometric_type)
    item.NumberOfContourPoints = int(points.shape[0])
    item.ContourData = _format_ds_list(flat)

    if contour.slice_indices:
        item.ContourImageSequence = _build_contour_image_sequence(
            slices, contour.slice_indices
        )
    return item


def build_rtstruct(
    primitives: Sequence[AnalyticalPrimitive],
    ref_image_folder: str | Path,
    out_path: str | Path,
    *,
    structure_set_label: str = "RT_MASK_VAL",
) -> Path:
    """Build a single RTSTRUCT containing every primitive in ``primitives``.

    Parameters
    ----------
    primitives : Sequence[AnalyticalPrimitive]
        One ROI per primitive will be added to the structure set.
    ref_image_folder : str | Path
        Folder containing the reference CT .dcm files.
    out_path : str | Path
        Where to write the RTSTRUCT .dcm.
    structure_set_label : str
        Value of the StructureSetLabel tag.

    Returns
    -------
    Path
        ``out_path`` as a Path. Raises if no contours could be produced.
    """
    ref_folder = Path(ref_image_folder)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    first_slice, slices, geometry = _read_ref_series_metadata(ref_folder)

    rt_series_uid = derived_uid(PROJECT_SALT, "rtstruct_series", str(out.name))
    rt_sop_uid = derived_uid(PROJECT_SALT, "rtstruct_sop", str(out.name))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RT_STRUCT_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = rt_sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = derived_uid(PROJECT_SALT, "implementation_class")
    file_meta.ImplementationVersionName = "rtmask_val_v1"

    ds = FileDataset(
        str(out),
        Dataset(),
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    # Patient + Study (copied from the reference CT).
    for tag in (
        "PatientID",
        "PatientName",
        "PatientBirthDate",
        "PatientSex",
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "AccessionNumber",
        "ReferringPhysicianName",
        "StudyID",
    ):
        if tag in first_slice:
            setattr(ds, tag, getattr(first_slice, tag))

    # RT Series module
    now = datetime.datetime.now()
    ds.SOPClassUID = RT_STRUCT_SOP_CLASS_UID
    ds.SOPInstanceUID = rt_sop_uid
    ds.Modality = "RTSTRUCT"
    ds.SeriesInstanceUID = rt_series_uid
    ds.SeriesNumber = 1
    ds.OperatorsName = ""  # Type 2 in the RT Series module: present, may be empty
    ds.SeriesDescription = "Generated by rtmask_validation"
    ds.Manufacturer = "rtmask_validation"
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")

    # RT Structure Set module
    ds.StructureSetLabel = structure_set_label
    ds.StructureSetDate = now.strftime("%Y%m%d")
    ds.StructureSetTime = now.strftime("%H%M%S")
    ds.FrameOfReferenceUID = first_slice.FrameOfReferenceUID
    ds.PositionReferenceIndicator = ""

    ds.ReferencedFrameOfReferenceSequence = _build_referenced_frame_of_reference_sequence(
        frame_of_reference_uid=first_slice.FrameOfReferenceUID,
        study_uid=first_slice.StudyInstanceUID,
        series_uid=first_slice.SeriesInstanceUID,
        slices=slices,
    )

    structure_set_roi_seq: list[Dataset] = []
    roi_contour_seq: list[Dataset] = []
    rt_roi_obs_seq: list[Dataset] = []

    roi_number = 0
    n_contours_total = 0
    for primitive in primitives:
        contours = primitive.get_contours(geometry)
        if not contours:
            continue
        roi_number += 1

        # StructureSetROI item.
        ss_roi = Dataset()
        ss_roi.ROINumber = roi_number
        ss_roi.ReferencedFrameOfReferenceUID = first_slice.FrameOfReferenceUID
        ss_roi.ROIName = primitive.name
        ss_roi.ROIGenerationAlgorithm = "MANUAL"
        structure_set_roi_seq.append(ss_roi)

        # ROIContour item.
        roi_contour = Dataset()
        color = _COLOR_PALETTE[(roi_number - 1) % len(_COLOR_PALETTE)]
        roi_contour.ROIDisplayColor = color
        roi_contour.ReferencedROINumber = roi_number
        contour_items: list[Dataset] = []
        for contour in contours:
            contour_items.append(_build_one_contour_item(contour=contour, slices=slices))
            n_contours_total += 1
        roi_contour.ContourSequence = DicomSequence(contour_items)
        roi_contour_seq.append(roi_contour)

        # RTROIObservations item.
        obs = Dataset()
        obs.ObservationNumber = roi_number
        obs.ReferencedROINumber = roi_number
        obs.RTROIInterpretedType = ""
        obs.ROIInterpreter = ""
        rt_roi_obs_seq.append(obs)

    if n_contours_total == 0:
        raise RuntimeError(
            f"No contours produced for any of {len(primitives)} primitive(s); "
            "RTSTRUCT not written."
        )

    ds.StructureSetROISequence = DicomSequence(structure_set_roi_seq)
    ds.ROIContourSequence = DicomSequence(roi_contour_seq)
    ds.RTROIObservationsSequence = DicomSequence(rt_roi_obs_seq)

    ds.save_as(
        str(out),
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    return out
