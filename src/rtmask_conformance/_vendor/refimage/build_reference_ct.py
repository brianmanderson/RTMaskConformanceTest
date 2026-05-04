"""Synthetic reference CT generator (vendored from rtmask_validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from ..common.coords import Geometry
from .uid_hierarchy import PROJECT_SALT, derived_uid


@dataclass(frozen=True)
class ReferenceCTSpec:
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[int, int, int] = (512, 512, 200)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    patient_id: str = "RTMASK_VAL_PHANTOM"
    patient_name: str = "Phantom^RTMaskValidation"
    accession_number: str = "RTMASKVAL001"
    series_description: str = "Synthetic reference CT for RT-mask validation"


def _build_volume(spec: ReferenceCTSpec) -> np.ndarray:
    cols, rows, slices = spec.size
    sx, sy, sz = spec.voxel_size

    x = (np.arange(cols, dtype=np.float32) + 0.5) * sx + spec.origin[0]
    y = (np.arange(rows, dtype=np.float32) + 0.5) * sy + spec.origin[1]
    z = (np.arange(slices, dtype=np.float32) + 0.5) * sz + spec.origin[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    f = (
        15.0 * np.sin(2 * np.pi * xx / 200.0)
        + 12.0 * np.cos(2 * np.pi * yy / 180.0)
        + 8.0 * np.sin(2 * np.pi * zz / 90.0)
    )
    volume_hu = np.clip(f, -200.0, 200.0).astype(np.int16)
    return np.transpose(volume_hu, (2, 0, 1))


def _build_dicom_dataset(
    *,
    pixel_data: np.ndarray,
    instance_number: int,
    image_position_patient: tuple[float, float, float],
    spec: ReferenceCTSpec,
    study_uid: str,
    series_uid: str,
    frame_of_reference_uid: str,
    sop_instance_uid: str,
) -> Dataset:
    cols, rows, _ = spec.size

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = derived_uid(PROJECT_SALT, "implementation_class")
    file_meta.ImplementationVersionName = "rtmask_val_v1"

    ds = Dataset()
    ds.file_meta = file_meta

    ds.PatientID = spec.patient_id
    ds.PatientName = spec.patient_name
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20260101"
    ds.StudyTime = "000000"
    ds.AccessionNumber = spec.accession_number
    ds.StudyID = "1"
    ds.ReferringPhysicianName = ""

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.Modality = "CT"
    ds.SeriesDescription = spec.series_description
    ds.SeriesDate = "20260101"
    ds.SeriesTime = "000000"
    ds.PatientPosition = "HFS"

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = "20260101"
    ds.InstanceCreationTime = "000000"
    ds.Manufacturer = "rtmask_validation"

    ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
    ds.AcquisitionNumber = 1
    ds.PatientOrientation = ["L", "P"]

    ds.KVP = "120"

    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.PositionReferenceIndicator = ""

    sx, sy, sz = spec.voxel_size
    ds.PixelSpacing = [sy, sx]
    ds.SliceThickness = sz
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = list(image_position_patient)
    ds.SliceLocation = float(image_position_patient[2])
    ds.InstanceNumber = instance_number

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "HU"

    ds.PixelData = pixel_data.astype(np.int16, copy=False).tobytes()

    return ds


def build_reference_ct(
    out_dir: str | Path,
    spec: ReferenceCTSpec | None = None,
) -> Geometry:
    spec = spec or ReferenceCTSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    study_uid = derived_uid(PROJECT_SALT, "study")
    series_uid = derived_uid(PROJECT_SALT, "ct_series")
    frame_of_reference_uid = derived_uid(PROJECT_SALT, "frame_of_reference")

    volume = _build_volume(spec)
    sx, sy, sz = spec.voxel_size

    for z_idx in range(volume.shape[0]):
        sop_uid = derived_uid(PROJECT_SALT, "ct_slice", str(z_idx))
        position = (
            spec.origin[0],
            spec.origin[1],
            spec.origin[2] + z_idx * sz,
        )
        ds = _build_dicom_dataset(
            pixel_data=volume[z_idx],
            instance_number=z_idx + 1,
            image_position_patient=position,
            spec=spec,
            study_uid=study_uid,
            series_uid=series_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uid=sop_uid,
        )
        ds.save_as(
            out / f"slice_{z_idx:04d}.dcm",
            enforce_file_format=True,
            little_endian=True,
            implicit_vr=False,
        )

    return Geometry(
        origin=spec.origin,
        spacing=spec.voxel_size,
        size=spec.size,
    )
