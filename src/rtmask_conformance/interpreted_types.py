"""RT ROI Interpreted Type and a TG-263-style description for every suite ROI.

The vendored RTSTRUCT writer leaves RT ROI Interpreted Type (3006,00A4) empty
and writes no ROI Description (3006,0028). :func:`apply_interpreted_types` is a
post-write pass that fills both in a saved RTSTRUCT, keyed by ROI name, and
changes nothing else: ROI names, ROI numbers, contour data, UIDs and the
transfer syntax stay as written.

The Interpreted Types are Defined Terms of DICOM PS3.3 §C.8.8.8 chosen to say
which clinical class each primitive stands in for. They are illustrative, not
clinical assertions, and the suite grades nothing on them: Interpreted Type is
Type 2 metadata and constrains no geometry. They exist so that a converter or
planning system that branches on the class (``MARKER``, ``BRACHY_CHANNEL``,
``BRACHY_SRC_APP``, ``PTV`` ...) is exercised by the fixture.

ROI names never change (threshold files key on them). The descriptions follow
the AAPM TG-263 naming style: 16 characters or fewer, components joined by
underscores, a class prefix where TG-263 has one (``PTV_``), dimensions in mm.

Every ROI of every bundle must have an entry in :data:`ROI_LABELS`; the pass
raises :class:`UnmappedRoiError` on an unmapped name, so a new primitive cannot
ship unlabelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydicom import dcmread

#: The Defined Terms of RT ROI Interpreted Type (3006,00A4), DICOM PS3.3 2026d §C.8.8.8.
DEFINED_TERMS: tuple[str, ...] = (
    "EXTERNAL",
    "PTV",
    "CTV",
    "GTV",
    "TREATED_VOLUME",
    "IRRAD_VOLUME",
    "OAR",
    "BOLUS",
    "AVOIDANCE",
    "ORGAN",
    "MARKER",
    "REGISTRATION",
    "ISOCENTER",
    "CONTRAST_AGENT",
    "CAVITY",
    "BRACHY_CHANNEL",
    "BRACHY_ACCESSORY",
    "BRACHY_SRC_APP",
    "BRACHY_CHNL_SHLD",
    "SUPPORT",
    "FIXATION",
    "DOSE_REGION",
    "CONTROL",
    "DOSE_MEASUREMENT",
    "DEVICE",
)

#: TG-263 limits a structure name to 16 characters; the descriptions keep to it.
MAX_DESCRIPTION_LENGTH: int = 16


class UnmappedRoiError(KeyError):
    """An RTSTRUCT holds an ROI with no entry in :data:`ROI_LABELS`."""


@dataclass(frozen=True)
class RoiLabel:
    """The Interpreted Type and the TG-263-style description of one ROI."""

    interpreted_type: str
    description: str

    def __post_init__(self) -> None:
        if self.interpreted_type not in DEFINED_TERMS:
            raise ValueError(f"{self.interpreted_type!r} is not a Defined Term of (3006,00A4)")
        if not self.description or len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"description {self.description!r} must be 1 to "
                f"{MAX_DESCRIPTION_LENGTH} characters"
            )


#: Interpreted Type and description per ROI name, in canonical bundle order.
ROI_LABELS: dict[str, RoiLabel] = {
    # planar: convex solids and the ellipsoid stand in for organs, the cube for a
    # target, the torus for an avoidance region, the two hollow shapes for organs.
    "sphere": RoiLabel("ORGAN", "Sphere_r40"),
    "cube": RoiLabel("PTV", "PTV_Cube60"),
    "cylinder": RoiLabel("ORGAN", "Cyl_r30_h80"),
    "ellipsoid": RoiLabel("ORGAN", "Ellips_30x50x60"),
    "torus": RoiLabel("AVOIDANCE", "Torus_R60_r20"),
    "hollow_sphere": RoiLabel("ORGAN", "HSph_R40r20"),
    "straw": RoiLabel("ORGAN", "Straw_R40r20"),
    # xor, keyhole: the same two hollow shapes in the standard's hole encodings.
    "hollow_sphere_xor": RoiLabel("ORGAN", "HSph_R40r20_XOR"),
    "straw_xor": RoiLabel("ORGAN", "Straw_R40r20_XOR"),
    "hollow_sphere_keyhole": RoiLabel("ORGAN", "HSph_R40r20_Key"),
    "straw_keyhole": RoiLabel("ORGAN", "Straw_R40r20_Key"),
    # pointline, POINT: fiducial and needle-track markers, a seed implant.
    "single_point_x256_y256_z150": RoiLabel("MARKER", "Marker_Point"),
    "needle_track_10pts_x150_y150": RoiLabel("MARKER", "Marker_Track10"),
    "ldr_seed_implant_x350_y350": RoiLabel("BRACHY_SRC_APP", "Brachy_Seeds147"),
    # pointline, OPEN_NONPLANAR: brachytherapy channels.
    "diagonal_needle_x100_y200_z40": RoiLabel("BRACHY_CHANNEL", "Brachy_Needle"),
    "helix_r12_pitch6_4turns_x300_y200": RoiLabel("BRACHY_CHANNEL", "Brachy_Helix"),
    "sinusoid_x150_y350_z20to80": RoiLabel("BRACHY_CHANNEL", "Brachy_Sinusoid"),
    "syed_bundle_5needles_x400_y100_z50": RoiLabel("BRACHY_CHANNEL", "Brachy_Syed5"),
    # pointline, OPEN_PLANAR: in-plane markers.
    "line_segment_x200_y200_to_x300_y280_z175": RoiLabel("MARKER", "Marker_Line128"),
    "open_arc_r40_90deg_x380_y300_z180": RoiLabel("MARKER", "Marker_Arc90"),
}


def label_for(roi_name: str) -> RoiLabel:
    """The label of ``roi_name``; raises :class:`UnmappedRoiError` if it has none."""
    try:
        return ROI_LABELS[roi_name]
    except KeyError:
        raise UnmappedRoiError(
            f"ROI {roi_name!r} has no RT ROI Interpreted Type mapping; add it to "
            "rtmask_conformance.interpreted_types.ROI_LABELS"
        ) from None


def apply_interpreted_types(path: str | Path) -> Path:
    """Write the Interpreted Type and the description of every ROI into a saved RTSTRUCT.

    Sets ``RTROIInterpretedType`` on each RT ROI Observations item and
    ``ROIDescription`` on each Structure Set ROI item, matching
    ``ReferencedROINumber`` to ``ROINumber`` and the ROI name to
    :data:`ROI_LABELS`, then saves the file in place with its transfer syntax
    unchanged. Every other element is left as read. Idempotent.

    Raises :class:`UnmappedRoiError` (and leaves the file untouched) if any ROI
    has no mapping, or if an observation references an ROI number the
    Structure Set ROI Sequence does not define.
    """
    path = Path(path)
    ds = dcmread(path)

    names = {int(item.ROINumber): str(item.ROIName) for item in ds.StructureSetROISequence}
    unmapped = sorted(name for name in names.values() if name not in ROI_LABELS)
    if unmapped:
        raise UnmappedRoiError(
            f"{path.name}: ROI(s) {unmapped} have no RT ROI Interpreted Type mapping; "
            "add them to rtmask_conformance.interpreted_types.ROI_LABELS"
        )

    for item in ds.StructureSetROISequence:
        item.ROIDescription = ROI_LABELS[str(item.ROIName)].description
    for obs in ds.RTROIObservationsSequence:
        number = int(obs.ReferencedROINumber)
        if number not in names:
            raise UnmappedRoiError(
                f"{path.name}: RT ROI Observations item references ROI number {number}, "
                "which the Structure Set ROI Sequence does not define"
            )
        obs.RTROIInterpretedType = ROI_LABELS[names[number]].interpreted_type

    ds.save_as(path, enforce_file_format=True)
    return path
