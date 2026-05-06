"""Plugin evaluator API for external programs.

This module exposes a thin, format-flexible entry point so other tools can
score their own prediction/ground-truth mask pairs without having to write
NIfTI files into a `<roi>.nii.gz` layout. Three input forms are accepted for
either argument: ``numpy.ndarray``, filesystem path (anything ``SimpleITK``
can read), or ``SimpleITK.Image``.

Public surface:

* :class:`MaskMetrics` — raw per-pair metrics, no thresholds applied.
* :func:`evaluate_masks` — returns ``MaskMetrics``.
* :func:`evaluate_masks_with_thresholds` — returns a ``ResultRecord`` with
  PASS/FAIL status and human-readable violations.
* :class:`GeometryMismatchError` — raised by ``evaluate_masks`` when both
  inputs carry geometry and that geometry disagrees.

Both functions reuse the vendored metric implementations and the geometry /
threshold helpers from :mod:`rtmask_conformance.verify` — no metric math is
re-implemented here.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Union

import numpy as np
import SimpleITK as sitk

from ._roi_set import CONFORMANCE_ROIS
from ._vendor.metrics import all_surface_metrics, binary_dsc, volume_metrics
from .thresholds import ConformanceConfig, Thresholds, load_config
from .verify import (
    GEOMETRY_TOLERANCE_MM,
    ResultRecord,
    Status,
    _evaluate_against_thresholds,
    _geometry_diff,
)

MaskLike = Union[np.ndarray, str, "os.PathLike[str]", sitk.Image]


class GeometryMismatchError(ValueError):
    """Raised when two mask inputs both carry geometry and it does not agree."""


@dataclass
class MaskMetrics:
    """Raw per-pair metrics. ``None`` means "not computable" — typically because
    one or both masks are empty (no surface to compare).
    """

    dice: float
    surface_dice: float | None
    hd95_mm: float | None
    msd_mm: float | None
    volume_rel_err: float | None
    volume_abs_err_mm3: float
    tool_volume_mm3: float
    reference_volume_mm3: float
    surface_dsc_tolerance_mm: float
    spacing_xyz: tuple[float, float, float]

    def as_dict(self) -> dict[str, float | None | tuple[float, float, float]]:
        return asdict(self)


def _coerce_mask(mask: MaskLike) -> tuple[np.ndarray, sitk.Image | None]:
    """Normalise a mask input to ``(zyx_array, sitk_image_or_None)``.

    Path or ``sitk.Image`` inputs return both the array and the image so the
    caller can read geometry from the image. A bare ``ndarray`` returns
    ``(array, None)`` — the caller must supply ``spacing_xyz`` explicitly.
    """
    if isinstance(mask, sitk.Image):
        return sitk.GetArrayFromImage(mask), mask
    if isinstance(mask, np.ndarray):
        return mask, None
    if isinstance(mask, (str, os.PathLike)):
        img = sitk.ReadImage(str(mask))
        return sitk.GetArrayFromImage(img), img
    raise TypeError(
        f"Unsupported mask input type {type(mask).__name__}; expected "
        f"numpy.ndarray, str/os.PathLike, or SimpleITK.Image."
    )


def _resolve_spacing(
    pred_img: sitk.Image | None,
    gt_img: sitk.Image | None,
    spacing_xyz: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Pick the spacing to use for surface metrics.

    Priority: explicit ``spacing_xyz`` > prediction image spacing > GT image
    spacing. When neither input carries geometry and no spacing is supplied,
    ``ValueError`` is raised because surface metrics are spacing-dependent.
    """
    if spacing_xyz is not None:
        return tuple(float(s) for s in spacing_xyz)  # type: ignore[return-value]
    if gt_img is not None:
        return tuple(float(s) for s in gt_img.GetSpacing())  # type: ignore[return-value]
    if pred_img is not None:
        return tuple(float(s) for s in pred_img.GetSpacing())  # type: ignore[return-value]
    raise ValueError(
        "spacing_xyz is required when both inputs are bare numpy arrays "
        "(no SimpleITK.Image / NIfTI path to infer spacing from)."
    )


def _compute_metrics(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    surface_dsc_tolerance_mm: float,
) -> MaskMetrics:
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    vm = volume_metrics(pred_arr, gt_arr, voxel_volume)
    dsc = binary_dsc(pred_arr, gt_arr)
    sm = all_surface_metrics(
        pred_arr, gt_arr, spacing_xyz, tolerance_mm=surface_dsc_tolerance_mm
    )
    return MaskMetrics(
        dice=dsc,
        surface_dice=sm.surface_dsc,
        hd95_mm=sm.hausdorff95_mm,
        msd_mm=sm.mean_surface_distance_mm,
        volume_rel_err=(vm.rel_error_pct / 100.0) if vm.rel_error_pct is not None else None,
        volume_abs_err_mm3=vm.abs_error_mm3,
        tool_volume_mm3=vm.tool_volume_mm3,
        reference_volume_mm3=vm.reference_volume_mm3,
        surface_dsc_tolerance_mm=float(surface_dsc_tolerance_mm),
        spacing_xyz=tuple(float(s) for s in spacing_xyz),  # type: ignore[arg-type]
    )


def evaluate_masks(
    prediction: MaskLike,
    ground_truth: MaskLike,
    spacing_xyz: tuple[float, float, float] | None = None,
    *,
    surface_dsc_tolerance_mm: float = 1.0,
    check_geometry: bool = True,
) -> MaskMetrics:
    """Compute raw metrics between two masks given as ndarrays, paths, or
    ``SimpleITK.Image`` objects. The two inputs may use different forms.

    Spacing is taken from the explicit ``spacing_xyz`` argument when provided;
    otherwise it is inferred from whichever input carries geometry (GT first,
    then prediction). When both inputs are bare ndarrays, ``spacing_xyz`` is
    required.

    ``check_geometry=True`` runs the same origin/spacing/size/direction
    precheck as :func:`evaluate_one` whenever both inputs carry geometry. On
    disagreement, raises :class:`GeometryMismatchError`. Set ``False`` to
    score arbitrary array pairs (the caller is then responsible for ensuring
    the arrays describe the same physical region).
    """
    pred_arr, pred_img = _coerce_mask(prediction)
    gt_arr, gt_img = _coerce_mask(ground_truth)

    if check_geometry and pred_img is not None and gt_img is not None:
        diag = _geometry_diff(pred_img, gt_img, GEOMETRY_TOLERANCE_MM)
        if diag is not None:
            raise GeometryMismatchError(
                f"Prediction and ground-truth geometry disagree: {diag}"
            )

    spacing = _resolve_spacing(pred_img, gt_img, spacing_xyz)
    return _compute_metrics(pred_arr, gt_arr, spacing, surface_dsc_tolerance_mm)


def _resolve_thresholds(
    *,
    thresholds: Thresholds | dict[str, float] | None,
    config: ConformanceConfig | None,
    roi: str,
) -> tuple[Thresholds, ConformanceConfig]:
    """Pick the Thresholds object to use, returning it alongside the config
    that supplied any defaults (so missing keys can be filled in).
    """
    cfg = config if config is not None else load_config()

    if isinstance(thresholds, Thresholds):
        return thresholds, cfg

    if isinstance(thresholds, dict):
        merged = dict(cfg.defaults)
        merged.update(thresholds)
        try:
            return Thresholds(**merged), cfg  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError(
                f"Custom thresholds dict {thresholds!r} merged with defaults "
                f"{cfg.defaults!r} is missing required metric keys."
            ) from e

    if roi in CONFORMANCE_ROIS:
        return cfg.thresholds_for(roi), cfg

    return Thresholds(**cfg.defaults), cfg  # type: ignore[arg-type]


def evaluate_masks_with_thresholds(
    prediction: MaskLike,
    ground_truth: MaskLike,
    spacing_xyz: tuple[float, float, float] | None = None,
    *,
    roi: str = "custom",
    thresholds: Thresholds | dict[str, float] | None = None,
    config: ConformanceConfig | None = None,
    surface_dsc_tolerance_mm: float = 1.0,
    check_geometry: bool = True,
) -> ResultRecord:
    """Compute metrics and grade them against thresholds, returning a
    ``ResultRecord`` (the same dataclass used by :func:`evaluate_one`).

    Threshold resolution (first non-None wins):
        1. ``thresholds`` argument — a ``Thresholds`` instance, or a ``dict``
           shallow-merged over ``config.defaults`` (or the shipped defaults
           if ``config`` is None).
        2. ``config.thresholds_for(roi)`` if ``roi`` is one of the seven
           shipped ``CONFORMANCE_ROIS``.
        3. ``config.defaults`` (the shipped baseline) for any other ``roi``.

    ``check_geometry=True`` causes a geometry mismatch to return a
    ``ResultRecord`` with ``Status.GEOMETRY_MISMATCH`` (mirroring
    :func:`evaluate_one`), rather than raising. Set ``False`` to skip the
    precheck entirely.
    """
    pred_arr, pred_img = _coerce_mask(prediction)
    gt_arr, gt_img = _coerce_mask(ground_truth)

    resolved, _ = _resolve_thresholds(thresholds=thresholds, config=config, roi=roi)
    threshold_dict = resolved.as_dict()

    if check_geometry and pred_img is not None and gt_img is not None:
        diag = _geometry_diff(pred_img, gt_img, GEOMETRY_TOLERANCE_MM)
        if diag is not None:
            return ResultRecord(
                roi=roi,
                status=Status.GEOMETRY_MISMATCH,
                thresholds=threshold_dict,
                violations=["geometry does not match ground truth"],
                geometry_diagnostic=diag,
            )

    spacing = _resolve_spacing(pred_img, gt_img, spacing_xyz)
    metrics = _compute_metrics(pred_arr, gt_arr, spacing, surface_dsc_tolerance_mm)

    # ResultRecord.metrics keys must match what _evaluate_against_thresholds
    # looks up: dice, surface_dice_1mm, hd95_mm, msd_mm, volume_rel_err.
    metrics_for_grade: dict[str, float | None] = {
        "dice": metrics.dice,
        "surface_dice_1mm": metrics.surface_dice,
        "hd95_mm": metrics.hd95_mm,
        "msd_mm": metrics.msd_mm,
        "volume_rel_err": metrics.volume_rel_err,
        "volume_abs_err_mm3": metrics.volume_abs_err_mm3,
        "tool_volume_mm3": metrics.tool_volume_mm3,
        "reference_volume_mm3": metrics.reference_volume_mm3,
    }

    violations = _evaluate_against_thresholds(metrics_for_grade, resolved)
    return ResultRecord(
        roi=roi,
        status=Status.PASS if not violations else Status.FAIL,
        metrics=metrics_for_grade,
        thresholds=threshold_dict,
        violations=violations,
    )
