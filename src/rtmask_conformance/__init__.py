"""rtmask_conformance — universal conformance test suite for DICOM-RT-to-NIfTI converters.

Public API:

* :data:`CONFORMANCE_ROIS` — the seven ROI names this v0.1 suite covers
* :func:`generate_fixture` — write a CT series + RTSTRUCT + GT NIfTIs to disk
* :func:`verify_predictions` — score a predictions directory against a GT directory
* :class:`ConformanceConfig` — load / construct threshold configuration
* :class:`ResultRecord` — per-ROI verify result

Plugin evaluator API (for use by external tools that hold masks in memory):

* :func:`evaluate_masks` — compute raw metrics from numpy arrays, file paths,
  or ``SimpleITK.Image`` objects.
* :func:`evaluate_masks_with_thresholds` — same but applies thresholds and
  returns a ``ResultRecord``.
* :class:`MaskMetrics` — dataclass returned by ``evaluate_masks``.

Two-step shipping contract: a consuming tool runs whatever it wants between
``generate_fixture`` and ``verify_predictions``, dropping ``<roi>.nii.gz`` files
into the predictions directory.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._roi_set import CONFORMANCE_ROIS
from .evaluator import (
    GeometryMismatchError,
    MaskMetrics,
    evaluate_masks,
    evaluate_masks_with_thresholds,
)
from .generate import generate_fixture
from .thresholds import ConformanceConfig, load_config
from .verify import ResultRecord, evaluate_one, verify_predictions

__all__ = [
    "CONFORMANCE_ROIS",
    "ConformanceConfig",
    "GeometryMismatchError",
    "MaskMetrics",
    "ResultRecord",
    "__version__",
    "evaluate_masks",
    "evaluate_masks_with_thresholds",
    "evaluate_one",
    "generate_fixture",
    "load_config",
    "verify_predictions",
]
