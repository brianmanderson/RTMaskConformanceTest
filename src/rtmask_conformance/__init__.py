"""rtmask_conformance — universal conformance test suite for DICOM-RT-to-NIfTI converters.

Public API:

* :data:`CONFORMANCE_ROIS` — the seven ROI names this v0.1 suite covers
* :func:`generate_fixture` — write a CT series + RTSTRUCT + GT NIfTIs to disk
* :func:`verify_predictions` — score a predictions directory against a GT directory
* :class:`ConformanceConfig` — load / construct threshold configuration
* :class:`ResultRecord` — per-ROI verify result

Two-step shipping contract: a consuming tool runs whatever it wants between
``generate_fixture`` and ``verify_predictions``, dropping ``<roi>.nii.gz`` files
into the predictions directory.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._roi_set import CONFORMANCE_ROIS
from .generate import generate_fixture
from .thresholds import ConformanceConfig, load_config
from .verify import ResultRecord, evaluate_one, verify_predictions

__all__ = [
    "CONFORMANCE_ROIS",
    "ConformanceConfig",
    "ResultRecord",
    "__version__",
    "evaluate_one",
    "generate_fixture",
    "load_config",
    "verify_predictions",
]
