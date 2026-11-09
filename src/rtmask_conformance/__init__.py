"""rtmask_conformance — universal conformance test suite for DICOM-RT-to-NIfTI converters.

Public API:

* :data:`BUNDLES` — the ROI bundles, one RTSTRUCT each, keyed by bundle name
* :data:`CONFORMANCE_ROIS` — the graded ROI names across all bundles, in order
* :func:`generate_fixture` — write a CT series + RTSTRUCTs + GT NIfTIs + manifest
* :func:`verify_fixture` — score a predictions directory against a fixture,
  driven by its manifest (preferred)
* :func:`verify_predictions` — score a predictions directory against a bare GT
  directory (the v0.1 form)
* :class:`ConformanceConfig` — load / construct threshold configuration
* :class:`ResultRecord` — per-ROI verify result

Plugin evaluator API (for use by external tools that hold masks in memory):

* :func:`evaluate_masks` — compute raw metrics from numpy arrays, file paths,
  or ``SimpleITK.Image`` objects.
* :func:`evaluate_masks_with_thresholds` — same but applies thresholds and
  returns a ``ResultRecord``.
* :class:`MaskMetrics` — dataclass returned by ``evaluate_masks``.

Two-step shipping contract: a consuming tool runs whatever it wants between
``generate_fixture`` and ``verify_fixture``, dropping ``<roi>.nii.gz`` files
into the predictions directory.
"""

from __future__ import annotations

__version__ = "0.2.0.dev0"

from ._roi_set import BUNDLES, CONFORMANCE_ROIS, ConformanceROI
from .evaluator import (
    GeometryMismatchError,
    MaskMetrics,
    evaluate_masks,
    evaluate_masks_with_thresholds,
)
from .generate import generate_fixture
from .manifest import FixtureManifest, read_manifest
from .thresholds import ConformanceConfig, load_config
from .verify import (
    FixtureIntegrityError,
    ResultRecord,
    evaluate_one,
    verify_fixture,
    verify_predictions,
)

__all__ = [
    "BUNDLES",
    "CONFORMANCE_ROIS",
    "ConformanceConfig",
    "ConformanceROI",
    "FixtureIntegrityError",
    "FixtureManifest",
    "GeometryMismatchError",
    "MaskMetrics",
    "ResultRecord",
    "__version__",
    "evaluate_masks",
    "evaluate_masks_with_thresholds",
    "evaluate_one",
    "generate_fixture",
    "load_config",
    "read_manifest",
    "verify_fixture",
    "verify_predictions",
]
