"""Score predicted ROI masks against analytic ground truth.

Public surface:

* :func:`verify_fixture` — read a fixture's ``manifest.json``, check every
  ground-truth file against its recorded SHA-256, score the manifest's graded
  ROIs, and return a ``Report`` that names the fixture it scored. The
  preferred entry point.
* :func:`verify_predictions` — the v0.1 path: score every ROI in
  ``CONFORMANCE_ROIS`` against a bare ground-truth directory, no manifest.
* :func:`check_fixture_integrity` — the hash check on its own.
* :func:`evaluate_one` — score a single ROI; the pytest harness calls this
  per parametrized test.
* :class:`ResultRecord` — per-ROI result, JSON-serializable.

The verifier runs a geometry precheck before computing any metrics: a
prediction whose ``(origin, spacing, size, direction)`` differs from the
ground-truth NIfTI by more than ``GEOMETRY_TOLERANCE_MM`` is flagged with the
distinct ``GEOMETRY_MISMATCH`` status, never silently scored. Most third-party
converter bugs are geometry, not labeling, and surfacing them separately is
diagnostic.

Each ROI resolves to one :class:`Status`, in this order:

1. prediction absent and the ROI's geometric type not claimed by the config
   -> ``UNSUPPORTED_TYPE`` (not a failure);
2. the ROI is report-only (named in ``report_only`` by ROI or geometric type,
   or ``graded: false`` in the manifest) -> ``REPORT_ONLY``: scored when it can
   be, never a failure;
3. prediction absent -> ``MISSING``;
4. geometry differs from the ground truth -> ``GEOMETRY_MISMATCH``;
5. otherwise ``PASS`` or ``FAIL`` against the thresholds.

``MISSING``, ``GEOMETRY_MISMATCH`` and ``FAIL`` are failures; the other three are not.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import __version__
from ._roi_set import BUNDLES, CONFORMANCE_ROIS
from ._vendor.metrics import all_surface_metrics, binary_dsc, volume_metrics
from .manifest import FixtureManifest, read_manifest
from .thresholds import ConformanceConfig, Thresholds, load_config

#: Version of the JSON report layout (``Report.as_dict()["version"]``).
#: 2.0 adds the ``fixture``, ``claims`` and ``report_only`` blocks, the
#: ``unsupported_type`` and ``report_only`` summary counts, and per-ROI
#: ``geometric_type`` and ``truth_kind``.
REPORT_VERSION: str = "2.0"

# Geometry agreement tolerance (mm). Tighter than 1e-4 risks false negatives on
# tools that internally round to single precision.
GEOMETRY_TOLERANCE_MM: float = 1e-4

SURFACE_DSC_TOLERANCE_MM: float = 1.0


class FixtureIntegrityError(ValueError):
    """A fixture's ground truth does not match the hashes in its manifest."""


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"
    GEOMETRY_MISMATCH = "GEOMETRY_MISMATCH"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    REPORT_ONLY = "REPORT_ONLY"


#: Statuses that make a run fail (CLI exit code 1).
FAILING_STATUSES: frozenset[Status] = frozenset(
    {Status.FAIL, Status.MISSING, Status.GEOMETRY_MISMATCH}
)

#: ``Status`` -> ``Report.summary`` key, in summary order.
_SUMMARY_KEYS: dict[Status, str] = {
    Status.PASS: "passed",
    Status.FAIL: "failed",
    Status.MISSING: "missing",
    Status.GEOMETRY_MISMATCH: "geometry_mismatch",
    Status.UNSUPPORTED_TYPE: "unsupported_type",
    Status.REPORT_ONLY: "report_only",
}

#: Geometric type of every suite ROI, for scoring paths without a manifest.
_KNOWN_GEOMETRIC_TYPES: dict[str, str] = {
    roi.name: roi.geometric_type.value for rois in BUNDLES.values() for roi in rois
}


@dataclass
class ResultRecord:
    roi: str
    status: Status
    metrics: dict[str, float | None] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    geometry_diagnostic: str | None = None  # populated when status == GEOMETRY_MISMATCH
    prediction_path: str | None = None
    groundtruth_path: str | None = None
    geometric_type: str | None = None  # Contour Geometric Type of the ROI, if known
    truth_kind: str | None = None  # kind of ground truth scored against

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Report:
    config_source: str
    summary: dict[str, int]
    results: list[ResultRecord]
    # Manifest summary of the fixture scored; None on the v0.1 groundtruth-only path.
    fixture: dict | None = None
    # What the config claimed and marked report-only (see thresholds.py).
    claims: dict | None = None
    report_only: list[str] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        """True when no ROI is FAIL, MISSING or GEOMETRY_MISMATCH."""
        s = self.summary
        return s["failed"] + s["missing"] + s["geometry_mismatch"] == 0

    def as_dict(self) -> dict:
        return {
            "version": REPORT_VERSION,
            "config_source": self.config_source,
            "fixture": self.fixture,
            "claims": self.claims,
            "report_only": self.report_only,
            "summary": self.summary,
            "results": [r.as_dict() for r in self.results],
        }


def _resolve_prediction_path(predictions_dir: Path, roi: str) -> Path | None:
    """Find ``<roi>.nii.gz`` in ``predictions_dir``.

    Tries an exact match first; on miss, falls back to a single
    case-insensitive match (warning) before giving up.
    """
    exact = predictions_dir / f"{roi}.nii.gz"
    if exact.is_file():
        return exact
    target = f"{roi}.nii.gz".lower()
    matches = [p for p in predictions_dir.glob("*.nii.gz") if p.name.lower() == target]
    if len(matches) == 1:
        warnings.warn(
            f"Prediction for {roi!r} matched case-insensitively as {matches[0].name!r}; "
            f"rename to {roi}.nii.gz for portability.",
            stacklevel=2,
        )
        return matches[0]
    return None


def _geometry_diff(pred: sitk.Image, gt: sitk.Image, tol_mm: float) -> str | None:
    """Return ``None`` if geometries agree within ``tol_mm``, else a diagnostic string."""
    diffs: list[str] = []

    pred_size = tuple(int(v) for v in pred.GetSize())
    gt_size = tuple(int(v) for v in gt.GetSize())
    if pred_size != gt_size:
        diffs.append(f"size: pred={pred_size} gt={gt_size}")

    for label, pred_t, gt_t in (
        ("origin", pred.GetOrigin(), gt.GetOrigin()),
        ("spacing", pred.GetSpacing(), gt.GetSpacing()),
        ("direction", pred.GetDirection(), gt.GetDirection()),
    ):
        diff = max(abs(p - g) for p, g in zip(pred_t, gt_t, strict=True))
        if diff > tol_mm:
            diffs.append(f"{label}: pred={tuple(round(v, 4) for v in pred_t)} "
                         f"gt={tuple(round(v, 4) for v in gt_t)} (max abs diff {diff:.6f})")

    return "; ".join(diffs) if diffs else None


def _compute_all_metrics(
    pred_arr: np.ndarray, gt_arr: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> dict[str, float | None]:
    voxel_volume = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    vm = volume_metrics(pred_arr, gt_arr, voxel_volume)
    dsc = binary_dsc(pred_arr, gt_arr)
    sm = all_surface_metrics(pred_arr, gt_arr, spacing_xyz, tolerance_mm=SURFACE_DSC_TOLERANCE_MM)
    return {
        "dice": dsc,
        "surface_dice_1mm": sm.surface_dsc,
        "hd95_mm": sm.hausdorff95_mm,
        "msd_mm": sm.mean_surface_distance_mm,
        "volume_rel_err": (vm.rel_error_pct / 100.0) if vm.rel_error_pct is not None else None,
        "volume_abs_err_mm3": vm.abs_error_mm3,
        "tool_volume_mm3": vm.tool_volume_mm3,
        "reference_volume_mm3": vm.reference_volume_mm3,
    }


def _evaluate_against_thresholds(
    metrics: dict[str, float | None], t: Thresholds
) -> list[str]:
    """Return a list of human-readable threshold-violation strings (empty if all pass).

    "Higher is better" metrics (dice, surface_dice_1mm) fail when below threshold;
    "lower is better" metrics (hd95_mm, msd_mm, volume_rel_err) fail when above.
    A metric whose value is ``None`` (not computable, e.g. one mask empty) fails
    against any threshold — the tool didn't produce something to score.
    """
    violations: list[str] = []

    def _hi_better(name: str, threshold: float) -> None:
        v = metrics.get(name)
        if v is None:
            violations.append(f"{name} not computable (one or both masks empty)")
        elif v < threshold:
            violations.append(f"{name} {v:.4f} < {threshold}")

    def _lo_better(name: str, threshold: float) -> None:
        v = metrics.get(name)
        if v is None:
            violations.append(f"{name} not computable (one or both masks empty)")
        elif v > threshold:
            violations.append(f"{name} {v:.4f} > {threshold}")

    _hi_better("dice", t.dice)
    _hi_better("surface_dice_1mm", t.surface_dice_1mm)
    _lo_better("hd95_mm", t.hd95_mm)
    _lo_better("msd_mm", t.msd_mm)
    _lo_better("volume_rel_err", t.volume_rel_err)
    return violations


def as_report_only(record: ResultRecord) -> ResultRecord:
    """Return ``record`` with status ``REPORT_ONLY``.

    Metrics, violations (what grading would have said) and any geometry
    diagnostic are kept for information; none of them affects the verdict.
    """
    record.status = Status.REPORT_ONLY
    return record


def evaluate_one(
    roi: str,
    prediction_path: str | Path | None,
    groundtruth_path: str | Path,
    config: ConformanceConfig,
    *,
    geometric_type: str | None = None,
    truth_kind: str = "binary_mask",
    graded: bool = True,
) -> ResultRecord:
    """Score one ROI and resolve its :class:`Status` (order in the module docstring).

    ``prediction_path=None`` means the tool wrote nothing for this ROI.
    ``geometric_type`` defaults to the suite's own record for ``roi`` (``None``
    for a ROI the suite does not know, which counts as claimed). ``graded=False``
    (from the manifest) makes the ROI report-only.
    """
    if geometric_type is None:
        geometric_type = _KNOWN_GEOMETRIC_TYPES.get(roi)
    thresholds = config.thresholds_for(roi)
    common = {
        "roi": roi,
        "thresholds": thresholds.as_dict(),
        "groundtruth_path": str(groundtruth_path),
        "geometric_type": geometric_type,
        "truth_kind": truth_kind,
    }

    if prediction_path is None and not config.claims(geometric_type):
        return ResultRecord(
            status=Status.UNSUPPORTED_TYPE,
            violations=[f"no prediction; geometric type {geometric_type} is not claimed"],
            **common,
        )

    record = _grade(roi, prediction_path, groundtruth_path, thresholds, common)
    if not graded or config.is_report_only(roi, geometric_type):
        return as_report_only(record)
    return record


def _grade(
    roi: str,
    prediction_path: str | Path | None,
    groundtruth_path: str | Path,
    thresholds: Thresholds,
    common: dict,
) -> ResultRecord:
    """MISSING, GEOMETRY_MISMATCH, PASS or FAIL for one ROI (the v0.1 grading)."""
    if prediction_path is None:
        return ResultRecord(
            status=Status.MISSING,
            violations=[f"prediction file <predictions>/{roi}.nii.gz not found"],
            **common,
        )

    pred_img = sitk.ReadImage(str(prediction_path))
    gt_img = sitk.ReadImage(str(groundtruth_path))

    geom_diag = _geometry_diff(pred_img, gt_img, GEOMETRY_TOLERANCE_MM)
    if geom_diag is not None:
        return ResultRecord(
            status=Status.GEOMETRY_MISMATCH,
            violations=["geometry does not match ground truth"],
            geometry_diagnostic=geom_diag,
            prediction_path=str(prediction_path),
            **common,
        )

    pred_arr = sitk.GetArrayFromImage(pred_img)
    gt_arr = sitk.GetArrayFromImage(gt_img)
    spacing_xyz = tuple(float(s) for s in gt_img.GetSpacing())

    metrics = _compute_all_metrics(pred_arr, gt_arr, spacing_xyz)
    violations = _evaluate_against_thresholds(metrics, thresholds)

    return ResultRecord(
        status=Status.PASS if not violations else Status.FAIL,
        metrics=metrics,
        violations=violations,
        prediction_path=str(prediction_path),
        **common,
    )


def verify_predictions(
    predictions_dir: str | Path,
    groundtruth_dir: str | Path,
    config: ConformanceConfig | None = None,
    *,
    rois: list[str] | None = None,
) -> Report:
    """Score every ROI in ``rois`` (default: ``CONFORMANCE_ROIS``).

    A missing prediction file always produces a MISSING result — the caller
    decides what to do based on ``Report.summary``. Both predictions and
    ground-truth dirs must be readable; a missing GT file is a setup error
    and raises ``FileNotFoundError`` (this is the test fixture, not the
    tool output).
    """
    config = config or load_config()
    rois = rois if rois is not None else list(CONFORMANCE_ROIS)

    pred_root = Path(predictions_dir)
    gt_root = Path(groundtruth_dir)
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_root}")

    return _score(
        pred_root, [RoiTarget(roi, gt_root / f"{roi}.nii.gz") for roi in rois], config
    )


@dataclass(frozen=True)
class RoiTarget:
    """One ROI to score: its ground truth and the manifest fields that resolve its status."""

    roi: str
    gt_path: Path
    geometric_type: str | None = None
    truth_kind: str = "binary_mask"
    graded: bool = True


def summarize(results: list[ResultRecord]) -> dict[str, int]:
    """``Report.summary``: ``total`` plus one count per status."""
    summary = {"total": len(results)}
    for status, key in _SUMMARY_KEYS.items():
        summary[key] = sum(1 for r in results if r.status == status)
    return summary


def _score(
    pred_root: Path,
    to_score: list[RoiTarget],
    config: ConformanceConfig,
    fixture: dict | None = None,
) -> Report:
    """Score each ROI in order and summarize."""
    results: list[ResultRecord] = []
    for item in to_score:
        if not item.gt_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth NIfTI for {item.roi!r} not found at {item.gt_path}. "
                "Re-run `rtmask-conformance generate` to rebuild the fixture."
            )
        pred_path = _resolve_prediction_path(pred_root, item.roi)
        results.append(evaluate_one(
            item.roi, pred_path, item.gt_path, config,
            geometric_type=item.geometric_type, truth_kind=item.truth_kind, graded=item.graded,
        ))

    return Report(
        config_source=config.config_source,
        summary=summarize(results),
        results=results,
        fixture=fixture,
        claims=config.claims_dict(),
        report_only=list(config.report_only),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_bundles(manifest: FixtureManifest, bundles: list[str] | None) -> list[str]:
    if bundles is None:
        return list(manifest.bundles)
    unknown = [b for b in bundles if b not in manifest.bundles]
    if unknown:
        raise ValueError(
            f"bundle(s) {unknown} not in this fixture (available: {list(manifest.bundles)})"
        )
    return [b for b in manifest.bundles if b in bundles]


def check_fixture_integrity(
    fixture_dir: str | Path,
    manifest: FixtureManifest | None = None,
    *,
    bundles: list[str] | None = None,
) -> FixtureManifest:
    """Check every ground-truth file of the selected bundles against the manifest.

    Returns the manifest. Raises ``FileNotFoundError`` for a missing truth file
    and :class:`FixtureIntegrityError` for a hash mismatch or a truth file the
    manifest records without a hash.
    """
    root = Path(fixture_dir)
    manifest = manifest or read_manifest(root)
    selected = set(_select_bundles(manifest, bundles))
    for roi, entry in manifest.rois.items():
        if entry.bundle not in selected or entry.truth_path is None:
            continue
        path = root / entry.truth_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Ground truth for {roi!r} not found at {path}. "
                "Re-run `rtmask-conformance generate` to rebuild the fixture."
            )
        if entry.truth_sha256 is None:
            raise FixtureIntegrityError(
                f"{path}: manifest records no SHA-256 for {roi!r}. "
                "Re-run `rtmask-conformance generate` to rebuild the fixture."
            )
        actual = _sha256(path)
        if actual != entry.truth_sha256:
            raise FixtureIntegrityError(
                f"{path}: SHA-256 {actual} does not match the manifest "
                f"({entry.truth_sha256}); the ground truth changed after generation. "
                "Re-run `rtmask-conformance generate` to rebuild the fixture."
            )
    return manifest


def fixture_summary(
    fixture_dir: str | Path, manifest: FixtureManifest, bundles: list[str]
) -> dict:
    """The ``fixture`` block of a report."""
    return {
        "path": str(Path(fixture_dir)),
        "manifest_version": manifest.manifest_version,
        "package_version": manifest.package_version,
        "git_sha": manifest.git_sha,
        "n_quadrature": manifest.n_quadrature,
        "pv_threshold": manifest.pv_threshold,
        "bundles": bundles,
    }


def verify_fixture(
    predictions_dir: str | Path,
    fixture_dir: str | Path,
    config: ConformanceConfig | None = None,
    *,
    bundles: list[str] | None = None,
) -> Report:
    """Score ``predictions_dir`` against the fixture at ``fixture_dir``.

    Reads ``<fixture_dir>/manifest.json``, checks the ground truth of the
    selected ``bundles`` (default: all) against the recorded SHA-256s, and
    scores the manifest's ROIs in those bundles exactly as
    :func:`verify_predictions` does. A ROI the manifest records as
    ``graded: false`` is scored as ``REPORT_ONLY`` when it has a binary-mask
    ground truth, and skipped otherwise. A fixture written by a different
    package version is scored, with a warning.
    """
    config = config or load_config()
    fixture_root = Path(fixture_dir)
    pred_root = Path(predictions_dir)
    if not fixture_root.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_root}")
    if not (fixture_root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"No manifest.json in {fixture_root}. Pass the directory "
            "`rtmask-conformance generate` wrote, or use --groundtruth for a bare "
            "ground-truth directory."
        )
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_root}")

    manifest = read_manifest(fixture_root)
    selected = _select_bundles(manifest, bundles)
    check_fixture_integrity(fixture_root, manifest, bundles=selected)
    if manifest.package_version != __version__:
        warnings.warn(
            f"fixture was generated by rtmask-conformance {manifest.package_version}; "
            f"verifying with {__version__}. Regenerate the fixture if results look off.",
            stacklevel=2,
        )

    to_score = fixture_targets(fixture_root, manifest, bundles=selected)
    return _score(pred_root, to_score, config, fixture_summary(fixture_root, manifest, selected))


def fixture_targets(
    fixture_dir: str | Path,
    manifest: FixtureManifest,
    *,
    bundles: list[str] | None = None,
) -> list[RoiTarget]:
    """The ROIs ``verify --fixture`` scores, in manifest order.

    Every ROI of the selected ``bundles`` (default: all) with a binary-mask
    ground truth, graded or not. An ungraded ROI without one is skipped; a
    graded ROI without one raises ``NotImplementedError`` (no grader yet).
    The pytest harness parametrizes over the same list.
    """
    root = Path(fixture_dir)
    selected = _select_bundles(manifest, bundles)
    targets: list[RoiTarget] = []
    for roi, entry in manifest.rois.items():
        if entry.bundle not in selected:
            continue
        scorable = entry.truth_kind == "binary_mask" and entry.truth_path is not None
        if not scorable:
            if not entry.graded:
                continue  # shipped for information; nothing to score against
            raise NotImplementedError(
                f"{roi}: no grader for truth_kind {entry.truth_kind!r} in this version"
            )
        targets.append(RoiTarget(
            roi,
            root / entry.truth_path,
            geometric_type=entry.geometric_type,
            truth_kind=entry.truth_kind,
            graded=entry.graded,
        ))
    return targets
