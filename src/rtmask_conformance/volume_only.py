"""Volume-only verification, for systems that report structure volumes but export no masks.

A treatment planning system imports the fixture's RTSTRUCTs and displays a
volume per structure; it usually cannot write a voxel mask. This mode grades
those reported volumes, and nothing else, against the closed-form analytic
volume of each primitive::

    rtmask-conformance verify --volumes volumes.csv --fixture <fixture> [--config claims.yaml]

Input CSV
---------
UTF-8 (a byte-order mark is accepted), comma-separated, one header row, one
row per ROI. Column names are matched case-insensitively; every column not
listed here is ignored.

``roi`` (required)
    The ROI name as written in the RTSTRUCT. Matched exactly, then
    case-insensitively (with a warning), as ``verify`` matches mask files.
``volume_cc`` or ``volume_mm3`` (one required)
    The reported volume; the header gives the unit. When neither is present,
    exactly one column named ``<system>_volume_cc`` or ``<system>_volume_mm3``
    is used instead (a console import record, e.g. ``raystation_volume_cc``).
``<system>_volume_cc_api`` / ``<system>_volume_mm3_api`` (optional)
    A full-precision volume read through the system's scripting interface,
    graded at the tolerance alone. Which volume is graded is set by
    ``volume_source`` (``--volume-source``): ``auto`` (default) grades the API
    volume where a row has one and the displayed volume otherwise;
    ``displayed`` ignores the API column; ``api`` grades only API volumes (a
    row without one has no volume). Each result says ``graded_volume``.
``volume_decimals`` (optional)
    How many decimals the system displayed. Sets the display quantisation below.
``import_outcome`` (optional)
    ``accepted``, ``warned``, ``rejected`` or ``rejected_file``. The last two
    mean the system holds no structure for the ROI (``rejected_file``: the
    whole RTSTRUCT was refused).
``message`` (optional; ``message_verbatim`` is an alias)
    The import message(s) for the ROI, echoed in the report.
``source`` (optional)
    When present, only rows whose ``source`` is empty or ``suite`` are read;
    other rows (a record that also lists another fixture's files) are skipped.

``volumes_template.csv`` in every generated fixture is a blank input file
with the ROI names filled in.

Grading
-------
Each ROI resolves to one status, in the order ``verify`` uses for masks:

1. no structure (no row, ``rejected``/``rejected_file``, or no volume) and the
   ROI's geometric type not claimed -> ``UNSUPPORTED_TYPE``;
2. report-only: a curve or point ROI (no enclosed volume; the reported value
   is echoed), a ROI or type listed under ``report_only``, or ``graded: false``
   in the manifest -> ``REPORT_ONLY``;
3. no structure -> ``MISSING`` (the result says which of the three cases);
4. ``PASS`` when ``|V_reported - V_ref| / V_ref <= max(tolerance, q)``, else ``FAIL``.

``tolerance`` defaults to 1 % (``volume_only.tolerance`` in the YAML,
``--volume-tolerance`` on the command line). ``q`` is the display
quantisation: half a unit in the last displayed decimal divided by ``V_ref``,
``q = 0.5 * 10**-volume_decimals / V_ref`` (both in the CSV's unit). Without
``volume_decimals``, or for an API volume, ``q = 0``. A row where ``q``
exceeds the tolerance is marked ``display_limited``: the display cannot
resolve an error as small as the tolerance.

End caps
--------
``V_ref`` is the analytic volume adjusted for the end-cap convention the
config declares under ``claims.end_cap`` (see :func:`reference_volume_mm3`).
Volume-only grading cannot see where a structure is: a shifted or rotated
structure with the right volume passes.
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pydicom

from .manifest import FixtureManifest, read_manifest
from .thresholds import ConformanceConfig, load_config

#: Mode name recorded in the JSON report.
MODE: str = "volume_only"

#: ``import_outcome`` values the reader accepts (empty means not recorded).
IMPORT_OUTCOMES: tuple[str, ...] = ("accepted", "warned", "rejected", "rejected_file")

#: Outcomes that mean the system holds no structure for the ROI.
NOT_IMPORTED: frozenset[str] = frozenset({"rejected", "rejected_file"})

#: Geometric types with no enclosed volume: always REPORT_ONLY in this mode.
NO_VOLUME_TYPES: frozenset[str] = frozenset({"POINT", "OPEN_PLANAR", "OPEN_NONPLANAR"})

#: Contour types whose polygons enclose area (the end-cap computation reads only these).
_CLOSED_TYPES: frozenset[str] = frozenset({"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"})

_UNIT_TO_MM3: dict[str, float] = {"cc": 1000.0, "mm3": 1.0}

#: Which reported volume to grade (``--volume-source``); the first is the default.
VOLUME_SOURCES: tuple[str, ...] = ("auto", "displayed", "api")

#: Columns of the blank template written into every fixture.
TEMPLATE_COLUMNS: tuple[str, ...] = (
    "roi", "volume_cc", "volume_decimals", "import_outcome", "message",
)
TEMPLATE_NAME: str = "volumes_template.csv"


# --------------------------------------------------------------------------- reading


@dataclass(frozen=True)
class VolumeRow:
    """One ROI row of the input CSV, normalised."""

    roi: str
    volume: float | None          # displayed volume, in ``unit``
    api_volume: float | None      # full-precision volume, in ``api_unit``
    volume_decimals: int | None
    import_outcome: str | None
    message: str | None
    line: int


@dataclass(frozen=True)
class VolumeTable:
    """The parsed input CSV: rows plus which columns supplied what."""

    path: str
    rows: list[VolumeRow]
    volume_column: str
    unit: str                     # "cc" or "mm3"
    api_column: str | None
    api_unit: str | None
    skipped_rows: int = 0         # rows of another ``source``


_SYSTEM_VOLUME = re.compile(r"^.+_volume_(cc|mm3)$")
_API_VOLUME = re.compile(r"^(?:.+_)?volume_(cc|mm3)_api$")


def _pick_volume_column(header: list[str], context: str) -> tuple[str, str]:
    """Return ``(column, unit)`` of the reported volume."""
    plain = [h for h in ("volume_cc", "volume_mm3") if h in header]
    if len(plain) == 2:
        raise ValueError(f"{context}: both volume_cc and volume_mm3 present; keep one")
    if plain:
        return plain[0], plain[0].rsplit("_", 1)[1]
    system = [h for h in header if _SYSTEM_VOLUME.match(h)]
    if len(system) > 1:
        raise ValueError(f"{context}: several volume columns {system}; keep one")
    if not system:
        raise ValueError(
            f"{context}: no volume column; expected volume_cc, volume_mm3 or "
            "<system>_volume_cc / <system>_volume_mm3"
        )
    return system[0], system[0].rsplit("_", 1)[1]


def _number(raw: str, what: str, context: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{context}: {what} {raw!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: {what} {raw!r} must be a finite, non-negative number")
    return value


def read_volumes_csv(path: str | Path) -> VolumeTable:
    """Read a volume-only input CSV or a console import record (module docstring)."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"{p}: empty file") from None
        body = list(reader)

    header = [h.strip().lower() for h in raw_header]
    dup = sorted({h for h in header if h and header.count(h) > 1})
    if dup:
        raise ValueError(f"{p}: duplicate column(s) {dup}")
    if "roi" not in header:
        raise ValueError(f"{p}: no 'roi' column")
    volume_column, unit = _pick_volume_column(header, str(p))
    api = [h for h in header if _API_VOLUME.match(h)]
    if len(api) > 1:
        raise ValueError(f"{p}: several API volume columns {api}; keep one")
    api_column = api[0] if api else None
    api_unit = _API_VOLUME.match(api_column).group(1) if api_column else None
    message_column = next((h for h in ("message", "message_verbatim") if h in header), None)
    col = {h: i for i, h in enumerate(header)}

    rows: list[VolumeRow] = []
    skipped = 0
    for line, cells in enumerate(body, start=2):
        if not any(c.strip() for c in cells):
            continue
        cells = cells + [""] * (len(header) - len(cells))

        def cell(name: str | None, cells: list[str] = cells) -> str:
            return cells[col[name]].strip() if name is not None and name in col else ""

        context = f"{p}:{line}"
        source = cell("source").lower()
        if source and source != "suite":
            skipped += 1
            continue
        roi = cell("roi")
        if not roi:
            raise ValueError(f"{context}: empty roi")
        outcome = cell("import_outcome").lower() or None
        if outcome is not None and outcome not in IMPORT_OUTCOMES:
            raise ValueError(
                f"{context}: import_outcome {outcome!r} not one of {list(IMPORT_OUTCOMES)}"
            )
        decimals_raw = cell("volume_decimals")
        decimals: int | None = None
        if decimals_raw:
            if not decimals_raw.isdigit():
                raise ValueError(
                    f"{context}: volume_decimals {decimals_raw!r} is not a non-negative integer"
                )
            decimals = int(decimals_raw)
        rows.append(VolumeRow(
            roi=roi,
            volume=_number(cell(volume_column), volume_column, context),
            api_volume=_number(cell(api_column), api_column, context) if api_column else None,
            volume_decimals=decimals,
            import_outcome=outcome,
            message=cell(message_column) or None,
            line=line,
        ))

    names = [r.roi for r in rows]
    dup_rois = sorted({n for n in names if names.count(n) > 1})
    if dup_rois:
        raise ValueError(f"{p}: more than one row for ROI(s) {dup_rois}")
    return VolumeTable(
        path=str(p), rows=rows, volume_column=volume_column, unit=unit,
        api_column=api_column, api_unit=api_unit, skipped_rows=skipped,
    )


# --------------------------------------------------------------------------- end caps


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_in_polygon(pt: np.ndarray, xy: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    xs, ys = xy[:, 0], xy[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.count_nonzero(crosses & (x < x_at)) % 2)


def slice_area_even_odd(polygons: list[np.ndarray]) -> float:
    """Area enclosed by coplanar closed polygons under the even-odd rule.

    Each polygon counts with sign ``(-1)**depth``, where ``depth`` is how many
    other polygons on the slice contain it: an annulus is outer minus inner.
    A single keyhole polygon needs no help; its shoelace area is already the
    outer area minus the hole.
    """
    total = 0.0
    for i, poly in enumerate(polygons):
        depth = sum(
            1 for j, other in enumerate(polygons) if j != i and _point_in_polygon(poly[0], other)
        )
        total += _polygon_area(poly) * (-1) ** depth
    return total


def end_contour_areas_mm2(rtstruct_path: str | Path) -> dict[str, tuple[float, float]]:
    """``{roi: (area at the lowest-z contour, area at the highest-z contour)}`` in mm².

    Read from the RTSTRUCT the system imported, closed contours only; a ROI
    without closed contours is left out.
    """
    ds = pydicom.dcmread(str(rtstruct_path))
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    out: dict[str, tuple[float, float]] = {}
    for rc in getattr(ds, "ROIContourSequence", []):
        by_z: dict[float, list[np.ndarray]] = {}
        for c in getattr(rc, "ContourSequence", []):
            if str(c.ContourGeometricType).strip().upper() not in _CLOSED_TYPES:
                continue
            pts = np.asarray([float(v) for v in c.ContourData], dtype=np.float64).reshape(-1, 3)
            by_z.setdefault(round(float(pts[0, 2]), 6), []).append(pts[:, :2])
        if not by_z:
            continue
        zs = sorted(by_z)
        out[names[int(rc.ReferencedROINumber)]] = (
            slice_area_even_odd(by_z[zs[0]]),
            slice_area_even_odd(by_z[zs[-1]]),
        )
    return out


def end_cap_volume_mm3(end_areas_mm2: tuple[float, float], slice_spacing_mm: float) -> float:
    """Half a slice of the end contour's cross-section at each end:
    ``(A_first + A_last) * dz / 2``."""
    return (end_areas_mm2[0] + end_areas_mm2[1]) * slice_spacing_mm / 2.0


def reference_volume_mm3(analytic_mm3: float, cap_mm3: float, end_cap: str) -> float:
    """The volume a system following ``end_cap`` should report, in mm³.

    The fixture's reference CT has slices at integer z (1 mm default spacing),
    and every flat cap of a shipped primitive lies on a half-integer z. So the
    end contours sit half a slice inside the true caps.

    * ``half_slice``: each contour stands for a slab one slice thick, so the
      structure reaches half a slice beyond its end contours, which is where
      the true caps are. The reference is the closed-form analytic volume.
    * ``none``: the structure stops at its end contours. The reference is the
      analytic volume less half a slice of end cross-section at each end:
      ``V_analytic - (A_first + A_last) * dz / 2``.

    On the shipped planar bundle at 1 mm slices the two references differ by
    1.67 % (cube), 1.25 % (cylinder: 1/80), 0.99 % (torus), 0.83 % (straw) and
    at most 0.11 % on the sphere, ellipsoid and hollow sphere, whose end
    contours are small. Summing the contours of the shipped RTSTRUCT under each
    convention lands within 0.6 % of its own reference on every ROI.
    """
    if end_cap == "half_slice":
        return analytic_mm3
    if end_cap == "none":
        return analytic_mm3 - cap_mm3
    raise ValueError(f"unknown end_cap {end_cap!r}")


# --------------------------------------------------------------------------- grading


@dataclass
class VolumeResult:
    """One ROI's volume-only result (JSON-serialisable via :meth:`as_dict`)."""

    roi: str
    status: object                          # verify.Status
    geometric_type: str | None = None
    import_outcome: str | None = None
    message: str | None = None
    analytic_volume_cc: float | None = None
    end_cap_volume_cc: float | None = None
    reference_volume_cc: float | None = None
    reported_volume_cc: float | None = None
    graded_volume: str | None = None        # "displayed", "api" or None
    volume_decimals: int | None = None
    error_pct: float | None = None
    tolerance_pct: float | None = None
    display_quantisation_pct: float | None = None
    effective_tolerance_pct: float | None = None
    display_limited: bool = False
    absent_reason: str | None = None        # no_row, rejected, rejected_file, no_volume
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class VolumeOnlyReport:
    """The report of one volume-only run; the CLI and ``report.py`` treat it like ``Report``."""

    config_source: str
    summary: dict[str, int]
    results: list[VolumeResult]
    fixture: dict | None
    claims: dict
    report_only: list[str]
    settings: dict
    unmatched_rows: list[str] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        s = self.summary
        return s["failed"] + s["missing"] + s["geometry_mismatch"] == 0

    def as_dict(self) -> dict:
        from .verify import REPORT_VERSION

        return {
            "version": REPORT_VERSION,
            "mode": MODE,
            "config_source": self.config_source,
            "fixture": self.fixture,
            "claims": self.claims,
            "report_only": self.report_only,
            MODE: self.settings,
            "summary": self.summary,
            "results": [r.as_dict() for r in self.results],
            "unmatched_rows": self.unmatched_rows,
        }


def grade_volume(
    reported_mm3: float,
    reference_mm3: float,
    tolerance: float,
    *,
    quantum_mm3: float | None = None,
) -> tuple[bool, float, float, float]:
    """``(passed, error, quantisation, effective_tolerance)``, all relative (fractions).

    ``quantum_mm3`` is one unit in the last displayed decimal; ``None`` grades
    at the tolerance alone.
    """
    error = (reported_mm3 - reference_mm3) / reference_mm3
    q = 0.5 * quantum_mm3 / reference_mm3 if quantum_mm3 is not None else 0.0
    effective = max(tolerance, q)
    return abs(error) <= effective, error, q, effective


@dataclass(frozen=True)
class VolumeTarget:
    """One fixture ROI: what the grader needs to know about it."""

    roi: str
    geometric_type: str | None
    graded: bool
    analytic_mm3: float | None
    end_areas_mm2: tuple[float, float] | None


def _match_rows(
    table: VolumeTable, targets: list[VolumeTarget]
) -> tuple[dict[str, VolumeRow], list[str]]:
    """Map target ROI -> row, exact then case-insensitive; return unmatched row names."""
    by_name = {r.roi: r for r in table.rows}
    matched: dict[str, VolumeRow] = {}
    used: set[str] = set()
    for t in targets:
        if t.roi in by_name:
            matched[t.roi] = by_name[t.roi]
            used.add(t.roi)
    for t in targets:
        if t.roi in matched:
            continue
        hits = [n for n in by_name if n not in used and n.lower() == t.roi.lower()]
        if len(hits) == 1:
            warnings.warn(
                f"Volume row {hits[0]!r} matched ROI {t.roi!r} case-insensitively; "
                f"use the RTSTRUCT name {t.roi!r}.",
                stacklevel=3,
            )
            matched[t.roi] = by_name[hits[0]]
            used.add(hits[0])
    unmatched = [r.roi for r in table.rows if r.roi not in used]
    if unmatched:
        warnings.warn(
            f"{table.path}: row(s) for ROI(s) not in the fixture, ignored: {unmatched}",
            stacklevel=3,
        )
    return matched, unmatched


def grade_rows(
    table: VolumeTable,
    targets: list[VolumeTarget],
    config: ConformanceConfig,
    *,
    tolerance: float | None = None,
    slice_spacing_mm: float = 1.0,
    volume_source: str = "auto",
) -> tuple[list[VolumeResult], list[str]]:
    """Grade every target ROI against the table (no I/O). Returns ``(results, unmatched)``."""
    from .verify import Status

    tol = config.volume_tolerance if tolerance is None else tolerance
    end_cap = config.end_cap
    matched, unmatched = _match_rows(table, targets)
    results: list[VolumeResult] = []
    for t in targets:
        row = matched.get(t.roi)
        res = VolumeResult(
            roi=t.roi,
            status=Status.PASS,
            geometric_type=t.geometric_type,
            import_outcome=row.import_outcome if row else None,
            message=row.message if row else None,
            volume_decimals=row.volume_decimals if row else None,
            tolerance_pct=100.0 * tol,
        )
        results.append(res)

        # Which volume, if any, the row offers.
        reported_mm3: float | None = None
        quantum_mm3: float | None = None
        if row is None:
            res.absent_reason = "no_row"
        elif row.import_outcome in NOT_IMPORTED:
            res.absent_reason = row.import_outcome
        elif row.api_volume is not None and volume_source != "displayed":
            res.graded_volume = "api"
            reported_mm3 = row.api_volume * _UNIT_TO_MM3[table.api_unit]
        elif volume_source == "api":
            res.absent_reason = "no_api_volume"
        elif row.volume is not None:
            res.graded_volume = "displayed"
            reported_mm3 = row.volume * _UNIT_TO_MM3[table.unit]
            if row.volume_decimals is not None:
                quantum_mm3 = 10.0 ** -row.volume_decimals * _UNIT_TO_MM3[table.unit]
        else:
            res.absent_reason = "no_volume"
        if reported_mm3 is not None:
            res.reported_volume_cc = reported_mm3 / 1000.0

        # 1. nothing to grade and the type is not claimed.
        if reported_mm3 is None and not config.claims(t.geometric_type):
            res.status = Status.UNSUPPORTED_TYPE
            res.violations.append(
                f"no structure ({res.absent_reason}); geometric type "
                f"{t.geometric_type} is not claimed"
            )
            continue

        # 2. report-only: no enclosed volume, or listed, or ungraded in the manifest.
        no_volume = t.geometric_type in NO_VOLUME_TYPES or not t.analytic_mm3
        if no_volume or not t.graded or config.is_report_only(t.roi, t.geometric_type):
            res.status = Status.REPORT_ONLY
            if no_volume:
                res.analytic_volume_cc = 0.0
                res.violations.append("no enclosed volume; reported volume echoed, not graded")
            else:
                _grade_into(res, t, reported_mm3, quantum_mm3, tol, end_cap, slice_spacing_mm)
                res.status = Status.REPORT_ONLY
            continue

        # 3. nothing to grade.
        if reported_mm3 is None:
            res.status = Status.MISSING
            res.violations.append({
                "no_row": "no row for this ROI in the volume CSV",
                "rejected": "import_outcome rejected: the ROI is not in the system",
                "rejected_file": "import_outcome rejected_file: the RTSTRUCT was refused",
                "no_volume": "no volume reported",
                "no_api_volume": "no API volume reported (--volume-source api)",
            }[res.absent_reason])
            continue

        # 4. PASS / FAIL.
        _grade_into(res, t, reported_mm3, quantum_mm3, tol, end_cap, slice_spacing_mm)
    return results, unmatched


def _grade_into(
    res: VolumeResult,
    t: VolumeTarget,
    reported_mm3: float | None,
    quantum_mm3: float | None,
    tol: float,
    end_cap: str,
    slice_spacing_mm: float,
) -> None:
    from .verify import Status

    cap = end_cap_volume_mm3(t.end_areas_mm2, slice_spacing_mm) if t.end_areas_mm2 else 0.0
    ref = reference_volume_mm3(t.analytic_mm3, cap, end_cap)
    res.analytic_volume_cc = t.analytic_mm3 / 1000.0
    res.end_cap_volume_cc = cap / 1000.0
    res.reference_volume_cc = ref / 1000.0
    if reported_mm3 is None:
        return
    passed, err, q, eff = grade_volume(reported_mm3, ref, tol, quantum_mm3=quantum_mm3)
    res.error_pct = 100.0 * err
    res.display_quantisation_pct = 100.0 * q
    res.effective_tolerance_pct = 100.0 * eff
    res.display_limited = q > tol
    res.status = Status.PASS if passed else Status.FAIL
    if not passed:
        res.violations.append(
            f"volume error {100.0 * err:+.3f} % exceeds {100.0 * eff:.3f} %"
        )


# --------------------------------------------------------------------------- fixture


def fixture_volume_targets(
    fixture_dir: str | Path, manifest: FixtureManifest, bundles: list[str]
) -> list[VolumeTarget]:
    """Every ROI of the selected bundles, with its analytic volume and end-contour areas.

    Each bundle's RTSTRUCT is checked against its manifest SHA-256 first.
    """
    from .verify import FixtureIntegrityError, _sha256

    root = Path(fixture_dir)
    areas: dict[str, tuple[float, float]] = {}
    for name in bundles:
        entry = manifest.bundles[name]
        path = root / entry.rtstruct_path
        if not path.is_file():
            raise FileNotFoundError(f"RTSTRUCT of bundle {name!r} not found at {path}")
        if _sha256(path) != entry.rtstruct_sha256:
            raise FixtureIntegrityError(
                f"{path}: SHA-256 does not match the manifest; the RTSTRUCT changed "
                "after generation. Re-run `rtmask-conformance generate`."
            )
        areas.update(end_contour_areas_mm2(path))

    targets: list[VolumeTarget] = []
    for roi, entry in manifest.rois.items():
        if entry.bundle not in bundles:
            continue
        analytic: float | None = None
        if entry.spec_path is not None:
            spec_file = root / entry.spec_path
            if not spec_file.is_file():
                raise FileNotFoundError(f"spec for {roi!r} not found at {spec_file}")
            analytic = json.loads(spec_file.read_text(encoding="utf-8")).get(
                "analytical_volume_mm3"
            )
        targets.append(VolumeTarget(
            roi=roi,
            geometric_type=entry.geometric_type,
            graded=entry.graded,
            analytic_mm3=float(analytic) if analytic else None,
            end_areas_mm2=areas.get(roi),
        ))
    return targets


def verify_volumes(
    volumes_csv: str | Path,
    fixture_dir: str | Path,
    config: ConformanceConfig | None = None,
    *,
    bundles: list[str] | None = None,
    tolerance: float | None = None,
    volume_source: str = "auto",
) -> VolumeOnlyReport:
    """Grade the volumes in ``volumes_csv`` against the fixture at ``fixture_dir``."""
    from .verify import _select_bundles, fixture_summary, summarize

    config = config or load_config()
    tol = config.volume_tolerance if tolerance is None else tolerance
    if not 0 < tol < 1:
        raise ValueError(f"volume tolerance {tol!r} must be a fraction in (0, 1)")
    root = Path(fixture_dir)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"No manifest.json in {root}; pass a generated fixture")
    csv_path = Path(volumes_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Volume CSV not found: {csv_path}")

    manifest = read_manifest(root)
    selected = _select_bundles(manifest, bundles)
    targets = fixture_volume_targets(root, manifest, selected)
    table = read_volumes_csv(csv_path)
    if volume_source not in VOLUME_SOURCES:
        raise ValueError(f"volume_source {volume_source!r} not one of {list(VOLUME_SOURCES)}")
    if volume_source == "api" and table.api_column is None:
        raise ValueError(f"{csv_path}: --volume-source api needs a <system>_volume_cc_api column")
    spacing_z = float(manifest.geometry["spacing"][2])
    results, unmatched = grade_rows(
        table, targets, config, tolerance=tol, slice_spacing_mm=spacing_z,
        volume_source=volume_source,
    )
    return VolumeOnlyReport(
        config_source=config.config_source,
        summary=summarize(results),
        results=results,
        fixture=fixture_summary(root, manifest, selected),
        claims={**config.claims_dict(), "end_cap": config.end_cap},
        report_only=list(config.report_only),
        settings={
            "input": str(csv_path),
            "volume_column": table.volume_column,
            "unit": table.unit,
            "api_column": table.api_column,
            "volume_source": volume_source,
            "tolerance": tol,
            "end_cap": config.end_cap,
            "slice_spacing_mm": spacing_z,
            "skipped_rows_other_source": table.skipped_rows,
        },
        unmatched_rows=unmatched,
    )


def write_volumes_template(fixture_dir: str | Path, manifest: FixtureManifest) -> Path:
    """Write ``volumes_template.csv`` (every manifest ROI, empty cells) into the fixture."""
    out = Path(fixture_dir) / TEMPLATE_NAME
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(TEMPLATE_COLUMNS)
        for roi in manifest.rois:
            w.writerow([roi] + [""] * (len(TEMPLATE_COLUMNS) - 1))
    return out


__all__ = [
    "MODE",
    "VolumeOnlyReport",
    "VolumeResult",
    "end_cap_volume_mm3",
    "end_contour_areas_mm2",
    "grade_volume",
    "read_volumes_csv",
    "reference_volume_mm3",
    "verify_volumes",
    "write_volumes_template",
]
