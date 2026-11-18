"""Render a :class:`Report` as JSON or a fixed-width plain-text table.

No external dependencies (no ``tabulate``, no ``rich``). ANSI color is only
emitted when stdout is a TTY.

Badges: ``PASS``, ``FAIL``, ``MISS`` (MISSING), ``GEOM`` (GEOMETRY_MISMATCH),
``UNSUP`` (UNSUPPORTED_TYPE) and ``REPORT`` (REPORT_ONLY). The text report
has one table per kind of ground truth: binary masks (Dice, surface metrics,
volume), points and curves (their own metrics, distances in mm). The JSON report
(``Report.as_dict``) carries ``version``, ``config_source``, ``fixture``,
``claims``, ``report_only``, ``summary`` and per-ROI ``results``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .grading_pointline import POINTLINE_TRUTH_KINDS
from .verify import Report, ResultRecord, Status


def write_json(report: Report, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def _status_badge(status: Status) -> str:
    if status == Status.PASS:
        return _color("PASS  ", "32")  # green
    if status == Status.FAIL:
        return _color("FAIL  ", "31")  # red
    if status == Status.MISSING:
        return _color("MISS  ", "33")  # yellow
    if status == Status.GEOMETRY_MISMATCH:
        return _color("GEOM  ", "35")  # magenta
    if status == Status.UNSUPPORTED_TYPE:
        return _color("UNSUP ", "36")  # cyan
    if status == Status.REPORT_ONLY:
        return _color("REPORT", "34")  # blue
    return status.value


def _fmt_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "  -   "
    if kind == "ratio":
        return f"{value:.4f}"
    if kind == "mm":
        return f"{value:6.3f}"
    return f"{value:.4f}"


def render_text(report: Report) -> str:
    """Return a human-readable plain-text rendering of ``report``."""
    if getattr(report, "settings", None) is not None:  # volume-only report
        return render_volume_text(report)
    lines: list[str] = []
    lines.append(
        f"rtmask-conformance verify  config={report.config_source}  "
        f"{report.summary['passed']}/{report.summary['total']} passed"
    )
    if report.summary.get("missing"):
        lines.append(f"  {_color('MISSING', '33')}: {report.summary['missing']}")
    if report.summary.get("geometry_mismatch"):
        lines.append(f"  {_color('GEOMETRY_MISMATCH', '35')}: {report.summary['geometry_mismatch']}")
    if report.summary.get("failed"):
        lines.append(f"  {_color('FAILED', '31')}: {report.summary['failed']}")
    if report.summary.get("unsupported_type"):
        lines.append(
            f"  {_color('UNSUPPORTED_TYPE', '36')}: {report.summary['unsupported_type']}"
            "  (type not claimed; not a failure)"
        )
    if report.summary.get("report_only"):
        lines.append(
            f"  {_color('REPORT_ONLY', '34')}: {report.summary['report_only']}"
            "  (scored, not graded; not a failure)"
        )
    lines.append("")

    mask_rows = [r for r in report.results if r.truth_kind not in POINTLINE_TRUTH_KINDS]
    tables = [
        _mask_table(mask_rows),
        _pointline_table([r for r in report.results if r.truth_kind == "points"], _POINT_COLUMNS),
        _pointline_table([r for r in report.results if r.truth_kind == "curve"], _CURVE_COLUMNS),
    ]
    for i, table in enumerate(t for t in tables if t):
        if i:
            lines.append("")
        lines.extend(table)

    failures = [r for r in report.results if r.status != Status.PASS]
    if failures:
        lines.append("")
        lines.append("Details:")
        for r in failures:
            note = "  (for information; not graded)" if r.status == Status.REPORT_ONLY else ""
            lines.append(f"  {r.roi} [{r.status.value}]{note}")
            if r.geometry_diagnostic:
                lines.append(f"    geometry: {r.geometry_diagnostic}")
            for v in r.violations:
                lines.append(f"    - {v}")
    return "\n".join(lines)


def _mask_table(rows: list[ResultRecord]) -> list[str]:
    """Binary-mask ROIs: Dice, surface DSC, HD95, MSD, volume error. Empty if no rows."""
    if not rows:
        return []
    headers = ["status", "ROI                                ", " dice ", "sDSC1 ", "HD95mm", "MSD mm", "dV%   "]
    lines = [" ".join(headers), "-" * len(" ".join(headers))]
    for r in rows:
        m = r.metrics or {}
        rel = m.get("volume_rel_err")
        rel_pct = (rel * 100.0) if rel is not None else None
        cells = [
            _status_badge(r.status),
            f"{r.roi:<35}",
            _fmt_metric(m.get("dice"), "ratio"),
            _fmt_metric(m.get("surface_dice_1mm"), "ratio"),
            _fmt_metric(m.get("hd95_mm"), "mm"),
            _fmt_metric(m.get("msd_mm"), "mm"),
            f"{rel_pct:6.2f}" if rel_pct is not None else "  -   ",
        ]
        lines.append(" ".join(cells))
    return lines


#: (header, metric key, format kind) for the point and the curve table.
_POINT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("recall", "point_recall", "ratio"),
    ("FPrate", "point_fp_rate", "ratio"),
    ("dMeanm", "point_mean_displacement_mm", "mm"),
    ("dMaxmm", "point_max_displacement_mm", "mm"),
)
_CURVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("covRec", "curve_coverage_recall", "ratio"),
    ("dMeanm", "curve_mean_distance_mm", "mm"),
    ("dMaxmm", "curve_max_distance_mm", "mm"),
    ("v2c mm", "curve_voxel_to_curve_mean_mm", "mm"),
    ("arcL% ", "arc_length_error_pct", "pct"),
)


def _pointline_table(
    rows: list[ResultRecord], columns: tuple[tuple[str, str, str], ...]
) -> list[str]:
    """Point (or curve) ROIs with their own metric columns. Empty if no rows."""
    if not rows:
        return []
    headers = ["status", "ROI                                ", *(h for h, _, _ in columns)]
    lines = [" ".join(headers), "-" * len(" ".join(headers))]
    for r in rows:
        m = r.metrics or {}
        cells = [_status_badge(r.status), f"{r.roi:<35}"]
        for _, key, kind in columns:
            v = m.get(key)
            cells.append(f"{v:6.2f}" if kind == "pct" and v is not None else _fmt_metric(v, kind))
        lines.append(" ".join(cells))
    return lines


def render_volume_text(report) -> str:
    """Plain-text rendering of a volume-only report (``verify --volumes``).

    One row per ROI: status, ROI, analytic volume, reference volume (analytic
    adjusted for the declared end cap), reported volume, error and the
    tolerance it was graded at, all volumes in cc. A trailing ``*`` marks a
    display-limited row; ``api`` marks a row graded on its API volume.
    """
    st, s = report.settings, report.summary
    lines = [
        f"rtmask-conformance verify --volumes  config={report.config_source}  "
        f"{s['passed']}/{s['total']} passed  (volume only)",
        f"  input {st['input']}  column={st['volume_column']}  "
        f"end_cap={st['end_cap']}  tolerance={100.0 * st['tolerance']:.2f} %",
    ]
    for key, label in (("failed", "FAILED"), ("missing", "MISSING"),
                       ("unsupported_type", "UNSUPPORTED_TYPE"), ("report_only", "REPORT_ONLY")):
        if s.get(key):
            lines.append(f"  {label}: {s[key]}")
    lines.append("")

    def cc(v: float | None) -> str:
        return f"{v:10.4f}" if v is not None else "      -   "

    headers = ["status", "ROI                                ", "analyt cc ", "ref cc    ",
               "reported  ", "err %  ", "tol %  ", "graded"]
    lines.append(" ".join(headers))
    lines.append("-" * len(" ".join(headers)))
    for r in report.results:
        err = f"{r.error_pct:+7.3f}" if r.error_pct is not None else "   -   "
        tol = (f"{r.effective_tolerance_pct:6.3f}{'*' if r.display_limited else ' '}"
               if r.effective_tolerance_pct is not None else "   -   ")
        lines.append(" ".join([
            _status_badge(r.status), f"{r.roi:<35}", cc(r.analytic_volume_cc),
            cc(r.reference_volume_cc), cc(r.reported_volume_cc), err, tol,
            r.graded_volume or r.absent_reason or "-",
        ]))
    if any(r.display_limited for r in report.results):
        lines.append("  * display-limited: half a unit in the last displayed decimal exceeds "
                     "the tolerance; graded at the display quantisation")

    details = [r for r in report.results if r.status != Status.PASS or r.message]
    if details or report.unmatched_rows:
        lines.append("")
        lines.append("Details:")
        for r in details:
            outcome = f"  import_outcome={r.import_outcome}" if r.import_outcome else ""
            lines.append(f"  {r.roi} [{r.status.value}]{outcome}")
            for v in r.violations:
                lines.append(f"    - {v}")
            if r.message:
                lines.append(f"    message: {r.message}")
        if report.unmatched_rows:
            lines.append(f"  rows for ROIs not in the fixture (ignored): "
                         f"{', '.join(report.unmatched_rows)}")
    return "\n".join(lines)


def print_report(report: Report) -> None:
    """Print the plain-text rendering to stdout."""
    print(render_text(report))


__all__ = ["ResultRecord", "print_report", "render_text", "render_volume_text", "write_json"]
