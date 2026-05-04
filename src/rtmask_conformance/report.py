"""Render a :class:`Report` as JSON or a fixed-width plain-text table.

No external dependencies (no ``tabulate``, no ``rich``). ANSI color is only
emitted when stdout is a TTY.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    lines.append("")

    headers = ["status", "ROI                                ", " dice ", "sDSC1 ", "HD95mm", "MSD mm", "dV%   "]
    lines.append(" ".join(headers))
    lines.append("-" * len(" ".join(headers)))

    for r in report.results:
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

    failures = [r for r in report.results if r.status != Status.PASS]
    if failures:
        lines.append("")
        lines.append("Details:")
        for r in failures:
            lines.append(f"  {r.roi} [{r.status.value}]")
            if r.geometry_diagnostic:
                lines.append(f"    geometry: {r.geometry_diagnostic}")
            for v in r.violations:
                lines.append(f"    - {v}")
    return "\n".join(lines)


def print_report(report: Report) -> None:
    """Print the plain-text rendering to stdout."""
    print(render_text(report))


__all__ = ["ResultRecord", "print_report", "render_text", "write_json"]
