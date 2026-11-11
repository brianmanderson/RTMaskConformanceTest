#!/usr/bin/env python3
"""Propose the ``pointline`` threshold block from a forward-benchmark results CSV.

Usage:

    python tools/calibrate_pointline_thresholds.py path/to/forward.csv [--margin 1.5]

The CSV is the per-(tool, primitive) table the upstream benchmark harness
writes for its forward pass (columns ``tool``, ``primitive``,
``geometric_type``, ``empty_output`` and the ``point_*``, ``curve_*`` and
``arc_length_*`` metric columns). It is read, never copied into this
repository.

Calibration rule:

1. Curves. A tool *traces a thin path* when, on every ``OPEN_NONPLANAR``
   primitive, its mask is non-empty and every marked voxel lies within one
   voxel diagonal of the analytic curve (``curve_voxel_to_curve_max_mm`` at
   most the diagonal). Tools whose masks are non-empty but violate that, for
   example by filling the polygon the curve's vertices span, are the negative
   calibrators.
2. Points. A tool *marks the points* when, on every ``POINT`` primitive, its
   mask is non-empty and finds every point (``point_recall`` = 1).
3. For each graded lower-is-better metric, the ceiling is the worst value any
   positive calibrator reaches on any primitive of that family, times
   ``--margin``, rounded up to a fixed step (0.05 mm, 0.5 %). One ceiling per
   family, written into every ROI's sub-block: several primitives score
   exactly 0 for every positive calibrator (an axis-aligned needle, a point on
   a voxel centre), so a per-primitive multiple would leave no tolerance.
4. Recall floors are the lowest value a positive calibrator reaches, with no
   margin: recall counts discrete points or curve samples, and one missed
   point is a defect.

The script prints the proposed YAML block, then a per-tool, per-primitive
pass/fail table under it and the separation between the two groups. Exit
status 1 if any positive calibrator would fail or any negative one would pass.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

VOXEL_DIAGONAL_MM = math.sqrt(3.0)  # the fixture grid is 1 mm isotropic

POINT_FLOORS = ("point_recall",)
POINT_CEILINGS = ("point_max_displacement_mm",)
CURVE_CEILINGS = (
    "arc_length_error_pct",
    "curve_max_distance_mm",
    "curve_voxel_to_curve_mean_mm",
)
STEP = {"arc_length_error_pct": 0.5}  # every other ceiling: 0.05 mm
DEFAULT_STEP = 0.05


def _num(v: str) -> float | None:
    if v is None or v.strip() == "" or v.strip().lower() == "nan":
        return None
    return float(v)


def _empty(row: dict) -> bool:
    return str(row.get("empty_output", "")).strip().lower() == "true"


def _ceil_to(value: float, step: float) -> float:
    return round(math.ceil(value / step - 1e-9) * step, 6)


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("direction", "forward") == "forward"]


def classify(rows: list[dict], gtype: str, ok) -> tuple[list[str], list[str]]:
    """(positive calibrators, other tools that wrote a non-empty mask on this family)."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["geometric_type"] == gtype:
            by_tool[r["tool"]].append(r)
    positive, negative = [], []
    for tool, rs in by_tool.items():
        if all(not _empty(r) and ok(r) for r in rs):
            positive.append(tool)
        elif any(not _empty(r) for r in rs):
            negative.append(tool)
    return sorted(positive), sorted(negative)


def _curve_ok(r: dict) -> bool:
    v = _num(r["curve_voxel_to_curve_max_mm"])
    return v is not None and v <= VOXEL_DIAGONAL_MM


def _point_ok(r: dict) -> bool:
    return _num(r["point_recall"]) == 1.0


def propose(rows: list[dict], margin: float) -> tuple[dict, dict]:
    """Return ({roi: {metric: threshold}}, calibration notes)."""
    block: dict[str, dict[str, float]] = {}
    notes: dict[str, dict] = {}
    families = (
        ("POINT", _point_ok, POINT_FLOORS, POINT_CEILINGS),
        ("OPEN_NONPLANAR", _curve_ok, (), CURVE_CEILINGS),
    )
    for gtype, ok, floors, ceilings in families:
        positive, negative = classify(rows, gtype, ok)
        fam = [r for r in rows if r["geometric_type"] == gtype]
        pos_rows = [r for r in fam if r["tool"] in positive]
        limits: dict[str, float] = {}
        for m in floors:
            vals = [_num(r[m]) for r in pos_rows if _num(r[m]) is not None]
            limits[m] = min(vals)
            notes[m] = {"calibrator_min": min(vals), "margin": "none"}
        for m in ceilings:
            vals = [_num(r[m]) for r in pos_rows if _num(r[m]) is not None]
            worst = max(vals)
            limits[m] = _ceil_to(worst * margin, STEP.get(m, DEFAULT_STEP))
            notes[m] = {"calibrator_max": worst, "margin": margin, "ceiling": limits[m]}
        notes[gtype] = {"positive": positive, "negative": negative}
        for prim in dict.fromkeys(r["primitive"] for r in fam):
            block[prim] = dict(limits)
    return block, notes


def grade(row: dict, limits: dict[str, float]) -> tuple[str, list[str]]:
    if _empty(row):
        return "FAIL", ["empty mask"]
    bad = []
    for m, lim in limits.items():
        v = _num(row.get(m, ""))
        if v is None:
            bad.append(f"{m} n/a")
        elif m in POINT_FLOORS and v < lim:
            bad.append(f"{m} {v:.4g} < {lim}")
        elif m not in POINT_FLOORS and v > lim:
            bad.append(f"{m} {v:.4g} > {lim}")
    return ("FAIL" if bad else "PASS"), bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="forward-pass results CSV")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="multiplier on the worst positive-calibrator value (default 1.5)")
    args = ap.parse_args(argv)

    rows = load(args.csv)
    block, notes = propose(rows, args.margin)

    print("pointline:")
    for prim, limits in block.items():
        print(f"  {prim}:")
        for m, v in limits.items():
            print(f"    {m}: {v}")

    print("\n# calibration")
    for k, v in notes.items():
        print(f"#   {k}: {v}")

    print("\n# per-tool verdicts under the proposed block")
    inconsistent = 0
    for gtype in ("POINT", "OPEN_NONPLANAR"):
        positive = set(notes[gtype]["positive"])
        negative = set(notes[gtype]["negative"])
        for r in rows:
            if r["geometric_type"] != gtype:
                continue
            verdict, why = grade(r, block[r["primitive"]])
            group = "positive" if r["tool"] in positive else (
                "negative" if r["tool"] in negative else "no-output")
            if (group == "positive" and verdict == "FAIL") or (
                    group == "negative" and verdict == "PASS" and gtype == "OPEN_NONPLANAR"):
                inconsistent += 1
            print(f"#   {r['tool']:<20} {r['primitive']:<38} {group:<9} {verdict}  "
                  f"{'; '.join(why)}")

    print("\n# separation on curves (ceiling vs the negative calibrators' range)")
    for m in CURVE_CEILINGS:
        neg = [_num(r[m]) for r in rows if r["geometric_type"] == "OPEN_NONPLANAR"
               and r["tool"] in notes["OPEN_NONPLANAR"]["negative"] and _num(r[m]) is not None]
        if neg:
            print(f"#   {m}: ceiling {notes[m]['ceiling']}, "
                  f"negative calibrators {min(neg):.4g} .. {max(neg):.4g}")
    return 1 if inconsistent else 0


if __name__ == "__main__":
    sys.exit(main())
