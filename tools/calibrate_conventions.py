#!/usr/bin/env python3
"""Measure the boundary-convention references and calibrate their thresholds.

Usage:

    rtmask-conformance generate ./fixture            # default n_quadrature 8
    python tools/calibrate_conventions.py ./fixture [--supersample 8]

Runs on a fixture written by ``rtmask-conformance generate``, nothing else.
For every binary-mask ROI of the ``planar`` bundle (the ``xor`` and
``keyhole`` twins share their parent's inside test, so their references are
the parent's, voxel for voxel) it prints:

1. **References against the closed form.** The partial-volume field's sum
   times the voxel volume, and each convention's voxel count times the voxel
   volume, as a signed residual (%) against the primitive's analytic volume.
2. **Ideal 3-D rasterisations across conventions.** Each convention's own
   reference, used as a prediction, scored against each convention's
   reference with that convention's thresholds (the shipped YAML, resolved
   through ``thresholds.load_config``). The diagonal passes with Dice 1 by
   construction; an off-diagonal cell says whether the gate tells the two
   conventions apart on that ROI.
3. **Contour-based rasterisations.** What a converter reading the fixture's
   RTSTRUCT can compute: per slice, from the planar contours only, the voxels
   whose centre is inside (``centre_in`` by scan line), or whose in-plane
   area fraction, from ``--supersample`` x ``--supersample`` sub-pixel centres,
   is >= 0.5 (``pv_ge_0.5``) or > 0 (``pv_gt_0``). Each is scored against its
   own convention's reference and the others'. These are the suite's own
   positive calibrators: a correct converter of each convention should pass
   under its own convention, with the margin printed.
4. **Volume ceilings.** For ``pv_gt_0`` and ``centre_in``, per ROI, the
   ceiling the calibration rule gives and the one the shipped YAML resolves to.
   Rule: start at the ``pv_ge_0.5`` value; where the contour-based
   rasterisation of the convention reads more than two thirds of it, raise the
   ceiling to 1.5 x that error, rounded up to 0.5 percentage points (the
   ``pointline`` family rule's margin and step); never lower it. The table also
   prints the smallest volume error of another convention's ideal mask against
   the convention's reference: a ceiling below it keeps the conventions apart.

Contours are filled even-odd (the ``planar`` bundle's two coplanar rings of the
hollow primitives). Exit status 1 if a contour-based rasterisation fails under
its own convention with the shipped thresholds, or if a shipped volume ceiling
differs from the rule's.

The calibration is only meaningful on a fixture at the generator's default
quadrature (``thresholds.CONVENTION_CALIBRATION_N_QUADRATURE``): the
``pv_gt_0`` reference grows with it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon2mask

from rtmask_conformance.manifest import read_manifest
from rtmask_conformance.thresholds import (
    CONVENTION_CALIBRATION_N_QUADRATURE,
    CONVENTIONS,
    DEFAULT_CONVENTION,
    load_config,
)
from rtmask_conformance.verify import _compute_all_metrics, _evaluate_against_thresholds

PAD = 3  # background voxels kept around the union bounding box when cropping

METRICS = ("dice", "surface_dice_1mm", "hd95_mm", "msd_mm", "volume_rel_err")

MARGIN = 1.5     # over the convention's own contour-based error
STEP = 0.005     # ceilings rounded up to 0.5 percentage points


def rule_ceiling(start: float, own_err: float) -> float:
    """The calibration rule of section 4 (module docstring)."""
    proposed = math.ceil(round(MARGIN * own_err / STEP, 6)) * STEP
    return round(max(start, proposed), 4)


def _read(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _crop_box(*masks: np.ndarray) -> tuple[slice, ...]:
    nz = np.argwhere(np.logical_or.reduce([m > 0 for m in masks]))
    lo = np.maximum(nz.min(axis=0) - PAD, 0)
    hi = np.minimum(nz.max(axis=0) + PAD + 1, masks[0].shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))


def score(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    box = _crop_box(pred, ref)
    m = _compute_all_metrics(pred[box].astype(np.uint8), ref[box].astype(np.uint8), spacing)
    return {k: m[k] for k in METRICS}


def contour_rasterisations(
    rtstruct: Path, roi: str, shape_zyx: tuple[int, int, int], geometry: dict, k: int
) -> dict[str, np.ndarray]:
    """Per-slice rasterisations of ``roi``'s contours under the three conventions."""
    ds = pydicom.dcmread(str(rtstruct))
    number = next(
        r.ROINumber for r in ds.StructureSetROISequence if r.ROIName == roi
    )
    item = next(c for c in ds.ROIContourSequence if c.ReferencedROINumber == number)
    ox, oy, oz = geometry["origin"]
    sx, sy, sz = geometry["spacing"]
    nz, ny, nx = shape_zyx
    by_slice: dict[int, list[np.ndarray]] = {}
    for c in item.ContourSequence:
        pts = np.asarray(c.ContourData, dtype=np.float64).reshape(-1, 3)
        z = int(round((pts[0, 2] - oz) / sz))
        # (row, col) in continuous voxel-index units.
        rc = np.stack([(pts[:, 1] - oy) / sy, (pts[:, 0] - ox) / sx], axis=1)
        by_slice.setdefault(z, []).append(rc)

    out = {name: np.zeros(shape_zyx, dtype=np.uint8) for name in CONVENTIONS}
    for z, rings in by_slice.items():
        allpts = np.concatenate(rings)
        r0 = max(int(np.floor(allpts[:, 0].min())) - 2, 0)
        r1 = min(int(np.ceil(allpts[:, 0].max())) + 3, ny)
        c0 = max(int(np.floor(allpts[:, 1].min())) - 2, 0)
        c1 = min(int(np.ceil(allpts[:, 1].max())) + 3, nx)
        h, w = r1 - r0, c1 - c0
        centre = np.zeros((h, w), dtype=bool)
        fine = np.zeros((h * k, w * k), dtype=bool)
        for rc in rings:  # even-odd: XOR the rings
            local = rc - np.array([r0, c0])
            centre ^= polygon2mask((h, w), local)
            fine ^= polygon2mask((h * k, w * k), (local + 0.5) * k - 0.5)
        frac = fine.reshape(h, k, w, k).mean(axis=(1, 3))
        out["centre_in"][z, r0:r1, c0:c1] = centre
        out["pv_ge_0.5"][z, r0:r1, c0:c1] = frac >= 0.5
        out["pv_gt_0"][z, r0:r1, c0:c1] = frac > 0
    return out


def _fmt(m: dict) -> str:
    return (f"{m['dice']:.4f} {m['surface_dice_1mm']:.4f} {m['hd95_mm']:5.2f} "
            f"{m['msd_mm']:5.3f} {100 * m['volume_rel_err']:6.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--supersample", type=int, default=8,
                    help="in-plane sub-pixels per axis for the contour area fraction")
    ap.add_argument("--bundle", default="planar")
    args = ap.parse_args(argv)

    manifest = read_manifest(args.fixture)
    config = load_config()
    spacing = tuple(float(v) for v in manifest.geometry["spacing"])
    voxel_mm3 = float(np.prod(spacing))
    rtstruct = args.fixture / manifest.bundles[args.bundle].rtstruct_path
    print(f"fixture {args.fixture}  n_quadrature={manifest.n_quadrature}  "
          f"supersample={args.supersample}")
    if manifest.n_quadrature != CONVENTION_CALIBRATION_N_QUADRATURE:
        print(f"warning: thresholds are calibrated at n_quadrature="
              f"{CONVENTION_CALIBRATION_N_QUADRATURE}", file=sys.stderr)

    rows_ref, rows_ideal, rows_contour = [], [], []
    own_failures: list[str] = []
    for roi in manifest.bundles[args.bundle].rois:
        entry = manifest.rois[roi]
        if entry.truth_kind != "binary_mask":
            continue
        spec = json.loads((args.fixture / entry.spec_path).read_text(encoding="utf-8"))
        v_true = float(spec["analytical_volume_mm3"])
        field = _read(args.fixture / entry.pv_field["path"]).astype(np.float64)
        refs = {c: _read(args.fixture / entry.truth_for(c)[0]) for c in CONVENTIONS}
        res = {"field_sum": 100 * (field.sum() * voxel_mm3 - v_true) / v_true}
        for c in CONVENTIONS:
            res[c] = 100 * (int(refs[c].sum()) * voxel_mm3 - v_true) / v_true
        rows_ref.append((roi, v_true, res))

        for pc in CONVENTIONS:
            for rc in CONVENTIONS:
                m = score(refs[pc], refs[rc], spacing)
                v = _evaluate_against_thresholds(m, config.thresholds_for(roi, rc))
                rows_ideal.append((roi, pc, rc, m, v))

        rasters = contour_rasterisations(
            rtstruct, roi, refs[CONVENTIONS[0]].shape, manifest.geometry, args.supersample
        )
        for pc in CONVENTIONS:
            for rc in CONVENTIONS:
                m = score(rasters[pc], refs[rc], spacing)
                t = config.thresholds_for(roi, rc)
                v = _evaluate_against_thresholds(m, t)
                rows_contour.append((roi, pc, rc, m, v, t))
                if pc == rc and v:
                    own_failures.append(f"{roi}: contour {pc} fails its own convention: {v}")

    print("\n1. References against the closed form (signed residual, % of analytic volume)")
    print(f"{'ROI':<16}{'V mm3':>12}{'sum(PV)':>10}" + "".join(f"{c:>11}" for c in CONVENTIONS))
    for roi, v, res in rows_ref:
        print(f"{roi:<16}{v:12.1f}{res['field_sum']:+10.4f}"
              + "".join(f"{res[c]:+11.4f}" for c in CONVENTIONS))

    header = "dice   sDSC1  HD95  MSD    dV%   verdict"
    print("\n2. Ideal 3-D rasterisation (prediction convention) vs reference (convention)")
    print(f"{'ROI':<16}{'pred':<11}{'ref':<11}{header}")
    for roi, pc, rc, m, v in rows_ideal:
        verdict = "PASS" if not v else "FAIL " + "; ".join(x.split(" ")[0] for x in v)
        print(f"{roi:<16}{pc:<11}{rc:<11}{_fmt(m)}  {verdict}")

    print("\n3. Contour-based rasterisation (rule) vs reference (convention)")
    print(f"{'ROI':<16}{'rule':<11}{'ref':<11}{header}")
    for roi, pc, rc, m, v, t in rows_contour:
        verdict = "PASS" if not v else "FAIL " + "; ".join(x.split(" ")[0] for x in v)
        margin = ""
        if pc == rc:
            margin = (f"   margins: vol {100 * (t.volume_rel_err - m['volume_rel_err']):+.2f} pp,"
                      f" dice {m['dice'] - t.dice:+.4f}, hd95 {t.hd95_mm - m['hd95_mm']:+.2f} mm,"
                      f" msd {t.msd_mm - m['msd_mm']:+.3f} mm,"
                      f" sDSC {m['surface_dice_1mm'] - t.surface_dice_1mm:+.4f}")
        print(f"{roi:<16}{pc:<11}{rc:<11}{_fmt(m)}  {verdict}{margin}")

    print("\n4. Volume ceilings (%): start = pv_ge_0.5 value; own = contour-based error under "
          "its own convention; rule = max(start, 1.5 x own rounded up to 0.5); shipped = YAML; "
          "cross = smallest error of another convention's ideal mask")
    print(f"{'ROI':<16}{'convention':<11}{'start':>7}{'own':>7}{'rule':>7}{'shipped':>8}"
          f"{'cross':>7}")
    mismatches: list[str] = []
    for roi in dict.fromkeys(r[0] for r in rows_contour):
        for c in CONVENTIONS:
            if c == DEFAULT_CONVENTION:
                continue
            start = config.thresholds_for(roi, DEFAULT_CONVENTION).volume_rel_err
            own = next(m["volume_rel_err"] for r, pc, rc, m, _, _ in rows_contour
                       if r == roi and pc == c and rc == c)
            cross = min(m["volume_rel_err"] for r, pc, rc, m, _ in rows_ideal
                        if r == roi and rc == c and pc != c)
            rule = rule_ceiling(start, own)
            shipped = config.thresholds_for(roi, c).volume_rel_err
            flag = "" if math.isclose(rule, shipped) else "   <- shipped differs from rule"
            if flag:
                mismatches.append(f"{roi} {c}: rule {rule} shipped {shipped}")
            print(f"{roi:<16}{c:<11}{100 * start:7.2f}{100 * own:7.2f}{100 * rule:7.2f}"
                  f"{100 * shipped:8.2f}{100 * cross:7.2f}{flag}")

    for line in own_failures + mismatches:
        print(line, file=sys.stderr)
    return 1 if own_failures or mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
