"""argparse-based CLI: ``rtmask-conformance generate ...`` and ``... verify ...``.

Exit codes:
    0   no ROI failed: every ROI is PASS, UNSUPPORTED_TYPE or REPORT_ONLY
        (or generate succeeded)
    1   any FAIL / MISSING / GEOMETRY_MISMATCH (verify only)
    2   usage / IO error, a malformed --config, or fixture ground truth that
        fails its manifest hash check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .generate import GenerateOptions, generate_fixture
from .manifest import read_manifest
from .report import print_report, write_json
from .thresholds import load_config
from .verify import Report, verify_fixture, verify_predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtmask-conformance",
        description=(
            "Universal conformance test suite for DICOM-RT-to-NIfTI converters. "
            "Generate a synthetic CT + RTSTRUCT + analytic ground-truth NIfTIs, "
            "then verify your tool's per-ROI predictions against them."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser(
        "generate",
        help="Write the synthetic CT, RTSTRUCT, and analytic GT NIfTIs to disk.",
    )
    g.add_argument("out_dir", type=Path, help="Output directory.")
    g.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Voxel size in mm (default: 1 1 1).",
    )
    g.add_argument(
        "--size",
        nargs=3,
        type=int,
        default=(512, 512, 200),
        metavar=("NX", "NY", "NZ"),
        help="Volume size in voxels (default: 512 512 200).",
    )
    g.add_argument(
        "--n-quadrature",
        type=int,
        default=8,
        help="Sub-voxel samples per axis for partial-volume GT (default: 8 -> 512/voxel).",
    )

    v = sub.add_parser(
        "verify",
        help="Score predicted ROI masks against ground-truth NIfTIs.",
    )
    v.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Directory containing <roi>.nii.gz files produced by your tool.",
    )
    truth = v.add_mutually_exclusive_group(required=True)
    truth.add_argument(
        "--fixture",
        type=Path,
        help=(
            "Fixture directory written by `generate` (preferred). Its manifest.json "
            "supplies the ROI list; ground-truth hashes are checked before scoring."
        ),
    )
    truth.add_argument(
        "--groundtruth",
        type=Path,
        help=(
            "Directory containing the ground-truth <roi>.nii.gz files (typically "
            "<fixture>/groundtruth). The v0.1 form: no manifest, no hash check."
        ),
    )
    v.add_argument(
        "--bundle",
        action="append",
        default=None,
        metavar="NAME",
        help="Score only this bundle (repeatable; requires --fixture). Default: all bundles.",
    )
    v.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML overriding default thresholds.",
    )
    v.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the structured JSON report to this path in addition to printing the table.",
    )
    v.add_argument(
        "--report-only",
        action="store_true",
        help="Print metrics but exit 0 regardless of pass/fail. Useful for tuning thresholds.",
    )

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    options = GenerateOptions(
        voxel_size=tuple(args.voxel_size),  # type: ignore[arg-type]
        size=tuple(args.size),  # type: ignore[arg-type]
        n_quadrature=int(args.n_quadrature),
    )
    out = generate_fixture(args.out_dir, options=options)
    manifest = read_manifest(out)
    print(f"fixture written to {out}")
    print(f"  CT series        : {out / manifest.refct_dir}")
    for name, bundle in manifest.bundles.items():
        print(f"  RTSTRUCT {name:<8}: {out / bundle.rtstruct_path}  ({len(bundle.rois)} ROIs)")
    print(f"  ground-truth dir : {out / manifest.groundtruth_dir}")
    print(f"  manifest         : {out / 'manifest.json'}")
    print(f"  next: run your tool, then `rtmask-conformance verify --predictions <dir> "
          f"--fixture {out}`")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.bundle and args.fixture is None:
        print("error: --bundle requires --fixture", file=sys.stderr)
        return 2

    try:
        if args.fixture is not None:
            report: Report = verify_fixture(
                predictions_dir=args.predictions,
                fixture_dir=args.fixture,
                config=config,
                bundles=args.bundle,
            )
        else:
            report = verify_predictions(
                predictions_dir=args.predictions,
                groundtruth_dir=args.groundtruth,
                config=config,
            )
    except (FileNotFoundError, ValueError) as e:  # FixtureIntegrityError is a ValueError
        print(f"error: {e}", file=sys.stderr)
        return 2

    if report.fixture is not None:
        f = report.fixture
        print(
            f"fixture {f['path']}  rtmask-conformance {f['package_version']} "
            f"({f['git_sha']})  n_quadrature={f['n_quadrature']}  "
            f"bundles={','.join(f['bundles'])}"
        )
    print_report(report)
    if args.report_json is not None:
        write_json(report, args.report_json)

    if args.report_only:
        return 0
    return 0 if report.overall_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
