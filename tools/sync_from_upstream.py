#!/usr/bin/env python3
"""Re-vendor the rtmask_validation source files into ``src/rtmask_conformance/_vendor/``.

Usage:

    python tools/sync_from_upstream.py /path/to/Dicom_RT_Images_Csharp

This copies the minimal set of modules needed for the conformance suite, then
records the upstream commit hash in ``tools/UPSTREAM_VERSION.txt``. Run any time
you want to pull bug fixes or behavior improvements from upstream.

The script does NOT touch the conformance-specific layer (``cli.py``,
``generate.py``, ``verify.py``, etc.) — only the ``_vendor/`` subpackage.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

# (relative path under upstream src/rtmask_validation, relative path under _vendor)
FILES_TO_VENDOR: list[tuple[str, str]] = [
    ("common/coords.py", "common/coords.py"),
    ("common/io.py", "common/io.py"),
    ("primitives/base.py", "primitives/base.py"),
    ("primitives/closed_planar.py", "primitives/closed_planar.py"),
    ("primitives/closed_planar_xor.py", "primitives/closed_planar_xor.py"),
    ("refimage/uid_hierarchy.py", "refimage/uid_hierarchy.py"),
    ("refimage/build_reference_ct.py", "refimage/build_reference_ct.py"),
    ("groundtruth/partial_volume.py", "groundtruth/partial_volume.py"),
    ("rtstruct/geometric_types.py", "rtstruct/geometric_types.py"),
    ("rtstruct/pydicom_writer.py", "rtstruct/pydicom_writer.py"),
    # benchmark/metrics.py needs a slim — handled separately below
]


def _slim_metrics(src_text: str) -> str:
    """Strip metrics.py down to the closed-shape subset.

    Drops ``point_metrics``, ``curve_metrics``, ``arc_length_metrics``,
    ``consensus_mask``, and the helpers/imports they need (``Geometry``,
    ``cKDTree``, ``densify_polyline``, ``skimage``).

    Best-effort: if the upstream layout drifts, this raises and you re-do the
    slim by hand.
    """
    lines = src_text.splitlines(keepends=True)
    # Cut everything from the first "Inter-tool agreement" or "point/curve metrics"
    # banner onward. Those banners exist in upstream metrics.py and bracket the
    # closed-shape section above from the curve/point section below.
    cut_marker = "#  Inter-tool agreement helpers"
    for i, line in enumerate(lines):
        if cut_marker in line:
            return "".join(lines[: i - 1])  # drop the comment-banner separator too
    raise RuntimeError(
        f"Could not find slim cut marker {cut_marker!r} in upstream metrics.py — "
        "upstream layout has drifted. Re-do the slim manually."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_repo",
        type=Path,
        help="Path to a checkout of brianmanderson/Dicom_RT_Images_Csharp",
    )
    args = parser.parse_args()

    upstream_root: Path = args.upstream_repo.resolve()
    upstream_pkg = upstream_root / "PythonCode" / "src" / "rtmask_validation"
    if not upstream_pkg.exists():
        print(f"error: {upstream_pkg} not found", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    vendor_root = repo_root / "src" / "rtmask_conformance" / "_vendor"
    if not vendor_root.exists():
        print(f"error: {vendor_root} missing — run from a populated repo", file=sys.stderr)
        return 2

    # Plain copies.
    for src_rel, dst_rel in FILES_TO_VENDOR:
        src = upstream_pkg / src_rel
        dst = vendor_root / dst_rel
        if not src.exists():
            print(f"error: upstream file missing: {src}", file=sys.stderr)
            return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copied  {src_rel}")

    # Slimmed metrics.
    src_metrics = upstream_pkg / "benchmark" / "metrics.py"
    dst_metrics = vendor_root / "metrics.py"
    metrics_slim = _slim_metrics(src_metrics.read_text(encoding="utf-8"))
    dst_metrics.write_text(metrics_slim, encoding="utf-8")
    print("copied  benchmark/metrics.py (slimmed to closed-shape subset)")

    # Capture upstream commit.
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream_root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = "UNKNOWN"

    version_file = repo_root / "tools" / "UPSTREAM_VERSION.txt"
    version_file.write_text(
        f"upstream_repo: https://github.com/brianmanderson/Dicom_RT_Images_Csharp\n"
        f"upstream_path: PythonCode/src/rtmask_validation\n"
        f"upstream_commit: {commit}\n"
        f"synced_on: {date.today().isoformat()}\n",
        encoding="utf-8",
    )
    print(f"\nrecorded upstream commit {commit} in {version_file.relative_to(repo_root)}")
    print(
        "\nNote: re-run unit tests after sync. If upstream introduced Python 3.11+ "
        "syntax (e.g. `Self`, `tomllib`), backport manually."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
