"""Generate the figures referenced by ``docs/VALIDATION.md``.

This script reproduces every numeric value cited in the validation page by
calling the same metric functions the test suite calls, then renders the
results to PNG.

Run from the repo root:

    python docs/validation/generate_figures.py

Outputs land in ``docs/validation/figures/``. The script is deterministic —
re-running on the same metric implementations produces byte-identical PNGs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.ndimage import binary_erosion

from rtmask_conformance._vendor.metrics import (
    all_surface_metrics,
    binary_dsc,
    volume_metrics,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cube(shape: tuple[int, int, int], origin: tuple[int, int, int], side: int) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    z, y, x = origin
    arr[z:z + side, y:y + side, x:x + side] = 1
    return arr


def sphere(shape: tuple[int, int, int], center: tuple[int, int, int], radius: float) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = center
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure 1: Cube-overlap test cases (schematic + analytical vs measured Dice)
# ---------------------------------------------------------------------------


def fig_cube_overlap_cases() -> Path:
    """Six cube-on-cube cases rendered with one ``bar3d`` per shape.

    Each case has at most three boxes drawn into the same axes:
      * **A** — semi-transparent (red)
      * **B** — semi-transparent (blue)
      * **A ∩ B** — fully opaque (purple)

    Because every shape here is axis-aligned, both A, B, and their
    intersection are themselves rectangular cuboids — drawn as a single
    `bar3d` each rather than per-voxel cubes. This keeps the geometry
    legible: two transparent boxes with an opaque solid where they overlap.
    Drawn order is A → B → intersection so the opaque solid sits on top.
    """
    Box = tuple[tuple[float, float, float], tuple[float, float, float]]
    # Coordinates in mm (= voxels at 1 mm spacing). Each box is (origin, size).
    cases: list[tuple[str, str, Box | None, Box | None, float]] = [
        ("Identity", "A = B (10³)",
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         1.0),
        ("Half overlap", "B shifted +5 along x",
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         ((5.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         0.5),
        ("Eighth overlap", "B shifted +5 along x, y and z",
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         ((5.0, 5.0, 5.0), (10.0, 10.0, 10.0)),
         0.125),
        ("Subset", "5³ centred inside 10³",
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         ((2.5, 2.5, 2.5), (5.0, 5.0, 5.0)),
         250.0 / 1125.0),
        ("Disjoint", "no overlap",
         ((0.0, 0.0, 0.0), (5.0, 5.0, 5.0)),
         ((10.0, 10.0, 10.0), (5.0, 5.0, 5.0)),
         0.0),
        ("One empty", "A = 10³, B = ∅",
         ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
         None,
         0.0),
    ]

    a_color = "#cc4444"        # red
    b_color = "#4466cc"        # blue
    inter_color = "#6a3a8a"    # purple
    soft_alpha = 0.25
    solid_alpha = 1.00

    def draw_box(ax, box: Box, color: str, alpha: float) -> None:
        (ox, oy, oz), (sx, sy, sz) = box
        ax.bar3d(
            ox, oy, oz, sx, sy, sz,
            color=color, alpha=alpha, shade=True,
            edgecolor="black", linewidth=0.6,
        )

    def intersect(a: Box | None, b: Box | None) -> Box | None:
        if a is None or b is None:
            return None
        (ax_, ay_, az_), (asx, asy, asz) = a
        (bx_, by_, bz_), (bsx, bsy, bsz) = b
        lo = (max(ax_, bx_), max(ay_, by_), max(az_, bz_))
        hi = (min(ax_ + asx, bx_ + bsx),
              min(ay_ + asy, by_ + bsy),
              min(az_ + asz, bz_ + bsz))
        size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if min(size) <= 0:
            return None
        return (lo, size)

    def measure_dice(a: Box | None, b: Box | None) -> float:
        """Voxelize both boxes onto a shared integer grid and call binary_dsc.

        Boxes use 1 mm spacing, so 1 voxel = 1 mm³. Expanding to ints is
        loss-free here — every box edge lands on a whole-mm grid point.
        """
        boxes = [x for x in (a, b) if x is not None]
        if not boxes:
            return 1.0  # both empty, by convention
        max_xyz = [int(max(o[i] + s[i] for o, s in boxes)) + 1 for i in range(3)]
        ag = np.zeros((max_xyz[2], max_xyz[1], max_xyz[0]), dtype=np.uint8)
        bg = np.zeros_like(ag)

        def fill(grid: np.ndarray, box: Box) -> None:
            (ox, oy, oz), (sx, sy, sz) = box
            grid[int(oz):int(oz + sz), int(oy):int(oy + sy), int(ox):int(ox + sx)] = 1

        if a is not None:
            fill(ag, a)
        if b is not None:
            fill(bg, b)
        return binary_dsc(ag, bg)

    fig = plt.figure(figsize=(15, 9.5))
    fig.suptitle("Unit-test Dice cases — analytical vs measured", fontsize=15, y=0.985)

    for idx, (title, desc, a_box, b_box, expected) in enumerate(cases):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")

        # Draw order matters under transparency: A first, then B, then the
        # opaque intersection on top. matplotlib's depth sorting on
        # transparent surfaces is approximate, but for two cuboids + one
        # opaque interior cuboid this order reads correctly.
        if a_box is not None:
            draw_box(ax, a_box, a_color, soft_alpha)
        if b_box is not None:
            draw_box(ax, b_box, b_color, soft_alpha)
        ibox = intersect(a_box, b_box)
        if ibox is not None:
            draw_box(ax, ibox, inter_color, solid_alpha)

        # Bound the axes to the union of A and B with a small margin.
        present = [x for x in (a_box, b_box) if x is not None]
        if present:
            lo = min(o[i] for o, _ in present for i in range(3))
            hi = max(o[i] + s[i] for o, s in present for i in range(3))
            pad = 0.05 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_zlim(lo - pad, hi + pad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
        ax.set_xlabel("x (mm)", fontsize=8, labelpad=-6)
        ax.set_ylabel("y (mm)", fontsize=8, labelpad=-6)
        ax.set_zlabel("z (mm)", fontsize=8, labelpad=-6)
        ax.tick_params(axis="both", which="major", labelsize=7, pad=-2)

        measured = measure_dice(a_box, b_box)
        ax.set_title(
            f"{title} — {desc}\n"
            f"analytical Dice = {expected:.4f}   measured Dice = {measured:.4f}",
            fontsize=10,
        )

    legend = [
        mpatches.Patch(color=a_color, alpha=soft_alpha + 0.20, label="A (prediction)"),
        mpatches.Patch(color=b_color, alpha=soft_alpha + 0.20, label="B (ground truth)"),
        mpatches.Patch(color=inter_color, alpha=solid_alpha, label="A ∩ B (opaque)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.30, wspace=0.05,
                        left=0.03, right=0.97)

    path = OUT / "fig1_cube_overlap_cases.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Expected-vs-measured bar chart
# ---------------------------------------------------------------------------


def fig_expected_vs_measured_dice() -> Path:
    cases = [
        ("Identity", 1.0, *(cube((20, 20, 20), (5, 5, 5), 10),) * 2),
        ("Half overlap", 0.5, cube((20, 20, 20), (5, 5, 0), 10), cube((20, 20, 20), (5, 5, 5), 10)),
        ("Eighth overlap", 0.125,
         cube((20, 20, 20), (0, 0, 0), 10), cube((20, 20, 20), (5, 5, 5), 10)),
        ("Subset 5³⊂10³", 250.0 / 1125.0,
         cube((20, 20, 20), (5, 5, 5), 10), cube((20, 20, 20), (7, 7, 7), 5)),
        ("Disjoint", 0.0,
         cube((20, 20, 20), (0, 0, 0), 5), cube((20, 20, 20), (15, 15, 15), 5)),
        ("Both empty", 1.0,
         np.zeros((10, 10, 10), np.uint8), np.zeros((10, 10, 10), np.uint8)),
    ]
    labels = [c[0] for c in cases]
    expected = [c[1] for c in cases]
    measured = [binary_dsc(c[2], c[3]) for c in cases]

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, expected, width=w, color="#4c72b0", label="Analytical")
    ax.bar(x + w / 2, measured, width=w, color="#dd8452", label="Measured")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Dice similarity coefficient")
    ax.set_ylim(0, 1.08)
    ax.set_title("Dice: hand-computed expected value vs `binary_dsc` measurement")
    for xi, (e, m) in enumerate(zip(expected, measured, strict=True)):
        ax.text(xi - w / 2, e + 0.02, f"{e:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    path = OUT / "fig2_expected_vs_measured_dice.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Surface metrics under increasing voxel shift
# ---------------------------------------------------------------------------


def fig_metrics_vs_shift() -> Path:
    base = cube((40, 40, 40), (15, 15, 15), 10)
    shifts = [0, 1, 2, 3, 4, 5, 7, 10]
    spacing = (1.0, 1.0, 1.0)
    dice_vals: list[float] = []
    hd95_vals: list[float] = []
    for s in shifts:
        b = cube((40, 40, 40), (15, 15, 15 + s), 10)
        dice_vals.append(binary_dsc(base, b))
        sm = all_surface_metrics(base, b, voxel_size_mm=spacing)
        hd95_vals.append(sm.hausdorff95_mm if sm.hausdorff95_mm is not None else np.nan)

    fig, ax1 = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    line_dice = ax1.plot(shifts, dice_vals, "o-", color="#4c72b0", label="Dice")
    ax1.set_xlabel("Voxel shift along x  (1 mm spacing)")
    ax1.set_ylabel("Dice", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()
    line_hd = ax2.plot(shifts, hd95_vals, "s--", color="#dd8452", label="HD95 (mm)")
    ax2.plot(shifts, shifts, ":", color="0.4", linewidth=1, label="ideal HD95 = shift")
    ax2.set_ylabel("HD95 (mm)", color="#dd8452")
    ax2.tick_params(axis="y", labelcolor="#dd8452")

    lines = line_dice + line_hd + [ax2.lines[-1]]
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    ax1.set_title("Two 10³ cubes, one shifted along x — Dice falls, HD95 tracks the shift")

    path = OUT / "fig3_metrics_vs_shift.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: HD95 honours per-axis spacing
# ---------------------------------------------------------------------------


def fig_anisotropic_spacing() -> Path:
    a = cube((40, 40, 40), (15, 15, 15), 10)
    b = cube((40, 40, 40), (16, 15, 15), 10)  # +1 voxel along z

    spacings = [
        ("Isotropic 1 mm", (1.0, 1.0, 1.0)),
        ("z-anisotropic\n(0.5, 1.0, 2.0) mm", (0.5, 1.0, 2.0)),
        ("z-anisotropic\n(0.5, 1.0, 5.0) mm", (0.5, 1.0, 5.0)),
    ]
    measured = []
    for _, sp in spacings:
        sm = all_surface_metrics(a, b, voxel_size_mm=sp)
        measured.append(sm.hausdorff95_mm)
    expected = [1.0, 2.0, 5.0]  # 1-voxel z-shift × sz

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(len(spacings))
    w = 0.38
    ax.bar(x - w / 2, expected, width=w, color="#4c72b0", label="Expected = 1·sz")
    ax.bar(x + w / 2, measured, width=w, color="#dd8452", label="Measured HD95")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in spacings])
    ax.set_ylabel("HD95 (mm)")
    ax.set_title("Per-axis spacing on the z dimension propagates into HD95")
    ax.legend()
    for xi, (e, m) in enumerate(zip(expected, measured, strict=True)):
        ax.text(xi - w / 2, e + 0.05, f"{e:.2f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    path = OUT / "fig4_anisotropic_spacing.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: Surface DSC vs tolerance
# ---------------------------------------------------------------------------


def fig_surface_dsc_vs_tolerance() -> Path:
    a = cube((40, 40, 40), (15, 15, 15), 10)
    b_close = cube((40, 40, 40), (15, 15, 16), 10)  # 1-voxel x shift @ 0.5 mm = 0.5 mm
    b_far = cube((40, 40, 40), (15, 15, 17), 10)    # 2-voxel x shift @ 1.0 mm = 2.0 mm

    tols = np.linspace(0.0, 3.0, 31)
    sdsc_close = [
        all_surface_metrics(a, b_close, voxel_size_mm=(0.5, 0.5, 0.5),
                            tolerance_mm=float(t)).surface_dsc for t in tols
    ]
    sdsc_far = [
        all_surface_metrics(a, b_far, voxel_size_mm=(1.0, 1.0, 1.0),
                            tolerance_mm=float(t)).surface_dsc for t in tols
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(tols, sdsc_close, "o-", color="#4c72b0",
            label="0.5 mm offset (1 vx @ 0.5 mm spacing)")
    ax.plot(tols, sdsc_far, "s--", color="#dd8452",
            label="2.0 mm offset (2 vx @ 1.0 mm spacing)")
    ax.axvline(0.5, linestyle=":", color="#4c72b0", alpha=0.6)
    ax.axvline(2.0, linestyle=":", color="#dd8452", alpha=0.6)
    ax.set_xlabel("Surface-DSC tolerance (mm)")
    ax.set_ylabel("Surface DSC")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Surface DSC saturates at 1.0 once tolerance ≥ surface offset")
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)

    path = OUT / "fig5_surface_dsc_vs_tolerance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: Volume rel-err under 1-voxel erosion (per shape)
# ---------------------------------------------------------------------------


def fig_erosion_volume_loss() -> Path:
    """Erode a sphere, ellipsoid, and cube by 1 voxel; show the rel-err and
    overlay the analytic surface-area / volume ratio for each.

    Shapes are sized so the comparison is meaningful — small enough to
    render fast, large enough that boundary discretisation doesn't dominate.
    """
    shape = (60, 60, 60)
    center = (30, 30, 30)
    cases = []
    # (name, mask, surface_area_mm2, volume_mm3) — analytical for reference.
    cases.append(("sphere r=15", sphere(shape, center, 15.0),
                  4 * np.pi * 15.0 ** 2, (4 / 3) * np.pi * 15.0 ** 3))
    # Cube: side 30 voxels.
    cases.append(("cube 30³", cube(shape, (15, 15, 15), 30),
                  6 * 30.0 ** 2, 30.0 ** 3))
    # "Hollow shell" — sphere of r=15 minus inner r=10.
    outer = sphere(shape, center, 15.0)
    inner = sphere(shape, center, 10.0)
    hollow = (outer & ~inner).astype(np.uint8)
    cases.append(("hollow sphere\nr_out=15, r_in=10",
                  hollow,
                  4 * np.pi * (15.0 ** 2 + 10.0 ** 2),
                  (4 / 3) * np.pi * (15.0 ** 3 - 10.0 ** 3)))

    rel_errs = []
    sa_over_v = []
    for _, gt, sa, vol in cases:
        eroded = binary_erosion(gt.astype(bool)).astype(np.uint8)
        vm = volume_metrics(eroded, gt, voxel_volume_mm3=1.0)
        rel_errs.append(vm.rel_error_pct)
        sa_over_v.append(100.0 * sa / vol)  # in %

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(cases))
    w = 0.38
    ax.bar(x - w / 2, sa_over_v, width=w, color="#4c72b0",
           label="Analytic surface/volume × 100")
    ax.bar(x + w / 2, rel_errs, width=w, color="#dd8452",
           label="Measured rel-err after 1-voxel erosion")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel("Percent")
    ax.set_title("1-voxel erosion: measured volume loss tracks surface/volume ratio")
    ax.legend()
    for xi, (s, r) in enumerate(zip(sa_over_v, rel_errs, strict=True)):
        ax.text(xi - w / 2, s + 0.5, f"{s:.1f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, r + 0.5, f"{r:.1f}", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    path = OUT / "fig6_erosion_volume_loss.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    figs = [
        fig_cube_overlap_cases(),
        fig_expected_vs_measured_dice(),
        fig_metrics_vs_shift(),
        fig_anisotropic_spacing(),
        fig_surface_dsc_vs_tolerance(),
        fig_erosion_volume_loss(),
    ]
    for p in figs:
        print(f"wrote {p.relative_to(Path.cwd()) if p.is_absolute() else p}")


if __name__ == "__main__":
    main()
