#!/usr/bin/env python3
"""
Generate single-panel figures sized for projection.

The report figures pack four to six panels each, which is right on a page a
reader can study but wrong on a slide seen for thirty seconds from the back of
a room.  These redraw the same model output one idea at a time, with type large
enough to read projected.

Writes ``results/slides/``.

    python export_slide_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from smr import Reformer2D, analyze_coking, default_config        # noqa: E402
from smr.assessment import closure_sensitivity, dimensionality_comparison
from smr.optimization import operating_envelope                   # noqa: E402
from smr.visualization import (INK_MUTED, INK_PRIMARY,            # noqa: E402
                               INK_SECONDARY, SERIES_COLORS, SURFACE)

#: Figure geometry: rendered at twice the size it is shown at, for crispness.
FIGSIZE = (11.0, 6.2)
DPI = 200


def use_slide_style() -> None:
    """Projection-scale type on the package's own palette."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 20,
        "axes.titlesize": 23,
        "axes.labelsize": 21,
        "axes.titleweight": "semibold",
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e2dc",
        "grid.linewidth": 1.0,
        "lines.linewidth": 3.2,
        "lines.markersize": 9,
        "legend.frameon": False,
        "legend.fontsize": 19,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


def figure() -> tuple:
    return plt.subplots(figsize=FIGSIZE)


def axial_temperatures(solution, path: Path) -> Path:
    """Wall, bulk and centreline temperature against axial position."""
    fig, ax = figure()
    z = solution.z
    ax.plot(z, solution.wall_surface_temperature(), color=SERIES_COLORS[7],
            label="inner wall surface")
    ax.plot(z, solution.bulk_temperature(), color=SERIES_COLORS[0],
            label="bulk gas")
    ax.plot(z, solution.centreline_temperature(), color=SERIES_COLORS[2],
            label="centreline")
    limit = solution.config.operating.wall_temperature_limit
    ax.axhline(limit, color=INK_MUTED, linestyle=(0, (5, 4)), linewidth=2.0)
    ax.annotate(f"metallurgical limit  {limit:.0f} K", xy=(0.99, limit),
                xycoords=("axes fraction", "data"), xytext=(0, 8),
                textcoords="offset points", ha="right", color=INK_MUTED,
                fontsize=18)
    ax.set_xlim(z[0], z[-1])
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("temperature  [K]")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def radial_profiles(solution, path: Path) -> Path:
    """Radial temperature profiles at four axial stations."""
    fig, ax = figure()
    r_mm = solution.grid.centers * 1.0e3
    radius = np.concatenate([r_mm, [solution.grid.radius * 1.0e3]])
    length = solution.config.geometry.length
    for slot, fraction in enumerate((0.0, 0.1, 0.3, 1.0)):
        k = int(round(fraction * (solution.n_axial - 1)))
        profile = np.concatenate([solution.temperature[k],
                                  [solution.wall_surface_temperature()[k]]])
        ax.plot(radius, profile, color=SERIES_COLORS[slot],
                label=f"z = {solution.z[k]:.1f} m")
        ax.plot(radius[-1], profile[-1], marker="s", markersize=11,
                color=SERIES_COLORS[slot], markeredgecolor=SURFACE,
                markeredgewidth=2.0)
    ax.set_xlabel("radial position  r  [mm]   (0 = centreline, 49 = wall)")
    ax.set_ylabel("temperature  [K]")
    ax.legend(loc="upper left", title="axial station", title_fontsize=18)
    ax.annotate("squares: inner wall surface,\nacross the wall film",
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                color=INK_SECONDARY, fontsize=18)
    fig.savefig(path)
    plt.close(fig)
    return path


def dimensionality(comparison: dict, sensitivity: dict, path: Path) -> Path:
    """What resolving the radial direction changes, and what it does not.

    Plotted as change relative to each metric's own value.  An absolute axis
    would put conversion points beside kelvin, which are not comparable
    quantities; the relative form is one axis and makes the contrast the point.
    """
    fig, ax = figure()
    delta, two_d = comparison["delta"], comparison["two_dimensional"]
    items = [
        ("CH$_4$\nconversion", "methane_conversion", "0.04 points", 0),
        ("outlet\nbulk $T$", "outlet_bulk_temperature_K", "0.93 K", 0),
        ("peak wall\nsurface $T$", "max_wall_surface_temperature_K",
         "5.8 K", 1),
        ("peak radial\n" r"$\Delta T$", "max_radial_dT_K", "43.3 K", 1),
    ]
    labels = [label for label, *_ in items]
    relative = [100.0 * abs(delta[key]) / abs(two_d[key])
                for _, key, _, _ in items]
    colors = [SERIES_COLORS[slot] for *_, slot in items]

    bars = ax.bar(labels, relative, color=colors, width=0.6)
    for bar, value, (_, _, absolute, _) in zip(bars, relative, items):
        ax.annotate(f"{value:.2f}%\n{absolute}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", color=INK_PRIMARY,
                    fontsize=19, fontweight="semibold")
    ax.axhline(0.0, color=INK_PRIMARY, linewidth=1.4)
    ax.set_ylabel("relative change  [%]")
    ax.set_ylim(0, max(relative) * 1.30)
    ax.grid(axis="x", visible=False)
    # A legend rather than floating labels: the two short bars carry no visible
    # colour, so the grouping has to be stated somewhere the eye can find it.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=SERIES_COLORS[0],
                             label="what the tube produces"),
                       Patch(facecolor=SERIES_COLORS[1],
                             label="how hot its wall runs")],
              loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    return path


def coking(report, path: Path) -> Path:
    """Carbon activity by route along the tube."""
    fig, ax = figure()
    z = report.z
    names = ("methane cracking", "Boudouard", "CO reduction")
    for j, name in enumerate(names):
        ax.plot(z, report.activity_actual[:, :, j].max(axis=1),
                color=SERIES_COLORS[j], label=name)
    ax.axhline(1.0, color=INK_PRIMARY, linestyle=(0, (5, 4)), linewidth=2.2)
    ceiling = float(np.nanmax(report.activity_actual))
    ax.set_ylim(1.0e-3, max(10.0 * ceiling, 10.0))
    ax.set_yscale("log")
    ax.axvspan(0.0, 0.65, color=SERIES_COLORS[1], alpha=0.12, linewidth=0)
    ax.annotate("carbon can deposit\n(first 0.65 m)", xy=(0.9, 2.2),
                color=INK_PRIMARY, fontsize=19)
    ax.annotate("deposition boundary  $a_C = 1$", xy=(0.99, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, -26),
                textcoords="offset points", ha="right", color=INK_PRIMARY,
                fontsize=18)
    ax.set_xlim(z[0], z[-1])
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"carbon activity  $a_C$  [-]")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def envelope(data: dict, path: Path) -> Path:
    """Hydrogen production along the feasible firing envelope."""
    fig, ax = figure()
    sc = data["steam_to_carbon"]
    hydrogen = data["hydrogen_production"]
    binding = np.array(data["binding"])
    wall = binding == "wall temperature"

    ax.plot(sc, hydrogen, color=INK_MUTED, linewidth=2.0, zorder=1)
    ax.plot(sc[wall], hydrogen[wall], color=SERIES_COLORS[0], marker="o",
            linestyle="none", markersize=12,
            label="metallurgical limit binds")
    ax.plot(sc[~wall], hydrogen[~wall], color=SERIES_COLORS[1], marker="s",
            linestyle="none", markersize=12, label="pressure-drop limit binds")
    best = int(np.argmax(hydrogen))
    ax.plot(sc[best], hydrogen[best], marker="*", markersize=30,
            color=SERIES_COLORS[7], linestyle="none", zorder=5)
    # Labelled as the corner rather than by its S/C: this sweep is on an
    # 11-point grid, while the optimiser locates the corner at S/C 4.15.
    ax.annotate("the corner:\nboth limits meet here",
                xy=(sc[best], hydrogen[best]), xytext=(0, 22),
                textcoords="offset points", ha="center",
                color=INK_PRIMARY, fontsize=20, fontweight="semibold")
    span = float(hydrogen.max() - hydrogen.min())
    ax.set_ylim(hydrogen.min() - 0.08 * span, hydrogen.max() + 0.34 * span)
    ax.set_xlabel("steam-to-carbon ratio  [-]")
    ax.set_ylabel("H$_2$ production  [mol s$^{-1}$ per tube]")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    return path


def closure_band(sensitivity: dict, path: Path) -> Path:
    """The radial gradient's sensitivity to the bed conductivity."""
    fig, ax = figure()
    k = sensitivity["conductivities"]
    gradient = sensitivity["metrics"]["max_radial_dT_K"]
    ax.plot(k, gradient, color=SERIES_COLORS[0], marker="o")
    reference = sensitivity["reference"]["max_radial_dT_K"]
    ax.plot([30.0], [reference], marker="*", markersize=30,
            color=SERIES_COLORS[7], linestyle="none", zorder=5)
    ax.annotate(f"correlated model\n{reference:.0f} K", xy=(30.0, reference),
                xytext=(18, 18), textcoords="offset points",
                color=INK_PRIMARY, fontsize=19, fontweight="semibold")
    ax.fill_between(k, gradient.min(), gradient.max(),
                    color=SERIES_COLORS[0], alpha=0.08, linewidth=0)
    ax.set_xlabel("effective radial conductivity  $k_{er}$"
                  "  [W m$^{-1}$ K$^{-1}$]")
    ax.set_ylabel(r"peak radial $\Delta T$  [K]")
    ax.annotate("published correlations disagree\nby a factor approaching two",
                xy=(0.97, 0.90), xycoords="axes fraction", ha="right",
                va="top", color=INK_SECONDARY, fontsize=19)
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    use_slide_style()
    outdir = Path("results/slides")
    outdir.mkdir(parents=True, exist_ok=True)

    print("  solving the baseline tube")
    solution = Reformer2D(default_config()).solve()
    report = analyze_coking(solution, include_equilibrated=False)

    written = [
        axial_temperatures(solution, outdir / "s1_axial_temperature.png"),
        radial_profiles(solution, outdir / "s2_radial_profiles.png"),
        coking(report, outdir / "s4_coking.png"),
    ]

    print("  running the dimensionality comparison")
    comparison = dimensionality_comparison(verbose=False)
    sensitivity = closure_sensitivity(verbose=False)
    written.append(dimensionality(comparison, sensitivity,
                                  outdir / "s3_dimensionality.png"))
    written.append(closure_band(sensitivity, outdir / "s6_closure_band.png"))

    print("  running the operating envelope")
    data = operating_envelope(steam_to_carbon=np.linspace(2.0, 4.5, 11),
                              verbose=False)
    written.append(envelope(data, outdir / "s5_envelope.png"))

    for path in sorted(written):
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
