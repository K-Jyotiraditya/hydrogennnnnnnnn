r"""
Matplotlib figures for the 2D reformer model.

Design rules followed here
--------------------------
* Colour is assigned by the job it does.  Series identity uses a fixed,
  colour-vision-deficiency validated categorical order (never cycled); the
  temperature field uses a single-hue light-to-dark sequential ramp because it
  encodes magnitude; the carbon-activity field uses a two-hue diverging ramp
  with a neutral midpoint pinned at ``a_C = 1``, because what matters there is
  polarity about a threshold.
* No dual-axis panels anywhere.  Two quantities of different scale get two
  panels.
* Every panel with two or more series carries a legend, and panels with four or
  fewer series also carry direct end labels, so identity never rests on colour
  alone.  Three of the categorical slots fall below 3:1 contrast on the chart
  surface, which obliges that relief.
* Grid and spines are recessive; values are never printed on every point.

Every function returns the :class:`matplotlib.figure.Figure` it creates, so the
caller can adjust or save it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
import numpy as np

from .coking import CARBON_REACTIONS, CokingReport
from .optimization import ParetoFront
from .reformer import PA_PER_BAR, ReformerSolution
from .species import IDX, SPECIES

__all__ = [
    "SERIES_COLORS",
    "use_style",
    "plot_axial_overview",
    "plot_radial_temperature",
    "plot_radial_profiles",
    "plot_kinetic_diagnostics",
    "plot_coking",
    "plot_carbon_boundary_map",
    "plot_pareto_front",
    "plot_operating_envelope",
    "plot_sensitivity",
    "plot_grid_convergence",
    "plot_model_assessment",
]

#: Validated categorical order (light surface).  Assign by slot, never cycle.
SERIES_COLORS = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"

#: Sequential ramp for magnitude fields (single hue, light to dark).
SEQUENTIAL_CMAP = "Reds"
#: Diverging ramp for threshold fields (two hues, neutral midpoint).
DIVERGING_CMAP = "RdBu_r"

_SPECIES_LABEL = {
    "CH4": r"CH$_4$", "H2O": r"H$_2$O", "H2": r"H$_2$",
    "CO": "CO", "CO2": r"CO$_2$", "N2": r"N$_2$",
}


def use_style() -> None:
    """Apply the package plotting style to the global rcParams."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.titleweight": "semibold",
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e2dc",
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.9,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _label_end(ax, x, y, text, color, dx: float = 0.012,
               va: str = "center") -> None:
    """Direct end-of-line label in neutral ink with a coloured marker.

    Text wears text tokens rather than the series colour; the small marker
    beside it carries identity.
    """
    span = ax.get_xlim()
    offset = dx * (span[1] - span[0])
    ax.plot([x], [y], marker="o", markersize=4.0, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5,
            clip_on=False)
    ax.annotate(text, xy=(x + offset, y), xycoords="data", color=INK_SECONDARY,
                fontsize=8.0, va=va, ha="left", annotation_clip=False)


# ---------------------------------------------------------------------------
# Axial overview
# ---------------------------------------------------------------------------

def plot_axial_overview(solution: ReformerSolution,
                        title: Optional[str] = None):
    """Six-panel axial overview of the solved tube."""
    import matplotlib.pyplot as plt

    use_style()
    z = solution.z
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    fig.subplots_adjust(hspace=0.33, wspace=0.30)

    # (a) temperatures -----------------------------------------------------
    ax = axes[0, 0]
    series = [
        ("inner wall surface", solution.wall_surface_temperature(), SERIES_COLORS[7]),
        ("bed, near wall", solution.near_wall_temperature(), SERIES_COLORS[1]),
        ("bulk (mixing cup)", solution.bulk_temperature(), SERIES_COLORS[0]),
        ("centreline", solution.centreline_temperature(), SERIES_COLORS[2]),
    ]
    for name, values, color in series:
        ax.plot(z, values, color=color, label=name)
    ax.set_xlim(z[0], z[-1])
    # The four curves converge at the outlet, so direct end labels would
    # collide; the legend carries identity here and the contrast relief is the
    # table view written alongside the figures.
    ax.axhline(solution.config.operating.wall_temperature_limit,
               color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.annotate("metallurgical limit",
                xy=(0.98, solution.config.operating.wall_temperature_limit),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right",
                color=INK_MUTED, fontsize=7.5)
    ax.set_title("(a)  Temperature")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("temperature  [K]")
    ax.legend(loc="lower right", ncols=1)

    # (b) dry composition --------------------------------------------------
    ax = axes[0, 1]
    y_dry = solution.dry_mole_fractions()
    order = ["H2", "CH4", "CO", "CO2", "N2"]
    for slot, name in enumerate(order):
        ax.plot(z, y_dry[:, IDX[name]], color=SERIES_COLORS[slot],
                label=_SPECIES_LABEL[name])
    ax.set_title("(b)  Bulk composition, dry basis")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("mole fraction  [-]")
    ax.legend(loc="center right", ncols=2)

    # (c) conversion and approach to equilibrium ---------------------------
    ax = axes[0, 2]
    conv = solution.methane_conversion()
    approach = solution.flow_average(
        np.clip(solution.diagnostics["approach_to_equilibrium"][:, :, 0], 0, 2))
    ax.plot(z, conv, color=SERIES_COLORS[0], label="CH$_4$ conversion")
    ax.plot(z, np.clip(approach, 0.0, 1.05), color=SERIES_COLORS[3],
            label="approach to equilibrium  $Q_1/K_1$")
    ax.set_xlim(z[0], z[-1] * 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(c)  Conversion and equilibrium approach")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("fraction  [-]")
    ax.legend(loc="lower right")

    # (d) pressure ---------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(z, solution.pressure / PA_PER_BAR, color=SERIES_COLORS[6])
    ax.set_title(f"(d)  Pressure  "
                 f"($\\Delta P$ = {solution.pressure_drop() / PA_PER_BAR:.2f} bar)")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("pressure  [bar]")

    # (e) wall heat flux ---------------------------------------------------
    ax = axes[1, 1]
    ax.plot(z, solution.diagnostics["q_wall"] / 1.0e3, color=SERIES_COLORS[1])
    ax.set_title(f"(e)  Inner-wall heat flux  "
                 f"({solution.heat_duty() / 1e3:.0f} kW per tube)")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("heat flux  [kW m$^{-2}$]")

    # (f) radial temperature difference ------------------------------------
    ax = axes[1, 2]
    ax.plot(z, solution.radial_temperature_difference(), color=SERIES_COLORS[2],
            label="wall surface $-$ centreline")
    ax.plot(z, solution.near_wall_temperature() - solution.centreline_temperature(),
            color=SERIES_COLORS[4], label="bed near wall $-$ centreline")
    ax.set_title("(f)  Radial temperature gradient")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"$\Delta T$  [K]")
    ax.legend(loc="upper right")

    head = title or f"2D pseudo-heterogeneous SMR tube  --  {solution.config.name}"
    fig.suptitle(head, fontsize=12, color=INK_PRIMARY, fontweight="semibold",
                 x=0.012, ha="left", y=0.985)
    return fig


# ---------------------------------------------------------------------------
# Radial fields
# ---------------------------------------------------------------------------

def plot_radial_temperature(solution: ReformerSolution,
                            slice_positions: Optional[Sequence[float]] = None):
    """Temperature field in the (z, r) plane plus radial slices.

    The field panel uses a single-hue sequential ramp (magnitude), mirrored
    about the axis so the tube cross-section reads naturally.
    """
    import matplotlib.pyplot as plt

    use_style()
    z = solution.z
    r_mm = solution.grid.centers * 1.0e3
    faces_mm = solution.grid.faces * 1.0e3
    T = solution.temperature

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(wspace=0.26)

    # Mirror the field about r = 0 for a full tube section.
    r_full = np.concatenate([-r_mm[::-1], r_mm])
    T_full = np.concatenate([T[:, ::-1], T], axis=1)

    ax = axes[0]
    mesh = ax.pcolormesh(z, r_full, T_full.T, cmap=SEQUENTIAL_CMAP,
                         shading="nearest", rasterized=True)
    levels = np.arange(np.floor(T.min() / 25) * 25, T.max() + 25, 25.0)
    contour = ax.contour(z, r_full, T_full.T, levels=levels,
                         colors=INK_SECONDARY, linewidths=0.5, alpha=0.55)
    ax.set_ylim(-faces_mm[-1], faces_mm[-1])
    ax.clabel(contour, levels[::2], fmt="%.0f", fontsize=6.5,
              colors=INK_SECONDARY)
    ax.set_title("(a)  Gas temperature field  $T(r, z)$")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("radial position  r  [mm]")
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax, pad=0.015)
    bar.set_label("temperature  [K]", color=INK_SECONDARY)
    bar.outline.set_visible(False)

    # Radial slices ---------------------------------------------------------
    if slice_positions is None:
        length = solution.config.geometry.length
        slice_positions = [0.0, 0.1 * length, 0.3 * length, length]
    ax = axes[1]
    for slot, z_target in enumerate(slice_positions[:4]):
        k = int(np.argmin(np.abs(z - z_target)))
        color = SERIES_COLORS[slot]
        profile = np.concatenate([T[k], [solution.wall_surface_temperature()[k]]])
        radius = np.concatenate([r_mm, [solution.grid.radius * 1.0e3]])
        ax.plot(radius, profile, color=color, label=f"z = {z[k]:.2f} m")
        ax.plot(radius[-1], profile[-1], marker="s", markersize=4.5,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_title("(b)  Radial temperature profiles\n"
                 "(square marker: inner wall surface)")
    ax.set_xlabel("radial position  r  [mm]")
    ax.set_ylabel("temperature  [K]")
    ax.legend(loc="upper left", title="axial station", title_fontsize=8)
    return fig


def plot_radial_profiles(solution: ReformerSolution,
                         species: Sequence[str] = ("CH4", "H2", "CO")):
    """Radial profiles of composition, rate and effectiveness factor."""
    import matplotlib.pyplot as plt

    use_style()
    z = solution.z
    r_mm = solution.grid.centers * 1.0e3
    length = solution.config.geometry.length
    stations = [0.05 * length, 0.2 * length, 0.5 * length, length]
    indices = [int(np.argmin(np.abs(z - zt))) for zt in stations]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.30)

    # (a) species radial profiles, normalised to the centreline value -------
    ax = axes[0]
    y = solution.mole_fractions()
    for slot, name in enumerate(species[:3]):
        k = indices[1]
        ax.plot(r_mm, y[k, :, IDX[name]], color=SERIES_COLORS[slot],
                label=_SPECIES_LABEL[name])
    ax.set_title(f"(a)  Radial composition at z = {z[indices[1]]:.2f} m")
    ax.set_xlabel("radial position  r  [mm]")
    ax.set_ylabel("mole fraction  [-]")
    ax.legend(loc="best")

    # (b) reforming rate ----------------------------------------------------
    ax = axes[1]
    rate = solution.diagnostics["rates"][:, :, 0]
    for slot, k in enumerate(indices):
        ax.plot(r_mm, rate[k] * 3.6, color=SERIES_COLORS[slot],
                label=f"z = {z[k]:.2f} m")
    ax.set_title("(b)  Observed reforming rate  $\\eta_1 r_1$")
    ax.set_xlabel("radial position  r  [mm]")
    ax.set_ylabel(r"rate  [kmol kg$^{-1}$ h$^{-1}$]")
    ax.legend(loc="best", title="axial station", title_fontsize=8)

    # (c) gas-solid temperature difference ----------------------------------
    ax = axes[2]
    d_t = solution.diagnostics["solid_temperature"] - solution.temperature
    for slot, k in enumerate(indices):
        ax.plot(r_mm, d_t[k], color=SERIES_COLORS[slot], label=f"z = {z[k]:.2f} m")
    ax.set_title("(c)  Catalyst minus gas temperature")
    ax.set_xlabel("radial position  r  [mm]")
    ax.set_ylabel(r"$T_s - T_g$  [K]")
    ax.legend(loc="best", title="axial station", title_fontsize=8)
    return fig


def plot_kinetic_diagnostics(solution: ReformerSolution):
    """Rates, effectiveness factors and bed transport coefficients."""
    import matplotlib.pyplot as plt

    use_style()
    z = solution.z
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    fig.subplots_adjust(hspace=0.34, wspace=0.26)

    short = ("R1 reforming", "R2 shift", "R3 overall")

    # (a) cross-section averaged observed rates -----------------------------
    ax = axes[0, 0]
    for j in range(3):
        rate = solution.area_average(solution.diagnostics["rates"][:, :, j])
        ax.plot(z, rate * 3.6, color=SERIES_COLORS[j], label=short[j])
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    ax.set_title("(a)  Observed reaction rates (area average)")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"rate  [kmol kg$^{-1}$ h$^{-1}$]")
    ax.legend(loc="best")

    # (b) effectiveness factors ---------------------------------------------
    ax = axes[0, 1]
    for j in range(3):
        eta = solution.area_average(solution.diagnostics["eta"][:, :, j])
        ax.plot(z, eta, color=SERIES_COLORS[j], label=short[j])
    ax.set_title("(b)  Intraparticle effectiveness factor")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"$\eta$  [-]")
    ax.set_yscale("log")
    ax.legend(loc="best")

    # (c) effective radial conductivity and wall coefficient ----------------
    ax = axes[1, 0]
    ax.plot(z, solution.area_average(solution.diagnostics["k_er"]),
            color=SERIES_COLORS[0], label=r"$k_{er}$")
    ax.set_title("(c)  Effective radial conductivity")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"$k_{er}$  [W m$^{-1}$ K$^{-1}$]")
    ax.legend(loc="best")

    # (d) wall and film heat transfer coefficients --------------------------
    ax = axes[1, 1]
    ax.plot(z, solution.diagnostics["h_wall"][:, -1],
            color=SERIES_COLORS[1], label=r"wall film  $h_w$")
    ax.plot(z, solution.area_average(solution.diagnostics["h_film"]),
            color=SERIES_COLORS[2], label=r"pellet film  $h_f$")
    ax.set_title("(d)  Heat transfer coefficients")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"$h$  [W m$^{-2}$ K$^{-1}$]")
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# Coking
# ---------------------------------------------------------------------------

def plot_coking(report: CokingReport):
    """Carbon activity along and across the tube, with the deposition boundary."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    use_style()
    z = report.z
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    fig.subplots_adjust(hspace=0.36, wspace=0.26)

    short = ("C1 methane cracking", "C2 Boudouard", "C3 CO reduction")

    # (a) axial carbon activity by route ------------------------------------
    ax = axes[0, 0]
    for j in range(3):
        ax.plot(z, report.activity_actual[:, :, j].max(axis=1),
                color=SERIES_COLORS[j], label=short[j])
    ax.axhline(1.0, color=INK_PRIMARY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate("deposition boundary  $a_C = 1$", xy=(z[-1], 1.0),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                color=INK_PRIMARY, fontsize=7.5)
    ax.set_yscale("log")
    # The inlet gas carries no CO, so the Boudouard and CO-reduction
    # activities start at zero; without a floor the log axis spans twenty
    # decades of irrelevant range and hides the boundary crossing.
    ceiling = float(np.nanmax(report.activity_actual))
    ax.set_ylim(1.0e-4, max(10.0 * ceiling, 10.0))
    ax.set_title("(a)  Peak carbon activity along the tube (actual gas)")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel(r"carbon activity  $a_C$  [-]")
    ax.legend(loc="best")

    # (b) worst-route field, diverging about the threshold -------------------
    ax = axes[0, 1]
    worst = np.clip(report.worst_activity, 1.0e-6, None)
    log_worst = np.log10(worst)
    span = max(0.3, float(np.max(np.abs(log_worst))))
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    mesh = ax.pcolormesh(z, report.r * 1.0e3, log_worst.T, cmap=DIVERGING_CMAP,
                         norm=norm, shading="nearest", rasterized=True)
    ax.set_title("(b)  Worst-route carbon activity field")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("radial position  r  [mm]")
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax, pad=0.015)
    bar.set_label(r"$\log_{10} a_C$", color=INK_SECONDARY)
    bar.outline.set_visible(False)

    # (c) steam-to-carbon margin -------------------------------------------
    ax = axes[1, 0]
    ax.plot(z, report.local_sc, color=SERIES_COLORS[0], label="local S/C")
    if np.isfinite(report.critical_sc).any():
        ax.plot(z, report.critical_sc, color=SERIES_COLORS[7],
                label="critical S/C (equilibrated gas)")
        ax.fill_between(z, report.critical_sc, report.local_sc,
                        where=report.local_sc >= report.critical_sc,
                        color=SERIES_COLORS[2], alpha=0.14, linewidth=0)
    ax.set_title("(c)  Steam-to-carbon margin")
    ax.set_xlabel("axial position  z  [m]")
    ax.set_ylabel("steam / carbon  [-]")
    ax.legend(loc="best")

    # (d) activity against temperature -------------------------------------
    ax = axes[1, 1]
    temperature = report.solid_temperature.ravel()
    for j, slot in enumerate((0, 1)):
        ax.scatter(temperature, report.activity_actual[:, :, slot].ravel(),
                   s=6, color=SERIES_COLORS[slot], alpha=0.45,
                   edgecolors="none", label=short[slot])
    ax.axhline(1.0, color=INK_PRIMARY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, max(10.0 * float(np.nanmax(report.activity_actual)), 10.0))
    ax.set_title("(d)  Carbon activity against catalyst temperature")
    ax.set_xlabel("catalyst temperature  [K]")
    ax.set_ylabel(r"carbon activity  $a_C$  [-]")
    ax.legend(loc="best", markerscale=2.0)
    return fig


def plot_carbon_boundary_map(boundary: dict, pressure: float):
    """Critical S/C versus temperature -- the classical carbon-limit diagram."""
    import matplotlib.pyplot as plt

    use_style()
    T = boundary["temperature"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    names = [CARBON_REACTIONS[i].split(":")[0] + " "
             + CARBON_REACTIONS[i].split("(")[-1].rstrip(")")
             for i in boundary["routes"]]
    for slot, name in enumerate(names):
        ax.plot(T, boundary["critical_sc_by_route"][:, slot],
                color=SERIES_COLORS[slot], label=name)
    ax.plot(T, boundary["critical_sc"], color=INK_PRIMARY, linewidth=2.4,
            linestyle=(0, (5, 2)), label="controlling limit")
    ax.set_title(f"Carbon-free boundary, equilibrated gas at "
                 f"{pressure / PA_PER_BAR:.0f} bar")
    ax.set_xlabel("temperature  [K]")
    ax.set_ylabel("critical steam-to-carbon ratio  [-]")
    ax.annotate("carbon free", xy=(0.70, 0.86), xycoords="axes fraction",
                color=INK_SECONDARY, fontsize=9)
    ax.annotate("carbon formation possible", xy=(0.06, 0.08),
                xycoords="axes fraction", color=INK_SECONDARY, fontsize=9)
    ax.legend(loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def plot_pareto_front(front: ParetoFront):
    """Pareto front with the decision variables along it."""
    import matplotlib.pyplot as plt

    use_style()
    keep = front.non_dominated_mask()
    if front.variable == "wall_temperature":
        cost = front.max_wall_temperature
        cost_label = "peak inner wall surface temperature  [K]"
    else:
        cost = front.steam_flow
        cost_label = "steam fed  [mol s$^{-1}$]"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    ax.plot(cost[keep], front.hydrogen_production[keep], color=SERIES_COLORS[0],
            marker="o", label="non-dominated")
    dominated = ~keep & front.feasible
    if dominated.any():
        ax.scatter(cost[dominated], front.hydrogen_production[dominated],
                   s=26, facecolors="none", edgecolors=INK_MUTED,
                   linewidths=1.0, label="dominated")
    # Selective annotation: the two ends of the front only.  The lower end sits
    # in the bottom-left corner, so its label goes above its marker to stay
    # inside the axes.
    idx = np.flatnonzero(keep)
    ends = (idx[:1].tolist() + idx[-1:].tolist()) if idx.size else []
    for position, k in enumerate(ends):
        offset = (8, 6) if position == 0 else (-8, -22)
        align = "left" if position == 0 else "right"
        ax.annotate(f"S/C {front.steam_to_carbon[k]:.2f}\n"
                    f"q {front.heat_flux[k] / 1e3:.0f} kW/m$^2$",
                    xy=(cost[k], front.hydrogen_production[k]),
                    xytext=offset, textcoords="offset points", ha=align,
                    color=INK_SECONDARY, fontsize=7.5)
    ax.set_title("(a)  Hydrogen production against tube-wall temperature"
                 if front.variable == "wall_temperature"
                 else "(a)  Hydrogen production against steam consumption")
    ax.set_xlabel(cost_label)
    ax.set_ylabel("H$_2$ production  [mol s$^{-1}$ per tube]")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(cost, front.methane_conversion, color=SERIES_COLORS[1],
            marker="o", label="CH$_4$ conversion  [-]")
    ax.plot(cost, front.methane_slip, color=SERIES_COLORS[4],
            marker="s", label="CH$_4$ slip, dry  [-]")
    ax.set_title("(b)  Conversion and slip along the front")
    ax.set_xlabel(cost_label)
    ax.set_ylabel("fraction  [-]")
    ax.legend(loc="center right")
    return fig


def plot_operating_envelope(envelope: dict):
    """Feasible firing envelope and the hydrogen optimum it implies.

    Panel (a) shows the largest admissible mean wall flux against S/C, split
    into the stretch where the metallurgical limit binds and the stretch where
    the pressure-drop limit binds.  Panel (b) shows the hydrogen production
    achieved along that envelope, whose maximum sits at the corner between the
    two regimes.
    """
    import matplotlib.pyplot as plt

    use_style()
    sc = envelope["steam_to_carbon"]
    q = envelope["max_heat_flux"] / 1.0e3
    binding = np.array(envelope["binding"])
    wall = binding == "wall temperature"
    drop = ~wall
    corner = envelope["corner_steam_to_carbon"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    fig.subplots_adjust(wspace=0.26)

    ax = axes[0]
    ax.plot(sc, q, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.plot(sc[wall], q[wall], color=SERIES_COLORS[0], marker="o",
            linestyle="none", label="metallurgical limit binds")
    ax.plot(sc[drop], q[drop], color=SERIES_COLORS[1], marker="s",
            linestyle="none", label="pressure-drop limit binds")
    if np.isfinite(corner):
        ax.axvline(corner, color=INK_MUTED, linestyle=(0, (4, 3)),
                   linewidth=1.1)
        ax.annotate(f"corner  S/C ~ {corner:.2f}",
                    xy=(corner, 0.06), xycoords=("data", "axes fraction"),
                    xytext=(5, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=8)
    ax.set_title("(a)  Largest admissible wall heat flux")
    ax.set_xlabel("steam-to-carbon ratio  [-]")
    ax.set_ylabel("mean wall heat flux  [kW m$^{-2}$]")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.plot(sc, envelope["hydrogen_production"], color=SERIES_COLORS[2],
            marker="o", label="H$_2$ on the envelope")
    best = int(np.argmax(envelope["hydrogen_production"]))
    ax.plot(sc[best], envelope["hydrogen_production"][best], marker="*",
            markersize=14, color=SERIES_COLORS[7], linestyle="none",
            label="maximum")
    ax.annotate(f"S/C {sc[best]:.2f}\n"
                f"{envelope['max_heat_flux'][best] / 1e3:.1f} kW/m$^2$",
                xy=(sc[best], envelope["hydrogen_production"][best]),
                xytext=(-16, -44), textcoords="offset points", ha="right",
                color=INK_SECONDARY, fontsize=8)
    ax.set_title("(b)  Hydrogen production along the envelope")
    ax.set_xlabel("steam-to-carbon ratio  [-]")
    ax.set_ylabel("H$_2$ production  [mol s$^{-1}$ per tube]")
    ax.legend(loc="upper left")
    return fig


def plot_sensitivity(grids: dict, constraints=None):
    """Design-space maps over the S/C and wall-flux plane."""
    import matplotlib.pyplot as plt

    use_style()
    sc = grids["steam_to_carbon"]
    q = grids["heat_flux"] / 1.0e3
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    fig.subplots_adjust(wspace=0.30)

    panels = [
        ("methane_conversion", "(a)  CH$_4$ conversion  [-]", 1.0),
        ("hydrogen_production_mol_per_s",
         "(b)  H$_2$ production  [mol s$^{-1}$]", 1.0),
        ("max_wall_surface_temperature_K",
         "(c)  Peak wall surface temperature  [K]", 1.0),
    ]
    limit = (constraints.max_wall_temperature if constraints is not None
             else 1150.0)

    for ax, (key, title, scale) in zip(axes, panels):
        field = grids[key] * scale
        mesh = ax.pcolormesh(q, sc, field, cmap=SEQUENTIAL_CMAP,
                             shading="nearest", rasterized=True)
        contour = ax.contour(q, sc, field, colors=INK_SECONDARY,
                             linewidths=0.6, alpha=0.6)
        ax.clabel(contour, fontsize=6.5, fmt="%.3g", colors=INK_SECONDARY)
        if key == "max_wall_surface_temperature_K":
            feasibility = ax.contour(q, sc, field, levels=[limit],
                                     colors=[SERIES_COLORS[0]], linewidths=2.2)
            ax.clabel(feasibility, fmt=f"{limit:.0f} K limit", fontsize=7.5,
                      colors=SERIES_COLORS[0])
        ax.set_title(title)
        ax.set_xlabel("mean wall heat flux  [kW m$^{-2}$]")
        ax.set_ylabel("steam-to-carbon ratio  [-]")
        ax.grid(False)
        bar = fig.colorbar(mesh, ax=ax, pad=0.015)
        bar.outline.set_visible(False)
    return fig


def plot_grid_convergence(study: dict):
    """Mesh-refinement study of the radial discretisation."""
    import matplotlib.pyplot as plt

    use_style()
    n = np.asarray(study["n_radial"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    ax.plot(n, study["methane_conversion"], color=SERIES_COLORS[0], marker="o",
            label="CH$_4$ conversion  [-]")
    ax.set_title("(a)  Outlet conversion against radial resolution")
    ax.set_xlabel("radial cells  [-]")
    ax.set_ylabel("conversion  [-]")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(n, study["max_radial_dT_K"], color=SERIES_COLORS[1], marker="o",
            label=r"peak radial $\Delta T$  [K]")
    ax.plot(n, study["max_wall_surface_temperature_K"]
            - np.min(study["max_wall_surface_temperature_K"]),
            color=SERIES_COLORS[2], marker="s",
            label="peak wall $T$ minus coarsest  [K]")
    ax.set_title("(b)  Thermal metrics against radial resolution")
    ax.set_xlabel("radial cells  [-]")
    ax.set_ylabel("temperature difference  [K]")
    ax.legend(loc="best")
    return fig


def plot_model_assessment(sensitivity: dict, variants: dict):
    """What the radial resolution buys, and how much the closures move it.

    Panel (a) sweeps the effective radial conductivity from the low end of the
    published range up to the lumped limit, so the radial gradient can be seen
    collapsing onto the wall-film contribution that survives it.  Panel (b)
    shows the signed deviation each discrete closure choice produces, drawn as
    bars about zero because the sign is the point.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.subplots_adjust(wspace=0.34)

    # (a) radial gradient against the conductivity ------------------------
    ax = axes[0]
    k = sensitivity["conductivities"]
    ax.plot(k, sensitivity["metrics"]["max_radial_dT_K"],
            color=SERIES_COLORS[0], marker="o",
            label=r"peak radial $\Delta T$")
    reference = sensitivity["reference"]
    ax.plot([30.0], [reference["max_radial_dT_K"]], marker="*", markersize=15,
            color=SERIES_COLORS[7], linestyle="none",
            label="correlated $k_{er}$")
    lumped = sensitivity.get("lumped_radial_dT")
    if lumped is not None:
        ax.axhline(lumped, color=INK_MUTED, linestyle=(0, (4, 3)),
                   linewidth=1.1)
        ax.annotate("wall-film contribution alone\n(radially lumped limit)",
                    xy=(0.04, lumped), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=7.5)
    ax.set_title("(a)  Radial gradient against the bed conductivity")
    ax.set_xlabel("effective radial conductivity  $k_{er}$  [W m$^{-1}$ K$^{-1}$]")
    ax.set_ylabel(r"peak radial $\Delta T$  [K]")
    ax.legend(loc="upper right")

    # (b) one-at-a-time closure deviations --------------------------------
    ax = axes[1]
    rows = variants["rows"][1:]
    base = variants["baseline"]
    labels = [row["variant"] for row in rows]
    wall = [row["max_wall_surface_temperature_K"]
            - base["max_wall_surface_temperature_K"] for row in rows]
    radial = [row["max_radial_dT_K"] - base["max_radial_dT_K"] for row in rows]

    y = np.arange(len(labels))
    height = 0.38
    ax.barh(y + height / 2 + 0.01, wall, height=height,
            color=SERIES_COLORS[0], label="peak wall surface $T$")
    ax.barh(y - height / 2 - 0.01, radial, height=height,
            color=SERIES_COLORS[1], label=r"peak radial $\Delta T$")
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title("(b)  Effect of changing one closure at a time")
    ax.set_xlabel("deviation from baseline  [K]")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return fig
