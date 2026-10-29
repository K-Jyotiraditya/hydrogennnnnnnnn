#!/usr/bin/env python3
"""
Command-line driver for the 2D pseudo-heterogeneous SMR tube model.

Usage
-----
    python main.py baseline      # solve the baseline tube, report, plot
    python main.py verify        # convergence and balance-closure study
    python main.py coking        # carbon-deposition analysis and limit map
    python main.py optimize      # tune S/C and wall heat flux
    python main.py pareto        # epsilon-constraint Pareto fronts
    python main.py envelope      # feasible firing envelope and its corner
    python main.py assess        # what 2D buys, and closure uncertainty
    python main.py sensitivity   # design-space scan over S/C and flux
    python main.py all           # everything, in order

Common options
-------------
    --outdir DIR     where figures and tables are written (default: results)
    --no-figures     skip plotting (text and CSV only)
    --radial N       radial cells
    --sc X           inlet steam-to-carbon ratio
    --flux X         mean inner-wall heat flux [kW/m^2]
    --wall-mode      use a specified wall temperature profile instead of flux

Every command writes a text report to stdout and to ``<outdir>/report.txt``,
and tabulates its numerical output as CSV so the figures always have a table
view beside them.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from smr import Reformer2D, analyze_coking, default_config
from smr.assessment import (
    REPORTED_METRICS,
    closure_sensitivity,
    closure_variants,
    dimensionality_comparison,
)
from smr.coking import coking_boundary_map
from smr.optimization import (
    Constraints,
    DesignSpace,
    ObjectiveWeights,
    operating_envelope,
    optimize_operating_point,
    pareto_front,
    sensitivity_scan,
)
from smr.species import IDX, SPECIES

# --------------------------------------------------------------------------
# Output plumbing
# --------------------------------------------------------------------------


class Tee:
    """Write to stdout and accumulate for the report file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 72)
        self(f"  {title}")
        self("=" * 72)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def build_config(args: argparse.Namespace):
    """Apply the command-line overrides to the baseline configuration."""
    cfg = default_config()
    if args.radial is not None:
        cfg = cfg.with_numerics(n_radial=args.radial)
    if args.rtol is not None:
        cfg = cfg.with_numerics(rtol=args.rtol)
    if args.sc is not None:
        # Hold the methane feed fixed and add steam, as a plant would.
        ch4 = cfg.feed.molar_flows()[IDX["CH4"]]
        minor = (cfg.feed.hydrogen_fraction + cfg.feed.co2_fraction
                 + cfg.feed.nitrogen_fraction + cfg.feed.co_fraction)
        total = ch4 * (1.0 + args.sc) / (1.0 - minor)
        cfg = cfg.with_feed(steam_to_carbon=args.sc, molar_flow=total)
    if args.flux is not None:
        cfg = cfg.with_operating(average_heat_flux=args.flux * 1.0e3)
    if args.wall_mode:
        cfg = cfg.with_operating(mode="wall_temperature")
    if args.eta is not None:
        cfg = cfg.with_catalyst(effectiveness_model=args.eta)
    return cfg


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_baseline(args, out: Tee, outdir: Path) -> dict:
    """Solve the baseline tube and write profiles, tables and figures."""
    cfg = build_config(args)
    out.rule("Baseline simulation")
    out(cfg.describe())
    out("")

    model = Reformer2D(cfg)
    solution = model.solve()
    if not solution.success:
        out(f"  WARNING: integrator reported '{solution.message}'")
    out(solution.report())

    # --- tables ---------------------------------------------------------
    y_bulk = solution.bulk_mole_fractions()
    header = (["z_m", "P_bar", "T_bulk_K", "T_centre_K", "T_near_wall_K",
               "T_wall_surface_K", "T_wall_outer_K", "q_wall_kW_m2",
               "X_CH4", "H2_yield", "k_er_W_mK", "eta_R1"]
              + [f"y_{s}" for s in SPECIES])
    rows = []
    for k in range(solution.n_axial):
        rows.append([
            f"{solution.z[k]:.4f}",
            f"{solution.pressure[k] / 1e5:.5f}",
            f"{solution.bulk_temperature()[k]:.3f}",
            f"{solution.temperature[k, 0]:.3f}",
            f"{solution.temperature[k, -1]:.3f}",
            f"{solution.wall_surface_temperature()[k]:.3f}",
            f"{solution.outer_wall_temperature()[k]:.3f}",
            f"{solution.diagnostics['q_wall'][k] / 1e3:.4f}",
            f"{solution.methane_conversion()[k]:.6f}",
            f"{solution.hydrogen_yield()[k]:.6f}",
            f"{solution.area_average(solution.diagnostics['k_er'])[k]:.4f}",
            f"{solution.area_average(solution.diagnostics['eta'][:, :, 0])[k]:.6f}",
        ] + [f"{y_bulk[k, i]:.6f}" for i in range(len(SPECIES))])
    write_csv(outdir / "axial_profiles.csv", header, rows)

    radial_header = ["z_m"] + [f"T_r{r * 1e3:.1f}mm_K"
                              for r in solution.grid.centers]
    radial_rows = [[f"{solution.z[k]:.4f}"]
                   + [f"{t:.3f}" for t in solution.temperature[k]]
                   for k in range(solution.n_axial)]
    write_csv(outdir / "radial_temperature.csv", radial_header, radial_rows)
    out("")
    out(f"  Tables  -> {outdir / 'axial_profiles.csv'}")
    out(f"            {outdir / 'radial_temperature.csv'}")

    if args.figures:
        from smr import visualization as viz
        figures = {
            "fig01_axial_overview.png": viz.plot_axial_overview(solution),
            "fig02_radial_temperature.png": viz.plot_radial_temperature(solution),
            "fig03_radial_profiles.png": viz.plot_radial_profiles(solution),
            "fig04_kinetic_diagnostics.png":
                viz.plot_kinetic_diagnostics(solution),
        }
        for name, fig in figures.items():
            fig.savefig(outdir / name)
            out(f"  Figure  -> {outdir / name}")
        _close_all()
    return {"solution": solution, "config": cfg}


def cmd_verify(args, out: Tee, outdir: Path) -> dict:
    """Grid and tolerance convergence, plus attribution of the balance error."""
    cfg = build_config(args)
    out.rule("Verification: convergence and conservation")

    out("Radial grid refinement (rtol = 1e-7)")
    out("  n_r   t[s]   nfev    X_CH4      T_out[K]  T_wall[K]  dT_rad[K]"
        "  dP[bar]   enthalpy closure")
    study = {k: [] for k in ("n_radial", "methane_conversion",
                             "max_radial_dT_K", "max_wall_surface_temperature_K")}
    rows = []
    for n_radial in (4, 6, 8, 12, 16, 24, 32):
        solution = Reformer2D(
            cfg.with_numerics(n_radial=n_radial, rtol=1.0e-7)).solve()
        m = solution.summary()
        out(f"  {n_radial:3d}  {solution.solve_time:5.2f} "
            f"{solution.n_rhs_evaluations:6d}  {m['methane_conversion']:.6f} "
            f"{m['outlet_bulk_temperature_K']:9.3f} "
            f"{m['max_wall_surface_temperature_K']:9.2f} "
            f"{m['max_radial_dT_K']:9.2f} {m['pressure_drop_bar']:8.3f}  "
            f"{m['energy_balance_rel_error']:+.3e}")
        for key in study:
            study[key].append(n_radial if key == "n_radial" else m[key])
        rows.append([n_radial, f"{m['methane_conversion']:.8f}",
                     f"{m['outlet_bulk_temperature_K']:.4f}",
                     f"{m['max_wall_surface_temperature_K']:.4f}",
                     f"{m['max_radial_dT_K']:.4f}",
                     f"{m['pressure_drop_bar']:.5f}",
                     f"{m['energy_balance_rel_error']:.4e}"])
    write_csv(outdir / "grid_convergence.csv",
              ["n_radial", "X_CH4", "T_out_K", "T_wall_max_K", "dT_radial_K",
               "dP_bar", "enthalpy_closure"], rows)

    out("")
    out("Integration tolerance (12 radial cells)")
    out("  rtol     t[s]   nfev    X_CH4      T_out[K]")
    for rtol in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8):
        solution = Reformer2D(cfg.with_numerics(rtol=rtol)).solve()
        m = solution.summary()
        out(f"  {rtol:.0e}  {solution.solve_time:5.2f} "
            f"{solution.n_rhs_evaluations:6d}  {m['methane_conversion']:.6f} "
            f"{m['outlet_bulk_temperature_K']:9.3f}")

    out("")
    out("Attribution of the global enthalpy-balance residual")
    out("  The finite-volume radial operator is exactly conservative, so with")
    out("  both optional transport terms switched off the balance must close")
    out("  to integrator tolerance.  The residual of the full model is the")
    out("  enthalpy carried by radial species dispersion, which is omitted on")
    out("  purpose: the same eddy transport is already represented in k_er,")
    out("  and counting it twice would overstate radial heat transfer.")
    out("")
    out("  model variant                    residual[W]   relative")
    variants = [
        ("full model", {}),
        ("no radial mass dispersion", {"radial_mass_dispersion": False}),
        ("no gas-solid film resistance", {"film_resistance": False}),
        ("neither", {"radial_mass_dispersion": False,
                     "film_resistance": False}),
    ]
    balance_rows = []
    for label, kwargs in variants:
        solution = Reformer2D(cfg.with_numerics(rtol=1.0e-7, **kwargs)).solve()
        eb = solution.energy_balance_error()
        mb = solution.mass_balance_error()
        out(f"  {label:32s} {eb['residual_W']:+10.2f}   "
            f"{eb['relative_error']:+.3e}")
        balance_rows.append([label, f"{eb['residual_W']:.4f}",
                             f"{eb['relative_error']:.4e}",
                             f"{mb['C']:.3e}", f"{mb['H']:.3e}",
                             f"{mb['O']:.3e}"])
    write_csv(outdir / "balance_attribution.csv",
              ["variant", "residual_W", "relative", "C_balance", "H_balance",
               "O_balance"], balance_rows)

    out("")
    out("Atomic balances of the full model (relative error)")
    solution = Reformer2D(cfg.with_numerics(rtol=1.0e-7)).solve()
    for element, error in solution.mass_balance_error().items():
        out(f"  {element}: {error:+.3e}")

    out("")
    out("Equilibrium cross-check 1: two independent routes to the composition")
    out("  Reaction extents with the Twigg correlations, against direct Gibbs")
    out("  energy minimisation under atom conservation.  The two share nothing")
    out("  beyond the NASA-7 polynomials.")
    out("")
    from smr.coking import (equilibrium_composition,
                            equilibrium_composition_gibbs)
    from smr.species import SPECIES as _SPECIES
    n_eq = np.zeros(len(_SPECIES))
    n_eq[IDX["CH4"]], n_eq[IDX["H2O"]] = 1.0, 3.0
    out("     T[K]  route        " + "  ".join(f"{s:>8}" for s in _SPECIES))
    eq_rows = []
    for T in (900.0, 1000.0, 1100.0, 1200.0):
        a = equilibrium_composition(T, cfg.feed.inlet_pressure, n_eq)
        b = equilibrium_composition_gibbs(T, cfg.feed.inlet_pressure, n_eq)
        out(f"  {T:7.0f}  extents      "
            + "  ".join(f"{v:8.5f}" for v in a))
        out(f"           gibbs-min    "
            + "  ".join(f"{v:8.5f}" for v in b))
        out(f"           max difference {np.max(np.abs(a - b)):.2e}")
        eq_rows.append([f"{T:.0f}"] + [f"{v:.6f}" for v in a]
                       + [f"{v:.6f}" for v in b]
                       + [f"{np.max(np.abs(a - b)):.3e}"])
    write_csv(outdir / "equilibrium_cross_check.csv",
              ["T_K"] + [f"extents_{s}" for s in _SPECIES]
              + [f"gibbs_{s}" for s in _SPECIES] + ["max_difference"], eq_rows)
    out(f"  Table   -> {outdir / 'equilibrium_cross_check.csv'}")

    out("")
    out("Equilibrium cross-check 2: Twigg correlations vs NASA-7 polynomials")
    from smr.kinetics import (equilibrium_constants,
                              equilibrium_constants_from_thermo)
    out("     T[K]      K1 Twigg     K1 thermo    K2 Twigg     K2 thermo")
    for T in (700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0):
        a = equilibrium_constants(T)
        b = equilibrium_constants_from_thermo(T)
        out(f"  {T:7.0f}  {a[0]:12.5g} {b[0]:12.5g} {a[1]:12.5g} {b[1]:12.5g}")

    if args.figures:
        from smr import visualization as viz
        fig = viz.plot_grid_convergence(study)
        fig.savefig(outdir / "fig05_grid_convergence.png")
        out("")
        out(f"  Figure  -> {outdir / 'fig05_grid_convergence.png'}")
        _close_all()
    return {}


def cmd_coking(args, out: Tee, outdir: Path) -> dict:
    """Carbon-deposition analysis of the solved tube and the limit diagram."""
    cfg = build_config(args)
    out.rule("Carbon deposition (coking) analysis")
    solution = Reformer2D(cfg).solve()
    report = analyze_coking(solution)
    out(report.report())

    rows = []
    for k in range(solution.n_axial):
        rows.append([
            f"{solution.z[k]:.4f}",
            f"{report.worst_activity[k].max():.6g}",
            f"{report.activity_actual[k, :, 0].max():.6g}",
            f"{report.activity_actual[k, :, 1].max():.6g}",
            f"{report.activity_actual[k, :, 2].max():.6g}",
            f"{report.local_sc[k]:.4f}",
            f"{report.critical_sc[k]:.4f}",
            int(report.node_flagged[k].any()),
        ])
    write_csv(outdir / "coking_axial.csv",
              ["z_m", "a_C_worst", "a_C_cracking", "a_C_boudouard",
               "a_C_co_reduction", "local_SC", "critical_SC_equilibrated",
               "flagged"], rows)
    out(f"  Table   -> {outdir / 'coking_axial.csv'}")

    out("")
    out("Carbon-free boundary of the equilibrated gas")
    temperatures = np.linspace(700.0, 1300.0, 31)
    boundary = coking_boundary_map(temperatures, cfg.feed.inlet_pressure)
    out("     T[K]   critical S/C   cracking   Boudouard   CO reduction")
    for i in range(0, temperatures.size, 3):
        by = boundary["critical_sc_by_route"][i]
        out(f"  {temperatures[i]:7.0f}   {boundary['critical_sc'][i]:12.3f}"
            f"   {by[0]:8.3f}   {by[1]:9.3f}   {by[2]:12.3f}")
    write_csv(outdir / "carbon_boundary.csv",
              ["T_K", "critical_SC", "SC_cracking", "SC_boudouard",
               "SC_co_reduction"],
              [[f"{temperatures[i]:.2f}",
                f"{boundary['critical_sc'][i]:.5f}",
                *[f"{v:.5f}" for v in boundary['critical_sc_by_route'][i]]]
               for i in range(temperatures.size)])
    out(f"  Table   -> {outdir / 'carbon_boundary.csv'}")

    if args.figures:
        from smr import visualization as viz
        viz.plot_coking(report).savefig(outdir / "fig06_coking.png")
        viz.plot_carbon_boundary_map(
            boundary, cfg.feed.inlet_pressure).savefig(
                outdir / "fig07_carbon_boundary.png")
        out(f"  Figure  -> {outdir / 'fig06_coking.png'}")
        out(f"  Figure  -> {outdir / 'fig07_carbon_boundary.png'}")
        _close_all()
    return {"report": report}


def cmd_optimize(args, out: Tee, outdir: Path) -> dict:
    """Tune the steam-to-carbon ratio and the mean wall heat flux."""
    cfg = build_config(args)
    constraints = Constraints(max_wall_temperature=args.wall_limit,
                              carbon_free=args.carbon_free)
    weights = ObjectiveWeights(steam=args.steam_weight)
    space = DesignSpace()

    out.rule("Operating-point optimisation")
    out(f"  Objective  : maximise H2 production, penalise CH4 slip"
        f" (w = {weights.methane_slip:.2f}) and steam"
        f" (w = {weights.steam:.2f})")
    out(f"  Constraints: wall surface T <= {constraints.max_wall_temperature:.0f} K,"
        f" dP <= {constraints.max_pressure_drop / 1e5:.1f} bar"
        + (", carbon free" if constraints.carbon_free else ""))
    out(f"  Bounds     : S/C in {space.steam_to_carbon},"
        f" q in {tuple(v / 1e3 for v in space.heat_flux)} kW/m2")
    out("")

    results = {}
    for method in args.methods:
        result = optimize_operating_point(
            cfg, space=space, weights=weights, constraints=constraints,
            method=method, coking=args.carbon_free,
            max_iterations=args.max_iterations, verbose=False)
        out(result.report())
        out("")
        results[method] = result

    best = min(results.values(), key=lambda r: r.objective)
    out(f"  Best method: {best.method} (objective {best.objective:.5f})")

    write_csv(outdir / "optimization.csv",
              ["method", "S_C", "q_mean_kW_m2", "objective", "X_CH4",
               "H2_mol_s", "slip_dry", "T_wall_max_K", "dP_bar", "duty_kW",
               "evaluations", "wall_time_s"],
              [[r.method, f"{r.steam_to_carbon:.4f}",
                f"{r.heat_flux / 1e3:.3f}", f"{r.objective:.6f}",
                f"{r.metrics['methane_conversion']:.6f}",
                f"{r.metrics['hydrogen_production_mol_per_s']:.5f}",
                f"{r.metrics['methane_slip_dry']:.6f}",
                f"{r.metrics['max_wall_surface_temperature_K']:.3f}",
                f"{r.metrics['pressure_drop_bar']:.4f}",
                f"{r.metrics['heat_duty_kW']:.2f}",
                r.n_evaluations, f"{r.wall_time:.2f}"]
               for r in results.values()])
    out(f"  Table   -> {outdir / 'optimization.csv'}")

    if args.figures:
        from smr import visualization as viz
        fig = viz.plot_axial_overview(
            best.solution, title="Optimised operating point  --  "
            f"S/C = {best.steam_to_carbon:.2f}, "
            f"q = {best.heat_flux / 1e3:.1f} kW/m2")
        fig.savefig(outdir / "fig08_optimum_axial.png")
        out(f"  Figure  -> {outdir / 'fig08_optimum_axial.png'}")
        _close_all()
    return {"results": results, "best": best}


def cmd_pareto(args, out: Tee, outdir: Path) -> dict:
    """Epsilon-constraint Pareto fronts."""
    cfg = build_config(args)
    out.rule("Pareto fronts (epsilon-constraint)")
    fronts = {}

    out("Front 1: hydrogen production against the tube-wall temperature limit")
    out("  (the production / tube-life trade-off)")
    fronts["wall_temperature"] = pareto_front(
        cfg, variable="wall_temperature",
        epsilon=np.linspace(1060.0, 1200.0, args.points), verbose=False)
    out(fronts["wall_temperature"].to_table())

    out("")
    out("Front 2: hydrogen production against steam consumption")
    out("  (the production / utility-cost trade-off)")
    fronts["steam"] = pareto_front(
        cfg, variable="steam",
        epsilon=np.linspace(2.0, 4.5, args.points), verbose=False)
    out(fronts["steam"].to_table())

    for name, front in fronts.items():
        keep = front.non_dominated_mask()
        write_csv(outdir / f"pareto_{name}.csv",
                  ["epsilon", "S_C", "q_mean_kW_m2", "T_wall_max_K",
                   "X_CH4", "H2_mol_s", "slip_dry", "steam_mol_s",
                   "feasible", "non_dominated"],
                  [[f"{front.epsilon[i]:.4f}",
                    f"{front.steam_to_carbon[i]:.4f}",
                    f"{front.heat_flux[i] / 1e3:.3f}",
                    f"{front.max_wall_temperature[i]:.3f}",
                    f"{front.methane_conversion[i]:.6f}",
                    f"{front.hydrogen_production[i]:.5f}",
                    f"{front.methane_slip[i]:.6f}",
                    f"{front.steam_flow[i]:.5f}",
                    int(front.feasible[i]), int(keep[i])]
                   for i in range(front.epsilon.size)])
        out(f"  Table   -> {outdir / f'pareto_{name}.csv'}")

    if args.figures:
        from smr import visualization as viz
        for i, (name, front) in enumerate(fronts.items(), start=9):
            fig = viz.plot_pareto_front(front)
            path = outdir / f"fig{i:02d}_pareto_{name}.png"
            fig.savefig(path)
            out(f"  Figure  -> {path}")
        _close_all()
    return {"fronts": fronts}


def cmd_envelope(args, out: Tee, outdir: Path) -> dict:
    """Feasible firing envelope and the constraint corner."""
    cfg = build_config(args)
    constraints = Constraints(max_wall_temperature=args.wall_limit)
    out.rule("Feasible operating envelope")
    out("  For each S/C the mean wall heat flux is raised until a constraint")
    out("  becomes tight, and the binding constraint is recorded.")
    out("")
    envelope = operating_envelope(
        cfg, steam_to_carbon=np.linspace(2.0, 4.5, 2 * args.points - 1),
        constraints=constraints, verbose=False)

    out("   S/C   q_max[kW/m2]  binding            X_CH4   H2[mol/s]"
        "  T_wall[K]  dP[bar]")
    rows = []
    for i, sc in enumerate(envelope["steam_to_carbon"]):
        out(f"  {sc:5.2f}  {envelope['max_heat_flux'][i] / 1e3:11.2f}"
            f"  {envelope['binding'][i]:<17s}"
            f"{envelope['methane_conversion'][i]:.5f}"
            f"  {envelope['hydrogen_production'][i]:9.3f}"
            f"  {envelope['max_wall_temperature'][i]:9.1f}"
            f"  {envelope['pressure_drop'][i]:7.3f}")
        rows.append([f"{sc:.4f}", f"{envelope['max_heat_flux'][i] / 1e3:.3f}",
                     envelope["binding"][i],
                     f"{envelope['methane_conversion'][i]:.6f}",
                     f"{envelope['hydrogen_production'][i]:.5f}",
                     f"{envelope['max_wall_temperature'][i]:.3f}",
                     f"{envelope['pressure_drop'][i]:.4f}"])
    write_csv(outdir / "operating_envelope.csv",
              ["S_C", "q_max_kW_m2", "binding_constraint", "X_CH4",
               "H2_mol_s", "T_wall_max_K", "dP_bar"], rows)
    out("")
    out(f"  Constraint corner (last wall-limited S/C) : "
        f"{envelope['corner_steam_to_carbon']:.2f}")
    out(f"  S/C maximising hydrogen on the envelope   : "
        f"{envelope['best_steam_to_carbon']:.2f}")
    out("")
    out("  Below the corner the metallurgical limit binds and extra steam buys")
    out("  conversion at constant peak wall temperature.  Above it the Ergun")
    out("  pressure drop binds, the firing has to be cut faster than the")
    out("  chemistry improves, and hydrogen production falls.  The optimum")
    out("  therefore sits at the corner -- which is an independent, purely")
    out("  geometric confirmation of what the numerical optimiser reports.")
    out(f"  Table   -> {outdir / 'operating_envelope.csv'}")

    if args.figures:
        from smr import visualization as viz
        fig = viz.plot_operating_envelope(envelope)
        fig.savefig(outdir / "fig12_operating_envelope.png")
        out(f"  Figure  -> {outdir / 'fig12_operating_envelope.png'}")
        _close_all()
    return {"envelope": envelope}


def cmd_assess(args, out: Tee, outdir: Path) -> dict:
    """Does radial resolution change an answer, and how much do closures move it?"""
    cfg = build_config(args)
    out.rule("Model assessment: dimensionality and closure uncertainty")

    out("Is the second dimension earning its cost?")
    out("  The same tube is run against its own radially lumped limit, taken by")
    out("  driving k_er to 1e5 W/m/K so the bed is isothermal in r while every")
    out("  other closure, the wall film included, is left untouched.  The")
    out("  difference is the contribution of the resolved radial profile alone.")
    out("")
    comparison = dimensionality_comparison(cfg, verbose=False)
    two_d, lumped, delta = (comparison["two_dimensional"],
                            comparison["lumped"], comparison["delta"])
    out(f"  {'metric':<30} {'2D':>12} {'lumped':>12} {'difference':>12}")
    out("  " + "-" * 68)
    rows = []
    for key, label, scale in REPORTED_METRICS:
        out(f"  {label:<30} {two_d[key] * scale:12.3f} "
            f"{lumped[key] * scale:12.3f} {delta[key] * scale:12.3f}")
        rows.append([label, f"{two_d[key] * scale:.5f}",
                     f"{lumped[key] * scale:.5f}", f"{delta[key] * scale:.5f}"])
    write_csv(outdir / "dimensionality.csv",
              ["metric", "two_dimensional", "radially_lumped", "difference"],
              rows)
    out("")
    out(f"  VERDICT: {comparison['verdict']}")

    out("")
    out("Effective radial conductivity sweep")
    out("  Published k_er correlations for comparable geometries disagree by a")
    out("  factor approaching two; the radial gradient scales roughly as 1/k_er.")
    out("")
    sensitivity = closure_sensitivity(cfg, verbose=False)
    out(f"  {'k_er [W/m/K]':>12} {'peak wall T [K]':>18} "
        f"{'peak radial dT [K]':>20} {'X_CH4 [%]':>12}")
    sens_rows = []
    for i, k in enumerate(sensitivity["conductivities"]):
        m = sensitivity["metrics"]
        out(f"  {k:12.1f} {m['max_wall_surface_temperature_K'][i]:18.2f} "
            f"{m['max_radial_dT_K'][i]:20.2f} "
            f"{m['methane_conversion'][i] * 100:12.3f}")
        sens_rows.append([f"{k:.2f}",
                          f"{m['max_wall_surface_temperature_K'][i]:.4f}",
                          f"{m['max_radial_dT_K'][i]:.4f}",
                          f"{m['methane_conversion'][i]:.6f}"])
    write_csv(outdir / "closure_sensitivity.csv",
              ["k_er_W_mK", "T_wall_max_K", "dT_radial_K", "X_CH4"], sens_rows)
    lo, hi, half = sensitivity["band"]["max_radial_dT_K"]
    out("")
    out(f"  Peak radial dT spans {lo:.1f} to {hi:.1f} K over this range.  The")
    out(f"  correlated value of {sensitivity['reference']['max_radial_dT_K']:.1f}"
        f" K therefore carries about +/-{half:.0f} K of closure uncertainty,")
    out("  and should be quoted with it.")

    out("")
    out("One-at-a-time closure variants")
    variants = closure_variants(cfg, verbose=False)
    base = variants["baseline"]
    out(f"  {'variant':<40} {'X_CH4 [%]':>10} {'T_wall [K]':>12} "
        f"{'dT_rad [K]':>12}")
    var_rows = []
    for row in variants["rows"]:
        out(f"  {row['variant']:<40} "
            f"{row['methane_conversion'] * 100:10.2f} "
            f"{row['max_wall_surface_temperature_K']:12.1f} "
            f"{row['max_radial_dT_K']:12.1f}")
        var_rows.append([row["variant"], f"{row['methane_conversion']:.6f}",
                         f"{row['max_wall_surface_temperature_K']:.3f}",
                         f"{row['max_radial_dT_K']:.3f}",
                         f"{row['outlet_bulk_temperature_K']:.3f}"])
    write_csv(outdir / "closure_variants.csv",
              ["variant", "X_CH4", "T_wall_max_K", "dT_radial_K", "T_out_K"],
              var_rows)
    out("")
    out("  Largest deviation from baseline over all variants:")
    for key, label, scale in REPORTED_METRICS:
        out(f"    {label:<30} {variants['largest_deviation'][key] * scale:10.3f}")
    out("")
    out("  Reading: the duty predictions are robust to every closure tested,")
    out("  while the thermal diagnostics carry a band of tens of kelvin.  That")
    out("  is the right way round for a heat-transfer-limited tube, and it is")
    out("  why the conversion number is quoted plainly while the radial")
    out("  gradient is quoted with an uncertainty.")

    if args.figures:
        from smr import visualization as viz
        sensitivity["lumped_radial_dT"] = lumped["max_radial_dT_K"]
        fig = viz.plot_model_assessment(sensitivity, variants)
        fig.savefig(outdir / "fig13_model_assessment.png")
        out(f"  Figure  -> {outdir / 'fig13_model_assessment.png'}")
        _close_all()
    return {"comparison": comparison, "sensitivity": sensitivity,
            "variants": variants}


def cmd_sensitivity(args, out: Tee, outdir: Path) -> dict:
    """Full-factorial scan of the design space."""
    cfg = build_config(args)
    out.rule("Design-space sensitivity scan")
    n = args.points
    grids = sensitivity_scan(
        cfg,
        steam_to_carbon=np.linspace(2.0, 4.5, n),
        heat_flux=np.linspace(4.0e4, 9.0e4, n),
        coking=True, verbose=False)

    out("   S/C   q[kW/m2]   X_CH4    H2[mol/s]  slip[%]  T_wall[K]"
        "  dT_rad[K]  dP[bar]   a_C,max")
    rows = []
    for i, sc in enumerate(grids["steam_to_carbon"]):
        for j, q in enumerate(grids["heat_flux"]):
            out(f"  {sc:4.2f}  {q / 1e3:8.2f}  {grids['methane_conversion'][i, j]:.5f}"
                f"  {grids['hydrogen_production_mol_per_s'][i, j]:9.3f}"
                f"  {grids['methane_slip_dry'][i, j] * 100:7.2f}"
                f"  {grids['max_wall_surface_temperature_K'][i, j]:9.1f}"
                f"  {grids['max_radial_dT_K'][i, j]:9.1f}"
                f"  {grids['pressure_drop_bar'][i, j]:7.3f}"
                f"  {grids['max_carbon_activity'][i, j]:8.3g}")
            rows.append([f"{sc:.4f}", f"{q / 1e3:.3f}",
                         f"{grids['methane_conversion'][i, j]:.6f}",
                         f"{grids['hydrogen_production_mol_per_s'][i, j]:.5f}",
                         f"{grids['methane_slip_dry'][i, j]:.6f}",
                         f"{grids['max_wall_surface_temperature_K'][i, j]:.3f}",
                         f"{grids['max_radial_dT_K'][i, j]:.3f}",
                         f"{grids['pressure_drop_bar'][i, j]:.4f}",
                         f"{grids['max_carbon_activity'][i, j]:.5g}"])
    write_csv(outdir / "sensitivity.csv",
              ["S_C", "q_mean_kW_m2", "X_CH4", "H2_mol_s", "slip_dry",
               "T_wall_max_K", "dT_radial_K", "dP_bar", "a_C_max"], rows)
    out(f"  Table   -> {outdir / 'sensitivity.csv'}")

    if args.figures:
        from smr import visualization as viz
        fig = viz.plot_sensitivity(
            grids, constraints=Constraints(max_wall_temperature=args.wall_limit))
        fig.savefig(outdir / "fig11_sensitivity.png")
        out(f"  Figure  -> {outdir / 'fig11_sensitivity.png'}")
        _close_all()
    return {"grids": grids}


def _close_all() -> None:
    import matplotlib.pyplot as plt
    plt.close("all")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

COMMANDS = {
    "baseline": cmd_baseline,
    "verify": cmd_verify,
    "coking": cmd_coking,
    "optimize": cmd_optimize,
    "pareto": cmd_pareto,
    "envelope": cmd_envelope,
    "assess": cmd_assess,
    "sensitivity": cmd_sensitivity,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS) + ["all"],
                        help="which analysis to run")
    parser.add_argument("--outdir", default="results",
                        help="output directory (default: results)")
    parser.add_argument("--no-figures", dest="figures", action="store_false",
                        help="skip plotting; write text and CSV only")
    parser.add_argument("--radial", type=int, default=None,
                        help="number of radial finite-volume cells")
    parser.add_argument("--rtol", type=float, default=None,
                        help="integrator relative tolerance")
    parser.add_argument("--sc", type=float, default=None,
                        help="inlet steam-to-carbon ratio")
    parser.add_argument("--flux", type=float, default=None,
                        help="mean inner-wall heat flux [kW/m^2]")
    parser.add_argument("--wall-mode", action="store_true",
                        help="specify a wall temperature profile instead of flux")
    parser.add_argument("--eta", default=None,
                        choices=["thiele", "constant", "none"],
                        help="intraparticle effectiveness-factor model")
    parser.add_argument("--wall-limit", type=float, default=1150.0,
                        help="metallurgical wall temperature limit [K]")
    parser.add_argument("--carbon-free", action="store_true",
                        help="add the carbon-activity constraint to the optimiser")
    parser.add_argument("--steam-weight", type=float, default=0.10,
                        help="objective weight on steam consumption")
    parser.add_argument("--methods", nargs="+", default=["slsqp"],
                        choices=["slsqp", "nelder-mead",
                                 "differential_evolution"],
                        help="optimisation methods to run")
    parser.add_argument("--max-iterations", type=int, default=60,
                        help="iteration budget for the optimiser")
    parser.add_argument("--points", type=int, default=8,
                        help="points per axis in Pareto and scan commands")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.figures:
        import matplotlib
        matplotlib.use("Agg")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = Tee()
    out("2D pseudo-heterogeneous steam methane reformer -- "
        "thermal-hydraulic simulation and performance analysis")
    out(f"run: {time.strftime('%Y-%m-%d %H:%M:%S')}   command: {args.command}")

    start = time.perf_counter()
    commands = list(COMMANDS) if args.command == "all" else [args.command]
    for name in commands:
        COMMANDS[name](args, out, outdir)
    out("")
    out(f"Total wall time: {time.perf_counter() - start:.1f} s")

    report_path = outdir / f"report_{args.command}.txt"
    out.save(report_path)
    print(f"\nReport -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
