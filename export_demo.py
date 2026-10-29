#!/usr/bin/env python3
"""
Precompute a grid of reformer solutions for the interactive web demonstration.

The browser cannot run the Python model, so the demo interpolates between
solutions computed here.  The grid spans the same design space the optimiser
searches, and every point is a full 2D solve -- nothing in the demo is fitted,
smoothed or invented, it is bilinear interpolation between real model output.

Writes ``results/demo_data.json``.  Run it again after changing the model.

    python export_demo.py [--sc-points 11] [--flux-points 11]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from smr import Reformer2D, analyze_coking, default_config
from smr.optimization import Constraints, fast_numerics
from smr.species import IDX, SPECIES

#: Axial stations at which the radial temperature profile is stored.
RADIAL_STATION_FRACTIONS = (0.0, 0.05, 0.15, 0.5, 1.0)


def build_case(steam_to_carbon: float, heat_flux: float,
               n_axial: int, constraints: Constraints) -> dict:
    """Solve one operating point and reduce it to the demo payload."""
    base = fast_numerics(default_config(), n_radial=10,
                         n_axial_output=n_axial, rtol=1.0e-6)
    feed = base.feed
    methane = feed.molar_flows()[IDX["CH4"]]
    minor = (feed.hydrogen_fraction + feed.co2_fraction
             + feed.nitrogen_fraction + feed.co_fraction)
    total = methane * (1.0 + steam_to_carbon) / (1.0 - minor)
    config = (base
              .with_feed(steam_to_carbon=steam_to_carbon, molar_flow=total)
              .with_operating(average_heat_flux=heat_flux))

    solution = Reformer2D(config).solve()
    report = analyze_coking(solution, include_equilibrated=False)
    summary = solution.summary()
    dry = solution.dry_mole_fractions()

    wall_ok = (summary["max_wall_surface_temperature_K"]
               <= constraints.max_wall_temperature)
    drop_ok = (summary["pressure_drop_bar"] * 1.0e5
               <= constraints.max_pressure_drop)

    def round_list(values, places=2):
        return [round(float(v), places) for v in values]

    stations = []
    for fraction in RADIAL_STATION_FRACTIONS:
        k = int(round(fraction * (solution.n_axial - 1)))
        stations.append({
            "z": round(float(solution.z[k]), 3),
            "T": round_list(solution.temperature[k], 1),
            "T_wall": round(float(solution.wall_surface_temperature()[k]), 1),
        })

    return {
        "sc": round(steam_to_carbon, 4),
        "flux": round(heat_flux, 1),
        "profiles": {
            "T_bulk": round_list(solution.bulk_temperature(), 1),
            "T_centre": round_list(solution.temperature[:, 0], 1),
            "T_wall": round_list(solution.wall_surface_temperature(), 1),
            "P": round_list(solution.pressure / 1.0e5, 4),
            "X": round_list(solution.methane_conversion(), 5),
            "y_H2": round_list(dry[:, IDX["H2"]], 5),
            "y_CH4": round_list(dry[:, IDX["CH4"]], 5),
            "y_CO": round_list(dry[:, IDX["CO"]], 5),
            "y_CO2": round_list(dry[:, IDX["CO2"]], 5),
            "dT_radial": round_list(
                solution.radial_temperature_difference(), 2),
            "a_C": round_list(report.worst_activity.max(axis=1), 4),
        },
        "radial": stations,
        "summary": {
            "X": round(summary["methane_conversion"], 5),
            "H2": round(summary["hydrogen_production_mol_per_s"], 4),
            "yield": round(summary["hydrogen_yield_mol_per_mol_CH4"], 4),
            "slip": round(summary["methane_slip_dry"], 5),
            "T_out": round(summary["outlet_bulk_temperature_K"], 1),
            "T_wall_max": round(summary["max_wall_surface_temperature_K"], 1),
            "T_outer_max": round(summary["max_outer_wall_temperature_K"], 1),
            "dT_radial_max": round(summary["max_radial_dT_K"], 1),
            "dP": round(summary["pressure_drop_bar"], 4),
            "duty": round(summary["heat_duty_kW"], 1),
            "a_C_max": round(float(np.max(report.activity_actual)), 4),
            "coking_zone_end": round(float(_coking_zone_end(report)), 3),
            "eta1": round(summary["mean_effectiveness_R1"], 5),
            "wall_ok": bool(wall_ok),
            "drop_ok": bool(drop_ok),
            "feasible": bool(wall_ok and drop_ok),
        },
    }


def _coking_zone_end(report) -> float:
    """Axial position [m] where the last flagged node occurs, else 0."""
    flagged = report.node_flagged.any(axis=1)
    return float(report.z[flagged][-1]) if flagged.any() else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc-points", type=int, default=11)
    parser.add_argument("--flux-points", type=int, default=11)
    parser.add_argument("--axial-points", type=int, default=69)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    constraints = Constraints()
    sc_values = np.linspace(2.0, 4.5, args.sc_points)
    flux_values = np.linspace(4.0e4, 9.0e4, args.flux_points)

    reference = Reformer2D(
        fast_numerics(default_config(), n_radial=10,
                      n_axial_output=args.axial_points)).solve()

    cases = []
    start = time.perf_counter()
    total = sc_values.size * flux_values.size
    for i, sc in enumerate(sc_values):
        for j, flux in enumerate(flux_values):
            cases.append(build_case(float(sc), float(flux),
                                    args.axial_points, constraints))
            done = i * flux_values.size + j + 1
            print(f"  {done:4d}/{total}  S/C {sc:.2f}  "
                  f"q {flux / 1e3:.1f} kW/m2", flush=True)

    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d"),
            "model": "2D pseudo-heterogeneous SMR tube",
            "tube_length_m": default_config().geometry.length,
            "tube_id_mm": default_config().geometry.inner_diameter * 1e3,
            "inlet_T_K": default_config().feed.inlet_temperature,
            "inlet_P_bar": default_config().feed.inlet_pressure / 1e5,
            "wall_limit_K": constraints.max_wall_temperature,
            "pressure_drop_limit_bar": constraints.max_pressure_drop / 1e5,
            "n_cases": len(cases),
            "species": list(SPECIES),
        },
        "z": [round(float(v), 4) for v in reference.z],
        "r_mm": [round(float(v) * 1e3, 2) for v in reference.grid.centers],
        "radial_station_fractions": list(RADIAL_STATION_FRACTIONS),
        "sc_values": [round(float(v), 4) for v in sc_values],
        "flux_values": [round(float(v), 1) for v in flux_values],
        "cases": cases,
    }

    path = Path(args.outdir) / "demo_data.json"
    path.write_text(json.dumps(payload, separators=(",", ":")),
                    encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    print(f"\n  {len(cases)} cases in {time.perf_counter() - start:.1f} s")
    print(f"  -> {path}  ({size_kb:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
