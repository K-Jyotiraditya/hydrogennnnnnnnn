"""
Verification tests for the 2D SMR tube model.

These are physics checks, not smoke tests: each one compares the code against
an independently known result -- a published rate constant, a thermodynamic
identity, an analytical asymptote, or a conservation law that the
discretisation must satisfy exactly.

Run with ``pytest tests`` or directly with ``python tests/test_smr.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smr import Reformer2D, analyze_coking, default_config          # noqa: E402
from smr import kinetics as kin                                      # noqa: E402
from smr import species as sp                                        # noqa: E402
from smr import transport as tr                                      # noqa: E402
from smr.assessment import (closure_sensitivity, closure_variants,   # noqa: E402
                            dimensionality_comparison)
from smr.coking import (ATOM_MATRIX,                                 # noqa: E402
                        carbon_activities,
                        carbon_equilibrium_constants,
                        critical_steam_to_carbon,
                        equilibrium_composition,
                        equilibrium_composition_gibbs)
from smr.config import Catalyst, Geometry, Numerics                  # noqa: E402
from smr.optimization import (Constraints, DesignSpace,               # noqa: E402
                              ObjectiveWeights, ReformerEvaluator,
                              fast_numerics, operating_envelope,
                              optimize_operating_point)
from smr.pellet import sphere_effectiveness, thiele_modulus          # noqa: E402
from smr.reformer import PA_PER_BAR, RadialGrid                      # noqa: E402
from smr.species import IDX, N_SPECIES, SPECIES                      # noqa: E402


# ---------------------------------------------------------------------------
# Thermochemistry
# ---------------------------------------------------------------------------

def test_cp_is_the_temperature_derivative_of_enthalpy():
    """dh/dT must equal cp for every species (NASA-7 self-consistency)."""
    T = np.array([600.0, 750.0, 900.0, 1150.0, 1300.0])
    dT = 0.05
    numerical = (sp.enthalpy_molar(T + dT) - sp.enthalpy_molar(T - dT)) / (2 * dT)
    assert np.allclose(numerical, sp.cp_molar(T), rtol=2.0e-4)


def test_cp_over_T_is_the_derivative_of_entropy():
    """ds/dT must equal cp/T for every species."""
    T = np.array([700.0, 900.0, 1100.0])
    dT = 0.05
    numerical = ((sp.entropy_molar_standard(T + dT)
                  - sp.entropy_molar_standard(T - dT)) / (2 * dT))
    assert np.allclose(numerical, sp.cp_molar(T) / T[:, None], rtol=2.0e-4)


def test_standard_reaction_enthalpies_match_literature():
    """dH(298 K) of the three reactions, in kJ/mol: +206, -41, +165."""
    dh = kin.reaction_enthalpies(298.15) / 1.0e3
    assert abs(dh[0] - 206.0) < 1.0
    assert abs(dh[1] + 41.0) < 1.0
    assert abs(dh[2] - 165.0) < 1.0


def test_graphite_reference_state():
    """Graphite cp at 298 K is 8.5 J/(mol K) and h(298) = 0 by definition."""
    assert abs(sp.cp_graphite(298.15) - 8.53) < 0.2
    assert abs(sp.enthalpy_graphite(298.15)) < 1.0e-9
    assert abs(sp.entropy_graphite(298.15) - 5.74) < 1.0e-9


def test_mixture_transport_properties_are_physical():
    """Viscosity, conductivity and cp of a reformer gas at 900 K."""
    y = np.zeros(N_SPECIES)
    y[IDX["CH4"]], y[IDX["H2O"]], y[IDX["H2"]] = 0.20, 0.65, 0.15
    y /= y.sum()
    assert 2.0e-5 < sp.mixture_viscosity(900.0, y) < 4.5e-5     # Pa s
    assert 0.08 < sp.mixture_thermal_conductivity(900.0, y) < 0.40   # W/(m K)
    assert 2000.0 < sp.mixture_cp_mass(900.0, y) < 4500.0       # J/(kg K)


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

def test_rate_constants_match_xu_froment_table():
    """k_j at the reference temperature of 648 K, from Xu & Froment Table 3."""
    k = kin.rate_constants(648.0)
    assert abs(k[0] / 1.842e-4 - 1.0) < 0.05      # kmol bar^0.5/(kg h)
    assert abs(k[1] / 7.558 - 1.0) < 0.05         # kmol/(kg h bar)
    assert abs(k[2] / 2.193e-5 - 1.0) < 0.05      # kmol bar^0.5/(kg h)


def test_adsorption_constants_decrease_with_temperature():
    """CH4, CO and H2 adsorb less strongly as temperature rises."""
    K = kin.adsorption_constants(np.array([700.0, 1100.0]))
    for name in ("CH4", "CO", "H2"):
        assert K[name][1] < K[name][0]
    # Dissociative water adsorption has the opposite sign of dH.
    assert K["H2O"][1] > K["H2O"][0]


def test_twigg_and_nasa_equilibrium_constants_agree():
    """Two independent routes to K1 and K2 must agree over the working range.

    Inside 700-1150 K, which covers everything the reformer gas sees, the
    Twigg correlations and the NASA-7 polynomials agree to better than 4%.
    Above that the correlations drift -- they were fitted over a narrower
    range -- reaching about 8% on K2 at 1300 K, so the wider band is asserted
    separately rather than hidden behind one loose tolerance.
    """
    working = np.linspace(700.0, 1150.0, 10)
    a, b = (kin.equilibrium_constants(working),
            kin.equilibrium_constants_from_thermo(working))
    assert np.max(np.abs(a[:, 0] / b[:, 0] - 1.0)) < 0.04
    assert np.max(np.abs(a[:, 1] / b[:, 1] - 1.0)) < 0.04

    extended = np.linspace(1150.0, 1300.0, 7)
    a, b = (kin.equilibrium_constants(extended),
            kin.equilibrium_constants_from_thermo(extended))
    assert np.max(np.abs(a[:, 0] / b[:, 0] - 1.0)) < 0.10
    assert np.max(np.abs(a[:, 1] / b[:, 1] - 1.0)) < 0.10


def test_rates_vanish_at_equilibrium():
    """At the equilibrium composition the net rates must be zero.

    This couples the kinetics and the equilibrium solver: if the rate law and
    the equilibrium constants were inconsistent, the reactor would relax to the
    wrong state.
    """
    for T in (900.0, 1050.0, 1200.0):
        n = np.zeros(N_SPECIES)
        n[IDX["CH4"]], n[IDX["H2O"]] = 1.0, 3.0
        y = equilibrium_composition(T, 26.0 * PA_PER_BAR, n)
        r = kin.reaction_rates(T, y * 26.0)
        reference = np.max(np.abs(kin.reaction_rates(
            T, np.where(np.arange(N_SPECIES) == IDX["H2"], 0.5, y * 26.0))))
        assert np.max(np.abs(r)) < 1.0e-6 * max(reference, 1.0)


def test_two_independent_equilibrium_routes_agree():
    """Reaction extents with Twigg constants vs direct Gibbs minimisation.

    The two share nothing beyond the NASA-7 polynomials: one solves the two
    reaction equilibria using the Twigg correlations, the other minimises the
    mixture Gibbs energy under atom conservation.  Agreement checks the extent
    solver, the Twigg correlations and the stoichiometry simultaneously.
    """
    n = np.zeros(N_SPECIES)
    n[IDX["CH4"]], n[IDX["H2O"]], n[IDX["N2"]] = 1.0, 3.0, 0.02
    for T in (900.0, 1000.0, 1100.0, 1200.0):
        extents = equilibrium_composition(T, 26.0 * PA_PER_BAR, n)
        gibbs = equilibrium_composition_gibbs(T, 26.0 * PA_PER_BAR, n)
        assert np.max(np.abs(extents - gibbs)) < 5.0e-3, (
            f"routes disagree at {T} K: {extents} vs {gibbs}")


def test_gibbs_minimisation_conserves_atoms():
    """The constrained minimiser must respect the atom balance it is given."""
    n = np.zeros(N_SPECIES)
    n[IDX["CH4"]], n[IDX["H2O"]], n[IDX["CO2"]] = 1.0, 2.5, 0.1
    y = equilibrium_composition_gibbs(1050.0, 26.0 * PA_PER_BAR, n)
    # Mole fractions scale to moles by the total, which the C balance fixes.
    carbon_in = float(ATOM_MATRIX[0] @ n)
    total = carbon_in / float(ATOM_MATRIX[0] @ y)
    moles = y * total
    assert np.allclose(ATOM_MATRIX @ moles, ATOM_MATRIX @ n, rtol=1.0e-6)
    assert abs(y.sum() - 1.0) < 1.0e-12
    assert np.all(y >= 0.0)


def test_equilibrium_conversion_decreases_with_pressure():
    """Le Chatelier: reforming increases the mole count, so pressure hurts."""
    n = np.zeros(N_SPECIES)
    n[IDX["CH4"]], n[IDX["H2O"]] = 1.0, 3.0
    low = equilibrium_composition(1100.0, 5.0 * PA_PER_BAR, n)
    high = equilibrium_composition(1100.0, 30.0 * PA_PER_BAR, n)
    assert low[IDX["CH4"]] < high[IDX["CH4"]]


def test_stoichiometry_conserves_atoms():
    """Every reaction must conserve C, H and O."""
    atoms = np.array([[1, 0, 0, 1, 1, 0],      # C
                      [4, 2, 2, 0, 0, 0],      # H
                      [0, 1, 0, 1, 2, 0]],     # O
                     dtype=float)
    assert np.allclose(kin.STOICHIOMETRY @ atoms.T, 0.0)


# ---------------------------------------------------------------------------
# Pellet
# ---------------------------------------------------------------------------

def test_effectiveness_factor_asymptotes():
    """eta -> 1 for small Thiele modulus and eta -> 1/phi for large."""
    assert abs(sphere_effectiveness(1.0e-6) - 1.0) < 1.0e-9
    assert abs(sphere_effectiveness(0.1) - (1.0 - 0.6 * 0.01)) < 2.0e-3
    for phi in (20.0, 50.0, 100.0):
        assert abs(sphere_effectiveness(phi) * phi - 1.0) < 0.02


def test_effectiveness_factor_is_monotone_in_particle_size():
    """A bigger pellet is more strongly diffusion limited."""
    y = np.zeros(N_SPECIES)
    y[IDX["CH4"]], y[IDX["H2O"]], y[IDX["H2"]] = 0.15, 0.55, 0.20
    y[IDX["CO"]], y[IDX["CO2"]] = 0.05, 0.05
    rates = np.array([1.0, 2.0, 0.5])          # mol/(kg s), forward
    previous = 1.0
    for d_p in (0.005, 0.010, 0.015, 0.025):
        catalyst = Catalyst(particle_diameter=d_p)
        phi = thiele_modulus(rates, 1050.0, 26.0 * PA_PER_BAR, y, catalyst)
        eta = sphere_effectiveness(phi)[0]
        assert eta < previous
        previous = eta


def test_effectiveness_factor_is_in_the_industrial_range():
    """15 mm pellets at reformer conditions give eta of order 0.01-0.2."""
    y = np.zeros(N_SPECIES)
    y[IDX["CH4"]], y[IDX["H2O"]], y[IDX["H2"]] = 0.12, 0.50, 0.25
    y[IDX["CO"]], y[IDX["CO2"]] = 0.06, 0.07
    rates = np.array([2.0, 5.0, 1.0])
    phi = thiele_modulus(rates, 1100.0, 25.0 * PA_PER_BAR, y, Catalyst())
    eta = sphere_effectiveness(phi)
    assert 0.005 < eta[0] < 0.25


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def test_radial_grid_partitions_the_cross_section():
    for n in (2, 5, 12, 33):
        for grading in (1.0, 1.5):
            grid = RadialGrid.build(0.049, n, grading)
            assert abs(grid.area.sum() - np.pi * 0.049**2) < 1.0e-15
            assert grid.faces[0] == 0.0
            assert abs(grid.faces[-1] - 0.049) < 1.0e-15
            assert np.all(np.diff(grid.centers) > 0.0)
            assert np.all(grid.centers > grid.faces[:-1])
            assert np.all(grid.centers < grid.faces[1:])


def test_radial_operator_is_conservative():
    """The area-weighted sum of the radial energy divergence must equal the
    wall input exactly: that is the whole point of the finite-volume form."""
    config = default_config()
    model = Reformer2D(config)
    grid = model.grid
    state = model.initial_state()
    flux, temperature, pressure = model._unpack(state)
    # Impose an artificial radial temperature profile.
    temperature = 800.0 + 120.0 * (grid.centers / grid.radius) ** 2
    props = model._local_properties(flux, temperature, float(pressure))
    q_wall, _ = model._wall_flux(1.0, temperature, props)

    k_face = model._face_mean(props["k_er"])
    cond = grid.circumference[1:-1] * k_face / np.diff(grid.centers)
    g = cond * np.diff(temperature)
    divergence = np.zeros(grid.n)
    divergence[:-1] += g
    divergence[1:] -= g
    divergence[-1] += grid.circumference[-1] * float(q_wall)
    total = float(divergence.sum())             # [W/m] of tube
    expected = float(grid.circumference[-1] * q_wall)
    assert abs(total - expected) < 1.0e-9 * abs(expected)


def test_ergun_terms_scale_correctly():
    """Viscous term linear in u, inertial term quadratic."""
    catalyst = Catalyst()
    grad1 = abs(tr.ergun_pressure_gradient(1.0, 6.0, 3.0e-5, catalyst))
    grad2 = abs(tr.ergun_pressure_gradient(2.0, 6.0, 3.0e-5, catalyst))
    assert 2.0 < grad2 / grad1 < 2.0 * 2.0      # between linear and quadratic
    assert tr.ergun_pressure_gradient(1.0, 6.0, 3.0e-5, catalyst) < 0.0


def test_transport_closures_are_in_range():
    """k_er, h_w and h_f at industrial reformer conditions."""
    catalyst, geometry = Catalyst(), Geometry()
    numerics = Numerics()
    k_er = tr.effective_radial_conductivity(1050.0, 0.12, 5000.0, 0.7,
                                            catalyst, geometry, numerics)
    h_w = tr.wall_heat_transfer_coefficient(0.12, 5000.0, 0.7, catalyst,
                                            geometry, numerics)
    h_f = tr.film_heat_transfer_coefficient(0.12, 5000.0, 0.7, catalyst)
    assert 1.0 < k_er < 60.0           # W/(m K)
    assert 200.0 < h_w < 4000.0        # W/(m^2 K)
    assert 200.0 < h_f < 5000.0        # W/(m^2 K)
    assert 0.5 < tr.tube_wall_temperature_drop(5.8e4, geometry) < 100.0


# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------

def _baseline_solution(**numerics):
    return Reformer2D(default_config().with_numerics(**numerics)).solve()


def test_reactor_conserves_atoms():
    solution = _baseline_solution(rtol=1.0e-7)
    for element, error in solution.mass_balance_error().items():
        assert abs(error) < 1.0e-8, f"{element} balance drifted by {error:.2e}"


def test_enthalpy_balance_closes_without_the_omitted_dispersion_term():
    """With radial dispersion and the film off, the only error left is the
    integrator's, so the enthalpy balance must close to tolerance."""
    solution = _baseline_solution(rtol=1.0e-7, radial_mass_dispersion=False,
                                  film_resistance=False)
    assert abs(solution.energy_balance_error()["relative_error"]) < 1.0e-5


def test_enthalpy_balance_residual_is_small_for_the_full_model():
    solution = _baseline_solution(rtol=1.0e-7)
    assert abs(solution.energy_balance_error()["relative_error"]) < 5.0e-3


def test_radial_grid_independence():
    """Outlet metrics must not move materially between 8 and 24 cells."""
    coarse = _baseline_solution(n_radial=8, rtol=1.0e-7).summary()
    fine = _baseline_solution(n_radial=24, rtol=1.0e-7).summary()
    assert abs(coarse["methane_conversion"] - fine["methane_conversion"]) < 1.0e-3
    assert abs(coarse["outlet_bulk_temperature_K"]
               - fine["outlet_bulk_temperature_K"]) < 0.5
    assert abs(coarse["max_wall_surface_temperature_K"]
               - fine["max_wall_surface_temperature_K"]) < 2.0


def test_profiles_are_physically_ordered():
    solution = _baseline_solution()
    assert solution.success
    # Heat enters at the wall, so temperature increases outwards everywhere.
    assert np.all(np.diff(solution.temperature, axis=1) >= -1.0e-6)
    # The wall surface is hotter still, and the outer metal hotter again.
    assert np.all(solution.wall_surface_temperature()
                  >= solution.near_wall_temperature() - 1.0e-6)
    assert np.all(solution.outer_wall_temperature()
                  >= solution.wall_surface_temperature() - 1.0e-6)
    # Pressure falls monotonically; composition stays physical.
    assert np.all(np.diff(solution.pressure) < 0.0)
    y = solution.mole_fractions()
    assert np.all(y >= -1.0e-9) and np.all(y <= 1.0 + 1.0e-9)
    assert np.allclose(y.sum(axis=2), 1.0, atol=1.0e-9)
    # Methane is consumed monotonically and hydrogen produced monotonically.
    assert np.all(np.diff(solution.methane_conversion()) >= -1.0e-6)
    assert np.all(np.diff(solution.hydrogen_production()) >= -1.0e-6)


def test_outlet_is_near_but_short_of_equilibrium():
    """An industrial reformer leaves a real approach to equilibrium."""
    solution = _baseline_solution()
    n_out = solution.species_molar_flow()[-1]
    t_out = float(solution.bulk_temperature()[-1])
    p_out = float(solution.pressure[-1])
    y_eq = equilibrium_composition(t_out, p_out, n_out)
    y_out = solution.bulk_mole_fractions()[-1]
    # Methane slip must be at or above the equilibrium value at the same
    # temperature and pressure, and within a few mole per cent of it.
    assert y_out[IDX["CH4"]] >= y_eq[IDX["CH4"]] - 0.01
    assert y_out[IDX["CH4"]] - y_eq[IDX["CH4"]] < 0.05


def test_outlet_metrics_are_industrially_plausible():
    summary = _baseline_solution().summary()
    assert 0.60 < summary["methane_conversion"] < 0.90
    assert 1050.0 < summary["outlet_bulk_temperature_K"] < 1200.0
    assert 1.0 < summary["pressure_drop_bar"] < 5.0
    assert 0.60 < summary["outlet_H2_dry"] < 0.80
    assert 0.08 < summary["outlet_CO_dry"] < 0.20
    assert 20.0 < summary["max_radial_dT_K"] < 200.0


def test_wall_temperature_mode_runs_and_reports_flux():
    config = default_config().with_operating(mode="wall_temperature")
    solution = Reformer2D(config).solve()
    assert solution.success
    assert np.all(solution.diagnostics["q_wall"] > 0.0)
    assert solution.heat_duty() > 0.0
    assert 0.5 < solution.summary()["methane_conversion"] < 0.95


def test_higher_steam_to_carbon_raises_conversion():
    low = Reformer2D(default_config().with_feed(steam_to_carbon=2.5)).solve()
    high = Reformer2D(default_config().with_feed(steam_to_carbon=4.0)).solve()
    assert (high.summary()["methane_conversion"]
            > low.summary()["methane_conversion"])


def test_higher_flux_raises_conversion_and_wall_temperature():
    low = Reformer2D(default_config().with_operating(
        average_heat_flux=4.5e4)).solve().summary()
    high = Reformer2D(default_config().with_operating(
        average_heat_flux=7.5e4)).solve().summary()
    assert high["methane_conversion"] > low["methane_conversion"]
    assert (high["max_wall_surface_temperature_K"]
            > low["max_wall_surface_temperature_K"])


def test_effectiveness_model_switches_behave():
    intrinsic = Reformer2D(default_config().with_catalyst(
        effectiveness_model="none")).solve().summary()
    limited = Reformer2D(default_config()).solve().summary()
    # Removing the diffusion limitation cannot reduce conversion.
    assert intrinsic["methane_conversion"] >= limited["methane_conversion"] - 1e-3
    assert intrinsic["mean_effectiveness_R1"] > limited["mean_effectiveness_R1"]


# ---------------------------------------------------------------------------
# Coking
# ---------------------------------------------------------------------------

def test_boudouard_reverses_near_970_K():
    """2CO <-> C + CO2 has K = 1 at about 970 K, and flips sign across it."""
    T = np.linspace(850.0, 1100.0, 501)
    K = carbon_equilibrium_constants(T)[:, 1]
    crossing = float(T[np.argmin(np.abs(np.log(K)))])
    assert 930.0 < crossing < 1010.0
    assert K[0] > 1.0 and K[-1] < 1.0


def test_methane_cracking_becomes_favourable_above_about_820_K():
    """CH4 <-> C + 2H2 has dG = 0 near 820-870 K at unit activities."""
    T = np.linspace(700.0, 1000.0, 601)
    K = carbon_equilibrium_constants(T)[:, 0]
    crossing = float(T[np.argmin(np.abs(np.log(K)))])
    assert 790.0 < crossing < 900.0


def test_carbon_activity_is_unity_on_its_own_boundary():
    """Construct a gas sitting exactly on the cracking boundary."""
    T = 900.0
    K = carbon_equilibrium_constants(T)[0]       # bar
    p = np.zeros(N_SPECIES)
    p[IDX["H2"]] = 2.0
    p[IDX["CH4"]] = p[IDX["H2"]] ** 2 / K        # makes a_C exactly 1
    p[IDX["H2O"]] = 1.0
    assert abs(carbon_activities(T, p)[0] - 1.0) < 1.0e-10


def test_more_steam_always_suppresses_carbon():
    """Carbon activity of the equilibrated gas falls as S/C rises."""
    activities = []
    for sc in (1.0, 2.0, 3.0, 4.0):
        n = np.zeros(N_SPECIES)
        n[IDX["CH4"]], n[IDX["H2O"]] = 1.0, sc
        y = equilibrium_composition(850.0, 26.0 * PA_PER_BAR, n)
        a = carbon_activities(850.0, y * 26.0)
        activities.append(float(np.max(a)))
    assert all(b < a for a, b in zip(activities, activities[1:]))


def test_critical_steam_to_carbon_is_bracketed():
    for T in (800.0, 950.0, 1100.0):
        sc = critical_steam_to_carbon(T, 26.0 * PA_PER_BAR)
        assert 0.3 <= sc <= 8.0
        # Just above the critical ratio the gas must be carbon free.
        n = np.zeros(N_SPECIES)
        n[IDX["CH4"]], n[IDX["H2O"]] = 1.0, sc + 0.2
        y = equilibrium_composition(T, 26.0 * PA_PER_BAR, n)
        assert np.max(carbon_activities(T, y * 26.0)) <= 1.0 + 1.0e-6


def test_coking_report_flags_the_inlet_region():
    """The cool inlet, where CO first appears, is the classic risk zone."""
    solution = _baseline_solution()
    report = analyze_coking(solution, include_equilibrated=False)
    assert report.activity_actual.shape == (solution.n_axial, solution.grid.n, 3)
    assert np.all(report.activity_actual >= 0.0)
    assert report.worst_activity.argmax() // solution.grid.n < solution.n_axial // 4
    summary = report.summary()
    assert summary["max_carbon_activity"] == report.activity_actual.max()


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def test_evaluator_caches_and_respects_the_feed_convention():
    evaluator = ReformerEvaluator(fast_numerics(default_config()))
    first = evaluator.evaluate([3.0, 5.8e4])
    second = evaluator.evaluate([3.0, 5.8e4])
    assert first is second
    assert evaluator.n_evaluations == 1
    # Raising S/C must add steam, not cut methane.
    low = evaluator.configure([2.5, 5.8e4]).feed.molar_flows()[IDX["CH4"]]
    high = evaluator.configure([4.0, 5.8e4]).feed.molar_flows()[IDX["CH4"]]
    assert abs(low - high) < 1.0e-12


def test_constraint_sign_convention():
    evaluator = ReformerEvaluator(fast_numerics(default_config()),
                                  constraints=Constraints(
                                      max_wall_temperature=1150.0))
    metrics = evaluator.evaluate([3.0, 5.8e4]).metrics
    assert np.all(evaluator.constraint_values(metrics) >= 0.0)
    assert metrics["feasible"]
    hot = evaluator.evaluate([3.0, 9.5e4]).metrics
    assert hot["max_wall_surface_temperature_K"] > 1150.0
    assert not hot["feasible"]
    assert evaluator.violation_from_metrics(hot) > 0.0


def test_optimiser_returns_a_feasible_improvement():
    weights = ObjectiveWeights(steam=0.10)
    constraints = Constraints(max_wall_temperature=1150.0)
    result = optimize_operating_point(method="slsqp", weights=weights,
                                      constraints=constraints,
                                      max_iterations=40, verbose=False)
    base = ReformerEvaluator(fast_numerics(default_config()), weights,
                             constraints).evaluate([3.0, 5.8e4]).metrics
    assert result.metrics["feasible"]
    assert result.metrics["objective"] < base["objective"]
    assert (result.metrics["max_wall_surface_temperature_K"]
            <= constraints.max_wall_temperature + 1.0)
    assert 2.0 <= result.steam_to_carbon <= 4.5
    assert 3.5e4 <= result.heat_flux <= 9.5e4


def test_design_space_handles_a_pinned_dimension():
    """A pinned bound (lower == upper) must map to 0, not divide by zero.

    The steam Pareto front fixes S/C and optimises the flux alone, which is
    exactly this case; an unguarded division produced a NaN start point and a
    NaN feed composition.
    """
    space = DesignSpace(steam_to_carbon=(3.2, 3.2))
    unit = space.to_unit(np.array([3.2, 6.0e4]))
    assert np.all(np.isfinite(unit))
    assert unit[0] == 0.0
    assert np.allclose(space.from_unit(unit), [3.2, 6.0e4])


def test_operating_envelope_has_a_constraint_corner():
    """The feasible envelope must switch from wall-limited to pressure-limited,
    and hydrogen production must peak at that corner."""
    envelope = operating_envelope(
        steam_to_carbon=np.linspace(2.0, 4.5, 6),
        constraints=Constraints(max_wall_temperature=1150.0), verbose=False)
    binding = envelope["binding"]
    assert "wall temperature" in binding
    assert "pressure drop" in binding
    # Wall-limited stretch comes first, then the pressure-limited stretch.
    first_drop = binding.index("pressure drop")
    assert all(b == "wall temperature" for b in binding[:first_drop])
    # The admissible flux rises then falls, and H2 peaks at the turning point.
    hydrogen = envelope["hydrogen_production"]
    assert int(np.argmax(hydrogen)) == first_drop - 1
    assert np.isfinite(envelope["corner_steam_to_carbon"])


# ---------------------------------------------------------------------------
# Model assessment
# ---------------------------------------------------------------------------

def test_lumped_limit_removes_the_bed_gradient_but_not_the_film():
    """Driving k_er up must flatten the bed while leaving the wall film.

    This is what makes the lumped run a fair stand-in for a 1D model: a 1D
    model still needs a wall coefficient, so the film jump must survive.
    """
    comparison = dimensionality_comparison(verbose=False)
    two_d, lumped = comparison["two_dimensional"], comparison["lumped"]
    # The bed itself becomes isothermal.
    config = default_config().with_numerics(
        k_er_model="constant", k_er_constant=1.0e5)
    solution = Reformer2D(config).solve()
    bed_spread = float(np.max(solution.temperature[:, -1]
                              - solution.temperature[:, 0]))
    assert bed_spread < 0.5
    # But a substantial film contribution remains.
    assert lumped["max_radial_dT_K"] > 20.0
    assert lumped["max_radial_dT_K"] < two_d["max_radial_dT_K"]


def test_radial_resolution_barely_changes_the_duty():
    """The project's own premise, tested rather than assumed.

    For a heat-transfer-limited tube the integrated chemistry is set by the
    heat crossing the wall, so resolving the radial profile should move the
    duty predictions very little while moving the thermal diagnostics a lot.
    """
    delta = dimensionality_comparison(verbose=False)["delta"]
    assert abs(delta["methane_conversion"]) < 5.0e-3        # <0.5 points
    assert abs(delta["hydrogen_production_mol_per_s"]) < 0.02
    assert abs(delta["outlet_bulk_temperature_K"]) < 3.0
    assert abs(delta["pressure_drop_bar"]) < 0.05
    # The gradient itself, by contrast, is a 2D-only quantity.
    assert abs(delta["max_radial_dT_K"]) > 20.0


def test_radial_gradient_falls_monotonically_with_conductivity():
    sensitivity = closure_sensitivity(verbose=False)
    gradient = sensitivity["metrics"]["max_radial_dT_K"]
    assert np.all(np.diff(gradient) < 0.0)
    wall = sensitivity["metrics"]["max_wall_surface_temperature_K"]
    assert np.all(np.diff(wall) < 0.0)
    # Conversion must stay essentially put across the whole range.
    conversion = sensitivity["metrics"]["methane_conversion"]
    assert float(conversion.max() - conversion.min()) < 1.0e-3


def test_closure_variants_move_diagnostics_not_duty():
    """One-at-a-time closure swaps: conversion robust, thermals uncertain."""
    variants = closure_variants(verbose=False)
    spread = variants["largest_deviation"]
    assert spread["methane_conversion"] < 0.01              # <1 point
    assert spread["hydrogen_production_mol_per_s"] < 0.05
    assert spread["max_wall_surface_temperature_K"] > 5.0
    assert len(variants["rows"]) >= 8
    assert variants["rows"][0]["variant"] == "baseline"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:                 # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
