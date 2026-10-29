r"""
Thermodynamic carbon-deposition (coking) analysis for the reformer tube.

Three carbon-forming routes are evaluated at every node of the solved tube:

=== ================================== ===================================
 id  reaction                            favoured by
=== ================================== ===================================
 C1  CH4  <-> C(s) + 2 H2               high temperature, low S/C
 C2  2 CO <-> C(s) + CO2  (Boudouard)   low temperature, high CO
 C3  CO + H2 <-> C(s) + H2O             low temperature, low steam
=== ================================== ===================================

For each route the **carbon activity** is formed from the local partial
pressures and the equilibrium constant of that route,

    a_C,1 = K1 p_CH4 / p_H2^2          (K1 in bar)
    a_C,2 = K2 p_CO^2 / p_CO2          (K2 in bar^-1)
    a_C,3 = K3 p_CO p_H2 / p_H2O       (K3 in bar^-1)

with ``K = exp(-dG(T)/RT)`` computed from the NASA-7 polynomials for the gases
and Kelley's graphite data (see :mod:`smr.species`).  Carbon deposition is
thermodynamically possible wherever ``a_C > 1``; the corresponding affinity

    dG_dep = -R T ln(a_C)

is the driving force for deposition, so a node with ``a_C = 2`` at 900 K has
about 5 kJ/mol of driving force.

Two gas bases are reported, following Rostrup-Nielsen's distinction:

*actual gas*
    the composition the 2D simulation actually predicts at that node.  This is
    the relevant basis for the entrance region, where the gas is far from
    equilibrium and methane cracking can run ahead of reforming.

*equilibrated gas*
    the composition that the reforming and shift equilibria would give at the
    same T, P and elemental composition.  This is the basis for the classical
    "principle of equilibrated gas" carbon limit and for the critical S/C
    curves produced by :func:`critical_steam_to_carbon`.

Caveat: these are *thermodynamic* boundaries for graphitic carbon.  Whisker
carbon on nickel nucleates at activities somewhat above one (the carbide-like
surface carbon has a different reference state), while gum formation on cooler
surfaces is kinetically controlled.  A node flagged here is therefore a node to
investigate, not a guarantee of deposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .kinetics import STOICHIOMETRY, equilibrium_constants
from .reformer import PA_PER_BAR, ReformerSolution
from .species import (
    IDX,
    N_SPECIES,
    R_GAS,
    SPECIES,
    gibbs_graphite,
    gibbs_molar_standard,
)

__all__ = [
    "CARBON_REACTIONS",
    "ATOM_MATRIX",
    "carbon_equilibrium_constants",
    "carbon_activities",
    "equilibrium_composition",
    "equilibrium_composition_gibbs",
    "critical_steam_to_carbon",
    "CokingReport",
    "analyze_coking",
    "coking_boundary_map",
]

#: Human-readable names of the carbon-forming routes.
CARBON_REACTIONS = (
    "C1: CH4 <-> C(s) + 2H2 (methane cracking)",
    "C2: 2CO <-> C(s) + CO2 (Boudouard)",
    "C3: CO + H2 <-> C(s) + H2O (CO reduction)",
)

# Gas-phase stoichiometry of the three carbon routes, in SPECIES order.
# Solid carbon is handled separately (activity = 1 for the reference state).
_CARBON_STOICH = np.zeros((3, N_SPECIES))
_CARBON_STOICH[0, [IDX["CH4"], IDX["H2"]]] = [-1.0, +2.0]
_CARBON_STOICH[1, [IDX["CO"], IDX["CO2"]]] = [-2.0, +1.0]
_CARBON_STOICH[2, [IDX["CO"], IDX["H2"], IDX["H2O"]]] = [-1.0, -1.0, +1.0]

#: Moles of solid carbon formed per unit extent of each route.
_CARBON_COEFF = np.array([1.0, 1.0, 1.0])


def carbon_equilibrium_constants(T) -> np.ndarray:
    """Equilibrium constants of the three carbon routes, shape ``(..., 3)``.

    Units: ``K1`` in bar, ``K2`` and ``K3`` in bar^-1, consistent with the
    1 bar standard state used by :mod:`smr.species`.
    """
    T = np.asarray(T, dtype=float)
    g_gas = gibbs_molar_standard(T)                      # (..., nsp)
    g_c = gibbs_graphite(T)                              # (...)
    dg = (np.einsum("ji,...i->...j", _CARBON_STOICH, g_gas)
          + _CARBON_COEFF * g_c[..., None])
    return np.exp(-dg / (R_GAS * T[..., None]))


def carbon_activities(T, p_bar) -> np.ndarray:
    """Carbon activity of each route, shape ``(..., 3)``.

    Parameters
    ----------
    T : array_like
        Temperature [K].
    p_bar : array_like
        Partial pressures [bar], shape ``(..., nsp)``.
    """
    T = np.asarray(T, dtype=float)
    p = np.clip(np.asarray(p_bar, dtype=float), 1.0e-12, None)
    K = carbon_equilibrium_constants(T)
    a1 = K[..., 0] * p[..., IDX["CH4"]] / p[..., IDX["H2"]] ** 2
    a2 = K[..., 1] * p[..., IDX["CO"]] ** 2 / p[..., IDX["CO2"]]
    a3 = K[..., 2] * p[..., IDX["CO"]] * p[..., IDX["H2"]] / p[..., IDX["H2O"]]
    return np.stack([a1, a2, a3], axis=-1)


def deposition_affinity(T, p_bar) -> np.ndarray:
    """Driving force ``-R T ln a_C`` [J/mol] of each route, shape ``(..., 3)``.

    Negative values mean deposition is thermodynamically favourable.
    """
    T = np.asarray(T, dtype=float)
    a = np.clip(carbon_activities(T, p_bar), 1.0e-300, None)
    return -R_GAS * T[..., None] * np.log(a)


# ---------------------------------------------------------------------------
# Equilibrated-gas composition
# ---------------------------------------------------------------------------

def equilibrium_composition(T: float, P: float, n_feed: Sequence[float],
                            x0: Optional[Sequence[float]] = None) -> np.ndarray:
    """Equilibrium mole fractions of the reforming/shift system.

    Solves the two independent equilibria R1 (reforming) and R2 (shift)
    simultaneously for their extents, which fixes the composition of the
    C-H-O-N system at the given ``T`` and ``P``.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Total pressure [Pa].
    n_feed : sequence of float, shape (nsp,)
        Feed amounts (any consistent unit) in :data:`smr.species.SPECIES` order.
    x0 : sequence of float, optional
        Initial guess for the two extents.

    Returns
    -------
    ndarray, shape (nsp,)
        Equilibrium mole fractions.
    """
    n0 = np.asarray(n_feed, dtype=float)
    K = equilibrium_constants(T)          # (3,), K1 [bar^2], K2 [-]
    p_total_bar = P / PA_PER_BAR
    floor = 1.0e-14 * max(n0.sum(), 1.0)

    nu1 = STOICHIOMETRY[0]
    nu2 = STOICHIOMETRY[1]

    def moles(extents: np.ndarray) -> np.ndarray:
        return n0 + extents[0] * nu1 + extents[1] * nu2

    def residual(extents: np.ndarray) -> np.ndarray:
        n = np.clip(moles(extents), floor, None)
        y = n / n.sum()
        p = np.clip(y * p_total_bar, 1.0e-14, None)
        q1 = (p[IDX["H2"]] ** 3 * p[IDX["CO"]]
              / (p[IDX["CH4"]] * p[IDX["H2O"]]))
        q2 = (p[IDX["H2"]] * p[IDX["CO2"]]
              / (p[IDX["CO"]] * p[IDX["H2O"]]))
        return np.array([np.log(q1) - np.log(K[0]),
                         np.log(q2) - np.log(K[1])])

    # Extent bounds: R1 cannot consume more CH4 or steam than is present; R2 is
    # bounded by the CO available after R1 and by the steam.  Generous but
    # finite bounds keep ``least_squares`` inside the physical region.
    e1_max = min(n0[IDX["CH4"]], n0[IDX["H2O"]]) * (1.0 - 1.0e-9)
    lower = np.array([-n0[IDX["CO"]] - e1_max, -n0[IDX["CO2"]]])
    upper = np.array([e1_max, n0[IDX["CO"]] + e1_max + n0[IDX["H2O"]]])
    guess = (np.array([0.5 * e1_max, 0.25 * e1_max]) if x0 is None
             else np.asarray(x0, dtype=float))
    guess = np.clip(guess, lower + 1.0e-12, upper - 1.0e-12)

    sol = least_squares(residual, guess, bounds=(lower, upper),
                        xtol=1.0e-14, ftol=1.0e-14, gtol=None)
    n = np.clip(moles(sol.x), floor, None)
    return n / n.sum()


#: Atoms per molecule of each species, rows C, H, O, N, in SPECIES order.
ATOM_MATRIX = np.array([
    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # C
    [4.0, 2.0, 2.0, 0.0, 0.0, 0.0],      # H
    [0.0, 1.0, 0.0, 1.0, 2.0, 0.0],      # O
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],      # N
])


def equilibrium_composition_gibbs(T: float, P: float, n_feed: Sequence[float],
                                  tolerance: float = 1.0e-12) -> np.ndarray:
    r"""Equilibrium mole fractions by Gibbs minimisation, element-potential form.

    Minimising the mixture Gibbs energy subject to conservation of carbon,
    hydrogen, oxygen and nitrogen gives the stationarity condition

    .. math::
        rac{g_i^\circ(T)}{RT} + \ln y_i + \ln rac{P}{P^\circ}
            = \sum_k \lambda_k a_{ki}

    where :math:`a_{ki}` is the number of atoms of element *k* in species *i*
    and :math:`\lambda_k` are the element potentials.  Solving for the amounts,

    .. math::
        n_i = N \exp\!\left( \sum_k \lambda_k a_{ki}
              - rac{g_i^\circ}{RT} - \ln rac{P}{P^\circ} 
ight)

    leaves a small square system in the element potentials and the total moles
    *N*: one equation per element from the atom balance, plus the closure
    :math:`\sum_i n_i = N`.  This is far more robust than minimising the Gibbs
    energy directly under linear constraints, because every iterate satisfies
    non-negativity identically and the unknowns number five rather than six.

    This is an *independent* route to the same answer as
    :func:`equilibrium_composition`, which instead solves the two reaction
    equilibria using the Twigg correlations.  The two share no equations beyond
    the NASA-7 polynomials, so agreement between them checks the
    reaction-extent solver, the Twigg correlations and the stoichiometry all at
    once.  It is used by the test suite and by the verification command rather
    than inside the reactor solver.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Total pressure [Pa].
    n_feed : sequence of float, shape (nsp,)
        Feed amounts in :data:`smr.species.SPECIES` order.

    Returns
    -------
    ndarray, shape (nsp,)
        Equilibrium mole fractions.
    """
    n0 = np.asarray(n_feed, dtype=float)
    total = max(float(n0.sum()), 1.0e-30)
    b_all = ATOM_MATRIX @ n0

    # An element absent from the feed forces every species containing it to
    # zero, and its potential to minus infinity.  Drop both rather than let the
    # solver chase the singularity -- this is what makes the routine work for a
    # nitrogen-free feed.
    present = b_all > 1.0e-14 * total
    atoms = ATOM_MATRIX[present]
    alive = np.flatnonzero(~(ATOM_MATRIX[~present] > 0).any(axis=0))
    atoms = atoms[:, alive]
    b = b_all[present] / total                       # atoms per mole of feed

    g_rt = gibbs_molar_standard(float(T))[alive] / (R_GAS * float(T))
    ln_p = float(np.log(P / 1.0e5))                  # 1 bar standard state

    def fractions(lam: np.ndarray) -> np.ndarray:
        """Mole fractions implied by a set of element potentials."""
        return np.exp(np.clip(atoms.T @ lam - g_rt - ln_p, -600.0, 600.0))

    def residual(x: np.ndarray) -> np.ndarray:
        """Closure on the mole fractions, then the atom balances.

        The unknowns are the element potentials and the log of the total moles
        per mole of feed.  Writing the closure on the *fractions* rather than on
        the amounts is essential: the amounts are proportional to the total, so
        a closure written on them cancels the total out and leaves the system
        singular in that unknown.
        """
        y = fractions(x[:-1])
        moles = np.exp(np.clip(x[-1], -600.0, 600.0))
        return np.concatenate([[y.sum() - 1.0], moles * (atoms @ y) - b])

    # Initialise the potentials by linear least squares against the feed.
    # Taking logs makes the fraction expression linear in the potentials,
    #     ln y_i = sum_k lambda_k a_ki - g_i/RT - ln(P/P0),
    # so fitting it to the (floored) feed composition gives a start of the
    # right order of magnitude for every species.
    y_feed = np.clip(n0[alive] / total, 1.0e-8, None)
    lam0 = np.linalg.lstsq(atoms.T, np.log(y_feed) + g_rt + ln_p,
                           rcond=None)[0]
    guess = np.concatenate([lam0, [0.0]])

    tol = max(float(tolerance), 1.0e-14)
    solution = least_squares(residual, guess, xtol=tol, ftol=tol, gtol=tol,
                             method="lm", max_nfev=20000)

    y = np.zeros(N_SPECIES)
    y[alive] = fractions(solution.x[:-1])
    return y / max(y.sum(), 1.0e-30)


def critical_steam_to_carbon(T: float, P: float,
                             routes: Sequence[int] = (0, 1, 2),
                             hydrogen_fraction: float = 0.0,
                             sc_bounds: tuple[float, float] = (0.3, 8.0),
                             tolerance: float = 1.0e-3,
                             threshold: float = 1.0) -> float:
    """Minimum S/C ratio for a carbon-free *equilibrated* gas.

    Bisects on the steam-to-carbon ratio until the largest carbon activity of
    the selected routes equals ``threshold``.  Returns ``sc_bounds[0]`` if the
    gas is already carbon-free at the lower bound, and ``nan`` if no S/C inside
    the bounds is sufficient.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    routes : sequence of int
        Which carbon routes to include (0 = cracking, 1 = Boudouard,
        2 = CO reduction).
    hydrogen_fraction : float
        Mole fraction of hydrogen co-fed with the methane, as in a real
        recycle-bearing feed.
    threshold : float
        Carbon activity that defines the boundary.  ``1.0`` is the graphite
        reference state, which is the thermodynamically exact criterion for
        bulk graphite.  Whisker carbon growing on a nickel crystallite is less
        stable than graphite, and the usual experimental correlation puts its
        onset near ``a_C`` of 1.5-3 on this basis; raising ``threshold``
        reproduces the shallower critical-S/C curves reported for nickel
        catalysts.  Gum formation on cooler surfaces is kinetic and is not
        described by any activity criterion.
    """
    routes = tuple(routes)

    def max_activity(sc: float) -> float:
        n = np.zeros(N_SPECIES)
        n[IDX["CH4"]] = 1.0
        n[IDX["H2O"]] = sc
        if hydrogen_fraction > 0.0:
            # hydrogen_fraction refers to the total feed basis
            total = (1.0 + sc) / (1.0 - hydrogen_fraction)
            n[IDX["H2"]] = hydrogen_fraction * total
        y = equilibrium_composition(T, P, n)
        a = carbon_activities(T, y * P / PA_PER_BAR)
        return float(np.max(a[list(routes)]))

    low, high = sc_bounds
    if max_activity(low) <= threshold:
        return low
    if max_activity(high) > threshold:
        return float("nan")
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if max_activity(mid) > threshold:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CokingReport:
    """Node-by-node carbon-deposition assessment of a solved tube."""

    z: np.ndarray                     # (nz,)        [m]
    r: np.ndarray                     # (nr,)        [m]
    temperature: np.ndarray           # (nz, nr)     gas T [K]
    solid_temperature: np.ndarray     # (nz, nr)     catalyst T [K]
    activity_actual: np.ndarray       # (nz, nr, 3)  a_C, actual gas
    activity_equilibrated: np.ndarray  # (nz, nr, 3) a_C, equilibrated gas
    affinity: np.ndarray              # (nz, nr, 3)  [J/mol]
    critical_sc: np.ndarray           # (nz,)        min S/C, equilibrated gas
    local_sc: np.ndarray              # (nz,)        local steam/carbon ratio
    threshold: float = 1.0
    marginal_threshold: float = 0.5

    # -- flags -------------------------------------------------------------
    @property
    def flagged(self) -> np.ndarray:
        """Boolean ``(nz, nr, 3)``: route thermodynamically favourable."""
        return self.activity_actual > self.threshold

    @property
    def node_flagged(self) -> np.ndarray:
        """Boolean ``(nz, nr)``: any route favourable at that node."""
        return self.flagged.any(axis=2)

    @property
    def worst_activity(self) -> np.ndarray:
        """Largest carbon activity at each node, shape ``(nz, nr)``."""
        return self.activity_actual.max(axis=2)

    @property
    def controlling_route(self) -> np.ndarray:
        """Index of the route with the largest activity, shape ``(nz, nr)``."""
        return self.activity_actual.argmax(axis=2)

    def axial_risk(self) -> np.ndarray:
        """Largest carbon activity over the radius at each ``z``, shape (nz,)."""
        return self.worst_activity.max(axis=1)

    def risk_zones(self) -> list[tuple[float, float, str]]:
        """Contiguous axial intervals where any node is flagged.

        Returns a list of ``(z_start, z_end, dominant_route_name)``.
        """
        flagged = self.node_flagged.any(axis=1)
        zones: list[tuple[float, float, str]] = []
        start: Optional[int] = None
        for k, f in enumerate(flagged):
            if f and start is None:
                start = k
            elif not f and start is not None:
                zones.append(self._zone(start, k - 1))
                start = None
        if start is not None:
            zones.append(self._zone(start, len(flagged) - 1))
        return zones

    def _zone(self, k0: int, k1: int) -> tuple[float, float, str]:
        block = self.activity_actual[k0:k1 + 1]
        route = int(np.unravel_index(np.argmax(block), block.shape)[2])
        return float(self.z[k0]), float(self.z[k1]), CARBON_REACTIONS[route]

    def margin(self) -> float:
        """Smallest safety margin ``1 - a_C`` over the whole tube.

        Positive means every node is carbon-free on the actual-gas basis.
        """
        return float(1.0 - np.max(self.worst_activity))

    def summary(self) -> dict:
        worst_idx = np.unravel_index(np.argmax(self.activity_actual),
                                     self.activity_actual.shape)
        return {
            "any_coking_risk": bool(self.flagged.any()),
            "max_carbon_activity": float(np.max(self.activity_actual)),
            "max_activity_route": CARBON_REACTIONS[int(worst_idx[2])],
            "max_activity_position_m": float(self.z[worst_idx[0]]),
            "max_activity_radius_m": float(self.r[worst_idx[1]]),
            "max_activity_temperature_K": float(
                self.temperature[worst_idx[0], worst_idx[1]]),
            "margin": self.margin(),
            "fraction_of_nodes_flagged": float(self.node_flagged.mean()),
            "n_risk_zones": len(self.risk_zones()),
            "max_critical_sc_equilibrated": (
                float(np.nanmax(self.critical_sc))
                if np.isfinite(self.critical_sc).any() else float("nan")),
            "min_local_sc": float(np.min(self.local_sc)),
            "peak_cracking_activity": float(np.max(self.activity_actual[:, :, 0])),
            "peak_boudouard_activity": float(np.max(self.activity_actual[:, :, 1])),
            "peak_co_reduction_activity": float(
                np.max(self.activity_actual[:, :, 2])),
        }

    def report(self) -> str:
        """Formatted text report."""
        s = self.summary()
        critical = (f"{s['max_critical_sc_equilibrated']:.2f}"
                    if np.isfinite(s["max_critical_sc_equilibrated"])
                    else "not evaluated")
        lines = [
            "Coking (carbon deposition) analysis",
            "-" * 64,
            f"  Thermodynamic risk anywhere     : "
            f"{'YES' if s['any_coking_risk'] else 'no'}",
            f"  Peak carbon activity (actual)   : "
            f"{s['max_carbon_activity']:.4g}"
            f"  at z = {s['max_activity_position_m']:.2f} m,"
            f" r = {s['max_activity_radius_m'] * 1e3:.1f} mm,"
            f" T = {s['max_activity_temperature_K']:.0f} K",
            f"  Controlling route               : {s['max_activity_route']}",
            f"  Safety margin (1 - a_C,max)     : {s['margin']:+.4f}",
            f"  Nodes flagged                   : "
            f"{s['fraction_of_nodes_flagged'] * 100:.2f} %",
            "",
            "  Peak activity by route",
            f"    C1 methane cracking           : "
            f"{s['peak_cracking_activity']:.4g}",
            f"    C2 Boudouard                  : "
            f"{s['peak_boudouard_activity']:.4g}",
            f"    C3 CO reduction               : "
            f"{s['peak_co_reduction_activity']:.4g}",
            "",
            f"  Local S/C minimum along tube    : {s['min_local_sc']:.2f}",
            f"  Critical S/C (equilibrated gas) : "
            f"{critical} (worst axial station)",
        ]
        zones = self.risk_zones()
        if zones:
            lines.append("")
            lines.append("  Flagged axial zones")
            for z0, z1, route in zones:
                lines.append(f"    {z0:6.2f} - {z1:6.2f} m : {route}")
        return "\n".join(lines)


def analyze_coking(solution: ReformerSolution,
                   include_equilibrated: bool = True,
                   equilibrated_stride: int = 8) -> CokingReport:
    """Build a :class:`CokingReport` from a solved tube.

    Parameters
    ----------
    solution : ReformerSolution
        Output of :meth:`smr.reformer.Reformer2D.solve`.
    include_equilibrated : bool
        Also compute the equilibrated-gas carbon activities and the critical
        S/C profile.  This costs one equilibrium solve per sampled station and
        can be switched off inside optimisation loops.
    equilibrated_stride : int
        Compute the equilibrated-gas quantities every ``stride`` axial points
        and interpolate in between.
    """
    p_bar = solution.partial_pressures("bar")              # (nz, nr, nsp)
    t_gas = solution.temperature
    t_solid = solution.diagnostics.get("solid_temperature", t_gas)

    # Carbon deposits on the catalyst surface, so the solid temperature is the
    # correct one for the activities; the difference is a few kelvin.
    a_actual = carbon_activities(t_solid, p_bar)
    affinity = deposition_affinity(t_solid, p_bar)

    nz, nr = t_gas.shape
    a_equil = np.full((nz, nr, 3), np.nan)
    critical = np.full(nz, np.nan)

    # Local steam-to-carbon ratio on a total-carbon basis.
    y_bulk = solution.bulk_mole_fractions()
    carbon = (y_bulk[:, IDX["CH4"]] + y_bulk[:, IDX["CO"]]
              + y_bulk[:, IDX["CO2"]])
    local_sc = y_bulk[:, IDX["H2O"]] / np.clip(carbon, 1.0e-30, None)

    if include_equilibrated:
        stations = list(range(0, nz, max(equilibrated_stride, 1)))
        if stations[-1] != nz - 1:
            stations.append(nz - 1)
        flows = solution.species_molar_flow()
        for k in stations:
            t_cup = float(solution.bulk_temperature()[k])
            p_k = float(solution.pressure[k])
            y_eq = equilibrium_composition(t_cup, p_k, flows[k])
            a_eq = carbon_activities(t_cup, y_eq * p_k / PA_PER_BAR)
            a_equil[k, :, :] = a_eq
            n_norm = flows[k] / max(flows[k, IDX["CH4"]]
                                    + flows[k, IDX["CO"]]
                                    + flows[k, IDX["CO2"]], 1.0e-30)
            critical[k] = critical_steam_to_carbon(
                t_cup, p_k, hydrogen_fraction=float(
                    np.clip(n_norm[IDX["H2"]] / max(n_norm.sum(), 1e-30),
                            0.0, 0.5)))
        # Interpolate over the skipped stations.
        valid = np.isfinite(critical)
        critical = np.interp(solution.z, solution.z[valid], critical[valid])
        for j in range(3):
            col = a_equil[:, 0, j]
            good = np.isfinite(col)
            interp = np.interp(solution.z, solution.z[good], col[good])
            a_equil[:, :, j] = interp[:, None]

    return CokingReport(
        z=solution.z, r=solution.grid.centers,
        temperature=t_gas, solid_temperature=t_solid,
        activity_actual=a_actual, activity_equilibrated=a_equil,
        affinity=affinity, critical_sc=critical, local_sc=local_sc,
    )


def coking_boundary_map(temperatures: np.ndarray, pressure: float,
                        hydrogen_fraction: float = 0.0,
                        routes: Sequence[int] = (0, 1, 2),
                        threshold: float = 1.0) -> dict:
    """Critical S/C versus temperature, the classic carbon-limit diagram.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures [K] at which to evaluate the boundary.
    pressure : float
        Pressure [Pa].
    hydrogen_fraction : float
        Hydrogen mole fraction in the feed.
    routes : sequence of int
        Carbon routes to include.

    Returns
    -------
    dict with keys ``temperature``, ``critical_sc`` and ``critical_sc_by_route``.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    overall = np.empty(temperatures.size)
    by_route = np.empty((temperatures.size, len(routes)))
    for i, T in enumerate(temperatures):
        overall[i] = critical_steam_to_carbon(
            float(T), pressure, routes=routes,
            hydrogen_fraction=hydrogen_fraction, threshold=threshold)
        for j, route in enumerate(routes):
            by_route[i, j] = critical_steam_to_carbon(
                float(T), pressure, routes=(route,),
                hydrogen_fraction=hydrogen_fraction, threshold=threshold)
    return {
        "temperature": temperatures,
        "critical_sc": overall,
        "critical_sc_by_route": by_route,
        "routes": tuple(routes),
        "threshold": threshold,
    }
