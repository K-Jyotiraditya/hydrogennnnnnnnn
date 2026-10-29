r"""
Operating-point optimisation for the reformer tube.

Decision variables
------------------
``x = [S/C, q_mean]``

* ``S/C``    -- inlet steam-to-carbon ratio [mol H2O / mol CH4]
* ``q_mean`` -- area-average inner-wall heat flux [W/m^2]; the axial *shape* of
  the flux profile is held fixed and only its level is tuned, which is how a
  reformer is actually controlled (firing rate at fixed burner arrangement)

Objective
---------
Scalarised, with all terms made dimensionless so the weights are meaningful:

    J = -(H2 produced / H2 stoichiometric maximum)
        + w_slip  * (outlet CH4 slip / slip reference)
        + w_steam * (S/C - S/C reference)
        + w_duty  * (wall heat duty / duty reference)

The first two terms are the two objectives named in the problem statement
(maximise hydrogen yield, minimise methane slip).  ``w_steam`` prices the steam
that has to be raised and later condensed; without it the optimiser simply
drives S/C to its upper bound, because excess steam always helps the chemistry.
``w_duty`` is available to price fuel gas and defaults to zero.

Constraints
-----------
* peak inner wall surface temperature <= metallurgical limit (default 1150 K)
* total pressure drop <= limit (default 4 bar)
* optionally, peak carbon activity <= 1 (thermodynamically carbon free)

Methods
-------
``"slsqp"``
    Gradient-based, constraints handled natively.  Fast; decision variables are
    internally scaled to the unit box so the finite-difference step is
    well-conditioned.
``"nelder-mead"``
    Derivative-free, constraints via a smooth exterior penalty.  Robust when
    the constraint boundary is active.
``"differential_evolution"``
    Global search with the same penalty.  Use when the feasible region may be
    disconnected (it can be, because a high flux can both raise conversion and
    violate the wall limit).

Pareto fronts
-------------
:func:`pareto_front` uses the epsilon-constraint method, which produces
genuinely non-dominated points rather than the gaps a weighted sum leaves on
non-convex fronts.  Two fronts are supported:

* ``"wall_temperature"`` -- hydrogen production against the tube-wall
  temperature limit, i.e. the production/tube-life trade-off;
* ``"steam"`` -- hydrogen production against steam consumption, i.e. the
  production/utility-cost trade-off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .coking import analyze_coking
from .config import ReformerConfig, default_config
from .reformer import PA_PER_BAR, Reformer2D, ReformerSolution
from .species import IDX

__all__ = [
    "ObjectiveWeights",
    "Constraints",
    "DesignSpace",
    "ReformerEvaluator",
    "OptimizationResult",
    "ParetoFront",
    "optimize_operating_point",
    "pareto_front",
    "sensitivity_scan",
    "operating_envelope",
    "fast_numerics",
]


def fast_numerics(config: ReformerConfig, n_radial: int = 8,
                  n_axial_output: int = 69,
                  rtol: float = 1.0e-6) -> ReformerConfig:
    """Coarsen the discretisation for use inside optimisation loops.

    The optimiser needs hundreds of solves; a 8-cell radial grid reproduces the
    outlet metrics of the 12-cell baseline to well under a per cent (see
    ``main.py grid-study``), which is far tighter than the uncertainty in the
    transport closures.
    """
    return config.with_numerics(n_radial=n_radial,
                                n_axial_output=n_axial_output,
                                rtol=rtol)


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveWeights:
    """Weights of the scalarised objective (all terms dimensionless)."""

    hydrogen: float = 1.0        # weight on normalised H2 production
    methane_slip: float = 1.0    # weight on normalised CH4 slip
    steam: float = 0.10          # price on steam above the reference S/C
    duty: float = 0.0            # price on firing duty
    slip_reference: float = 0.05        # [-] reference dry CH4 slip
    steam_reference: float = 2.5        # [-] reference S/C
    duty_reference: float = 2.5e5       # [W] reference tube duty


@dataclass
class Constraints:
    """Inequality constraints, written as ``g(x) >= 0`` internally."""

    max_wall_temperature: float = 1150.0   # [K] metallurgical limit
    max_pressure_drop: float = 4.0e5       # [Pa]
    carbon_free: bool = False              # require max carbon activity <= 1
    min_methane_conversion: float = 0.0    # [-] optional production floor
    #: Violations below this (in the scaled constraint units used internally,
    #: so 5e-3 is ~0.25 K on the wall limit or ~500 Pa on the pressure drop)
    #: count as feasible.  Without it a point the optimiser has driven exactly
    #: onto an active constraint -- and then re-solved on the finer grid -- is
    #: reported as infeasible on the last digit.
    feasibility_tolerance: float = 5.0e-3


@dataclass
class DesignSpace:
    """Bounds on the decision variables."""

    steam_to_carbon: tuple[float, float] = (2.0, 4.5)
    heat_flux: tuple[float, float] = (3.5e4, 9.5e4)   # [W/m^2]

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.steam_to_carbon[0], self.heat_flux[0]])

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.steam_to_carbon[1], self.heat_flux[1]])

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map physical variables to the unit box.

        A dimension may be pinned (``lower == upper``), which is how the steam
        Pareto front holds S/C fixed while optimising the flux alone.  Such a
        dimension maps to 0 rather than dividing by a zero width.
        """
        width = self.upper - self.lower
        return np.where(width > 0.0,
                        (np.asarray(x, dtype=float) - self.lower)
                        / np.where(width > 0.0, width, 1.0),
                        0.0)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map unit-box variables back to physical values."""
        return self.lower + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) \
            * (self.upper - self.lower)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    """One simulation of the tube at a given operating point."""

    x: np.ndarray
    metrics: dict
    solution: Optional[ReformerSolution] = None


class ReformerEvaluator:
    """Caching wrapper that turns an operating point into scalar metrics.

    Parameters
    ----------
    base_config : ReformerConfig
        Template configuration; the decision variables overwrite the feed S/C
        and the mean wall heat flux.
    weights, constraints : ObjectiveWeights, Constraints
        Objective and constraint definitions.
    coking : bool
        Evaluate the carbon activity (needed for the ``carbon_free``
        constraint, and reported either way when enabled).
    keep_solutions : bool
        Retain the full :class:`ReformerSolution` of every evaluation.  Off by
        default to keep memory flat during a global search.
    """

    def __init__(self, base_config: Optional[ReformerConfig] = None,
                 weights: Optional[ObjectiveWeights] = None,
                 constraints: Optional[Constraints] = None,
                 coking: bool = False,
                 keep_solutions: bool = False):
        cfg = default_config() if base_config is None else base_config
        self.base_config = cfg.with_operating(mode="flux")
        self.weights = weights or ObjectiveWeights()
        self.constraints = constraints or Constraints()
        self.coking = coking
        self.keep_solutions = keep_solutions
        self._cache: dict[tuple[float, float], Evaluation] = {}
        self.history: list[Evaluation] = []

    # -- configuration -----------------------------------------------------
    def configure(self, x: Sequence[float]) -> ReformerConfig:
        """Build the configuration for operating point ``x = [S/C, q_mean]``."""
        sc, q = float(x[0]), float(x[1])
        cfg = self.base_config
        # Hold the methane feed rate fixed and add steam, which is what varying
        # S/C means operationally: the plant does not cut its feedstock to raise
        # the steam ratio.
        feed = cfg.feed
        ch4_flow = feed.molar_flows()[IDX["CH4"]]
        minor = (feed.hydrogen_fraction + feed.co2_fraction
                 + feed.nitrogen_fraction + feed.co_fraction)
        total = ch4_flow * (1.0 + sc) / max(1.0 - minor, 1.0e-6)
        return (cfg
                .with_feed(steam_to_carbon=sc, molar_flow=total)
                .with_operating(average_heat_flux=q))

    # -- evaluation --------------------------------------------------------
    def evaluate(self, x: Sequence[float]) -> Evaluation:
        """Simulate the tube at ``x``, with caching."""
        key = (round(float(x[0]), 6), round(float(x[1]), 3))
        if key in self._cache:
            return self._cache[key]

        cfg = self.configure(x)
        solution = Reformer2D(cfg).solve()
        metrics = dict(solution.summary())
        metrics["steam_to_carbon"] = float(x[0])
        metrics["heat_flux_W_m2"] = float(x[1])
        metrics["converged"] = solution.success
        metrics["steam_flow_mol_per_s"] = float(
            solution.species_molar_flow()[0, IDX["H2O"]])
        metrics["methane_feed_mol_per_s"] = float(
            solution.species_molar_flow()[0, IDX["CH4"]])

        if self.coking or self.constraints.carbon_free:
            report = analyze_coking(solution, include_equilibrated=False)
            metrics["max_carbon_activity"] = float(
                np.max(report.activity_actual))
            metrics["coking_margin"] = report.margin()
        else:
            metrics["max_carbon_activity"] = float("nan")
            metrics["coking_margin"] = float("nan")

        metrics["objective"] = self.objective_from_metrics(metrics)
        metrics["constraint_violation"] = self.violation_from_metrics(metrics)
        metrics["feasible"] = (metrics["constraint_violation"]
                               <= self.constraints.feasibility_tolerance)

        ev = Evaluation(np.asarray(x, dtype=float), metrics,
                        solution if self.keep_solutions else None)
        self._cache[key] = ev
        self.history.append(ev)
        return ev

    __call__ = evaluate

    # -- objective and constraints ----------------------------------------
    def objective_from_metrics(self, m: dict) -> float:
        """Scalar objective (to be minimised)."""
        w = self.weights
        h2_max = 4.0 * m["methane_feed_mol_per_s"]      # CH4 + 2H2O -> CO2 + 4H2
        j = -w.hydrogen * m["hydrogen_production_mol_per_s"] / max(h2_max, 1e-30)
        j += w.methane_slip * m["methane_slip_dry"] / w.slip_reference
        j += w.steam * (m["steam_to_carbon"] - w.steam_reference)
        j += w.duty * (m["heat_duty_kW"] * 1.0e3) / w.duty_reference
        if not m.get("converged", True):
            j += 10.0
        return float(j)

    def constraint_values(self, m: dict) -> np.ndarray:
        """Constraint vector ``g(x) >= 0`` (feasible when all entries >= 0)."""
        c = self.constraints
        g = [
            (c.max_wall_temperature - m["max_wall_surface_temperature_K"]) / 50.0,
            (c.max_pressure_drop - m["pressure_drop_bar"] * PA_PER_BAR) / 1.0e5,
        ]
        if c.min_methane_conversion > 0.0:
            g.append((m["methane_conversion"] - c.min_methane_conversion) / 0.05)
        if c.carbon_free:
            g.append(1.0 - m["max_carbon_activity"])
        return np.array(g, dtype=float)

    def violation_from_metrics(self, m: dict) -> float:
        """Largest constraint violation (``<= 0`` when feasible)."""
        return float(np.max(-self.constraint_values(m)))

    def penalty_objective(self, x: Sequence[float],
                          penalty: float = 50.0) -> float:
        """Objective with an exterior quadratic penalty, for derivative-free
        and global methods."""
        m = self.evaluate(x).metrics
        g = self.constraint_values(m)
        violation = np.clip(-g, 0.0, None)
        return m["objective"] + penalty * float(np.sum(violation**2))

    @property
    def n_evaluations(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Outcome of :func:`optimize_operating_point`."""

    x: np.ndarray
    steam_to_carbon: float
    heat_flux: float
    objective: float
    metrics: dict
    solution: ReformerSolution
    config: ReformerConfig
    history: list[Evaluation] = field(default_factory=list)
    success: bool = True
    message: str = ""
    method: str = ""
    n_evaluations: int = 0
    wall_time: float = 0.0

    def report(self) -> str:
        m = self.metrics
        lines = [
            f"Optimisation result ({self.method})",
            "-" * 64,
            f"  Status                          : "
            f"{'converged' if self.success else 'NOT converged'} -- "
            f"{self.message}",
            f"  Evaluations / wall time         : "
            f"{self.n_evaluations} / {self.wall_time:.1f} s",
            "",
            "  Optimal operating point",
            f"    Steam-to-carbon ratio         : {self.steam_to_carbon:8.3f}",
            f"    Mean wall heat flux           : "
            f"{self.heat_flux / 1e3:8.2f} kW/m2",
            f"    Tube duty                     : {m['heat_duty_kW']:8.1f} kW",
            "",
            "  Performance at the optimum",
            f"    Methane conversion            : "
            f"{m['methane_conversion'] * 100:8.2f} %",
            f"    Hydrogen production           : "
            f"{m['hydrogen_production_mol_per_s']:8.3f} mol/s per tube",
            f"    Hydrogen yield                : "
            f"{m['hydrogen_yield_mol_per_mol_CH4']:8.3f} mol/mol CH4",
            f"    Methane slip (dry)            : "
            f"{m['methane_slip_dry'] * 100:8.2f} %",
            f"    Outlet bulk temperature       : "
            f"{m['outlet_bulk_temperature_K']:8.1f} K",
            f"    Pressure drop                 : "
            f"{m['pressure_drop_bar']:8.2f} bar",
            "",
            "  Constraint status",
            f"    Peak wall surface temperature : "
            f"{m['max_wall_surface_temperature_K']:8.1f} K",
            f"    Largest violation (<=0 is ok) : "
            f"{m['constraint_violation']:+8.4f}",
        ]
        if np.isfinite(m.get("max_carbon_activity", float("nan"))):
            lines.append(f"    Peak carbon activity          : "
                         f"{m['max_carbon_activity']:8.4g}")
        return "\n".join(lines)


@dataclass
class ParetoFront:
    """Non-dominated set produced by :func:`pareto_front`."""

    variable: str
    epsilon: np.ndarray                 # the constrained objective's bound
    hydrogen_production: np.ndarray     # [mol/s]
    methane_conversion: np.ndarray      # [-]
    methane_slip: np.ndarray            # [-] dry
    steam_to_carbon: np.ndarray
    heat_flux: np.ndarray               # [W/m^2]
    max_wall_temperature: np.ndarray    # [K]
    steam_flow: np.ndarray              # [mol/s]
    feasible: np.ndarray                # bool
    metrics: list[dict] = field(default_factory=list)

    def non_dominated_mask(self) -> np.ndarray:
        """Mask of genuinely non-dominated points.

        For the wall-temperature front the two objectives are (maximise H2
        production, minimise peak wall temperature); for the steam front they
        are (maximise H2 production, minimise steam flow).
        """
        cost = (self.max_wall_temperature if self.variable == "wall_temperature"
                else self.steam_flow)
        gain = self.hydrogen_production
        keep = np.ones(gain.size, dtype=bool)
        for i in range(gain.size):
            if not self.feasible[i]:
                keep[i] = False
                continue
            dominated = ((gain >= gain[i]) & (cost <= cost[i])
                         & ((gain > gain[i]) | (cost < cost[i])) & self.feasible)
            keep[i] = not dominated.any()
        return keep

    def to_table(self) -> str:
        """Fixed-width table of the front."""
        head = (f"{'eps':>9} {'S/C':>6} {'q[kW/m2]':>9} {'Twall[K]':>9} "
                f"{'X_CH4[%]':>9} {'H2[mol/s]':>10} {'slip[%]':>8} {'ok':>4}")
        rows = [head, "-" * len(head)]
        for i in range(self.epsilon.size):
            rows.append(
                f"{self.epsilon[i]:9.3f} {self.steam_to_carbon[i]:6.2f} "
                f"{self.heat_flux[i] / 1e3:9.2f} "
                f"{self.max_wall_temperature[i]:9.1f} "
                f"{self.methane_conversion[i] * 100:9.2f} "
                f"{self.hydrogen_production[i]:10.3f} "
                f"{self.methane_slip[i] * 100:8.3f} "
                f"{'y' if self.feasible[i] else 'n':>4}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------

def optimize_operating_point(
        config: Optional[ReformerConfig] = None,
        space: Optional[DesignSpace] = None,
        weights: Optional[ObjectiveWeights] = None,
        constraints: Optional[Constraints] = None,
        method: Literal["slsqp", "nelder-mead",
                        "differential_evolution"] = "slsqp",
        x0: Optional[Sequence[float]] = None,
        coking: bool = False,
        coarse: bool = True,
        max_iterations: int = 60,
        verbose: bool = True) -> OptimizationResult:
    """Tune S/C and the mean wall heat flux.

    Parameters
    ----------
    config : ReformerConfig, optional
        Template configuration (default: the baseline industrial tube).
    space : DesignSpace, optional
        Bounds on the decision variables.
    weights, constraints : optional
        Objective weights and constraint limits.
    method : str
        ``"slsqp"``, ``"nelder-mead"`` or ``"differential_evolution"``.
    x0 : sequence of float, optional
        Starting point ``[S/C, q_mean]``; defaults to the centre of the box.
    coking : bool
        Evaluate carbon activities at every point.
    coarse : bool
        Use :func:`fast_numerics` during the search; the returned solution is
        always recomputed on the full grid.
    max_iterations : int
        Iteration budget passed to the underlying optimiser.
    """
    cfg = default_config() if config is None else config
    space = space or DesignSpace()
    constraints = constraints or Constraints()
    weights = weights or ObjectiveWeights()
    search_cfg = fast_numerics(cfg) if coarse else cfg

    evaluator = ReformerEvaluator(search_cfg, weights, constraints, coking=coking)
    t_start = time.perf_counter()

    if verbose:
        print(f"Optimising with {method}: "
              f"S/C in [{space.steam_to_carbon[0]:.2f},"
              f" {space.steam_to_carbon[1]:.2f}], "
              f"q in [{space.heat_flux[0] / 1e3:.1f},"
              f" {space.heat_flux[1] / 1e3:.1f}] kW/m2")

    if method == "slsqp":
        def f_unit(u):
            return evaluator.evaluate(space.from_unit(u)).metrics["objective"]

        def g_unit(u):
            m = evaluator.evaluate(space.from_unit(u)).metrics
            return evaluator.constraint_values(m)

        u0 = (space.to_unit(x0) if x0 is not None
              else np.array([0.35, 0.5]))
        res = minimize(f_unit, np.clip(u0, 0.0, 1.0), method="SLSQP",
                       bounds=[(0.0, 1.0)] * 2,
                       constraints=[{"type": "ineq", "fun": g_unit}],
                       options={"maxiter": max_iterations, "ftol": 1.0e-6,
                                "eps": 2.0e-3, "disp": False})
        x_opt = space.from_unit(res.x)
        success, message = bool(res.success), str(res.message)

    elif method == "nelder-mead":
        def f_pen(u):
            return evaluator.penalty_objective(space.from_unit(u))

        u0 = (space.to_unit(x0) if x0 is not None else np.array([0.35, 0.5]))
        res = minimize(f_pen, np.clip(u0, 0.0, 1.0), method="Nelder-Mead",
                       bounds=[(0.0, 1.0)] * 2,
                       options={"maxiter": max_iterations, "xatol": 1.0e-3,
                                "fatol": 1.0e-5})
        x_opt = space.from_unit(res.x)
        success, message = bool(res.success), str(res.message)

    elif method == "differential_evolution":
        def f_pen(u):
            return evaluator.penalty_objective(space.from_unit(u))

        res = differential_evolution(
            f_pen, bounds=[(0.0, 1.0)] * 2, maxiter=max_iterations,
            popsize=8, tol=1.0e-4, polish=True, seed=0, init="sobol")
        x_opt = space.from_unit(res.x)
        success, message = bool(res.success), str(res.message)

    else:  # pragma: no cover
        raise ValueError(f"Unknown method {method!r}")

    # Re-solve the optimum on the full-fidelity grid.
    final_evaluator = ReformerEvaluator(cfg, weights, constraints,
                                        coking=coking, keep_solutions=True)
    final = final_evaluator.evaluate(x_opt)
    elapsed = time.perf_counter() - t_start

    result = OptimizationResult(
        x=np.asarray(x_opt, dtype=float),
        steam_to_carbon=float(x_opt[0]),
        heat_flux=float(x_opt[1]),
        objective=final.metrics["objective"],
        metrics=final.metrics,
        solution=final.solution,
        config=final_evaluator.configure(x_opt),
        history=evaluator.history,
        success=success, message=message, method=method,
        n_evaluations=evaluator.n_evaluations, wall_time=elapsed,
    )
    if verbose:
        print(result.report())
    return result


def pareto_front(config: Optional[ReformerConfig] = None,
                 variable: Literal["wall_temperature", "steam"] = "wall_temperature",
                 epsilon: Optional[Sequence[float]] = None,
                 space: Optional[DesignSpace] = None,
                 weights: Optional[ObjectiveWeights] = None,
                 constraints: Optional[Constraints] = None,
                 coking: bool = False,
                 max_iterations: int = 40,
                 verbose: bool = True) -> ParetoFront:
    """Epsilon-constraint Pareto front.

    Parameters
    ----------
    variable : str
        ``"wall_temperature"`` sweeps the metallurgical limit (production vs
        tube life); ``"steam"`` fixes S/C at each point and optimises the flux
        alone (production vs steam cost).
    epsilon : sequence of float, optional
        Values of the swept bound.  Defaults are 1060-1200 K for the wall
        temperature front and S/C 2.0-4.5 for the steam front.
    """
    cfg = default_config() if config is None else config
    space = space or DesignSpace()
    weights = weights or ObjectiveWeights()
    base_constraints = constraints or Constraints()

    if epsilon is None:
        epsilon = (np.linspace(1060.0, 1200.0, 8) if variable == "wall_temperature"
                   else np.linspace(space.steam_to_carbon[0],
                                    space.steam_to_carbon[1], 8))
    epsilon = np.asarray(epsilon, dtype=float)

    # Maximise hydrogen production alone along the front; the trade-off is
    # carried by the swept constraint, not by the weights.
    front_weights = replace(weights, methane_slip=0.0, steam=0.0, duty=0.0)

    records: list[dict] = []
    for i, eps in enumerate(epsilon):
        if variable == "wall_temperature":
            local_constraints = replace(base_constraints,
                                        max_wall_temperature=float(eps))
            local_space = space
            x0 = None
        else:
            local_constraints = base_constraints
            local_space = replace(space, steam_to_carbon=(float(eps), float(eps)))
            x0 = np.array([float(eps), 0.5 * (space.heat_flux[0]
                                              + space.heat_flux[1])])

        if verbose:
            label = (f"T_wall <= {eps:.0f} K" if variable == "wall_temperature"
                     else f"S/C = {eps:.2f}")
            print(f"  [{i + 1}/{epsilon.size}] {label}", flush=True)

        res = optimize_operating_point(
            cfg, space=local_space, weights=front_weights,
            constraints=local_constraints, method="slsqp", x0=x0,
            coking=coking, coarse=True, max_iterations=max_iterations,
            verbose=False)
        records.append(res.metrics)

    def column(key: str) -> np.ndarray:
        return np.array([r[key] for r in records], dtype=float)

    front = ParetoFront(
        variable=variable,
        epsilon=epsilon,
        hydrogen_production=column("hydrogen_production_mol_per_s"),
        methane_conversion=column("methane_conversion"),
        methane_slip=column("methane_slip_dry"),
        steam_to_carbon=column("steam_to_carbon"),
        heat_flux=column("heat_flux_W_m2"),
        max_wall_temperature=column("max_wall_surface_temperature_K"),
        steam_flow=column("steam_flow_mol_per_s"),
        feasible=np.array([bool(r["feasible"]) for r in records]),
        metrics=records,
    )
    if verbose:
        print(front.to_table())
    return front


def sensitivity_scan(config: Optional[ReformerConfig] = None,
                     steam_to_carbon: Optional[Sequence[float]] = None,
                     heat_flux: Optional[Sequence[float]] = None,
                     coking: bool = True,
                     coarse: bool = True,
                     verbose: bool = True) -> dict:
    """Full-factorial scan of the design space.

    Returns a dict of ``(n_sc, n_q)`` arrays suitable for contour plotting:
    ``methane_conversion``, ``hydrogen_production``, ``methane_slip``,
    ``max_wall_temperature``, ``max_carbon_activity``, ``pressure_drop``.
    """
    cfg = default_config() if config is None else config
    search_cfg = fast_numerics(cfg) if coarse else cfg
    sc = np.asarray(steam_to_carbon if steam_to_carbon is not None
                    else np.linspace(2.0, 4.5, 6), dtype=float)
    q = np.asarray(heat_flux if heat_flux is not None
                   else np.linspace(4.0e4, 9.0e4, 6), dtype=float)

    evaluator = ReformerEvaluator(search_cfg, coking=coking)
    keys = ("methane_conversion", "hydrogen_production_mol_per_s",
            "methane_slip_dry", "max_wall_surface_temperature_K",
            "max_carbon_activity", "pressure_drop_bar",
            "outlet_bulk_temperature_K", "max_radial_dT_K")
    grids = {k: np.empty((sc.size, q.size)) for k in keys}

    for i, s in enumerate(sc):
        for j, flux in enumerate(q):
            m = evaluator.evaluate([s, flux]).metrics
            for k in keys:
                grids[k][i, j] = m[k]
        if verbose:
            print(f"  S/C = {s:.2f} done ({(i + 1) * q.size}"
                  f"/{sc.size * q.size} points)", flush=True)

    grids["steam_to_carbon"] = sc
    grids["heat_flux"] = q
    return grids


def operating_envelope(config: Optional[ReformerConfig] = None,
                       steam_to_carbon: Optional[Sequence[float]] = None,
                       space: Optional[DesignSpace] = None,
                       constraints: Optional[Constraints] = None,
                       coarse: bool = True,
                       flux_tolerance: float = 50.0,
                       verbose: bool = True) -> dict:
    """Largest admissible wall heat flux as a function of S/C.

    For each steam-to-carbon ratio this brackets and then bisects on the mean
    wall flux until the binding constraint is exactly satisfied, and records
    which constraint it was.  It is the cheapest way to see the *shape* of the
    feasible region, and it explains the optimum that
    :func:`optimize_operating_point` finds:

    * at low S/C the **metallurgical limit** binds.  Adding steam raises
      conversion at essentially constant peak wall temperature, so hydrogen
      production climbs.
    * above a corner S/C the **pressure-drop limit** takes over.  The extra
      steam raises the mass flux and the Ergun term with it, so the firing has
      to be cut faster than the chemistry improves, and hydrogen production
      *falls* with further steam.

    Hydrogen production is therefore maximised exactly at the corner where the
    two constraints meet -- a geometric argument that is completely independent
    of the numerical optimiser, and a useful check on it.

    Returns
    -------
    dict with keys ``steam_to_carbon``, ``max_heat_flux``, ``binding``,
    ``hydrogen_production``, ``methane_conversion``,
    ``max_wall_temperature``, ``pressure_drop`` and ``corner_steam_to_carbon``.
    """
    from scipy.optimize import brentq

    cfg = default_config() if config is None else config
    space = space or DesignSpace()
    constraints = constraints or Constraints()
    search_cfg = fast_numerics(cfg) if coarse else cfg
    evaluator = ReformerEvaluator(search_cfg, constraints=constraints)

    ratios = np.asarray(steam_to_carbon if steam_to_carbon is not None
                        else np.linspace(space.steam_to_carbon[0],
                                         space.steam_to_carbon[1], 11),
                        dtype=float)
    low, high = space.heat_flux

    def slack(sc: float, q: float) -> float:
        """Positive inside the feasible region."""
        return -evaluator.evaluate([sc, q]).metrics["constraint_violation"]

    records: list[dict] = []
    for sc in ratios:
        if slack(sc, high) > 0.0:
            q_max = high
        elif slack(sc, low) < 0.0:
            q_max = low
        else:
            q_max = brentq(lambda q: slack(sc, q), low, high, xtol=flux_tolerance)
        m = evaluator.evaluate([sc, q_max]).metrics
        # Which constraint is tight?  Compare each slack in its own scaled
        # units, exactly as the optimiser sees them.
        g = evaluator.constraint_values(m)
        binding = ("wall temperature", "pressure drop")[int(np.argmin(g[:2]))]
        if q_max >= high - flux_tolerance:
            binding = "flux bound"
        records.append({"steam_to_carbon": float(sc), "max_heat_flux": q_max,
                        "binding": binding, "metrics": m})
        if verbose:
            print(f"  S/C {sc:4.2f}: q_max {q_max / 1e3:6.2f} kW/m2 "
                  f"({binding}), X = {m['methane_conversion'] * 100:5.2f}%, "
                  f"H2 = {m['hydrogen_production_mol_per_s']:.3f} mol/s",
                  flush=True)

    hydrogen = np.array([r["metrics"]["hydrogen_production_mol_per_s"]
                         for r in records])
    binding = [r["binding"] for r in records]
    # The corner is the last ratio at which the wall temperature still binds.
    wall_limited = [i for i, b in enumerate(binding) if b == "wall temperature"]
    corner = float(ratios[wall_limited[-1]]) if wall_limited else float("nan")

    return {
        "steam_to_carbon": ratios,
        "max_heat_flux": np.array([r["max_heat_flux"] for r in records]),
        "binding": binding,
        "hydrogen_production": hydrogen,
        "methane_conversion": np.array(
            [r["metrics"]["methane_conversion"] for r in records]),
        "max_wall_temperature": np.array(
            [r["metrics"]["max_wall_surface_temperature_K"] for r in records]),
        "pressure_drop": np.array(
            [r["metrics"]["pressure_drop_bar"] for r in records]),
        "corner_steam_to_carbon": corner,
        "best_steam_to_carbon": float(ratios[int(np.argmax(hydrogen))]),
        "constraints": constraints,
    }
