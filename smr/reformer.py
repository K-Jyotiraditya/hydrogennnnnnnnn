r"""
Two-dimensional pseudo-heterogeneous model of a single industrial SMR tube.

Model formulation
-----------------
Axisymmetric packed bed, plug flow in ``z`` with radial dispersion of heat and
species.  The state variables are the superficial species molar fluxes
``N_i(r, z)`` [mol/(m^2 s)], the gas temperature ``T(r, z)`` [K] and the
cross-section pressure ``P(z)`` [Pa]:

Species (one equation per species per radial node)

    dN_i/dz = (1/r) d/dr ( r D_er dC_i/dr ) + rho_b sum_j nu_ji eta_j r_j

Energy

    ( sum_i N_i cp_i ) dT/dz = (1/r) d/dr ( r k_er dT/dr )
                               - rho_b sum_j dH_j eta_j r_j

Momentum (cross-section averaged)

    dP/dz = -[ 150 mu (1-eps)^2/(eps^3 d_p^2) u_s
               + 1.75 rho (1-eps)/(eps^3 d_p) u_s^2 ]

Boundary conditions

    r = 0      dT/dr = 0,  dC_i/dr = 0                (symmetry)
    r = R      dC_i/dr = 0                            (impermeable wall)
               -k_er dT/dr = -q_w(z)                  (specified flux), or
               -k_er dT/dr = U_w (T|_R - T_w(z))      (specified wall temp.)

The pseudo-heterogeneous character enters twice: the observed rate of each
reaction is the intrinsic rate times an intraparticle effectiveness factor
(:mod:`smr.pellet`), and the rates are evaluated at the catalyst *surface*
temperature obtained from the external film energy balance

    h_f a_v (T_g - T_s) = -rho_b sum_j dH_j eta_j r_j (T_s)

solved by Picard iteration at every node.

Numerics
--------
The radial direction is discretised with a conservative cell-centred finite
volume scheme on annular cells, which handles the ``1/r`` operator exactly and
guarantees that radial transport neither creates nor destroys mass or energy.
The resulting system of ``n_radial * (n_species + 1) + 1`` ODEs in ``z`` is
integrated with :func:`scipy.integrate.solve_ivp`.

Note on the energy equation: enthalpy transported by radial *species*
dispersion is deliberately omitted.  The dominant radial dispersion mechanism
is already represented in ``k_er`` through the ``Re Pr / Pe_r`` term, so adding
a species-dispersion enthalpy flux would double-count the same eddy transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import kinetics as kin
from . import transport as tr
from .config import ReformerConfig
from .pellet import effectiveness_factors
from .species import (
    IDX,
    N_SPECIES,
    SPECIES,
    cp_molar,
    enthalpy_molar,
    gas_density,
    molar_density,
    mixture_cp_mass,
    mixture_molar_mass,
    mixture_thermal_conductivity,
    mixture_viscosity,
)

__all__ = ["RadialGrid", "Reformer2D", "ReformerSolution"]

PA_PER_BAR = 1.0e5


def _smooth_positive(x: np.ndarray, delta: float) -> np.ndarray:
    r"""Smooth, non-negative replacement for ``max(x, 0)``.

    ``f(x) = (x + sqrt(x^2 + delta^2)) / 2`` is C-infinity, equals ``x`` to
    within ``delta`` for ``x >> delta`` and decays to ``0`` for ``x << -delta``.

    Hard clipping would do the same job but puts a kink in the right-hand side,
    which wrecks the finite-difference Jacobian the stiff integrator builds and
    is what makes a method-of-lines packed-bed model fail with "repeated
    convergence failures".  Negative fluxes have to be excluded because the
    Xu-Froment rate law contains ``p_H2**2.5``, which is not real-valued for a
    negative partial pressure.
    """
    return 0.5 * (x + np.sqrt(x * x + delta * delta))


# ---------------------------------------------------------------------------
# Radial grid
# ---------------------------------------------------------------------------

@dataclass
class RadialGrid:
    """Conservative annular finite-volume grid on ``0 <= r <= R``.

    Attributes
    ----------
    faces : ndarray, shape (n+1,)
        Cell face radii, ``faces[0] = 0`` and ``faces[-1] = R``.
    centers : ndarray, shape (n,)
        Area-centroid radii of the cells.
    area : ndarray, shape (n,)
        Cross-sectional area of each annular cell [m^2].
    circumference : ndarray, shape (n+1,)
        ``2 pi r`` at each face [m].
    """

    faces: np.ndarray
    centers: np.ndarray
    area: np.ndarray
    circumference: np.ndarray

    @property
    def n(self) -> int:
        return self.centers.size

    @property
    def radius(self) -> float:
        return float(self.faces[-1])

    @property
    def total_area(self) -> float:
        return float(self.area.sum())

    @property
    def wall_half_thickness(self) -> float:
        """Distance from the last cell centre to the wall [m]."""
        return float(self.faces[-1] - self.centers[-1])

    @classmethod
    def build(cls, radius: float, n: int, grading: float = 1.0) -> "RadialGrid":
        """Create a grid with ``n`` cells.

        ``grading > 1`` clusters cells towards the wall, where the temperature
        gradient is steepest; ``grading = 1`` gives uniform spacing in ``r``.
        """
        if n < 2:
            raise ValueError("n_radial must be at least 2.")
        xi = np.linspace(0.0, 1.0, n + 1)
        if abs(grading - 1.0) > 1.0e-12:
            xi = 1.0 - (1.0 - xi) ** grading
        faces = radius * xi
        area = np.pi * (faces[1:] ** 2 - faces[:-1] ** 2)
        # Area centroid of an annulus, so that cell values are second-order
        # accurate representations of the cell average.
        centers = np.sqrt(0.5 * (faces[1:] ** 2 + faces[:-1] ** 2))
        circumference = 2.0 * np.pi * faces
        return cls(faces=faces, centers=centers, area=area,
                   circumference=circumference)


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

@dataclass
class ReformerSolution:
    """Result of a :class:`Reformer2D` integration.

    Field shapes: ``nz`` axial output points, ``nr`` radial cells,
    ``nsp = 6`` species.
    """

    config: ReformerConfig
    grid: RadialGrid
    z: np.ndarray                 # (nz,)      [m]
    flux: np.ndarray              # (nz, nr, nsp) superficial N_i [mol/(m^2 s)]
    temperature: np.ndarray       # (nz, nr)   gas T [K]
    pressure: np.ndarray          # (nz,)      [Pa]
    diagnostics: dict = field(default_factory=dict)
    success: bool = True
    message: str = ""
    n_rhs_evaluations: int = 0
    solve_time: float = 0.0

    # -- basic derived fields ---------------------------------------------
    @property
    def n_axial(self) -> int:
        return self.z.size

    @property
    def total_flux(self) -> np.ndarray:
        """Total superficial molar flux [mol/(m^2 s)], shape (nz, nr)."""
        return self.flux.sum(axis=2)

    def mole_fractions(self) -> np.ndarray:
        """Local mole fractions, shape (nz, nr, nsp)."""
        total = np.clip(self.total_flux, 1.0e-30, None)[:, :, None]
        return self.flux / total

    def partial_pressures(self, unit: str = "Pa") -> np.ndarray:
        """Local partial pressures, shape (nz, nr, nsp)."""
        p = self.mole_fractions() * self.pressure[:, None, None]
        return p / PA_PER_BAR if unit == "bar" else p

    def species_molar_flow(self) -> np.ndarray:
        """Cross-section integrated species molar flow [mol/s], shape (nz, nsp)."""
        return np.einsum("r,zrs->zs", self.grid.area, self.flux)

    def total_molar_flow(self) -> np.ndarray:
        """Total molar flow [mol/s], shape (nz,)."""
        return self.species_molar_flow().sum(axis=1)

    def bulk_mole_fractions(self) -> np.ndarray:
        """Flow-averaged (mixing-cup) mole fractions, shape (nz, nsp)."""
        f = self.species_molar_flow()
        return f / np.clip(f.sum(axis=1, keepdims=True), 1.0e-30, None)

    def bulk_temperature(self) -> np.ndarray:
        """Enthalpy-consistent mixing-cup temperature [K], shape (nz,).

        Found by inverting ``H(T_cup) = H_actual`` with the mixture heat
        capacity, which is exact for the ideal-gas enthalpy model used here.
        """
        flows = self.species_molar_flow()                        # (nz, nsp)
        h_local = enthalpy_molar(self.temperature)               # (nz, nr, nsp)
        h_total = np.einsum("r,zrs,zrs->z", self.grid.area, self.flux, h_local)
        # Newton iteration on the cup temperature.
        t_cup = self.flow_average(self.temperature)
        for _ in range(40):
            h_try = np.einsum("zs,zs->z", flows, enthalpy_molar(t_cup))
            cp_try = np.einsum("zs,zs->z", flows, cp_molar(t_cup))
            step = (h_total - h_try) / np.clip(cp_try, 1.0e-12, None)
            t_cup = t_cup + step
            if np.max(np.abs(step)) < 1.0e-9:
                break
        return t_cup

    def flow_average(self, field_zr: np.ndarray) -> np.ndarray:
        """Molar-flow-weighted radial average of a ``(nz, nr)`` field."""
        weight = self.grid.area[None, :] * self.total_flux
        return (np.sum(weight * field_zr, axis=1)
                / np.clip(weight.sum(axis=1), 1.0e-30, None))

    def area_average(self, field_zr: np.ndarray) -> np.ndarray:
        """Area-weighted radial average of a ``(nz, nr)`` field."""
        return (np.einsum("r,zr->z", self.grid.area, field_zr)
                / self.grid.total_area)

    # -- performance metrics ----------------------------------------------
    def methane_conversion(self) -> np.ndarray:
        """Methane conversion [-], shape (nz,)."""
        f = self.species_molar_flow()[:, IDX["CH4"]]
        return (f[0] - f) / max(f[0], 1.0e-30)

    def carbon_conversion(self) -> np.ndarray:
        """Conversion of methane carbon to CO + CO2 [-], shape (nz,)."""
        f = self.species_molar_flow()
        carbon_oxides = f[:, IDX["CO"]] + f[:, IDX["CO2"]]
        return (carbon_oxides - carbon_oxides[0]) / max(f[0, IDX["CH4"]], 1.0e-30)

    def hydrogen_yield(self) -> np.ndarray:
        """Hydrogen produced per mole of methane fed [-], shape (nz,)."""
        f = self.species_molar_flow()
        return (f[:, IDX["H2"]] - f[0, IDX["H2"]]) / max(f[0, IDX["CH4"]], 1.0e-30)

    def hydrogen_production(self) -> np.ndarray:
        """Net hydrogen production rate [mol/s], shape (nz,)."""
        f = self.species_molar_flow()[:, IDX["H2"]]
        return f - f[0]

    def methane_slip(self) -> float:
        """Outlet methane mole fraction on a wet basis [-]."""
        return float(self.bulk_mole_fractions()[-1, IDX["CH4"]])

    def dry_mole_fractions(self) -> np.ndarray:
        """Bulk mole fractions on a dry (water-free) basis, shape (nz, nsp)."""
        y = self.bulk_mole_fractions().copy()
        y[:, IDX["H2O"]] = 0.0
        return y / np.clip(y.sum(axis=1, keepdims=True), 1.0e-30, None)

    def pressure_drop(self) -> float:
        """Total pressure drop over the tube [Pa]."""
        return float(self.pressure[0] - self.pressure[-1])

    # -- thermal fields ----------------------------------------------------
    def centreline_temperature(self) -> np.ndarray:
        """Temperature of the innermost cell [K], shape (nz,)."""
        return self.temperature[:, 0]

    def near_wall_temperature(self) -> np.ndarray:
        """Temperature of the outermost cell [K], shape (nz,)."""
        return self.temperature[:, -1]

    def wall_surface_temperature(self) -> np.ndarray:
        """Inner tube wall *surface* temperature [K], shape (nz,).

        Adds the near-wall half-cell conduction and the wall film resistance to
        the outermost bed temperature, so this is the temperature a pyrometer
        looking at the inside of the tube would see.
        """
        return self.diagnostics["T_wall_surface"]

    def outer_wall_temperature(self) -> np.ndarray:
        """Outer tube metal temperature [K], shape (nz,)."""
        return (self.wall_surface_temperature()
                + tr.tube_wall_temperature_drop(self.diagnostics["q_wall"],
                                                self.config.geometry))

    def radial_temperature_difference(self) -> np.ndarray:
        """Inner wall surface minus centreline temperature [K], shape (nz,)."""
        return self.wall_surface_temperature() - self.centreline_temperature()

    def max_wall_temperature(self) -> float:
        """Peak inner wall surface temperature [K]."""
        return float(np.max(self.wall_surface_temperature()))

    def heat_duty(self) -> float:
        """Total heat transferred through the wall [W]."""
        q = self.diagnostics["q_wall"]
        return float(np.pi * self.config.geometry.inner_diameter
                     * np.trapezoid(q, self.z))

    # -- verification ------------------------------------------------------
    def energy_balance_error(self) -> dict:
        """Global enthalpy balance audit.

        Returns absolute and relative closure of
        ``H_out - H_in = Q_wall``, which is an independent check on the
        discretisation: every term is recomputed from the stored solution
        rather than from the integrator's internal state.
        """
        h_in = float(np.einsum("r,rs,rs->", self.grid.area, self.flux[0],
                               enthalpy_molar(self.temperature[0])))
        h_out = float(np.einsum("r,rs,rs->", self.grid.area, self.flux[-1],
                                enthalpy_molar(self.temperature[-1])))
        q_wall = self.heat_duty()
        residual = (h_out - h_in) - q_wall
        return {
            "enthalpy_in_W": h_in,
            "enthalpy_out_W": h_out,
            "wall_heat_input_W": q_wall,
            "residual_W": residual,
            "relative_error": residual / max(abs(q_wall), 1.0e-30),
        }

    def mass_balance_error(self) -> dict:
        """Atomic balance audit for C, H and O [relative error]."""
        f_in = self.species_molar_flow()[0]
        f_out = self.species_molar_flow()[-1]
        # Atoms per molecule: CH4, H2O, H2, CO, CO2, N2
        atoms = {
            "C": np.array([1, 0, 0, 1, 1, 0], dtype=float),
            "H": np.array([4, 2, 2, 0, 0, 0], dtype=float),
            "O": np.array([0, 1, 0, 1, 2, 0], dtype=float),
        }
        out = {}
        for name, a in atoms.items():
            n_in, n_out = float(f_in @ a), float(f_out @ a)
            out[name] = (n_out - n_in) / max(n_in, 1.0e-30)
        return out

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        """Key outlet metrics and verification figures."""
        y_out = self.bulk_mole_fractions()[-1]
        y_dry = self.dry_mole_fractions()[-1]
        eb = self.energy_balance_error()
        return {
            "outlet_bulk_temperature_K": float(self.bulk_temperature()[-1]),
            "outlet_centreline_temperature_K": float(self.temperature[-1, 0]),
            "outlet_near_wall_temperature_K": float(self.temperature[-1, -1]),
            "outlet_pressure_bar": float(self.pressure[-1] / PA_PER_BAR),
            "pressure_drop_bar": self.pressure_drop() / PA_PER_BAR,
            "methane_conversion": float(self.methane_conversion()[-1]),
            "hydrogen_yield_mol_per_mol_CH4": float(self.hydrogen_yield()[-1]),
            "hydrogen_production_mol_per_s": float(self.hydrogen_production()[-1]),
            "methane_slip_wet": self.methane_slip(),
            "methane_slip_dry": float(y_dry[IDX["CH4"]]),
            "outlet_H2_dry": float(y_dry[IDX["H2"]]),
            "outlet_CO_dry": float(y_dry[IDX["CO"]]),
            "outlet_CO2_dry": float(y_dry[IDX["CO2"]]),
            "outlet_H2_CO_ratio": float(y_out[IDX["H2"]]
                                        / max(y_out[IDX["CO"]], 1.0e-30)),
            "max_wall_surface_temperature_K": self.max_wall_temperature(),
            "max_outer_wall_temperature_K": float(
                np.max(self.outer_wall_temperature())),
            "max_radial_dT_K": float(np.max(self.radial_temperature_difference())),
            "heat_duty_kW": self.heat_duty() / 1.0e3,
            "energy_balance_rel_error": eb["relative_error"],
            "carbon_balance_rel_error": self.mass_balance_error()["C"],
            "mean_effectiveness_R1": float(
                np.mean(self.diagnostics["eta"][:, :, 0])),
            "solve_time_s": self.solve_time,
            "rhs_evaluations": self.n_rhs_evaluations,
        }

    def report(self) -> str:
        """Formatted text report of the outlet state and verification."""
        s = self.summary()
        lines = [
            "Outlet state",
            "-" * 64,
            f"  Bulk (mixing-cup) temperature   : "
            f"{s['outlet_bulk_temperature_K']:8.1f} K",
            f"  Centreline / near-wall T        : "
            f"{s['outlet_centreline_temperature_K']:8.1f} /"
            f" {s['outlet_near_wall_temperature_K']:.1f} K",
            f"  Pressure (drop)                 : "
            f"{s['outlet_pressure_bar']:8.2f} bar "
            f"({s['pressure_drop_bar']:.2f} bar)",
            f"  Methane conversion              : "
            f"{s['methane_conversion'] * 100:8.2f} %",
            f"  Hydrogen yield                  : "
            f"{s['hydrogen_yield_mol_per_mol_CH4']:8.3f} mol H2 / mol CH4 fed",
            f"  Hydrogen production             : "
            f"{s['hydrogen_production_mol_per_s']:8.3f} mol/s per tube",
            f"  Methane slip (wet / dry)        : "
            f"{s['methane_slip_wet'] * 100:8.2f} / "
            f"{s['methane_slip_dry'] * 100:.2f} %",
            f"  Dry H2 / CO / CO2               : "
            f"{s['outlet_H2_dry'] * 100:8.2f} / {s['outlet_CO_dry'] * 100:.2f} /"
            f" {s['outlet_CO2_dry'] * 100:.2f} %",
            f"  H2/CO ratio                     : {s['outlet_H2_CO_ratio']:8.2f}",
            "",
            "Thermal and mechanical limits",
            "-" * 64,
            f"  Peak inner wall surface T       : "
            f"{s['max_wall_surface_temperature_K']:8.1f} K"
            f"   (limit {self.config.operating.wall_temperature_limit:.0f} K)",
            f"  Peak outer tube metal T         : "
            f"{s['max_outer_wall_temperature_K']:8.1f} K",
            f"  Peak radial dT (wall-centre)    : {s['max_radial_dT_K']:8.1f} K",
            f"  Heat duty per tube              : {s['heat_duty_kW']:8.1f} kW",
            "",
            "Verification",
            "-" * 64,
            f"  Enthalpy balance closure        : "
            f"{s['energy_balance_rel_error']:8.2e} (relative)",
            f"  Carbon atom balance             : "
            f"{s['carbon_balance_rel_error']:8.2e} (relative)",
            f"  Mean effectiveness factor R1    : "
            f"{s['mean_effectiveness_R1']:8.4f}",
            f"  Integration                     : "
            f"{s['solve_time_s']:8.2f} s, {s['rhs_evaluations']} RHS calls",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Reformer2D:
    """2D pseudo-heterogeneous reformer tube solver.

    Examples
    --------
    >>> from smr import default_config, Reformer2D
    >>> model = Reformer2D(default_config())
    >>> solution = model.solve()
    >>> round(solution.summary()["methane_conversion"], 3)  # doctest: +SKIP
    0.714
    """

    def __init__(self, config: ReformerConfig):
        self.config = config
        self.grid = RadialGrid.build(config.geometry.radius,
                                     config.numerics.n_radial,
                                     config.numerics.radial_grading)
        self._n_calls = 0
        # Flattened state layout: [N (nr*nsp) | T (nr) | P (1)]
        nr = self.grid.n
        self._slice_flux = slice(0, nr * N_SPECIES)
        self._slice_temp = slice(nr * N_SPECIES, nr * N_SPECIES + nr)
        self._index_pressure = nr * N_SPECIES + nr
        self._n_state = nr * N_SPECIES + nr + 1
        feed = config.feed
        self._flux_scale = feed.molar_flow / config.geometry.cross_section
        self._jac_sparsity = self._build_jacobian_sparsity()

    # -- Jacobian structure -----------------------------------------------
    def _build_jacobian_sparsity(self):
        """Sparsity pattern of ``d(rhs)/d(state)``.

        The radial finite-volume stencil couples each cell only to its two
        neighbours, so the Jacobian is block-tridiagonal apart from the
        pressure row and column, which are dense because the Ergun term uses
        cross-section averages.  Handing this pattern to the stiff integrator
        turns the Jacobian cost from ``O(n^2)`` right-hand-side evaluations
        into ``O(n)`` and -- more importantly here -- keeps the finite-
        difference Jacobian well-conditioned, which is what the strongly
        dispersive radial operator needs.
        """
        from scipy.sparse import lil_matrix

        nr = self.grid.n
        n = self._n_state
        pattern = lil_matrix((n, n), dtype=np.int8)

        def flux_rows(node: int) -> range:
            return range(node * N_SPECIES, (node + 1) * N_SPECIES)

        def temp_row(node: int) -> int:
            return nr * N_SPECIES + node

        for i in range(nr):
            neighbours = [j for j in (i - 1, i, i + 1) if 0 <= j < nr]
            dependencies = []
            for j in neighbours:
                dependencies.extend(flux_rows(j))
                dependencies.append(temp_row(j))
            dependencies.append(self._index_pressure)
            for row in list(flux_rows(i)) + [temp_row(i)]:
                for col in dependencies:
                    pattern[row, col] = 1

        # The pressure equation sees every cell.
        pattern[self._index_pressure, :] = 1
        return pattern.tocsr()

    # -- state packing -----------------------------------------------------
    def _unpack(self, state: np.ndarray):
        nr = self.grid.n
        flux = state[self._slice_flux].reshape(nr, N_SPECIES)
        temperature = state[self._slice_temp]
        pressure = state[self._index_pressure]
        return flux, temperature, pressure

    def _pack(self, flux: np.ndarray, temperature: np.ndarray,
              pressure: float) -> np.ndarray:
        state = np.empty(self._n_state)
        state[self._slice_flux] = flux.ravel()
        state[self._slice_temp] = temperature
        state[self._index_pressure] = pressure
        return state

    def initial_state(self) -> np.ndarray:
        """Inlet state vector: uniform composition and temperature over ``r``."""
        feed = self.config.feed
        flux_total = feed.molar_flow / self.config.geometry.cross_section
        flux = np.tile(flux_total * feed.mole_fractions(), (self.grid.n, 1))
        temperature = np.full(self.grid.n, feed.inlet_temperature)
        return self._pack(flux, temperature, feed.inlet_pressure)

    def absolute_tolerance(self):
        """Per-variable absolute tolerance vector for the integrator."""
        num = self.config.numerics
        if num.atol is not None:
            return num.atol
        atol = np.empty(self._n_state)
        atol[self._slice_flux] = num.atol_flux_relative * self._flux_scale
        atol[self._slice_temp] = num.atol_temperature
        atol[self._index_pressure] = num.atol_pressure
        return atol

    # -- local property evaluation ----------------------------------------
    def _local_properties(self, flux: np.ndarray, temperature: np.ndarray,
                          pressure: float) -> dict:
        """Per-node gas properties and bed transport coefficients."""
        cfg = self.config
        cat, geo, num = cfg.catalyst, cfg.geometry, cfg.numerics

        total = np.clip(flux.sum(axis=1), 1.0e-30, None)
        y = np.clip(flux, 0.0, None) / total[:, None]
        y = y / np.clip(y.sum(axis=1, keepdims=True), 1.0e-30, None)

        mw = mixture_molar_mass(y)                                   # (nr,)
        mass_flux = total * mw                                       # [kg/(m^2 s)]
        rho = gas_density(temperature, pressure, y)                  # [kg/m^3]
        u_s = mass_flux / np.clip(rho, 1.0e-6, None)                 # [m/s]
        mu = mixture_viscosity(temperature, y)
        k_gas = mixture_thermal_conductivity(temperature, y)
        cp_mass = mixture_cp_mass(temperature, y)
        cp_mol = cp_molar(temperature)                               # (nr, nsp)

        re_p = tr.particle_reynolds(mass_flux, cat.particle_diameter, mu)
        pr = tr.prandtl(cp_mass, mu, k_gas)

        k_er = tr.effective_radial_conductivity(temperature, k_gas, re_p, pr,
                                                cat, geo, num)
        h_wall = tr.wall_heat_transfer_coefficient(k_gas, re_p, pr, cat, geo, num)
        h_film = tr.film_heat_transfer_coefficient(k_gas, re_p, pr, cat)

        u_s_mean = float(np.einsum("r,r->", self.grid.area, u_s)
                         / self.grid.total_area)
        d_er = tr.effective_radial_diffusivity(u_s_mean, cat, geo, num)

        return {
            "y": y,
            "total_flux": total,
            "molar_mass": mw,
            "mass_flux": mass_flux,
            "density": rho,
            "velocity": u_s,
            "viscosity": mu,
            "k_gas": k_gas,
            "cp_mass": cp_mass,
            "cp_molar": cp_mol,
            "reynolds": re_p,
            "prandtl": pr,
            "k_er": k_er,
            "h_wall": h_wall,
            "h_film": h_film,
            "d_er": d_er,
        }

    def _face_mean(self, values: np.ndarray) -> np.ndarray:
        """Harmonic mean of a cell-centred transport coefficient at interior
        faces, which is the consistent choice for series conduction."""
        a, b = values[:-1], values[1:]
        return 2.0 * a * b / np.clip(a + b, 1.0e-30, None)

    def _reaction_source(self, temperature: np.ndarray, pressure: float,
                         props: dict) -> dict:
        """Per-node reaction rates, effectiveness factors and heat release.

        Includes the external film energy balance (gas -> catalyst surface),
        solved by Picard iteration, which is what makes the model
        pseudo-heterogeneous rather than pseudo-homogeneous.
        """
        cfg = self.config
        cat, num = cfg.catalyst, cfg.numerics
        nr = self.grid.n
        y = props["y"]
        p_bar = y * pressure / PA_PER_BAR

        a_v = cat.specific_surface_area
        film_conductance = np.clip(props["h_film"] * a_v, 1.0e-6, None)

        t_solid = temperature.copy()
        rates = np.zeros((nr, kin.N_REACTIONS))
        eta = np.ones((nr, kin.N_REACTIONS))
        heat = np.zeros(nr)
        d_h = np.zeros((nr, kin.N_REACTIONS))

        n_iter = num.film_iterations if num.film_resistance else 1
        for _ in range(n_iter):
            r_int, r_fwd = kin.reaction_rates_si(t_solid, p_bar,
                                                 num.p_h2_floor,
                                                 return_forward=True)
            eta = effectiveness_factors(r_fwd, t_solid, pressure, y, cat, num)
            rates = eta * r_int                                   # [mol/(kg s)]
            d_h = kin.reaction_enthalpies(t_solid)                # [J/mol]
            # Volumetric heat release (negative for the endothermic reformer).
            heat = -cat.bulk_density * np.einsum("nj,nj->n", d_h, rates)
            if not num.film_resistance:
                break
            t_solid = temperature + heat / film_conductance
            t_solid = np.clip(t_solid, 300.0, 1600.0)

        species_source = cat.bulk_density * np.einsum(
            "ji,nj->ni", kin.STOICHIOMETRY, rates)                 # [mol/(m^3 s)]

        return {
            "rates_intrinsic": kin.reaction_rates_si(temperature, p_bar,
                                                     num.p_h2_floor),
            "rates": rates,
            "eta": eta,
            "heat": heat,
            "species_source": species_source,
            "solid_temperature": t_solid,
            "reaction_enthalpy": d_h,
            "p_bar": p_bar,
        }

    def _wall_flux(self, z: float, temperature: np.ndarray,
                   props: dict) -> tuple[np.ndarray, np.ndarray]:
        """Wall heat flux [W/m^2] and inner wall surface temperature [K].

        The flux is positive when heat flows into the bed.
        """
        cfg = self.config
        op, geo = cfg.operating, cfg.geometry
        delta = self.grid.wall_half_thickness
        k_er_wall = props["k_er"][-1]
        h_wall = props["h_wall"][-1]
        # Series resistance from the last cell centre to the wall surface.
        resistance = delta / k_er_wall + 1.0 / h_wall

        if op.mode == "flux":
            q = float(op.heat_flux(np.atleast_1d(z), geo.length)[0])
            t_surface = temperature[-1] + q * resistance
        else:
            t_wall = float(op.wall_temperature(np.atleast_1d(z), geo.length)[0])
            q = (t_wall - temperature[-1]) / resistance
            t_surface = t_wall
        return np.asarray(q, dtype=float), np.asarray(t_surface, dtype=float)

    # -- right-hand side ---------------------------------------------------
    def rhs(self, z: float, state: np.ndarray) -> np.ndarray:
        """``d(state)/dz`` for :func:`scipy.integrate.solve_ivp`."""
        self._n_calls += 1
        cfg = self.config
        num = cfg.numerics
        grid = self.grid
        nr = grid.n

        flux, temperature, pressure = self._unpack(state)
        flux = _smooth_positive(flux, self._flux_scale * 1.0e-7)
        temperature = np.clip(temperature, 300.0, 1600.0)
        pressure = float(max(pressure, num.min_pressure))

        props = self._local_properties(flux, temperature, pressure)
        source = self._reaction_source(temperature, pressure, props)
        q_wall, _ = self._wall_flux(z, temperature, props)

        # --- radial conduction (energy) ----------------------------------
        k_face = self._face_mean(props["k_er"])
        dr_center = np.diff(grid.centers)
        cond = grid.circumference[1:-1] * k_face / dr_center          # [W/(m K)]
        dt_cell = np.diff(temperature)
        g_face = cond * dt_cell                                      # [W/m]
        energy_radial = np.zeros(nr)
        energy_radial[:-1] += g_face
        energy_radial[1:] -= g_face
        # Wall contribution on the outermost cell.
        energy_radial[-1] += grid.circumference[-1] * q_wall
        energy_radial /= grid.area                                   # [W/m^3]

        # --- radial dispersion (species) ---------------------------------
        # The dispersive flux is written on the MOLE FRACTION gradient,
        #     J_i = -C_tot D_er d(y_i)/dr,
        # so that sum_i J_i = -C_tot D_er d(sum_i y_i)/dr = 0 identically and
        # dispersion transports no net moles.  The apparently equivalent
        # -D_er d(C_i)/dr form is wrong in a non-isothermal bed: C_tot varies
        # with the radial temperature profile, so it drives a spurious net
        # molar flux towards the hot wall -- here about 6% of the through-flow,
        # which accumulates over the tube and eventually destroys the near-wall
        # composition.
        if num.radial_mass_dispersion and props["d_er"] > 0.0:
            c_total = molar_density(temperature, pressure)            # [mol/m^3]
            c_face = 0.5 * (c_total[:-1] + c_total[1:])
            cond_m = (grid.circumference[1:-1] * c_face * props["d_er"]
                      / dr_center)[:, None]
            g_species = cond_m * np.diff(props["y"], axis=0)          # [mol/(m s)]
            species_radial = np.zeros((nr, N_SPECIES))
            species_radial[:-1] += g_species
            species_radial[1:] -= g_species
            species_radial /= grid.area[:, None]
        else:
            species_radial = 0.0

        d_flux = species_radial + source["species_source"]

        # --- energy equation ---------------------------------------------
        heat_capacity_flux = np.einsum("ns,ns->n", flux, props["cp_molar"])
        d_temperature = ((energy_radial + source["heat"])
                         / np.clip(heat_capacity_flux, 1.0e-9, None))

        # --- momentum (Ergun, cross-section averaged) --------------------
        rho_mean = float(np.einsum("r,r->", grid.area, props["density"])
                         / grid.total_area)
        mu_mean = float(np.einsum("r,r->", grid.area, props["viscosity"])
                        / grid.total_area)
        mass_flux_mean = float(np.einsum("r,r->", grid.area, props["mass_flux"])
                               / grid.total_area)
        u_s_mean = mass_flux_mean / max(rho_mean, 1.0e-6)
        d_pressure = tr.ergun_pressure_gradient(u_s_mean, rho_mean, mu_mean,
                                                cfg.catalyst)

        return self._pack(d_flux, d_temperature, d_pressure)

    # -- integration -------------------------------------------------------
    def solve(self, z_span: Optional[tuple[float, float]] = None,
              initial_state: Optional[np.ndarray] = None,
              verbose: bool = False) -> ReformerSolution:
        """Integrate the tube and return a :class:`ReformerSolution`."""
        import time

        cfg = self.config
        num = cfg.numerics
        if z_span is None:
            z_span = (0.0, cfg.geometry.length)
        y0 = self.initial_state() if initial_state is None else initial_state
        z_out = np.linspace(z_span[0], z_span[1], num.n_axial_output)
        atol = self.absolute_tolerance()

        self._n_calls = 0
        t_start = time.perf_counter()
        options: dict = {}
        if num.method in ("BDF", "Radau", "LSODA"):
            # LSODA takes a banded structure rather than a sparsity pattern, so
            # the pattern is only passed to the implicit Runge-Kutta/BDF pair.
            if num.method in ("BDF", "Radau"):
                options["jac_sparsity"] = self._jac_sparsity
        result = solve_ivp(
            self.rhs, z_span, y0,
            method=num.method, t_eval=z_out,
            rtol=num.rtol, atol=atol, max_step=num.max_step,
            **options,
        )
        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"solve_ivp: {result.message} "
                  f"({self._n_calls} RHS evaluations, {elapsed:.2f} s)")

        nz = result.t.size
        nr = self.grid.n
        flux = result.y[self._slice_flux, :].T.reshape(nz, nr, N_SPECIES)
        flux = np.clip(flux, 0.0, None)
        temperature = result.y[self._slice_temp, :].T
        pressure = result.y[self._index_pressure, :]

        solution = ReformerSolution(
            config=cfg, grid=self.grid, z=result.t, flux=flux,
            temperature=temperature, pressure=pressure,
            success=bool(result.success), message=str(result.message),
            n_rhs_evaluations=self._n_calls, solve_time=elapsed,
        )
        solution.diagnostics = self.collect_diagnostics(solution)
        return solution

    def collect_diagnostics(self, solution: ReformerSolution) -> dict:
        """Re-evaluate the closures along a stored solution.

        Everything a post-processor or plot needs, on the ``(nz, nr)`` grid.
        """
        nz, nr = solution.z.size, self.grid.n
        n_rxn = kin.N_REACTIONS
        out = {
            "k_er": np.zeros((nz, nr)),
            "h_wall": np.zeros((nz, nr)),
            "h_film": np.zeros((nz, nr)),
            "reynolds": np.zeros((nz, nr)),
            "prandtl": np.zeros((nz, nr)),
            "velocity": np.zeros((nz, nr)),
            "density": np.zeros((nz, nr)),
            "viscosity": np.zeros((nz, nr)),
            "solid_temperature": np.zeros((nz, nr)),
            "heat_release": np.zeros((nz, nr)),
            "eta": np.zeros((nz, nr, n_rxn)),
            "rates": np.zeros((nz, nr, n_rxn)),
            "rates_intrinsic": np.zeros((nz, nr, n_rxn)),
            "reaction_enthalpy": np.zeros((nz, nr, n_rxn)),
            "approach_to_equilibrium": np.zeros((nz, nr, n_rxn)),
            "q_wall": np.zeros(nz),
            "T_wall_surface": np.zeros(nz),
            "d_er": np.zeros(nz),
        }
        for k in range(nz):
            flux = solution.flux[k]
            temperature = solution.temperature[k]
            pressure = float(solution.pressure[k])
            props = self._local_properties(flux, temperature, pressure)
            source = self._reaction_source(temperature, pressure, props)
            q_wall, t_surface = self._wall_flux(float(solution.z[k]),
                                                temperature, props)
            for key in ("k_er", "h_wall", "h_film", "reynolds", "prandtl",
                        "velocity", "density", "viscosity"):
                out[key][k] = props[key]
            out["solid_temperature"][k] = source["solid_temperature"]
            out["heat_release"][k] = source["heat"]
            out["eta"][k] = source["eta"]
            out["rates"][k] = source["rates"]
            out["rates_intrinsic"][k] = source["rates_intrinsic"]
            out["reaction_enthalpy"][k] = source["reaction_enthalpy"]
            out["approach_to_equilibrium"][k] = kin.approach_to_equilibrium(
                temperature, source["p_bar"], self.config.numerics.p_h2_floor)
            out["q_wall"][k] = float(q_wall)
            out["T_wall_surface"][k] = float(t_surface)
            out["d_er"][k] = props["d_er"]
        return out
