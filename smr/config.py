r"""
Configuration objects for the 2D SMR tube model.

Every default carries a provenance comment.  Numbers tagged ``[NOB21]`` follow
the industrial reformer of Nobandegani et al. (2021); ``[XF89]`` follows Xu &
Froment (1989); ``[CORR]`` means the value comes from a standard correlation or
accepted industrial practice rather than one specific paper.

Items tagged ``[CHECK]`` are representative industrial values chosen to be
consistent with the cited geometry and duty.  They are the ones to reconcile
against the source tables before quoting the model as a reproduction of a
specific published case.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional

import numpy as np

from .species import SPECIES, N_SPECIES, IDX

__all__ = [
    "Geometry",
    "Catalyst",
    "Feed",
    "Operating",
    "Numerics",
    "ReformerConfig",
    "SPECIES",
    "default_config",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """Reformer tube geometry (single tube)."""

    length: float = 13.6              # [m]  heated tube length        [NOB21]
    inner_diameter: float = 0.098     # [m]  tube inner diameter       [NOB21]
    wall_thickness: float = 0.012     # [m]  tube wall thickness       [CHECK]
    wall_conductivity: float = 28.0   # [W/(m K)] HP-Nb alloy @ ~1200 K [CORR]

    @property
    def radius(self) -> float:
        """Inner radius [m]."""
        return 0.5 * self.inner_diameter

    @property
    def cross_section(self) -> float:
        """Inner cross-sectional area [m^2]."""
        return np.pi * self.radius**2

    @property
    def inner_surface_area(self) -> float:
        """Heated inner surface area of the whole tube [m^2]."""
        return np.pi * self.inner_diameter * self.length

    @property
    def bed_volume(self) -> float:
        """Total (bulk) bed volume [m^3]."""
        return self.cross_section * self.length


# ---------------------------------------------------------------------------
# Catalyst and bed
# ---------------------------------------------------------------------------

@dataclass
class Catalyst:
    """Packed bed of Ni/MgAl2O4 reforming catalyst."""

    void_fraction: float = 0.45          # [-]  bed voidage                [NOB21]
    particle_diameter: float = 0.015     # [m]  equivalent sphere diameter [NOB21]
    bulk_density: float = 1016.0         # [kg/m^3] bed bulk density       [CHECK]
    shape_factor: float = 1.0            # [-]  1.0 sphere, <1 ring/cylinder
    pellet_porosity: float = 0.53        # [-]  intraparticle porosity     [XF89]
    tortuosity: float = 3.54             # [-]  pore tortuosity            [XF89]
    pore_radius: float = 1.0e-8          # [m]  mean pore radius           [XF89]
    solid_conductivity: float = 0.3      # [W/(m K)] catalyst solid        [CORR]
    emissivity: float = 0.85             # [-]  pellet surface emissivity  [CORR]

    #: How the intraparticle effectiveness factor is obtained:
    #: ``"thiele"``   -- generalised Thiele modulus per reaction (default)
    #: ``"constant"`` -- fixed values from :attr:`eta_constant`
    #: ``"none"``     -- eta = 1 (intrinsic kinetics, no diffusion limitation)
    effectiveness_model: Literal["thiele", "constant", "none"] = "thiele"
    eta_constant: tuple[float, float, float] = (0.05, 0.20, 0.05)  # [CORR]
    eta_scale: float = 1.0               # calibration multiplier on eta

    @property
    def solid_density(self) -> float:
        """Apparent pellet density [kg/m^3] consistent with the bulk density."""
        return self.bulk_density / (1.0 - self.void_fraction)

    @property
    def specific_surface_area(self) -> float:
        """External pellet area per unit bed volume a_v [m^2/m^3]."""
        return 6.0 * (1.0 - self.void_fraction) / self.particle_diameter

    @property
    def characteristic_length(self) -> float:
        """V_p/S_p characteristic length [m] (d_p/6 for a sphere)."""
        return self.particle_diameter / 6.0


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

@dataclass
class Feed:
    """Inlet stream of one tube.

    The composition is set by the steam-to-carbon ratio together with the
    hydrogen, carbon-dioxide and nitrogen fractions typical of a recycle-bearing
    industrial feed.  ``molar_flow`` is the total inlet flow per tube.
    """

    steam_to_carbon: float = 3.0        # [mol H2O / mol CH4]              [NOB21]
    inlet_temperature: float = 793.0    # [K]   mixed feed inlet            [NOB21]
    inlet_pressure: float = 2.6e6       # [Pa]  ~26 bar                     [NOB21]
    molar_flow: float = 4.45            # [mol/s] total per tube            [CHECK]

    # Non-methane, non-steam fractions of the dry-ish feed, on a total basis.
    hydrogen_fraction: float = 0.025    # [-]  recycle H2                   [CHECK]
    co2_fraction: float = 0.010         # [-]  recycle CO2                  [CHECK]
    nitrogen_fraction: float = 0.005    # [-]  inert                        [CHECK]
    co_fraction: float = 0.0            # [-]                               [CHECK]

    def mole_fractions(self) -> np.ndarray:
        """Inlet mole fractions in :data:`smr.species.SPECIES` order."""
        y = np.zeros(N_SPECIES)
        y[IDX["H2"]] = self.hydrogen_fraction
        y[IDX["CO2"]] = self.co2_fraction
        y[IDX["N2"]] = self.nitrogen_fraction
        y[IDX["CO"]] = self.co_fraction
        remainder = 1.0 - y.sum()
        if remainder <= 0.0:
            raise ValueError("Minor feed fractions exceed unity.")
        # Split the remainder between CH4 and H2O at the requested S/C ratio.
        x_ch4 = remainder / (1.0 + self.steam_to_carbon)
        y[IDX["CH4"]] = x_ch4
        y[IDX["H2O"]] = remainder - x_ch4
        return y

    def molar_flows(self) -> np.ndarray:
        """Inlet species molar flows [mol/s]."""
        return self.molar_flow * self.mole_fractions()

    def mass_flow(self) -> float:
        """Inlet mass flow [kg/s]."""
        from .species import MW
        return float(np.sum(self.molar_flows() * MW))


# ---------------------------------------------------------------------------
# Firing / wall boundary condition
# ---------------------------------------------------------------------------

#: Signature of a user-supplied axial boundary profile: f(z[m]) -> value.
AxialProfile = Callable[[np.ndarray], np.ndarray]


@dataclass
class Operating:
    """Thermal boundary condition at the inner tube wall.

    Two modes are supported, mirroring the two ways reformer duty is reported:

    ``"flux"``
        A specified heat flux profile q_w(z) [W/m^2] on the inner wall.  The
        inner wall surface temperature is then a model output -- this is the
        mode used by the optimiser, which treats the average flux as a decision
        variable.

    ``"wall_temperature"``
        A specified inner wall temperature profile T_w(z) [K], the quantity an
        operator actually measures with a pyrometer.  The flux becomes an
        output.
    """

    mode: Literal["flux", "wall_temperature"] = "flux"

    # --- flux mode ---------------------------------------------------------
    average_heat_flux: float = 5.8e4    # [W/m^2] area-average inner flux   [CHECK]
    flux_profile: Literal["uniform", "industrial", "custom"] = "industrial"
    #: Peak position of the "industrial" flux profile as a fraction of length.
    flux_peak_location: float = 0.25    # [-]                               [CORR]
    flux_peak_ratio: float = 1.45       # [-] peak/mean flux                [CORR]
    custom_flux: Optional[AxialProfile] = None

    # --- wall temperature mode --------------------------------------------
    wall_temperature_inlet: float = 1030.0   # [K]                          [CHECK]
    wall_temperature_outlet: float = 1180.0  # [K]                          [CHECK]
    custom_wall_temperature: Optional[AxialProfile] = None

    #: Metallurgical design limit on the inner tube wall temperature [K].
    wall_temperature_limit: float = 1150.0   #                              [CHECK]

    def heat_flux(self, z: np.ndarray, length: float) -> np.ndarray:
        """Inner-wall heat flux profile [W/m^2] at axial positions ``z``."""
        z = np.asarray(z, dtype=float)
        if self.mode != "flux":
            raise RuntimeError("heat_flux() is only defined in 'flux' mode.")
        if self.flux_profile == "custom":
            if self.custom_flux is None:
                raise ValueError("flux_profile='custom' requires custom_flux.")
            return np.asarray(self.custom_flux(z), dtype=float)
        if self.flux_profile == "uniform":
            return np.full_like(z, self.average_heat_flux)
        # "industrial": a skewed profile that rises quickly in the first metres
        # (where the endothermic duty is largest) then decays, normalised to the
        # requested area-average flux.
        shape = self._industrial_shape(z / length)
        return self.average_heat_flux * shape

    #: Sharpness of the "industrial" flux bump (higher = narrower peak).
    flux_peak_sharpness: float = 2.0
    #: Background level of the bump, which sets the inlet flux.
    flux_background: float = 0.55
    #: Exponential decay towards the outlet, where the gas nears equilibrium.
    flux_decay: float = 0.55

    def _industrial_shape(self, zeta: np.ndarray) -> np.ndarray:
        r"""Normalised flux shape on ``zeta = z/L``, with unit mean.

        Reproduces the shape of a measured top-fired reformer firing profile:
        a substantial but sub-average flux at the inlet, a maximum in the first
        quarter of the tube where the endothermic duty is largest, and a decay
        towards the outlet where the gas approaches equilibrium and absorbs
        less heat.  It is built from a gamma-like bump on a background, damped
        exponentially,

            g(zeta) = [c + (1-c) (zeta/zp)^a e^{a(1 - zeta/zp)}] e^{-d zeta}

        and then rescaled affinely about its own mean,

            shape = 1 + lambda (g - mean g),
            lambda = (peak_ratio - 1) / (max g - mean g),

        so that the profile has unit mean *and* a peak-to-mean ratio of exactly
        :attr:`flux_peak_ratio`, whatever the shape parameters.  The maximum
        sits slightly upstream of :attr:`flux_peak_location` because of the
        exponential damping.
        """
        zeta = np.clip(np.asarray(zeta, dtype=float), 0.0, 1.0)
        zp = float(np.clip(self.flux_peak_location, 0.02, 0.98))
        a = float(max(self.flux_peak_sharpness, 0.1))
        c = float(np.clip(self.flux_background, 0.0, 1.0))
        d = float(max(self.flux_decay, 0.0))

        def raw(x):
            ratio = np.clip(x / zp, 0.0, None)
            bump = ratio**a * np.exp(a * (1.0 - ratio))
            return (c + (1.0 - c) * bump) * np.exp(-d * x)

        zz = np.linspace(0.0, 1.0, 4001)
        g = raw(zz)
        mean_g, max_g = float(np.trapezoid(g, zz)), float(g.max())
        lam = (self.flux_peak_ratio - 1.0) / max(max_g - mean_g, 1.0e-6)
        return np.clip(1.0 + lam * (raw(zeta) - mean_g), 0.02, None)

    def wall_temperature(self, z: np.ndarray, length: float) -> np.ndarray:
        """Inner-wall temperature profile [K] at axial positions ``z``."""
        z = np.asarray(z, dtype=float)
        if self.mode != "wall_temperature":
            raise RuntimeError(
                "wall_temperature() is only defined in 'wall_temperature' mode.")
        if self.custom_wall_temperature is not None:
            return np.asarray(self.custom_wall_temperature(z), dtype=float)
        zeta = np.clip(z / length, 0.0, 1.0)
        # Smooth monotone ramp, steeper near the inlet where firing is hardest.
        return (self.wall_temperature_inlet
                + (self.wall_temperature_outlet - self.wall_temperature_inlet)
                * (1.0 - np.exp(-2.5 * zeta)) / (1.0 - np.exp(-2.5)))

    def total_duty(self, geometry: Geometry) -> float:
        """Total heat input over the tube [W] (flux mode only)."""
        z = np.linspace(0.0, geometry.length, 2001)
        q = self.heat_flux(z, geometry.length)
        return float(np.pi * geometry.inner_diameter * np.trapezoid(q, z))


# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------

@dataclass
class Numerics:
    """Discretisation and integrator settings."""

    n_radial: int = 12                  # radial finite-volume cells
    radial_grading: float = 1.0         # >1 clusters cells towards the wall
    n_axial_output: int = 273           # axial points stored in the solution
    method: str = "BDF"                 # scipy.integrate.solve_ivp method
    rtol: float = 1.0e-6
    max_step: float = np.inf

    # Absolute tolerances are set per state variable rather than globally: the
    # species fluxes are O(100) mol/(m^2 s) while the pressure is O(1e6) Pa, and
    # a single scalar atol either over-resolves the fluxes (which is what makes
    # a naive method-of-lines packed-bed model take minutes) or under-resolves
    # the trace species.  ``atol`` overrides all three when given.
    atol: Optional[float] = None
    atol_flux_relative: float = 1.0e-6   # x inlet total molar flux density
    atol_temperature: float = 1.0e-4     # [K]
    atol_pressure: float = 1.0           # [Pa]

    # Model switches -------------------------------------------------------
    radial_mass_dispersion: bool = True   # include D_er radial species mixing
    film_resistance: bool = True          # gas->solid film DT (pseudo-het.)
    film_iterations: int = 3              # Picard sweeps for the solid T
    radiation_in_bed: bool = True         # radiative term in k_er^0

    # Closure selections ---------------------------------------------------
    #: ``"correlation"`` uses the stagnant + turbulent k_er closure;
    #: ``"constant"`` uses :attr:`k_er_constant`.
    k_er_model: Literal["correlation", "constant"] = "correlation"
    k_er_constant: float = 10.0           # [W/(m K)]
    radial_peclet_heat: float = 8.0       # Pe_r,h asymptote               [CORR]
    radial_peclet_mass: float = 9.0       # Pe_r,m asymptote               [CORR]
    #: ``"li_finlayson"`` or ``"dixon"`` wall Nusselt correlation.
    wall_htc_model: Literal["li_finlayson", "dixon"] = "li_finlayson"

    # Numerical safeguards -------------------------------------------------
    p_h2_floor: float = 1.0e-4            # [bar] floor on p_H2 in the rates
    min_mole_fraction: float = 1.0e-12
    min_pressure: float = 1.0e5           # [Pa] abort below this


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class ReformerConfig:
    """Everything the solver needs, in one object."""

    geometry: Geometry = field(default_factory=Geometry)
    catalyst: Catalyst = field(default_factory=Catalyst)
    feed: Feed = field(default_factory=Feed)
    operating: Operating = field(default_factory=Operating)
    numerics: Numerics = field(default_factory=Numerics)
    name: str = "industrial SMR tube (Nobandegani et al. 2021 geometry)"

    # -- convenience constructors -----------------------------------------
    def with_feed(self, **kwargs) -> "ReformerConfig":
        """Copy with modified :class:`Feed` fields."""
        return replace(self, feed=replace(self.feed, **kwargs))

    def with_operating(self, **kwargs) -> "ReformerConfig":
        """Copy with modified :class:`Operating` fields."""
        return replace(self, operating=replace(self.operating, **kwargs))

    def with_numerics(self, **kwargs) -> "ReformerConfig":
        """Copy with modified :class:`Numerics` fields."""
        return replace(self, numerics=replace(self.numerics, **kwargs))

    def with_catalyst(self, **kwargs) -> "ReformerConfig":
        """Copy with modified :class:`Catalyst` fields."""
        return replace(self, catalyst=replace(self.catalyst, **kwargs))

    # -- derived quantities ------------------------------------------------
    @property
    def mass_flux(self) -> float:
        """Superficial mass flux G [kg/(m^2 s)]."""
        return self.feed.mass_flow() / self.geometry.cross_section

    def describe(self) -> str:
        """Human-readable summary of the configuration."""
        g, c, f, o = self.geometry, self.catalyst, self.feed, self.operating
        lines = [
            f"Configuration: {self.name}",
            "-" * 64,
            f"  Tube length / inner diameter : {g.length:.2f} m / "
            f"{g.inner_diameter * 1e3:.1f} mm",
            f"  Bed voidage / particle dia.  : {c.void_fraction:.3f} / "
            f"{c.particle_diameter * 1e3:.1f} mm",
            f"  Bed bulk density             : {c.bulk_density:.0f} kg/m3 "
            f"({c.bulk_density * g.bed_volume:.1f} kg per tube)",
            f"  Feed  T / P                  : {f.inlet_temperature:.1f} K / "
            f"{f.inlet_pressure / 1e5:.2f} bar",
            f"  Feed  total / CH4 flow       : {f.molar_flow:.3f} / "
            f"{f.molar_flows()[IDX['CH4']]:.3f} mol/s",
            f"  Steam-to-carbon ratio        : {f.steam_to_carbon:.2f}",
            f"  Superficial mass flux G      : {self.mass_flux:.2f} kg/(m2 s)",
        ]
        if o.mode == "flux":
            lines += [
                f"  Wall BC                      : specified flux "
                f"({o.flux_profile}), mean {o.average_heat_flux / 1e3:.1f} kW/m2",
                f"  Total duty per tube          : "
                f"{o.total_duty(g) / 1e3:.1f} kW",
            ]
        else:
            lines += [
                f"  Wall BC                      : specified wall temperature "
                f"{o.wall_temperature_inlet:.0f} -> "
                f"{o.wall_temperature_outlet:.0f} K",
            ]
        lines += [
            f"  Radial cells / integrator    : {self.numerics.n_radial} / "
            f"{self.numerics.method}",
            f"  Effectiveness factor model   : {c.effectiveness_model}",
        ]
        return "\n".join(lines)


def default_config() -> ReformerConfig:
    """Return the baseline industrial configuration."""
    return ReformerConfig()
