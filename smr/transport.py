r"""
Bed-scale transport closures for the packed reformer tube.

Pressure drop
-------------
Ergun (1952), written for superficial velocity ``u_s``:

    -dP/dz = 150 mu (1-eps)^2 / (eps^3 d_p^2) u_s
             + 1.75 rho (1-eps) / (eps^3 d_p) u_s^2

Effective radial thermal conductivity
-------------------------------------
Additive stagnant + flow contributions, the standard packed-bed form used in
two-dimensional reformer models:

    k_er = k_er^0 + k_g Re Pr / Pe_r
    Pe_r = Pe_r,inf (1 + 46 (d_p/d_t)^2)          Bauer & Schluender

with the stagnant conductivity from the Kunii-Smith form plus a void radiation
term that matters at reformer temperatures:

    k_er^0 = k_g [eps + (1-eps)/(phi_k + (2/3) k_g/k_s)] + k_rad
    k_rad  = 4 sigma (e/(2-e)) T^3 d_p

Wall heat transfer
------------------
Inner-wall film coefficient from Li & Finlayson (1977) for gases in packed
tubes, or Dixon's correlation as an alternative:

    Nu_w = 0.17 Re^0.79                           Li & Finlayson
    Nu_w = (1 - 1.5 (d_t/d_p)^-1.5) Re^0.59 Pr^(1/3)   Dixon

Gas-solid film heat transfer
----------------------------
Wakao & Kaguei (1982):

    Nu_p = 2 + 1.1 Re^0.6 Pr^(1/3)

All Reynolds numbers are particle Reynolds numbers ``Re = G d_p / mu`` built on
the superficial mass flux ``G``.
"""

from __future__ import annotations

import numpy as np

from .config import Catalyst, Geometry, Numerics

__all__ = [
    "STEFAN_BOLTZMANN",
    "particle_reynolds",
    "prandtl",
    "ergun_pressure_gradient",
    "stagnant_conductivity",
    "effective_radial_conductivity",
    "effective_radial_diffusivity",
    "wall_heat_transfer_coefficient",
    "film_heat_transfer_coefficient",
    "film_mass_transfer_coefficient",
    "tube_wall_temperature_drop",
]

STEFAN_BOLTZMANN = 5.670374419e-8  # [W/(m^2 K^4)]


def particle_reynolds(mass_flux: float | np.ndarray, particle_diameter: float,
                      viscosity: float | np.ndarray) -> float | np.ndarray:
    """Particle Reynolds number ``Re = G d_p / mu`` [-]."""
    return mass_flux * particle_diameter / viscosity


def prandtl(cp_mass: float | np.ndarray, viscosity: float | np.ndarray,
            conductivity: float | np.ndarray) -> float | np.ndarray:
    """Prandtl number ``Pr = cp mu / k`` [-] with ``cp`` per unit mass."""
    return cp_mass * viscosity / conductivity


def ergun_pressure_gradient(u_s: float, rho: float, mu: float,
                            catalyst: Catalyst) -> float:
    """Axial pressure gradient ``dP/dz`` [Pa/m] (negative for flow in +z).

    Parameters
    ----------
    u_s : float
        Superficial velocity [m/s].
    rho : float
        Gas mass density [kg/m^3].
    mu : float
        Gas viscosity [Pa s].
    """
    eps = catalyst.void_fraction
    d_p = catalyst.particle_diameter
    viscous = 150.0 * mu * (1.0 - eps)**2 / (eps**3 * d_p**2) * u_s
    inertial = 1.75 * rho * (1.0 - eps) / (eps**3 * d_p) * u_s**2
    return -(viscous + inertial)


def stagnant_conductivity(T: float | np.ndarray, k_gas: float | np.ndarray,
                          catalyst: Catalyst,
                          include_radiation: bool = True) -> float | np.ndarray:
    """Stagnant (no-flow) effective bed conductivity ``k_er^0`` [W/(m K)]."""
    eps = catalyst.void_fraction
    phi_k = 0.13  # Kunii-Smith particle-contact parameter for packed spheres
    conduction = k_gas * (eps + (1.0 - eps)
                          / (phi_k + (2.0 / 3.0) * k_gas / catalyst.solid_conductivity))
    if not include_radiation:
        return conduction
    e = catalyst.emissivity
    k_rad = (4.0 * STEFAN_BOLTZMANN * (e / (2.0 - e))
             * np.asarray(T, dtype=float)**3 * catalyst.particle_diameter)
    return conduction + k_rad


def effective_radial_conductivity(T, k_gas, re_p, pr, catalyst: Catalyst,
                                  geometry: Geometry, numerics: Numerics):
    """Effective radial thermal conductivity ``k_er`` [W/(m K)]."""
    if numerics.k_er_model == "constant":
        return np.full_like(np.asarray(T, dtype=float), numerics.k_er_constant)
    k0 = stagnant_conductivity(T, k_gas, catalyst,
                               include_radiation=numerics.radiation_in_bed)
    aspect = catalyst.particle_diameter / geometry.inner_diameter
    pe_r = numerics.radial_peclet_heat * (1.0 + 46.0 * aspect**2)
    return k0 + k_gas * re_p * pr / pe_r


def effective_radial_diffusivity(u_s: float, catalyst: Catalyst,
                                 geometry: Geometry, numerics: Numerics) -> float:
    """Effective radial mass dispersion coefficient ``D_er`` [m^2/s].

    ``D_er = u_s d_p / (Pe_r,m (1 + 19.4 (d_p/d_t)^2))`` on the *superficial*
    velocity, which is the Froment & Bischoff convention for the radial mass
    Peclet number; the aspect-ratio correction is the mass-transfer analogue of
    the Bauer-Schluender wall damping.  Returns 0 when radial dispersion is
    switched off.
    """
    if not numerics.radial_mass_dispersion:
        return 0.0
    aspect = catalyst.particle_diameter / geometry.inner_diameter
    pe_m = numerics.radial_peclet_mass * (1.0 + 19.4 * aspect**2)
    return u_s * catalyst.particle_diameter / pe_m


def wall_heat_transfer_coefficient(k_gas, re_p, pr, catalyst: Catalyst,
                                   geometry: Geometry, numerics: Numerics):
    """Inner-wall film heat transfer coefficient ``h_w`` [W/(m^2 K)]."""
    d_p = catalyst.particle_diameter
    re_p = np.clip(np.asarray(re_p, dtype=float), 1.0, None)
    if numerics.wall_htc_model == "dixon":
        ratio = geometry.inner_diameter / d_p
        nu_w = ((1.0 - 1.5 * ratio**-1.5) * re_p**0.59
                * np.asarray(pr, dtype=float)**(1.0 / 3.0))
    else:  # Li & Finlayson
        nu_w = 0.17 * re_p**0.79
    return np.clip(nu_w, 1.0, None) * k_gas / d_p


def film_heat_transfer_coefficient(k_gas, re_p, pr, catalyst: Catalyst):
    """Gas-to-pellet film heat transfer coefficient ``h_f`` [W/(m^2 K)]
    from the Wakao & Kaguei correlation."""
    re_p = np.clip(np.asarray(re_p, dtype=float), 1.0e-3, None)
    nu_p = 2.0 + 1.1 * re_p**0.6 * np.asarray(pr, dtype=float)**(1.0 / 3.0)
    return nu_p * k_gas / catalyst.particle_diameter


def film_mass_transfer_coefficient(d_mix, re_p, sc, catalyst: Catalyst):
    """Gas-to-pellet film mass transfer coefficient ``k_f`` [m/s]
    from the Wakao & Kaguei analogue, ``Sh = 2 + 1.1 Re^0.6 Sc^(1/3)``."""
    re_p = np.clip(np.asarray(re_p, dtype=float), 1.0e-3, None)
    sh = 2.0 + 1.1 * re_p**0.6 * np.asarray(sc, dtype=float)**(1.0 / 3.0)
    return sh * d_mix / catalyst.particle_diameter


def tube_wall_temperature_drop(q_wall, geometry: Geometry):
    """Temperature rise across the tube metal [K] for a given inner flux.

    Uses the cylindrical-shell conduction resistance, referenced to the inner
    surface flux ``q_wall`` [W/m^2]:

        dT = q_w r_i ln(r_o/r_i) / k_wall
    """
    r_i = geometry.radius
    r_o = r_i + geometry.wall_thickness
    return (np.asarray(q_wall, dtype=float) * r_i * np.log(r_o / r_i)
            / geometry.wall_conductivity)
