r"""
Intrinsic steam-reforming kinetics of Xu & Froment (1989) for Ni/MgAl2O4.

Three independent reactions are used:

=== ============================================ ================
  j  reaction                                      dH_298 [kJ/mol]
=== ============================================ ================
  1  CH4 + H2O  <-> CO  + 3 H2                      +206
  2  CO  + H2O  <-> CO2 +   H2                       -41
  3  CH4 + 2H2O <-> CO2 + 4 H2                      +165
=== ============================================ ================

Langmuir-Hinshelwood-Hougen-Watson rate equations, with partial pressures in
bar and rates in kmol/(kg_cat h) -- the units of the original paper.  The module
also exposes an SI wrapper returning mol/(kg_cat s).

    DEN = 1 + K_CO p_CO + K_H2 p_H2 + K_CH4 p_CH4 + K_H2O p_H2O / p_H2

    r1 = (k1 / p_H2^2.5) (p_CH4 p_H2O   - p_H2^3 p_CO  / K1) / DEN^2
    r2 = (k2 / p_H2    ) (p_CO  p_H2O   - p_H2   p_CO2 / K2) / DEN^2
    r3 = (k3 / p_H2^3.5) (p_CH4 p_H2O^2 - p_H2^4 p_CO2 / K3) / DEN^2

Arrhenius parameters (Xu & Froment 1989, Tables 3 and 4):

    k1 = 4.225e15 exp(-240100 / R T)   kmol bar^0.5 / (kg_cat h)
    k2 = 1.955e06 exp(-067130 / R T)   kmol / (kg_cat h bar)
    k3 = 1.020e15 exp(-243900 / R T)   kmol bar^0.5 / (kg_cat h)

    K_CH4 = 6.65e-04 exp(+38280 / R T)   bar^-1
    K_H2O = 1.77e+05 exp(-88680 / R T)   -
    K_CO  = 8.23e-05 exp(+70650 / R T)   bar^-1
    K_H2  = 6.12e-09 exp(+82900 / R T)   bar^-1

Equilibrium constants use the Twigg (1989) correlations, which are the ones
normally paired with this rate set:

    K1 = exp(30.114 - 26830 / T)  bar^2        (= 1.198e17 exp(-26830/T) kPa^2)
    K2 = exp(4400 / T - 4.036)    -
    K3 = K1 K2                    bar^2

:func:`equilibrium_constants_from_thermo` recomputes K1 and K2 from the NASA
polynomials in :mod:`smr.species` and is used by the test suite to confirm the
two routes agree to within a few per cent over 700-1300 K.
"""

from __future__ import annotations

import numpy as np

from .species import (
    IDX,
    N_SPECIES,
    R_GAS,
    enthalpy_molar,
    gibbs_molar_standard,
)

__all__ = [
    "N_REACTIONS",
    "REACTION_NAMES",
    "STOICHIOMETRY",
    "rate_constants",
    "adsorption_constants",
    "equilibrium_constants",
    "equilibrium_constants_from_thermo",
    "reaction_rates",
    "reaction_rates_si",
    "reaction_enthalpies",
    "approach_to_equilibrium",
]

N_REACTIONS = 3

REACTION_NAMES = (
    "R1: CH4 + H2O <-> CO + 3H2 (reforming)",
    "R2: CO + H2O <-> CO2 + H2 (water-gas shift)",
    "R3: CH4 + 2H2O <-> CO2 + 4H2 (overall reforming)",
)

# Stoichiometric matrix nu[j, i]: species i produced per unit extent of j.
STOICHIOMETRY = np.zeros((N_REACTIONS, N_SPECIES))
STOICHIOMETRY[0, [IDX["CH4"], IDX["H2O"], IDX["CO"], IDX["H2"]]] = [-1, -1, +1, +3]
STOICHIOMETRY[1, [IDX["CO"], IDX["H2O"], IDX["CO2"], IDX["H2"]]] = [-1, -1, +1, +1]
STOICHIOMETRY[2, [IDX["CH4"], IDX["H2O"], IDX["CO2"], IDX["H2"]]] = [-1, -2, +1, +4]

# --- Arrhenius / van't Hoff parameters -------------------------------------
_K_PRE = np.array([4.225e15, 1.955e06, 1.020e15])
_K_ACT = np.array([240100.0, 67130.0, 243900.0])          # [J/mol]

_ADS_PRE = {"CH4": 6.65e-04, "H2O": 1.77e05, "CO": 8.23e-05, "H2": 6.12e-09}
_ADS_DH = {"CH4": -38280.0, "H2O": 88680.0, "CO": -70650.0, "H2": -82900.0}

# Conversion from kmol/(kg h) to mol/(kg s).
KMOL_PER_H_TO_MOL_PER_S = 1000.0 / 3600.0


def rate_constants(T):
    """Forward rate constants ``(k1, k2, k3)`` at temperature ``T`` [K].

    Units: k1, k3 in kmol bar^0.5/(kg_cat h); k2 in kmol/(kg_cat h bar).
    Returns an array of shape ``(..., 3)``.
    """
    T = np.asarray(T, dtype=float)[..., None]
    return _K_PRE * np.exp(-_K_ACT / (R_GAS * T))


def adsorption_constants(T):
    """Adsorption equilibrium constants as a dict of arrays.

    ``K_CH4``, ``K_CO`` and ``K_H2`` are in bar^-1; ``K_H2O`` is dimensionless.
    """
    T = np.asarray(T, dtype=float)
    return {
        key: _ADS_PRE[key] * np.exp(-_ADS_DH[key] / (R_GAS * T))
        for key in ("CH4", "H2O", "CO", "H2")
    }


def equilibrium_constants(T):
    """Reaction equilibrium constants ``(K1, K2, K3)`` from Twigg's correlations.

    K1 and K3 are in bar^2, K2 is dimensionless.  Shape ``(..., 3)``.
    """
    T = np.asarray(T, dtype=float)
    K1 = np.exp(30.114 - 26830.0 / T)
    K2 = np.exp(4400.0 / T - 4.036)
    return np.stack([K1, K2, K1 * K2], axis=-1)


def equilibrium_constants_from_thermo(T):
    """Equilibrium constants recomputed from the NASA-7 polynomials.

    Provided as an independent cross-check of :func:`equilibrium_constants`.
    Units match that function (bar^2, -, bar^2); the pressure-reference
    conversion uses 1 bar = 1e5 Pa.
    """
    T = np.asarray(T, dtype=float)
    g = gibbs_molar_standard(T)                       # (..., nsp) at 1 bar
    dg = np.einsum("ji,...i->...j", STOICHIOMETRY, g)  # (..., 3)
    dn = STOICHIOMETRY.sum(axis=1)                    # (3,) = [2, 0, 2]
    # K_p in bar^dn follows directly since the standard state is 1 bar.
    return np.exp(-dg / (R_GAS * T[..., None]))


def reaction_enthalpies(T):
    """Reaction enthalpies ``(dH1, dH2, dH3)`` [J/mol] at ``T`` [K].

    Computed from the NASA-7 polynomials so that the energy balance and the
    equilibrium thermodynamics are mutually consistent.  Shape ``(..., 3)``.
    """
    h = enthalpy_molar(T)                              # (..., nsp)
    return np.einsum("ji,...i->...j", STOICHIOMETRY, h)


def reaction_rates(T, p_bar, p_h2_floor: float = 1.0e-4, return_den: bool = False,
                   return_forward: bool = False):
    """Intrinsic LHHW reaction rates in kmol/(kg_cat h).

    Parameters
    ----------
    T : array_like
        Temperature [K], shape ``(...)``.
    p_bar : array_like
        Partial pressures [bar], shape ``(..., nsp)`` in
        :data:`smr.species.SPECIES` order.
    p_h2_floor : float
        Lower bound applied to p_H2 before it appears in a denominator.  The
        rate law is singular at p_H2 -> 0; industrial feeds always carry some
        hydrogen, and the floor only guards the integrator's trial steps.
    return_den : bool
        Also return the LHHW denominator (useful for diagnostics).
    return_forward : bool
        Also return the forward (reverse-free) rates, i.e. the same expressions
        with the ``-p_H2^a p_x / K_j`` term dropped.  These are what the pellet
        model needs: the Thiele modulus of a *reversible* reaction is built on
        the sum of the forward and reverse rate coefficients, which stays
        finite at equilibrium, whereas the net rate vanishes there.  Using the
        net rate would make the effectiveness factor tend to one exactly where
        an industrial reformer spends most of its length.

    Returns
    -------
    r : ndarray, shape ``(..., 3)``
        Signed reaction rates; negative means the reaction runs in reverse.
    """
    T = np.asarray(T, dtype=float)
    p = np.clip(np.asarray(p_bar, dtype=float), 0.0, None)

    p_ch4 = p[..., IDX["CH4"]]
    p_h2o = p[..., IDX["H2O"]]
    p_h2 = np.clip(p[..., IDX["H2"]], p_h2_floor, None)
    p_co = p[..., IDX["CO"]]
    p_co2 = p[..., IDX["CO2"]]

    k = rate_constants(T)
    K = equilibrium_constants(T)
    Kads = adsorption_constants(T)

    den = (1.0
           + Kads["CO"] * p_co
           + Kads["H2"] * p_h2
           + Kads["CH4"] * p_ch4
           + Kads["H2O"] * p_h2o / p_h2)
    den2 = den * den

    drive1 = p_ch4 * p_h2o - p_h2**3 * p_co / K[..., 0]
    drive2 = p_co * p_h2o - p_h2 * p_co2 / K[..., 1]
    drive3 = p_ch4 * p_h2o**2 - p_h2**4 * p_co2 / K[..., 2]

    r1 = k[..., 0] / p_h2**2.5 * drive1 / den2
    r2 = k[..., 1] / p_h2 * drive2 / den2
    r3 = k[..., 2] / p_h2**3.5 * drive3 / den2

    r = np.stack([r1, r2, r3], axis=-1)

    out = [r]
    if return_den:
        out.append(den)
    if return_forward:
        f1 = k[..., 0] / p_h2**2.5 * (p_ch4 * p_h2o) / den2
        f2 = k[..., 1] / p_h2 * (p_co * p_h2o) / den2
        f3 = k[..., 2] / p_h2**3.5 * (p_ch4 * p_h2o**2) / den2
        out.append(np.stack([f1, f2, f3], axis=-1))
    return out[0] if len(out) == 1 else tuple(out)


def reaction_rates_si(T, p_bar, p_h2_floor: float = 1.0e-4,
                      return_forward: bool = False):
    """Intrinsic reaction rates in mol/(kg_cat s).

    With ``return_forward`` a ``(net, forward)`` tuple is returned.
    """
    if return_forward:
        r, f = reaction_rates(T, p_bar, p_h2_floor, return_forward=True)
        return r * KMOL_PER_H_TO_MOL_PER_S, f * KMOL_PER_H_TO_MOL_PER_S
    return reaction_rates(T, p_bar, p_h2_floor) * KMOL_PER_H_TO_MOL_PER_S


def approach_to_equilibrium(T, p_bar, p_h2_floor: float = 1.0e-4):
    """Ratio ``beta_j = Q_j / K_j`` for the three reactions.

    ``beta = 0`` at the pure-reactant limit and ``beta = 1`` at equilibrium;
    ``beta > 1`` means the reaction is running in reverse.  Dimensionless,
    shape ``(..., 3)``.
    """
    T = np.asarray(T, dtype=float)
    p = np.clip(np.asarray(p_bar, dtype=float), 1.0e-12, None)
    p_h2 = np.clip(p[..., IDX["H2"]], p_h2_floor, None)
    K = equilibrium_constants(T)
    Q1 = p_h2**3 * p[..., IDX["CO"]] / (p[..., IDX["CH4"]] * p[..., IDX["H2O"]])
    Q2 = p_h2 * p[..., IDX["CO2"]] / (p[..., IDX["CO"]] * p[..., IDX["H2O"]])
    Q3 = p_h2**4 * p[..., IDX["CO2"]] / (p[..., IDX["CH4"]] * p[..., IDX["H2O"]]**2)
    return np.stack([Q1 / K[..., 0], Q2 / K[..., 1], Q3 / K[..., 2]], axis=-1)
