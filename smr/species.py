r"""
Thermochemistry and transport properties for the SMR gas mixture.

Thermochemistry uses NASA 7-coefficient polynomials (GRI-Mech 3.0 set) for the
gas species, so that heat capacities, enthalpies, entropies, reaction enthalpies
and equilibrium constants are all derived from one consistent data set:

    cp/R  = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4
    h/RT  = a0 + a1 T/2 + a2 T^2/3 + a3 T^3/4 + a4 T^4/5 + a5/T
    s/R   = a0 ln T + a1 T + a2 T^2/2 + a3 T^3/3 + a4 T^4/4 + a6

Solid carbon (graphite) is handled separately -- it is not part of the GRI-Mech
set -- using Kelley's heat-capacity equation together with the standard-state
reference values h_f(298.15) = 0 and s(298.15) = 5.74 J/mol/K.  Graphite is
needed only by :mod:`smr.coking`.

Transport properties:
  * pure-component viscosity from Chapman-Enskog with the Neufeld collision
    integral, mixed with Wilke's rule;
  * pure-component conductivity from the modified Eucken relation, mixed with
    the Mason-Saxena (Wassiljewa) rule;
  * binary diffusivities from Fuller-Schettler-Giddings, mixed with Wilke's
    dilute rule to give an effective diffusivity of each species in the mixture.

All public functions take SI units unless the name says otherwise
(T in K, P in Pa, lengths in m, amounts in mol).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants and species bookkeeping
# ---------------------------------------------------------------------------

R_GAS = 8.314462618  # J/(mol K)
P_REF = 1.0e5  # Pa, reference pressure for standard-state entropies
T_REF = 298.15  # K

#: Gas species carried by the model, in state-vector order.
SPECIES = ("CH4", "H2O", "H2", "CO", "CO2", "N2")
N_SPECIES = len(SPECIES)
IDX = {name: i for i, name in enumerate(SPECIES)}

#: Molar masses [kg/mol].
MW = np.array([0.0160428, 0.0180153, 0.0020159, 0.0280101, 0.0440095, 0.0280134])

# NASA-7 polynomial coefficients (GRI-Mech 3.0): (T_switch, low[7], high[7]).
# Low range 200-1000 K, high range 1000-3500 K.
_NASA7 = {
    "CH4": (
        1000.0,
        (5.14987613e00, -1.36709788e-02, 4.91800599e-05, -4.84743026e-08,
         1.66693956e-11, -1.02466476e04, -4.64130376e00),
        (7.48514950e-02, 1.33909467e-02, -5.73285809e-06, 1.22292535e-09,
         -1.01815230e-13, -9.46834459e03, 1.84373180e01),
    ),
    "H2O": (
        1000.0,
        (4.19864056e00, -2.03643410e-03, 6.52040211e-06, -5.48797062e-09,
         1.77197817e-12, -3.02937267e04, -8.49032208e-01),
        (3.03399249e00, 2.17691804e-03, -1.64072518e-07, -9.70419870e-11,
         1.68200992e-14, -3.00042971e04, 4.96677010e00),
    ),
    "H2": (
        1000.0,
        (2.34433112e00, 7.98052075e-03, -1.94781510e-05, 2.01572094e-08,
         -7.37611761e-12, -9.17935173e02, 6.83010238e-01),
        (3.33727920e00, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
         2.00255376e-14, -9.50158922e02, -3.20502331e00),
    ),
    "CO": (
        1000.0,
        (3.57953347e00, -6.10353680e-04, 1.01681433e-06, 9.07005884e-10,
         -9.04424499e-13, -1.43440860e04, 3.50840928e00),
        (2.71518561e00, 2.06252743e-03, -9.98825771e-07, 2.30053008e-10,
         -2.03647716e-14, -1.41518724e04, 7.81868772e00),
    ),
    "CO2": (
        1000.0,
        (2.35677352e00, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09,
         -1.43699548e-13, -4.83719697e04, 9.90105222e00),
        (3.85746029e00, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10,
         -4.72084164e-14, -4.87591660e04, 2.27163806e00),
    ),
    "N2": (
        1000.0,
        (3.29867700e00, 1.40824040e-03, -3.96322200e-06, 5.64151500e-09,
         -2.44485400e-12, -1.02089990e03, 3.95037200e00),
        (2.92664000e00, 1.48797680e-03, -5.68476000e-07, 1.00970380e-10,
         -6.75335100e-15, -9.22797700e02, 5.98052800e00),
    ),
}

# Pack coefficients into arrays indexed like SPECIES for vectorised evaluation.
_T_SWITCH = np.array([_NASA7[s][0] for s in SPECIES])
_A_LOW = np.array([_NASA7[s][1] for s in SPECIES])   # (nsp, 7)
_A_HIGH = np.array([_NASA7[s][2] for s in SPECIES])  # (nsp, 7)

# Lennard-Jones parameters (Reid, Prausnitz & Poling): sigma [Angstrom], eps/k [K].
_LJ_SIGMA = np.array([3.758, 2.641, 2.827, 3.690, 3.941, 3.798])
_LJ_EPS_K = np.array([148.6, 809.1, 59.7, 91.7, 195.2, 71.4])

# Fuller-Schettler-Giddings atomic diffusion volumes [cm^3/mol].
_FULLER_V = np.array([24.42, 12.70, 6.12, 18.00, 26.90, 18.50])

# Linear/non-linear flag for the modified Eucken correction (unused placeholder
# kept explicit for clarity: all species here are treated as polyatomic).
_IS_MONATOMIC = np.zeros(N_SPECIES, dtype=bool)


# ---------------------------------------------------------------------------
# Ideal-gas thermochemistry
# ---------------------------------------------------------------------------

#: Half-width [K] of the blend window around the NASA polynomial switch.
_BLEND_HALF_WIDTH = 20.0


def _nasa_coeffs(T):
    """Select low/high NASA coefficient sets for scalar or array ``T``.

    Returns an array of shape ``(..., nsp, 7)``.  Temperatures are clipped to
    200-3500 K; the reformer never leaves 500-1500 K but the integrator can
    probe outside the physical range during step-size selection.

    The two coefficient sets are blended with a smoothstep weight over a
    +/-20 K window around the 1000 K switch instead of being swapped
    discontinuously.  The GRI-Mech sets are continuous in cp at the switch but
    not differentiable there, and 1000 K sits in the middle of a reformer's
    operating range: an un-smoothed switch puts a kink in the right-hand side
    at whichever axial position each radial node crosses 1000 K, which forces a
    stiff integrator to cut its step at a dozen separate places.  The two sets
    agree to about 0.01% in cp, h and s at the switch, so the blend costs no
    accuracy.
    """
    T = np.clip(np.asarray(T, dtype=float), 200.0, 3500.0)
    x = ((T[..., None] - (_T_SWITCH - _BLEND_HALF_WIDTH))
         / (2.0 * _BLEND_HALF_WIDTH))
    x = np.clip(x, 0.0, 1.0)
    w = (x * x * (3.0 - 2.0 * x))[..., None]          # smoothstep, C1 at 0 and 1
    return (1.0 - w) * _A_LOW + w * _A_HIGH, T


def cp_molar(T):
    """Molar isobaric heat capacity of each species [J/(mol K)].

    Shape: ``(..., nsp)`` following the shape of ``T``.
    """
    a, T = _nasa_coeffs(T)
    Tv = T[..., None]
    cp_over_R = (a[..., 0] + a[..., 1] * Tv + a[..., 2] * Tv**2
                 + a[..., 3] * Tv**3 + a[..., 4] * Tv**4)
    return R_GAS * cp_over_R


def enthalpy_molar(T):
    """Molar enthalpy of each species [J/mol], referenced to the elements at
    298.15 K (i.e. it includes the enthalpy of formation)."""
    a, T = _nasa_coeffs(T)
    Tv = T[..., None]
    h_over_RT = (a[..., 0] + a[..., 1] * Tv / 2.0 + a[..., 2] * Tv**2 / 3.0
                 + a[..., 3] * Tv**3 / 4.0 + a[..., 4] * Tv**4 / 5.0
                 + a[..., 5] / Tv)
    return R_GAS * Tv * h_over_RT


def entropy_molar_standard(T):
    """Standard-state (1 bar, pure) molar entropy of each species [J/(mol K)]."""
    a, T = _nasa_coeffs(T)
    Tv = T[..., None]
    s_over_R = (a[..., 0] * np.log(Tv) + a[..., 1] * Tv
                + a[..., 2] * Tv**2 / 2.0 + a[..., 3] * Tv**3 / 3.0
                + a[..., 4] * Tv**4 / 4.0 + a[..., 6])
    return R_GAS * s_over_R


def gibbs_molar_standard(T):
    """Standard-state molar Gibbs energy of each species [J/mol]."""
    T = np.asarray(T, dtype=float)
    return enthalpy_molar(T) - T[..., None] * entropy_molar_standard(T)


# ---------------------------------------------------------------------------
# Solid carbon (graphite)
# ---------------------------------------------------------------------------

#: Standard entropy of graphite at 298.15 K [J/(mol K)] (CODATA).
S_GRAPHITE_298 = 5.74
#: Enthalpy of formation of graphite is zero by definition of the reference state.
H_GRAPHITE_298 = 0.0

# Kelley's equation for graphite, converted from cal to J:
#   cp = 16.86 + 4.77e-3 T - 8.53e5 / T^2   [J/(mol K)], valid ~300-1500 K.
_C_KELLEY = (16.86, 4.77e-3, -8.53e5)


def cp_graphite(T):
    """Heat capacity of graphite [J/(mol K)] from Kelley's equation."""
    T = np.clip(np.asarray(T, dtype=float), 250.0, 2000.0)
    a, b, c = _C_KELLEY
    return a + b * T + c / T**2


def enthalpy_graphite(T):
    """Molar enthalpy of graphite [J/mol] relative to the 298.15 K reference."""
    T = np.clip(np.asarray(T, dtype=float), 250.0, 2000.0)
    a, b, c = _C_KELLEY
    # Integral of cp dT from T_REF to T.
    return (H_GRAPHITE_298
            + a * (T - T_REF)
            + 0.5 * b * (T**2 - T_REF**2)
            - c * (1.0 / T - 1.0 / T_REF))


def entropy_graphite(T):
    """Standard molar entropy of graphite [J/(mol K)]."""
    T = np.clip(np.asarray(T, dtype=float), 250.0, 2000.0)
    a, b, c = _C_KELLEY
    # Integral of cp/T dT from T_REF to T.
    return (S_GRAPHITE_298
            + a * np.log(T / T_REF)
            + b * (T - T_REF)
            - 0.5 * c * (1.0 / T**2 - 1.0 / T_REF**2))


def gibbs_graphite(T):
    """Standard molar Gibbs energy of graphite [J/mol]."""
    T = np.asarray(T, dtype=float)
    return enthalpy_graphite(T) - T * entropy_graphite(T)


# ---------------------------------------------------------------------------
# Mixture thermodynamic properties
# ---------------------------------------------------------------------------

def mixture_molar_mass(y):
    """Mixture molar mass [kg/mol] from mole fractions ``y`` of shape (..., nsp)."""
    return np.sum(np.asarray(y) * MW, axis=-1)


def mixture_cp_molar(T, y):
    """Mixture molar heat capacity [J/(mol K)]."""
    return np.sum(cp_molar(T) * np.asarray(y), axis=-1)


def mixture_cp_mass(T, y):
    """Mixture specific heat capacity [J/(kg K)]."""
    return mixture_cp_molar(T, y) / mixture_molar_mass(y)


def gas_density(T, P, y):
    """Ideal-gas mass density [kg/m^3]."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    return P * mixture_molar_mass(y) / (R_GAS * T)


def molar_density(T, P):
    """Ideal-gas molar density [mol/m^3]."""
    return np.asarray(P, dtype=float) / (R_GAS * np.asarray(T, dtype=float))


# ---------------------------------------------------------------------------
# Transport properties
# ---------------------------------------------------------------------------

def _omega_22(T_star):
    """Neufeld correlation for the reduced collision integral Omega(2,2)*."""
    T_star = np.clip(T_star, 0.3, 100.0)
    return (1.16145 * T_star**-0.14874
            + 0.52487 * np.exp(-0.77320 * T_star)
            + 2.16178 * np.exp(-2.43787 * T_star))


def pure_viscosity(T):
    """Pure-component viscosity [Pa s] from Chapman-Enskog kinetic theory.

    mu = 2.6693e-6 * sqrt(M[g/mol] * T) / (sigma[A]^2 * Omega(2,2)*)
    """
    T = np.asarray(T, dtype=float)[..., None]
    T_star = T / _LJ_EPS_K
    mw_g = MW * 1.0e3
    return 2.6693e-6 * np.sqrt(mw_g * T) / (_LJ_SIGMA**2 * _omega_22(T_star))


def mixture_viscosity(T, y):
    """Mixture viscosity [Pa s] using Wilke's rule."""
    y = np.clip(np.asarray(y, dtype=float), 1.0e-12, None)
    mu = pure_viscosity(T)                      # (..., nsp)
    mu_i = mu[..., :, None]                     # (..., nsp, 1)
    mu_j = mu[..., None, :]                     # (..., 1, nsp)
    m_ratio = MW[None, :] / MW[:, None]         # M_j / M_i
    phi = (1.0 + np.sqrt(mu_i / mu_j) * m_ratio**0.25)**2 \
        / np.sqrt(8.0 * (1.0 + 1.0 / m_ratio))
    denom = np.sum(y[..., None, :] * phi, axis=-1)
    return np.sum(y * mu / denom, axis=-1)


def pure_thermal_conductivity(T):
    """Pure-component thermal conductivity [W/(m K)] from the modified Eucken
    relation, k = mu/M * (1.32 cv + 1.77 R)."""
    mu = pure_viscosity(T)
    cv = cp_molar(T) - R_GAS
    return mu / MW * (1.32 * cv + 1.77 * R_GAS)


def mixture_thermal_conductivity(T, y):
    """Mixture thermal conductivity [W/(m K)] using the Mason-Saxena rule."""
    y = np.clip(np.asarray(y, dtype=float), 1.0e-12, None)
    k = pure_thermal_conductivity(T)
    mu = pure_viscosity(T)
    mu_i = mu[..., :, None]
    mu_j = mu[..., None, :]
    m_ratio = MW[None, :] / MW[:, None]
    phi = (1.0 + np.sqrt(mu_i / mu_j) * m_ratio**0.25)**2 \
        / np.sqrt(8.0 * (1.0 + 1.0 / m_ratio))
    denom = np.sum(y[..., None, :] * phi, axis=-1)
    return np.sum(y * k / denom, axis=-1)


def binary_diffusivity(T, P, i, j):
    """Fuller-Schettler-Giddings binary diffusivity D_ij [m^2/s].

    D_ij = 1.013e-2 T^1.75 (1/M_i + 1/M_j)^0.5 / (P [V_i^(1/3) + V_j^(1/3)]^2)
    with T in K, P in Pa, M in kg/mol -- the constant already folds in the unit
    conversions from the usual cm^2/s, atm, g/mol form.
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    mw_g = MW * 1.0e3
    m_term = np.sqrt(1.0 / mw_g[i] + 1.0 / mw_g[j])
    v_term = (_FULLER_V[i]**(1.0 / 3.0) + _FULLER_V[j]**(1.0 / 3.0))**2
    return 1.013e-2 * T**1.75 * m_term / (P * v_term)


def _binary_diffusivity_matrix(T, P):
    """Full (nsp, nsp) matrix of binary diffusivities [m^2/s] for scalar T, P."""
    mw_g = MW * 1.0e3
    m_term = np.sqrt(1.0 / mw_g[:, None] + 1.0 / mw_g[None, :])
    v13 = _FULLER_V**(1.0 / 3.0)
    v_term = (v13[:, None] + v13[None, :])**2
    D = 1.013e-2 * T**1.75 * m_term / (P * v_term)
    np.fill_diagonal(D, np.inf)  # self-diffusion is never used
    return D


def effective_mixture_diffusivity(T, P, y):
    """Diffusivity of each species in the mixture [m^2/s] via Wilke's rule.

    D_im = (1 - y_i) / sum_{j != i} (y_j / D_ij)

    Suitable for the dilute-reactant limit used in the pellet effectiveness
    calculation; it is not a rigorous Stefan-Maxwell solution.

    Fully vectorised: ``T`` of shape ``(...)``, ``P`` broadcastable to it and
    ``y`` of shape ``(..., nsp)`` give a result of shape ``(..., nsp)``.
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1.0e-12, None)
    y = y / y.sum(axis=-1, keepdims=True)

    # Binary diffusivities factorise as D_ij = f(T, P) * g_ij, so the (nsp, nsp)
    # geometry factor is built once and scaled per node.
    mw_g = MW * 1.0e3
    m_term = np.sqrt(1.0 / mw_g[:, None] + 1.0 / mw_g[None, :])
    v13 = _FULLER_V ** (1.0 / 3.0)
    g = m_term / (v13[:, None] + v13[None, :]) ** 2          # (nsp, nsp)
    scale = 1.013e-2 * T ** 1.75 / P                          # (...)
    inv_g = np.array(g, copy=True)
    np.fill_diagonal(inv_g, np.inf)
    inv_g = 1.0 / inv_g                                       # 1/g_ij, 0 on diag

    # sum_{j != i} y_j / D_ij = (1/scale) * sum_j y_j / g_ij
    inv_sum = np.einsum("ij,...j->...i", inv_g, y) / scale[..., None]
    return np.clip(1.0 - y, 1.0e-6, None) / np.clip(inv_sum, 1.0e-30, None)


def knudsen_diffusivity(T, pore_radius):
    """Knudsen diffusivity of each species [m^2/s].

    D_K,i = (2/3) r_pore sqrt(8 R T / (pi M_i))
    """
    T = np.asarray(T, dtype=float)[..., None]
    return (2.0 / 3.0) * pore_radius * np.sqrt(8.0 * R_GAS * T / (np.pi * MW))


__all__ = [
    "R_GAS",
    "P_REF",
    "T_REF",
    "SPECIES",
    "N_SPECIES",
    "IDX",
    "MW",
    "cp_molar",
    "enthalpy_molar",
    "entropy_molar_standard",
    "gibbs_molar_standard",
    "cp_graphite",
    "enthalpy_graphite",
    "entropy_graphite",
    "gibbs_graphite",
    "mixture_molar_mass",
    "mixture_cp_molar",
    "mixture_cp_mass",
    "gas_density",
    "molar_density",
    "pure_viscosity",
    "mixture_viscosity",
    "pure_thermal_conductivity",
    "mixture_thermal_conductivity",
    "binary_diffusivity",
    "effective_mixture_diffusivity",
    "knudsen_diffusivity",
]
