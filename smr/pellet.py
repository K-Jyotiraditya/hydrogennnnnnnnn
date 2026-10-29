r"""
Intraparticle (pellet-scale) transport: effective diffusivity and per-reaction
effectiveness factors.

Industrial reforming catalyst pellets are large (10-20 mm) and the intrinsic
rates of Xu & Froment are fast, so the observed rate is strongly diffusion
limited.  The pseudo-heterogeneous model used here lumps that limitation into
an effectiveness factor per reaction,

    r_obs,j = eta_j * r_int,j (T_s, p_s)

evaluated from a generalised Thiele modulus based on the pseudo-first-order
rate constant of the key reactant of each reaction (CH4 for R1 and R3, CO for
R2):

    k_v,j = rho_cat |r_int,j| / C_key,j        [1/s]
    phi_j = (V_p/S_p) sqrt(k_v,j / D_eff,key)  [-]
    eta_j = (1/phi)[1/tanh(3 phi) - 1/(3 phi)] (sphere, L = V_p/S_p = R_p/3)

Two properties of this closure are worth noting.

* It is self-limiting near equilibrium: as the net rate collapses, ``k_v -> 0``,
  ``phi -> 0`` and ``eta -> 1``, which is the correct limit because no
  intraparticle gradient is needed to sustain a vanishing flux.
* It is a linearisation.  A rigorous treatment solves the coupled multi-
  component pellet boundary-value problem (Xu & Froment 1989, part II), which
  gives effectiveness factors of order 0.01-0.1 for 15 mm pellets.  Because an
  industrial reformer is heat-transfer rather than kinetically limited over
  most of its length, the outlet state is weakly sensitive to eta; use
  :attr:`smr.config.Catalyst.eta_scale`, or
  ``effectiveness_model="constant"``, to pin eta to pellet-model or
  experimental values when a quantitative match matters.

The effective diffusivity combines bulk and Knudsen contributions through the
Bosanquet relation and corrects for the pore structure:

    D_eff,i = (eps_p / tau_p) [1/D_im + 1/D_K,i]^-1
"""

from __future__ import annotations

import numpy as np

from .config import Catalyst, Numerics
from .species import (
    IDX,
    R_GAS,
    effective_mixture_diffusivity,
    knudsen_diffusivity,
)

__all__ = [
    "effective_diffusivity",
    "thiele_modulus",
    "sphere_effectiveness",
    "effectiveness_factors",
]

#: Index of the key (rate-limiting) reactant of each reaction.
_KEY_REACTANT = (IDX["CH4"], IDX["CO"], IDX["CH4"])

#: Relative smoothing applied to the net rate magnitude in the Thiele modulus.
_RATE_SMOOTHING = 1.0e-3


def effective_diffusivity(T, P, y, catalyst: Catalyst) -> np.ndarray:
    """Effective intraparticle diffusivity of each species [m^2/s].

    Vectorised over nodes: ``T`` of shape ``(...)``, ``P`` broadcastable to it
    and ``y`` of shape ``(..., nsp)`` give a result of shape ``(..., nsp)``.

    Parameters
    ----------
    T, P : array_like
        Temperature [K] and total pressure [Pa] at the pellet surface.
    y : array_like
        Mole fractions at the pellet surface.
    catalyst : Catalyst
        Pellet structural parameters.
    """
    d_mix = effective_mixture_diffusivity(T, P, y)
    d_knud = knudsen_diffusivity(T, catalyst.pore_radius)
    d_comb = 1.0 / (1.0 / d_mix + 1.0 / d_knud)
    return catalyst.pellet_porosity / catalyst.tortuosity * d_comb


def sphere_effectiveness(phi: np.ndarray) -> np.ndarray:
    """Isothermal first-order effectiveness factor of a sphere.

    ``phi`` is the Thiele modulus built on the characteristic length
    ``L = V_p/S_p = R_p/3``.  The series limit is used for small ``phi`` to
    avoid cancellation in ``1/tanh(3 phi) - 1/(3 phi)``.
    """
    phi = np.asarray(phi, dtype=float)
    small = phi < 1.0e-3
    phi_safe = np.where(small, 1.0, phi)
    x = 3.0 * phi_safe
    eta = (1.0 / phi_safe) * (1.0 / np.tanh(x) - 1.0 / x)
    # Series: eta -> 1 - (3/5) phi^2 as phi -> 0.
    eta_small = 1.0 - 0.6 * phi ** 2
    return np.where(small, eta_small, eta)


def thiele_modulus(rates_si, T, P, y, catalyst: Catalyst) -> np.ndarray:
    """Generalised Thiele modulus of each reaction [-].

    Parameters
    ----------
    rates_si : array_like, shape (..., 3)
        Intrinsic rates in mol/(kg_cat s).
    T, P : array_like
        Pellet-surface temperature [K] and pressure [Pa], shape ``(...)``.
    y : array_like, shape (..., nsp)
        Pellet-surface mole fractions.

    Returns
    -------
    ndarray, shape (..., 3)
    """
    rates_si = np.asarray(rates_si, dtype=float)
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)

    d_eff = effective_diffusivity(T, P, y, catalyst)              # (..., nsp)
    c_total = P / (R_GAS * T)                                     # (...)
    length = catalyst.characteristic_length / max(catalyst.shape_factor, 1.0e-3)

    key = np.array(_KEY_REACTANT)
    c_key = np.clip(c_total[..., None] * y[..., key], 1.0e-8, None)  # (..., 3)
    d_key = np.clip(d_eff[..., key], 1.0e-14, None)                  # (..., 3)
    # ``rates_si`` is expected to be the FORWARD rate (see
    # kinetics.reaction_rates(return_forward=True)), which is strictly positive;
    # the smooth magnitude below only guards against a trial state in which it
    # underflows, and keeps the right-hand side differentiable there.
    scale = np.sqrt(np.sum(rates_si ** 2, axis=-1, keepdims=True))
    magnitude = np.sqrt(rates_si ** 2 + (_RATE_SMOOTHING * scale) ** 2)
    k_v = catalyst.solid_density * magnitude / c_key                  # [1/s]
    return length * np.sqrt(k_v / d_key)


def effectiveness_factors(rates_si, T, P, y, catalyst: Catalyst,
                          numerics: Numerics | None = None) -> np.ndarray:
    """Effectiveness factor of each reaction [-], shape ``(..., 3)``.

    Dispatches on :attr:`smr.config.Catalyst.effectiveness_model` and is
    vectorised over leading axes, so a whole radial stack of nodes is handled
    in one call.
    """
    rates_si = np.asarray(rates_si, dtype=float)
    model = catalyst.effectiveness_model
    if model == "none":
        eta = np.ones_like(rates_si)
    elif model == "constant":
        eta = np.broadcast_to(np.asarray(catalyst.eta_constant, dtype=float),
                              rates_si.shape).copy()
    elif model == "thiele":
        eta = sphere_effectiveness(thiele_modulus(rates_si, T, P, y, catalyst))
    else:  # pragma: no cover - guarded by the dataclass type annotation
        raise ValueError(f"Unknown effectiveness_model {model!r}")
    return np.clip(eta * catalyst.eta_scale, 1.0e-6, 1.0)
