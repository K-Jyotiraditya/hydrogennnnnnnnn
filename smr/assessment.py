r"""
Model assessment: what the two-dimensional resolution buys, and how much of
the answer is set by uncertain closures rather than by the physics.

These are questions about the *model*, not about the design, which is why they
live apart from :mod:`smr.optimization`. Both are the questions an examiner or
a reviewer asks first, and both are cheap to answer once the model exists.

Dimensionality
--------------
A two-dimensional model is only worth its cost if resolving the radial
direction changes an answer. :func:`dimensionality_comparison` tests that
directly rather than assuming it, by running the same tube against its own
radially-lumped limit.

The limit is taken by driving the effective radial conductivity to a very large
value, which makes the bed isothermal in r while leaving every other closure
untouched -- including the wall film, which a one-dimensional model would still
need. The difference between the two runs is therefore exactly the contribution
of the resolved radial temperature profile, isolated from everything else.

Closure sensitivity
-------------------
The effective radial conductivity is the most uncertain quantity in any packed
bed model: published correlations for comparable geometries disagree by a
factor approaching two, and the predicted radial gradient scales roughly as its
reciprocal. :func:`closure_sensitivity` sweeps it over that range and reports
the band on the outputs, so the headline numbers can be quoted with an
uncertainty instead of a false precision.

:func:`closure_variants` does the same for discrete modelling choices -- which
wall-coefficient correlation, whether bed radiation is counted, whether the
film resistance is resolved, which effectiveness-factor model -- by swapping
one at a time and tabulating the effect.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import ReformerConfig, default_config
from .reformer import Reformer2D

__all__ = [
    "LUMPED_CONDUCTIVITY",
    "REPORTED_METRICS",
    "dimensionality_comparison",
    "closure_sensitivity",
    "closure_variants",
]

#: Effective radial conductivity [W/(m K)] used to emulate a radially lumped
#: (one-dimensional) bed. Large enough that the residual radial gradient is
#: below 0.1 K, small enough not to stiffen the integration.
LUMPED_CONDUCTIVITY = 1.0e5

#: Metrics carried through every assessment, with display labels and scales.
REPORTED_METRICS = (
    ("methane_conversion", "CH4 conversion [%]", 100.0),
    ("hydrogen_production_mol_per_s", "H2 production [mol/s]", 1.0),
    ("outlet_bulk_temperature_K", "outlet bulk T [K]", 1.0),
    ("max_wall_surface_temperature_K", "peak wall surface T [K]", 1.0),
    ("max_outer_wall_temperature_K", "peak outer metal T [K]", 1.0),
    ("max_radial_dT_K", "peak radial dT [K]", 1.0),
    ("pressure_drop_bar", "pressure drop [bar]", 1.0),
)


def _summary(config: ReformerConfig) -> dict:
    return Reformer2D(config).solve().summary()


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------

def dimensionality_comparison(config: Optional[ReformerConfig] = None,
                              intermediate: Optional[Sequence[float]] = None,
                              verbose: bool = True) -> dict:
    """Compare the 2D model against its radially lumped limit.

    Parameters
    ----------
    config : ReformerConfig, optional
        Template configuration (default: the baseline tube).
    intermediate : sequence of float, optional
        Effective radial conductivities [W/(m K)] to run between the correlated
        value and the lumped limit, so the transition can be seen rather than
        only its endpoints.
    verbose : bool
        Print the comparison as it is computed.

    Returns
    -------
    dict with keys ``two_dimensional``, ``lumped``, ``delta`` (metric ->
    2D minus lumped), ``transition`` (list of per-conductivity summaries) and
    ``verdict`` (a one-line reading of the result).
    """
    cfg = default_config() if config is None else config

    two_d = _summary(cfg)
    lumped = _summary(cfg.with_numerics(k_er_model="constant",
                                        k_er_constant=LUMPED_CONDUCTIVITY))
    delta = {key: two_d[key] - lumped[key] for key, _, _ in REPORTED_METRICS}

    transition = []
    if intermediate:
        for k_er in intermediate:
            summary = _summary(cfg.with_numerics(k_er_model="constant",
                                                 k_er_constant=float(k_er)))
            summary["k_er"] = float(k_er)
            transition.append(summary)

    # The verdict is written from the numbers, not asserted in advance.
    conversion_points = abs(delta["methane_conversion"]) * 100.0
    wall_kelvin = abs(delta["max_wall_surface_temperature_K"])
    if conversion_points < 0.5 and wall_kelvin < 10.0:
        verdict = ("Radial resolution changes the duty predictions negligibly "
                   "(<0.5 conversion points, <10 K on peak wall temperature). "
                   "2D is needed for the gradient itself and for radially "
                   "resolved diagnostics such as coking, not for duty.")
    elif wall_kelvin >= 10.0:
        verdict = ("Radial resolution shifts the peak wall temperature by "
                   f"{wall_kelvin:.1f} K, which is material against a "
                   "metallurgical limit: 2D is required for tube-life work.")
    else:
        verdict = ("Radial resolution changes conversion by "
                   f"{conversion_points:.2f} points: 2D affects the duty "
                   "prediction materially.")

    if verbose:
        print(f"{'metric':<30} {'2D':>12} {'lumped':>12} {'difference':>12}")
        print("-" * 68)
        for key, label, scale in REPORTED_METRICS:
            print(f"{label:<30} {two_d[key] * scale:12.3f} "
                  f"{lumped[key] * scale:12.3f} {delta[key] * scale:12.3f}")
        print()
        print(f"  {verdict}")

    return {
        "two_dimensional": two_d,
        "lumped": lumped,
        "delta": delta,
        "transition": transition,
        "verdict": verdict,
        "lumped_conductivity": LUMPED_CONDUCTIVITY,
    }


# ---------------------------------------------------------------------------
# Closure sensitivity
# ---------------------------------------------------------------------------

def closure_sensitivity(config: Optional[ReformerConfig] = None,
                        conductivities: Optional[Sequence[float]] = None,
                        verbose: bool = True) -> dict:
    """Sweep the effective radial conductivity over its credible range.

    Parameters
    ----------
    conductivities : sequence of float, optional
        Values of ``k_er`` [W/(m K)] to impose. The default spans 15 to 60,
        which brackets the correlated value of roughly 30 by the factor of two
        that published correlations disagree by.

    Returns
    -------
    dict with the swept values, a per-metric array, the correlated-model
    reference, and ``band`` giving (min, max, half-width) for each metric.
    """
    cfg = default_config() if config is None else config
    values = np.asarray(conductivities if conductivities is not None
                        else (15.0, 22.0, 30.0, 45.0, 60.0), dtype=float)

    reference = _summary(cfg)
    rows = [_summary(cfg.with_numerics(k_er_model="constant",
                                       k_er_constant=float(k)))
            for k in values]

    metrics = {key: np.array([row[key] for row in rows])
               for key, _, _ in REPORTED_METRICS}
    band = {key: (float(v.min()), float(v.max()),
                  0.5 * float(v.max() - v.min()))
            for key, v in metrics.items()}

    if verbose:
        print(f"  {'k_er [W/m K]':>12} " + " ".join(
            f"{label:>24}" for _, label, _ in REPORTED_METRICS[3:6]))
        for i, k in enumerate(values):
            print(f"  {k:12.1f} " + " ".join(
                f"{metrics[key][i] * scale:24.2f}"
                for key, _, scale in REPORTED_METRICS[3:6]))
        print()
        lo, hi, half = band["max_radial_dT_K"]
        print(f"  Peak radial dT spans {lo:.1f} to {hi:.1f} K over this range: "
              f"the correlated value of "
              f"{reference['max_radial_dT_K']:.1f} K carries about "
              f"+/-{half:.0f} K of closure uncertainty.")

    return {
        "conductivities": values,
        "metrics": metrics,
        "band": band,
        "reference": reference,
    }


def closure_variants(config: Optional[ReformerConfig] = None,
                     verbose: bool = True) -> dict:
    """Swap discrete modelling choices one at a time and tabulate the effect.

    Each variant changes exactly one closure against the baseline, so the
    resulting table reads as a one-at-a-time sensitivity study over the
    modelling decisions that are judgement calls rather than measurements.
    """
    cfg = default_config() if config is None else config

    variants: list[tuple[str, ReformerConfig]] = [
        ("baseline", cfg),
        ("wall coefficient: Dixon",
         cfg.with_numerics(wall_htc_model="dixon")),
        ("no bed radiation in k_er",
         cfg.with_numerics(radiation_in_bed=False)),
        ("no gas-solid film resistance",
         cfg.with_numerics(film_resistance=False)),
        ("no radial mass dispersion",
         cfg.with_numerics(radial_mass_dispersion=False)),
        ("radial Peclet 6 (from 8)",
         cfg.with_numerics(radial_peclet_heat=6.0)),
        ("radial Peclet 12 (from 8)",
         cfg.with_numerics(radial_peclet_heat=12.0)),
        ("effectiveness: intrinsic (eta = 1)",
         cfg.with_catalyst(effectiveness_model="none")),
        ("effectiveness: fixed literature values",
         cfg.with_catalyst(effectiveness_model="constant")),
    ]

    rows = []
    for label, variant in variants:
        summary = _summary(variant)
        rows.append({"variant": label, **summary})
        if verbose:
            print(f"  {label:<38} X = "
                  f"{summary['methane_conversion'] * 100:6.2f}%  "
                  f"T_wall = {summary['max_wall_surface_temperature_K']:7.1f} K"
                  f"  dT_rad = {summary['max_radial_dT_K']:6.1f} K",
                  flush=True)

    baseline = rows[0]
    spread = {}
    for key, _, _ in REPORTED_METRICS:
        deltas = [abs(row[key] - baseline[key]) for row in rows[1:]]
        spread[key] = float(max(deltas)) if deltas else 0.0

    return {"rows": rows, "baseline": baseline, "largest_deviation": spread}
