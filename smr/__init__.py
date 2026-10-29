"""
smr -- Two-dimensional pseudo-heterogeneous model of an industrial steam
methane reformer (SMR) tube.

Package layout
--------------
config          Dataclasses holding geometry, catalyst, feed and numerical
                settings, with literature provenance for every number.
species         Thermochemistry (NASA-7 polynomials, graphite) and transport
                properties (viscosity, conductivity, diffusivity).
kinetics        Xu & Froment (1989) intrinsic Langmuir-Hinshelwood kinetics.
pellet          Intraparticle diffusion limitation -> effectiveness factors.
transport        Bed-scale closures: Ergun, effective radial conductivity,
                wall and film heat transfer coefficients.
reformer        The 2D axisymmetric pseudo-heterogeneous solver (method of
                lines in r, stiff ODE integration in z) and its solution
                object.
coking          Thermodynamic carbon-deposition boundary analysis.
optimization    S/C ratio and wall-flux optimisation, epsilon-constraint
                Pareto fronts, feasible operating envelope.
assessment      Model assessment: what the 2D resolution buys, and how much of
                the answer is set by uncertain closures.
visualization   Matplotlib figures (axial profiles, radial contours, fronts).

Primary references
------------------
[XF89]  Xu, J. & Froment, G. F. (1989). Methane steam reforming, methanation
        and water-gas shift: I. Intrinsic kinetics.
        AIChE Journal 35(1), 88-96.
[NOB21] Nobandegani, M. S. et al. (2021). One-dimensional modelling and
        optimisation of an industrial steam methane reformer.
        Chem. Biochem. Eng. Q. 35(4), 369-379.
[CMSL]  COMSOL Multiphysics Application Library, "Nonisothermal Reacting Flow
        in a Methane Steam Reformer" (Application ID 119401) -- architecture
        for the 2D axisymmetric porous-media/heat/species coupling.
"""

from .config import (
    Geometry,
    Catalyst,
    Feed,
    Operating,
    Numerics,
    ReformerConfig,
    SPECIES,
    default_config,
)
from .reformer import Reformer2D, ReformerSolution
from .coking import CokingReport, analyze_coking
from .optimization import (
    OptimizationResult,
    operating_envelope,
    optimize_operating_point,
    pareto_front,
)
from .assessment import (
    closure_sensitivity,
    closure_variants,
    dimensionality_comparison,
)

__version__ = "1.1.0"

__all__ = [
    "Geometry",
    "Catalyst",
    "Feed",
    "Operating",
    "Numerics",
    "ReformerConfig",
    "SPECIES",
    "default_config",
    "Reformer2D",
    "ReformerSolution",
    "CokingReport",
    "analyze_coking",
    "OptimizationResult",
    "operating_envelope",
    "optimize_operating_point",
    "pareto_front",
    "closure_sensitivity",
    "closure_variants",
    "dimensionality_comparison",
]
