# Two-dimensional pseudo-heterogeneous model of an industrial steam methane reformer tube

Thermal-hydraulic simulation and performance analysis of a single industrial SMR
tube, covering radial temperature gradients, pressure drop, catalytic coking
boundary limits and operating-point optimisation.

The model is a 2D axisymmetric pseudo-heterogeneous packed-bed formulation,
discretised by the method of lines in the radial direction and integrated as a
stiff initial-value problem along the tube. It combines

* the intrinsic Langmuir–Hinshelwood kinetics of **Xu & Froment (1989)** for
  Ni/MgAl₂O₄,
* the industrial tube geometry and duty of **Nobandegani et al. (2021)**
  (13.6 m × 98 mm i.d.),
* the 2D porous-media / radial-heat-transfer / multicomponent-transport
  architecture of the **COMSOL non-isothermal reacting-flow reformer model**
  (Application ID 119401),

All thermochemistry is derived from one consistent NASA-7 polynomial set, so that
heat capacities, reaction enthalpies, equilibrium constants and carbon activities
stay consistent with each other.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py baseline        # solve the tube, write profiles and figures
python main.py all             # every analysis, in order
python tests/test_smr.py       # 46 verification tests
```

Each command writes a text report, CSV tables and PNG figures into `results/`.

| command | what it does |
|---|---|
| `baseline` | Solves the tube; axial/radial profiles of T, P and composition |
| `verify` | Grid and tolerance convergence; conservation-law audit |
| `coking` | Carbon-activity analysis and the carbon-free S/C boundary map |
| `optimize` | Tunes S/C and wall heat flux under the metallurgical limit |
| `pareto` | ε-constraint Pareto fronts |
| `envelope` | Feasible firing envelope and the constraint corner |
| `assess` | What the 2D resolution buys, and the closure uncertainty band |
| `sensitivity` | Full-factorial scan of the S/C × heat-flux design space |

Useful options: `--sc 3.5`, `--flux 65` (kW/m²), `--radial 16`, `--wall-mode`
(specify a wall temperature profile instead of a flux), `--eta constant`,
`--no-figures`, `--outdir DIR`.

### As a library

```python
from smr import default_config, Reformer2D, analyze_coking

config   = default_config().with_feed(steam_to_carbon=3.5)
solution = Reformer2D(config).solve()

print(solution.report())
print(analyze_coking(solution).report())

T   = solution.temperature                 # (nz, nr) gas temperature  [K]
y   = solution.mole_fractions()            # (nz, nr, 6)
Twi = solution.wall_surface_temperature()  # (nz,) inner wall surface  [K]
```

---

## Model formulation

State variables: the superficial species molar fluxes `N_i(r,z)` [mol m⁻² s⁻¹],
the gas temperature `T(r,z)` [K], and the cross-section pressure `P(z)` [Pa].

**Species** — one equation per species per radial node

```
dN_i/dz = (1/r) d/dr ( r C_tot D_er dy_i/dr ) + ρ_b Σ_j ν_ji η_j r_j
```

**Energy**

```
( Σ_i N_i cp_i ) dT/dz = (1/r) d/dr ( r k_er dT/dr ) − ρ_b Σ_j ΔH_j η_j r_j
```

**Momentum** (Ergun, cross-section averaged)

```
dP/dz = −[ 150 μ (1−ε)²/(ε³ d_p²) u_s + 1.75 ρ (1−ε)/(ε³ d_p) u_s² ]
```

**Boundary conditions**

| boundary | species | energy |
|---|---|---|
| `r = 0` | `dy_i/dr = 0` | `dT/dr = 0` |
| `r = R` | `dy_i/dr = 0` | `−k_er dT/dr = −q_w(z)`, **or** `= U_w (T\|_R − T_w(z))` |

Two wall modes are supported, mirroring the two ways reformer duty is reported:
a specified **heat flux** profile (the inner wall temperature is then an output,
and the firing level is the optimiser's decision variable), or a specified
**wall temperature** profile (what a pyrometer measures; the flux becomes an
output).

### What makes it pseudo-heterogeneous

Two separate gas–solid resistances, both of which matter:

1. **Intraparticle diffusion** — each reaction carries an effectiveness factor
   from a generalised Thiele modulus on the characteristic length
   `L = V_p/S_p = d_p/6`, with an effective diffusivity combining bulk and
   Knudsen transport through the Bosanquet relation and the pellet's
   porosity/tortuosity.
2. **External film** — the rates are evaluated at the *catalyst surface*
   temperature from the film energy balance
   `h_f a_v (T_g − T_s) = −ρ_b Σ_j ΔH_j η_j r_j(T_s)`, solved by Picard
   iteration at every node. Over the tube the catalyst runs 1–5 K below the gas.

The Thiele modulus is built on the **forward** rate rather than the net rate.
A reversible reaction’s net rate vanishes at equilibrium, so a net-rate Thiele
modulus drives η to 1 over most of the length of an industrial reformer. With
the forward rate the model returns η₁ between 0.024 and 0.36, with a mean of
0.079 for 15 mm pellets, close to the 0.1 reported for a commercial reformer.
The integration cost also drops by two orders of magnitude, since the artificial
stiffness disappears along with the artificial rate.

### Closures

| quantity | correlation |
|---|---|
| Effective radial conductivity | `k_er = k_er⁰ + k_g Re Pr / Pe_r`, `Pe_r = 8 (1 + 46 (d_p/d_t)²)` (Bauer–Schlünder); `k_er⁰` from Kunii–Smith plus a void-radiation term `4σ(e/(2−e))T³d_p` |
| Wall film coefficient | Li & Finlayson, `Nu_w = 0.17 Re^0.79` (or Dixon) |
| Pellet film coefficient | Wakao & Kaguei, `Nu_p = 2 + 1.1 Re^0.6 Pr^⅓` |
| Radial mass dispersion | `D_er = u_s d_p / (Pe_m (1 + 19.4 (d_p/d_t)²))` |
| Viscosity / conductivity | Chapman–Enskog with the Neufeld collision integral; Wilke and Mason–Saxena mixing |
| Diffusivity | Fuller–Schettler–Giddings, Wilke mixture rule, Bosanquet with Knudsen |
| Pressure drop | Ergun |
| Equilibrium | Twigg correlations, cross-checked against the NASA-7 polynomials |

### Numerics

The radial direction uses a **conservative cell-centred finite-volume scheme on
annular cells**, which represents the `1/r` operator exactly and guarantees that
radial transport neither creates nor destroys mass or energy. The resulting
`n_r × 7 + 1` ODE system is integrated with `scipy.integrate.solve_ivp` (BDF)
using an analytically constructed **Jacobian sparsity pattern**, block tridiagonal
apart from the pressure row and column, together with **per-variable absolute
tolerances**. The species fluxes are O(10²) mol m⁻² s⁻¹ while the pressure is
O(10⁶) Pa, and a single scalar `atol` cannot serve both.

Three things a method-of-lines packed-bed model gets wrong if written naively.
Each is documented in the source at the point where it is handled:

* **The dispersive flux must be written on the mole-fraction gradient**,
  `J_i = −C_tot D_er ∂y_i/∂r`, so that `Σ_i J_i = 0`. The apparently equivalent
  `−D_er ∂C_i/∂r` form is wrong in a non-isothermal bed: `C_tot` varies with the
  radial temperature profile, so it drives a net molar flux towards the hot wall,
  here about 6% of the through-flow, which accumulates over 13.6 m and destroys
  the near-wall composition.
* **Negative fluxes must be excluded smoothly**, not clipped. The rate law
  contains `p_H2^2.5`, which is not real-valued for a negative partial pressure,
  but a hard `max(x, 0)` puts a kink in the right-hand side which wrecks the
  finite-difference Jacobian. A smooth `(x + √(x²+δ²))/2` does the same job.
* **The NASA polynomial switch at 1000 K sits in the middle of the operating
  range.** The GRI-Mech sets are continuous in `cp` there but not differentiable,
  so we blend them with a smoothstep weight over a ±20 K window. They agree to
  about 0.01% at the switch, so the blend costs no accuracy.

---

## Baseline results

Default configuration: 13.6 m × 98 mm i.d., S/C = 3.0, 793 K and 26 bar inlet,
4.45 mol s⁻¹ per tube, 58 kW m⁻² mean inner-wall flux on an industrial axial
profile (peak 1.45 × mean at 22% of length), 12 radial cells.

| quantity | value |
|---|---|
| Methane conversion | 76.7% |
| Outlet bulk (mixing-cup) temperature | 1115 K |
| Outlet centreline / near-wall temperature | 1107 / 1122 K |
| Pressure drop | 2.37 bar (26.0 → 23.6 bar) |
| Hydrogen production | 2.76 mol s⁻¹ per tube (2.59 mol per mol CH₄ fed) |
| Methane slip, dry | 6.2% |
| Dry outlet H₂ / CO / CO₂ | 71.7 / 12.8 / 8.7 % |
| Peak inner wall surface temperature | 1137 K (limit 1150 K) |
| Peak outer tube metal temperature | 1150 K |
| Peak radial ΔT (wall surface → centreline) | 106 K |
| Heat duty | 243 kW per tube |
| Mean effectiveness factor η₁ | 0.079 |
| Solve time | 0.7 s (802 right-hand-side evaluations) |

The radial gradient splits into roughly 45 K across the bed itself and 60 K
across the near-wall film at the point of peak flux, falling to 16 K and 16 K at
the outlet. Both are strongly front-loaded: the peak sits at z ≈ 0.7 m, where the
firing is near its maximum and the gas is still cold.

### Does the second dimension earn its cost?

Tested, not assumed. The same tube is run against its own radially lumped limit
(k_er driven to 1e5 W/m·K, every other closure untouched), so the difference is
the contribution of the resolved radial profile alone:

| Metric | 2D | Radially lumped | Difference |
|---|---|---|---|
| CH₄ conversion | 76.683% | 76.648% | **0.036 points** |
| H₂ production | 2.762 mol/s | 2.761 mol/s | **0.001 mol/s** |
| Outlet bulk temperature | 1114.96 K | 1114.03 K | **0.93 K** |
| Peak wall surface temperature | 1137.27 K | 1131.46 K | **5.81 K** |
| Peak radial ΔT | 106.04 K | 62.74 K | **43.30 K** |

**For duty the second dimension buys almost nothing.** A 1D model would have
produced the same performance numbers, because the tube is heat-transfer limited
and the integrated chemistry is set by how much heat crossed the wall rather than
by how it was distributed across the radius. **For thermal diagnostics it buys a
great deal:** the gradient itself, and a 5.8 K higher peak wall temperature.
Against a creep life that roughly halves per 20 K, a systematic 5.8 K
underestimate is worth about 20% of tube life, in the non-conservative direction.

We therefore claim that 2D is justified here by tube integrity and radially
resolved diagnostics, and not by a performance benefit it does not deliver.

### Closure uncertainty

Sweeping k_er across the factor-of-four range that published correlations span:

| k_er (W/m·K) | 15 | 22 | 30 | 45 | 60 |
|---|---|---|---|---|---|
| Peak radial ΔT (K) | 125.6 | 112.1 | 102.7 | 92.5 | 86.5 |
| CH₄ conversion | 0.7671 | 0.7669 | 0.7668 | 0.7667 | 0.7666 |

The gradient is therefore **106 ± 20 K**, not 106.0 K. Over every discrete
closure variant tested (wall-coefficient correlation, bed radiation, film
resistance, radial Peclet number, effectiveness model) the largest deviations
are 0.53 percentage points on conversion, 15.9 K on peak wall temperature and
23.1 K on the radial gradient. The duty predictions are robust to every closure;
the thermal diagnostics carry a band of tens of kelvin.

### Optimised operating point

Maximising hydrogen production with a penalty on methane slip and on steam, under
a 1150 K wall limit and a 4 bar pressure-drop limit:

| | baseline | optimum |
|---|---|---|
| S/C | 3.00 | **4.15** |
| Mean wall flux | 58.0 kW m⁻² | **67.8 kW m⁻²** |
| Methane conversion | 76.7% | **86.9%** |
| H₂ production | 2.76 mol s⁻¹ | **3.18 mol s⁻¹** (+15%) |
| Methane slip, dry | 6.2% | **3.1%** |
| Peak wall surface T | 1137 K | 1149.7 K (active) |
| Pressure drop | 2.37 bar | 4.00 bar (active) |

Both constraints are exactly active at the optimum: the tube is simultaneously
metallurgy-limited and pressure-drop-limited. SLSQP (15 evaluations) and
Nelder–Mead (91 evaluations) agree to three decimal places on both decision
variables.

### Why the optimum sits there: the constraint corner

`python main.py envelope` raises the firing at each S/C until a constraint
becomes tight and records which one it was. The feasible region has a **corner**,
and that corner, rather than any numerical search, is what fixes the optimum:

| S/C | max flux | binding constraint | H₂ |
|---|---|---|---|
| 2.00 | 51.4 kW m⁻² | wall temperature | 2.39 mol s⁻¹ |
| 3.00 | 60.1 kW m⁻² | wall temperature | 2.84 mol s⁻¹ |
| 4.00 | 66.9 kW m⁻² | wall temperature | **3.14 mol s⁻¹** |
| 4.25 | 61.9 kW m⁻² | pressure drop | 3.01 mol s⁻¹ |
| 4.50 | 47.0 kW m⁻² | pressure drop | 2.47 mol s⁻¹ |

Below the corner the **metallurgical limit** binds: extra steam buys conversion
at essentially constant peak wall temperature, so hydrogen production climbs.
Above it the **Ergun pressure drop** takes over: the extra steam raises the mass
flux, the firing has to be cut faster than the chemistry improves, and hydrogen
production *falls*. Production is therefore maximised exactly where the two
constraints meet, at S/C ≈ 4.15 — a purely geometric argument that confirms the
numerical optimiser without reference to it.

This is also why the steam Pareto front turns back on itself above S/C ≈ 4.2.
Those points are feasible but genuinely dominated, and `non_dominated_mask()`
excludes them; they are a real feature of the design space, not a solver
artefact.

---

## Coking analysis

Three carbon-forming routes are evaluated at every node from carbon activities
built on the same NASA-7 thermochemistry plus Kelley's graphite data:

```
a_C,1 = K₁ p_CH4 / p_H2²      CH4  ⇌ C(s) + 2H₂   (methane cracking)
a_C,2 = K₂ p_CO² / p_CO2      2CO  ⇌ C(s) + CO₂   (Boudouard)
a_C,3 = K₃ p_CO p_H2 / p_H2O  CO+H₂⇌ C(s) + H₂O   (CO reduction)
```

Deposition is thermodynamically possible wherever `a_C > 1`.

At the baseline the model flags **the first 0.65 m of the tube**, where the
Boudouard activity peaks at 8.0 at 758 K as CO first appears in a still-cool gas;
methane cracking peaks at 1.4 over the same stretch. Downstream of ~1 m every
activity falls below 1 and stays there. That is the classical result: the
entrance region, not the hot outlet, is the carbon-sensitive zone.

Two gas bases are reported, following Rostrup-Nielsen's distinction. On the
**equilibrated-gas** basis the critical S/C never exceeds 0.93 over 700–1300 K,
so S/C = 3 looks enormously safe; on the **actual-gas** basis the inlet is
flagged. That difference matters, because the inlet gas is far from equilibrium
and only the actual-gas criterion detects it.

One caveat: these are thermodynamic boundaries for **graphitic**
carbon. Whisker carbon on a nickel crystallite is less stable than graphite and
nucleates at activities nearer 1.5–3 on this scale, while gum formation on cooler
surfaces is kinetically controlled and no activity criterion describes it. Pass
`threshold=2.0` to `critical_steam_to_carbon` or `coking_boundary_map` to shift
the boundary onto a whisker-carbon basis. A flagged node is a node to
investigate, not a prediction of deposition.

---

## Verification

`python main.py verify` runs the whole audit; `python tests/test_smr.py` runs 46
physics tests. Headline results:

**Conservation.** Atomic balances on C, H and O close to 1.2 × 10⁻¹² relative.
The global enthalpy balance is decomposed to show exactly where its residual
comes from:

| model variant | residual | relative |
|---|---|---|
| full model | +339 W | 1.40 × 10⁻³ |
| no radial mass dispersion | +31 W | 1.27 × 10⁻⁴ |
| no gas–solid film resistance | +310 W | 1.28 × 10⁻³ |
| **neither** | **+0.19 W** | **7.6 × 10⁻⁷** |

With both optional terms off the balance closes to integrator tolerance, which
confirms the finite-volume discretisation is exactly conservative. The 339 W
residual of the full model is the enthalpy carried by radial species dispersion,
which is **omitted on purpose**: the same eddy transport already appears in
`k_er` through its `Re Pr / Pe_r` term, and counting it twice would overstate
radial heat transfer. It is 0.14% of the tube duty.

**Convergence.** Methane conversion is grid-independent to 6 × 10⁻⁴ relative
between 4 and 32 radial cells; peak wall temperature moves 2.9 K over the same
range and 0.6 K between 12 and 32 cells. Time integration is converged at
`rtol = 1e-5`.

**Independent cross-checks.**

* Rate constants reproduce Xu & Froment's Table 3 values at 648 K to within
  1.6% (`k₁ = 1.871e-4` vs 1.842e-4, `k₂ = 7.585` vs 7.558,
  `k₃ = 2.231e-5` vs 2.193e-5 — the residual is round-off in the published
  pre-exponential factors).
* Standard reaction enthalpies come out at +205.9, −41.2 and +164.7 kJ mol⁻¹,
  against the textbook +206, −41, +165.
* The Twigg equilibrium correlations and the NASA-7 polynomials agree to better
  than 4% over 700–1150 K, drifting to 8% on K₂ at 1300 K, outside the
  correlations' fitted range.
* The LHHW rates vanish identically at the composition returned by the
  independent equilibrium solver, so the kinetics and the thermodynamics are
  mutually consistent.
* The equilibrium composition is reproduced by **two independent routes** that
  share nothing beyond the NASA-7 polynomials: reaction extents with the Twigg
  correlations, and direct Gibbs-energy minimisation in element-potential form.
  They agree to within 2.4e-3 in mole fraction over 900–1200 K, and the residual
  is the known Twigg-versus-NASA drift rather than solver error.
* The reactor outlet sits within ~1 mol% of the equilibrium composition computed
  independently at the same outlet T and P, and on the correct side of it.
* The Boudouard reaction reverses at 970 K and methane cracking becomes
  favourable near 830 K, both matching the standard values.

---

## Validation status

Everything above is **verification** (the equations are solved correctly).
**Validation** (they are the right equations) is the weakest part of this work:

| Claim | Status |
|---|---|
| Equations solved correctly | Verified: conservation to 1e-12, residual attributed, convergence shown |
| Kinetics implemented correctly | Verified against Xu & Froment Table 3 to 1.6% |
| Thermochemistry correct | Verified: ΔH to <1 kJ/mol, two independent routes to K agree to 4% |
| Kinetics/thermodynamics consistent | Verified: rates vanish at the independent equilibrium |
| Outlet industrially typical | Plausibility only |
| Radial gradient magnitude | Order-of-magnitude agreement with literature (>80 °C reported) |
| Reproduces a specific published case | **Not validated** |
| Agrees with plant or experimental data | **Not validated** |

The last two are blocked on the four `[CHECK]` parameters below. The Nobandegani
paper is open access (CC BY 4.0) at
[hrcak.srce.hr/271019](https://hrcak.srce.hr/271019); with its tables the model
could be run on their case and compared against their reported profiles.

Until then the accurate description is that this work **uses** the Nobandegani
geometry rather than reproducing their results.

## Parameters to reconcile before quoting this as a reproduction

Every default carries a provenance tag in `smr/config.py`. Values tagged
`[NOB21]` or `[XF89]` come from the cited papers. Values tagged **`[CHECK]`** are
representative industrial values chosen for consistency with the cited geometry
and duty, and are the ones to reconcile against the source tables before
presenting the model as a reproduction of a specific published case:

* feed molar flow per tube (4.45 mol s⁻¹) and the minor feed fractions
  (H₂ 2.5%, CO₂ 1.0%, N₂ 0.5%),
* bed bulk density (1016 kg m⁻³),
* mean wall heat flux (58 kW m⁻²) and the shape of its axial profile,
* tube wall thickness (12 mm) and the 1150 K metallurgical limit.

Known model limitations:

* The energy equation omits enthalpy transport by radial species dispersion, for
  the double-counting reason given above (0.14% of duty).
* Axial dispersion of heat and mass is neglected, which is standard at these
  Peclet numbers (`u_s L / D_ax` ≫ 10³) but not exact near the inlet.
* The effectiveness factor is a linearised single-reactant Thiele modulus, not a
  full multicomponent pellet boundary-value solve. Because the tube is
  heat-transfer limited rather than kinetically limited, the outlet is weakly
  sensitive to it: forcing η = 1 everywhere changes conversion by under one
  percentage point. Use `effectiveness_model="constant"` with `eta_constant` to
  pin η to pellet-model or experimental values where a quantitative match matters.
* `k_er` is the single most uncertain closure in any 2D packed-bed model; the
  radial gradients scale roughly as `1/k_er`. Use `k_er_model="constant"` to test
  sensitivity directly.
* Radiation between the pellets and the tube wall is in `k_er⁰` but there is no
  separate wall-to-bed radiative path.

---

## Package layout

```
smr/
  config.py         Dataclasses for geometry, catalyst, feed, firing, numerics
  species.py        NASA-7 thermochemistry, graphite, transport properties
  kinetics.py       Xu & Froment LHHW rates, equilibrium constants
  pellet.py         Effective diffusivity, Thiele modulus, effectiveness factors
  transport.py      Ergun, k_er, wall and film heat transfer coefficients
  reformer.py       The 2D solver, radial grid, and solution object
  coking.py         Carbon activities, equilibrated gas, critical S/C
  optimization.py   Evaluator, constrained optimisers, Pareto fronts, scans
  assessment.py     Dimensionality test, closure sensitivity and variants
  visualization.py  Matplotlib figures
main.py             Command-line driver
tests/test_smr.py   46 verification tests
```

---

## Presentation materials

Two further deliverables are built from the same model output:

| Build step | Output |
|---|---|
| `python export_slide_figures.py` | Six single-panel figures sized for projection, in `results/slides/` |
| `python export_demo.py` | `results/demo_data.json` — a grid of 121 full solves (11 steam ratios x 11 firing rates) for the interactive simulator |

The report figures pack four to six panels each, which works on a page but not on
a slide. `export_slide_figures.py` redraws the same output one idea at a time
with projection-scale type. `export_demo.py` precomputes the grid that the
browser demo interpolates between. The page runs no Python, so every point it
shows is bilinear interpolation between real solves rather than a fit.

## Figures

| file | content |
|---|---|
| `fig01_axial_overview.png` | Temperature, composition, conversion, pressure, flux, radial ΔT |
| `fig02_radial_temperature.png` | `T(r,z)` field and radial slices |
| `fig03_radial_profiles.png` | Radial composition, rate and gas–solid ΔT |
| `fig04_kinetic_diagnostics.png` | Rates, η, `k_er`, heat transfer coefficients |
| `fig05_grid_convergence.png` | Mesh-refinement study |
| `fig06_coking.png` | Carbon activity along and across the tube |
| `fig07_carbon_boundary.png` | Critical S/C versus temperature |
| `fig08_optimum_axial.png` | Optimised operating point |
| `fig09/10_pareto_*.png` | ε-constraint Pareto fronts |
| `fig11_sensitivity.png` | Design-space maps |
| `fig12_operating_envelope.png` | Feasible firing envelope and its corner |
| `fig13_model_assessment.png` | Dimensionality test and closure sensitivity |

Figures use a fixed categorical colour order for series identity, checked for
colour-vision deficiency, a single-hue sequential ramp for the temperature field,
and a diverging ramp with its neutral midpoint at `a_C = 1` for carbon activity.
Every figure has a CSV table beside it.

---

## References

1. **Xu, J. & Froment, G. F.** (1989). Methane steam reforming, methanation and
   water-gas shift: I. Intrinsic kinetics. *AIChE Journal* **35**(1), 88–96.
   — rate equations, activation energies, adsorption constants.
2. **Nobandegani, M. S., et al.** (2021). One-dimensional modelling and
   optimisation of an industrial steam methane reformer. *Chemical and
   Biochemical Engineering Quarterly* **35**(4), 369–379.
   — tube dimensions, feed conditions, validation benchmarks.
3. **COMSOL Multiphysics Application Library**, *Nonisothermal Reacting Flow in
   a Methane Steam Reformer*, Application ID 119401.
   — 2D axisymmetric porous-media / heat / species coupling architecture.
4. Twigg, M. V. (1989). *Catalyst Handbook*, 2nd ed. — equilibrium correlations.
5. Rostrup-Nielsen, J. R. (1984). *Catalytic Steam Reforming*, Springer.
   — carbon formation criteria and the principle of equilibrated gas.
6. Froment, G. F. & Bischoff, K. B. *Chemical Reactor Analysis and Design.*
   — packed-bed radial transport closures.
7. Smith, G. P., et al. *GRI-Mech 3.0.* — NASA-7 thermochemical polynomials.
8. Li, C. H. & Finlayson, B. A. (1977). Heat transfer in packed beds — a
   reevaluation. *Chemical Engineering Science* **32**, 1055–1066.
9. Wakao, N. & Kaguei, S. (1982). *Heat and Mass Transfer in Packed Beds.*
