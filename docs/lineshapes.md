# Resonance dynamics and angular terms

Jax-PWA implements resonance dynamics directly in JAX. Laura++ and published LHCb amplitude analyses are principal references used to define and validate the conventions.

## Available dynamics models

```python
RelativisticBreitWigner()
Pole()
SigmaPole()
GounarisSakurai()
RhoOmegaMixing(component="rho")
PipiKKRescattering()
Flatte(...)
LASS(...)
KMatrix(...)
QMI(...)
QMI2D(...)
PolarFormFactorSymNR(...)
```

One-dimensional isobar dynamics use the ordinary `lineshape(mass, context)` interface through `Resonance`. A genuinely two-dimensional Dalitz amplitude such as `QMI2D` or `PolarFormFactorSymNR` is attached through `DalitzAmplitude` because it depends simultaneously on two invariant-mass-squared coordinates.

## Relativistic Breit-Wigner

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r)^2.
```

## Simple Pole

`Pole` exposes the simple fixed-width Breit-Wigner form listed as `BW` in Laura++ Appendix A:

```text
R(m) = 1 / (m - m0 - i Gamma0/2).
```

## Gounaris-Sakurai

`GounarisSakurai` follows Laura++ Appendix A for rho-like vector states,

```text
R(m) = [1 + D Gamma0/m0] /
       [m0^2 - m^2 + f(m) - i m0 Gamma(m)].
```

`RhoOmegaMixing` implements Eq. (15) of the LHCb isobar model. Its `rho`
and `omega` variants are the two effective terms obtained by splitting the
common denominator; the omega propagator uses its fixed pole width, without
momentum or barrier factors, as in Laura++ `LauRhoOmegaMix`.

`SigmaPole` is the separate pole convention used for the paper's
`f0(500)`, `sqrt(s_sigma) = m_sigma - i Gamma_sigma`, rather than the generic
fixed-width `Pole` convention.

`PipiKKRescattering` implements the source term and the inelastic amplitude
from Eqs. (17)--(21) of the paper and is explicitly zero outside
`1.0 <= m(pi pi) <= 1.5 GeV`.

The default `convention="paper"` uses the literal printed expression.
`convention="laura"` instead follows the production factor in `s=m**2`
and overall phase `i` in Laura++ 3.8 `LauRescatteringRes::amplitude`.
The source denominators then have units GeV squared. The explicit mass window
is retained; reproducing Laura++'s full support requires setting `mass_min`
to the KK threshold. These conventions must be matched to the coefficients:
a global phase of one component changes its interference with other components.

`convention="laura"` reproduces `LauRescatteringRes::amplitude` to float64
precision against a literal line-by-line port, including its `eta0=1`
override below the KK threshold and above `m_prime` and its behaviour at the
single point `m == 2*kaon_mass` where the C++ itself divides by zero (this
returns the well-defined two-sided-limit zero there instead of propagating a
NaN); see `docs/reviews/20260914_pipi_kk_rescattering.md`. This is inactive
at the default `mass_min=1.0`/`mass_max=m_prime=1.5` window, so it only
matters when `mass_min`/`mass_max` are widened. Laura++'s own
`LauRescatteringRes` constructor defaults are `lambdaPiPi=1.0`,
`lambdaKK=2.8` (i.e. `delta_kk_squared=7.84`); `PipiKKRescattering`'s class
default `delta_kk_squared=1.0` is not meant to reproduce that value, only to
give the unit test a neutral reference point -- pass the actual analysis's
`lambdaPiPi`/`lambdaKK` explicitly.

## Flatte

`Flatte` follows the coupled two-channel Laura++ form

```text
R(m) = 1 / [(m0^2-m^2) - i m0 (Gamma1(m)+Gamma2(m))].
```

with analytic continuation below threshold and optional Adler zero. Presets are provided for `f0(980)`, `K0*(1430)` and `a0(980)` charge states.

## LASS K-pi S-wave

`LASS` implements the coherent effective-range plus `K0*(1430)` S-wave form,

```text
R(m) = m / [q cot(delta_B) - i q]
     + exp(2 i delta_B)
       [m0 Gamma0 (m0/q0)] /
       [(m0^2-m^2) - i m0 Gamma0 (q/m)(m0/q0)],

cot(delta_B) = 1/(a q) + r q/2.
```

## Five-channel K-matrix pi-pi S-wave

`KMatrix` implements the standard five-pole, five-channel Anisovich-Sarantsev model used by Laura++. The channel order is `pi pi`, `K Kbar`, `4 pi`, `eta eta`, `eta eta'`.

```text
K_ij(s) = [ sum_alpha g_i^alpha g_j^alpha/(m_alpha^2-s)
          + f_ij^scatt (m_sq0-s0_scatt)/(s-s0_scatt) ] f_A0(s)

P_j(s) = sum_alpha beta_alpha g_j^alpha/(m_alpha^2-s)
       + f_1j^prod (m_sq0-s0_prod)/(s-s0_prod)

F = (I - i K rho)^(-1) P.
```

Here `m_sq0 = 1 GeV^2`. The numerator is constant in `s`, matching
`KMatrix`'s `_slowly_varying_factor`; writing `1-s0/s` instead would reduce
the whole factor to `1/s` and describe a different model.

The scattering constants are fixed by default while the process-dependent `betas` and `f_prod` may be complex fit parameters. `scattering_amplitude()` and `s_matrix()` expose the coupled-channel `T` and `S` matrices for unitarity diagnostics.

## Laura++ alternative rescattering model

`Rescattering2` ports `LauRescattering2Res` and can be used as an ordinary
spin-0 one-dimensional lineshape. It parameterizes

```text
A(m) = g_00(m) exp(i phi_00(m)) / (1 + m^2/Lambda^2)
```

with Chebyshev expansions in two mass regions: from the charged-kaon threshold
`2 m_K` to 1.47 GeV, and from 1.47 to 2.00 GeV. `phi_00` is evaluated as a
Chebyshev series in degrees and converted to radians once, after evaluation,
matching `LauRescattering2Res::resAmp`. `g_00(m)`, and therefore `A(m)`, is
exactly zero below the charged-kaon threshold `2 m_K`; region I is not
extrapolated into the sub-threshold region for the amplitude actually
returned by `__call__` (`Rescattering2.magnitude()` still exposes the raw,
un-cut Chebyshev value for diagnostics). The default `B`, `C`, `D`, `F`
coefficients and `Lambda = 1` GeV are the Laura++ defaults. The two Chebyshev
expansions are constructed to be continuous at 1.47 GeV.

```python
from jaxpwa import RealImag, Rescattering2, Resonance

rescattering = Resonance(
    "rescattering",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
    mass=1.47,
    width=0.0,
    spin=0,
    lineshape=Rescattering2(),
)
```

Because it uses the normal `ResonanceAmplitude` path, it participates in the
same deterministic normalization, CP-fit coefficient handling, caching, and
minimizer interface as the other one-dimensional lineshapes.

## QMI / QMIPWA S-wave

`QMI` implements a quasi-model-independent scalar amplitude specified at fixed two-body mass knots. The interpolation coordinate is

```text
s = m(pi pi)^2.
```

Thus

```text
A_S(s_k) = a_k exp(i delta_k)
A_S(s)   = a(s) exp(i delta(s)).
```

By default, magnitude and phase are interpolated separately. Four interpolation
modes are available:

```python
QMI(..., interpolation="linear")  # default; reproduces the published LHCb convention
QMI(..., interpolation="cubic")   # local smoothstep, two adjacent knots
QMI(..., interpolation="hermite") # local Hermite, finite-difference slopes
QMI(..., interpolation="natural") # global natural cubic spline in s=m^2
```

All modes are implemented in JAX and pass through the supplied knots; outside the knot range the nearest endpoint value is used. `natural` solves the global spline system: its second derivative vanishes at the two endpoints and its first and second derivatives are continuous at interior knots. Moving a node can affect every interval. With two knots it reduces to linear interpolation. Natural boundary conditions apply inside the knot range; constant continuation outside it need not have a matching first derivative.

The natural option supports polar and Cartesian parameters, prepared evaluation, JIT and automatic gradients. It solves a knot-sized system without storing an event-by-knot basis matrix. It uses ordinary JAX autodiff, so it does not have the grouped custom VJP optimization of the local modes; large-fit performance should be measured separately. As with other cubic splines, overshoot is possible, including negative interpolated polar magnitudes. `linear` remains the default, and `cubic` retains its local behavior.

`cubic`'s locality is what forces its derivative to exactly zero at every knot, from both sides: the smoothstep weight `3t^2 - 2t^3` is the only cubic blend on two points (`values[index]`, `values[index+1]`) whose derivative is guaranteed to match its neighboring interval's without any information about further knots. Because that zero-derivative value is fixed to the interval's own two endpoints and never to the surrounding trend of the curve, `cubic` visibly flattens at each knot and then bulges between knots to compensate; this is expected, not a defect, and is exercised by `test_qmi_local_cubic_uses_only_the_two_adjacent_knots`. `hermite` removes this artifact at no extra per-event cost — its knot tangents are precomputed once from neighboring knots (`_hermite_slopes`, a central difference for interior knots) and its custom VJP uses the same grouped-interval-sum reduction as `cubic` (see `docs/performance.md`) — so prefer `hermite` over `cubic` whenever a smooth-looking curve matters and the strict two-knot dependency of `cubic` is not itself a requirement.

The public `knots` argument is given as masses in GeV, matching the published tables; internally `QMI` squares the masses and interpolates in `s`. Entries of `magnitudes` and `phases` may be numerical constants or fit `Parameter` objects. Phases are expressed in radians and should be supplied as a continuous/unwrapped sequence; interpolation does not impose a `[-pi, pi)` branch cut.

Example:

```python
qmi = QMI(
    knots=(0.30, 0.50, 0.70, 0.90, 1.10),
    magnitudes=(a0, a1, a2, a3, a4),
    phases=(d0, d1, d2, d3, d4),
    interpolation="natural",
)
```

Published QMI values should be validated in analysis-specific studies before
being used in a production model.

For fits where the polar coordinates become poorly conditioned, the same class
accepts Cartesian knot values:

```python
qmi = QMI(
    knots=(0.30, 0.50, 0.70, 0.90, 1.10),
    real_parts=(x0, x1, x2, x3, x4),
    imaginary_parts=(y0, y1, y2, y3, y4),
    interpolation="linear",
)
```

In this form, the real and imaginary parts are interpolated directly and
independently in `s=m**2`, and the amplitude is

```text
A_S(s) = x(s) + i y(s).
```

This avoids phase-branch ambiguities and the magnitude-zero singularity during
minimization. `interpolated_magnitude_phase(mass)` remains available and derives
the polar coordinates from the interpolated complex value. A QMI declaration
must provide exactly one complete parameter set: either `magnitudes` and
`phases`, or `real_parts` and `imaginary_parts`.

### Per-knot CP violation

Cartesian knots may be the real/imaginary parts of a `CPRealImag`, so that every
knot carries its own `(x, y, dx, dy)` and the node value for charge `q = ±1` is
`(x + q dx) + i (y + q dy)`:

```python
nodes = [
    CPRealImag(
        Parameter.dynamics(f"S.node_{i:02d}.x", x0, owner="S"),
        Parameter.dynamics(f"S.node_{i:02d}.y", y0, owner="S"),
        Parameter.dynamics(f"S.node_{i:02d}.dx", 0.0, owner="S"),
        Parameter.dynamics(f"S.node_{i:02d}.dy", 0.0, owner="S"),
    )
    for i, (x0, y0) in enumerate(seed)
]

def qmi_for(charge):
    bound = [node.for_charge(charge) for node in nodes]
    return QMI(knots, real_parts=tuple(n.real_part for n in bound),
               imaginary_parts=tuple(n.imag_part for n in bound))
```

Build one `QMI` per charge (`qmi_for(+1)` for the B+ model, `qmi_for(-1)` for B-);
both share the same parameters, so the S-wave `A_CP` and phase difference are
free functions of the mass. Give the `Resonance` a fixed unit coefficient
(`RealImag(1.0, 0.0)`) rather than a free one -- a free global coefficient would
be degenerate with the nodes. `CPRealImag.real_part`/`imag_part` expose
`parameters` and `resolve(values)`, so these entries are collected as fit
parameters and differentiated like ordinary `Parameter` knots. The node
parameters must be `Parameter.dynamics` with the component owner.

### Optional QMI knot smoothing

All four interpolation modes support the same **optional NLL penalty**:

```python
penalty = qmi.smoothness_constraint(strength=0.0)  # lambda; off by default
session = session.with_constraint(penalty)        # FitSession or CPFitSession
# Low-level alternative: ConstrainedNLL(base_nll, penalty)
```

`QMISmoothnessConstraint(qmi, strength=..., weights=...)` is also available
from the top-level package. Creating the constraint alone does not activate
it; attach it once to the likelihood. No interpolation formula, component
normalization, or PDF normalization is changed. Existing QMI declarations
and serialized amplitude models are unchanged. Constraints belong to the
session/objective, so model-only export does not save this configuration.

For complex node values `q_i`, let `s_i = m_i**2`, `h_i = s_(i+1) - s_i`,
and `d_i = (q_(i+1) - q_i) / h_i`. The added term is

```text
lambda * sum_i w_i * 2 * |d_i - d_(i-1)|² / (h_i + h_(i-1)),
```

summed over interior nodes. The nonuniform-grid factors approximate the
integral of squared curvature in **s**, using differences of adjacent
complex slopes. This is a penalty on the **nodes**, not an integral of the
interpolated curve's second derivative: that distinction makes it useful
for linear interpolation too. It is identical across interpolation modes
for identical complex nodes, even though those interpolants differ between
nodes. Constant or affine-in-s complex nodes have zero penalty. Two knots
also give zero because they define no interior curvature. Boundary slopes
are not pinned.

For polar nodes, the constraint first forms `a_i * exp(1j * phi_i)` and
penalizes Re/Im. It does not penalize wrapped phase differences directly,
and is regular at zero magnitude. This does not resolve the existing need
to unwrap phases consistently for **polar interpolation** between nodes.
The penalty and its gradients/Hessians are pure JAX and operate only on
knot-sized arrays, independently of prepared event caches.

`strength` must be a fixed finite nonnegative number, not a fit `Parameter`.
`strength=0` is an exact no-op. There is no implicit factor of 1/2 or division
by the number of events. With masses in GeV the raw penalty has units of
amplitude squared / GeV^6; consequently a numerical lambda depends on the
node amplitude scale and the NLL convention. Validate its value using
independent toy ensembles, including bias, CPV recovery and interval coverage;
smaller penalized-Hessian errors do not by themselves demonstrate improved
frequentist coverage.

`weights` defaults to one per **interior** node (length `len(knots)-2`).
All weights must be finite and nonnegative. Set a weight to zero to disable
that node's entire three-node stencil. To protect a narrow structure, disable
all stencils overlapping that region, not just nodes centred inside it.

For CP QMI with independent charge shapes, attach one penalty for each:

```python
session = session.with_constraint(qmi_plus.smoothness_constraint(strength=lam))
session = session.with_constraint(qmi_minus.smoothness_constraint(strength=lam))
```

This does not constrain the two charges to agree. If the **same QMI object**
is shared by both charges, attach its penalty only once. A separately declared
chi_c0 is not directly penalized, although correlations can affect its fit.
The remaining scalar QMI can itself contain rapidly varying physical
structures such as f0(980); regularization can bias those too.

The penalty acts on the **raw node values**. Use a fixed scale convention,
for example `Resonance(..., normalize_component=False)` with a fixed external
QMI coefficient and an isobar reference. If a free global coefficient can
compensate arbitrary rescaling of all nodes, or the QMI is dynamically
unit-normalized with all node magnitudes free, the fit can evade the penalty
by shrinking the nodes without changing its PDF. The constraint deliberately
does not change those normalization choices for the caller.

## QMI2D Dalitz amplitude

`QMI2D` is the direct two-dimensional extension of the QMI idea. Every Dalitz cell carries one complex amplitude

```text
A_ij = a_ij exp(i phi_ij).
```

The axes are given directly in Dalitz invariants (`s12` and `s13`) through bin edges. Magnitudes and phases may contain ordinary numbers or dynamical `Parameter` objects, so every active cell can be floated in a fit.

Three evaluation modes are available:

```python
QMI2D(..., interpolation="none")
QMI2D(..., interpolation="linear")
QMI2D(..., interpolation="cubic")
```

- `none` is piecewise constant: every event receives exactly the complex number assigned to its bin.
- `linear` treats the cell values as located at bin centers and interpolates magnitude and phase bilinearly.
- `cubic` performs a local tensor-product bicubic Catmull-Rom interpolation, again separately for magnitude and phase.

All three implementations are JAX-native and therefore compatible with automatic differentiation and minimization.

### Physical Dalitz-bin mask

For a real three-body decay, the rectangular `s12 x s13` grid contains cells that never intersect the physical Dalitz region. `physical_bin_mask(...)` marks only cells with physical support using the exact analytic Dalitz boundary. The bin edges themselves should be built from the exact kinematic endpoints,

```text
s12_min = (m1 + m2)^2
s12_max = (M - m3)^2
```

rather than from the minimum/maximum of numerical integration samples. This is
important because quadrature nodes do not lie exactly on the kinematic
endpoints, while a physical boundary bin must extend all the way to `s12_max`.

Example:

```python
smin = (m1 + m2)**2
smax = (M - m3)**2
edges = np.linspace(smin, smax, 9)

mask = physical_bin_mask(
    tuple(edges), tuple(edges),
    mother_mass=M,
    masses=(m1, m2, m3),
    folded=True,
)

field = QMI2D(
    s12_edges=tuple(edges),
    s13_edges=tuple(edges),
    magnitudes=magnitudes,
    phases=phases,
    active_mask=mask,
    interpolation="cubic",
    folded=True,
)
```

For cubic interpolation, slopes use the physical distances between bin
centers. Nonuniform grids have continuous first derivatives at internal
centers; uniform grids retain the previous Catmull–Rom interpolation, including
its repeated-value boundary convention. Interpolation remains local to a 4x4
neighborhood.

`physical_bin_mask` clips each bin to the kinematic s12 interval before sampling
the analytic boundary. It accepts rectangular grids extending outside the
Dalitz domain. The intersection search is sampled, so check mask convergence
with `samples_per_bin` for very narrow intersections. Free parameters in cells
marked inactive are rejected; fixed values remain valid placeholders.

For `interpolation="none"`, inactive cells evaluate to zero. For linear/cubic interpolation, inactive rectangular cells act only as ghost support filled from the nearest active cell; they are not intended to carry independent physics parameters.

For channels with two identical particles, `folded=True` evaluates the field at

```text
s_low  = min(s12, s13)
s_high = max(s12, s13)
```

which imposes the exchange symmetry directly on the two-dimensional field. This is the natural default for studies of `D_s+ -> pi- pi+ pi+` when `s12` and `s13` correspond to the two `pi+ pi-` combinations.

`folded=True` requires `s12_edges` and `s13_edges` to be identical (the constructor raises otherwise): the lookup above puts `s_low` on the `s12` grid and `s_high` on the `s13` grid, so mismatched ranges would silently clamp whichever physical value happens to be smaller/larger to the narrower grid's boundary instead of producing the intended single symmetric field.

A QMI2D component is attached directly to the coherent amplitude model:

```python
model = DecayModel(
    channel,
    [DalitzAmplitude("qmi2d", field, RealImag(1.0, 0.0))],
)
```

The global complex normalization/phase ambiguity remains present, just as for a 1D QMI, and a fit must fix an appropriate reference convention. A completely free two-dimensional field can also develop poorly constrained or null directions; closure tests and Hessian/correlation diagnostics are therefore essential before using it on data.

## Polar form-factor nonresonant amplitude

`PolarFormFactorSymNR` ports Laura++ `LauPolarFormFactorSymNR` (Nogueira et al., PRD 92, 054010
(2015), modified for symmetric Dalitz plots). It is a real amplitude in the two invariants `s`, `t`
that pair each identical particle with the third one, with a scale `lambda_` in GeV:

```text
PolarFFSymNR:        1/(1 + s/lambda^2) + 1/(1 + t/lambda^2)
PolarFFSymNRNoInter: 1/(1 + min(s, t)/lambda^2)      (no_interference=True)
```

The sum of the two terms already is the identical-particle symmetrization, so it is attached through
`DalitzAmplitude`, not `Resonance`. Laura++ uses `s = m13^2`, `t = m23^2` with the identical
particles as tracks 1 and 2, which is the default `invariants=("s13", "s23")` for a channel ordered
like `("K+", "K+", "K-")`; pass the other two invariant names for a different ordering. As in
Laura++ there is no `1/sqrt(2)` or other overall normalization. Laura++'s default `lambda` is
1.0 GeV, fixed, within [0, 10]; `lambda_` may be a `Parameter.dynamics(...)` to float it:

```python
lam = Parameter.dynamics("NR.lambda", 1.5845, owner="NR", fixed=True, bounds=(0.0, 10.0))
DalitzAmplitude("NR", PolarFormFactorSymNR(lam), RealImag(1.0, 0.0))
```

The standalone call also resolves parameters, e.g.
`PolarFormFactorSymNR(lam)(data, {"NR.lambda": 2.0})`. Evaluation uses the
equivalent form `lambda^2 / (lambda^2 + s)` so the amplitude and its derivative
have a finite zero limit at `lambda=0` for positive physical invariants.
An initially configured numeric scale must be positive and finite. A zero
scale cannot supply a unit-normalized component on its own. Floating scales
must be `Parameter.dynamics` objects with the component owner, as for QMI
nodes; a generic free `Parameter` would otherwise be silently fixed by the
prepared cache and is rejected. Fixed generic parameters remain supported.

## QMI fit parameters and scale convention

Declare every free QMI/QMI2D node with
`Parameter.dynamics(name, value, owner=component_name)`. Generic free
`Parameter(...)` nodes are rejected because they would otherwise be exposed to
Minuit while remaining frozen in the amplitude cache. Fixed generic parameters
remain supported. Knot masses and QMI2D bin edges must be finite.

Prepared interpolation data are isolated per resonance amplitude, so multiple
QMI components on the same particle pair may use different knot grids.

With `normalize_component=True`, a common positive rescaling of all Cartesian
QMI nodes (or all polar magnitudes) cancels against the component norm. If all
those nodes float, this leaves an unidentifiable scale even when the component's
global coefficient is fixed. Either fix an appropriate node-scale convention,
or use `normalize_component=False` and a fixed QMI coefficient, together with
a separate reference amplitude that defines the total scale/phase convention.
Do not automatically fix a relative phase: that may constrain the physical
model. The fitter does not choose these conventions automatically.

## Component composition and normalization

`ResonanceAmplitude` multiplies one-dimensional isobar lineshapes by their barrier and angular terms. `DalitzAmplitude` bypasses the isobar construction and evaluates a full Dalitz-dependent complex function directly.

All amplitude-component and coherent-PDF normalization uses deterministic
mass-plane Gauss--Legendre or Square-Dalitz quadrature; `PhaseSpaceMC` is
retained only for toy/proposal generation.

## References

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.

V. V. Anisovich and A. V. Sarantsev, *K-matrix analysis of the (IJ^PC = 00++)-wave in the mass region below 1900 MeV*, Eur. Phys. J. A 16 (2003) 229.

LHCb Collaboration, *Amplitude analysis of the D_s+ -> pi- pi+ pi+ decay*, arXiv:2209.09840.

LHCb Collaboration, *Amplitude analysis of the B+ -> pi+ pi+ pi- decay*, Phys. Rev. D 101,
012006 (2020), arXiv:1909.05212. `SigmaPole`, `RhoOmegaMixing` and `PipiKKRescattering`
reproduce this paper's isobar conventions; see `docs/reviews/paper_isobar_conventions.md` for the
numeric reproduction against Laura++ and the published tables, including the remaining unresolved
discrepancies.

## Optional SymPy expressions

`SympyLineshape` is an additional plugin for `Resonance(..., lineshape=...)`.
Existing classes, custom JAX callables, and the default `RelativisticBreitWigner()`
are unchanged. Install the optional dependency with
`python -m pip install -e ".[sympy]"` from the repository root (or
`python -m pip install "jax-pwa[sympy]"` for an installed distribution).
Importing the package or using existing lineshapes does not import SymPy.

```python
import sympy as sp
from jaxpwa import Parameter, RealImag, Resonance, SympyLineshape

m, m0, gamma, alpha = sp.symbols("m m0 gamma alpha", real=True)
mass = Parameter.dynamics("custom.mass", 0.77, owner="custom", bounds=(0.7, 0.85))
width = Parameter.dynamics("custom.width", 0.15, owner="custom", fixed=True)
shape = SympyLineshape(
    sp.exp(-alpha*m**2) / (m-m0-sp.I*gamma/2),
    mass_symbol=m,
    parameters={
        m0: mass,
        gamma: width,
        alpha: Parameter.dynamics("custom.alpha", 0.2, owner="custom", bounds=(0, 2)),
    },
)
component = Resonance(
    "custom", (0, 1), RealImag(1, 0), lineshape=shape,
    mass=mass, width=width, spin=0,
)
```

Bindings use actual SymPy `Symbol` objects. `mass_symbol` receives the per-event
pair mass in GeV; `parameters` maps other symbols to scalar numerical constants
or `Parameter.dynamics` declarations. Fixed values, bounds, steps and backend
aliases follow the existing parameter API. Extra constants have user-defined
units; `alpha` in this example has units GeV^-2. Every free symbol needs exactly
one binding; unused bindings are rejected, except that `mass_symbol` may be
absent from a constant expression. Symbol names must be unique.

Reuse the same mass/width Parameters in `Resonance` to keep its barrier factors
and integration metadata consistent. Alternatively,
`context_symbols={m0: "pole_mass", gamma: "pole_width"}` binds symbols directly to
the resolved `ResonanceContext`. Available fields are `parent_mass`,
`daughter_masses[0]`, `daughter_masses[1]`, `bachelor_mass`, `spin`, `pole_mass`,
`pole_width`, `resonance_radius` and `parent_radius`.

The plugin exposes its parameters to `DecayModel` and resolves current values
without rebuilding the compiled function. Floating parameters must have kind
`DYNAMICS` and an `owner` matching the component name. Each dynamics parameter
currently belongs to one component. The existing cache recomputes affected
amplitudes and normalization matrix entries. Changing an expression or a
parameter's fixed/free status requires a new model/cache. The mapping properties
return copies; the wrapper itself is immutable.

### Evaluation and supported expressions

SymPy 1.14+ generates a JAX function once, with common-subexpression elimination.
Runtime arithmetic and derivatives are JAX operations; the enclosing likelihood
is compiled by the existing minimizer. Output is complex and has the shape of
the input mass array, including for constant expressions. Precision follows the
package's JAX configuration (normally float64/complex128).

Supported operations are `Add`, `Mul`, `Pow` (including `sqrt`), `exp`, `log`,
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `Abs`, `re`,
`im`, and `conjugate`, with finite numeric literals and constants `I`, `pi`, `E`.
`Piecewise`, arbitrary symbolic functions, matrices and unevaluated
integrals/derivatives are rejected. No NumPy/SciPy fallback is used. Construction
accepts trusted scalar SymPy expressions, not strings.

The default evaluates roots/logarithms on real inputs according to JAX rules.
For analytic continuation, `complex_domain=True` converts numerical arguments to
complex before evaluation, selecting the principal branches. It does not choose
a physical Riemann sheet or override simplifications already performed using
SymPy assumptions. Choose assumptions consistent with the intended domain.
Singularities and thresholds can have undefined gradients; floating parameters
and JAX Hessians require the corresponding differentiability. Test gradients
through the normalized likelihood, not only through the raw lineshape.

The expression supplies only the lineshape. The existing external angular,
barrier and identical-particle symmetrization factors remain in `Resonance`.
For a genuinely two-dimensional expression a different `DalitzAmplitude` plugin
would be needed; it is not part of this API. Automatic grid refinement only sees
the declared resonance mass and width; additional narrow structures in a formula
need an explicit normalization-convergence study.

### Saving definitions

`shape.to_spec()` returns a versioned JSON-compatible tree containing symbols,
assumptions, allowed operations, bindings and full Parameter declarations.
`SympyLineshape.from_spec(spec, parameter_registry={p.name: p, ...})` reconstructs
and recompiles it without `eval` or arbitrary formula parsing. A registry reuses
Parameter objects also supplied to `Resonance`; conflicting definitions fail.
This saves initial declarations, not fitted values: save the latter separately.
Compiled kernels, prepared samples and complete `DecayModel` objects are not
serialized by this API.

See [the runnable SymPy tutorial](../notebooks/tutorials/tutorial_11_sympy_lineshapes.ipynb)
for plots, an Asimov parameter-recovery fit, gradient validation and JSON roundtrip.
See also [the B → 3π Dalitz example](../notebooks/tutorials/tutorial_12_sympy_b3pi_dalitz.ipynb)
for a complete symmetrized Dalitz model with a user-defined pole.

### Constant-bin QMI (no interpolation)

`QMI(interpolation="none")` uses `knots` as **mass bin edges**. Supply one
complex value per interval, not per edge, in either Cartesian or polar form:

```python
qmi = QMI(knots=(0.28, 0.50, 0.80, 1.20),
          real_parts=(1.0, 0.7, 0.3), imaginary_parts=(0.0, 0.2, -0.1),
          interpolation="none")
```

Bins are `[left, right)`, with the final edge included in the last bin;
outside the edge range the first/last bin value is held constant. Values
jump at interior edges. AD differentiates the bin amplitudes; the mass
derivative is zero within bins and does not represent a derivative at a jump.
Prepared evaluation reuses bin assignments and grouped reductions for its VJP,
including polar/Cartesian gradients and Hessians. `size` counts edges in this mode.
The existing knot-curvature constraint is not defined for constant bins:
use `strength=0` (nonzero strength raises an error).

Use `Parameter.dynamics` for floating bin values. With a free global complex
coefficient, fix one bin to `1+0j` to remove the redundant scale and phase.
Bins completely removed by vetoes should also be fixed: their parameters have
no likelihood sensitivity. Discontinuous bins require checking normalization
quadrature convergence near their boundaries. No component normalization is
implicitly enabled by selecting this mode.

A typical application uses the 25 reference mass bins of the `B -> 3pi` S-wave with global
CPV coefficients.
