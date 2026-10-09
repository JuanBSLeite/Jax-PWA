# Fitting and statistical validation

Jax-PWA uses JAX to evaluate the objective and automatic gradient. `iminuit`
remains the default minimizer, and a projected Nesterov prefit is available for
large amplitude fits that benefit from a fast first pass.

## Nesterov prefit and Minuit continuation

`Minimizer.fit()` and both high-level fit sessions accept `method="nesterov"`
or `method="nesterov-minuit"`:

```python
result = session.fit(
    method="nesterov-minuit",
    strategy=2,
    hesse=True,
    nesterov_max_iter=1000,
)
```

`method="nesterov"` returns a `NesterovResult` with the fitted `values`, NLL,
status, and iteration history. It does not provide a covariance matrix. The
`nesterov-minuit` mode uses those values as the starting point for the ordinary
Minuit fit; `strategy=1` and `strategy=2` retain their usual meaning of one or
two MIGRAD stages. Nesterov methods accept strategies 1 and 2 only.

The existing `verbose` levels are shared by both optimizers: 0 is silent, 1
prints stage summaries, 2 prints periodic Nesterov progress, and 3 prints it
more frequently while also increasing Minuit's output level. Values above 3
are accepted for more detailed diagnostic output.

The Nesterov stage uses parameter-step scaling, box projection for declared
bounds, backtracking, and restart after a failed extrapolation. It is a
nonconvex heuristic here: the convex (O(1/k^2)) guarantee does not apply to
the amplitude likelihood, and the result is not a global-minimum guarantee.

Iterations, projected-gradient checks and both backtracking attempts execute
inside one JAX program. Only the final endpoint and history are transferred to
the host. The compiled solver is reused while the objective remains alive;
starting values, parameter scales, bounds and fixed values are runtime inputs.
Changing `nesterov_max_iter` or the parameter-name layout creates a different
program. In the local two-parameter Rosenbrock benchmark, repeated 530-evaluation
runs fell from about 201 ms to 27 ms, while the first JIT call increased from
about 281 ms to 418 ms. Large likelihoods can have a different balance between
objective execution and dispatch overhead.

The Nesterov stage stops with `status="stalled"` after five consecutive
accepted iterations without any NLL decrease. This happens when backtracking
has shrunk the step to floating-point resolution, typically near a point where
the objective is unbounded below and its gradient diverges. The classic case
is an event-weighted NLL with negative weights, where `-w_i log p_i` goes to
`-inf` as the density at a negative-weight event goes to zero. A stalled
endpoint is not a minimum, and a rising projected gradient in the progress
output is the signature of this case.

The Minuit continuation is checked against the Nesterov endpoint. A
non-finite or higher-NLL continuation is rejected and the Nesterov result is
returned. An invalid continuation (call limit reached, EDM above target, or a
failed error matrix) that does not raise the NLL is returned with its invalid
status, exactly as `method="minuit"` would return it: it is at least as good a
point as the Nesterov endpoint, which has no EDM or covariance check of its
own. Within Minuit, the best finite stage is restored if a later MIGRAD
stage, including the final polishing pass, worsens the NLL or becomes
non-finite. A retained Minuit result preserves its values, NLL, validity and
covariance together; resetting values alone would leave stale `FMin` metadata.
This prevents an unstable continuation from becoming
the reported fit, but it does not turn a non-converged Nesterov endpoint into
a valid statistical minimum; inspect `result.valid`, NLL, EDM, and covariance.

In an event-weighted `FitSession.fit` or `CPFitSession.fit(weights=..., covariance="sandwich"|"sumw2")`,
a returned Nesterov endpoint has no covariance to correct: the session warns
and skips the corrected covariance, as it already does for an invalid Minuit
result.

## NLL and Minuit convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`.

`Minimizer` uses a default tolerance of `1e-4`. Fit validity must not be judged from `valid` or EDM alone: always compare the fitted NLL with known reference points in closure tests and inspect pulls/covariance quality.

### Square-Dalitz coordinates

`coordinates="square-dalitz"` on `FitSession`/`CPFitSession` fits the events as points of
`(m', theta')` instead of the Dalitz plane: the NLL changes by the parameter-independent
`-sum_i w_i log |J_i|`, so fitted values and errors are unchanged. Requirements (Square-Dalitz
model normalization, dimensionless efficiencies, Dalitz-plot background densities) are in
[square_dalitz.md](square_dalitz.md#fitting-in-square-dalitz-coordinates).

## Normalization samples

By default, amplitude-component and PDF normalization integrals use mass-plane
Gauss--Legendre or Square-Dalitz quadrature. An external `normalization_sample`
selects `toy-mc` and is reused unchanged during the fit. See
[Monte Carlo normalization](mc_integration.md) for proposal weights and selection.

The default model configuration is

```python
model = DecayModel(
    channel,
    components,
    normalize_components=True,
    normalization_method="gauss-legendre",
    normalization_bin_width=0.005,
)
```

The quadrature is constructed lazily and reused for the lifetime of the model.
This mass-plane method is the same base Gauss--Legendre prescription used by
Laura++.

The normal fit workflow is simply

```python
data = model.generate_phase_space(...)
cache = model.prepare_cache(data)
```

where `generate_phase_space()` is used only to generate event/proposal samples. `prepare_cache()` uses the model-owned normalization sample unless an explicit integration sample is supplied.

For the default method, convergence should be checked by changing
`normalization_order_m13` and `normalization_order_m23`. For
`normalization_method="square-dalitz"`, change `normalization_resolution`.

```text
400 -> 600 -> 800 -> 1000 -> 1200.
```

## Component normalization convention

Every dynamical component is normalized by default according to

```text
integral dPhi |F_j|^2 = 1.
```

The normalization is applied before multiplication by the complex coefficient:

```text
A = sum_j c_j F_j_normalized.
```

For a floating dynamical parameter the scale is recalculated at the current parameter point using the deterministic grid:

```text
F_j(x;theta)
 -> F_j(x;theta) / sqrt(integral_grid |F_j(theta)|^2).
```

Detector efficiency is deliberately excluded from individual component normalization and enters only the total signal-PDF normalization.

Raw components can be requested explicitly with

```python
DecayModel(..., normalize_components=False)
```

To preserve the global convention while leaving one component unscaled, set
`normalize_component=False` on its `Resonance`, `NonResonant`, or
`DalitzAmplitude` declaration:

```python
qmi_component = Resonance(
    "S_wave_QMI",
    pair=(0, 2),
    coefficient=qmi_coefficient,
    lineshape=qmi,
    normalize_component=False,
)
```

The raw component is still included in the full Hermitian normalization matrix,
including its interference rows and columns, and therefore in the total PDF
normalization. `normalize_component=None` (the default) inherits the model-level
`normalize_components` setting; an explicit boolean overrides it.

## Cached normalization matrix

For linear complex coefficients,

```text
A(x) = sum_i c_i F_i(x)
```

and

```text
N(c) = integral |A|^2 dPhi = c^dagger M c,
```

with

```text
M_ij = integral conj(F_i) F_j dPhi.
```

For coefficient-only fits the component values and the full Hermitian normalization matrix are cached once. Changing magnitudes/phases therefore does not reevaluate lineshapes or reintegrate the Dalitz plot.

When a resonance mass, width, or another dynamical parameter floats, only the affected component and the corresponding normalization-matrix row/column are reevaluated.

## Fit-performance path

Parameter-independent event kinematics are prepared once:

```text
m_ij
p*
p
q
cos(theta)
```

During repeated likelihood evaluations only genuinely parameter-dependent quantities are recomputed, such as

```text
q0, p0
Blatt-Weisskopf pole factors
running width
lineshape
component normalization
interference row/column.
```

JAX has a one-time compilation cost on the first objective/gradient evaluation. Timing studies should distinguish cache preparation, first JIT compilation, steady-state likelihood evaluation, and full Minuit runtime.

## Gradient validation

`Minimizer.check_gradient()` compares the same JAX gradient supplied to Minuit with central finite differences:

```python
gradient_check = minimizer.check_gradient(
    start_values,
    step_scale=1e-5,
    print_table=True,
)
```

This should be used when introducing a new dynamical parameter or lineshape.

## sWeight / COW Dalitz fits

`FitSession.fit()` accepts per-event signal weights directly
(`CPFitSession.fit(weights=(plus_weights, minus_weights), ...)` is the joint
B+/B- equivalent with the same covariance options; see
[cp_coefficients.md](cp_coefficients.md#event-weighted-sweightcow-cp-fits)):

```python
result = session.fit(
    weights=sweights,
    covariance="sandwich",
    strategy=1,
    hessian="jax",
)
```

### Weighted estimating equation

The fitted point is obtained from the weighted objective

```text
Q(theta) = -sum_i w_i log p(x_i; theta),
```

equivalently from the weighted score equation

```text
U(theta) = sum_i w_i s_i(theta) = 0,
s_i(theta) = d log p_i(theta) / d theta.
```

Finite negative weights are allowed. With negative sWeights/COW weights, `Q`
is best regarded as an M-estimation / pseudo-likelihood objective rather than
a literal event probability likelihood. The weight multiplies the log-PDF; it
is never inserted inside the logarithm. A weighted `FitSession` must therefore
be signal-only: do not simultaneously configure an explicit Dalitz background
mixture, signal fraction/yield, or extended likelihood.

### Nominal covariance: Godambe / sandwich

For fixed event weights, define the sensitivity matrix at the fitted point
`theta_hat`

```text
A = -sum_i w_i d2 log p_i / dtheta dtheta^T
```

and the empirical score-variability matrix

```text
B = sum_i w_i^2 s_i s_i^T.
```

The Godambe/sandwich covariance is

```text
C_sandwich = inv(A) B inv(A).
```

This is the asymptotically correct covariance for the weighted estimating
equation under the usual regularity assumptions for fixed weights. In
particular, `B` is a score outer product, not a second Hessian. This distinction
matters because the information identity need not survive event weighting.
`covariance="sandwich"` implements this expression and is the recommended
fixed-weight covariance for signed sWeight/COW amplitude fits.

Jax-PWA computes `B` without materializing the huge event-by-parameter score
matrix. At `theta_hat` it defines the auxiliary scalar

```text
R(theta) = 0.5 * sum_i w_i^2
           * [log p_i(theta) - log p_i(theta_hat)]^2.
```

All residuals vanish at the reference point, therefore

```text
Hessian[R](theta_hat) = sum_i w_i^2 s_i s_i^T = B.
```

The existing memory-aware JAX Hessian/HVP backend can thus evaluate `B`
directly, including large floating-QMI fits, without constructing an
`N_events x N_parameters` Jacobian.

### SumW2 / squared-weight Hessian alternative

For compatibility with common RooFit-style weighted-error prescriptions,
Jax-PWA also provides

```text
H_w2 = -sum_i w_i^2 d2 log p_i / dtheta dtheta^T
C_sumw2 = inv(A) H_w2 inv(A).
```

Use `covariance="sumw2"` for this prescription. The historical
`covariance="sweight"` spelling is retained as a backwards-compatible alias
for `"sumw2"`. The squared-weight Hessian expression is commonly used but is
not generally identical to the Godambe covariance; equality requires an
additional score/Hessian information-identity relation.

Gaussian constraints are fixed penalties, not event fluctuations: they enter
the sensitivity matrix `A` but neither `B` nor `H_w2`. For a weighted
Gaussian location fit with a constraint of width `s`,
`A = sum_i w_i + 1/s^2` and `C_sumw2 = sum_i w_i^2 / A^2`.

A zero-weight event contributes exactly zero to the weighted objective, even
at a physical PDF zero where `log p_i = -inf`. Gradients stay finite there
only because Jax-PWA's signal log-densities use a safe logarithm; a custom
`logpdf` passed to `WeightedUnbinnedNLL` must do the same.

### JAX Hessian behavior during minimization

For all corrected weighted covariance modes (`"sandwich"`, `"sumw2"`, and
the `"sweight"` alias), `hessian="jax"` is intentionally **postfit-only**.
A signed-weight objective can have an indefinite exact Hessian away from its
minimum. Passing that matrix into MIGRAD's seed/search metric can trigger
negative-curvature recovery and forced positive-definite error matrices even
when the objective and gradient are finite. Jax-PWA therefore:

1. minimizes with the JAX gradient and Minuit's numerical search curvature;
2. evaluates the exact memory-aware JAX matrix `A` at `theta_hat`;
3. evaluates either `B` (sandwich) or `H_w2` (SumW2) with JAX;
4. replaces `result.covariance` and `result.errors` by the chosen corrected
   covariance.

If the optimizer result is invalid or its objective is non-finite, the corrected
covariance is skipped with a `RuntimeWarning`. The invalid result is returned
for diagnosis; any covariance it already contains is the uncorrected optimizer
estimate, not a sandwich result. Correcting errors cannot repair a failed fit.

The low-level `Minimizer` and ordinary `covariance="minuit"` fits retain the
usual behavior in which `hessian="jax"` can be supplied directly to MIGRAD.

### Constraints and weight uncertainty

The sandwich `A` matrix contains the curvature of the complete fitted
objective, including deterministic penalty/regularization terms. The `B`
matrix is built from event score fluctuations only. This is appropriate for a
fixed regularization penalty. A Gaussian constraint representing an actual
independent auxiliary measurement has its own sampling uncertainty and should
be modeled jointly or propagated separately if that contribution matters.

Likewise, the formulas above condition on the observed event weights. They do
not automatically propagate uncertainty from the mass fit or other procedure
used to determine sWeights/COWs. This is an important distinction for our
current two-stage mass-fit -> COW -> Dalitz-fit workflow. Langenbruch derives
additional nuisance/weight-estimation terms, while Dembinski et al. formulate
the complete two-step M-estimator and its sandwich covariance in Eqs. (33--35).
Implementing those first-stage terms would require carrying the mass-fit yield
and shape estimating equations (or their covariance/cross-derivatives) into the
Dalitz covariance. The current `"sandwich"` mode is therefore the **fixed-weight
Godambe covariance**, not yet the full two-step COW covariance. Toy or bootstrap
studies remain advisable for final coverage validation.

The usual sPlot assumption that the discriminating variable and control/Dalitz
variables are sufficiently independent within each species still applies to
ordinary sWeights. COWs generalize the weighting construction and can relax
that factorization under their stated conditions.

### Experimental event-dependent log barrier

The [COW log-barrier notebook](../notebooks/examples/ds_pipipi_lhcb2023_goofit_real_data_fit_cow_sweights_logbarrier.ipynb)
contains a notebook-local experiment, not a new public constraint API. It adds
`lambda * max(-w_i, 0) * h_i**2` to each event's weighted loss, where
`h_i = max(0, log(epsilon * q_i / p_i))`. The fixed reference density `q` is a
constant phase-space intensity multiplied by the same acceptance, normalized
with `mean(normalization_sample.weights * acceptance)` in the model's measure.

Unlike a fixed parameter-only penalty, this barrier changes the event score
fluctuations as well as the total Hessian. Its fixed-weight sandwich uses

```text
ell_i = -w_i log p_i + lambda max(-w_i, 0) h_i^2
u_i   = w_i + 2 lambda max(-w_i, 0) h_i
g_i   = grad(ell_i) = -u_i grad(log p_i)
A     = Hessian(sum_i ell_i)
B     = sum_i g_i g_i^T
C     = inv(A) B inv(A)^T
```

After a valid fit, the notebook evaluates `A` from the complete objective and
`B` as the Hessian of `0.5 * sum_i u_i(theta_hat)**2 *
[log p_i(theta) - log p_i(theta_hat)]**2` at `theta_hat`. Only this auxiliary
calculation freezes the effective score weights and reference log-PDF. It
uses `Minimizer.jax_hessian()` and the existing memory-aware HVP backend,
without constructing an event-by-parameter Jacobian. At zero barrier strength
this reduces to the ordinary fixed-weight sandwich.

`COMPUTE_BARRIER_COVARIANCE=True` enables the correction for the nominal fit
and optional scan. Successful correction replaces `result.covariance` and
`result.errors`, and enables fit-fraction error propagation.
`barrier_covariance_reports[id(result)]` records the correction status, `A`,
`B`, corrected and original covariance matrices, and the condition number of
`A`. Invalid fits or unusable matrices retain explicitly uncorrected optimizer
errors; no ridge or pseudoinverse is used to force a covariance. Running HESSE
or MIGRAD again can overwrite the correction.

This estimates local asymptotic variance conditional on the COW weights,
reference density and regularization hyperparameters. It excludes the mass-fit
and weight-estimation uncertainty, data-driven hyperparameter selection, and
regularization bias. Parameter boundaries or hinge thresholds can invalidate
ordinary local covariance assumptions. Toy/full-bootstrap coverage validation
is still required for a final measurement.

### References

- M. Pivk and F. R. Le Diberder, *sPlot: a statistical tool to unfold data
  distributions*, Nucl. Instrum. Meth. A **555** (2005) 356,
  arXiv:physics/0402083.
- C. Langenbruch, *Parameter uncertainties in weighted unbinned maximum
  likelihood fits*, Eur. Phys. J. C **82** (2022) 393,
  doi:10.1140/epjc/s10052-022-10254-8, arXiv:1911.01303. See especially the
  asymptotically correct weighted covariance and the discussion of sWeight
  nuisance-parameter uncertainty.
- H. Dembinski, M. Kenzie, C. Langenbruch and M. Schmelling, *Custom
  Orthogonal Weight functions (COWs) for event classification*, Nucl. Instrum.
  Meth. A **1040** (2022) 167270,
  doi:10.1016/j.nima.2022.167270, arXiv:2112.04574. The paper treats COWs,
  sWeights as a special case, and asymptotic covariance for parameters fitted
  to weighted control-variable distributions.

## Slow HESSE in large fits

`MnHesse: Using analytical gradient but a numerical Hessian calculator` means
Minuit received the JAX gradient but is estimating second derivatives with finite
differences. With many free QMI knots this requires quadratically many objective
evaluations. Strategies 1 and 2 may invoke HESSE inside MIGRAD, so `hesse=False`
only skips the final explicit HESSE call.

With iminuit >= 2.32, automatic second derivatives can be selected explicitly:

```python
result = session.fit(
    strategy=1,
    hessian="jax",
    hesse=True,
    verbose=1,
)
# Also supported by CPFitSession and both session.fit_multistart() methods.
# Low-level API:
minimizer = Minimizer(
    nll,
    parameters,
    hessian="jax",
    verbose=1,
)
```

`hessian="numerical"` remains the library default. For ordinary fits, `"jax"` also supplies the
Hessian during MIGRAD; Minuit still handles bounds, covariance inversion and the
`errordef` scaling. The exception is `FitSession.fit(..., covariance="sweight")`,
where `"jax"` is reserved for the postfit squared-weight covariance as described
above. It requires an objective differentiable twice. For floating
dynamics, Hessian-vector products are evaluated sequentially, which minimizes
peak memory. This supports QMI's custom VJPs
without requiring the full event-by-parameter batch. This reduces repeated
likelihood probes but does not guarantee a faster fit: second-order JIT
compilation and retained intermediate arrays can be expensive. Compilation is
lazy and reused across fits of the same session. `verbose=1` reports times and
function/gradient counts for each SIMPLEX, MIGRAD and final HESSE stage; stage
times include any first-use compilation.

The separate `Analytical calculator ... numerical ... g2 ...` warning compares
gradients. Near a minimum, cancellation in differences of large NLL values can
make this comparison unreliable. It does not by itself establish that automatic
gradients are wrong. Use `check_gradient()` at an interior parameter point with
several step sizes (for example `1e-4`, `1e-5`, `1e-6`), inspect absolute errors,
and ensure x64 is enabled. Automatic Hessians do not repair an incorrect custom
gradient, a nonsmooth objective, or an unidentified fit direction.

Upstream references: [iminuit derivatives and strategies](https://scikit-hep.org/iminuit/reference.html),
[Minuit HESSE implementation](https://root.cern.ch/doc/master/MnHesse_8cxx_source.html).

## RealImag coefficients

The supported complex coefficient parameterization is

```text
c = x + i y
```

through `RealImag`. One complex coefficient is normally fixed to remove the arbitrary global amplitude scale and phase.

## Floating dynamical parameters

Mass, width and other dynamics quantities may be `Parameter` objects. Dynamics parameters must have an `owner` equal to their amplitude-component name.

```python
mass = Parameter.dynamics(
    "rho.mass",
    0.760,
    owner="rho",
    bounds=(0.73, 0.81),
)
width = Parameter.dynamics(
    "rho.width",
    0.180,
    owner="rho",
    bounds=(0.10, 0.22),
)
```

Meson/Blatt-Weisskopf radii are fixed by default unless explicitly promoted to fit parameters.

## Multistart minimization

```python
minimizer = Minimizer(nll, model.parameters, verbose=1)
scan = minimizer.fit_multistart(
    n_starts=20,
    seed=314159,
    include_default=False,
    simplex=False,
)
result = scan.best
```

For the standard closure notebooks we intentionally use one randomized start unless the study is explicitly about multistart robustness.

Useful diagnostics are

```text
validity
EDM
NLL(truth)
NLL(start)
NLL(fit)
NLL(fit) - NLL(truth)
pulls
covariance/correlation matrix.
```

A result with tiny EDM but `NLL(fit)` far above `NLL(truth)` is a failed closure even if Minuit reports `valid=True`.

## Closure criterion

For each floating coordinate,

```text
pull = (value_fit - value_gen) / sigma_fit.
```

A single pseudoexperiment is a closure check, not a bias measurement. Bias requires an ensemble of toys.

## E791 examples and conventions

The E791 notebooks use the Fit-2 resonance content for

```text
D+ -> pi- pi+ pi+.
```

Historical E791 three-pion analyses used effective Blatt-Weisskopf radii

```text
parent_radius = 3.0 GeV^-1
resonance_radius = 3.0 GeV^-1.
```

The project RBW convention is

```text
1 / (m0^2 - m^2 - i m0 Gamma).
```

With `rho(770)=1+0i` retained as the reference coefficient, the E791 examples account for the propagator-sign convention by shifting the constant non-resonant phase by 180 degrees.

The step-by-step fitting workflow is
`notebooks/tutorials/tutorial_02_first_fit.ipynb`, with efficiency and backgrounds added in
`notebooks/tutorials/tutorial_04_acceptance_and_backgrounds.ipynb`.

## Fit fractions

After a fit, convert the Minuit values to a mapping and print the fractions:

```python
fit_values = {name: float(result.values[name]) for name in result.parameters}
model.print_fit_fractions(fit_values, include_interference=True)
```

By default this reports physical fractions. Pass the same efficiency callable
used in the likelihood through `efficiency=...` to report acceptance-weighted
fractions. The returned dictionary stores fractions as numbers rather than
percentages; the printed table uses percentages.

### Fit fraction errors (delta method)

`fit_fractions()`/`print_fit_fractions()` only report central values --
`FF_i = Re(conj(c_i) M_ii c_i) / (c^dagger M c)`, a nonlinear function of the
coefficient vector `c` and the Hermitian normalization matrix `M_ij =
integral conj(F_i) F_j dPhi` (see "Normalization: the central invariant" in
`CLAUDE.md`). `M` itself is a function of every floating `DYNAMICS`
parameter, not just the coefficients: a component's mass, width, or (for
`QMI`) every knot magnitude/phase all change `M_ii`/`M_ij` and therefore
every fit fraction, including other components' fractions through the shared
denominator `c^dagger M c`. `fit_fraction_errors` reports the standard error
on this nonlinear function via the delta method, the standard first-order
error-propagation technique also implicit in Minuit's own HESSE parameter
errors: expand `f(theta)` (here the vector of fit fractions) to first order
around the postfit point `theta_hat`,

```text
f(theta) ~= f(theta_hat) + J (theta - theta_hat),   J = df/dtheta |_theta_hat
Cov(f)   ~= J Cov(theta_hat) J^T
```

and take `sigma(f_i) = sqrt(Cov(f)_ii)`. `Cov(theta_hat)` is Minuit's postfit
covariance (`result.covariance` after HESSE, from `Minimizer.fit`) -- the
same curvature-based, locally-Gaussian estimate already used for every plain
parameter's `sigma` printed elsewhere in this document. Propagating it
through fit fractions this way is therefore no less (and no more) justified
than trusting those parameter errors themselves; see "Caveats" below for when
that stops being a good approximation.

**How the Jacobian is computed.** Earlier revisions of this kind of
propagation (and the still-common approach elsewhere in the field) estimate
`J` by finite differences: nudge each parameter by a small step, recompute
`f`, and take a symmetric difference quotient -- exactly what
`docs/cp_coefficients.md`'s hand-rolled CP-observable Jacobian still does,
since that one differentiates a plain NumPy closed-form expression outside
JAX. `fit_fraction_errors` instead differentiates
`PreparedAmplitudeCache.fit_fractions` -- a pure JAX function of the full
parameter mapping, exactly like the objective itself -- with reverse-mode
autodiff (the same VJP construction as
`jaxpwa.observables.delta_method_jacobian`), giving `J`
to floating-point precision with no step-size tuning and no truncation
error. Reverse mode is also the efficient *direction* here: a fit fraction
vector has far fewer entries (one per component) than a QMI-heavy model has
free parameters -- the S-wave QMI of a `B+ -> K+K-K+` CP fit alone contributes on
the order of a hundred floating knot magnitudes/phases -- so the cost scales
with the output count, not with how many parameters float.

That said, `delta_method_jacobian` does *not* use `jax.jacrev`'s default
`vmap`-batched sweep over every output row at once: it takes one
`jax.vjp` linearization pass, then loops one cotangent-basis row through the
resulting backward function per output, in an ordinary Python loop. For a
handful of outputs this is a minor difference in *speed* -- but it is not a
minor difference in *memory*. `fit_fractions`'s reverse pass touches the
normalization-matrix computation for every floating dynamics parameter (a
QMI knot, in that `B+ -> K+K-K+` fit), which is comparatively heavy
because it is built from the full normalization sample; `vmap`-batching
that backward pass across every output row at once multiplies an already
sizeable per-row intermediate by the output count *simultaneously*, which
was enough to exhaust GPU memory outright for that notebook's joint B+/B-
Jacobian (16 outputs) before the loop replaced it. The loop keeps peak
memory bounded to one row's cost regardless of output count, trading it for
`len(output)` sequential backward passes instead of one vectorized one --
worthwhile whenever outputs are few and each backward pass is heavy, as
here.

The built-in fit-fraction wrappers compile their sequential VJP loop with
`jax.lax.map`. Each model retains the executable for its parameter layout;
normalization arrays and current parameter values are passed as runtime inputs,
so later calls reuse compilation while reading fresh values, integration weights
and efficiencies. The executable does not retain the large integration arrays.
The fraction cache prepares just one event on its unused data side, while keeping
the **entire normalization sample** and its weights. The general-purpose
`delta_method_jacobian` helper retains its eager Python loop.

```python
errors = model.fit_fraction_errors(fit_values, result.covariance)
```

Internally this resolves `fit_values` against every `Parameter` in
`model.parameters` (so a partial dict falls back to each parameter's own
declared value, exactly like `fit_fractions()`), builds the same physical- or
acceptance-weighted `PreparedAmplitudeCache` `print_fit_fractions()` would
(`efficiency=...` selects the convention), differentiates its
`fit_fractions(values)` with respect to every non-fixed parameter by default,
and propagates `result.covariance` through that Jacobian. `result.covariance`
can be passed directly: the same helper as `delta_method_covariance` looks it up by parameter
name (`covariance[a, b]`) rather than assuming a particular positional order,
so which parameters happen to be free/fixed, or in what order Minuit stores
them, does not matter -- see "Pitfall" below for why that name lookup is
mandatory, not just convenient. `FitSession` exposes the same call as
`session.fit_fraction_errors(result)`.

**The CP joint case.** `CPFitSession.fit_fraction_errors(result)` does not
call the single-model version twice. `plus_model`/`minus_model` typically
share almost every fit parameter (every `CPRealImag` coefficient, every
dynamics parameter -- a QMI knot or a `GaussianConstraint`-anchored mass is
literally the same `Parameter` object for both charges), so `FF_plus` and
`FF_minus` are correlated, and that correlation matters for anything derived
from both of them together. The joint Jacobian is mathematically equivalent to
stacking both charges' fraction vectors and differentiating them together:

```python
def joint_fractions(values):
    return jnp.concatenate([plus_cache.fit_fractions(values), minus_cache.fit_fractions(values)])

J = delta_method_jacobian(joint_fractions, values, parameter_names)   # shape (2n, m)
full_covariance = J @ covariance_matrix @ J.T                         # shape (2n, 2n)
```

The built-in implementation evaluates each charge's Jacobian in its own
reusable compiled program, always using the same *union* of both models' free
parameter names. Stacking those rows gives the same `(2n, m)` Jacobian without
propagating zero cotangents through the other charge's integration graph. Its
top half is sensitivities of the `n`
B+ fractions and bottom half of the `n` B- fractions to the *same* `m`
parameters. `J @ C @ J.T` then gives the full `(2n, 2n)` covariance in one
step: its top-left and bottom-right `n x n` blocks are `Var(FF_plus)` and
`Var(FF_minus)` (what `errors["plus"]`/`errors["minus"]` report), and its
off-diagonal block's diagonal is `Cov(FF_plus_i, FF_minus_i)` for each
matching component `i`. That cross term is exactly what makes
`errors["mean"]` more than guesswork:

```text
FF_mean = 0.5*(FF_plus + FF_minus)
Var(FF_mean) = 0.25*(Var(FF_plus) + Var(FF_minus) + 2*Cov(FF_plus, FF_minus))
```

```python
errors = session.fit_fraction_errors(result)
errors["plus"]["rho0_770"]   # sigma(FF_plus) for one component
errors["minus"]["rho0_770"]  # sigma(FF_minus)
errors["mean"]["rho0_770"]   # sigma(FF_mean), including Cov(FF_plus, FF_minus)
```

Skipping the cross term (i.e. treating the two charges as independent and
adding their variances in quadrature) would silently overstate or understate
`sigma(FF_mean)` whenever `Cov(FF_plus, FF_minus)` is non-negligible -- which
it generally is here, precisely because the two charges share parameters
(see `docs/cp_coefficients.md`, "Derived-observable uncertainties").

**Pitfall: covariance ordering.** `delta_method_covariance`'s internal
`_covariance_matrix` helper always tries name-pair indexing
(`covariance[a, b]` for every `a, b` in the requested parameter names)
*before* ever treating `covariance` as a plain positionally-ordered array,
and only falls back to the latter if name indexing itself raises. This is
deliberate, not defensive boilerplate: an early version tried the array
interpretation first whenever `covariance`'s own shape matched
`(len(names), len(names))`, reasoning that a same-size object was "probably
already a dense matrix ordered like `names`". That reasoning breaks silently
for exactly the common case of requesting *every* free parameter, because
`iminuit`'s `Matrix` is itself array-convertible (`jnp.asarray(covariance)`
succeeds) *and* commonly has that exact same total dimension -- so the shape
check would "accidentally" pass and the code would use Minuit's own internal
parameter order instead of the caller's `parameter_names` order. The bug is
subtle to notice from symptoms alone: each fraction's *own* variance
(`v^T C v` for a single row `v` of `J`) is unaffected by a self-consistent
wrong permutation of `C`, so `errors["plus"]`/`errors["minus"]` still came
out correct -- only the *cross*-covariance between two different rows (e.g.
`Cov(FF_plus, FF_minus)`, or `Cov` between two different components) was
silently wrong. `tests/test_delta_method.py` has a regression test for this
exact scenario.

**Caveats.** This is a linear (Gaussian) approximation, not an exact
propagation:

- It is only as good as `Cov(theta_hat)` itself -- if HESSE did not run, did
  not converge, or `result.valid` is `False`, treat the propagated errors
  with the same skepticism as the raw parameter errors they are built from.
- It can understate the true uncertainty where `f(theta)` is strongly
  nonlinear near `theta_hat`, or where a fraction sits close to its hard
  lower bound of zero (a Gaussian is symmetric; a fraction is not, once its
  uncertainty band would otherwise cross zero). MINOS-style asymmetric scans
  or toy/bootstrap refits are the more robust (and far more expensive)
  alternative in that regime.
- Gaussian-constrained parameters (`GaussianConstraint`, e.g.
  `rho0_1450`/`rho0_1700`/`phi1020`/`chic0` mass and width in
  the `B+ -> K+K-K+` QMI CP fit) need no special handling: the constraint is
  part of the NLL that HESSE differentiates, so its effect on
  `Cov(theta_hat)` -- and hence on every propagated fit-fraction error -- is
  already included automatically, whether or not that parameter is fixed.

`jaxpwa.observables.delta_method_errors`/`delta_method_covariance`
are general-purpose: they work for any JAX-differentiable postfit quantity,
not just fit fractions (interference fractions, the CP-observable table in
`docs/cp_coefficients.md`, or any custom derived observable) -- see
`docs/catalog.md` ("Delta-method error propagation").
