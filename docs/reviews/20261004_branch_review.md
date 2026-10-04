# Branch and working-tree review — 2026-10-04

Reviewed `memory-optimization-2` at `ae7ed7f` against `main` at `49d9777`,
including the three branch commits and the uncommitted implementation present
at the start of the review. Also inspected the analysis notebooks under
`AmAn_B2KKK_Run2/Notebooks`; their measured-data fits were not rerun.

## Reproduced problems and applied fixes

### 1. Floating polar form-factor parameters could be silently fixed

`PolarFormFactorSymNR(Parameter("NR.lambda", 1.2))` was exposed as a free fit
parameter, but coefficient-only preparation cached its shape permanently.
For a B+ -> K+ K+ K- model with 20 phase-space events (seed 3, normalization
resolution 8), differentiating the summed intensity gave **0.0** through the
cache and **-0.0006613820017321026** through direct evaluation.

Floating scales now require `Parameter.dynamics` and their component owner,
consistent with the existing QMI/QMI2D safeguard. Fixed generic parameters
remain supported. The standalone public call now resolves `Parameter`
values as well; previously it attempted to convert the parameter object
itself to a JAX array and raised `TypeError`.

The equivalent expression `lambda^2 / (lambda^2 + s)` also replaces
`1 / (1 + s / lambda^2)`. At lambda=0 and positive invariant masses, the
original value was zero but its derivative was NaN. The new expression
retains the same positive-scale formula with a finite zero-limit derivative.
Numeric configuration rejects non-finite scales.

### 2. Clipping negative Chebyshev polynomials invalidated normalization

For `Chebyshev1D((2.0,), -1, 1)`, the original code integrated the signed
polynomial analytically, then clipped its negative density. Numerical
integration of the returned density was **1.125**, rather than one.

The PDF now checks the endpoints and interior stationary points over its
whole declared support. Invalid coefficient points return NaN everywhere,
including when all observed events happen to lie in positive regions. This
also prevents an invalid polynomial from hiding behind another positive
mixture component. Valid PDFs retain the analytic integral and ordinary JAX
gradients/Hessians. Trailing zero coefficients and the constant polynomial
are covered. The sufficient condition `sum(abs(c)) <= 1` skips root finding
in the common fit domain.

The polynomial and integral conventions agree with the
[ROOT RooChebychev source](https://root.cern.ch/doc/master/RooChebychev_8cxx_source.html).
Rejecting a globally negative polynomial is an explicit Jax-PWA validity
policy, not a change to that polynomial formula.

### 3. Crystal Ball overflow inside declared mass-fit bounds

At mean=0, sigma=1, alpha=0.05, n=100 and interval [-20, 20], the original
tail coefficient `(n/alpha)^n` overflowed. Evaluation at [-10, 0, 10]
returned [0, 0, 0], with a NaN alpha derivative of the summed log-density.

Canceling the common power scale before evaluation, using `log1p`, gives
densities approximately [0.0436577, 0.0718008, 1.38486e-23] and a finite
alpha derivative **12.670905499766743**. Tests cover both tail signs,
normalization and shape gradients against central differences, alongside
the existing SciPy reference checks.

### 4. Recursive mixtures accepted invalid fractions

`SumPDF1D` now rejects non-finite/out-of-range constant fractions and
non-callable components on construction. Fitted fractions outside [0, 1]
produce NaN rather than a signed mixture that might look positive at every
observed event. The recursive RooAddPdf weighting formula is preserved.

### 5. Amplitude projection curves lost interference cancellations

The original `weights / coherent_intensity` ratio was replaced by zero
where coherent intensity vanished. This incorrectly erased individual
nonzero amplitudes in phase-space projections.

For component amplitudes a=[1,1], b=[-1,1], the coherent intensity is [0,4].
On a unit integration measure the correct component curves are [1,1] each,
with interference [-2,2]. The corrected phase-space path uses its known
acceptance-weighted integration measure and preserves these contributions.
Generated signal toys retain their importance-weight ratio.

### 6. Fixed-amplitude CP asymmetries failed with yield-only covariance

When all amplitude parameters were fixed but fitted yields/backgrounds had
a covariance, `component_cp_asymmetries` attempted covariance propagation
through an empty amplitude-parameter list. Such component observables now
have zero propagated uncertainty (and remain NaN for zero total intensity).

### 7. Weighted sessions and regional analysis checks

`CPFitSession` now validates weight shapes, finiteness and incompatible
background/yield configuration on construction, before preparing caches.
Validated weight arrays are cached once. Documentation distinguishes
fit-only `weights=` from session-level `with_event_weights` for projections.

The regional-analysis notebook now rejects invalid 1D mixture points instead
of converting them into a finite log floor, ignores inactive zero-weight
events, and requires a valid fit before calculating/reporting covariance.
The repository guidance now describes the implemented Nesterov/Minuit
continuation policy consistently with `docs/fitting.md` and its tests.

## Validation

- Final focused run: **94 passed** across `test_discriminants.py`,
  `test_polar_form_factor_nr.py`, `test_cp_workflow.py` and
  `test_cp_toy_projection.py`.
- The broader targeted run of cache/dynamic normalization, minimizer,
  workflow and model-I/O tests also passed (200 tests on the initial review
  snapshot).
- Full run: **836 passed, 1 skipped**, 1030.45 seconds (17m10s).
  `test_sympy_lineshape.py` was skipped because the optional SymPy extra is
  absent. The full run started during the final PDF refinements; the focused
  94-test run above validates the final versions of all subsequently changed
  implementation modules. Counts overlap and should not be added together.
- 95 notebooks compiled after IPython input transformation, including the
  analysis notebooks. Compilation is not execution of their fits.
- Ruff passes for the discriminants, polar amplitude and their tests;
  existing lint debt elsewhere was not treated as a numerical failure.
- CPU benchmark: 10,000 events, 1,600 normalization nodes, 5 components,
  8 free coefficients, three repetitions. Compact cache retained; matrix
  diagonal range [0.9999999999999998, 1.0]; warm value+gradient about
  **0.95 ms**. This is a smoke measurement, not a controlled GPU comparison.
- Environment: Apple ARM64 CPU, JAX 0.11.2, NumPy 2.5.3, iminuit 2.33.0,
  x64 enabled.

## Scope and remaining validation boundaries

The weighted joint charge likelihood and its two covariance corrections
pass the existing independent weighted-Bernoulli closed-form checks.
Dynamic-cache normalization and derivative comparisons, chunk balancing,
memory-probe fallback and sequential Hessian checks are exercised by tests.
No changes to their established physics formulas were required here.

Component CP asymmetries respect each component's normalization convention:
normalizing both physical diagonals to one removes their raw dynamics scale
from the integrated asymmetry. The unnormalized common convention is needed
when extracting that asymmetry from QMI dynamics at a fixed coefficient.

Sandwich errors still condition on observed weights and fixed penalties;
they do not include uncertainty from the preceding mass fit or an independent
auxiliary measurement. That existing documented statistical limitation is
not repaired by a finite Hessian or by numerical test success.

GPU memory behavior and measured-data closure were not revalidated on this
CPU host. The NR checks establish the documented formula and its numerical
behavior; direct access to the upstream Laura++ class was blocked by the
hosting site's access protection, so this review does not certify a fresh
line-by-line C++ comparison.
