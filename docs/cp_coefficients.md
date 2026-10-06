# Direct-CP coefficient parameterization

Jax-PWA provides `CPRealImag` for charge-conjugate amplitude fits. It is a Cartesian extension of `RealImag` and keeps the CP-averaged amplitude and direct-CP difference in one shared parameter set.

For charge label `q = +1` or `q = -1`,

```text
c_q = (x + q dx) + i (y + q dy).
```

Therefore

```text
c_plus  = (x + dx) + i (y + dy)
c_minus = (x - dx) + i (y - dy).
```

`x` and `y` are the CP-even Cartesian components. `dx` and `dy` are CP-odd differences. Setting `dx = dy = 0` recovers the CP-conserving case.

All four entries may be ordinary numbers or `Parameter.coefficient(...)` objects. The same `Parameter` instances are shared between the two charge models.

## Joint CP likelihood

A direct-CP amplitude fit must preserve both the Dalitz-shape information and the relative rate of the two charge samples. `CPJointNLL` therefore treats charge as part of the fitted sample space.

For coherent charge amplitudes

```text
A_plus(phi)  = sum_j c_plus_j  F_j(phi)
A_minus(phi) = sum_j c_minus_j F_j(phi)
```

define

```text
I_plus  = integral |A_plus|^2 dPhi
I_minus = integral |A_minus|^2 dPhi.
```

The joint signal PDF is

```text
p(phi, +) = |A_plus(phi)|^2  / (I_plus + I_minus)
p(phi, -) = |A_minus(phi)|^2 / (I_plus + I_minus).
```

For unweighted positive- and negative-charge samples the corresponding NLL is

```text
NLL = - sum_plus  log |A_plus|^2
      - sum_minus log |A_minus|^2
      + (N_plus + N_minus) log(I_plus + I_minus).
```

This differs physically from summing two independently normalized Dalitz likelihoods. Independent normalizations would condition on the observed charge counts and remove sensitivity to the integrated charge asymmetry. With `CPJointNLL`, changes in CP coefficients can affect both local interference patterns and the predicted positive/negative yield fractions

```text
P(+) = I_plus  / (I_plus + I_minus)
P(-) = I_minus / (I_plus + I_minus).
```

Within each charge sample all amplitudes are still added coherently before taking the absolute square. The cached normalization uses the full quadratic form `c^dagger M c`, including all interference terms.

When only coefficient parameters float, the dynamical basis and normalization matrices remain cached; only the charge-dependent coefficient vectors and the two quadratic normalizations change.

At least one complex-amplitude convention must be fixed, as in an ordinary amplitude fit, to remove the overall scale/phase degeneracy.


## Derived-observable uncertainties

Post-fit tables that transform `(x, y, dx, dy)` into charge-specific
magnitudes and phases, coefficient asymmetries, or phase differences must
propagate the full HESSE covariance matrix. In particular, the correlations
between all four Cartesian parameters are retained through

```text
Cov(f) = J Cov(x, y, dx, dy) J^T,
```

where `J` is the Jacobian of the derived observables. Angular differences in
the numerical Jacobian are wrapped to the principal interval so that crossing
the `-pi`/`+pi` boundary does not generate a spurious large uncertainty.
Parameters fixed in the fit are absent from the Minuit covariance and are
treated as having zero covariance. Consequently, components fitted with
`dx = dy = 0` fixed have exactly zero propagated uncertainty on their
coefficient `A_CP` and CP phase difference, while their common magnitude and
phase can still carry uncertainty from `x` and `y`.

`jaxpwa.observables.delta_method_covariance`/`delta_method_errors`
implement exactly this `Cov(f) = J Cov(x) J^T` propagation as a reusable
library function, with `J` computed by exact autodiff (`jax.jacrev`) rather
than a hand-rolled finite-difference Jacobian, for any JAX-differentiable
postfit quantity -- not only the Cartesian-to-polar CP observables above.
`CPFitSession.fit_fraction_errors` is the built-in wrapper for fit fractions;
it propagates the *joint* B+/B- covariance in one Jacobian rather than two
independent ones (`plus_model`/`minus_model` share almost every fit
parameter here), which is what makes its `"mean"` entry's cross-term-aware
error correct rather than a naive quadrature sum. See `docs/fitting.md`
("Fit fraction errors") and `docs/catalog.md` ("Delta-method error
propagation").

### Integrated CP asymmetry per component

`CPFitSession.component_cp_asymmetries(result)` returns, for every amplitude
component,

```text
A_CP(k) = (I_k^- - I_k^+) / (I_k^- + I_k^+),   I_k^q = integral |c_k^q A_k^q|^2 dPhi
```

with `I_k^q` from `PreparedAmplitudeCache.component_intensities` (the
numerators of the fit fractions, on the common integration scale both charges
share). It differs from the coefficient asymmetry above whenever a component's
dynamics carry charge-dependent parameters: a QMI S-wave with a fixed global
coefficient and per-node `CPRealImag` nodes has a zero coefficient `A_CP` by
construction, but its integrated `A_CP` is the CP asymmetry of the whole
S-wave. It is physical (efficiency excluded) by default;
`acceptance_weighted=True` weights both integrals by each charge's efficiency.
Errors propagate the joint postfit covariance through both charges in one
Jacobian (NaN when the result has no covariance). With `YieldAsymmetry`, the
integrated amplitude-level asymmetry is still well defined, but it is not the
counting asymmetry of the fitted yields. The calculation respects component
normalization: with `normalize_component=True`, each physical diagonal
integral is one and its raw dynamics scale cancels. To retain an integrated
charge asymmetry from QMI nodes at a fixed global coefficient, use the common
unnormalized component convention (`normalize_component=False`) on both
charges. A component with no free amplitude parameters has zero propagated
error when a covariance exists only for yields or backgrounds.

## Efficiency and background mixtures

`CPJointNLL` also supports efficiency-weighted signal and background while preserving the same joint charge normalization.

With charge-dependent efficiencies `epsilon_q(phi)` define

```text
I_plus^eps  = integral epsilon_plus(phi)  |A_plus(phi)|^2  dPhi
I_minus^eps = integral epsilon_minus(phi) |A_minus(phi)|^2 dPhi.
```

The `PreparedAmplitudeCache` objects passed to `CPJointNLL` must therefore be prepared with the corresponding efficiency values on their normalization samples. Event-by-event efficiency arrays are passed separately to `CPJointNLL`.

For background functions `B_plus(phi)` and `B_minus(phi)`, define

```text
J_plus  = integral B_plus(phi)  dPhi
J_minus = integral B_minus(phi) dPhi.
```

### Non-extended convention

The reference mixture parameter is the **signal fraction**, `f_sig`,

```text
p(phi,+) = f_sig epsilon_plus(phi)|A_plus(phi)|^2
                     / (I_plus^eps + I_minus^eps)
           + (1-f_sig) B_plus(phi)/(J_plus + J_minus)

p(phi,-) = f_sig epsilon_minus(phi)|A_minus(phi)|^2
                     / (I_plus^eps + I_minus^eps)
           + (1-f_sig) B_minus(phi)/(J_plus + J_minus).
```

The total charge probabilities are

```text
P(+) = f_sig I_plus^eps/(I_plus^eps + I_minus^eps)
       + (1-f_sig) J_plus/(J_plus + J_minus)

P(-) = f_sig I_minus^eps/(I_plus^eps + I_minus^eps)
       + (1-f_sig) J_minus/(J_plus + J_minus).
```

Example:

```python
nll = CPJointNLL(
    plus_cache,
    minus_cache,
    plus_efficiency=eff_plus_data,
    minus_efficiency=eff_minus_data,
    plus_background=bkg_plus_data,
    minus_background=bkg_minus_data,
    plus_background_normalization=bkg_plus_norm,
    minus_background_normalization=bkg_minus_norm,
    signal_fraction=signal_fraction,
)
```

`signal_fraction` may be either a number or a fit `Parameter`.

### Extended convention

For an extended likelihood, the mixture is parameterized by component yields rather than fractions:

```text
lambda(phi,+) = N_sig S_plus(phi) + N_bkg B_plus(phi)
lambda(phi,-) = N_sig S_minus(phi) + N_bkg B_minus(phi),
```

where `S_plus + S_minus` and `B_plus + B_minus` each integrate to one over the combined charge-Dalitz sample space. The extended NLL is

```text
NLL_ext = N_sig + N_bkg
          - sum_plus  log lambda(phi,+)
          - sum_minus log lambda(phi,-),
```

up to the parameter-independent factorial constant.

Use

```python
nll = CPJointNLL(
    plus_cache,
    minus_cache,
    plus_efficiency=eff_plus_data,
    minus_efficiency=eff_minus_data,
    plus_background=bkg_plus_data,
    minus_background=bkg_minus_data,
    plus_background_normalization=bkg_plus_norm,
    minus_background_normalization=bkg_minus_norm,
    extended=True,
    signal_yield=signal_yield,
    background_yield=background_yield,
)
```

The total expected number of events is available through

```python
nll.expected_events(values)
```

Signal-only extended fits are also supported by setting `extended=True` and supplying only `signal_yield`.

The non-extended and extended parameterizations are deliberately mutually exclusive: a fit must use either `signal_fraction` or explicit yields, never both.

### Yield-asymmetry parameterization (`YieldAsymmetry`)

By default `signal_yield` is one number/`Parameter` shared by both charges, so
`S_plus`/`S_minus` above (each normalized jointly, `S_plus + S_minus`
integrating to one) alone determine how `N_sig` splits between `N_plus` and
`N_minus` — through the amplitude's own `I_plus`/`I_minus`. This is the
"central invariant" joint-normalization behavior and gives the fit
sensitivity to CP violation through Dalitz-plot interference.

`YieldAsymmetry(total, asymmetry)` is a drop-in replacement for `signal_yield`
that instead makes `N_plus`/`N_minus` independently fittable literal event
counts:

```text
N_plus  = N_s (1 - A_yield) / 2
N_minus = N_s (1 + A_yield) / 2,
```

so `N_plus + N_minus == N_s` for any `A_yield`. `A_yield` (the yield
asymmetry) is unrelated to `CPRealImag`'s coefficient-level `A_CP` discussed
above — the two are independent observables and should not be conflated when
naming fit parameters. Passing `YieldAsymmetry` changes the extended signal
model to

```text
lambda(phi,+) = N_plus  p_plus(phi)  + N_bkg B_plus(phi)
lambda(phi,-) = N_minus p_minus(phi) + N_bkg B_minus(phi),

p_plus(phi)  = |A_plus(phi)|^2  / I_plus
p_minus(phi) = |A_minus(phi)|^2 / I_minus,
```

i.e. each charge's isobar PDF is normalized *on its own*, not jointly. Each
`A_q(phi)` is still the full coherent sum of interfering components, so the
CP-violating *shape* within each charge's own Dalitz plot is preserved;
`A_yield` instead controls the *relative rate* between charges directly,
overriding whatever split `I_plus`/`I_minus` alone would predict. Use this
when the observable of interest is a raw/counting CP asymmetry (e.g. one
contaminated by production or detection asymmetries the amplitude model does
not capture) rather than the interference-driven asymmetry `CPJointNLL`
gives by default. `nll.expected_events(values)` still returns `N_s` exactly
(plus any background yields), and `nll.charge_probabilities(values)` returns
`N_plus/N_s`, `N_minus/N_s` (diluted by background, as usual, when
backgrounds are present).

```python
from jaxpwa import Parameter
from jaxpwa.likelihood import YieldAsymmetry

signal_yield = YieldAsymmetry(
    Parameter("n_signal", 5000.0),
    Parameter("a_cp_yield", 0.0),
)
nll = CPJointNLL(plus_cache, minus_cache, extended=True, signal_yield=signal_yield)
```

`total` and `asymmetry` may be ordinary numbers or `Parameter` objects, and
`CPFitSession(..., signal_yield=signal_yield)` collects their `Parameter`s
the same way it collects a plain `signal_yield` `Parameter`.

### Event-weighted (sWeight/COW) CP fits

`CPFitSession.fit(weights=(plus_weights, minus_weights), covariance=...)`
subtracts background with per-event signal weights (sWeights or COWs from a
fit to a discriminating variable such as the B mass) instead of modelling it
in the Dalitz plot. One weight array per charge, aligned with
`plus_data`/`minus_data`. The objective is the signal-only weighted
estimating function over the joint `(Dalitz, charge)` space,

```text
Q(theta) = - sum_{i in B+} w_i log S_plus(phi_i) - sum_{j in B-} w_j log S_minus(phi_j)
S_q(phi) = eps_q(phi) |A_q(phi)|^2 / (I_plus + I_minus),
```

i.e. the same joint normalization as the unweighted fit. The charge
asymmetry is therefore still measured through `I_plus`/`I_minus`: the
weighted sums `W_plus = sum w_i` and `W_minus = sum w_j` play the role of the
per-charge signal counts. For a single non-resonant component with fixed
`x = 1` the weighted fit reduces exactly to a weighted Bernoulli in charge,
`(1+dx)^2 / [(1+dx)^2 + (1-dx)^2] = W_plus / (W_plus + W_minus)`, which
`tests/test_cp_workflow.py` uses as a closed-form check of the fitted value
and of both corrected covariances.

`covariance` follows `FitSession.fit` (see the
[sWeight / COW section of fitting.md](fitting.md#sweight--cow-dalitz-fits)):
`"sandwich"` (recommended for signed weights) installs the fixed-weight
Godambe covariance with the score-outer matrix summed over both charges,
`"sumw2"` (alias `"sweight"`) the squared-weight Hessian correction, and
`"minuit"` keeps Minuit's uncorrected HESSE. Both share one implementation
with `FitSession`, including the postfit-only `hessian="jax"` behaviour and
the warnings for invalid or Nesterov-endpoint results. Neither correction
propagates the uncertainty of the mass fit that produced the weights.

A weighted session must be signal-only: no `backgrounds`, `signal_fraction`,
`signal_yield`/`YieldAsymmetry` or `extended=True` (the background is what the
weights remove; there is no extended Poisson term). Efficiency and vetoes are
kept: they enter the normalization `I_q`, and the efficiency factor in the
numerator is a parameter-independent additive constant in `log S_q`. Events
with zero weight do not contribute, even where the signal density vanishes.
`session.with_event_weights(plus_weights, minus_weights)` attaches the weights
to the session instead, so the rest of a workflow can stay unchanged: its
`objective` (and therefore `minimizer()`/`fit_multistart()`) is the weighted
one, `fit()` uses the session weights unless `weights=` is passed, and
`plot_projection`/`plot_projection_from_toy`/`prepare_projection_toy` scale the
signal model to the weighted sums `W_plus + W_minus` (split by `I_plus`/`I_minus`)
and draw weighted data histograms with `sqrt(sum w^2)` errors; their pulls use
`(o - e) / sqrt(sum w^2 + sigma_MC^2)`. The binned goodness-of-fit methods
(`goodness_of_fit_projection`, `goodness_of_fit_chi2`) histogram the per-bin
sum of weights and use each bin's sum of squared weights as its variance,
`chi2 = sum (sum w - e)^2 / sum w^2` (`chi2_from_histograms(observed_variance=)`);
that variance is estimated from the bin's own events, so the chi2 law is only
asymptotic and the bins must be well populated. The unbinned
`point_to_point_dissimilarity` assumes unweighted events and raises
`NotImplementedError` on an event-weighted session.

Session construction validates weight shapes, finiteness and incompatible
background/yield settings immediately, before preparing amplitude caches.
The `weights=` argument affects only that fit invocation; use
`with_event_weights` with the same weights to obtain weighted projections
and the matching weighted goodness-of-fit tests afterward.

The low-level equivalent is `CPJointNLL(..., plus_weights=..., minus_weights=...)`,
and `CPJointNLL.signal_log_densities(parameters)` returns the per-event
`(log S_plus, log S_minus)` used by both.

### Amplitude components on projections

`CPFitSession.plot_projection(...)` and `plot_projection_from_toy(...)` accept
`show_amplitude_components=True`. Each signal component `k` is then drawn as
its incoherent contribution `|c_k A_k|^2`, on the signal scale: every rendering
event (weighted phase-space MC or generated toy) keeps its signal weight times
`|c_k A_k|^2 / |sum_j c_j A_j|^2`, so the same folding, selection and binning
apply. The components do not add up to the signal; on a linear y axis the
remainder is drawn as `interference`. The per-component amplitudes come from
`DecayModel.component_amplitudes(data, values)`, whose coherent sum is
`DecayModel.amplitude(data, values)`.

With many components the legend crowds the histograms: `legend_panel=True`
(both methods, `axes=None`) moves it to a separate column on the right of the
figure, shared by the two charges, and `extra_panel=True` adds an empty panel
below it (`figure.extra_axes`) for the caller, e.g. a Dalitz map of the
projected region. `share_y=True` gives the two charges one common y axis
(histograms and pulls), labelled on the left panel only. The return value is
unchanged.

## B± -> K± pi+ pi- tutorial convention

The CP tutorial notebooks use the particle ordering

```text
(1, 2, 3) = (K±, pi±, pi∓)
```

and consistently display

```text
s13 = m^2(K± pi∓)
s23 = m^2(pi+ pi-).
```

The signal-only tutorial uses `CPJointNLL` directly. The efficiency/background tutorial uses `signal_fraction` by default and also shows how to instantiate the extended likelihood with `signal_yield` and `background_yield`. Both tutorials start the minimization from deliberately displaced values and report generated, start and fitted coefficients together with charge-separated fit fractions.

## Physical domain and acceptance validation

`CPJointNLL` rejects invalid initial yields/fractions. Yields must be finite and
non-negative, the signal fraction must lie in `[0, 1]`, and relative background
fractions must be non-negative with sum at most one. During a fit, parameter
points violating these conditions return positive infinite NLL through a
JAX-compatible branch. Use physical parameter bounds and feasible starts;
the domain check is a safeguard, not a replacement for those settings.

The NLL also returns infinity if the resulting density is non-positive or
non-finite at an observed event, or the joint signal normalization is invalid.
Negative densities are no longer silently floored. A zero component yield is
allowed when the remaining components provide a valid total density.

Each charge's efficiency must be a finite, non-negative vector of length equal
to that charge's sample size. Scalars are explicitly expanded. Column arrays
of shape `(N, 1)` are rejected, avoiding accidental broadcasting. CPFitSession
applies equivalent checks to efficiency and veto outputs on data and integration
samples. Supplied background arrays must also match their charge's data size.

For fixed dynamics, CPFitSession reuses model normalization templates separately
for each charge when that charge has neither efficiency nor veto.
