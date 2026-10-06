# Time-dependent neutral-meson Dalitz fits

`NeutralMesonMixing` and `TimeDependentDalitzNLL` implement an unbinned,
signal-only likelihood for `(Dalitz coordinates, decay time)`, conditional on
initial-flavour tag and, when supplied, per-event time uncertainty. Numerical
operations and derivatives are JAX; use the existing `Minimizer` directly.
This is a separate API from the charge-joint `CPJointNLL` for charged mesons.

## Conventions and reference

The numerical reference is Belle, **Phys. Rev. D 89, 091103 (2014)**,
[arXiv:1404.2412](https://arxiv.org/abs/1404.2412), specifically the **printed
rates in Eqs. (1–2)**. Let `u=t/tau`, `r=q/p`, `C=r*conj(A)*Abar`. The D0 rate is

```
exp(-u)/2 * [
    (|A|²+|r*Abar|²)*cosh(y*u)
  + (|A|²-|r*Abar|²)*cos(x*u)
  + 2*Re(C)*sinh(y*u) - 2*Im(C)*sin(x*u)
]
```

For D0bar, swap `A` and `Abar` and replace `r` with `1/r`. These interference
signs are part of the API; do not substitute a different eigenstate/g-minus
convention without transforming the parameters consistently.

`x`, `y` are fractions (Belle CP-conserving central values: `0.0056`, `0.0030`),
`q_over_p` is the positive magnitude, `phi` is its phase in radians, and `tau`
and all times share units (the example uses ps). Require `tau>0`, `|y|<1`,
`q_over_p>0`. Fields accept floats or `Parameter`s. Put physical bounds on
floating parameters. Invalid evaluations return infinite NLL.

## Mathematical construction

Let $z=(s_{12},s_{13})$ denote a point in the Dalitz plot, and let
$A(z)$ and $\bar A(z)$ be the coherent amplitudes for the two flavour states.
The implementation uses

$$
r = \frac{q}{p} = |r|e^{i\phi},\qquad u=\frac{t}{\tau},
$$

and the two mixing functions

$$
g_+(t)=\frac{1}{2}\left[
e^{(-1+y+i x)u/2}+e^{(-1-y-i x)u/2}
\right],
$$

$$
g_-(t)=\frac{1}{2}\left[
e^{(-1+y+i x)u/2}-e^{(-1-y-i x)u/2}
\right].
$$

Thus, for an initially produced $D^0$ or $\bar D^0$, the time-dependent
amplitudes are

$$
\mathcal A_{D^0}(z,t)=g_+(t)A(z)+r\,g_-(t)\bar A(z),
$$

$$
\mathcal A_{\bar D^0}(z,t)=g_+(t)\bar A(z)+r^{-1}g_-(t)A(z).
$$

The corresponding unnormalised rates can be written as

$$
R_{+}(z,t)=|g_+|^2|A|^2+|g_-|^2|r\bar A|^2
 +2\operatorname{Re}\!\left[g_+^*g_-\,r\,A^*\bar A\right],
$$

$$
R_{-}(z,t)=|g_+|^2|\bar A|^2+|g_-|^2|r^{-1}A|^2
 +2\operatorname{Re}\!\left[g_+^*g_-\,r^{-1}\,\bar A^*A\right].
$$

Equivalently, defining

$$
C=rA^*\bar A,
$$

the first rate is the Belle expression

$$
R_{+}(z,t)=\frac{e^{-u}}{2}\left[
(|A|^2+|r\bar A|^2)\cosh(yu)
+(|A|^2-|r\bar A|^2)\cos(xu)
+2\operatorname{Re}(C)\sinh(yu)
-2\operatorname{Im}(C)\sin(xu)
\right].
$$

$R_-$ is obtained by exchanging $A\leftrightarrow\bar A$ and replacing
$r\to r^{-1}$. This is the sign convention implemented by
`NeutralMesonMixing.basis`; changing the definition of $g_-$ requires changing
the amplitude convention and the interference terms together.

### Dalitz acceptance and conditional normalisation

With perfect time resolution and unit temporal acceptance, a Dalitz efficiency
$\epsilon_D(z)$ gives the tag-conditional PDFs

$$
p_{+}(z,t)=\frac{\epsilon_D(z)R_{+}(z,t)}{N_{+}},
\qquad
p_{-}(z,t)=\frac{\epsilon_D(z)R_{-}(z,t)}{N_{-}},
$$

where

$$
N_{\pm}=\int_{\mathcal D}d\Phi(z)\,\epsilon_D(z)
\int_{t_{\min}}^{t_{\max}}dt\,R_{\pm}(z,t).
$$

The cache supplies the Dalitz integrals of $|A|^2$, $|\bar A|^2$ and
$A^*\bar A$ using `mean(sample.weights * f)`. The time factors are integrated
separately and have units of time. Consequently, the two tags are normalised
separately; this likelihood does not infer the production fraction from the
number of $D^0$ and $\bar D^0$ tags.

For a factorised true-time acceptance $a(t)$ and Gaussian resolution
$G(t_{\rm obs}-t;\sigma_t)$, the observed rate is instead

$$
\widetilde R_{\pm}(z,t_{\rm obs})
=\int_0^\infty dt\,a(t)G(t_{\rm obs}-t;\sigma_t)R_{\pm}(z,t).
$$

The normalisation integrates this expression only over the selected observed
window. In the quadrature implementation this is evaluated as

$$
N_{\pm}=\int_{\mathcal D}d\Phi(z)\,\epsilon_D(z)
\int_0^\infty dt\,a(t)R_{\pm}(z,t)
\Pr(t_{\min}\le t_{\rm obs}\le t_{\max}\mid t,\sigma_t),
$$

where the final probability is the difference of two Gaussian CDF values.
Temporal quadrature therefore uses $\sum_i w_i f(t_i)$, while Dalitz
quadrature retains the package convention based on the weighted mean.

### Mistagging

For an observed tag $q=+1$ or $q=-1$, `wrong_tag` mixes the already normalised
flavour PDFs. For a scalar or event-wise mistag probability $w$, the observed
PDF is

$$
p_{\rm obs}(z,t\mid q=+1)=(1-w)p_+(z,t)+wp_-(z,t),
$$

$$
p_{\rm obs}(z,t\mid q=-1)=(1-w)p_-(z,t)+wp_+(z,t).
$$

Here $w$ is the posterior wrong-flavour probability **in the selected sample**;
it is not automatically a raw detector mistag rate before selection.

For an extended fit, `production_fraction` provides the true production
fraction $f_{\rm prod}=P(D^0)$ when `signal_tag_fraction` is not supplied
explicitly. The session converts it to the observed positive-tag fraction using
the selected-sample mistag probability $w$:

$$
P(q_{\rm obs}=+1)=f_{\rm prod}(1-w)+(1-f_{\rm prod})w.
$$

`signal_tag_fraction` remains available when the observed tag fraction is the
quantity calibrated directly. Supplying both options is rejected because they
are two parameterisations of the same tag probability.

## Preparing A and Abar

Use one `PreparedAmplitudeCache` containing all A components followed by all
Abar components. Both groups must refer to **the same final-state coordinates**
and the same integration measure; give every component a unique name. With
ordering `(KS, pi+, pi-)`, `s12=m²(KS pi+)` and `s13=m²(KS pi-)`. In the
no-direct-CPV convention, `Abar(s12,s13)=A(s13,s12)`. There is no identical-pion
symmetrization or folding: pi+ and pi- are distinct particles.

For a reflected callable, pass only the permuted raw invariants to the original
function, not its precomputed kinematic arrays. Retaining prepared arrays from
the unreflected point would silently evaluate the wrong amplitude. The runnable
example below implements this explicitly. Independently supplied A/Abar models
also support direct CP violation; reference phases and scales must be fixed to
remove unidentifiable directions, especially when fitting `q/p` simultaneously.

```python
from jaxpwa import (
    Parameter, PreparedAmplitudeCache, NeutralMesonMixing,
    TimeDependentDalitzNLL, Minimizer,
)

# components_A / components_Abar: low-level AmplitudeComponent sequences.
# Both groups evaluated at the observed final-state point, not separate datasets.
x = Parameter("mix.x", 0.0056, bounds=(-0.1, 0.1), step=0.001)
y = Parameter("mix.y", 0.0030, bounds=(-0.1, 0.1), step=0.001)
parameters = (*amplitude_parameters, x, y)
cache = PreparedAmplitudeCache.prepare(
    (*components_A, *components_Abar),
    data=data.as_dict(),
    normalization_data=integration_sample.as_dict(),
    normalization_weights=integration_sample.weights,
    efficiency_normalization=efficiency_on_integration_sample,
    parameters=parameters,
    normalize_components=False,
)
cache.check_parameters(parameters)
nll = TimeDependentDalitzNLL(
    cache, len(components_A), times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(0.0, 4.0),
)
result = Minimizer(nll, parameters).fit()
```

`tags=+1` means initially D0; `-1` means initially D0bar. Conditional PDFs are
normalized separately for the two tags. The likelihood does not fit initial
production fractions, tag counts, or an extended yield. `wrong_tag` is the
posterior wrong-flavour probability **within the selected sample** (scalar,
event array, or scalar `Parameter`); it mixes already-normalized PDFs. A raw
preselection mistag rate with production/selection asymmetries requires its own
conversion to this conditional probability.

The new `cache.coherent_groups(values, groups)` method evaluates both amplitudes
and their 2x2 Hermitian overlap in one pass. For the two groups it retains
`integral(conj(A)*Abar)` as well as both diagonal integrals. Every Dalitz integral
uses `mean(sample.weights*f)`; temporal integrals have units of time. Fixed
lineshapes and overlap matrices stay cached when only mixing or coefficients
float; floating dynamics update the affected blocks through existing cache paths.

## Time acceptance and Gaussian resolution

The default uses exact finite-window or `[0,infinity)` time integrals, unit
acceptance, and perfect resolution. For factorized true-time acceptance and/or
a Gaussian resolution, provide quadrature explicitly:

```python
import numpy as np
import jax.numpy as jnp

nodes, weights = np.polynomial.legendre.leggauss(300)
true_time_max = 8.0  # ps; verify convergence for the lifetime/acceptance in use
nodes = true_time_max*(nodes+1)/2
weights = true_time_max*weights/2
nll = TimeDependentDalitzNLL(
    cache, len(components_A), measured_times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(-0.5, 4.0),
    sigma_t=event_time_errors,  # positive, fixed; scalar also accepted
    time_nodes=nodes, time_weights=weights,
    time_acceptance=lambda t, values: 1-jnp.exp(-t/0.15),
)
```

Temporal quadrature follows **sum(weights*f)**, not the Dalitz sample's mean
convention. With resolution, true time is nonnegative while measured time may
be negative. Gaussian CDF differences account for migration into/out of the
observed time window in the per-event normalization. Accumulation scans time
nodes to avoid materializing an events-by-nodes response matrix.

Acceptance acts on true time and must factorize from the Dalitz efficiency.
The PDF is conditional on sigma_t; it assumes the supplied conditional response
and acceptance describe that category. Correlated time/Dalitz acceptance,
non-Gaussian signal resolution mixtures, floating signal-resolution calibration,
and automatic model serialization are not provided by this signal API.
Multiple background categories are composed separately by the session below. Repeat
the quadrature with increased order and true-time endpoint, particularly for
narrow sigma_t or long lifetime tails. A quadrature endpoint truncates the
convolution; it is not an exact infinite-time Gaussian convolution.

## High-level session: TimeDependentFitSession

`TimeDependentFitSession` (`docs/user_friendly_api.md` "Design principle": a
composition layer, not a replacement) removes the A/Abar cache-building
boilerplate above. It composes one `DecayModel` (the D0/A amplitude), a
`PhaseSpaceSample` of observed Dalitz coordinates, parallel `times`/`tags`
arrays, and a `NeutralMesonMixing`:

```python
from jaxpwa import NeutralMesonMixing, TimeDependentFitSession

session = TimeDependentFitSession(
    model, data, times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(0.0, 4.0),
)
result = session.fit()
session.report(result)
```

By default (no `abar_model`) it derives D0bar by reflection,
`Abar(s12,s13) = A(s13,s12)`, the same no-direct-CPV convention as the manual
example above -- it never re-derives or overrides the physics in
`TimeDependentDalitzNLL`/`NeutralMesonMixing`. Pass an independently built
`abar_model` (a second `DecayModel` sharing `model`'s daughter masses) for
direct CP violation; only its dynamics/coefficients are used, since the cache
is built once on `model`'s `data`/`normalization_sample` for both flavours.
`session.parameters` collects and deduplicates `Parameter`s from `model` (and
`abar_model`, if given), `mixing`, mistag, time acceptance and constraints.
Floating dynamics are registered for both cache groups: the reflected group
shares the same public parameter values, with owners remapped internally to
its `bar_` component names. Each model's `normalize_components` default and
each component's override are preserved. In particular, reflection inherits
the original component's normalization convention; it does not disable it.
Component scales use physical phase-space integrals, while efficiency/veto
weights enter the full overlap matrix.

`fit(update_model=True)` also returns
the model(s) with fitted values baked in, via `model_with_fitted_values`.
`session.signal_objective` remains the signal-only `TimeDependentDalitzNLL`.
`session.base_objective` composes it with `TimeDependentMixtureNLL` when
backgrounds or extended yields are requested; constraints wrap that mixture.
Without those options, the original signal-only behavior is unchanged.

`session.plot_time_projection(result)` overlays each observed tag's decay-time
histogram against the model's exact Dalitz-integrated curve:

```python
session.plot_time_projection(result, bins=60, time_unit="ps")
```

The curve comes from `TimeDependentDalitzNLL.dalitz_integrated_time_pdf`, the
Dalitz marginal `p(t | tag) = integral_DP rate(t, s12, s13, tag) dPhi / norm`,
built from the same `cache.coherent_groups` overlap and `mixing.basis`/
`integrals` the fit itself normalizes against -- not a rendering MC sample, so
it shares that method's scope: unit temporal acceptance and perfect
resolution only (raises if `time_nodes`/`sigma_t`/`time_acceptance` are set),
and a scalar `wrong_tag` (a single curve needs one representative mistag
probability, not per-event values). A tag with zero observed events is
skipped.

`session.plot_projection(result, variable)` is the Dalitz-variable analogue,
one subplot per tag (D0 | D0bar), mirroring `CPFitSession.plot_projection`'s
two-population layout:

```python
session.plot_projection(result, "s13", bins=60)
```

A weighted phase-space MC sample renders the histogram; the density comes
from `TimeDependentDalitzNLL.tag_marginal_density`, the *time*-integrated
(over `time_range`) counterpart of `dalitz_integrated_time_pdf` --
`p(s12,s13 | tag) = integral_t rate(t, s12, s13, tag) dt / norm`, built from
the same selected-time basis integrals used by the likelihood. The ideal case
uses analytic integrals; temporal acceptance and Gaussian resolution use the
configured true-time quadrature, including the probability of entering the
observed time window. Quadrature convergence remains the caller's responsibility.
These Dalitz diagnostics require scalar `wrong_tag` and scalar `sigma_t`
(if supplied). Event-wise values are rejected: a single prediction would need
an explicitly specified distribution of these conditioning variables.

Projections evaluate the fitted coefficients, dynamics and component scales,
and multiply by efficiency and veto at the rendering points. Low-level callers
of `tag_marginal_density(a, b, values, efficiency=...)` must likewise supply
amplitudes in the cache's convention and acceptance at the requested points;
omitting acceptance for an acceptance-weighted cache raises an error.
`show_pulls=True` adds a pull panel below each tag's histogram.

`dalitz_density_at_time(a, b, t, values, efficiency=...)` gives the density
conditional on the **observed** time and tag, with the same acceptance and
resolution support. It first mixes the selected-sample joint flavour PDFs,
then divides by the observed-tag time marginal. Because `wrong_tag` is defined
over the selected sample, its posterior value at a particular time generally
changes; independently normalizing each flavour snapshot before mixing would
be incorrect. With zero mistag and perfect resolution, the t=0 snapshot reduces
to the accepted, normalized |A|² or |Abar|². Times outside the selected range or
with zero probability have no conditional density (NaN).

`plot_contour(result, "mix.x", "mix.y")` (not session-specific -- see the
"Plotting" section of `docs/catalog.md`) draws the published-style (x, y)
confidence-region plot: the profile-likelihood 68%/95% contours from
`Minuit.mncontour` around the fitted point, exactly the kind of figure a
mixing-parameter paper publishes (e.g. Fig. 4 of the Belle reference above).
`result` is any fitted `Minuit` object, i.e. what `session.fit()` returns
with the default `update_model=False`.

## Multiple backgrounds and extended yields

`TimeDependentFitSession` follows the `FitSession` / `CPFitSession` conventions:
`backgrounds=`, `.with_background(...)`, `signal_fraction=`, or
`extended=True, signal_yield=...`. Every component is a PDF in **both Dalitz
and observed time**, conditional on the observed tag and `sigma_t`.

A `TimeDependentBackgroundSpec` describes a factorized background.
`shape(data)` is a fixed Dalitz shape, automatically normalized using
`mean(sample.weights * shape)` separately for each tag. It can read `tag` as
well as `s12`, `s13`, `s23`. A supplied `normalization_sample` must have valid
integration weights. Vetoes are applied by default (`apply_veto=False` opts
out). Signal efficiency is **not** applied to a background map: the map
should describe the selected background already.

`time_pdf(data, values)` receives `t`, `tag`, and `sigma_t` when supplied,
and must be normalized over the session's **selected observed-time range**
for every tag and sigma_t. Include the background's own resolution and
acceptance here. The signal's response is not automatically reused. This
explicit normalization contract also permits analytical time PDFs without
building an additional event-by-time integration array. `parameters=...`
registers their floating parameters; parameters declared on callable objects
are also collected.

For the factorized interface, background category $k$ is constructed as

$$
B_k(z,t\mid q,\sigma_t)=b_k(z\mid q)\,h_k(t\mid q,\sigma_t),
$$

with

$$
\int_{\mathcal D}d\Phi\,b_k(z\mid q)=1,
\qquad
\int_{t_{\min}}^{t_{\max}}dt\,h_k(t\mid q,\sigma_t)=1.
$$

The background time PDF includes its own temporal acceptance and resolution;
the signal efficiency is not automatically applied to a background map that
already describes the selected background.

For example, two illustrative backgrounds with exponential observed-time
PDFs (not a detector model for Belle):

```python
import jax.numpy as jnp
from jaxpwa import Parameter, TimeDependentFitSession

low, high = 0.0, 4.0
rate = Parameter("comb.rate", 1.5, bounds=(0.1, 10.0))

def truncated_exponential(t, r):
    norm = jnp.exp(-r * low) - jnp.exp(-r * high)
    return jnp.where((t >= low) & (t <= high), r*jnp.exp(-r*t)/norm, 0.0)

def comb_time(data, values):
    return truncated_exponential(data["t"], rate.resolve(values))

def partial_time(data, values):
    return truncated_exponential(data["t"], 0.5)

session = TimeDependentFitSession(
    model, data, times, tags, mixing, time_range=(low, high),
    signal_fraction=Parameter("f_sig", 0.9, bounds=(0.0, 1.0)),
).with_background(
    "combinatorial", lambda d: jnp.ones_like(d["s12"]),
    time_pdf=comb_time,
    fraction=Parameter("f_comb", 0.7, bounds=(0.0, 1.0)),
    parameters=(rate,),
).with_background(
    "partial", lambda d: d["s12"], time_pdf=partial_time,
)
result = session.fit()
session.plot_projection(result, "s12")
session.plot_time_projection(result, time_unit="ps")
```

As in the other sessions, `f_comb` is a fraction **within the background**:

$$
p(z,t\mid q,\sigma_t)=f_{\rm sig}S(z,t\mid q,\sigma_t)
+(1-f_{\rm sig})\left[f_{\rm comb}B_{\rm comb}(z,t\mid q,\sigma_t)
+(1-f_{\rm comb})B_{\rm partial}(z,t\mid q,\sigma_t)\right].
$$

For $K$ background categories, the first $K-1$ relative fractions are fitted
or fixed and the last is the remainder. In general,

$$
p=f_{\rm sig}S+(1-f_{\rm sig})\sum_{k=1}^{K}w_kB_k,
\qquad \sum_{k=1}^{K}w_k=1.
$$

For N backgrounds, the first N-1 carry relative fractions; the last is the
remainder. A single background needs no relative fraction. These conditional
mixture fractions are shared by the two tags. The NLL includes every event
without fitting the relative number of positive/negative tags in this mode.

In extended mode, replace fractions with `signal_yield` and a `yield_` for
**every** background. Each yield counts the sum of both tags. For an observed
tag $q$, define $\pi_k(q)=P(q\mid k)$ using `signal_tag_fraction` and
`tag_fraction`. The fitted intensity is

$$
\lambda(z,t,q\mid\sigma_t)=N_{\rm sig}\,\pi_{\rm sig}(q)
S(z,t\mid q,\sigma_t)
+\sum_{k=1}^{K}N_k\,\pi_k(q)B_k(z,t\mid q,\sigma_t).
$$

The extended negative log-likelihood, up to parameter-independent terms, is

$$
\mathcal N_{\rm ext}=N_{\rm sig}+\sum_{k=1}^{K}N_k
-\sum_{i=1}^{N_{\rm data}}\log\lambda(z_i,t_i,q_i\mid\sigma_{t,i}).
$$

Each yield counts both observed tags. If a tag fraction is omitted, the
session uses the observed positive-tag fraction as a fixed default; this does
not introduce a component-specific tag asymmetry.

Use `signal_tag_fraction=` for the signal's P(tag=+1), and `tag_fraction=`
for each background. These may be fixed numbers or Parameters in [0,1].
Unspecified tag fractions default to the observed positive-tag fraction,
fixed and shared; this default adds no component-specific tag-asymmetry
information. Set them explicitly to model or fit different tag populations.
Tag-fraction options are only meaningful in extended mode. For example:

```python
extended_session = TimeDependentFitSession(
    model, data, times, tags, mixing, time_range=(low, high),
    extended=True,
    signal_yield=Parameter("N_sig", 900., bounds=(0., None)),
    signal_tag_fraction=0.5,
).with_background(
    "combinatorial", lambda d: jnp.ones_like(d["s12"]), time_pdf=comb_time,
    yield_=Parameter("N_comb", 100., bounds=(0., None)),
    tag_fraction=0.5, parameters=(rate,),
)
```

### Correlated backgrounds and real D with random tags

Use `TimeDependentBackgroundCategory(name, density, ...)` for a general
joint PDF. `density(values)` returns the normalized conditional Dalitz-time
PDF on the fitted events; fixed arrays are accepted as well. Its normalization,
acceptance, veto and response are the caller's responsibility. Declare any
additional fit parameters in `parameters=(...)`. Existing `BackgroundCategory`
objects are also accepted, provided their densities include time and obey
this same conditional normalization (a Dalitz-only density is insufficient).

For a real D with random tag, one can reuse the signal kernel with a different
mistag probability, keeping its mixing dependence:

```python
from dataclasses import replace
from jaxpwa import TimeDependentBackgroundCategory

# Begin with a session without backgrounds; share its cache and response.
random_signal = replace(session.signal_objective, wrong_tag=0.5)
random_tag = TimeDependentBackgroundCategory(
    "random_pion", density=random_signal.densities,
)
# Configure the complete background tuple and its relative fractions as usual.
random_session = replace(session, backgrounds=(random_tag,))
```

This example assumes the random-tag component shares the signal acceptance
and resolution. Its mixing/amplitude parameters are already in the session;
declare an independently floated wrong-tag parameter explicitly if used.
The existing `wrong_tag` convention is a selected-sample posterior, which
must match the convention used to determine the background mistag fraction.

Projections draw the signal, each named background, and their total. Extended
projections preserve the fitted expected count for each tag. Factorized specs
provide their marginals automatically. General/precomputed categories require
`dalitz_density(data, values)` and/or `time_density(data, values)` callbacks for
the requested projection; missing callbacks raise an error. Data contains
`tag`, optional scalar `sigma_t`, and invariants or `t`. These callbacks must
use the same selection and normalization as the joint PDF. Existing signal
projection limits still apply: scalar mistag/resolution for Dalitz projections,
and ideal temporal response for the current time-projection helper.

## Reproducible validation and scope of the Belle reproduction

```
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20 --toy
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20 --backgrounds
pytest tests/test_time_dependent.py
```

The script uses physical D0 -> KS pi+ pi- kinematics and a **reduced** K*(892)-,
K*(892)+, rho model. The Asimov weights and optional sampled toy come from an
independent NumPy implementation of the printed Belle rates. It fits x,y with
JAX/Minuit, reports timing, and checks Asimov recovery at Belle's central mixing
values. With `--backgrounds`, it includes two independently normalized
background PDFs and also floats/reconstructs the signal fraction and the
relative composition of the background. The optional toy samples the finite Dalitz/time quadrature distribution;
it is a discrete approximation, not a new continuous public toy generator.
Neither this three-resonance model nor its fit fractions is Belle's full model.

Unit tests check both tags, overlap orientation, finite/infinite time integrals,
zero mixing, wrong tags, accepted and smeared normalization, independent SciPy
convolution, fixed-cache reuse, floating dynamics (chunked and unchunked), and
JAX gradients including x,y,tau,|q/p|,phase.

The paper provides a much fuller amplitude table, including K-matrix pi-pi and
K-pi S-waves, and describes detector resolution, efficiency and backgrounds.
Reproducing its full model requires a convention audit of those amplitudes and
external inputs cited by the paper. Reproducing the experimental best fit and
errors additionally requires event data and detector/background parameters not
supplied as numerical inputs here. This implementation validates the time
formalism and a controlled example; it does not certify reproduction of Belle's
experimental result.

A full-component Belle transcription (all 15 components, fit-fraction audit and Asimov fit)
was validated earlier; that notebook is no longer in the repository, but its sizeable remaining
FF discrepancies are recorded in the
[validation report](reviews/20260923_time_dependent_belle.md). Do not treat the Belle amplitude
conventions as an already closed reproduction. The time-dependent API is exercised in
[`babar_2008_d0_kspipi.ipynb`](../notebooks/examples/babar_2008_d0_kspipi.ipynb).

## Joint time-dependent toy generation

`generate_time_dependent_toy(session, size, ...)` generates a signal sample
with Dalitz coordinates, true decay times, observed tags and latent true tags:
`toy.data`, `toy.times`, `toy.tags` and `toy.true_tags`. It uses a uniform
phase-space proposal and an exponential lifetime proposal, then resamples using
the session's own time-dependent signal density. The generated Dalitz shape and
time distribution are therefore correlated through mixing.

The public return type is `TimeDependentToy`. The four arrays have the same
length and the tag convention is the one used by the likelihood:

| field | meaning |
|---|---|
| `data` | Dalitz coordinates, returned as a `PhaseSpaceSample`; momenta are omitted by default. |
| `times` | generated true decay times in the session's `time_range`. |
| `true_tags` | latent production flavour: `+1` for D0 and `-1` for D0bar. |
| `tags` | observed tags after the wrong-tag process, also `+1` or `-1`. |

For a production fraction $f=P(D^0)$ and a scalar wrong-tag probability $w$,
the latent and observed tags are drawn according to

$$
P(q_{\rm true}=+1)=f,\qquad
P(q_{\rm obs}=q_{\rm true})=1-w,
$$

so that

$$
\pi_+ = P(q_{\rm obs}=+1)=f(1-w)+(1-f)w,
$$

$$
\pi_- = P(q_{\rm obs}=-1)=fw+(1-f)(1-w).
$$

For each observed tag, the signal kernel is the same conditional density used
by `TimeDependentDalitzNLL.densities`:

$$
p_{\rm obs}(z,t\mid q=+1)=(1-w)p_+(z,t)+wp_-(z,t),
$$

$$
p_{\rm obs}(z,t\mid q=-1)=(1-w)p_-(z,t)+wp_+(z,t),
$$

where $p_+$ and $p_-$ are the separately normalised rates from the mixing
equations above. The intended joint signal target is therefore

$$
p(z,t,q)=\pi_q\,p_{\rm obs}(z,t\mid q).
$$

The implementation draws proposal times from the truncated exponential

$$
g_t(t)=\frac{\tau^{-1}e^{-t/\tau}}
 {e^{-t_{\min}/\tau}-e^{-t_{\max}/\tau}},
 \qquad t_{\min}\le t\le t_{\max},
$$

and draws Dalitz points from the model's phase-space generator. Candidate
weights are formed from the session density and inverse proposal-time density.
The observed-tag probability is already present in the tag proposal, so it
cancels from the importance ratio; applying it again would bias the requested
tag fraction. The candidates are then sampled with replacement. This is
importance resampling: increasing `proposal_size` reduces
the finite-proposal approximation, but does not turn the method into an exact
rejection sampler. Use a fixed `seed` for reproducible candidates and increase
`proposal_size` when validating small mixing or narrow Dalitz structures.

```python
from jaxpwa import generate_time_dependent_toy

toy = generate_time_dependent_toy(
    session,
    50_000,
    parameters=truth,
    production_fraction=0.65,
    wrong_tag=0.08,
    proposal_size=1_000_000,
    seed=7,
)
```

`production_fraction` must be in $[0,1]$. If `wrong_tag=None`, the generator
uses `session.wrong_tag`; otherwise the supplied scalar overrides it. Set
`include_momenta=True` only when downstream code needs four-momenta. The default
compact sample contains only `s12`, `s13` and `s23`, which is enough for the
likelihood and the standard Dalitz plots.

This sampler currently requires perfect time resolution and unit temporal
acceptance. Sessions configured with `time_nodes`, `time_acceptance` or
`sigma_t` raise `NotImplementedError`; sampling those detector effects requires
an additional true-time to observed-time response step. Background generation
is separate and remains the responsibility of `ToyBackground`,
`TimeDependentBackgroundSpec` or `TimeDependentBackgroundCategory`.

Useful diagnostics of a production-and-tag toy are the Dalitz population, time
distributions by observed tag, the true/observed tag migration matrix, and the wrong-tag rate in
time bins. These plots diagnose the generated sample; they are not substitutes
for the likelihood projections, which evaluate the analytic model curves.
