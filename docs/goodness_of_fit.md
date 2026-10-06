# Goodness of fit

Jax-PWA provides two independent goodness-of-fit (GOF) methods for an unbinned
amplitude fit, both following M. Williams, "How good are your fits? Unbinned multivariate
goodness-of-fit tests in high energy physics" (arXiv:1006.3019) — the reference that studies
exactly this use case (a Dalitz-plot analysis) and is the method Laura++'s literature uses for
GOF in isobar fits:

- **Binned Pearson chi2** (`FitSession.goodness_of_fit_projection`/`goodness_of_fit_chi2`,
  `CPFitSession` equivalents), in 1D or on the Dalitz/Square-Dalitz plane.
- **Point-to-point dissimilarity (PPD)** (`FitSession.point_to_point_dissimilarity`,
  `CPFitSession.point_to_point_dissimilarity`), an unbinned test with a permutation-test
  p-value.

Notebooks: `notebooks/tutorials/tutorial_09_goodness_of_fit.ipynb` (self-contained walkthrough of
both methods, including a deliberately wrong model to show what a rejected fit looks like),
`notebooks/tutorials/tutorial_46_kdtree_local_residuals.ipynb` and
`notebooks/tutorials/tutorial_47_point_to_point.ipynb`.

The low-level, physics-agnostic building blocks (`chi2_from_histograms`,
`point_to_point_dissimilarity`, `BinnedChi2Result`, `PointToPointResult`) live in
`jaxpwa.goodness_of_fit` and consume plain arrays, not a session — use them directly
for a custom binning or a workflow the session layer doesn't cover, per the project's
[low-level/high-level design principle](user_friendly_api.md).

## Local spatial residuals

`kdtree_local_residuals` uses SciPy's spatial tree on host arrays to define a
reference-neighbour radius for each observed point. It supports `k=1`, including
when the reference sample has only one point; the returned radii retain one
entry per observed point even though SciPy squeezes its neighbour axis in that
case. These local residuals are diagnostics, not independent-bin chi2 terms.

## Binned Pearson chi2

```python
gof = session.goodness_of_fit_projection(result, "s13", bins=40)
print(gof.chi2, gof.n_bins, gof.p_value_min, gof.p_value_max)

gof2d = session.goodness_of_fit_chi2(result, bins=25)
plot_pulls(gof2d)
```

Both reuse the same reweighted-MC-projection histogram trick as `plot_projection`
(`_projection_components`/`_scaled_projection_weights`): the expected counts come from a large
weighted phase-space sample reweighted to the fitted density, histogrammed exactly like the data.
`goodness_of_fit_chi2` supports `square_dalitz=True` (Laura++ `(m', theta')` coordinates, via
`invariants_to_square_dalitz`) and `folded=True` (identical-daughter folding), matching
`plot_square_dalitz`.

### The degrees-of-freedom caveat

Pearson's chi2 (Williams Eq. 3.2) is `sum_c (o_c - e_c)^2 / e_c`. When the model's free
parameters are obtained from a binned fit that minimizes this same chi2, the degrees of freedom
are exactly `n_bins - n_free_parameters - 1`. **That is never the case here**: every
Jax-PWA fit is an *unbinned* maximum-likelihood fit, so the true number of degrees of
freedom is not known. Williams shows only the bound

```text
chi2(dof = n_bins - n_free_parameters - 1) <= chi2 <= chi2(dof = n_bins - 1)
```

holds. `BinnedChi2Result` therefore reports **both** bounding degrees of freedom (`dof_min`,
`dof_max`) and their p-values (`p_value_min`, `p_value_max`) rather than a single, potentially
wrong, pair. Read them as a bracket on the true p-value: if even `p_value_max` is small, the fit
can be safely rejected; if even `p_value_min` is comfortably large, it can be safely retained.
Only when the two are far apart (many free parameters relative to `n_bins`) does the ambiguity
actually matter in practice — use more bins to shrink it, subject to the usual sparse-bin caveat
below.

`n_free_parameters` defaults to the number of floating (non-fixed) `Parameter`s the session
collects (`sum(1 for p in session.parameters if not p.fixed)`); pass it explicitly to override.

Bins with zero expected count are dropped from `n_bins` (and hence from `dof_min`/`dof_max`) and
show as `nan` in `pulls` — dividing by zero cannot be averaged over, and a model that predicts
zero density anywhere real data is observed is a separate, more serious problem than a
chi2/dof number.

## Point-to-point dissimilarity (PPD)

```python
result = session.point_to_point_dissimilarity(result, sigma_bar=0.01, mc_size=2000)
print(result.statistic, result.p_value)
```

This is the unbinned test Williams recommends for exactly the Dalitz-plot case (Sec. 3.3), and
avoids the "curse of dimensionality" binning-choice problem entirely. The test statistic
(Williams Eqs. 3.12-3.13) is

```text
T = (1/n_d^2) * sum_{i<j in data} psi(x_i, x_j)
    - (1/(n_d*n_mc)) * sum_{i in data, j in reference} psi(x_i, x_j)

psi(x_i, x_j) = exp(-|x_i - x_j|^2 / (2 * sigma(x_i) * sigma(x_j)))
sigma(x) = sigma_bar / (f0(x) * phase_space_area)
```

where `f0` is the fitted total density (signal plus backgrounds, fraction-weighted, unit
integral) and the `reference` sample is a large unweighted Monte Carlo replica drawn from `f0`
(built internally via `weighted_resample` on a large phase-space projection sample). Larger `T`
means worse agreement. `sigma_bar` is a **tunable nuisance parameter**, not a physical constant:
Williams notes its optimal value should be picked by examining `f0` (denser regions want a
smaller effective bandwidth); there is no single correct value, so compare across a couple of
choices rather than trusting one blindly, and note that `phase_space_area` only enters `sigma(x)`
as an overall constant that is itself absorbed into `sigma_bar` — any consistent normalization of
`f0` works.

### The permutation-test p-value

`T`'s null distribution is not known analytically, so its p-value is estimated by the
**permutation test** (Williams Appendix C, after Fisher 1935): the pooled data+reference sample
is repeatedly relabeled into a random size-`n_data` "data" subset, `T` is recomputed for each
relabeling, and the p-value is the fraction of relabelings whose statistic is at least as large
as the observed one (`n_permutations`, default 200). Because `sigma(x_i)` depends only on
position (through `f0`), not on the data/reference label, the implementation builds the full
pairwise kernel matrix once and reduces every permutation to one matrix-vector product against
it — mathematically the same permutation test, just reformulated as JAX-friendly linear algebra
instead of recomputing pairwise `exp()` calls per permutation.

Because this is a Monte Carlo estimate, `p_value == 0.0` only means no permutation reached the
observed statistic among `n_permutations` trials — not that the true p-value is exactly zero.
Raise `n_permutations` for a tighter estimate near small p-values.

### Cost: this test is O(n^2)

The pairwise kernel matrix is `n x n` where `n = n_data + n_reference`, so both memory and compute
scale quadratically. `max_total_events` (default 5,000) raises a clear `ValueError` rather than
attempting a huge dense array; `mc_size` defaults to `min(10 * n_data, max_total_events -
n_data)` (Williams' own `n_mc = 10*n_d` recommendation, capped by the guard). If you hit the
guard, lower `mc_size`, subsample `data` before constructing the session, or raise
`max_total_events` deliberately if you have the memory to spare — a few thousand pooled events
already means multiple dense float64 arrays of order 200 MB each.

For a session already built on a large real dataset (hundreds of thousands of candidates,
routine for a real analysis), rebuilding it from scratch just to subsample is unnecessary: since
`FitSession`/`CPFitSession` are frozen dataclasses, `dataclasses.replace(session, data=subsample)`
(or `plus_data=`/`minus_data=` for `CPFitSession`) gives a second session sharing the same model,
backgrounds, efficiency and veto, with only the data swapped — the pattern a large real-data
CP fit needs for its PPD test.

## CP fits

`CPFitSession` mirrors all three entry points per charge:

```python
gof = cp_session.goodness_of_fit_projection(result, "s13", bins=40)  # {"plus": ..., "minus": ...}
plus_only = cp_session.goodness_of_fit_chi2(result, charge="plus", bins=25)
ppd_plus = cp_session.point_to_point_dissimilarity(result, charge="plus")
```

`goodness_of_fit_projection`/`goodness_of_fit_chi2` return a `{"plus": BinnedChi2Result, "minus":
BinnedChi2Result}` dict by default, or a single result when `charge` is given.
`point_to_point_dissimilarity` requires `charge` explicitly: Williams' statistic compares one
data sample against one reference sample from one density, so a "joint" PPD across both charges
at once is not a documented statistic and is not offered here — run it once per charge.

## Plotting pulls

```python
from jaxpwa import plot_pulls

plot_pulls(session.goodness_of_fit_projection(result, "s13", bins=40))
plot_pulls(session.goodness_of_fit_chi2(result, bins=25))
```

`plot_pulls` dispatches on `result.pulls.ndim`: a 1D result draws a bar plot of
`(observed-expected)/sqrt(expected)` against bin center with a shaded +-2 band; a 2D result draws
a diverging `pcolormesh` centered at zero over `result.edges`.

## Pulls directly on `plot_projection`

For the common case of eyeballing one 1D projection together with its pulls, pass
`show_pulls=True` to `plot_projection` itself instead of a separate `goodness_of_fit_projection`
call:

```python
ax, ax_pulls = session.plot_projection(result, "s13", bins=40, show_pulls=True)
axes_grid = cp_session.plot_projection(result, "s13", bins=40, show_pulls=True)  # (2, 2)
```

This adds a `(observed-expected)/sqrt(expected)` panel below the histogram, sharing its x axis
(drawn with the same helper `plot_pulls` uses for the 1D case), using `plot_projection`'s own
histogram binning -- not the independent binning `goodness_of_fit_projection` would choose for an
actual GOF test; it is a quick visual diagnostic, not a substitute for the chi2/PPD numbers
above. Because it needs two rows (one column per charge for `CPFitSession`), it builds its own
figure and therefore requires `ax=None`/`axes=None`; the return value is then `(ax, ax_pulls)` for
`FitSession` or the full 2x2 axes grid (row 0 histograms, row 1 pulls) for `CPFitSession`, instead
of the usual single `ax`/length-2 `axes`.
