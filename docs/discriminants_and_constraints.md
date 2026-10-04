# Discriminating variables and external constraints

## Factorized discriminating-variable PDFs

Amplitude fits may include observables beyond the Dalitz coordinates, such as a reconstructed parent mass, BDT response, PID discriminator, or isolation variable.

The basic implementation uses a factorized component model

\[
P_k(\Phi,x_1,\ldots,x_n)
=
P_k^{DP}(\Phi)\prod_a P_{k,a}(x_a),
\]

where `k` labels signal or a background category.  Each one-dimensional factor is normalized on its declared fit range, so the Dalitz and discriminating-variable normalizations remain separable.

Available basic models are:

- `Gaussian1D(mean, sigma, low, high)`;
- `CrystalBall1D(mean, sigma, alpha, n, low, high)` — Gaussian core with a
  power-law tail (`alpha > 0`, `n > 1`, the `scipy.stats.crystalball`
  convention), normalized on `[low, high]` in closed form (`erf` for the
  Gaussian half, the power law's elementary antiderivative for the tail) so
  it stays cheap and JAX-differentiable under a many-event unbinned
  likelihood, unlike wrapping `scipy.stats.crystalball` directly. The
  power-law branch is evaluated only on its own domain (its argument is
  clamped to `z <= -alpha` before the power); evaluating it at core points
  beyond the pole `z = n/alpha - alpha` gave an infinite discarded value and
  NaN gradients for every shape parameter whenever that pole lay inside the
  fit range (small `n`, e.g. `n = 3`, `alpha = 1.5`);
  A negative `alpha` puts the tail on the high side (the `RooCBShape`/LHCb
  convention where the sign of `alpha` selects the side): the density is the
  mirror image `z -> -z` with `|alpha|`. Tail powers are evaluated via
  `log1p` after canceling their common scale, avoiding overflow of
  `(n/|alpha|)^n` (e.g. `|alpha|=0.05`, `n=100` within the notebook bounds);
- `Exponential1D(slope, low, high)`;
- `Chebyshev1D(coefficients, low, high)` — `1 + sum_k c_k T_k(t)` on `t in [-1, 1]`
  (`RooChebychev` convention), normalized in closed form; not positive by
  construction. A coefficient point that makes the polynomial negative
  anywhere on the interval returns NaN everywhere, allowing a
  likelihood to reject it. The check includes endpoints and interior
  stationary points, even when no data lie in the negative region. Valid
  polynomials retain their analytic normalization and JAX derivatives;
- `SumPDF1D(pdfs, fractions)` — recursive-fraction sum (the `RooAddPdf`
  convention) `f1 p1 + (1-f1) f2 p2 + ... + (1-f1)...(1-f_{k-1}) p_k`, e.g. the
  LHCb B -> hhh signal model (LHCb-ANA-2019-052, Eq. 7) Gaussian + left-tail
  Crystal Ball + right-tail Crystal Ball as
  `SumPDF1D((Gaussian1D(...), CrystalBall1D(..., a1, ...), CrystalBall1D(..., -|a2|, ...)), (fG, fCB))`.
  Constant fractions must be finite and in `[0, 1]`; fitted fractions outside
  this domain return NaN everywhere (so an invalid PDF cannot be masked by
  another component of a larger mixture);
- `Histogram1D(edges, values)`;
- `FactorizedDensity(base_density, observables, pdfs)`.

Example:

```python
signal_full = FactorizedDensity(
    base_density=lambda values: signal_dp(data, values),
    observables={"mass": mass, "bdt": bdt},
    pdfs={
        "mass": Gaussian1D(mass_mean, 0.014, 5.20, 5.35),
        "bdt": Histogram1D(edges, signal_bdt_shape),
    },
)
```

The factorization is an analysis assumption.  Correlations between the Dalitz position and a discriminating variable, or between two discriminating variables, require a multidimensional model and are intentionally outside the basic implementation.

`FactorizedDensity` can be used independently for the signal and for every named background category, so combinatorial, partially reconstructed, and misidentified backgrounds can have distinct mass/BDT shapes.

## Gaussian external constraints

`GaussianConstraint` adds

\[
\Delta \mathrm{NLL}
=
\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\]

up to an additive parameter-independent constant.

```python
constraint = GaussianConstraint(signal_fraction, mean=0.70, sigma=0.04)
constrained_nll = ConstrainedNLL(base_nll, constraint)
```

Any number of independent constraints can be supplied to `ConstrainedNLL`.  Typical uses include external signal fractions, background yields, efficiency nuisance parameters, and calibration quantities.

Correlated multivariate Gaussian constraints are not part of the basic implementation yet.

## Optional QMI smoothness constraints

`QMISmoothnessConstraint(qmi, strength=..., weights=...)`, or equivalently
`qmi.smoothness_constraint(...)`, penalizes differences of adjacent complex
node slopes in `s=m**2`. It supports all 1D QMI interpolation modes and both
Cartesian and polar nodes, including JAX gradients and Hessians. Attach it
with `FitSession.with_constraint`, `CPFitSession.with_constraint`, or
`ConstrainedNLL`; it is never added implicitly by the lineshape.
Strength zero disables it exactly. See [the QMI smoothing convention](lineshapes.md#optional-qmi-knot-smoothing)
for the nonuniform-grid weights, fixed-scale requirement, CP usage and
validation of the bias/uncertainty tradeoff.

## Notebooks

- `notebooks/tutorials/tutorial_42_discriminant_pdfs.ipynb` and
  `notebooks/tutorials/tutorial_43_factorized_density_and_constraints.ipynb` cover the
  discriminating-variable PDFs and their factorized combination with the Dalitz density.
- `notebooks/tests/b2kpipi_gaussian_constraints.ipynb` compares constrained and unconstrained fits and plots the corresponding NLL scans.
