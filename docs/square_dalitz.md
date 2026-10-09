# Square Dalitz integration

`SquareDalitzGrid` provides deterministic integration on the square-Dalitz variables commonly used in three-body amplitude analyses.

For a selected two-body pair `(i, j)`, with invariant mass `m_ij`, the coordinates follow the Laura++ convention

```text
m'     = acos(2 (m_ij - m_min)/(m_max - m_min) - 1) / pi
theta' = theta_ij / pi
```

with

```text
m_min = m_i + m_j
m_max = M - m_k
```

and `k` the bachelor index. Both transformed coordinates lie in `[0, 1]`.

The pair is **ordered**. The literal Laura++ convention is `pair=(0, 1)`,
corresponding to `(d1, d2)`, with `theta_12` defined as the angle between `d1`
and the bachelor `d3` in the `d1 d2` rest frame. Reversing the order to
`pair=(1, 0)` leaves `m'` unchanged and reflects the angular coordinate,
`theta' -> 1 - theta'`.

The map is not equal-area in the original Dalitz invariants. `SquareDalitzGrid.sample()` therefore stores the absolute transformation Jacobian in `PhaseSpaceSample.weights`. With the package convention

```text
integral(f) = mean(sample.weights * f)
```

the physical integral is

```text
integral_DP f(s12,s13,s23) ds_a ds_b
=
integral_0^1 integral_0^1 f(s(m',theta')) |J| dm' dtheta'.
```

For the implemented convention,

```text
|J| = 2 pi^2 Delta_m m_ij q p sin(pi m') sin(pi theta')
```

where `q` is the daughter momentum in the `ij` rest frame and `p` is the bachelor momentum in that frame.

## Quadrature

Two deterministic quadratures are available:

```python
quadrature="midpoint"
quadrature="gauss-legendre"
```

The Laura++-compatible **default** is `gauss-legendre` with `resolution=1000`,
meaning 1000 nodes on each axis (one million two-dimensional nodes). This
matches Laura++'s Square-Dalitz prescription for treating narrow structures
along the diagonal pair. The Gauss-Legendre weights are folded into
`PhaseSpaceSample.weights` together with the physical Jacobian and scaled so
that the package-wide convention `mean(weights * f)` remains valid.

`midpoint` remains available explicitly. It uses a regular, uniform grid of cell centers in both transformed variables,

```text
m'_a     = (a + 1/2) / N
theta'_b = (b + 1/2) / N
```

so the sampling itself is uniform on `[0,1] x [0,1]`; all non-uniformity of the physical measure enters through the Jacobian.

Gauss-Legendre often converges faster for smooth functions, but narrow or highly localized amplitudes still require sufficient resolution. The Laura++ default therefore does not replace a convergence study of the normalization matrix.

## What must converge in a fit

A comparison of only the total truth normalization

```text
I = c^dagger M c
```

is not sufficient to validate a quadrature. Different errors in matrix elements can cancel for one particular coefficient vector. For fit closure the relevant numerical object is the full complex normalization matrix

```text
M_ij = integral F_i^* F_j dPhi.
```

A robust convergence study should therefore compare all diagonal and interference elements against a denser reference grid. This is especially important for narrow resonances and for interference between structures oriented along different Dalitz axes.

## B+ -> K+ pi+ pi- convention

For particle ordering

```text
(1, 2, 3) = (K+, pi+, pi-)
```

For a `B+ -> K+ pi+ pi-` model one may use

```python
pair=(0, 2)
```

which corresponds to the `(1,3)` pair in one-based notation and therefore transforms

```text
m_13 = m(K+ pi-).
```

The Square-Dalitz sample can be used consistently for both individual-component
and total-PDF normalization.

## Fitting in Square-Dalitz coordinates

`FitSession(..., coordinates="square-dalitz")` and `CPFitSession(..., coordinates="square-dalitz")`
fit the events as points of the Square-Dalitz plane `(m', theta')` of the model's
`normalization_pair` instead of `(s_a, s_b)`. Every event density is then per `dm' dtheta'`,
normalized over the unit square:

```text
p_square(m', theta') = |A|^2 eps |J| / integral_0^1 integral_0^1 |A|^2 eps |J| dm' dtheta'
                     = p_dalitz(s_a, s_b) |J(m', theta')| .
```

The denominator is exactly what `normalization_method="square-dalitz"` integrates, so the model
must use it (the session raises otherwise). Because `|J|` does not depend on the fit parameters,
the NLL becomes

```text
NLL_square(theta) = NLL_dalitz(theta) - sum_i w_i log |J_i|
```

(`w_i = 1` unless the fit is event-weighted), which is what `SquareDalitzNLL` computes;
`square_dalitz_log_jacobian(model, sample)` returns the per-event `log |J_i|`. Fitted values,
HESSE errors, covariances (including the weighted sandwich/SumW2 corrections) and pulls are
therefore identical to the Dalitz-plot fit; only the NLL value changes. For CP fits the B+ and B-
terms are added and the joint B+/B- normalization is unchanged.

**Efficiencies and backgrounds.** The equality above holds for mixtures only if every component
gains the same `|J|`:

- an efficiency is a dimensionless probability, the same in either coordinates. A
  `SquareDalitzHistogramEfficiency` must keep `divide_jacobian=False` (rejected otherwise);
- a background shape is always interpreted as a density in the ordinary Dalitz measure,
  normalized by `integral b ds_a ds_b`. A Square-Dalitz histogram is naturally a density per
  `dm' dtheta'`, so a `SquareDalitzHistogramBackground` must set `divide_jacobian=True`
  (`b / |J|`, whose Dalitz-plot normalization is `integral b dm' dtheta'`); the session rejects
  `divide_jacobian=False` in Square-Dalitz coordinates. Generic callables cannot be checked:
  they must be Dalitz-plot densities.

**Input in `(m', theta')`.** `PhaseSpaceSample.from_square_dalitz(mprime, thetaprime,
mother_mass=..., masses=..., pair=...)` builds a sample from Square-Dalitz coordinates (amplitudes
are evaluated from the invariants it computes), `sample.square_dalitz(...)` returns them, and
`read_phase_space_sample(..., mprime="mp", thetaprime="thp", mother_mass=..., masses=...,
pair=...)` reads them from ROOT (also through `FitSession.from_root`). Projections in
`(m', theta')` use `plot_projection(..., square_dalitz=True)`.

`tests/test_square_dalitz_fit.py` fits the same toys in both coordinates (single sample and CP,
with Square-Dalitz efficiency and background) and checks equal values and errors and an NLL
difference of `sum_i log |J_i|`.

## Validation

`tests/test_square_dalitz.py` checks:

- Laura++-compatible defaults and ordered-pair convention;
- reflection of `theta'` when the ordered pair is reversed;
- the Jacobian against Laura++'s factorized expression;
- invariant -> square-Dalitz -> invariant round trips;
- the midpoint integral of a constant against the ordinary Dalitz area;
- Gauss-Legendre constant and smooth-moment integrals;
- convergence for a narrow Breit-Wigner-like structure at sufficiently high resolution.
