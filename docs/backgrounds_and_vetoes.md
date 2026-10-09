# Multiple background categories and veto maps

Jax-PWA supports arbitrary named background categories and Laura++-style Dalitz vetoes.

## Multiple background categories

For a non-extended fit, the total signal fraction remains the primary mixture parameter:

```text
p(x) = f_sig S(x) + (1 - f_sig) sum_k w_k B_k(x)
```

where every `B_k` is independently normalized and the relative background weights satisfy

```text
sum_k w_k = 1.
```

For `N` background categories, the first `N-1` categories may carry relative `fraction` parameters and the final category is the remainder. This avoids introducing a redundant normalization parameter.

Example:

```python
comb = BackgroundCategory(
    "combinatorial",
    values=comb_data,
    normalization=comb_norm,
    fraction=Parameter("f_comb", 0.6, bounds=(0.0, 1.0)),
)
partial = BackgroundCategory(
    "partially_reconstructed",
    values=partial_data,
    normalization=partial_norm,
)

nll = MultiBackgroundNLL(
    signal_density=lambda values: signal_pdf(values),
    backgrounds=(comb, partial),
    signal_fraction=Parameter("f_sig", 0.8, bounds=(0.0, 1.0)),
)
```

The mixture is then

```text
p = f_sig S + (1-f_sig) [f_comb B_comb + (1-f_comb) B_partial].
```

### Extended mode

In an extended fit every component has its own expected yield:

```text
lambda(x) = N_sig S(x) + sum_k N_k B_k(x)
```

and

```text
NLL = N_sig + sum_k N_k - sum_events log(lambda(x)).
```

Each `BackgroundCategory` then uses `yield_=` rather than `fraction=`.

## Multiple backgrounds in CP fits

`CPJointNLL` accepts `CPBackgroundCategory` objects. Each category has separate positive- and negative-charge shapes and normalization integrals but is normalized in the joint `(Dalitz, charge)` sample space:

```text
B_k(phi,+) = B^raw_{k,+}(phi) / (J_{k,+}+J_{k,-})
B_k(phi,-) = B^raw_{k,-}(phi) / (J_{k,+}+J_{k,-}).
```

The non-extended convention remains

```text
p_q = f_sig S_q + (1-f_sig) sum_k w_k B_{k,q}.
```

In extended mode each CP background category has an independent global yield.

The older single-background arguments of `CPJointNLL` are retained for compatibility with existing notebooks.

## Laura++-style veto maps

A veto is represented by a binary acceptance function

```text
V(phi) = 1  accepted
V(phi) = 0  vetoed.
```

For the signal PDF,

```text
P_sig(phi) = V(phi) epsilon(phi) |A(phi)|^2
             / integral V epsilon |A|^2 dPhi.
```

Thus the same veto must be applied to the event sample and to all normalization integrals.

### Mass-window vetoes

`MassWindowVeto` follows the Laura++ `addMassVeto` convention: bounds are specified in invariant mass in GeV, not mass squared.

```python
charm_veto = MassWindowVeto((0, 2), 1.84, 1.89)
```

This rejects points with

```text
1.84 <= m13 <= 1.89 GeV.
```

Several vetoes can be combined:

```python
veto = CompositeVeto(
    MassWindowVeto((0, 2), 1.84, 1.89),
    MassWindowVeto((1, 2), 3.00, 3.20),
)
```

Arbitrary accepted regions can be represented by `FunctionalVeto`.

### Applying vetoes consistently

For data or generated phase-space samples:

```python
accepted_sample = veto.apply(sample)
```

For a signal PDF:

```python
pdf = SignalPDF(
    intensity=intensity,
    integrator=integrator,
    efficiency=efficiency,
    veto=veto,
)
```

For a background shape, wrap it with the same veto:

```python
vetoed_background = VetoedDensity(background, veto)
```

The background normalization must then be computed from `vetoed_background` on the normalization sample. This ensures signal, every background category, generated toys and fitted data all use exactly the same accepted Dalitz region.

## Importing Square-Dalitz histogram densities

`SquareDalitzHistogramBackground` returns piecewise-constant bin values at
the transformed coordinates by default. Set `interpolation="linear"` for
bilinear interpolation between bin centres, or `interpolation="spline"` for
smooth cubic interpolation. If an external histogram represents a
density `h(mprime, thetaprime)` in Square-Dalitz area, convert it to the
invariant-mass measure used by the signal likelihood:
`b(s13, s23) = h(mprime, thetaprime) / square_dalitz_jacobian(...)`.
Normalizing the unconverted shape cannot repair the position-dependent
distortion. An efficiency is dimensionless and does not receive this factor.
With `divide_jacobian=True`, both toy methods sample this background directly
in Square-Dalitz coordinates; CP charge fractions use the converted PDF and
post-veto integration weights. See [toy generation](toy_generation.md#square-dalitz-histogram-backgrounds).

Match the external interpolation setting as well, and normalize the converted
background after vetoes. In CP fits with a prescribed background counting
asymmetry, normalize each charge separately before applying its charge scale;
otherwise the integrals of the two shapes also change the yield split.
See the [Laura CP comparison audit](reviews/laura_cpvfit_415489.md) for a
reproduced example and analysis-specific adapters.

## Folded Dalitz Plot / Square Dalitz Plot efficiency and background

For a channel with two identical final-state particles, `DecayChannel`
already detects them automatically from PDG IDs
(`DecayChannel("D+", ("pi-", "pi+", "pi+")).final_state_ids`), and `Resonance`
amplitudes are symmetrized under their exchange without any extra
configuration (`docs/dynamics_structure.md`). An unbinned fit therefore
already gives correct results on the full Dalitz Plot / Square Dalitz Plot
with no folding required — folding is a statistics tool, not a correctness
requirement, and is most useful when *building* an efficiency or background
map from a limited MC/data sample: since the physics is symmetric under
exchanging the identical pair, folding effectively doubles the events per bin.

`HistogramEfficiency`/`HistogramBackground` (`efficiency`/`background`
modules, plain Dalitz invariants) and `SquareDalitzHistogramEfficiency`/
`SquareDalitzHistogramBackground` (`square_histograms.py`, `(m', theta')`)
all accept `folded=True`:

```python
from jaxpwa.efficiency import HistogramEfficiency

# s12/s13 are the two invariants exchanged when swapping the identical pair
# (e.g. the two pi+ in D+ -> pi- pi+ pi+); x_edges must equal y_edges.
efficiency = HistogramEfficiency(
    x_edges=edges, y_edges=edges, values=values,
    x_variable="s12", y_variable="s13", folded=True,
)
```

```python
from jaxpwa import SquareDalitzHistogramEfficiency

# pair must be the identical daughters themselves, so m' is already symmetric
# and only theta' -> 1 - theta' needs folding; thetaprime_edges <= 0.5.
efficiency = SquareDalitzHistogramEfficiency(
    mprime_edges=mp_edges, thetaprime_edges=tp_edges, values=values,
    mother_mass=mother_mass, masses=masses, pair=(1, 2), folded=True,
)
```

Both raise at construction if the edges can't represent a folded domain
(`x_edges != y_edges`, or `thetaprime_edges` extending past 0.5) — the same
protection `QMIPixel(folded=True)` already applies to its own `s12_edges`/
`s13_edges`. `square_dalitz_efficiency_from_root`/`square_dalitz_background_from_root`
and their plain-Dalitz equivalents accept `folded=True` too, for loading an
already-folded histogram straight from a ROOT file. `plot_dalitz`/
`plot_square_dalitz` also accept `folded=True`, to visualize data or
projections on the same folded half. These fold at evaluation time using
`fold_thetaprime` (or plain `min`/`max` for the plain-Dalitz case) — a
`folded=True` efficiency/background composes directly with `FitSession` and
`CPFitSession` exactly like any other efficiency/background model.

`FitSession.plot_projection`/`CPFitSession.plot_projection` accept
`folded=True` with `partner_variable=` (the other exchange-symmetric
invariant, e.g. `variable="s12"`, `partner_variable="s13"`) and
`fold_side="low"|"high"` (default `"low"`): this is the same `min`/`max` fold
used everywhere else in this feature, applied event by event to project onto
`s_low = min(variable, partner_variable)` or `s_high = max(...)` instead of
`variable` alone. `s_low` and `s_high` are genuinely different distributions
(not two views of the same one), so call `plot_projection` twice — once per
`fold_side` — to get the usual pair of folded spectra; each call still plots
exactly one histogram, holding the same number of events as the unfolded
case. `CPFitSession.plot_projection` always renders one panel per charge
(`axes`, when given, must be a `[B+ axis, B- axis]` pair, never a single
axis); folding is applied *within* each charge's own panel, B+ and B- are
never mixed with each other.

This feature deliberately does **not** fold the normalization/generation
grid itself (`SquareDalitzGrid`, `DecayModel`'s automatic normalization
selection): halving that grid and doubling the result would only be correct
if the *coherent amplitude* is itself exactly symmetric under the exchange,
and nothing here can safely verify that for an arbitrary user model. The
existing automatic `Resonance` symmetrization (or a `QMIPixel(folded=True)`
component) already makes the full-domain integral correct without needing
to fold it.

## Relation to SCF

Vetoes act on the accepted reconstructed phase space. When SCF is enabled, the SCF migration map should be constructed for the same accepted region, or vetoed reconstructed bins should carry zero accepted probability. The SCF machinery and veto maps are intentionally kept as separate objects so detector migration and analysis selection remain independently testable.

## Validation and zero support

Non-CP and CP mixtures reject non-finite or unphysical starting fractions/yields.
During minimization, invalid parameter points return infinite NLL. Relative
background fractions must form a simplex: their explicit sum cannot exceed one.
Yields must be non-negative.

An event with zero total density has infinite NLL. SignalPDF returns exactly zero
outside signal support; its `floor` constructor argument is retained for
compatibility but no longer replaces physical zeros or clips positive densities.
A background may provide support where the signal is zero. Negative or non-finite
densities are invalid, not probabilities to be floored.

FitSession accepts scalar efficiency/veto values (explicitly expanded) or a vector
with exactly one entry per event. Values must be finite and non-negative. Relative
efficiency values may exceed one. SignalPDF enforces the same shapes; under JAX,
invalid numerical inputs produce invalid densities/infinite NLL rather than a
host-side validation exception.

A CP background may have zero normalization for one charge if all its supplied
values in that charge are zero. Individual integrals must be finite and
non-negative, and their sum must be positive and finite. Ordinary non-CP
backgrounds still require a strictly positive normalization.
