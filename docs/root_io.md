# ROOT input with uproot

Jax-PWA reads ROOT files with `uproot`; PyROOT is not required.

## TTree input

Use `read_root_tree` when arbitrary branches are needed:

```python
from jaxpwa import read_root_tree

arrays = read_root_tree(
    "data.root",
    "DecayTree",
    {"s13": "B_s13", "s23": "B_s23", "mass": "B_M"},
    cut="B_M > 5200 && B_M < 5350",
)
```

The mapping is `{output_name: ROOT_branch_name}`. Returned arrays are JAX arrays
by default. Flat scalar and fixed-size numeric branches are supported.

For a large tree containing many toys, use `library="np"` to keep the input in
host RAM and select events **before** transferring to the JAX device:

```python
import jax.numpy as jnp

raw = read_root_tree(
    "toys.root", "fitTree", ["s12", "s13", "s23", "iExpt"], library="np",
)
mask = raw["iExpt"] == 0
s12 = jnp.asarray(raw["s12"][mask])
```

`jnp.asarray(raw["s12"])[mask]` transfers the entire column first. Converting
the output of the default JAX reader back to NumPy also incurs the initial
device allocation; choose `library="np"` at read time. The host mode supports
the same branch renaming, cuts and entry ranges as the default mode.

For a three-body fit sample use `read_phase_space_sample`:

```python
from jaxpwa import read_phase_space_sample

data = read_phase_space_sample(
    "data.root", "DecayTree",
    s12="S12", s13="S13", s23="S23",
    weight="eventWeight",
)
```

If no weight branch is supplied, all event weights are one. Optional four-momenta use `(E, px, py, pz)` branch tuples for `p1`, `p2` and `p3`. `entry_start`, `entry_stop` and uproot `cut` are supported.

Events stored in Square-Dalitz coordinates are read by naming the `m'` and `theta'` branches
instead of the invariants; the ordered `pair` defining `m'` and the channel masses are required:

```python
data = read_phase_space_sample(
    "data.root", "DecayTree",
    mprime="mPrime", thetaprime="thPrime",
    mother_mass=channel.parent_mass, masses=channel.daughter_masses, pair=(1, 2),
)
```

## ROOT histograms in ordinary Dalitz coordinates

```python
from jaxpwa import histogram_background_from_root, histogram_efficiency_from_root

efficiency = histogram_efficiency_from_root(
    "maps.root", "efficiency_s13_s23",
    x_variable="s13", y_variable="s23",
)
background = histogram_background_from_root(
    "maps.root", "background_s13_s23",
    x_variable="s13", y_variable="s23",
)
```

These return `HistogramEfficiency` and `HistogramBackground`.

## ROOT histograms in Square Dalitz coordinates

For B-decay analyses it is common to store efficiency and background maps directly in Square-Dalitz coordinates `(m', theta')`. This is supported natively:

```python
from jaxpwa import (
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
)

kwargs = dict(
    mother_mass=channel.parent_mass,
    masses=channel.daughter_masses,
    pair=(0, 2),
)

efficiency = square_dalitz_efficiency_from_root(
    "maps.root", "efficiency_sdp", **kwargs
)
background = square_dalitz_background_from_root(
    "maps.root", "background_sdp", **kwargs
)
```

The TH2 x axis is interpreted as `m'` and the y axis as `theta'`. Usually both axes cover `[0,1]`. The ordered `pair` must be exactly the same Square-Dalitz convention used when producing the histogram.

Both constructors accept `interpolation="linear"` to evaluate bilinearly
between bin centres (the `useInterpolation` convention), or
`interpolation="spline"` for smooth tensor-product cubic interpolation. Use
`interpolation="none"` for piecewise-constant lookup. For a
background density stored per Square-Dalitz area, use
`divide_jacobian=True` on `square_dalitz_background_from_root`. The same object
then exposes `generation_value()` for the raw height used when sampling
directly in Square-Dalitz coordinates.

The fitter itself continues to pass ordinary invariants `(s12,s13,s23)`. The histogram model converts every evaluation point internally with `invariants_to_square_dalitz`, then performs the TH2 bin lookup. This means the same object works transparently in:

```text
event PDF evaluation
signal normalization
toy generation
background normalization
CP fits
```

Efficiency histograms are dimensionless and never receive a Jacobian. For a
background histogram that is a density in Square-Dalitz area, pass
`divide_jacobian=True` so its callable value is the density in ordinary
invariant coordinates used by the likelihood. With `SquareDalitzGrid`, the
integration weights already contain the Jacobian. With this option, both toy
methods sample directly in Square-Dalitz coordinates using `generation_value()`
and the raw histogram height. CP charge-split integrals use the callable PDF
(`h/J`) with the ordinary integration weights. Without `divide_jacobian=True`,
the callable histogram is treated as a density per ordinary Dalitz area for
both generation and fitting.

The classes can also be constructed directly without ROOT:

```python
from jaxpwa import (
    SquareDalitzHistogramEfficiency,
    SquareDalitzHistogramBackground,
)

efficiency = SquareDalitzHistogramEfficiency(
    mprime_edges,
    thetaprime_edges,
    values,
    channel.parent_mass,
    channel.daughter_masses,
    pair=(0, 2),
)
```

## Normalization

Efficiency histograms enter the signal normalization as

```text
integral epsilon(Phi) |A(Phi)|^2 dPhi.
```

A background histogram remains an unnormalized shape until its Dalitz integral is computed, for example

```python
bkg_norm = jnp.mean(norm.weights * background(norm.as_dict()))
```

For Square-Dalitz quadrature, `norm.weights` already contain the
transformation Jacobian. A background with `divide_jacobian=True` therefore
has the expected cancellation in the product `weights * background(...)`.

## Examples

- `notebooks/tests/b2kpipi_root_tree_input.ipynb`: TTree -> `PhaseSpaceSample` -> amplitude fit;
- `notebooks/tests/b2kpipi_square_dalitz_eff_background.ipynb`: ROOT TH2 efficiency/background maps in `(m', theta')`, with SDP and ordinary-Dalitz plots and a signal/background fit.
