```{raw} html
<div class="jp-hero">
  <img src="_static/logo.png" alt="Jax-PWA logo">
  <h1>Jax-PWA</h1>
  <p class="jp-tagline">JAX-native unbinned amplitude fits of three-body decays
  (Dalitz-plot analyses), with initial scalar four-body support.</p>
  <div class="jp-badges">
    <a href="https://github.com/JuanBSLeite/Jax-PWA/actions/workflows/tests.yml"><img src="https://github.com/JuanBSLeite/Jax-PWA/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
    <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue" alt="Python">
    <img src="https://img.shields.io/badge/backend-JAX-orange" alt="JAX">
    <a href="https://github.com/JuanBSLeite/Jax-PWA"><img src="https://img.shields.io/badge/source-GitHub-black?logo=github" alt="GitHub"></a>
  </div>
</div>
<div class="jp-stats">
  <div class="jp-stat"><b>100% JAX</b><span>pipeline, gradients, no TensorFlow</span></div>
  <div class="jp-stat"><b>float64</b><span>complex128 by default</span></div>
  <div class="jp-stat"><b>CPU + GPU</b><span>CUDA 12 / 13 extras</span></div>
  <div class="jp-stat"><b>~840</b><span>tests, Laura++-validated</span></div>
</div>
```

**Jax-PWA** fits a coherent sum of resonances to a sample of reconstructed decays such as
`B+ -> K+ pi+ pi-` or `D+ -> pi- pi+ pi+`, to extract resonance parameters, fit fractions,
CP asymmetries and interference. Phase space, kinematics, lineshapes, normalization,
likelihoods and gradients are all JAX arrays and ops; `iminuit` only performs the final
minimization. Conventions follow Laura++, and every deliberate deviation is documented.

## Quick start

::::{tab-set}
:::{tab-item} Install
```bash
git clone https://github.com/JuanBSLeite/Jax-PWA.git
cd Jax-PWA
python -m pip install -e ".[dev]"        # CPU
python -m pip install -e ".[dev,cuda12]"  # or NVIDIA GPU (cuda13 also available)
```
:::
:::{tab-item} Fit
```python
from jaxpwa import FitSession

session = FitSession(model, data)     # model: DecayModel, data: PhaseSpaceSample
result = session.fit(simplex=True)    # JAX value_and_grad + iminuit
session.report(result)                # parameters, fit fractions, GoF
session.plot_projection(result, "s13")
```
:::
:::{tab-item} From ROOT
```python
session = FitSession.from_root(
    model, "data.root", "DecayTree", s12="S12", s13="S13", s23="S23",
)
result = session.fit()
```
:::
:::{tab-item} Toys
```python
from jaxpwa import generate_toy

toy = generate_toy(model, 100_000, parameters=truth, seed=1)
```
:::
::::

## Where do you want to go?

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 First steps
:link: user_friendly_api
:link-type: doc
`FitSession`, `CPFitSession`, toys, ROOT input, efficiency, vetoes and backgrounds in a few lines.
:::

:::{grid-item-card} 🎓 Tutorial course
:link: https://github.com/JuanBSLeite/Jax-PWA/blob/main/notebooks/tutorials/TUTORIALS.md
Nine-part notebook course: phase space, a first fit, normalization, floating dynamics, CP, QMI, GoF.
:::

:::{grid-item-card} 📚 API catalog
:link: catalog
:link-type: doc
One line for every public name in `jaxpwa`, with a pointer to the page that covers it.
:::

:::{grid-item-card} 🧬 Lineshapes
:link: lineshapes
:link-type: doc
Breit-Wigner, Gounaris-Sakurai, Flatté, LASS, K-matrix, QMI, poles, rescattering and more.
:::

:::{grid-item-card} ⚖️ CP and time dependence
:link: cp_coefficients
:link-type: doc
Joint B+/B- normalization, sWeight fits, and `D0`-`D0bar` mixing with `TimeDependentFitSession`.
:::

:::{grid-item-card} ⚡ Performance
:link: performance
:link-type: doc
What is cached, what is recomputed, and how memory is controlled with `normalization_chunk_size`.
:::
::::

## How it works

```{raw} html
<div class="jp-pipeline">
  <div class="jp-step"><b>1. Data</b><small>PhaseSpaceSample from arrays or ROOT</small></div>
  <div class="jp-step"><b>2. Model</b><small>DecayChannel + resonances / non-resonant</small></div>
  <div class="jp-step"><b>3. Dynamics</b><small>JAX kinematics, lineshapes, barriers</small></div>
  <div class="jp-step"><b>4. Normalization</b><small>Gauss-Legendre, Square Dalitz or toy MC</small></div>
  <div class="jp-step"><b>5. Likelihood</b><small>efficiency, SCF, backgrounds, constraints</small></div>
  <div class="jp-step"><b>6. Minimize</b><small>JAX gradient + Nesterov / iminuit</small></div>
</div>
```

Two API layers coexist by design: the convenience sessions (`FitSession`, `CPFitSession`,
`TimeDependentFitSession`) compose the low-level public classes (`SignalPDF`,
`PreparedAmplitudeCache`, `MultiBackgroundNLL`, `CPJointNLL`, `Minimizer`), which stay
available for custom likelihoods and validation work.

## Highlights

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
- **Amplitude models**: coherent resonance sums, 2D amplitudes, identical-particle symmetrization
- **Normalization**: deterministic quadrature with automatic refinement of narrow states
- **Caching**: coefficient-only fits never re-evaluate lineshapes
:::
:::{grid-item}
- **Fit extras**: efficiency, vetoes, SCF, many backgrounds, discriminating variables, constraints
- **Toy generation**: inverse-transform and Laura++-style accept-reject samplers
- **I/O**: ROOT through `uproot`, no PyROOT; model serialization
:::
::::

```{admonition} Reading guide
:class: tip
Start with the **Getting started** pages, use the **API catalog** to find a class, and consult
the **Validation reviews** to see how each convention was reproduced numerically before it
was trusted.
```

```{toctree}
:caption: Getting started
:maxdepth: 1

_generated/overview
user_friendly_api
catalog
```

```{toctree}
:caption: Amplitude models
:maxdepth: 1

dynamics_structure
lineshapes
cp_coefficients
square_dalitz
mc_integration
four_body
```

```{toctree}
:caption: Fitting
:maxdepth: 1

fitting
backgrounds_and_vetoes
scf
discriminants_and_constraints
convolution_resolution
goodness_of_fit
time_dependent
toy_generation
```

```{toctree}
:caption: Input/output
:maxdepth: 1

root_io
model_io
```

```{toctree}
:caption: Performance and environment
:maxdepth: 1

performance
gpu_ubuntu_24_04
```

```{toctree}
:caption: API reference
:maxdepth: 1

_generated/api/index
```

```{toctree}
:caption: Validation reviews
:maxdepth: 1
:glob:

reviews/*
```
