# Jax-PWA

Unbinned amplitude fits of three-body (and scalar four-body) decays, with JAX
likelihoods and gradients end to end and `iminuit` for the final minimization.

- **New here?** Start with the [overview](_generated/overview.md) and the
  [nine-part notebook course](https://github.com/JuanBSLeite/Jax-PWA/blob/main/notebooks/tutorials/TUTORIALS.md).
- **Looking for a class or function?** The [API catalog](catalog.md) lists every public
  name with one line on what it does; the [API reference](_generated/api/index.md) has
  the full docstrings.
- **Conventions and formulas** are in the user guide pages below.

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
