# API catalog

An index of every name exported from `jaxpwa` (i.e. `import jaxpwa as m;
m.__all__`): what it is, one line on what it does, and where to read more. Everything below is
importable directly as `from jaxpwa import <Name>` — no submodule path needed.

This page is a map, not a tutorial. For conventions, formulas and worked examples, follow the
"docs" links to the per-topic file, and the "notebooks" links to a runnable example. For anything
not covered here (private helpers, submodule-only exports like the `Simultaneous`/`Unbinned`/
`WeightedUnbinned` NLL variants in `jaxpwa.likelihood`), read the module directly —
this catalog only covers the public top-level surface.

For a guided introduction, start with the [nine-part notebook course](../notebooks/tutorials/TUTORIALS.md).
The lessons explain the main fitting workflow and link to specialized examples below.

## Model construction

Initial four-body API (scalar external particles): see [four-body support](four_body.md)
and [closure notebook](../notebooks/tutorials/tutorial_66_four_body_closure.ipynb).

| Name | Kind | What it does |
|---|---|---|
| `NBodyDecayChannel` | class | Declares N-body masses and optional identical-particle labels; `from_particles(...)` resolves named external particles. |
| `NBodySample` | class | Covariant event storage, subset invariants, orientation and integration weights; JAX pytree. |
| `NBodyPhaseSpaceMC` | class | Recursive physical weighted N-body phase-space generation. |
| `FourBodyDecayModel` | class | Four-body coherent model using the existing normalization/cache/fit core; `with_fixed_parameters(...)` returns an updated immutable model. |
| `Isobar` | class | Spin, mass, width, radius and reusable lineshape; `from_particle(name)` fills nominal properties from `particle`. |
| `PairChain` | class | One LS wave for `P -> (ab)(cd)`. |
| `CascadeChain` | class | One LS wave for `P -> R a, R -> S b, S -> c d`. |
| `pair_coordinates` | function | Five invariant mass/helicity coordinates for `(ab)(cd)`. |
| `cascade_coordinates` | function | Five invariant mass/helicity coordinates for `a(b(cd))`. |
| `pair_coordinates_to_momenta` | function | Canonical four-vectors reconstructed from five pair-chain coordinates. |

| Name | Kind | What it does |
|---|---|---|
| `DecayChannel` | class | Parent particle and ordered three-body final state; `from_particles(...)` mirrors the N-body named constructor. |
| `DecayModel` | class | Build a coherent amplitude model with deterministic Dalitz-plane normalization; `with_fixed_parameters(...)` fixes selected parameters in a new model. |
| `Resonance` | class | Declarative one-dimensional resonance component; `from_particle(...)` uses the same nominal-property resolver as `Isobar`, with interchangeable `lineshape`/`angular` plugins and form-factor conventions. |
| `NonResonant` | class | Constant (S-wave, isotropic) non-resonant component with a complex coefficient. |
| `DalitzAmplitude` | class | Attach a genuinely two-dimensional amplitude (e.g. `QMI2D`) that depends on both Dalitz invariants at once, bypassing the isobar construction. |
| `AmplitudeComponent` | class | Named dynamical component `F_i(x)` with a coefficient; the base type `Resonance`/`NonResonant`/`DalitzAmplitude` all produce. |
| `CoherentAmplitudeModel` | class | Coherent sum `A(x) = sum_i c_i F_i(x)` of prepared components. |
| `ConstantAmplitude` | class | Non-resonant constant dynamical amplitude (the piece `NonResonant` wraps). |
| `PreparedAmplitudeCache` | class | Pre-evaluated component values and Hermitian normalization matrix for a data/normalization sample; the object that makes repeated NLL evaluation cheap. |

Docs: `docs/fitting.md`, `docs/dynamics_structure.md`, `docs/performance.md` (caching). Notebooks: `notebooks/tutorials/tutorial_02_first_fit.ipynb`, `notebooks/examples/b2kpipi_toy_fit.ipynb`.

## Complex coefficients

| Name | Kind | What it does |
|---|---|---|
| `RealImag` | class | Complex coefficient `c = x + i y` for one amplitude component. |
| `CPRealImag` | class | Cartesian CP coefficient `c_q = (x + q*dx) + i(y + q*dy)` shared between the B+/B- charge models. |

Docs: `docs/fitting.md` ("RealImag coefficients"), `docs/cp_coefficients.md`.

## One-dimensional resonance dynamics (lineshapes)

Each is passed as `Resonance(..., lineshape=...)`; all implement `lineshape(mass, context)`.

| Name | Kind | What it does |
|---|---|---|
| `SympyLineshape` | class | Optional SymPy scalar expression with explicit parameter/context bindings, compiled to JAX; see `docs/lineshapes.md` and `tutorial_11_sympy_lineshapes.ipynb`. |
| `RelativisticBreitWigner` | class | Standard relativistic Breit-Wigner with running width and Blatt-Weisskopf barrier factor. |
| `Pole` | class | Simple fixed-width Breit-Wigner pole, `1/(m - m0 - i*Gamma0/2)` (Laura++ `BW`). |
| `SigmaPole` | class | LHCb 3pi-isobar `f0(500)` pole, `1/((pole_mass - i*pole_width)**2 - m**2)`; distinct from `Pole`'s convention. |
| `GounarisSakurai` | class | Gounaris-Sakurai lineshape for rho-like vector states. |
| `RhoOmegaMixing` | class | Coherent rho-omega mixing (LHCb 3pi-isobar Eq. 15); `component="rho"`/`"omega"` selects one of the two effective split terms. |
| `PipiKKRescattering` | class | LHCb 3pi-isobar phenomenological pi-pi to K-Kbar S-wave (Eqs. 17-21), zero outside `1.0-1.5 GeV`; `convention="paper"` (default) or `"laura"`. |
| `Flatte` | class | Coupled two-channel Flatte lineshape (generic; construct channel masses directly). |
| `BaBarFlatte` | class | Flatte form as parameterized in the BaBar `B± -> K± pi∓ pi±` analysis (arXiv:0803.4451). |
| `LASS` | class | Effective-range + `K0*(1430)` coherent S-wave form for `K pi`. |
| `KMatrix` | class | Five-pole, five-channel Anisovich-Sarantsev pi-pi S-wave K-matrix; exposes `scattering_amplitude()`/`s_matrix()` for unitarity checks. |
| `QMI` | class | Quasi-model-independent S-wave specified at fixed mass knots; `interpolation=` selects constant bins (`none`) or `linear`/`cubic`/`hermite`/`natural`, polar or Cartesian knot parameters. |
| `Rescattering2` | class | Port of Laura++ `LauRescattering2Res`: two-region Chebyshev pi-pi/KK rescattering S-wave, zero below the `2*m_K` threshold. |

Docs: `docs/lineshapes.md` (formulas + references). Notebooks: `notebooks/tutorials/tutorial_25_rescattering2.ipynb`
(Rescattering2 diagnostics) and
`notebooks/validation/paper_isobar_benchmark_squaredp_01_laura_toys.ipynb` (`SigmaPole`/`RhoOmegaMixing`/
`PipiKKRescattering` reproduction of the LHCb `B -> 3pi` isobar model, Phys. Rev. D 101, 012006).
`docs/reviews/paper_isobar_conventions.md` documents the numeric reproduction, the confirmed
angular-orientation/ACP fixes, and the remaining unresolved discrepancies in that benchmark.

## Angular models

Passed as `Resonance(..., angular=...)`; default is `CovariantAngular()`.

| Name | Kind | What it does |
|---|---|---|
| `CovariantAngular` | class | Default covariant angular factor (spin `L=0..4`). |
| `ZemachP` / `Zemach_P` | class | Laura++ `Zemach_P`: bachelor momentum `p` evaluated in the resonance rest frame. Both names are the same class. |
| `ZemachPstar` / `Zemach_Pstar` | class | Laura++ `Zemach_Pstar`: bachelor momentum `p*` evaluated in the parent rest frame. Both names are the same class. |
| `GooFitLegacyAngular` | class | Legacy GooFit `Ds -> pi pi pi` angular convention (`L=0..2` only), for reproducing historical fits. |

Docs: `docs/dynamics_structure.md`.

## Resonance assembly internals

| Name | Kind | What it does |
|---|---|---|
| `ResonanceAmplitude` | class | Complete one-dimensional resonance amplitude assembled from a lineshape + angular + barrier plugin; what `Resonance` builds under the hood. |
| `ResonanceContext` | class | Kinematic/particle-property bundle (`parent_mass`, `daughter_masses`, `bachelor_mass`, `spin`, `pole_mass`, `pole_width`, radii) passed to lineshape/angular callables. |

Docs: `docs/dynamics_structure.md`, `docs/lineshapes.md`.

## Two-dimensional Dalitz amplitudes

| Name | Kind | What it does |
|---|---|---|
| `QMI2D` | class | Complex amplitude field defined bin-by-bin over `(s12, s13)`; three interpolation modes (`none`/`linear`/`cubic`), optional identical-particle folding. |
| `physical_bin_mask` | function | Mark which `QMI2D` grid cells intersect the physical Dalitz boundary, using the exact analytic boundary. |
| `PolarFormFactorSymNR` | class | Laura++ `LauPolarFormFactorSymNR` nonresonant amplitude for symmetric Dalitz plots, `1/(1+s/lambda^2) + 1/(1+t/lambda^2)` (or the `min(s,t)` no-interference form). |

Docs: `docs/lineshapes.md` ("QMI2D Dalitz amplitude", "Polar form-factor nonresonant amplitude").

## Kinematics and phase space

| Name | Kind | What it does |
|---|---|---|
| `PhaseSpaceSample` | class | Core data container: invariants, weights, optional four-momenta. Every integral in the package is `mean(sample.weights * f)`. |
| `PhaseSpaceMC` | class | Generate weighted three-body phase-space events in the physical weight convention `ds12 ds13 / (128*pi^3*M^2)`. |
| `CovariantKinematics` | class | Bundle of `resonance_mass, p_star, p, q, cos_theta` used by covariant/Zemach angular factors. |
| `covariant_kinematics` | function | Compute covariant-spin kinematics from daughter four-vectors. |
| `covariant_kinematics_from_invariants` | function | Compute the same kinematics directly from Dalitz invariants (no four-vectors needed). |
| `boost_to_rest_frame` | function | Lorentz-boost a four-momentum into the rest frame of another. |
| `dalitz_s13_limits` | function | Exact physical `s13` bounds at fixed `s12` (the Dalitz boundary). |

Docs: `docs/mc_integration.md`, `docs/toy_generation.md`.

## Square-Dalitz coordinates and quadrature

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzGrid` | class | Laura++-convention `(m', theta')` deterministic integration grid with Jacobian weights folded into `PhaseSpaceSample.weights`. |
| `DalitzGaussLegendreGrid` | class | Tensor-product Gauss-Legendre grid directly in `(m13, m23)` (the default `normalization_method="gauss-legendre"` engine). |
| `invariants_to_square_dalitz` | function | Convert `(s12,s13,s23)` to Laura++ `(m', theta')`. |
| `square_dalitz_to_invariants` | function | Inverse of the above. |
| `square_dalitz_jacobian` | function | Absolute Jacobian `|d(s_ij,s_ik)/d(m',theta')|` for the SDP map. |
| `fold_thetaprime` | function | Map `theta'` onto `[0, 0.5]` (identical-particle exchange fold); used by `SquareDalitzHistogramEfficiency`/`Background(folded=True)` and `plot_square_dalitz(folded=True)`, see `docs/backgrounds_and_vetoes.md`. |

Docs: `docs/square_dalitz.md`.

## Fit parameters and minimization

| Name | Kind | What it does |
|---|---|---|
| `Parameter` | class | Configuration for one scalar fit parameter (value, bounds, `owner`, `ParameterKind`); use `Parameter.coefficient(...)`/`Parameter.dynamics(...)` constructors. |
| `ParameterKind` | class | Enum-like role tag (`COEFFICIENT` vs `DYNAMICS`) that determines what a floating parameter invalidates in the cache. |
| `Minimizer` | class | Wraps `iminuit` around a JAX `value_and_grad` objective; `fit()`, `fit_multistart()`, `check_gradient()`. |
| `MultiStartResult` | class | Collection of independent minimizations from `fit_multistart`, plus the best valid minimum. |
| `NesterovResult` | class | Minuit-compatible result object returned by `Minimizer.fit(method="nesterov")`. |

Docs: `docs/fitting.md`.

## Low-level PDFs and likelihoods

These are what `FitSession`/`CPFitSession` compose automatically; use them directly for custom or CP workflows the convenience layer doesn't cover.

| Name | Kind | What it does |
|---|---|---|
| `SignalPDF` | class | Efficiency-corrected, normalized signal density built from a `PreparedAmplitudeCache`. |
| `SCFSignalPDF` | class | Signal PDF including correctly-reconstructed *and* self-cross-feed (SCF) migrated events. |
| `MultiBackgroundNLL` | class | Unbinned NLL: signal plus an arbitrary number of named background categories (non-CP). |
| `WeightedUnbinnedNLL` | class | Weighted unbinned NLL `-sum_i w_i log p(x_i)` for an externally-weighted `logpdf` (e.g. sWeights); does not implement the covariance correction needed for statistically valid HESSE errors on such fits. |
| `NeutralMesonMixing` | class | Exact neutral-meson time kernel with x, y, lifetime and complex q/p; see [time-dependent fits](time_dependent.md). |
| `TimeDependentMixtureNLL` | class | Multiple normalized Dalitz-time backgrounds with floating fractions or extended yields and component tag probabilities; see `docs/time_dependent.md`. |
| `TimeDependentDalitzNLL` | class | Tagged time-dependent Dalitz signal NLL, coherent A/Abar overlap, factorized acceptance and optional Gaussian time resolution; see [time-dependent fits](time_dependent.md). |
| `CPJointNLL` | class | Unbinned NLL for simultaneous B+/B- fits with one joint `(Dalitz, charge)` normalization — charge is part of the sample space, not fit independently per charge. Optional `plus_weights`/`minus_weights` give the signal-only sWeight/COW objective. |
| `YieldAsymmetry` | class | Extended-fit `signal_yield` replacement splitting a total `N_s` into independent `N_plus`/`N_minus` via a yield asymmetry, overriding `CPJointNLL`'s default amplitude-driven charge split. |

Docs: `docs/fitting.md`, `docs/cp_coefficients.md`, `docs/backgrounds_and_vetoes.md`, `docs/scf.md`, `docs/performance.md`, `docs/time_dependent.md`.

## Backgrounds

| Name | Kind | What it does |
|---|---|---|
| `BackgroundCategory` | class | One normalized background category (density + normalization integral) for `MultiBackgroundNLL`. |
| `TimeDependentBackgroundSpec` | class | Fixed Dalitz shape normalized per tag plus a caller-normalized, optionally parameterized observed-time PDF for `TimeDependentFitSession`. |
| `TimeDependentBackgroundCategory` | class | General normalized joint Dalitz-time background callback or array; optional projection marginal callbacks. |
| `BackgroundSpec` | class | Background *shape* for `FitSession`, normalized automatically on the fit's own measure — no manual integral needed. |
| `CPBackgroundCategory` | class | One background category in the joint `(Dalitz, charge)` space, for `CPJointNLL`. |
| `CPBackgroundSpec` | class | Charge-aware background shape for `CPFitSession` (`plus_shape`/`minus_shape`, or one shared shape). |
| `ToyBackground` | class | One background component for high-level (`generate_toy`) non-CP toy generation. |
| `CPToyBackground` | class | Charge-aware background component for `generate_cp_toy`. |

Docs: `docs/backgrounds_and_vetoes.md`, `docs/user_friendly_api.md`, `docs/toy_generation.md`.

## Vetoes

| Name | Kind | What it does |
|---|---|---|
| `VetoMap` | class | Base type for a binary Dalitz-acceptance mask; `.apply(sample)` selects data, `.apply(sample, for_integration=True)` selects and rescales an integration sample. |
| `MassWindowVeto` | class | Reject one invariant-mass window, Laura++ `addMassVeto` convention (bounds in GeV, not GeV²). |
| `CompositeVeto` | class | Logical AND of any number of veto maps. |
| `FunctionalVeto` | class | Wrap an arbitrary callable returning accept/reject per event as a veto map. |
| `VetoedDensity` | class | Apply a veto map to any density/shape callable (e.g. a background shape). |
| `vetoed_signal_pdf` | function | Build a veto-aware `SignalPDF` directly from a `DecayModel`. |

Docs: `docs/backgrounds_and_vetoes.md`.

## Self-cross-feed (SCF)

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzSCFMap` | class | Uniform Square-Dalitz SCF fraction + true->reconstructed migration map (dense or sparse). |
| `SparseMigration` | class | COO representation of `P(reco_bin \| true_bin)` for large migration maps; `O(nnz)` memory instead of `O(n_bins^2)`. |

Docs: `docs/scf.md`.

## Efficiency/background maps from ROOT histograms

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzHistogramEfficiency` | class | Piecewise-constant efficiency map in `(m', theta')`, constructed directly from arrays (no ROOT needed). |
| `SquareDalitzHistogramBackground` | class | Piecewise-constant background shape in `(m', theta')`, constructed directly from arrays. |
| `square_dalitz_efficiency_from_root` | function | Build a `SquareDalitzHistogramEfficiency` from a ROOT TH2 with `(m', theta')` axes. |
| `square_dalitz_background_from_root` | function | Build a `SquareDalitzHistogramBackground` from a ROOT TH2 with `(m', theta')` axes. |
| `histogram_efficiency_from_root` | function | Build an efficiency map from a ROOT TH2 in ordinary `(s_ij, s_ik)`-style Dalitz coordinates. |
| `histogram_background_from_root` | function | Build a background map from a ROOT TH2 in ordinary Dalitz coordinates. |
| `read_root_histogram2d` | function | Low-level: read a ROOT TH2 into `(values, x_edges, y_edges)` JAX arrays via uproot. |

Docs: `docs/root_io.md`. Notebooks: `notebooks/tests/b2kpipi_square_dalitz_eff_background.ipynb`, `notebooks/tutorials/tutorial_36_histogram_maps_from_root.ipynb`.

## ROOT tree I/O

| Name | Kind | What it does |
|---|---|---|
| `read_root_tree` | function | Read arbitrary named ROOT TTree branches into JAX arrays (or host NumPy arrays with `library="np"`), with an optional `cut`; see `root_io.md`. |
| `read_phase_space_sample` | function | Read a ROOT TTree directly into a `PhaseSpaceSample` (`s12/s13/s23`, optional `weight` and four-momenta). |
| `write_phase_space_sample` | function | Write one `PhaseSpaceSample` to a ROOT TTree with uproot. |
| `write_phase_space_samples` | function | Write several `PhaseSpaceSample` objects to ROOT TTrees in one call. |
| `write_cp_phase_space_sample` | function | Write B+ and B- samples to one TTree with a signed `charge` branch. |

Docs: `docs/root_io.md`. Notebooks: `notebooks/tests/b2kpipi_root_tree_input.ipynb`, `notebooks/tutorials/tutorial_07_root_io.ipynb`, `notebooks/tutorials/tutorial_37_root_low_level_io.ipynb`.

## Model import/export

| Name | Kind | What it does |
|---|---|---|
| `export_model` | function | Write a `DecayModel`'s (or `FitSession`'s `.model`) channel, components (lineshape/angular/coefficient plugins, embedded `Parameter`s) and normalization settings to a JSON file. |
| `import_model` | function | Reconstruct a `DecayModel` from a file written by `export_model`. |
| `model_to_spec` | function | In-memory `dict` form of `export_model`, for embedding a model definition without file I/O. |
| `model_from_spec` | function | In-memory counterpart of `import_model`; accepts an optional `parameter_registry` to preserve shared `Parameter` identity. |
| `export_cp_models` | function | Like `export_model`, for a B+/B- model pair or a `CPFitSession` (its `.plus_model`/`.minus_model`), sharing one parameter registry. |
| `import_cp_models` | function | Reconstruct a `(plus_model, minus_model)` pair from a file written by `export_cp_models`. |
| `cp_models_to_spec` | function | In-memory `dict` form of `export_cp_models`. |
| `cp_models_from_spec` | function | In-memory counterpart of `import_cp_models`. |
| `model_with_fitted_values` | function | Return a new `DecayModel` (or `FitSession`'s `.model`) with every `Parameter.value` taken from a `{name: float}` mapping (e.g. `session.result_values(result)`) -- fitting never mutates the original model's frozen `Parameter`s. |
| `cp_models_with_fitted_values` | function | Like `model_with_fitted_values`, for a B+/B- model pair, keeping a coefficient shared between charges shared in the result. |

`FitSession.fit(update_model=True)`/`CPFitSession.fit(update_model=True)` call these
automatically, returning `(result, updated_model)`/`(result, plus_model, minus_model)` instead of
plain `result` (default `False`, existing callers are unaffected).

Only the built-in jaxpwa component/plugin classes (`Resonance`/`NonResonant`/
`DalitzAmplitude`, every lineshape and angular model, `RealImag`/`CPRealImag`, `QMI2D`) round-trip;
a custom plugin needs its own `to_spec`/`from_spec` pair, as `SympyLineshape` already provides.
`normalization_method="toy-mc"` is not supported, since its external `normalization_sample` is not
part of the specification. None of these capture a session's `data`/`efficiency`/`veto`/
`backgrounds`/`constraints` -- only the amplitude model(s). Docs: `docs/model_io.md`.

## Toy (pseudo-data) generation

| Name | Kind | What it does |
|---|---|---|
| `generate_toy` | function | Generate signal/background pseudo-data; `method="inverse-transform"` (default) or `"accept-reject"`. |
| `generate_signal_toy` | function | Generate an unweighted signal-only toy. |
| `generate_cp_toy` | function | Generate a CP toy for both charges at once, with the accepted-integral charge split and optional single-ROOT-file output. |
| `generate_time_dependent_toy` | function | Generate a tagged neutral-meson Dalitz/time toy with latent true tags, observed wrong-tagged tags, and importance resampling; see [time-dependent toy generation](time_dependent.md#joint-time-dependent-toy-generation). |
| `TimeDependentToy` | class | Immutable container for the generated `data`, `times`, `tags` and `true_tags` arrays. |
| `prepare_inverse_toy_generator` | function | Precompute the inverse-CDF tables once for repeated toys at fixed model parameters. |
| `PreparedInverseToyGenerator` | class | The reusable object `prepare_inverse_toy_generator` returns; `.generate(n, seed=...)`. |
| `weighted_resample` | function | Draw unweighted events from a weighted phase-space sample (the resampling building block behind `method="resample"`). |

Docs: `docs/toy_generation.md`. Notebooks: `notebooks/tests/user_friendly_toy_generation.ipynb`, `notebooks/tutorials/tutorial_38_toy_methods_comparison.ipynb`, `notebooks/tutorials/tutorial_39_prepared_inverse_toy_generator.ipynb`.

## Discriminating-variable PDFs and external constraints

| Name | Kind | What it does |
|---|---|---|
| `Gaussian1D` | class | Gaussian PDF normalized on a finite interval. |
| `Exponential1D` | class | Exponential PDF `exp(slope*x)` normalized on a finite interval. |
| `Chebyshev1D` | class | Chebyshev-polynomial background PDF (`RooChebychev` convention) with closed-form normalization on a finite interval. See [discriminants_and_constraints.md](discriminants_and_constraints.md). |
| `CrystalBall1D` | class | Crystal Ball PDF (Gaussian core, left power-law tail, or right tail for negative `alpha`; `scipy.stats.crystalball` convention) with closed-form normalization on a finite interval. |
| `SumPDF1D` | class | Recursive-fraction sum of normalized 1D PDFs (`RooAddPdf` convention), e.g. Gaussian + two-sided Crystal Ball mass models. See [discriminants_and_constraints.md](discriminants_and_constraints.md). |
| `Histogram1D` | class | Piecewise-constant normalized histogram PDF from edges + values. |
| `BreitWigner1D` | class | Constant-width Breit-Wigner PDF normalized on a finite mass interval. |
| `LineshapeIntensity1D` | class | Turn an existing complex dynamics lineshape (e.g. `RelativisticBreitWigner`) into a normalized 1D intensity PDF. |
| `FactorizedDensity` | class | Multiply a base Dalitz density by independent 1D discriminant PDFs (mass, BDT, PID, ...). |
| `GaussianConstraint` | class | Gaussian penalty `0.5*((x-mu)/sigma)^2` on one parameter. |
| `QMISmoothnessConstraint` | class | Optional complex-node curvature penalty in mass squared for any 1D QMI interpolation; also built by `qmi.smoothness_constraint(...)`. |
| `ConstrainedNLL` | class | Add callable penalties, including Gaussian or QMI smoothness constraints, to an existing NLL. |

Docs: `docs/discriminants_and_constraints.md`. Notebooks: `notebooks/tutorials/tutorial_42_discriminant_pdfs.ipynb`, `notebooks/tutorials/tutorial_43_factorized_density_and_constraints.ipynb`, `notebooks/tests/b2kpipi_gaussian_constraints.ipynb`.

## Detector-resolution convolution (1D)

| Name | Kind | What it does |
|---|---|---|
| `ConvolvedPDF1D` | class | Numerically convolve a normalized 1D PDF with a resolution kernel, with finite-observed-window normalization. |
| `GaussianResolution1D` | class | Gaussian conditional resolution kernel `R(x_obs \| x_true)`. |

Docs: `docs/convolution_resolution.md`. Notebooks: `notebooks/tutorials/tutorial_44_convolution.ipynb`, `notebooks/tests/pdf_convolution_resolution.ipynb`.

## Delta-method error propagation

| Name | Kind | What it does |
|---|---|---|
| `delta_method_jacobian` | function | Exact reverse-mode-autodiff Jacobian of a JAX-differentiable observable with respect to named fit parameters, at a given parameter point (one `jax.vjp` linearization plus a per-output-row loop, not `jax.jacrev`'s batched sweep -- see `docs/fitting.md` for why). |
| `delta_method_covariance` | function | Propagate a postfit covariance (e.g. `result.covariance` from `Minimizer.fit`) through that Jacobian: `J @ C @ J.T`. |
| `delta_method_errors` | function | Standard errors only: `sqrt(diag(delta_method_covariance(...)))`. |

General-purpose linear (Gaussian) error propagation for any JAX-differentiable
quantity derived from postfit parameter values -- not just fit fractions,
though `DecayModel.fit_fraction_errors`/`FitSession.fit_fraction_errors`/
`CPFitSession.fit_fraction_errors` are the built-in convenience wrappers for
that specific case (`CPFitSession`'s propagates the *joint* B+/B- covariance
in one Jacobian, since the two charges share almost every fit parameter, and
returns the correct cross-term-aware error for the mean fraction too). This
is the same linear approximation implicit in Minuit's own HESSE errors, and
can accept either a Minuit-style name-indexable covariance or a plain dense
array already ordered like the requested parameter names. Docs:
`docs/fitting.md` ("Fit fractions"), `docs/cp_coefficients.md`.

## Goodness of fit

| Name | Kind | What it does |
|---|---|---|
| `BinnedChi2Result` | class | Binned Pearson chi2 result: chi2, dof bounds, p-value bounds, per-bin pulls and edges. |
| `chi2_from_histograms` | function | Low-level binned Pearson chi2 test between observed/expected count arrays (1D or 2D); `observed_variance=` (e.g. per-bin sum of squared sWeights/COW weights) replaces the Poisson variance. |
| `PointToPointResult` | class | Point-to-point dissimilarity (PPD) result: statistic, permutation-test p-value. |
| `point_to_point_dissimilarity` | function | Low-level unbinned PPD test (Williams, arXiv:1006.3019) between plain coordinate/density arrays. |
| `kdtree_local_residuals` | function | Adaptive k-neighbour local-residual diagnostic: per-point Pearson pulls from a KD-tree-selected neighbourhood, in any coordinate pair. |

`FitSession`/`CPFitSession` expose these as `goodness_of_fit_projection`/`goodness_of_fit_chi2`/
`point_to_point_dissimilarity` methods, reusing the same reweighted-MC-projection machinery as
`plot_projection`. Docs: `docs/goodness_of_fit.md`. Notebooks:
`notebooks/tutorials/tutorial_09_goodness_of_fit.ipynb`,
`notebooks/tutorials/tutorial_46_kdtree_local_residuals.ipynb`,
`notebooks/tutorials/tutorial_47_point_to_point.ipynb`.

## Plotting

| Name | Kind | What it does |
|---|---|---|
| `plot_dalitz` | function | Plot a 2D Dalitz histogram in one call. |
| `plot_square_dalitz` | function | Plot a 2D Square-Dalitz histogram from ordinary invariant coordinates. |
| `plot_binned_data` | function | Plot 1D data as black points with statistical error bars. |
| `binned_data` | function | Return bin centers, counts, uncertainties and edges without plotting (for custom figures). |
| `plot_pulls` | function | Plot per-bin pulls from a `BinnedChi2Result` (1D bar plot or 2D diverging heatmap). |
| `plot_contour` | function | Plot Minos profile-likelihood confidence-region contour(s) in a 2D parameter plane (e.g. `(x, y)` mixing) from a fitted `Minuit` result (any `Minimizer.fit()`/`*FitSession.fit()` return value), via `Minuit.mncontour` -- not the Gaussian/covariance-ellipse approximation. Not session-specific. |

Docs: `docs/user_friendly_api.md` ("Automatic projections", "Plot helpers"), `docs/goodness_of_fit.md`.

## High-level sessions

Composition layers over everything above; see `docs/user_friendly_api.md` "Design principle" for what they intentionally do *not* replace.

| Name | Kind | What it does |
|---|---|---|
| `FitSession` | class | Compose PDF + likelihood + backgrounds + constraints + minimizer for one sample in a few lines; `fit()`, `report()`, `plot_projection()`, `goodness_of_fit_projection()`/`goodness_of_fit_chi2()`/`point_to_point_dissimilarity()`, `.from_root(...)`. |
| `CPFitSession` | class | Same composition for simultaneous B+/B- fits over `CPJointNLL`; shared `Parameter`s collected once; same goodness-of-fit methods, per charge. `fit(weights=(plus, minus), covariance="sandwich")` or `with_event_weights(plus, minus)` runs a signal-only sWeight/COW fit; `plot_projection(..., show_amplitude_components=True)` also draws each amplitude component and the interference ([cp_coefficients.md](cp_coefficients.md#event-weighted-sweightcow-cp-fits)). |
| `CPFitSession.component_cp_asymmetries` | method | Integrated CP asymmetry `(I_k^- - I_k^+)/(I_k^- + I_k^+)` of every component, including CP violation in its dynamics (e.g. a QMI S-wave's nodes), with joint-covariance delta-method errors. See `docs/cp_coefficients.md`. |
| `CPProjectionToy` | class | Reusable host signal/background toys returned by `CPFitSession.prepare_projection_toy`; `plot_projection_from_toy` preserves all CP projection options. See [generated CP projections](user_friendly_api.md#cp-projections-from-generated-toys). |
| `TimeDependentFitSession` | class | Composes `TimeDependentDalitzNLL`: builds the shared A+Abar `PreparedAmplitudeCache` (Abar derived by reflection unless an explicit `abar_model` is given, for direct CPV) and collects `Parameter`s from the model(s) and `mixing`. `fit()`/`fit_multistart()`/`report()`/`print_result()`/`print_fit_fractions()`/`fit_fraction_errors()`, Supports `.with_background(...)`, conditional fractions or extended yields with component tag fractions; `signal_objective` retains the signal kernel. Projections include all backgrounds when marginal callbacks are available. `plot_time_projection()` overlays each tag's decay-time histogram against the exact Dalitz-integrated curve (unit acceptance/perfect resolution only). `plot_projection()` overlays each tag's Dalitz-variable histogram (one subplot per tag) against the time-integrated, tag-conditional density, mirroring `CPFitSession.plot_projection`'s two-population layout. |

Docs: `docs/user_friendly_api.md`, `docs/time_dependent.md`. Notebooks: `notebooks/tutorials/tutorial_02_first_fit.ipynb`, `notebooks/tutorials/tutorial_06_joint_cp_fit.ipynb`, `notebooks/examples/babar_2008_d0_kspipi.ipynb` (time-dependent).

## Configuration

| Name | Kind | What it does |
|---|---|---|
| `enable_x64` | function | Enable or disable JAX 64-bit floating-point precision. Importing `jaxpwa` already calls this with `enabled=True` unless `JAX_ENABLE_X64` was set explicitly first; call `enable_x64(False)` directly only to opt back into an explicit, unvalidated float32 experiment. |

Docs: `README.md` "Installation", `docs/fitting.md`.
