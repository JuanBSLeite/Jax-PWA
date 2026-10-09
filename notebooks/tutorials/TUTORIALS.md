# Learn Jax-PWA

A progressive course in English, based on the [public API catalog](../../docs/catalog.md).
Each notebook contains explanations, executable examples, diagnostic checks and exercises.
Start with lessons 1–2; proceed to normalization before adapting a fit to your own data.
Every lesson is self-contained and generates its own data. The models are illustrative,
not reproductions of published measurements.

## Setup

From the repository root, create or activate a Python environment (Python 3.12 or newer), then:

```bash
python -m pip install -e ".[dev]"
python -m pip install jupyterlab
python -m jupyter lab notebooks
```

Select the kernel belonging to this environment. Run each notebook from top to bottom
in a fresh kernel. Installation is done once in a terminal, not inside the lessons.
`enable_x64()` is called before numerical work. A CPU is sufficient; first-use JAX
compilation can take longer than subsequent evaluations. Runtime depends on the machine.
Examples use modest event counts and grids; increase resolutions and establish convergence
before drawing physics conclusions. Inverse-CDF and normalization resolutions control
separate approximations. No remote data, GPU, or ROOT installation is required.

## Learning path

| Lesson | Notebook | Main public APIs |
|---|---|---|
| 1 | [From a decay channel to a Dalitz plot](tutorial_01_phase_space_and_models.ipynb) | DecayChannel, PhaseSpaceSample, DecayModel, RealImag, plotting |
| 2 | [Your first unbinned amplitude fit](tutorial_02_first_fit.ipynb) | FitSession, Parameter, generate_toy, fit fractions |
| 3 | [Normalization, integration weights and convergence](tutorial_03_normalization.ipynb) | normalization_sample, MassWindowVeto, integration weights |
| 4 | [Efficiency, vetoes, backgrounds and constraints](tutorial_04_acceptance_and_backgrounds.ipynb) | BackgroundSpec, ToyBackground, GaussianConstraint |
| 5 | [Floating dynamics and the low-level fit API](tutorial_05_dynamics_and_low_level_api.ipynb) | PreparedAmplitudeCache, Minimizer, MultiBackgroundNLL |
| 6 | [A joint fit to both charges](tutorial_06_joint_cp_fit.ipynb) | CPRealImag, generate_cp_toy, CPFitSession |
| 7 | [From ROOT events to a reproducible fit](tutorial_07_root_io.ipynb) | write_phase_space_sample, read_phase_space_sample, FitSession.from_root |
| 8 | [Como usar o QMI](tutorial_08_qmi.ipynb) | QMI polar/cartesiano, interpolação, uso em um modelo de Dalitz |
| 9 | [Goodness of fit](tutorial_09_goodness_of_fit.ipynb) | BinnedChi2Result, chi2 1D/2D, point_to_point_dissimilarity, plot_pulls |
| 10 | [QMI S-wave isobar closure](../validation/b2pipipi_qmi_isobar_closure_no_cp_mag_phase.ipynb) | QMI, magnitude/phase recovery, closure diagnostics |
| 11 | [QMI Cartesian isobar closure](../validation/b2pipipi_qmi_isobar_closure_no_cp.ipynb) | Cartesian QMI nodes, coefficient recovery, fit validation |
| 12 | [QMIPixel: campo de Dalitz e ajuste de toy](tutorial_10_qmi_pixel_dalitz_closure.ipynb) | QMIPixel, DalitzAmplitude, máscara física, folding, interpolação e ajuste de magnitude/fase |

| 13 | [Line shapes opcionais com SymPy](tutorial_11_sympy_lineshapes.ipynb) | SympyLineshape, parâmetros explícitos, gradientes, ajuste Asimov e JSON; requer o extra `sympy` |
| 14 | [SymPy no Dalitz de B → 3π](tutorial_12_sympy_b3pi_dalitz.ipynb) | Polo definido por SymPy, spin=1 com Zemach_P, fatores de forma, simetrização de π⁺ idênticos e gradientes |

## Feature reference (one notebook per API, lessons 15-64)

Lessons 1-14 cover the main fitting workflow. The 50 short, focused notebooks below each
demonstrate exactly one remaining public class or function in isolation (a plot, a minimal
model/toy/fit, and a sanity-check assertion) -- not a progressive course, a lookup set. Start
from [the feature index](tutorial_64_feature_index.ipynb) for a decision table grouped the same
way as [`docs/catalog.md`](../../docs/catalog.md), or jump straight to a notebook below.

| # | Notebook | Demonstrates |
|---|---|---|
| 15 | [Pole](tutorial_15_pole.ipynb) | `Pole` fixed-width Breit-Wigner pole |
| 16 | [SigmaPole](tutorial_16_sigma_pole.ipynb) | `SigmaPole` LHCb f0(500) convention |
| 17 | [GounarisSakurai](tutorial_17_gounaris_sakurai.ipynb) | `GounarisSakurai` spin-1 lineshape |
| 18 | [RhoOmegaMixing](tutorial_18_rho_omega_mixing.ipynb) | `RhoOmegaMixing` coherent mixing |
| 19 | [PipiKKRescattering](tutorial_19_pipi_kk_rescattering.ipynb) | `PipiKKRescattering` pi-pi to K-Kbar |
| 20 | [Flatte (generic)](tutorial_20_flatte_generic.ipynb) | `Flatte` two-channel coupled lineshape |
| 21 | [Flatte presets](tutorial_21_flatte_presets.ipynb) | `Flatte.f0_980()`/`.k0star_1430_*()`/`.a0_980_*()` |
| 22 | [BaBarFlatte](tutorial_22_babar_flatte.ipynb) | `BaBarFlatte` BaBar convention |
| 23 | [LASS](tutorial_23_lass.ipynb) | `LASS` effective-range + K0*(1430) |
| 24 | [KMatrix](tutorial_24_kmatrix.ipynb) | `KMatrix` five-pole five-channel S-wave |
| 25 | [Rescattering2](tutorial_25_rescattering2.ipynb) | `Rescattering2` Laura++ port |
| 26 | [Angular models](tutorial_26_angular_models.ipynb) | `CovariantAngular` vs `ZemachP`/`ZemachPstar` |
| 27 | [Narrow-resonance grid](tutorial_27_narrow_resonance_grid.ipynb) | Automatic narrow-resonance refinement |
| 28 | [Export/import a model](tutorial_28_export_import_model.ipynb) | `export_model`/`import_model` |
| 29 | [Export/import CP models](tutorial_29_export_import_cp_models.ipynb) | `export_cp_models`/`import_cp_models` |
| 30 | [Model with fitted values](tutorial_30_model_with_fitted_values.ipynb) | `model_with_fitted_values`, `fit(update_model=True)` |
| 31 | [Vetoes](tutorial_31_vetoes.ipynb) | `MassWindowVeto`/`CompositeVeto`/`FunctionalVeto` |
| 32 | [Vetoed density](tutorial_32_vetoed_density.ipynb) | `VetoedDensity`/`vetoed_signal_pdf` |
| 33 | [SCF map (dense)](tutorial_33_scf_map_dense.ipynb) | `SquareDalitzSCFMap` |
| 34 | [Sparse migration](tutorial_34_sparse_migration.ipynb) | `SparseMigration` |
| 35 | [Histogram maps from arrays](tutorial_35_histogram_maps_from_arrays.ipynb) | `SquareDalitzHistogramEfficiency`/`Background` |
| 36 | [Histogram maps from ROOT](tutorial_36_histogram_maps_from_root.ipynb) | `histogram_efficiency_from_root`/`histogram_background_from_root` |
| 37 | [Low-level ROOT I/O](tutorial_37_root_low_level_io.ipynb) | `write_cp_phase_space_sample`/`read_root_histogram2d` |
| 38 | [Toy method comparison](tutorial_38_toy_methods_comparison.ipynb) | `method="accept-reject"` vs `"inverse-transform"` |
| 39 | [Prepared inverse toy generator](tutorial_39_prepared_inverse_toy_generator.ipynb) | `prepare_inverse_toy_generator` |
| 40 | [Weighted resample](tutorial_40_weighted_resample.ipynb) | `weighted_resample` |
| 41 | [Generate CP toy](tutorial_41_generate_cp_toy.ipynb) | `generate_cp_toy` |
| 42 | [Discriminant PDFs](tutorial_42_discriminant_pdfs.ipynb) | `Gaussian1D`/`Exponential1D`/`Histogram1D`/`BreitWigner1D` |
| 43 | [Factorized density + constraints](tutorial_43_factorized_density_and_constraints.ipynb) | `FactorizedDensity`/`GaussianConstraint`/`ConstrainedNLL` |
| 44 | [Convolution](tutorial_44_convolution.ipynb) | `ConvolvedPDF1D`/`GaussianResolution1D` |
| 45 | [Delta method](tutorial_45_delta_method.ipynb) | `delta_method_jacobian`/`covariance`/`errors` |
| 46 | [KD-tree local residuals](tutorial_46_kdtree_local_residuals.ipynb) | `kdtree_local_residuals` |
| 47 | [Point-to-point](tutorial_47_point_to_point.ipynb) | `PointToPointResult`/`point_to_point_dissimilarity` |
| 48 | [Plotting helpers](tutorial_48_plotting_helpers.ipynb) | `plot_dalitz`/`plot_square_dalitz`/`binned_data`/`plot_pulls` |
| 49 | [Fit fractions](tutorial_49_fit_fractions.ipynb) | `DecayModel.fit_fractions`/`interference_fractions` |
| 50 | [Multistart fitting](tutorial_50_fit_multistart.ipynb) | `fit_multistart`/`MultiStartResult` |
| 51 | [Nesterov optimizer](tutorial_51_nesterov_optimizer.ipynb) | `method="nesterov"`/`NesterovResult` |
| 52 | [Gradient check](tutorial_52_check_gradient.ipynb) | `Minimizer.check_gradient` |
| 53 | [JAX Hessian](tutorial_53_hessian_jax.ipynb) | `hessian="jax"` vs `"numerical"` |
| 54 | [Square-Dalitz conversions](tutorial_54_square_dalitz_conversions.ipynb) | `invariants_to_square_dalitz`/`square_dalitz_jacobian`/`fold_thetaprime` |
| 55 | [Yield asymmetry](tutorial_55_yield_asymmetry.ipynb) | `YieldAsymmetry` |
| 56 | [Shared parameters](tutorial_56_shared_parameters.ipynb) | Sharing/conflicting `Parameter` objects |
| 57 | [float32 experiment](tutorial_57_float32_experiment.ipynb) | `enable_x64(False)` |
| 58 | [Manual CoherentAmplitudeModel](tutorial_58_coherent_amplitude_model_manual.ipynb) | `CoherentAmplitudeModel`/`AmplitudeComponent`/`ConstantAmplitude` |
| 59 | [PreparedAmplitudeCache direct](tutorial_59_prepared_cache_direct.ipynb) | `PreparedAmplitudeCache` used directly |
| 60 | [BackgroundCategory vs BackgroundSpec](tutorial_60_background_category_vs_spec.ipynb) | `BackgroundCategory` vs `BackgroundSpec` |
| 61 | [CP background spec/category](tutorial_61_cp_background_spec_and_category.ipynb) | `CPBackgroundSpec`/`CPBackgroundCategory` |
| 62 | [Resonance internals](tutorial_62_resonance_internals.ipynb) | `ResonanceAmplitude`/`ResonanceContext` |
| 63 | [DalitzAmplitude + QMIPixel](tutorial_63_dalitz_amplitude_qmi_pixel.ipynb) | `DalitzAmplitude` wrapping `QMIPixel` |
| 64 | [Feature index](tutorial_64_feature_index.ipynb) | Decision table linking back to all of the above |
| 65 | [Weighted unbinned NLL](tutorial_65_weighted_unbinned_nll.ipynb) | `WeightedUnbinnedNLL` |

## Continue with focused examples

The course covers the main fitting workflow, not every specialized catalog entry.
Use these existing notebooks and documents for extensions:

| Topic | Next example or reference |
|---|---|
| Four-body amplitudes and toy closure | [Four-body closure](tutorial_66_four_body_closure.ipynb), [physics and scope](../../docs/four_body.md) |
| Multiple backgrounds and extended fits | [Multiple backgrounds](../tests/b2kpipi_multiple_backgrounds.ipynb), [background conventions](../../docs/backgrounds_and_vetoes.md) |
| Detector migration / self-cross-feed | [SCF migration](../tests/b2kpipi_scf_migration.ipynb), [SCF with veto](../tests/b2kpipi_scf_with_veto.ipynb) |
| Discriminating variables | [Discriminant PDFs](tutorial_42_discriminant_pdfs.ipynb) |
| Square-Dalitz histogram maps | [Efficiency and background maps](../tests/b2kpipi_square_dalitz_eff_background.ipynb) |
| Repeated toy generation | [Toy generation](../tests/user_friendly_toy_generation.ipynb), [prepared generators](../../docs/toy_generation.md) |
| One-dimensional resolution | [PDF convolution](../tests/pdf_convolution_resolution.ipynb) |
| Identical-particle folding | [Folded histogram maps](tutorial_35_histogram_maps_from_arrays.ipynb), [folding conventions](../../docs/backgrounds_and_vetoes.md) |
| Alternative lineshapes, QMI and QMIPixel | [Lineshape documentation](../../docs/lineshapes.md), [dynamics structure](../../docs/dynamics_structure.md) |

The `../examples/` and `../validation/` directories contain worked analyses, closure studies and
numerical reproductions rather than introductory lessons. Consult the
[convention review](../../docs/reviews/paper_isobar_conventions.md) for the remaining
publication-reproduction discrepancies in SigmaPole, RhoOmegaMixing and PipiKKRescattering.
