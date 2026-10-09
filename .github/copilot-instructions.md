# Copilot instructions for Jax-PWA

## What this is

Jax-PWA is an unbinned amplitude-fit package for three-body decays. The numerical
pipeline is **JAX end to end**: phase-space generation, kinematics, amplitude dynamics,
normalization, likelihoods and gradients all run as JAX arrays/ops. `iminuit` only performs the
final minimization, consuming a JAX `value_and_grad` objective. `particle` supplies standard
particle properties and `uproot` provides ROOT I/O without PyROOT. There is no TensorFlow
dependency anywhere in the numerical path.

Laura++ is the primary physics reference for resonance, barrier-factor, angular and
Square-Dalitz conventions, but classes use neutral names rather than Laura++-specific ones.
When a convention is ported from Laura++ (e.g. `Rescattering2`, the K-matrix, `Zemach_P`/
`Zemach_Pstar`), the docstring says so explicitly and any deliberate deviation from the
upstream C++ (e.g. a typo it contains) is documented and justified in the docstring — don't
assume the literal upstream formula is followed if the docstring says otherwise.

## Commands

```bash
python -m pip install -e ".[dev]"     # install with test + ruff extras

pytest                                 # full suite (testpaths = tests/, ~770 tests, several minutes)
pytest tests/test_rescattering2.py -v  # single file
pytest tests/test_rescattering2.py::test_rescattering2_matches_laura_reference_points  # single test

ruff check .                           # lint (configured, not CI-enforced; repo has existing lint debt,
                                        # so check new/changed files rather than expecting a clean run)

python benchmarks/benchmark_fit_evaluation.py --events 100000 --normalization-resolution 1000 --repeats 20
python benchmarks/benchmark_toy_generation.py --size 100000
python benchmarks/benchmark_scf_migration.py --bins-mprime 40 --bins-thetaprime 40
python benchmarks/benchmark_cache_stages.py --events 100000 --normalization-resolution 1000
python benchmarks/benchmark_qmi_memory_speed.py
python benchmarks/benchmark_time_dependent.py --resolution 20
```

CI (`.github/workflows/tests.yml`) runs `pytest tests` on Python 3.12, 3.13, and 3.14, plus a notebook
sanity check that parses every notebook under `notebooks/tutorials/`, `notebooks/examples/`,
`notebooks/validation/` and `notebooks/tests/` with `nbformat` and compiles (not executes) each
code cell. The check fails if any of those directories is missing or empty, so a notebook
reorganization must update the directory list in both `tests.yml` and `full-validation.yml`.
Because cells are only compiled, a notebook that imports an undeclared package (e.g. `pandas`)
or calls a renamed API still passes CI.
`full-validation.yml` and `toy-benchmark.yml` are `workflow_dispatch`-only and
not run on every push. There is no GPU CI; `docs/gpu_ubuntu_24_04.md` documents the manual
WSL2/CUDA reference environment used for GPU validation.

Importing `jaxpwa` enables JAX 64-bit precision automatically (unless
`JAX_ENABLE_X64` was already set in the environment, which stays authoritative) — the project
deliberately runs float64/complex128 for amplitude-analysis stability, and this is not JAX's own
default. `enable_x64(False)` (`from jaxpwa import enable_x64`) remains available to opt
back into float32 for an explicit, validated experiment; nothing in the numerical path is expected
to be correct or to converge reliably (Minuit's EDM-based convergence check in particular) under
float32.

## Architecture

### Pipeline

```
PhaseSpaceSample (data or generated)
  -> DecayChannel + amplitude components (Resonance / NonResonant / DalitzAmplitude)
  -> pure-JAX kinematics and dynamics
  -> deterministic Dalitz / Square-Dalitz quadrature, or external toy-MC normalization
  -> coherent amplitude -> PreparedAmplitudeCache
  -> optional efficiency/veto/SCF/backgrounds/discriminating-variable PDFs/1D convolution
  -> optional Gaussian constraints
  -> FitSession / CPFitSession (optional convenience layer)
  -> JAX NLL + automatic gradient -> Minimizer (Nesterov prefit and/or iminuit)
```

### Two API layers, by design

`FitSession`/`CPFitSession` (`workflow.py`, `cp_workflow.py`) are *composition* layers over the
low-level public classes (`SignalPDF`, `PreparedAmplitudeCache`, `MultiBackgroundNLL`,
`CPJointNLL`, `Minimizer`) — they do not replace them. Advanced/validation work should still go
through the low-level classes directly; see `docs/user_friendly_api.md` "Design principle".

`Minimizer.fit(method="nesterov")` provides a projected, parameter-scaled
Nesterov first-order fit. `method="nesterov-minuit"` runs that prefit before
the existing Minuit strategy 1/2 stages. Invalid or worsened MIGRAD results
are rejected, and a later stage cannot replace an earlier stage with a higher
NLL. Nesterov-only results have no covariance and must not be used for
uncertainty reporting without a separate Hessian calculation.

### One-dimensional lineshapes vs. full 2D Dalitz amplitudes

`Resonance` composes a lineshape through the ordinary `lineshape(mass, context)` interface
(`dynamics/lineshape/*.py`, one physics model per file: `relativistic_breit_wigner.py`,
`gounaris_sakurai.py`, `flatte.py`, `babar_flatte.py`, `pole.py` (`Pole`, `SigmaPole`),
`lass.py`, `kmatrix.py`, `qmi.py`, `rescattering2.py`, `pipi_kk_rescattering.py`, `rho_omega.py`,
plus `sympy.py`'s `SympyLineshape` for user-written symbolic lineshapes via the optional `sympy`
extra) combined with an angular factor (`dynamics/angular.py`) and Blatt-Weisskopf
barriers. `DalitzAmplitude` bypasses that isobar construction entirely for amplitudes that are
intrinsically two-dimensional (`QMIPixel`, `dynamics/lineshape/qmi_pixel.py`), evaluated directly over
`(s12, s13)`.

`QMI`'s `interpolation="cubic"` is not a cubic spline in the usual sense: it is a strictly local
smoothstep between the two knots bordering an event's interval, and that locality is exactly what
forces its derivative to zero at every knot (the zero-derivative value is the only one guaranteed
to match across neighboring intervals without consulting further knots) — the visible
flatten-then-bulge look this produces is by design, not a bug. `interpolation="hermite"` removes
it at the same per-event cost (knot tangents are precomputed once from neighbors, then reused
through the same grouped-VJP reduction as `cubic`) and should be preferred whenever a smooth
curve matters; see `docs/lineshapes.md` and `docs/performance.md`.

### Normalization: the central invariant

Every integral in the codebase (grid quadrature *and* Monte Carlo) follows one convention:
`integral(f) = mean(sample.weights * f)`. This is why `PhaseSpaceSample` weight semantics matter
so much — an unweighted signal toy is *not* generally a valid unit-weight integration sample; see
`docs/mc_integration.md` before touching anything that consumes `normalization_sample`.

Three `DecayModel` normalization methods exist (`normalization_method=`): `gauss-legendre`
(default, mass-plane quadrature), `square-dalitz` (Square-Dalitz quadrature, matches Laura++'s
prescription for diagonal narrow structures), and `toy-mc` (auto-selected when
`normalization_sample=` is supplied). Narrow resonances (width <= 20 MeV by default) trigger
automatic local grid refinement; see the "Normalization" section of `README.md`.

Two independent normalization switches exist and are easy to conflate:
- `DecayModel(normalize_components=...)` — model-wide default for per-component unit-integral
  rescaling before the complex coefficient is applied.
- `Resonance(..., normalize_component=True|False|None)` — per-component override; `None`
  inherits the model default. A component with `normalize_component=False` still enters the full
  Hermitian normalization matrix `M_ij = integral conj(F_i) F_j dPhi` (interference is never
  skipped), only its own unit-integral rescaling is skipped.

### Prepared caching is a performance-critical pattern, not an implementation detail

`PreparedAmplitudeCache` (`amplitude/cache.py`) separates what is fixed for a Minuit step
(component values + normalization matrix on the data/normalization sample) from what must be
recomputed when a `ParameterKind.DYNAMICS` parameter floats (only the affected component and its
normalization-matrix row/column). Coefficient-only fits never re-evaluate lineshapes or
re-integrate the Dalitz plot after the first prepare. When changing anything in the fit hot path
(`decay.py`, `amplitude/cache.py`, `likelihood/*.py`), check `docs/performance.md` for which
computation is supposed to be cached vs. recomputed, and re-run the relevant benchmark rather
than assuming a change is free.

The fixed/floating split for each `ParameterKind.DYNAMICS` parameter is baked in once, from the
`parameters` given to `PreparedAmplitudeCache.prepare()`. If a low-level caller then builds
`Minimizer` with a *different* `Parameter` list for the same names (bypassing `DecayModel`/
`FitSession`, which always thread one shared list and can't drift), the cache silently keeps
ignoring that parameter — a structural zero gradient, not an approximate one, with no error.
Call `cache.check_parameters(parameters)` before handing a separately-built parameter list to
`Minimizer`; see `docs/performance.md`.

### CP fits share one normalization across charges

`CPJointNLL` (`likelihood/cp.py`) treats charge as part of the fitted sample space: B+/B- are
normalized jointly (`p(phi,+) = |A_plus|^2 / (I_plus + I_minus)`), not as two independently
normalized Dalitz likelihoods. This is what gives the fit sensitivity to the integrated charge
asymmetry. `cp_workflow.py`'s projection helpers must preserve this — e.g.
`_joint_scaled_weights` divides each charge's projection sample by its *own* size before summing,
since adaptive/external MC samples for B+ and B- need not have equal size. See
`docs/cp_coefficients.md`.

`YieldAsymmetry` (`likelihood/cp.py`) is the one sanctioned escape hatch from this joint split: it
replaces a plain `signal_yield` with independently fittable literal `N_plus`/`N_minus`, split from
a total `N_s` via a raw counting yield asymmetry, for when the observable of interest is a
production/detection-driven counting asymmetry rather than the amplitude-driven one. Each charge's
isobar PDF is then normalized on its own (`|A_plus|^2 / I_plus`, `|A_minus|^2 / I_minus`) instead
of jointly, so `N_plus`/`N_minus` are already standalone per-charge counts — `cp_workflow.py`'s
signal-projection scaling must *not* reweight them by `integral_q / norm` the way a shared
`signal_yield` is. See `docs/cp_coefficients.md`, "Yield-asymmetry parameterization".

### Time-dependent neutral-meson mixing

`NeutralMesonMixing`/`TimeDependentDalitzNLL` (`likelihood/time_dependent.py`) are a separate API
from `CPJointNLL`: an unbinned `(Dalitz, decay-time)` likelihood conditional on an observed
initial-flavour tag (D0 vs D0bar), following Belle PRD 89, 091103 (2014) Eqs. (1-2). One
`PreparedAmplitudeCache` holds A components followed by Abar components, evaluated at the SAME
final-state coordinates and integration sample -- there is no automatic identical-particle
folding here (e.g. pi+/pi- stay distinct). `TimeDependentFitSession`
(`time_dependent_workflow.py`) is the composition layer, following the same "does not replace the
low-level classes" principle as `FitSession`/`CPFitSession`: by default it derives Abar by
reflection, `Abar(s12,s13)=A(s13,s12)` (no-direct-CPV); pass an independently built `abar_model`
for direct CPV instead.

Three `TimeDependentDalitzNLL` methods marginalize the joint density differently and back the
session's plotting, rather than each plot re-deriving the physics: `dalitz_integrated_time_pdf`
(Dalitz-integrated, keeps t; requires unit temporal acceptance/perfect resolution),
`tag_marginal_density` (time-integrated over `time_range`, keeps Dalitz), and
`dalitz_density_at_time` (neither marginalized -- a Dalitz snapshot at one fixed t, normalized to
integrate to 1 *at that t*, isolating the mixing-driven shape evolution from the trivial
exp(-t/tau) yield decay). All three require a scalar `wrong_tag` (a single curve/snapshot needs
one representative mistag probability, not the per-event values a fit may use). `plot_contour`
(`plotting.py`) is not session-specific -- it draws Minos profile-likelihood contours via
`Minuit.mncontour` for any fitted `Minuit` result (any session's `fit()` return value), e.g. the
`(x,y)` confidence region a mixing measurement publishes. See `docs/time_dependent.md`.

### Folded Dalitz Plot / Square Dalitz Plot for identical particles

For a channel with two identical final-state particles, `DecayChannel` detects them
automatically from PDG IDs, and `Resonance` amplitudes are symmetrized under their exchange with
no extra configuration — an unbinned fit is therefore already correct on the full (unfolded)
plane. `folded=True` is instead a **statistics** tool for building efficiency/background maps
from a limited sample or for diagnostic plots, not a correctness requirement:
`HistogramEfficiency`/`HistogramBackground` (plain `s12`/`s13`, `efficiency`/`background`
modules), `SquareDalitzHistogramEfficiency`/`SquareDalitzHistogramBackground` (`(m', theta')`,
`square_histograms.py`), `plot_dalitz`/`plot_square_dalitz`, and `FitSession`/
`CPFitSession.plot_projection` (`fold_side="low"|"high"`) all fold with the same `min`/`max`
convention `QMIPixel(folded=True)` already uses. `pair` (or `x_edges == y_edges`) **must be the
actual identical pair** — nothing can check this from `masses` alone, since two *distinct*
particles (e.g. `pi+`/`pi-`) can share a mass without being identical; picking the wrong pair
folds by a symmetry the data don't have, silently. See `docs/backgrounds_and_vetoes.md`. The
normalization/generation grid (`SquareDalitzGrid`, `DecayModel`'s automatic selection) is
deliberately never folded — halving it would only be correct if the coherent amplitude were
itself exactly symmetric, which nothing here can verify automatically.

### Toy generation: two independent public samplers

`inverse-transform` (default; numerical Rosenblatt transform, tabulated CDFs, `docs/toy_generation.md`)
and `accept-reject` (Laura++-style envelope/restart algorithm, explicit `method="accept-reject"`).
They are validated against each other, not derived from one another — don't assume one is a
special case of the other. Both keep event arrays on the JAX device: inverse-transform builds its
CDF tables and draws candidates with `jax.random`, so its first call in a process pays a one-time
JIT compilation cost that repeated `prepared.generate(...)` calls of the same size reuse.

### Four-body amplitudes are an additive, narrower API

`FourBodyDecayModel`/`NBodyDecayChannel` (`four_body.py`, `kinematics/four_body.py`,
`kinematics/nbody.py`) cover a spin-zero parent decaying to four spin-zero daughters, built from
`Isobar`, `PairChain` and `CascadeChain` over the same `lineshape(mass, context)` contract,
amplitude cache, `mean(weights*f)` normalization (weighted `NBodyPhaseSpaceMC`, never a 2D grid)
and `Minimizer`. It does not extend `DecayModel`/`PhaseSpaceSample`, and most three-body
conveniences (Dalitz/Square-Dalitz maps, SCF, ROOT/model serialization, CP and time-dependent
sessions, `FitSession.report`) are deliberately not four-body APIs yet. Check
`docs/four_body.md`'s boundary table before assuming a three-body feature carries over.

## Where to look before changing behavior

`docs/catalog.md` indexes every name in `jaxpwa.__all__` (the whole public,
top-level-importable API) — one line per class/function plus a pointer to the doc/notebook that
covers it in depth. Check it first for "does something already do X" or "where is X" questions
before searching the source directly.

Each subsystem also has one focused doc under `docs/` (`fitting.md`, `lineshapes.md`,
`mc_integration.md`, `backgrounds_and_vetoes.md`, `cp_coefficients.md`, `scf.md`,
`square_dalitz.md`, `toy_generation.md`, `discriminants_and_constraints.md`,
`convolution_resolution.md`, `dynamics_structure.md`, `performance.md`, `root_io.md`,
`model_io.md`, `user_friendly_api.md`, `goodness_of_fit.md`, `time_dependent.md`,
`four_body.md`). `docs/reviews/` contains dated, adversarial numeric-reproduction review
write-ups (concrete inputs, reproduced numbers, "Applied fixes" sections, or — as in
`paper_isobar_conventions.md` — an explicit "Remaining discrepancy" section when a reproduction
is not yet closed) — this repo's working style is to reproduce a suspected discrepancy
numerically before changing formulas, and to update the corresponding `docs/*.md` in the same
change that fixes the code; an open "Remaining discrepancy" means the corresponding lineshape
(`SigmaPole`, `RhoOmegaMixing`, `PipiKKRescattering`) is not yet validated to publication
precision and should not be treated as certified. `notebooks/tutorials/` are the tutorial/example
set referenced by the docs and README (`notebooks/tutorials/TUTORIALS.md`);
`notebooks/examples/` holds complete worked analyses on published or realistic configurations
(e.g. `B+ -> K+ pi+ pi-` fits, the BaBar 2008 `D0 -> KS pi pi` model, the LHCb 2023
`Ds -> 3pi` fit, BESIII `D0 -> 4pi`, and Laura++-generated Square-Dalitz toys for the LHCb
`B -> 3pi` isobar model); `notebooks/validation/` holds closure and pull studies;
`notebooks/tests/` holds informal per-feature check notebooks (constraints, SCF, vetoes, ROOT
input). All of these consume the same public API and can break silently when a lineshape or
normalization convention changes underneath them, since CI only compiles their cells. In-progress
physics analyses are kept outside the package notebooks (e.g. the untracked
`AmAn_B2KKK_Run2/`), not under `notebooks/`.

### Histogram interpolation and Square-Dalitz acceptance

Histogram efficiency and background models accept `interpolation="none"`,
`"linear"`, or `"spline"`. The spline mode is JAX-differentiable cubic
interpolation over bin-centre values. Efficiency maps do not get normalized to
unit integral; use `clip=True` only when reproducing the reference convention
that bounds interpolated efficiencies to `[0, 1]`. Square-Dalitz backgrounds use
`divide_jacobian=True` for PDF evaluation and their raw `generation_value()` for
toy generation. Apply vetoes before acceptance/background normalization.
