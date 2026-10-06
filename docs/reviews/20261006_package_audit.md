# Package audit: lineshapes, integration, fitting and caching

Date: 2026-10-06. Scope: numerical contracts and regression coverage in the
existing package, with targeted reproductions and preparation benchmarks.
This is not a new publication-level validation of every physics convention.

## Environment and method

- Repository virtual environment: Python 3.12, JAX 0.11.2, iminuit 2.33.0.
- CPU: Intel Core i7-11800H; GPU: GeForce RTX 3050 Ti Laptop, 4 GiB.
- Float64/complex128 retained. CPU tests explicitly select `JAX_PLATFORMS=cpu`;
  GPU runs select `JAX_PLATFORMS=cuda`, with preallocation disabled.
- Read the catalog, subsystem documentation and earlier physics reviews before
  examining the numerical implementation. Reproduce defects before changing it.
- The untracked analysis directory `AmAn_B2KKK_Run2/` was not modified.
- Timings are local diagnostics with other CPU work running, not an isolated
  hardware benchmark or a guarantee of complete-fit speedup.

## Reproduced defects and applied fixes

### Cache validation missed changed bindings within the same component

`PreparedAmplitudeCache.check_parameters()` previously compared only the set of
floating dynamics owners. Renaming `a.slope` to `a.new_slope`, retaining owner
`a`, passed the check even though the cache still read the original name. The
reproduced derivative in `a.new_slope` was exactly zero. Adding/removing a
parameter, changing its backend alias, or changing a fixed value on an already
floating component also went undetected.

The check now compares each dynamics binding's name, owner, effective backend
alias, fixed status and fixed value. Changes to a free starting value, optimizer
step or bounds remain allowed. This is still an explicit low-level check, not
automatic discovery of every parameter nested inside user dynamics.

### Compact preparation incorrectly rejected unnormalized zero components

A fixed zero-valued component with `normalize_component=False` was rejected
because compact preparation required every raw matrix diagonal to be positive.
The dynamic path already restricted that requirement to normalized components.
Compact preparation now follows the same rule, preserving zero rows/columns and
avoiding dormant `1/sqrt(0)` operations. Components requesting unit-integral
rescaling still reject a zero integral; a zero total PDF remains invalid.

### Shared Minuit backend omitted the Hessian memory schedule

Two minimizers wrapping the same objective and parameter names shared compiled
callbacks even if a parameter changed from coefficient to dynamics. The second
minimizer could inherit the coefficient-only Hessian program instead of the
checkpointed, sequential-HVP program. This affects memory scheduling, not the
analytic Hessian's mathematical definition. Floating-dynamics status now enters
the shared-backend key.

### Final polishing could return a worse Minuit stage

The restoration guard ran before the final invalid-result polishing pass, with
no guard after it. A controlled test using real Minuit, an objective `x**2`,
`ncall=1`, and a deliberately displaced polish returned NLL 40,000 after an
earlier stage had NLL 4. The final check now falls back to a retained Minuit
result, keeping parameter values, FMin metadata, validity and covariance
together. The reproduced fallback remains invalid; it does not manufacture
convergence. Compiled Python callbacks are shared by the result snapshot.

### K-matrix documentation disagreed with its implementation

`docs/lineshapes.md` showed `(1-s0/s)/(s-s0)` for the slowly varying terms,
whereas the code and its docstring use `(m_sq0-s0)/(s-s0)`, with
`m_sq0 = 1 GeV^2`. Corrected the displayed formulas. No lineshape formula or
physics convention was changed.

### Single-neighbour local residuals raised an indexing error

The follow-up NumPy inventory reproduced `IndexError` for observed points
`[[0], [2]]`, reference points `[[0], [1]]` and `k=1`. SciPy returns one
distance per observation without a neighbour axis in this case. Local residuals
now preserve this one-dimensional result instead of indexing a missing axis;
the same correction covers a reference sample containing only one point.

## Measured preparation improvement

The same NumPy Gauss-Legendre rule was repeatedly constructed for equal orders,
including inside adaptive axis-segment loops. A shared, bounded LRU now retains
up to 32 one-dimensional rules as read-only host arrays. Full grids, models and
JAX device buffers are not retained. Scaled intervals create independent arrays.
Mass-plane, Square-Dalitz, adaptive Square-Dalitz, discriminant and convolution
users share this helper. The cached rule is bitwise identical to NumPy's result.

Final uncached/cached comparison in fresh processes, means of the second and
third constructions (raw output: `20261006_package_audit_results.json`):

| Grid | Points | Before | After | Approximate ratio |
|---|---:|---:|---:|---:|
| Square-Dalitz, resolution 1000 | 1,000,000 | 117.0 ms | 15.1 ms | 7.7x |
| Adaptive mass plane, narrow bands on both axes | 1,480,776 | 584.8 ms | 107.4 ms | 5.4x |

Mean integration weights remained respectively `5.063084362537261` and
`5.063266806352790`. These are construction gains, not changes to the quadrature
accuracy or steady-state objective. The first call includes cold JAX kinematics
compilation, so it is reported separately by the benchmark.

Reproduce with `OPENBLAS_NUM_THREADS=1` and the intended JAX backend:

```bash
python benchmarks/benchmark_quadrature_preparation.py --uncached
python benchmarks/benchmark_quadrature_preparation.py
python benchmarks/benchmark_fit_evaluation.py --events 100000 --normalization-resolution 1000 --repeats 20
```

The initial five-component GPU fit benchmark reported compact preparation
6.129 s, warm value/gradient 1.601 ms, NLL `671442.1957703558`, gradient norm
`137948.69787917935`, and matrix diagonal within `2.3e-16` of unity. The third
dataset reusing model normalization prepared in 32.8 ms. These measurements use
phase-space events and do not establish minimization convergence.

The post-change run gave NLL `671442.1957703556` (difference about `2.3e-10`),
the identical displayed gradient norm, and the same normalization-matrix
diagonal extrema. Warm evaluation averaged 1.396 ms; cache preparation was
6.867 s. With concurrent work and no changes to coefficient-only likelihood
algebra, these timing fluctuations do not establish a faster complete fit.

A subsequent compilation audit found that initial compact preparation created
a duplicate data JIT instead of using the model-owned function retained for
later datasets. Sharing that function reduced second-dataset preparation from
3.199 s to 0.0281 s, preserving the NLL and gradient norm. The final run's
initial preparation was 7.15 s and warm value/gradient was 1.59 ms; the measured
benefit is avoided recompilation. A separate persistent-cache experiment reduced
normalization/data compilation in a second process from 2.68/2.37 s to 28/31 ms.
See the [NumPy and compilation follow-up](20261006_numpy_compilation.md) for the
24-module inventory, configuration and additional measured candidates.

## Pure-JAX quadrature experiment

`benchmarks/benchmark_legendre_rules.py` contains a pure-JAX Newton/Legendre
recurrence candidate, including a configurable loop-unrolling comparison.
The production default remains cached NumPy, as requested after comparing the
tradeoff. Both algorithms can cache a completed rule; comparing repeated JAX
recomputation with uncached NumPy would not measure production cache reuse.

At order 1000, maximum observed node and weight differences from NumPy were
`1.11e-16` and `6.19e-14`. The checked exact polynomial moments agreed within
`5e-16`. Relative differences in the smallest endpoint weights are larger,
around `8e-9`; this is a comparison between algorithms, not proof that either
set is a high-precision reference.

A preliminary GPU run with ordinary loops took about 409 ms including JIT and
56 ms on repeated recomputation. Unrolling 16 recurrence steps reduced warm
recomputation to roughly 10 ms but increased the first call to 554 ms. At order
2000 the ordinary-loop candidate took 322 ms cold versus NumPy's 790 ms.
Thus JAX is feasible and may help larger cold rules, but is not uniformly faster
for the current orders. This candidate is kept as an experiment, not silently
substituted into production grids.

## Remaining validation boundaries and performance opportunities

- Existing tests cover reference values, prepared/direct equivalence, QMI
  interpolation/derivatives, K-matrix unitarity, weighted matrix integrals,
  chunked value/gradient/Hessian consistency, and fit closure on test models.
  They do not certify every parameter point or every external analysis setup.
- The unresolved publication reproduction in `paper_isobar_conventions.md`
  remains unresolved, particularly mixed rho/omega normalization and rescattering
  interference. Passing its current regression tests does not close that issue.
- QMI `natural` still solves a knot-sized dense system. Factoring its fixed
  geometry or using a differentiable tridiagonal solve is a candidate to measure
  against values, gradients, Hessians and complete fits. Local QMI modes already
  have prepared geometry and grouped reverse reductions.
- The Hermitian dynamic matrix reduction is already optimized structurally;
  fewer pair integrals do not guarantee faster device execution. Do not replace
  it or change normalization resolution based only on operation counts.
- Parameter kinds remain important: a coefficient accidentally nested inside
  unchecked dynamics can still produce a cached, flat fit direction. The stronger
  list-consistency check does not infer such intent. Broader construction-time
  validation needs compatibility tests for custom and shared parameters.
- Floating narrow masses/widths require convergence checks over the fitted
  region as well as at the initial point; chunk size controls memory, not
  integration accuracy. No reduction of grid accuracy or precision was used.

## Validation results

Validation was run on both CPU and CUDA:

- The initial regression reproducer had 9 failures and 3 passes before fixes;
  all 12 passed after the corresponding cache/backend fixes.
- A CPU full-suite run completed 238 passing cases before it was deliberately
  interrupted to avoid duplicating expensive Hessian checks running on CUDA.
- The focused CUDA suite passed 328 cases covering lineshapes, integration,
  caches, minimization and related physics regressions.
- The remaining 34 CPU test files passed 410 cases. Comparing collected node
  IDs verified that the union of these runs and the regression follow-up
  covered all 882 cases collected at the initial audit checkpoint. Counts
  overlap and must not be added as unique cases; this was not one uninterrupted
  full-suite run.
- The final Minuit/regression check at that checkpoint passed 51 cases on CPU
  and 51 on CUDA, including the deliberately worsened-polish reproduction.
- The later shared-compilation-kernel change passed 96 relevant CUDA
  cache/model/workflow/regression cases.
- The final NumPy follow-up passed 28 CPU cases: the 19 audit regression cases
  and the goodness-of-fit tests, including both single-neighbour reproductions.
- The candidate-migration follow-up passed 72 focused CPU cases after retaining
  the device-resident Nesterov loop, reverting slower toy-generation paths and
  fixing the time-dependent toy tag-probability ratio. The detailed benchmark
  and distribution checks are in `20261006_numpy_compilation.md`.
- Ruff passed for the changed files without existing lint debt. Comparing
  baseline/current diagnostics for `decay.py`, `goodness_of_fit.py` and the
  existing fit benchmark found no added diagnostics; their 27, 10 and 2 existing
  diagnostics remain. `git diff --check` passed. The standalone envelope
  experiment also passed a CPU smoke run with exact output comparisons.

Concurrent edits to existing test parametrizations and removal of the notebook
example test were present during the follow-up. They were preserved and are not
part of this audit's changes; the 882 count refers to the saved initial
collection, not a claim about the current collected total. Benchmarks verify
normalization weights, finite matrices and unchanged objective/gradient values
at the specified points. Existing open physics-reproduction boundaries above
remain open.
