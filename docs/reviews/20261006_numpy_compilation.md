# NumPy use and compilation review

Date: 2026-10-06. Same environment and numerical precision as the
[package audit](20261006_package_audit.md). Timings are local GPU diagnostics,
not guarantees across models or hardware. Raw measurements are in
`20261006_package_audit_results.json`.

## Inventory and migration assessment

An AST inventory of `src/jaxpwa/**/*.py` found 24 modules importing NumPy.
The table accounts for all of them; paths below are relative to `src/jaxpwa`.
NumPy imports alone do not indicate host work in the repeated JAX likelihood.

| Modules | NumPy role | Assessment |
|---|---|---|
| `_quadrature.py` | Gauss-Legendre nodes and weights | Keep the measured, bounded host cache, as requested. Pure-JAX prototype is available; cold cost depends on order/device. |
| `integration/gauss_legendre.py`, `integration/adaptive_gauss_legendre.py`, `integration/adaptive_square_dalitz.py`, `kinematics/square_dalitz.py` | Grid setup, interval segmentation and static geometry | Keep host setup. Repeated root construction is already eliminated. Adaptive variable-length grids would need padding or host compaction to be fully JIT-compatible. |
| `resolution/convolution.py` | Quadrature preparation, including Gauss-Hermite | Keep setup on host; Gauss-Legendre now shares the rule cache. Measure repeated Hermite construction before adding another cache. |
| `discriminants.py` | Static Chebyshev polynomial transformation | Fitted polynomial evaluation is already JAX. Putting a constant transform into the traced graph offers no demonstrated benefit. |
| `four_body.py` | Square root of the static exchange count | No meaningful runtime migration target. |
| `kinematics/nbody.py`, `likelihood/time_dependent.py` | Input validation | Keep validation; large-array checks could reduce on device and transfer scalar flags if setup profiling identifies a bottleneck. |
| `fit/minimizer.py` | Minuit interface, parameter vectors and result/Hessian handling | Host boundary is needed by iminuit. Small vectors and matrices are suitable for NumPy; objective and differentiation remain JAX. |
| `fit/nesterov.py` | Projected optimizer loop and result construction | The complete iteration/backtracking loop is now device-resident. NumPy remains only for small input/output vectors. |
| `likelihood/weighted.py` | Post-fit covariance assembly/linear algebra | Keep small parameter-space matrices on CPU unless large-parameter benchmarks justify migration. Event-side derivatives remain JAX. |
| `io/root.py`, `plotting.py` | ROOT/Matplotlib boundaries and histograms | Keep the host interface. A GPU histogram path could help repeated large projections but needs an end-to-end measurement. |
| `workflow.py`, `cp_workflow.py`, `time_dependent_workflow.py`, `projection_toys.py` | Projection histograms, covariance propagation and mixture bookkeeping | Mostly post-fit or orchestration work. Consolidating transfers may help large plots; no evidence that a blanket rewrite improves fitting. |
| `goodness_of_fit.py` | Histograms and SciPy spatial trees | Keep the spatial tree; replacing it with dense pair distances changes memory scaling. The PPD kernel is already JAX. Fixed a separate single-neighbour shape bug. |
| `resolution/scf.py` | Dense-to-sparse construction | Prefer constructing `SparseMigration` directly when possible. Extracting a variable number of nonzeros is setup, not repeated PDF evaluation. |
| `toy_inverse.py` | Small category counts and seeds | Event-level inverse transforms already run in JAX. Little expected benefit from moving seed bookkeeping. |
| `toy_accept.py` | Local-envelope reductions, accepted-index compaction and seeds | Tested end to end; device compaction slowed the 100,000-event benchmark and was reverted. |
| `time_dependent_toy.py` | Event-sized proposals and weighted resampling | Device resampling was much slower for the large proposal CDF and was reverted. A separate tag-probability bias found during the experiment was fixed. |

No direct NumPy import was found in `dynamics/lineshape/`. Static quadrature
construction may be invoked while a PDF is traced, but cached NumPy rules do
not recompute or transfer event arrays on each likelihood evaluation.

### Envelope experiment: a faster kernel is not necessarily faster generation

`benchmark_envelope_reduction.py` compares the existing host update against
`zeros.at[cells].max(scores)` followed by the same envelope maximum. It includes
device synchronization and return of the 400-cell result to the host; fresh
input arrays avoid hiding device-to-host transfers. Outputs agreed exactly for
all benchmark samples.

| Proposal points | NumPy, mean after first call | JAX, mean after first call | First JAX call |
|---|---:|---:|---:|
| 100,000 | 0.900 ms | 0.311 ms | 40.8 ms |
| 1,000,000 | 11.95 ms | 0.844 ms | 36.5 ms |

This update is called only when an envelope is exceeded and generation
restarts. Many calls generate successfully without using it at all. The new
compilation can outweigh the saved transfer on one or two restarts, so the
candidate remains in the benchmark, not production. Moving pilot reduction,
proposal generation and accepted-event compaction together deserves a complete
toy-generation benchmark before adoption. Current physics/RNG behaviour is
unchanged.

An end-to-end follow-up confirmed why the isolated result is insufficient:
moving envelope setup and accepted-index compaction to JAX increased warm
100,000-event accept-reject generation from roughly 1.08 s to 1.46 s on the
local GPU. Those production changes were reverted.

## Device-resident Nesterov: applied

The projected Nesterov loop now keeps its iterations, gradients, projections,
two-attempt restart and backtracking on the JAX device. It transfers the final
endpoint and history once. Origin, scales, bounds, fixed values and `gtol` are
runtime inputs, so repeated fits of the same live objective reuse the compiled
solver when their parameter layout and `max_iter` agree.

For the benchmark's two-parameter Rosenbrock objective, the endpoint, status,
530 objective evaluations and NLL agreed with the former implementation.
Repeated runs fell from about 201 ms to 27 ms (7.4x). Cold time rose from about
281 ms to 418 ms. Tests cover bounds, changed runtime scales/fixed values,
monotone history, failed line searches, stalling, compiled reuse and release of
dead objectives.

## Time-dependent toy: migration rejected, probability fix retained

Moving proposal generation and weighted CDF sampling to JAX preserved the
tested distributions but performed poorly at the production default of 20
proposal points per output event. For 100,000 output events, both
`jnp.searchsorted` and a vectorized binary search over the two-million-point CDF
took tens of seconds, versus about 0.8 s warm for the existing NumPy resampling
path. The migration was reverted.

The experiment exposed an independent correctness issue. Tags are already
drawn with probability `pi_q`, while the target is `pi_q * p(z,t|q)`; therefore
`pi_q` cancels from the importance ratio. The old code multiplied it into the
weights again. With `production_fraction=0.75` and zero wrong-tag rate, the old
benchmark returned about 90% positive tags instead of 75%. The final corrected
NumPy run returned 75.4%. Distribution tests now cover finite/infinite time ranges,
lifetime mean, true and observed tag fractions, wrong-tag rate, momenta and
fixed-seed reproducibility.

## Compilation: applied reuse fix

`DecayModel` retained a data-preparation JIT, but initial compact preparation
created a second equivalent data JIT inside its wrapper. The next dataset
therefore compiled again despite having the same shape and model.

Initial compact preparation now uses the model-owned data kernel as well.
Efficiency-specific normalization wrappers share it. In the five-component,
100,000-event benchmark, preparing the second dataset fell from **3.199 s to
0.0281 s**. The third took 0.0290 s. The NLL and displayed gradient norm were
unchanged compared with the immediately preceding run. Initial preparation
still cost 7.15 s, and warm value/gradient evaluation averaged 1.59 ms; those
are separate costs and were not improved by this reuse change.

Tests cover both construction orders and efficiency-wrapper identity, followed
by prepared-value checks. The relevant GPU cache/model/workflow suite passed
96 tests. This addresses duplicate compilation in the coefficient-only
three-body preparation path, not every possible model or floating-dynamics
setup.

## Persistent compilation cache: measured across fresh processes

The JAX disk cache was enabled only in an isolated temporary directory for this
experiment. No user's global configuration was changed. Two fresh processes ran:

```bash
JAX_COMPILATION_CACHE_DIR=/path/to/private/cache python benchmarks/benchmark_cache_stages.py --events 100000 --normalization-resolution 1000 --repeats 3
```

| Stage | Empty cache: compile | Populated cache: load/compile |
|---|---:|---:|
| Normalization chunk | 2.682 s | 0.0278 s |
| Data components | 2.370 s | 0.0311 s |

Tracing/lowering still took about 54–90 ms per stage in these runs. Cache loads
also do not eliminate grid construction, data transfer or execution. The first
process still compiles; subsequent compatible processes benefit. JAX's default
minimum compile-time threshold of one second was retained.

For repeated scripts/notebook sessions, configure a persistent directory before
the first compilation, for example:

```python
import os
import jax

jax.config.update(
    "jax_compilation_cache_dir", os.path.expanduser("~/.cache/jaxpwa")
)
```

See JAX's [persistent-cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
for compatibility, thresholds and cache administration.

## Further ways to avoid compilation cost

- Reuse the same `DecayModel` for repeated datasets and fits; keep data shapes
  and dtypes stable where practical. The existing normalization chunking already
  keeps most integration blocks at a stable shape.
- Use `JAX_LOG_COMPILES=1 JAX_EXPLAIN_CACHE_MISSES=1` temporarily to locate
  unexpected retracing. JAX's [compilation guide](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
  explains function identity and changing signatures. Avoid recreating jitted
  closures inside loops.
- Four-body data preparation and floating-dynamics block preparation still
  build some kernels per prepared cache. Sharing those kernels needs correct
  model/parameter ownership keys; the three-body fix should not be copied
  blindly across these paths.
- The Nesterov solver is shared per live objective, parameter layout,
  `max_iter` and verbosity cadence. Different iteration budgets still compile
  separate programs.
- Do not trade float64 precision or normalization resolution for shorter
  compilation. Neither was reduced in these changes. Likewise, aggressive loop
  unrolling can increase cold compilation even when it helps execution, as the
  Legendre experiment demonstrates.

The verified gains remove repeated preparation or compilation. Reducing the
first-ever compilation of a new physics model requires profiling that model's
graph; this review does not establish a universal compiler-speed improvement.
