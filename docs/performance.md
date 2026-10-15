# Fit performance and GPU execution

Jax-PWA separates expensive one-time preparation from repeated likelihood evaluation. This distinction is especially important on GPUs, where recomputing resonance dynamics on a large normalization grid can dominate the fit even when the final coefficient algebra is small.

## Reusing quadrature rules

Gauss-Legendre users share a bounded host cache of the one-dimensional NumPy
nodes and weights on `[-1, 1]`, keyed by order (up to 32 rules). This avoids
repeating the same root calculation across adaptive mass segments, axes,
models, discriminant PDFs and convolution setup. Cached arrays are read-only;
interval scaling creates fresh arrays. Full grids and device buffers are not
retained by this cache, and neither the nodes nor the integration measure changes.

Run `benchmarks/benchmark_quadrature_preparation.py` with and without
`--uncached` in fresh processes to compare first-use and repeated construction.
The 2026-10-06 local GPU check reduced repeated construction of a million-point
Square-Dalitz grid from about 117 ms to 15 ms, and of a 1,480,776-point adaptive
mass grid from 585 ms to 107 ms. These are preparation timings, not faster
likelihood evaluations; see [the review](reviews/20261006_package_audit.md)
for scope and validation.

## Avoiding repeated compilation

Compact preparation and later data-only preparation on the same `DecayModel`
share one data-side JIT function, including across efficiency-specific
normalization wrappers. In the local five-component, 100,000-event benchmark,
this reduced second-dataset preparation from 3.20 s to 28 ms by eliminating a
duplicate compilation. Reuse the model and compatible array shapes/dtypes to
benefit; the first preparation still compiles.

For reuse across Python processes, JAX also supports an optional persistent
compilation cache. Set `JAX_COMPILATION_CACHE_DIR` to a private directory before
running the script, or configure `jax_compilation_cache_dir` before the first
compilation. In two fresh local processes, the normalization/data compilation
stages went from 2.68/2.37 s with an empty cache to 28/31 ms with a populated
cache. Tracing, grid construction and execution remain separate costs.

See the [NumPy and compilation review](reviews/20261006_numpy_compilation.md)
for the full inventory, reproducible benchmarks, configuration example and
remaining candidates. Persistent caching is optional and is not enabled
globally by importing Jax-PWA.

The optional Nesterov prefit also keeps its complete iteration and backtracking
loop on device. Repeated runs of the same live objective reuse the solver while
starts, scales, bounds and fixed values remain runtime inputs. A small local
Rosenbrock benchmark improved from about 201 ms to 27 ms after compilation;
the cold call became slower, so this mainly benefits repeated or expensive
prefits. See [fitting](fitting.md) for convergence semantics.

## Prepared single-sample fits

`FitSession` prepares a `PreparedAmplitudeCache` before repeated likelihood calls. For fixed resonance dynamics, the cache stores the component values on the data and the normalization matrix

\[
M_{ij}=\int d\Phi\,\epsilon(\Phi) F_i^*(\Phi)F_j(\Phi).
\]

A coefficient-only likelihood evaluation then requires only

\[
A_n=\sum_i c_i F_i(x_n),\qquad
\mathcal N=c^\dagger M c,
\]

plus the event-wise log-likelihood reduction. The resonance functions are not reevaluated on the normalization grid at every Minuit step.

Efficiency and veto values on the data and normalization sample are also evaluated once by `FitSession` and reused.

## Compact coefficient-only cache

If no dynamical parameter is floating, `PreparedAmplitudeCache` uses a dedicated compact preparation path. Fixed component evaluations, component normalization and construction of the normalization matrix are compiled with JAX, while the large normalization sample is processed in fixed-size chunks.

For these fits the chunk size is 100,000 points (`normalization_chunk_size="auto"` resolves to it; an integer overrides it). Their chunks only amortize XLA compilation: for a one-million-point Square-Dalitz grid, ten chunks reuse the same XLA executable instead of compiling one very large graph specialized to one million points. There is no automatic differentiation through the grid here, so memory is not the constraint.

The matrix is accumulated as sums over chunks,

\[
M_{ij}=\frac{1}{N}\sum_k\sum_{n\in k}
 w_n F_i^*(x_n)F_j(x_n),
\]

so chunking changes only the execution schedule, not the quadrature convention. A partial final chunk is padded with a valid physical event and zero integration weights, so padded entries do not contribute.

When no efficiency map is present, the bare component matrix is needed to normalize individual components. After obtaining scales

\[
s_i = 1/\sqrt{M_{ii}},
\]

the normalized matrix is obtained algebraically as

\[
M'_{ij}=s_i M_{ij}s_j.
\]

The code therefore does not perform a second full normalization-grid reduction after scaling. With an efficiency map, the efficiency-weighted matrix is accumulated explicitly because in general it cannot be derived from the bare matrix by per-component scaling alone.

After preparation, the compact cache releases the large prepared normalization-event mapping and the per-point normalization component array. Only the data component matrix, the small fixed normalization matrix and the per-component scales remain resident for coefficient-only minimization.

The fixed model normalization is also retained by `DecayModel`. Subsequent `FitSession` objects using the same model but different datasets reuse the same normalization matrix and component scales and only evaluate the data-side amplitudes. This is particularly useful for toy, bootstrap and repeated-fit campaigns.

If a mass, width, radius, lineshape parameter, or other `ParameterKind.DYNAMICS` quantity floats, the relevant prepared data are retained. Only components owned by floating dynamical parameters are reevaluated.

For multiple floating dynamical components, all affected normalization-matrix rows are updated in one batched accelerator reduction rather than one full normalization-grid reduction per component.

### Hermitian dynamic normalization blocks

For every amplitude type, real integration/acceptance weights imply
`M_ji = conj(M_ij)`. Both the chunked and unchunked dynamic-cache paths
integrate only the diagonal and upper triangle of the dynamic-dynamic block,
then fill the lower triangle by conjugation. With `d` floating components,
this evaluates `d * (d + 1) / 2` unique component-pair integrals instead of
`d**2`. The dynamic-fixed rectangular block is also integrated once and
mirrored; the fixed-fixed block remains cached. This applies equally to
floating resonance masses/widths, QMI nodes, and direct Dalitz amplitudes.

Each matrix entry retains its original integral, with no factor of two.
Restoring the opposite triangle means the full quadratic form already gives

\[
c^\dagger M c = \sum_i |c_i|^2 M_{ii}
+ 2\operatorname{Re}\sum_{i<j} c_i^* M_{ij} c_j.
\]

The factor of two belongs only to an explicit upper-triangle sum of
interference terms. Diagonal integrals are counted once. Multiplying stored
off-diagonal entries by two would double the interference again when
evaluating the full quadratic form.

The triangular reduction uses a real diagonal reduction and one contraction
per upper row, without building an event-by-component-pair tensor. Its custom
JVP implements the exact identity
`dM = F^H W dF + (F^H W dF)^H + F^H dW F`.
The cross product `F^H W dF` is generally not Hermitian, so that derivative
uses a full matrix product. This preserves efficient JAX gradients and
forward-over-reverse Hessians while avoiding duplicated primal integrals.
Global component scales are applied after summing all chunks, with the
original `mean(weights * f)` convention and sample-size denominator.

Fewer pair integrals do not guarantee a shorter fit: dense matrix products
can be very efficient on the target device. A benchmark that replaced only the
dynamic matrix reduction by a dense reference, before JAX compilation, without
changing sample points, weights, free parameters or component-normalization
conventions, gave the following (measured with the former chunking-sweep
benchmark, which was removed together with the chunking options).

On the local CPU on 2026-10-01, with 40 repetitions
used 412,000 adaptive Square-Dalitz points and two floating resonance
components. The dense reference averaged 36.35 ms per NLL/gradient evaluation;
the Hermitian reduction averaged 38.54 ms (about 6% slower). First-use
NLL/gradient compilation plus execution took 1.98 s and 2.01 s respectively.
Both returned NLL `791730.5476162778` and gradient norm `281289.1308668142`
within floating-point rounding, and retained 67,232,256 cache bytes.
This checks evaluation cost and equivalence at phase-space starting points,
not fit convergence or a GPU speedup.

For floating dynamics, `normalization_chunk_size` also sets the size of the
prepared normalization blocks. Each block is accumulated with `jax.lax.scan` and
checkpointed, so gradients and Hessian-vector products retain the forward
residuals of one block, not of the whole grid. Every block is prepared on its
own, which is what QMI needs because its cached sort indices are valid only for
the block they were computed on. The effective block width is balanced below
the requested size to minimize tail padding. The grid resolution still controls
quadrature accuracy and should not be reduced without a normalization-
convergence check; the chunk size changes only the schedule.

### `compact_prepared_data` must be defined for every floating component type

`PreparedAmplitudeCache._prepare_chunked_dynamics` retains, per normalization chunk, only the
union of what each *floating* (`ParameterKind.DYNAMICS`-owned) component's
`compact_prepared_data(data)` reports it needs (`_compact_prepared_component_data` in
`amplitude/cache.py`). This union is computed by iterating the floating components and returning
the **entire** shared prepared mapping unchanged as soon as any one of them lacks a
`compact_prepared_data` method — so a single component type in the model without one silently
disables compaction for every other floating component too, not just the one missing it.

`ResonanceAmplitude.compact_prepared_data` previously also fell back to `dict(data)` (a shallow
copy of the whole shared mapping) for any resonance with spin != 0 or a non-`CovariantAngular`
angular factor — i.e. effectively every P-wave/D-wave resonance (rho, omega, K*, f2, ...). Since a
shallow copy still holds a live reference to every value in the original mapping, this defeated
compaction for the whole model whenever such a resonance floated alongside anything else. It now
retains only the five kinematics arrays (`mass`, `pstar`, `p`, `q`, `costheta`) each of its
pairings needs plus its own namespaced prepared-lineshape entry, falling back to the full copy
only if those keys are unexpectedly absent.

`QMIPixel` (via `DalitzAmplitude`/`_ResolvedDirectDynamics` in `decay.py`) had no
`compact_prepared_data` at all, so any model combining a floating `QMIPixel` field with floating 1D
resonances lost compaction entirely even after the `ResonanceAmplitude` fix above. It now retains
only `s12`/`s13`, the sole keys `interpolated_magnitude_phase` reads; `_ResolvedDirectDynamics`
delegates to the wrapped dynamics object's own `compact_prepared_data` when present. Any future
`AmplitudeComponent`-compatible dynamics object used as a floating component needs the same method
for compaction to remain effective across the rest of the model — see
`tests/test_resonance_amplitude.py` and `tests/test_qmi_pixel.py` for the expected contract.

## Input memory in multi-toy studies

`read_root_tree` defaults to JAX arrays on the active device. Keeping a complete
multi-toy file in that form consumes VRAM throughout every fit, even if each
fit only selects one toy. Use `library="np"` to read into host RAM and apply
the toy/charge mask before `jnp.asarray`; see [ROOT input](root_io.md).
Restart an existing notebook kernel to release arrays and compiled programs
from previous runs before comparing memory use.

The notebook-23 `SqDP_FreeMasses` input has 47,560,440 entries. Its three float64
invariants and two int32 labels occupy 1,521,934,080 bytes (1.417 GiB). Keeping
these arrays on the host removes that retained device payload without changing
events, precision, quadrature, free parameters, or likelihood conventions.

For this notebook's actual model, maps and complete toy 0, run:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  python benchmarks/benchmark_pull_study_memory.py --require-gpu --hessian \
  --normalization-resolution 1000
```

This diagnostic requires the notebook's local ROOT inputs. It reads a bounded
host entry range containing toy 0, executes setup only, and reports memory after
input selection, model/map preparation, amplitude caches, value/gradient, and
optionally the automatic Hessian. It does not run the multi-toy loop or write
CSV results. `--hessian` reproduces the automatic-Hessian measurement.
`--fit-ncall N` additionally runs one fit with a call limit; a
limited run need not converge. CPU allocator statistics can be unavailable,
and JAX's live/peak allocation counters exclude some CUDA/runtime overhead.
Measure the device process as well when assessing a laptop's VRAM budget.

On the RTX 3050 Ti (4 GiB), the 2026-09-19 end-to-end check used the notebook's
95,074 accepted events, 41 free parameters, and one million normalization
points per charge. With dynamic microbatching (a since-removed option) and sequential Hessian-vector
products, the full toy-0 MIGRAD+HESSE fit completed with `hessian="jax"`,
`valid=True`, accurate covariance, EDM `9.433e-8`, and NLL
`-494077.2905459427`. The complete diagnostic, including explicit pre-fit
value/gradient and Hessian evaluations, took 219.4 s; the already-compiled fit
stage took 95.8 s. A corresponding `hessian="numerical"` run took about 1066 s
and reached the same NLL to six decimal places.

The current implementation also works with JAX's default BFC allocator. At the
notebook's 500-by-500 setting, its effective pool limit was 2.76 GiB; the
Hessian completed with a peak live JAX allocation of 438,583,552 bytes
(418.3 MiB) and a 543,162,368-byte allocator pool (518 MiB). The benchmark
finished preparation, value/gradient, and Hessian in 94.6 s. At 1000-by-1000,
the Hessian took 74.4 s and the whole diagnostic took 129.9 s. Peak live JAX
memory was 1,144,694,784 bytes (1.066 GiB), and the allocator pool reached
2,153,775,104 bytes (2.006 GiB). This check validates one complete toy, not the
full pull distribution.

## JAX and iminuit

`Minimizer` compiles its `jax.value_and_grad` evaluator lazily. The compiled backend is reused both inside one `Minimizer` and across short-lived `Minimizer` instances that wrap the same live objective and parameter layout. This is the common pattern produced by repeated `FitSession.fit()` or `CPFitSession.fit()` calls, so a second fit of the same session does not pay the same XLA compilation cost again.

The shared lookup stores only a weak reference to the objective. A completed fit session can therefore be garbage-collected normally instead of being retained by the compilation cache.

The backend key also distinguishes floating dynamics from coefficient-only
parameters, since they require different Hessian memory schedules. Reusing the
same objective and names with different parameter kinds cannot bypass the
checkpointed, sequential-HVP path.

The Minuit value and gradient callbacks also share the last evaluated parameter point, so requesting the value and gradient at the same point causes only one JAX device evaluation and one device-to-host transfer.

`Minimizer(..., hessian="jax")` (or `session.fit(hessian="jax")`) adds an
automatic Hessian for both MIGRAD and HESSE. It uses forward-over-reverse AD,
including through QMI's custom VJPs. For floating dynamics, each Hessian
column is a separate Hessian-vector product, run one at a time to minimize
peak memory. For coefficient-only full Hessians, one linearization is
reused inside a single executable.
The last Hessian is cached independently of the value/gradient point. Compiled
programs are shared across minimizers of the same live objective and fixed-
parameter layout. No Hessian program runs or compiles on the default
`hessian="numerical"` path. The JAX path supplies a diagonal callback too:
iminuit 2.32's negative-curvature recovery can call it even when a full Hessian
is supplied. This G2 callback now has a separate JAX program: it computes
`H_ii` with forward-over-reverse directional derivatives inside one bounded
`lax.map`, retaining and transferring only the diagonal. It reuses an
already-cached full Hessian at the same
point. At a new point it does not construct or cache the full matrix.
The full-Hessian callback remains available for MIGRAD and HESSE.

This removes full matrix assembly and per-column host synchronization from
G2-only requests, but still requires one directional derivative per free
parameter. It does not eliminate the cost of repeated curvature evaluations
during `NegativeG2LineSearch`, or the first full Hessian used to seed MIGRAD.
Use `python benchmarks/benchmark_hesse.py --model qmi --events 5000
--repeats 3 --g2` to compare diagonal-only evaluation with the previous
full-Hessian extraction at identical points. G2 timings distinguish the first
call from warm calls; the full-Hessian executable has already been compiled
by the preceding HESSE measurements.

A 2026-10-01 CPU check used the local `B+ -> K+K-K+` QMI CP-fit analysis
configuration (`13_b2kkk_cpvfit_qmi.ipynb`, kept outside `notebooks/`): 137 free parameters, 335,313 data events, one million shared
toy-MC normalization points per charge, efficiency/background maps. Two seed points differed only by `1e-4` and `2e-4`
in the first free parameter. G2 was evaluated before the full Hessians to
avoid full-Hessian cache hits; diagonals agreed within `rtol=1e-9,
atol=1e-7`.

| Callback | First call, including its JIT | Second point, compiled |
|---|---:|---:|
| Full Hessian followed by diagonal extraction | 32.82 s | 23.73 s |
| Diagonal-only G2 | 34.04 s | 22.99 s |

These single-run timings show only a roughly 3% warm-evaluation improvement
for this large normalization sample, and an additional separate first-use
compilation. They do not establish a shorter complete fit. Removing full
matrix assembly therefore does not by itself resolve the notebook's
multi-minute negative-curvature search.

The differentiation direction matters on GPU. Applying reverse mode again to
the QMI gradient reintroduces scatter-adds through the saved knot gathers,
including highly contended FP64 updates. The grouped first-derivative VJP alone
does not prevent this. Applying a forward-mode JVP to the **gradient** preserves
its grouped reductions. An isolated prepared linear-QMI graph (49 knots, 10,000
events) contained 16 scatter operations with reverse-over-reverse and none with
the current forward-over-reverse implementation. Columns are still evaluated
sequentially to bound intermediate memory.

Run `python benchmarks/benchmark_hesse.py --model qmi --events 5000 --repeats 3`
to compare numerical and automatic HESSE. The benchmark reports cold and warm
times separately and changes each evaluation point to bypass the host cache.
It evaluates curvature at initial model parameters on phase-space data, **not**
at a fitted minimum; covariance flags are reported and timings do not establish
fit convergence or physics precision. Large samples still require substantial
second-order work and memory, so measure on the device used for the analysis.
With `verbose=1`, full fits report elapsed time and call counts for each optimizer
stage, including first-use JIT and any internal HESSE work within MIGRAD.

### GPU Hessian validation (2026-09-12)

The corrected backend passed all 29 `tests/test_minimizer.py` tests on a GeForce
RTX 3050 Ti Laptop GPU (4 GB), using Python 3.14, JAX 0.11.1, iminuit 2.32.0,
CUDA 13 and NVIDIA driver 595.91.07. Runs required `JAX_PLATFORMS=cuda`, enabled
x64, disabled preallocation, and explicitly checked `jax.default_backend()`;
there was no CPU fallback.

The `13_b2kkk_cpvfit_qmi` model was also checked with its 49 knots, 125 free
parameters and 1,037,441 adaptive Gauss-Legendre normalization points per charge
(nominal resolution 20, with narrow-resonance refinement). On 500/450 generated
phase-space events, the Hessian took 15.77 s on its first call, including JIT, and
2.442–2.443 s on three subsequent changed-point calls. Against central differences
of the JAX gradient (`step=1e-5`), the relative Frobenius-norm difference was
`3.03e-8`. Peak live JAX allocations were 711,382,016 bytes; the allocator pool
peaked at 1,075,838,976 bytes (these exclude some CUDA/runtime overhead).

A second run used 179,188/162,598 generated events (341,786 total), keeping the
same 125 free parameters and normalization grids. The first Hessian took 22.27 s;
three subsequent changed-point evaluations took 3.572, 3.596 and 3.596 s. The
relative Frobenius-norm difference against gradient finite differences was
`1.72e-8`. Peak live JAX allocations were 975,183,104 bytes and the allocator pool
peaked at 2,149,580,800 bytes. No out-of-memory error occurred. Timings include
device synchronization and transfer of the resulting matrix to NumPy.

Both checks retained the notebook's yield-asymmetry convention and mass/width
constraints, but used generated events without measured efficiency/background
maps. They validate derivatives and GPU execution, not the fitted data result.
The previous reverse-over-reverse implementation was interrupted before its
first GPU Hessian returned; its earlier CPU-only timing is not a measurement of
the corrected implementation.

The established strategy-2 refinement is intentionally retained: refined fits still run the existing two MIGRAD passes followed by HESSE. Removing the second pass changed convergence/precision in the regression suite. It should therefore only be reconsidered as an explicit fast-fit mode after dedicated closure studies.

### Fit-fraction uncertainties

`DecayModel.fit_fraction_errors` and the session wrappers use a compiled,
sequential reverse-mode Jacobian specialized to the model's parameter layout.
Normalization arrays, efficiencies and current parameter values are runtime
inputs, so the same executable can serve later parameter points and integration
samples without retaining their arrays. The fraction cache prepares just one
event on its unused data side; the full integration sample and weight convention
are unchanged. First-use compilation still costs time.

For CP fits, the two charges' Jacobian rows are computed separately in the same
union-of-parameter-names order, then stacked before evaluating `J @ C @ J.T`.
This avoids zero-cotangent work through the opposite charge while preserving
the cross-charge covariance needed for the mean fraction's error. The generic
`delta_method_jacobian` helper remains available with its eager row loop.

GPU comparison on the RTX 3050 Ti described above used the notebook-13 model
(125 free parameters, 16 charge/component fractions, 1,037,441 normalization
points per charge), physical fractions at the model starting values, and a
synthetic positive-definite covariance:

| Measurement | Previous eager implementation | Compiled fraction kernels |
| --- | ---: | ---: |
| First full uncertainty call, including preparation and JIT | 14.83 s | 15.06 s |
| Second full uncertainty call | 4.89 s | 1.19 s |
| Jacobian portion of the second call | 3.69 s | 0.097 s |
| Peak live JAX array allocation | 1.398 GB | 0.661 GB |

These are two calls in each of two fresh GPU processes, after constructing the
normalization grids. The benefit is primarily repeated evaluation and lower
memory use, not a faster first call. The largest absolute difference in standard
errors was `1.73e-18`; the Jacobian relative Frobenius-norm difference was
`3.00e-16`. Runtime/driver memory is additional to the JAX allocation numbers.
This comparison does not replace evaluating the actual fitted covariance and
measured efficiency maps. Tests also cover efficiency-weighted fractions,
floating QMI knots, component normalization, covariance ordering and
changed values/weights/efficiencies on a reused executable.

For a reproducible single-charge comparison, run each method in a fresh process:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/benchmark_fit_fraction_errors.py --method eager --require-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/benchmark_fit_fraction_errors.py --method prepared --require-gpu
```

### Hazard: mismatched parameter lists between cache and `Minimizer`

`PreparedAmplitudeCache.prepare()` decides, once, which DYNAMICS parameters go
through the compact fixed-evaluation path versus the dynamic-recompute path,
based on the `fixed` flag of the `parameters` it was given. `Minimizer` is a
separate, decoupled class that accepts its own `parameters` sequence. `FitSession`/
`CPFitSession`/`DecayModel.prepare_cache` always thread the same `model.parameters`
into both, so this cannot drift in the documented high-level workflow. But an
advanced caller assembling `PreparedAmplitudeCache` and `Minimizer` directly (see
`docs/user_friendly_api.md`) must pass the *same* parameter list to both: handing
`Minimizer` a list that marks a DYNAMICS parameter as floating when the cache was
prepared with it fixed produces no error, and Minuit sees an exact zero gradient
along that direction instead of a small one — the value simply never reaches the
cache's compact evaluation path. Call `cache.check_parameters(parameters)` before
constructing `Minimizer` whenever the two parameter lists are not obviously the
same object.

The check compares every dynamics binding: public name, owner, backend alias,
fixed status and fixed value. Comparing only floating component owners is
insufficient: renaming or adding a parameter on an already-floating component
can also produce an ignored fit direction. Free starting values, bounds and
optimizer steps may change without rebuilding the cache.

A component with `normalize_component=False` may have a zero integral without
being rescaled; the compact cache accepts it and keeps its zero matrix row and
column. A component requesting unit-integral normalization still requires a
positive diagonal integral. A valid total PDF must of course have nonzero
normalization.

### Memory-aware normalization chunks (`normalization_chunk_size="auto"`)

`normalization_chunk_size` is a positive integer or `"auto"`, which is the
default of `DecayModel`, `FourBodyDecayModel` and `PreparedAmplitudeCache.prepare`.
It is the only memory option. Earlier versions also had
`dynamics_microbatch_size`, `dynamics_microbatch_parallelism` and
`hessian_batch_size`; they were removed (see
[the 2026-09-19 review](reviews/20260919_dynamics_chunking_and_hessian_review.md)).

The reverse-mode AD pass through floating lineshapes needs memory proportional
to the number of normalization points in one chunk. On a 4 GB GPU, one
unchunked million-point grid through several floating lineshapes
(a `B -> 3pi` CP model with 41 free parameters, 4 floating mass/width
resonances) once exhausted memory with a 1.4 GiB allocation on the first
MIGRAD gradient call. `"auto"` therefore sizes the chunk from the device:

1. **Budget.** `MEMORY_FRACTION = 0.5` of the free memory,
   `bytes_limit - bytes_in_use` from `device.memory_stats()`. The remainder is
   headroom for the allocator pool, the CUDA context, autotuning scratch space
   and what is allocated after preparation. The fraction is an internal
   constant in `amplitude/memory.py`, not a public option.
2. **Cost per point, measured.** The forward-over-reverse Hessian-vector product
   of the normalization is compiled, not run, on the first 20,000 points, and
   `compiled.memory_analysis().temp_size_in_bytes / points` gives bytes per
   point. This is the heaviest program a fit needs, so value, gradient and
   Hessian all fit. The scaling is linear (5,000 to 40,000 points: about 533 B
   per point for the gradient and 1,156 B for the Hessian-vector product, a
   stable ratio of 2.17), so one probe suffices.
3. **Resident blocks.** The prepared blocks stay on the device for the whole fit
   whatever the chunk size; their measured size is subtracted from the budget.
4. **Chunk.** `(budget - resident) / bytes_per_point`, between 2,048 points and
   the sample size, then balanced to minimize padding. If the whole sample fits
   there is a single block and no scan.

Without device memory statistics (the CPU backend, or
`XLA_PYTHON_CLIENT_ALLOCATOR=platform`) the previous fixed 100,000 is used and no
probe is compiled; samples of at most 20,000 points are one block. The result
is memoized per model, so a toy loop compiles the probe once. Low-level callers
of `PreparedAmplitudeCache.prepare` can pass `chunk_size_memo` to share it. An
`INFO Jax-PWA normalization: automatic chunk size ...` line reports the choice.

Measured on the RTX 3050 Ti (4 GiB) with a two-floating-component model
(Gounaris-Sakurai rho plus a Hermite QMI S-wave) and a one-million-point grid,
for one value+gradient plus one Hessian-vector product:

| chunk | peak live JAX | process VRAM | first call | warm value+grad |
|---|---:|---:|---:|---:|
| whole grid (what `"auto"` chose, 1185 B/point) | 852 MiB | 2926 MiB | 37.4 s | 49 ms |
| 100,000 | 271 MiB | 1024 MiB | 11.0 s | 74 ms |

NLL, gradient norm and Hessian-vector-product norm were identical to six
decimals. A smaller chunk lowers memory and the first-call compilation time and
costs about 50% in warm time, because of the scan. With
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`, `"auto"` chose 379,553 points (three
blocks of 333,334); with `0.15`, 129,145 points (eight blocks of 125,000), with
unchanged results. If cold-start time matters more than warm time, pass an
integer such as `100_000`.

Limits of the estimate. It covers the normalization side only: the data-side
amplitudes of floating components are evaluated on all events at once and are not
bounded by the chunk size. The budget counts the program's temporary buffers and
the resident blocks, not what other programs (the Hessian-vector product of the
full NLL, data amplitudes, other processes sharing the GPU) allocate later; the
`0.5` margin is a heuristic validated on the model above, not on the
137-parameter analysis fits. `peak_bytes_in_use` also includes the preparation
phase, so it is not a clean measure of the AD pass.

### Bounded-memory `hessian="jax"` for floating dynamics

The automatic Hessian originally put all columns into one XLA executable:
`jax.linearize(jax.grad(vector_objective), point)` produced one pushforward,
then `jax.lax.map` evaluated the basis vectors. Checkpointing the gradient
reduced its program estimate substantially, but the real notebook model still
required about 3.30 GiB at 500-by-500 resolution. That exceeded the default
BFC allocator's 2.76 GiB pool on the 4 GiB RTX 3050 Ti.

For floating `DYNAMICS` parameters, `Minimizer` checkpoints the gradient
and compiles a single forward-over-reverse Hessian-vector product:

```python
gradient = jax.checkpoint(jax.grad(vector_objective))
hvp = jax.jit(lambda point, tangent: jax.jvp(
    gradient, (point,), (tangent,),
)[1])
```

The host invokes this reusable program
once per basis vector and transfers each column before starting the next.
Temporary buffers from different columns therefore cannot overlap, and XLA
compiles one HVP rather than a program that contains the entire Hessian loop.
The differentiation remains
forward-over-reverse, preserving QMI's grouped custom-VJP reductions. The
result is symmetrized once after the columns are assembled, as before.

Coefficient-only fits keep the previous single-linearization program. They do
not need the extra memory boundary, and a small eight-parameter GPU benchmark
showed that applying the checkpoint there increased the warm Hessian time from
about 18 ms to 22 ms.

On the real `B -> 3pi` model at 500-by-500 resolution, the sequential-HVP
implementation completed with the default allocator in 46.2 s for the Hessian.
Peak live JAX memory was 418.3 MiB and the allocator pool reached 518 MiB,
compared with the previous 3.30 GiB program requirement. The computed Hessian
norm was identical to the platform-allocator run (`4892002.165911924`).

At 1000-by-1000 resolution, also with the default allocator, the Hessian took
74.4 s. Peak live JAX memory was 1.066 GiB and the pool reached 2.006 GiB,
below its 2.76 GiB limit. A contemporaneous `nvidia-smi` sample showed
3,524 MiB including CUDA/runtime memory outside JAX's counters.

The end-to-end default-allocator check at one million normalization points per
charge completed MIGRAD+HESSE with `valid=True`, accurate covariance, EDM
`9.433e-8`, and NLL matching the numerical-HESSE run to six decimal places.
MIGRAD used 39 function calls; the numerical version took about 1066 s and
needed 2411. A live notebook should still avoid retaining unrelated device
arrays, but the platform allocator is no longer part of the required
configuration.

## QMI preparation

For the local one-dimensional QMI modes (`linear`, `cubic`, `hermite`), `prepare_mass` caches the
fixed knot interval index and interpolation fraction (plus the event order and per-interval
`starts`/`ends` used by the reverse pass) once; each mode's forward evaluation and its
hand-written `jax.custom_vjp` then cost one gather and one grouped-interval-sum reduction per
event, with no global linear solve and no event-sized reverse scatter-add. `natural` (the global
natural cubic spline) is different: it is *not* on this fixed/cached path. Every evaluation
solves its knot-sized tridiagonal system from scratch with `jnp.linalg.solve`, through ordinary
JAX autodiff rather than a custom VJP, so changing magnitudes or phases re-solves the system;
benchmark it separately from the local modes for large fits (see `docs/lineshapes.md`).

For `QMIPixel`, fixed interpolation geometry is also cached: bin edges, bin centres, active masks and the nearest-active gather map used to fill ghost cells. Floating magnitudes/phases therefore update only the value field and interpolation algebra, not the geometry construction.

## K-matrix preparation

The five-channel K-matrix has a particularly expensive fixed operation,

\[
D(s)=\left[I-iK(s)\rho(s)\right]^{-1}.
\]

For the pi-pi production amplitude only the first row of `D(s)` is needed. During prepared resonance evaluation, Jax-PWA now stores that row once for each event/normalization point. A later change of production coefficients therefore evaluates

\[
F_{\pi\pi}(s)=D_{0j}(s)P_j(s)
\]

as a five-term complex contraction rather than repeating a 5x5 linear solve for every point at every likelihood evaluation.

Only five complex values per point are retained, rather than the full 5x5 inverse matrix. This keeps the GPU memory cost substantially lower while removing the dominant repeated K-matrix linear algebra. The ordinary standalone `KMatrix.amplitude_vector()` API still performs the full solve because it returns all five output channels.

## Sparse SCF migration

`SquareDalitzSCFMap` caches its fixed Square-Dalitz bin centres, invariant coordinates and phase-space areas. It also supports sparse migration storage.

A dense migration matrix has memory complexity

\[
O(N_{\rm bin}^2),
\]

which becomes costly very quickly. A `100 x 100` Square-Dalitz map has 10,000 bins and therefore 100 million migration elements, or roughly 800 MB in float64.

For local SCF migration, most of those elements are zero. `SparseMigration` stores only

```text
true bin index
reconstructed bin index
probability
```

for non-zero transitions, so memory becomes `O(nnz)`. Migration is evaluated with JAX gather/scatter operations and remains JIT-compatible and differentiable.

Dense input remains supported. `SquareDalitzSCFMap(storage="auto")`, the default, compresses a dense matrix when its non-zero fraction is at most 25%. For very large maps, construct `SparseMigration` directly so the dense matrix never exists.

A dedicated benchmark compares dense and sparse execution on the active JAX device:

```bash
python benchmarks/benchmark_scf_migration.py \
  --bins-mprime 40 \
  --bins-thetaprime 40 \
  --neighbors 9 \
  --repeats 50
```

The output reports storage reduction, first JIT call, steady-state execution time and the sparse/dense speed ratio. The best representation can be device- and sparsity-dependent; sparse storage is primarily essential for controlling memory at fine binning.

## Benchmarking cache compilation stages

Use the dedicated cache-stage benchmark to separate XLA lowering, compilation and execution:

```bash
python benchmarks/benchmark_cache_stages.py \
  --events 100000 \
  --normalization-resolution 1000 \
  --repeats 5
```

For the realistic five-component B+ -> K+ pi+ pi- model, one CUDA benchmark with one million normalization points showed the original full-grid normalization graph spending about 15.2 s in XLA compilation while the actual one-million-point execution took only about 0.25 s. This identified compilation, not arithmetic throughput, as the dominant cold-start cost.

With 100,000-point normalization chunks, the same device measured approximately:

- 3.18 s to compile the normalization chunk kernel;
- 0.034 s for the first 100,000-point chunk execution;
- 0.029 s per warm chunk;
- about 0.43 s for a warm traversal of all ten chunks;
- 1.30 s to compile the separate 100,000-event data kernel;
- about 0.026 s for a warm 100,000-event data evaluation.

These numbers are hardware- and load-dependent; they are provided as a representative diagnostic, not a guaranteed performance target.

## Benchmarking the full fit on the target GPU

Use the realistic five-component B+ -> K+ pi+ pi- benchmark:

```bash
python benchmarks/benchmark_fit_evaluation.py \
  --events 100000 \
  --normalization-resolution 1000 \
  --repeats 20
```

The JSON output reports:

- JAX backend and device;
- whether x64 is enabled;
- number of data and normalization points;
- normalization-grid construction time;
- phase-space data generation time;
- `prepared_cache_seconds`, including first compact-cache compilation and execution;
- `prepared_cache_reuse_seconds`, the cost of accessing the already prepared session cache;
- preparation times for second and third datasets sharing the same `DecayModel` normalization;
- first value+gradient time, which includes likelihood JIT compilation;
- steady-state value+gradient timing after compilation;
- whether the amplitude cache is compact;
- minimum and maximum normalization-matrix diagonal values as a quick component-normalization sanity check.

On the same representative CUDA setup with 100,000 data events and one million normalization points, the coefficient-only cache preparation improved from an initial baseline of about 28.95 s to about 7.24 s after compact-cache fusion, elimination of the redundant matrix reduction, model-level normalization reuse and chunked normalization compilation. The steady-state value+gradient time remained about 4.8 ms, and the normalized matrix diagonal stayed at unity to floating-point precision.

The first compiled call should not be confused with steady-state fit throughput. GPU/XLA compilation can be significant, while subsequent iterations are much faster.

## Precision

The project uses 64-bit real and 128-bit complex arithmetic when `enable_x64()` is enabled. This is deliberate for amplitude-analysis stability. Consumer GPUs can have much lower FP64 throughput than data-centre GPUs, but changing the default to float32/complex64 should only be done after explicit likelihood, parameter, fit-fraction, and toy-closure studies.

## Time-dependent Dalitz fits

`TimeDependentDalitzNLL` uses `PreparedAmplitudeCache.coherent_groups` to evaluate
A and Abar and their cross-normalization in one dynamic pass. Mixing-only and
coefficient-only fits reuse the fixed component basis and Gram matrix. Time
normalization is analytic by default; acceptance/Gaussian response uses explicit
true-time quadrature and a JAX scan with event-sized accumulators. See
[time-dependent fits](time_dependent.md) and
`benchmarks/benchmark_time_dependent.py` for a runnable D0 example and benchmark.
