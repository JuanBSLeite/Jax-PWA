# Dynamics microbatching and bounded-memory Hessian review — 2026-09-19

> **Partly superseded (2026-10-02).** `dynamics_microbatch_size`,
> `dynamics_microbatch_parallelism` and `hessian_batch_size` were removed. Only
> `normalization_chunk_size` survives, now also accepting `"auto"` (memory-aware
> sizing, see [performance.md](../performance.md)). This write-up is kept as the
> historical record of the removed code; its microbatch tests and
> `benchmark_dynamics_chunking_sweep.py` no longer exist.

Adversarial review of commit `a77f130` ("improve chucks for free dynamics"), which added the
`dynamics_microbatch_size` AD microbatching path in `PreparedAmplitudeCache` (see "AD
microbatching for floating-dynamics normalization on constrained GPUs" and "Bounded-memory
`hessian="jax"` for floating dynamics" in [performance.md](../performance.md)) together with its
own new tests and `benchmarks/benchmark_pull_study_memory.py`. One compatibility bug was
confirmed by a crashing benchmark. The review also found a benchmark accounting gap and a
mutable-configuration hazard in a shared callback. A reported shared-backend "bug" was instead
an invalid test assertion about the identity of a private wrapper tuple; the compiled callbacks
were already shared correctly. Commit `15ce685` supplied the compatibility and callback fixes,
and this follow-up corrects that test and the review record.

## Confirmed findings

### 1. Medium: `_matrix_from_dynamic` did not handle the chunked cache representation

`PreparedAmplitudeCache._matrix_from_dynamic` (`amplitude/cache.py`) unconditionally required
`self.normalization_components`, but `_prepare_chunked_dynamics` always sets that field to `None`
on the new chunked path (fixed-component normalization columns live in
`self.normalization_chunks[3]` instead). The public `evaluate`/`normalization`/
`normalization_matrix` wrappers each got their own `normalization_chunks is not None` early-return
branch in the same commit and so never hit this method on the chunked path, but the sibling
compatibility helper `_evaluate_components` (used by `_matrix_with_dynamic_blocks`) *was* patched
to read `self.normalization_chunks[3][column]` — an asymmetry that pointed to the omission being
an oversight, not an intentional narrowing.

Reproduction (fails on `a77f130`, before the fix):

```bash
python benchmarks/benchmark_qmi_memory_speed.py --events 2000 --normalization-resolution 150 --repeats 1
```

```
RuntimeError: Dynamic cache is missing fixed normalization components
```

This benchmark builds `matrix_only = jax.jit(lambda dynamic_norm: cache._matrix_from_dynamic(dynamic_norm))`
and calls it directly, the same pattern any other internal caller following
`_evaluate_dynamic_components` + `_matrix_from_dynamic` would use. The model's QMI component and
`normalization_resolution=150` already push the normalization sample past the default
`dynamics_microbatch_size=20_000`, so this is not an exotic configuration. The supported public
`evaluate`, `normalization`, and `normalization_matrix` methods already took their dedicated
chunked path, so this was an internal compatibility/benchmark failure rather than a wrong fit
result.

**Fix**: `_matrix_from_dynamic` now derives the fixed-component columns from
`self.normalization_chunks[3]` (reshaped and truncated to `self.normalization_weights.size`,
mirroring `_evaluate_components`) when `self.normalization_chunks is not None`, instead of only
accepting `self.normalization_components`.

### 2. Test issue: backend tuple identity is not part of cache reuse

The shared-backend branch of `_backend()` (`fit/minimizer.py`) deliberately rebuilt the outer
tuple `(free, names, fcn, grad, hessian)` so that `free` came from the calling `Minimizer` while
reusing the compiled `fcn`/`grad`/`hessian` callbacks. The new regression test asserted identity
of the entire private tuple. That assertion was stronger than the intended behavior and did not
test what matters: callback reuse for an equal batch size and separation for a different batch
size.

Reproduction (fails on `a77f130`, before the fix):

```bash
pytest tests/test_minimizer.py::test_hessian_batch_size_participates_in_shared_backend_key
```

```
assert first._backend() is same._backend()
AssertionError
```

**Fix**: keep the production behavior that always threads the calling instance's `free` tuple.
The test now asserts identity of the shared callback objects and verifies that a different
`hessian_batch_size` gets a different Hessian callback.

## Additional findings

### 3. Fixed: `_evaluate_dynamic_components`'s chunked branch skipped the `self.data is None` guard

The non-chunked branch raised `RuntimeError("Dynamic cache is missing prepared event data")` for
a cache built with `self.data is None`; the chunked branch (added in the same commit) had no
equivalent check and would instead fail deep inside a lineshape's `.function(None, pars)` call
with a far less diagnostic exception. Fixed by adding the same guard at the top of the chunked
branch.

### 4. Fixed: `hessian_batch_size` was read from a mutable attribute inside a shared closure

The Hessian closure built inside `_backend()` read `self.hessian_batch_size` directly at call
time rather than at closure-construction time. Since `fcn`/`grad`/`hessian` are shared across
`Minimizer` instances that hit the same `_SHARED_BACKENDS` entry (see finding 2), a later mutation
of `hessian_batch_size` on whichever instance happened to compile the backend would silently
change the batching behavior for every other instance still using that shared closure. Fixed by
snapshotting `self.hessian_batch_size` into a local (`configured_hessian_batch_size`) once, at
closure-construction time, and reading that local inside `hessian()` instead.

### 5. Fixed: the QMI memory benchmark undercounted retained memory on the chunked path

`benchmarks/benchmark_qmi_memory_speed.py`'s `retained` accounting read
`cache.normalization_data`/`cache.normalization_components` directly, both of which
`_prepare_chunked_dynamics` always leaves `None`; the benchmark never accounted for
`cache.normalization_chunks`, which holds the actual retained state (stacked prepared geometry,
weights, efficiency and fixed columns across every macro-chunk) on the chunked path. Since this
benchmark's whole purpose is quantifying retained cache memory, and the chunked path is exactly
the QMI/floating-dynamics configuration it exists to characterize, this was a silent, large
undercount rather than a crash. Fixed by adding `"normalization_chunks":
_array_payload_bytes(cache.normalization_chunks)` to the `retained` dict and blocking on it before
stopping the prepare-time timer.

Effect of the fix, same reproduction as finding 1
(`--events 2000 --normalization-resolution 150 --repeats 1`):

| | before fix | after fix |
|---|---|---|
| `retained_cache_bytes_total` | 2,204,080 B | 34,448,032 B |
| `normalization_chunks` category | not reported | 32,243,952 B |

### 6. Fixed in the accompanying sweep: total padding was undercounted

The proposed `benchmark_dynamics_chunking_sweep.py` originally reported only inner padding for
one full macro-chunk. It omitted outer padding of the final macro-chunk, which can dominate when
the normalization sample size is not divisible by `normalization_chunk_size`. The calculation
now uses the actual point count and reports all executed padded slots. The sweep also clears JAX
caches between configurations so cold-compilation timings and device residency from an earlier
configuration do not contaminate the next one.

## Flagged but not fixed

- **`diagonal <= 0` guard is asymmetric between the fixed and dynamic chunked paths.**
  `_prepare_chunked_dynamics`'s one-time, eager fixed-component preparation raises
  `ValueError("Component normalization requires positive diagonal integrals")` when a normalized
  component's diagonal integral is non-positive. The per-Minuit-step dynamic path
  (`_chunked_dynamic_normalization`) has no equivalent: it only avoids a dormant
  `1/sqrt(0)` derivative for *unnormalized* components, not a hard failure for a normalized one
  whose diagonal becomes non-positive during minimization. Left unfixed because
  `_chunked_dynamic_normalization` runs inside a `jax.jit`-traced, `jax.lax.scan`-based reduction;
  a Python-level `if bool(...): raise` is not valid there without `jax.experimental.checkify` or a
  `jax.debug.callback`, either of which is a larger, per-step-cost change than this review's scope.
- **The automatic-Hessian path now forks on parameter *kind* (`has_floating_dynamics`), not on
  whether chunking was actually needed**, so any `hessian="jax"` fit with even one floating
  `DYNAMICS` parameter takes the new per-column/batch `jax.jvp` path by default
  (`hessian_batch_size=1`, i.e. one sequential forward+backward pass per parameter) regardless of
  model size. The only new benchmark validating this (`benchmarks/benchmark_pull_study_memory.py`)
  is GPU-only, and every number in `performance.md`'s new sections comes from a single RTX 3050 Ti
  run; no CPU or small-model wall-clock comparison establishes this default change is free, as
  [CLAUDE.md](../../CLAUDE.md) asks for changes to the fit hot path. Left open as a design question
  rather than fixed, since resolving it (e.g. gating on an actual chunking decision, or measuring a
  CPU baseline) is a judgment call outside a bug-fix pass.

## Applied corrections and validation

`pytest tests/test_minimizer.py`: the original implementation failed only the invalid outer-tuple
identity assertion. The corrected test checks callback sharing directly.

`python benchmarks/benchmark_qmi_memory_speed.py --events 2000 --normalization-resolution 150 --repeats 1`:
crashed with `RuntimeError` before the fix; produces a valid JSON payload after.

Full combined run after the follow-up corrections —
`pytest tests/test_minimizer.py tests/test_dynamic_normalization_chunks.py tests/test_decay_model.py tests/test_model_io.py tests/test_root_io.py`:
**109 passed in 396.43 s**.

The corrected sweep also completed a two-point GPU smoke test with identical NLL and gradient
norms. It reported 24.2% total padding for macro/micro sizes 5,000/2,000 and 3.5% for
5,000/5,000, matching the executed padded point counts.
