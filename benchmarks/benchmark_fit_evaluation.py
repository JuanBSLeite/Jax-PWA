"""Benchmark preparation, JIT compilation and steady-state Dalitz-fit evaluation.

This benchmark uses the realistic five-component B+ -> K+ pi+ pi- model used
throughout the example notebooks. It is intended to be run on the actual CPU
or GPU used for fits; it does not form part of the unit-test suite.

Example
-------
python benchmarks/benchmark_fit_evaluation.py --events 100000 --normalization-resolution 1000
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from jaxpwa import (
    BaBarFlatte,
    DecayChannel,
    DecayModel,
    FitSession,
    LASS,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    enable_x64,
)


enable_x64()

_BENCHMARK_VERSION = 2


def _coefficient(name: str, x: float, y: float, *, fixed: bool = False):
    return RealImag(
        Parameter.coefficient(
            f"{name}.x",
            x,
            fixed=fixed,
            owner=name,
        ),
        Parameter.coefficient(
            f"{name}.y",
            y,
            fixed=fixed,
            owner=name,
        ),
    )


def make_model(normalization_resolution: int) -> DecayModel:
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    return DecayModel(
        channel,
        [
            Resonance(
                "Kstar892",
                (0, 2),
                _coefficient("Kstar892", 1.00, 0.00, fixed=True),
                mass=0.8958,
                width=0.0474,
                spin=1,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "KpiS",
                (0, 2),
                _coefficient("KpiS", 1.40, -0.60),
                lineshape=LASS(2.07, 3.32, 1.8),
                mass=1.425,
                width=0.270,
                spin=0,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "rho770",
                (1, 2),
                _coefficient("rho770", 0.65, 0.10),
                mass=0.7753,
                width=0.1491,
                spin=1,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "f0_980",
                (1, 2),
                _coefficient("f0_980", -0.20, 1.00),
                lineshape=BaBarFlatte(),
                mass=0.965,
                width=0.0,
                spin=0,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            NonResonant(_coefficient("NR", -0.50, 0.10)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=normalization_resolution,
        normalization_pair=(0, 2),
    )


def _block_tree(value, _seen: set[int] | None = None):
    """Synchronize all device arrays reachable from a benchmark result.

    JAX only traverses explicitly registered pytrees. Several fitter objects are
    ordinary dataclasses, so relying on ``jax.tree_util.tree_leaves`` alone can
    leave their arrays executing asynchronously and under-report timings. This
    recursive walker handles dataclasses and standard containers explicitly.
    """

    if _seen is None:
        _seen = set()

    object_id = id(value)
    if object_id in _seen:
        return value
    _seen.add(object_id)

    blocker = getattr(value, "block_until_ready", None)
    if blocker is not None:
        blocker()
        return value

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _block_tree(getattr(value, field.name), _seen)
        return value

    if isinstance(value, Mapping):
        for item in value.values():
            _block_tree(item, _seen)
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            _block_tree(item, _seen)
        return value

    return value


def _seconds(function):
    start = time.perf_counter()
    result = function()
    _block_tree(result)
    return result, time.perf_counter() - start


def _parameter_point(session: FitSession) -> dict[str, float]:
    return {parameter.name: float(parameter.value) for parameter in session.parameters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--normalization-resolution", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260902)
    args = parser.parse_args()

    model = make_model(args.normalization_resolution)

    _, normalization_seconds = _seconds(lambda: model.normalization_sample)
    data, data_seconds = _seconds(
        lambda: model.generate_phase_space(args.events, seed=args.seed)
    )
    session = FitSession(model, data)

    # First session: model normalization plus first XLA compilation and data-side
    # evaluation.
    cache, cache_seconds = _seconds(lambda: session.signal_cache)
    _, cache_reuse_seconds = _seconds(lambda: session.signal_cache)

    # Second dataset: normalization and the data executable are already cached
    # on DecayModel. The same shapes/dtypes reuse the first preparation's JIT.
    second_data, second_data_seconds = _seconds(
        lambda: model.generate_phase_space(args.events, seed=args.seed + 1)
    )
    second_session = FitSession(model, second_data)
    second_cache, second_session_cache_seconds = _seconds(
        lambda: second_session.signal_cache
    )

    # Third dataset: both model normalization and the data-only executable are
    # warm. This is representative of long toy/bootstrapping campaigns.
    third_data, third_data_seconds = _seconds(
        lambda: model.generate_phase_space(args.events, seed=args.seed + 2)
    )
    third_session = FitSession(model, third_data)
    third_cache, third_session_cache_seconds = _seconds(
        lambda: third_session.signal_cache
    )

    minimizer = session.minimizer()
    free, names, fcn, grad, _ = minimizer._backend()
    point = _parameter_point(session)
    vector = [point[name] for name in names]

    start = time.perf_counter()
    first_value = fcn(*vector)
    first_gradient = grad(*vector)
    first_seconds = time.perf_counter() - start

    times = []
    values = []
    for index in range(args.repeats):
        shifted = list(vector)
        if shifted:
            shifted[index % len(shifted)] += 1e-7 * (index + 1)
        start = time.perf_counter()
        values.append(fcn(*shifted))
        grad(*shifted)
        times.append(time.perf_counter() - start)

    diagonal = jnp.real(jnp.diag(cache.normalization_matrix_fixed))
    second_diagonal = jnp.real(jnp.diag(second_cache.normalization_matrix_fixed))
    third_diagonal = jnp.real(jnp.diag(third_cache.normalization_matrix_fixed))
    payload = {
        "benchmark_version": _BENCHMARK_VERSION,
        "device": str(jax.devices()[0]),
        "platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "events": args.events,
        "normalization_resolution": args.normalization_resolution,
        "normalization_points": model.normalization_sample.size,
        "components": len(model.components),
        "free_parameters": len(free),
        "normalization_grid_seconds": normalization_seconds,
        "data_generation_seconds": data_seconds,
        "prepared_cache_seconds": cache_seconds,
        "prepared_cache_reuse_seconds": cache_reuse_seconds,
        "second_data_generation_seconds": second_data_seconds,
        "second_session_prepared_cache_seconds": second_session_cache_seconds,
        "third_data_generation_seconds": third_data_seconds,
        "third_session_prepared_cache_seconds": third_session_cache_seconds,
        "second_session_cache_speedup": (
            cache_seconds / second_session_cache_seconds
            if second_session_cache_seconds > 0.0
            else float("inf")
        ),
        "third_session_cache_speedup": (
            cache_seconds / third_session_cache_seconds
            if third_session_cache_seconds > 0.0
            else float("inf")
        ),
        "cache_is_compact": bool(cache.is_compact),
        "normalization_matrix_diagonal_min": float(jnp.min(diagonal)),
        "normalization_matrix_diagonal_max": float(jnp.max(diagonal)),
        "second_normalization_matrix_diagonal_min": float(jnp.min(second_diagonal)),
        "second_normalization_matrix_diagonal_max": float(jnp.max(second_diagonal)),
        "third_normalization_matrix_diagonal_min": float(jnp.min(third_diagonal)),
        "third_normalization_matrix_diagonal_max": float(jnp.max(third_diagonal)),
        "first_value_and_gradient_seconds": first_seconds,
        "steady_value_and_gradient_seconds_mean": float(jnp.mean(jnp.asarray(times))),
        "steady_value_and_gradient_seconds_min": float(jnp.min(jnp.asarray(times))),
        "steady_value_and_gradient_seconds_max": float(jnp.max(jnp.asarray(times))),
        "first_nll": float(first_value),
        "first_gradient_norm": float(jnp.linalg.norm(jnp.asarray(first_gradient))),
        "last_nll": float(values[-1]) if values else float(first_value),
    }
    print("FIT_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
