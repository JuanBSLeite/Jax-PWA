"""Compare a pure-JAX Gauss-Legendre prototype with NumPy.

The JAX candidate uses Newton refinement of cosine root estimates and the
three-term Legendre recurrence. This is a preparation benchmark, not a fit
benchmark. First calls include JIT; subsequent calls deliberately recompute
instead of using the production rule cache. Run on an idle device, e.g.:

JAX_PLATFORMS=cuda python benchmarks/benchmark_legendre_rules.py --unroll 16
JAX_PLATFORMS=cpu python benchmarks/benchmark_legendre_rules.py
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa import enable_x64


@partial(jax.jit, static_argnames=("order", "unroll"))
def jax_legendre_rule(order: int, unroll: int = 1):
    """Compute a float64 rule in JAX; order is static and must be positive."""
    if order < 1:
        raise ValueError("order must be positive")
    nodes = jnp.cos(
        jnp.pi * (jnp.arange(order, dtype=jnp.float64) + 0.75) / (order + 0.5)
    )

    def polynomial(x):
        def recurrence(k, state):
            previous, current = state
            return current, ((2 * k + 1) * x * current - k * previous) / (k + 1)

        previous, current = jax.lax.fori_loop(
            1, order, recurrence, (jnp.ones_like(x), x), unroll=unroll,
        )
        derivative = order * (previous - x * current) / (1 - x * x)
        return current, derivative

    def newton(_, x):
        value, derivative = polynomial(x)
        return x - value / derivative

    nodes = jax.lax.fori_loop(0, 8, newton, nodes)
    nodes = 0.5 * (nodes - nodes[::-1])
    _, derivative = polynomial(nodes)
    weights = 2 / ((1 - nodes * nodes) * derivative**2)
    weights = 0.5 * (weights + weights[::-1])
    weights = 2 * weights / weights.sum()
    return nodes[::-1], weights[::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", nargs="+", type=int, default=[16, 100, 500, 1000, 2000],
    )
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.orders, args.unroll, args.repeats) < 1:
        parser.error("orders, unroll and repeats must be positive")
    enable_x64()
    for order in args.orders:
        started = perf_counter()
        expected_nodes, expected_weights = np.polynomial.legendre.leggauss(order)
        numpy_seconds = perf_counter() - started
        times = []
        for _ in range(args.repeats + 1):
            started = perf_counter()
            result = jax.block_until_ready(jax_legendre_rule(order, args.unroll))
            times.append(perf_counter() - started)
        nodes, weights = (np.asarray(array) for array in result)
        moment_error = max(
            abs(weights @ nodes**k - (0.0 if k % 2 else 2.0 / (k + 1)))
            for k in range(min(40, 2 * order))
        )
        print(json.dumps({
            "order": order, "unroll": args.unroll,
            "device": str(jax.devices()[0]), "numpy_seconds": numpy_seconds,
            "jax_first_seconds": times[0], "jax_warm_seconds": times[1:],
            "max_node_difference": float(np.max(abs(nodes - expected_nodes))),
            "max_weight_difference": float(np.max(abs(weights - expected_weights))),
            "max_moment_error": float(moment_error),
        }), flush=True)


if __name__ == "__main__":
    main()
