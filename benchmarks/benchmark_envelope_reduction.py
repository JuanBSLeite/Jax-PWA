"""Compare the former host envelope update with the production JAX reduction.

This isolates a rare accept-reject restart operation, not complete generation.
The first repetition includes JIT; subsequent repetitions reuse that executable.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa.toy_accept import _update_local_envelopes


def _host_update(envelopes, cells, scores):
    cells = np.asarray(jax.device_get(cells), dtype=np.int32)
    scores = np.asarray(jax.device_get(scores), dtype=float)
    maxima = np.zeros_like(envelopes)
    np.maximum.at(maxima, cells, scores)
    return np.maximum(envelopes, 1.2 * maxima)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.points):
        parser.error("points and repeats must be positive")
    envelopes = np.ones(400)
    for n in args.points:
        old_times, new_times = [], []
        for i in range(args.repeats):
            # Fresh arrays avoid reusing a cached device-to-host copy.
            key = jax.random.key(i)
            cells = jax.random.randint(key, (n,), 0, 400)
            scores = jax.random.uniform(
                jax.random.fold_in(key, 1), (n,), dtype=jnp.float64,
            ) * 2
            jax.block_until_ready((cells, scores))
            start = perf_counter()
            old = _host_update(envelopes, cells, scores)
            old_times.append(perf_counter() - start)
            start = perf_counter()
            new = np.asarray(jax.device_get(
                _update_local_envelopes(
                    jnp.asarray(envelopes), cells, scores, envelope_safety=1.2,
                )
            ))
            new_times.append(perf_counter() - start)
            np.testing.assert_array_equal(old, new)
        print(json.dumps({
            "points": n, "numpy_seconds": old_times, "jax_seconds": new_times,
            "device": str(jax.devices()[0]), "x64": jax.config.jax_enable_x64,
        }), flush=True)


if __name__ == "__main__":
    main()
