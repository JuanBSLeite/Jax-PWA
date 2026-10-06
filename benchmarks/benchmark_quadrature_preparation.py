"""Measure cold/repeated grid construction with or without host rule reuse.

Run both modes in fresh processes, e.g. with OPENBLAS_NUM_THREADS=1:
    python benchmarks/benchmark_quadrature_preparation.py --uncached
    python benchmarks/benchmark_quadrature_preparation.py
Timings include device synchronization, but no amplitude preparation or fitting.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax

from jaxpwa import SquareDalitzGrid
from jaxpwa._quadrature import legendre_rule
from jaxpwa.integration import AdaptiveDalitzGaussLegendreGrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncached", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.uncached:
        from jaxpwa.integration import gauss_legendre
        from jaxpwa.kinematics import square_dalitz

        gauss_legendre.legendre_rule = legendre_rule.__wrapped__
        square_dalitz.legendre_rule = legendre_rule.__wrapped__
    grids = {
        "square_1000": SquareDalitzGrid(1.86966, (0.13957,) * 3, resolution=1000),
        "adaptive": AdaptiveDalitzGaussLegendreGrid(
            1.86966, (0.13957,) * 3,
            m13_narrow_resonances=((0.78265, 0.00849),),
            m23_narrow_resonances=((0.78265, 0.00849),),
        ),
    }
    for name, grid in grids.items():
        legendre_rule.cache_clear()
        times = []
        for _ in range(args.repeats):
            start = perf_counter()
            sample = grid.sample()
            jax.block_until_ready(sample.weights)
            times.append(perf_counter() - start)
        print(json.dumps({
            "grid": name,
            "cached_rules": not args.uncached,
            "device": str(jax.devices()[0]),
            "x64": jax.config.jax_enable_x64,
            "seconds": times,
            "points": sample.size,
            "weight_mean": float(sample.weights.mean()),
        }), flush=True)


if __name__ == "__main__":
    main()
