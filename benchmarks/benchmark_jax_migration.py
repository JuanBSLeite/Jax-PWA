"""Compare cold/warm toy generation and Nesterov across source snapshots.

Use --baseline-dir with copies of nesterov.py, toy_accept.py and
 time_dependent_toy.py to measure the old implementations in a fresh process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa import (
    DecayChannel,
    DecayModel,
    NeutralMesonMixing,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    TimeDependentFitSession,
    time_dependent_toy,
    toy_accept,
)
from jaxpwa.fit import nesterov


def load_snapshot(directory, name, current):
    if directory is None:
        return current
    module_name = f"jaxpwa._migration_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.events < 1 or args.repeats < 1:
        parser.error("events and repeats must be positive")
    optimizer = load_snapshot(args.baseline_dir, "nesterov", nesterov)
    accept = load_snapshot(args.baseline_dir, "toy_accept", toy_accept)
    temporal = load_snapshot(
        args.baseline_dir, "time_dependent_toy", time_dependent_toy
    )
    result = {
        "device": str(jax.devices()[0]),
        "events": args.events,
        "baseline": args.baseline_dir is not None,
        "x64": jax.config.jax_enable_x64,
    }

    parameters = (Parameter("x", -1.5, step=0.1), Parameter("y", 2.0, step=0.1))

    def rosenbrock(p):
        return (1 - p["x"]) ** 2 + 100 * (p["y"] - p["x"] ** 2) ** 2

    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        fit = optimizer.minimize(rosenbrock, parameters, max_iter=300)
        times.append(perf_counter() - start)
    result["nesterov"] = {
        "seconds": times,
        "fval": fit.fval,
        "nfcn": fit.nfcn,
        "status": fit.status,
        "values": fit.values,
    }

    model = DecayModel(
        DecayChannel("D0", ("K(S)0", "pi+", "pi-")),
        [
            NonResonant(RealImag(1.0, 0.0)),
            Resonance(
                "rho", (1, 2), RealImag(0.3, 0.2), mass=0.775, width=0.149, spin=1
            ),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=30,
    )
    times = []
    for i in range(args.repeats):
        start = perf_counter()
        toy = accept.generate_signal_toy(
            model,
            args.events,
            seed=100 + i,
            pool_size=max(args.events, 4096),
            include_momenta=False,
        )
        jax.block_until_ready(toy.s12)
        times.append(perf_counter() - start)
    result["accept_reject"] = {"seconds": times, "mean_s12": float(jnp.mean(toy.s12))}

    data = model.generate_phase_space(8, seed=1, include_momenta=False)
    session = TimeDependentFitSession(
        model,
        data,
        jnp.linspace(0.05, 1, 8),
        jnp.asarray([1, -1] * 4),
        NeutralMesonMixing(0.01, 0.005, 0.4103),
        time_range=(0.0, 4.0),
    )
    times = []
    for i in range(args.repeats):
        start = perf_counter()
        toy = temporal.generate_time_dependent_toy(
            session,
            args.events,
            seed=200 + i,
            proposal_size=20 * args.events,
            production_fraction=0.75,
        )
        jax.block_until_ready((toy.times, toy.data.s12))
        times.append(perf_counter() - start)
    result["time_dependent"] = {
        "seconds": times,
        "mean_time": float(jnp.mean(toy.times)),
        "plus_fraction": float(np.mean(np.asarray(toy.tags) == 1)),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
