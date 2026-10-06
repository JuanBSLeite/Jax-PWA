"""Projected accelerated gradient prefit for smooth bounded objectives."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

# Consecutive accepted iterations without any NLL decrease before the run is
# reported as ``status="stalled"``.
STALL_ITERATIONS = 5


@dataclass(frozen=True)
class NesterovResult:
    """Minuit-compatible result returned by a Nesterov-only fit."""

    values: Mapping[str, float]
    fval: float
    valid: bool
    status: str
    converged: bool
    history: tuple[Mapping[str, float], ...]
    nfcn: int
    errors: Mapping[str, float]
    covariance: object = None
    optimizer: str = "nesterov"

    @property
    def fmin(self):
        """Minuit-``FMin``-like stand-in; ``edm`` is always ``nan`` (no EDM check)."""

        return SimpleNamespace(edm=float("nan"))


# Objectives own the lifetime of compiled solvers, as in the Minuit backend.
_SOLVERS = {}


def _build_solver(objective, names, fixed_names, max_iter, report_every):
    def solve(origin, scales, fixed_values, lower, upper, gtol):
        def scaled_nll(z):
            physical = origin + scales * z
            values = dict(zip(fixed_names, fixed_values, strict=True))
            values.update({name: physical[i] for i, name in enumerate(names)})
            return objective(values)

        evaluate = jax.value_and_grad(scaled_nll)

        def residual(x, g):
            return jnp.max(jnp.abs(x - jnp.clip(x - g, lower, upper)))

        def search(yy, fy, gy, f, step):
            def trial_body(state):
                _, _, _, trial, count, _ = state
                candidate = jnp.clip(yy - trial * gy, lower, upper)
                delta = candidate - yy
                fc, gc = evaluate(candidate)
                majorant = fy + gy @ delta + delta @ delta / (2.0 * trial)
                accepted = (
                    jnp.isfinite(fc)
                    & jnp.all(jnp.isfinite(gc))
                    & (fc <= majorant + 1e-12 * jnp.maximum(1.0, jnp.abs(fy)))
                    & (fc <= f)
                )
                return (
                    candidate,
                    fc,
                    gc,
                    jnp.where(accepted, trial, trial * 0.5),
                    count + 1,
                    accepted,
                )

            return jax.lax.while_loop(
                lambda state: (state[4] < 60) & ~state[5],
                trial_body,
                (yy, fy, gy, step * 1.1, jnp.int32(0), jnp.bool_(False)),
            )

        def iteration(state):
            x, y, f, g, t, step, nfcn, i, stalled, status, history = state
            same = jnp.all(y == x)
            fy, gy = jax.lax.cond(same, lambda: (f, g), lambda: evaluate(y))
            first = search(y, fy, gy, f, step)
            second = jax.lax.cond(
                first[5],
                lambda: first,
                lambda: search(x, f, g, f, step),
            )
            candidate, fc, gc, trial, count, accepted = second
            nfcn += (~same).astype(jnp.int32) + first[4]
            nfcn += jnp.where(first[5], 0, count)
            stalled = jnp.where(fc >= f, stalled + 1, 0)
            next_t = (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0
            next_y = jnp.clip(
                candidate + (t - 1.0) / next_t * (candidate - x),
                lower,
                upper,
            )
            x = jnp.where(accepted, candidate, x)
            f, g = jnp.where(accepted, fc, f), jnp.where(accepted, gc, g)
            i += accepted.astype(jnp.int32)
            r = residual(x, g)
            history = jax.lax.cond(
                accepted,
                lambda h: h.at[i].set(jnp.stack((f, r))),
                lambda h: h,
                history,
            )
            # 0=max_iter/running, 1=line_search_failed, 2=stalled.
            status = jnp.where(
                accepted, jnp.where(stalled >= STALL_ITERATIONS, 2, 0), 1
            ).astype(jnp.int32)
            if report_every is not None:

                def report():
                    jax.debug.callback(
                        lambda it, fv, rv: print(
                            f"[Nesterov] {int(it)}: NLL={float(fv):.9f}, "
                            f"projected gradient={float(rv):.4g}",
                            flush=True,
                        ),
                        i,
                        f,
                        r,
                    )

                jax.lax.cond(accepted & (i % report_every == 0), report, lambda: None)
            return (x, next_y, f, g, next_t, trial, nfcn, i, stalled, status, history)

        x = jnp.zeros_like(origin)
        f, g = evaluate(x)
        history = jnp.zeros((max_iter + 1, 2), dtype=origin.dtype)
        history = history.at[0].set(jnp.stack((f, residual(x, g))))
        state = (
            x,
            x,
            f,
            g,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            history,
        )
        state = jax.lax.while_loop(
            lambda s: (s[7] < max_iter) & (s[9] == 0) & ~(residual(s[0], s[3]) <= gtol),
            iteration,
            state,
        )
        x, _, f, g, _, _, nfcn, i, _, status, history = state
        status = jnp.where(residual(x, g) <= gtol, 3, status)
        return origin + scales * x, f, nfcn, i, status, history

    return jax.jit(solve)


def _solver(objective, names, fixed_names, max_iter, report_every):
    key = (id(objective), names, fixed_names, max_iter, report_every)
    cached = _SOLVERS.get(key)
    if cached is not None and cached[0]() is objective:
        return cached[1]
    try:
        ref = weakref.ref(objective, lambda _: _SOLVERS.pop(key, None))
    except TypeError:
        return _build_solver(objective, names, fixed_names, max_iter, report_every)

    def call(values):
        return ref()(values)

    solver = _build_solver(call, names, fixed_names, max_iter, report_every)
    _SOLVERS[key] = (ref, solver)
    return solver


def minimize(
    objective: Callable,
    parameters: Sequence,
    *,
    start_values: Mapping[str, float] | None = None,
    max_iter: int = 1000,
    gtol: float = 1e-4,
    verbose: int = 0,
) -> NesterovResult:
    """Run projected Nesterov with device-resident iterations and backtracking.

    NumPy is used only for small input vectors and the returned host result.
    Starts, scales, bounds and fixed values are runtime inputs to a solver
    shared while the objective is alive. The full history is transferred once.
    """
    if max_iter < 1 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("max_iter must be positive and gtol must be positive")
    supplied = {} if start_values is None else dict(start_values)
    free = tuple(p for p in parameters if not p.fixed)
    if not free:
        raise ValueError("At least one free parameter is required")
    fixed = {p.name: float(p.value) for p in parameters if p.fixed}
    origin = np.asarray([supplied.get(p.name, p.value) for p in free], dtype=float)
    scales = np.asarray(
        [
            p.step if p.step is not None else max(abs(float(origin[i])), 1.0)
            for i, p in enumerate(free)
        ],
        dtype=float,
    )
    bounds = [p.bounds or (None, None) for p in free]
    lower = np.asarray(
        [
            (-np.inf if b[0] is None else b[0] - origin[i]) / scales[i]
            for i, b in enumerate(bounds)
        ]
    )
    upper = np.asarray(
        [
            (np.inf if b[1] is None else b[1] - origin[i]) / scales[i]
            for i, b in enumerate(bounds)
        ]
    )
    report_every = 50 if verbose == 2 else 10 if verbose >= 3 else None
    solver = _solver(
        objective, tuple(p.name for p in free), tuple(fixed), max_iter, report_every
    )
    physical, f, nfcn, iterations, status_code, history_array = jax.device_get(
        solver(
            jnp.asarray(origin),
            jnp.asarray(scales),
            jnp.asarray(tuple(fixed.values())),
            jnp.asarray(lower),
            jnp.asarray(upper),
            jnp.asarray(gtol),
        )
    )
    status = ("max_iter", "line_search_failed", "stalled", "converged")[
        int(status_code)
    ]
    history = tuple(
        {"iteration": i, "nll": float(row[0]), "projected_gradient": float(row[1])}
        for i, row in enumerate(history_array[: int(iterations) + 1])
    )
    values = {p.name: float(physical[i]) for i, p in enumerate(free)}
    values.update(fixed)
    return NesterovResult(
        values=values,
        fval=float(f),
        valid=status == "converged",
        status=status,
        converged=status == "converged",
        history=history,
        nfcn=int(nfcn),
        errors={p.name: float("nan") for p in free},
    )
