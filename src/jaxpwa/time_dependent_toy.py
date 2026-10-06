"""Importance-sampled toys for tagged time-dependent Dalitz fits."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp

from .kinematics import PhaseSpaceSample
from .time_dependent_workflow import _acceptance


@dataclass(frozen=True)
class TimeDependentToy:
    """A generated time-dependent Dalitz sample and its observed tags."""

    data: PhaseSpaceSample
    times: jnp.ndarray
    tags: jnp.ndarray
    true_tags: jnp.ndarray


@partial(jax.jit, static_argnames=("size",))
def _time_proposal(key, size, low, high, rate, production_fraction, wrong):
    """Device-resident tags and a stable truncated exponential proposal."""
    tag_key, flip_key, time_key = jax.random.split(key, 3)
    true_tags = jnp.where(
        jax.random.uniform(tag_key, (size,)) < production_fraction, 1, -1
    )
    tags = jnp.where(
        jax.random.uniform(flip_key, (size,)) < wrong, -true_tags, true_tags
    )
    low = jnp.maximum(low, 0.0)
    cdf_range = -jnp.expm1(-rate * (high - low))
    uniform = jax.random.uniform(time_key, (size,), dtype=jnp.float64)
    times = low - jnp.log1p(-uniform * cdf_range) / rate
    density = rate * jnp.exp(-rate * (times - low)) / cdf_range
    return times, tags.astype(jnp.int32), true_tags.astype(jnp.int32), density


@partial(jax.jit, static_argnames=("size",))
def _resample_indices(key, density, proposal_time, size):
    # The proposal already draws tags with the requested observed probability.
    # That factor cancels between target and proposal; multiplying it again
    # would bias, for example, a 75% tag fraction towards 90%.
    weights = density / jnp.maximum(proposal_time, 1e-300)
    total = jnp.sum(weights)
    valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & (total > 0)
    valid &= jnp.isfinite(total)
    cdf = jnp.cumsum(weights) / total
    cdf = cdf.at[-1].set(1.0)
    uniform = jax.random.uniform(key, (size,), dtype=cdf.dtype)
    selected = jnp.searchsorted(cdf, uniform, side="right")
    return selected, valid


def generate_time_dependent_toy(
    session,
    size: int,
    *,
    parameters=None,
    production_fraction: float = 0.5,
    wrong_tag: float | None = None,
    proposal_size: int | None = None,
    seed: int | None = None,
    include_momenta: bool = False,
) -> TimeDependentToy:
    """Generate a joint ``(Dalitz, true time, observed tag)`` signal toy.

    The target is the same tag-conditional signal density used by the session,
    multiplied by the requested production/tag probability. Dalitz points are
    drawn uniformly in phase space and true times from a lifetime exponential;
    importance resampling then applies mixing, efficiency, temporal acceptance,
    and the optional wrong-tag mixture. This is a validation sampler, not an
    exact rejection sampler, so increase ``proposal_size`` for precision.

    ``production_fraction`` is the probability of a true ``D0`` tag. The
    observed tag is flipped with ``wrong_tag`` before the joint density is
    evaluated. Background generation is intentionally separate from this first
    signal generator and remains the responsibility of the session's background
    APIs. Proposals and resampling run in JAX. A seed is reproducible within
    this implementation, but does not reproduce the former NumPy RNG samples.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 <= float(production_fraction) <= 1.0:
        raise ValueError("production_fraction must lie in [0, 1]")
    wrong = float(session.wrong_tag if wrong_tag is None else wrong_tag)
    if not 0.0 <= wrong <= 1.0:
        raise ValueError("wrong_tag must lie in [0, 1]")
    if (
        session.time_nodes is not None
        or session.time_acceptance is not None
        or session.sigma_t is not None
    ):
        raise NotImplementedError(
            "time-dependent toy generation currently requires perfect time "
            "resolution and unit temporal acceptance"
        )
    proposal_size = (
        max(20 * size, 20_000) if proposal_size is None else int(proposal_size)
    )
    if proposal_size < size:
        raise ValueError("proposal_size must be at least size")

    values = {} if parameters is None else parameters
    tau = float(session.mixing.resolved(values)[2])
    if not jnp.isfinite(tau) or tau <= 0:
        raise ValueError("mixing lifetime must be finite and positive")
    key = jax.random.key(secrets.randbits(32) if seed is None else int(seed))
    proposal_key, resample_key = jax.random.split(key)
    times, observed_tags, true_tags, proposal_time = _time_proposal(
        proposal_key,
        proposal_size,
        *session.time_range,
        1.0 / tau,
        float(production_fraction),
        wrong,
    )
    sample = session.model.generate_phase_space(
        proposal_size, seed=seed, include_momenta=include_momenta
    )
    candidate_sample = sample.without_momenta()
    cache = session._prepare_cache(candidate_sample)
    objective = replace(
        session.signal_objective,
        cache=cache,
        times=jnp.asarray(times),
        tags=jnp.asarray(observed_tags),
        wrong_tag=wrong,
        efficiency=_acceptance(
            session.efficiency, session.veto, candidate_sample.as_dict()
        ),
    )
    selected, valid = _resample_indices(
        resample_key,
        objective.densities(values),
        proposal_time,
        size,
    )
    if not bool(valid):
        raise ValueError("time-dependent toy proposal produced invalid weights")
    selected_data = sample.take(jnp.asarray(selected))
    if not include_momenta:
        selected_data = selected_data.without_momenta()
    return TimeDependentToy(
        data=selected_data,
        times=jnp.asarray(times[selected]),
        tags=jnp.asarray(observed_tags[selected]),
        true_tags=jnp.asarray(true_tags[selected]),
    )


__all__ = ["TimeDependentToy", "generate_time_dependent_toy"]
