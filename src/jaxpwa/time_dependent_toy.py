"""Importance-sampled toys for tagged time-dependent Dalitz fits."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from .kinematics import PhaseSpaceSample
from .time_dependent_workflow import _acceptance


@dataclass(frozen=True)
class TimeDependentToy:
    """A generated time-dependent Dalitz sample and its observed tags."""

    data: PhaseSpaceSample
    times: jnp.ndarray
    tags: jnp.ndarray
    true_tags: jnp.ndarray


def _truncated_exponential_sample(rng, size, low, high, rate):
    low = max(float(low), 0.0)
    if np.isinf(high):
        return low - np.log(rng.random(size)) / rate
    high = float(high)
    cdf_range = np.exp(-rate * low) - np.exp(-rate * high)
    uniform = rng.random(size)
    return -np.log(np.exp(-rate * low) - uniform * cdf_range) / rate


def _truncated_exponential_density(times, low, high, rate):
    low = max(float(low), 0.0)
    times = np.asarray(times)
    if np.isinf(high):
        normalisation = np.exp(-rate * low)
    else:
        normalisation = np.exp(-rate * low) - np.exp(-rate * float(high))
    density = rate * np.exp(-rate * times) / normalisation
    return np.where((times >= low) & (times <= high), density, 0.0)


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
    APIs.
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
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("mixing lifetime must be finite and positive")
    rng = np.random.default_rng(seed)
    true_tags = np.where(
        rng.random(proposal_size) < float(production_fraction), 1, -1
    ).astype(np.int32)
    observed_tags = np.where(
        rng.random(proposal_size) < wrong, -true_tags, true_tags
    ).astype(np.int32)
    times = _truncated_exponential_sample(
        rng,
        proposal_size,
        session.time_range[0],
        session.time_range[1],
        1.0 / tau,
    )
    proposal_time = _truncated_exponential_density(
        times, session.time_range[0], session.time_range[1], 1.0 / tau
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
    density = np.asarray(objective.densities(values), dtype=float)
    # Tags were already drawn from the requested observed-tag probability.
    # It cancels between target and proposal in the importance ratio.
    weights = density / np.maximum(proposal_time, 1e-300)
    if not np.all(np.isfinite(weights) & (weights >= 0)) or not np.any(weights > 0):
        raise ValueError("time-dependent toy proposal produced invalid weights")
    probabilities = weights / weights.sum()
    selected = rng.choice(proposal_size, size=size, replace=True, p=probabilities)
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
