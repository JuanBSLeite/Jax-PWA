"""Numerical and distribution checks for device-resident optimizer/toy paths."""

import gc
import weakref
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    DecayChannel,
    DecayModel,
    NeutralMesonMixing,
    NonResonant,
    Parameter,
    TimeDependentFitSession,
    generate_time_dependent_toy,
)
from jaxpwa.fit.nesterov import _SOLVERS, minimize


def test_nesterov_bounds_fixed_values_and_runtime_inputs_reuse():
    traces = []

    def objective(p):
        traces.append(None)
        return (p["x"] - p["target"]) ** 2

    parameters = [
        Parameter("x", 0.0, bounds=(-1.0, 1.0), step=0.1),
        Parameter("target", 2.0, fixed=True),
    ]
    first = minimize(objective, parameters, max_iter=200)
    n_traces = len(traces)
    assert first.valid
    assert first.values["x"] == pytest.approx(1.0)
    assert first.fval == pytest.approx(1.0)
    parameters = [
        replace(parameters[0], step=0.2, bounds=(-3.0, 3.0)),
        replace(parameters[1], value=-2.0),
    ]
    second = minimize(objective, parameters, start_values={"x": 1.0}, max_iter=200)
    assert second.valid
    assert second.values["x"] == pytest.approx(-2.0, abs=3e-4)
    assert second.values["target"] == -2.0
    assert len(traces) == n_traces
    for result in (first, second):
        assert np.all(np.diff([h["nll"] for h in result.history]) <= 0)
        assert result.covariance is None


def test_nesterov_failed_search_keeps_initial_point_and_counts_evaluations():
    # The only finite point has a nonzero gradient; no proposed move is valid.
    def objective(p):
        return jnp.where(p["x"] == 0, p["x"], jnp.nan)

    result = minimize(objective, [Parameter("x", 0.0)], max_iter=10)
    assert result.status == "line_search_failed"
    assert result.values["x"] == 0
    assert result.fval == 0
    assert len(result.history) == 1
    assert result.nfcn == 121  # Initial point + two 60-step searches.


def test_nesterov_solver_does_not_retain_objective():
    def objective(p):
        return p["x"] ** 2

    ref = weakref.ref(objective)
    identity = id(objective)
    minimize(objective, [Parameter("x", 1.0)], max_iter=5)
    assert any(key[0] == identity for key in _SOLVERS)
    del objective
    gc.collect()
    assert ref() is None
    assert not any(key[0] == identity for key in _SOLVERS)


@pytest.mark.parametrize("high", [1.4, float("inf")])
def test_temporal_toy_lifetime_tags_reproducibility_and_momenta(high):
    low, tau = 0.2, 0.4103
    model = DecayModel(
        DecayChannel("D0", ("K(S)0", "pi+", "pi-")),
        [NonResonant(1.0)],
        normalization_method="square-dalitz",
        normalization_resolution=8,
    )
    sample = model.generate_phase_space(8, seed=1, include_momenta=False)
    session = TimeDependentFitSession(
        model,
        sample,
        jnp.linspace(0.3, 1.0, 8),
        jnp.array([1, -1] * 4),
        NeutralMesonMixing(0.0, 0.0, tau),
        time_range=(low, high),
    )
    kwargs = dict(
        size=12_000,
        proposal_size=60_000,
        production_fraction=0.75,
        wrong_tag=0.2,
        seed=104,
        include_momenta=True,
    )
    toy = generate_time_dependent_toy(session, **kwargs)
    expected_mean = low + tau
    if np.isfinite(high):
        expected_mean -= (high - low) / np.expm1((high - low) / tau)
    assert float(jnp.mean(toy.times)) == pytest.approx(expected_mean, abs=0.015)
    assert np.mean(np.asarray(toy.tags) == 1) == pytest.approx(0.65, abs=0.02)
    assert np.mean(np.asarray(toy.true_tags) == 1) == pytest.approx(0.75, abs=0.02)
    assert np.mean(np.asarray(toy.tags != toy.true_tags)) == pytest.approx(
        0.2, abs=0.02
    )
    assert np.all((np.asarray(toy.times) >= low) & (np.asarray(toy.times) <= high))
    assert toy.data.p1.shape == (12_000, 4)
    repeated = generate_time_dependent_toy(session, **kwargs)
    for a, b in (
        (toy.times, repeated.times),
        (toy.tags, repeated.tags),
        (toy.data.p1, repeated.data.p1),
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("efficiency_value", [0.0, float("nan")])
def test_temporal_toy_rejects_invalid_public_importance_weights(efficiency_value):
    model = DecayModel(
        DecayChannel("D0", ("K(S)0", "pi+", "pi-")),
        [NonResonant(1.0)],
        normalization_method="square-dalitz",
        normalization_resolution=8,
    )
    sample = model.generate_phase_space(8, seed=1, include_momenta=False)
    session = TimeDependentFitSession(
        model,
        sample,
        jnp.linspace(0.1, 1.0, 8),
        jnp.array([1, -1] * 4),
        NeutralMesonMixing(0.0, 0.0, 0.4103),
        efficiency=lambda data: jnp.full(data["s12"].shape, efficiency_value),
    )
    with pytest.raises(ValueError, match="invalid weights|finite and non-negative"):
        generate_time_dependent_toy(session, 10, proposal_size=100, seed=7)
