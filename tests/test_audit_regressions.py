"""Regressions reproduced by the October 2026 numerical/cache review."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import Minimizer, Parameter
from jaxpwa._quadrature import legendre_rule
from jaxpwa.amplitude import AmplitudeComponent, PreparedAmplitudeCache


def _shape(data, parameters=None):
    parameters = {} if parameters is None else parameters
    return 1 + parameters.get("slope", 0.2) * data["x"]


def _cache(parameters, *, components=None, chunk_size=100):
    data = {"x": jnp.linspace(0.0, 1.0, 7)}
    return PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", _shape, 1.0),) if components is None else components,
        data=data,
        normalization_data=data,
        normalization_weights=jnp.ones(7),
        parameters=parameters,
        normalization_chunk_size=chunk_size,
    )


def test_cache_check_rejects_renamed_parameter_on_same_floating_component():
    parameter = Parameter.dynamics("a.slope", 0.2, owner="a", backend_name="slope")
    cache = _cache((parameter,))
    renamed = replace(parameter, name="a.new_slope")
    # The stale cache reads a.slope; the incoming fit would vary a.new_slope.
    gradient = jax.grad(lambda x: cache.intensity({renamed.name: x}).sum())(0.5)
    assert gradient == 0.0
    with pytest.raises(ValueError, match="inconsistent"):
        cache.check_parameters((renamed,))


@pytest.mark.parametrize("change", ["fixed", "value", "backend", "missing", "added"])
def test_cache_check_covers_each_dynamic_binding(change):
    slope = Parameter.dynamics("a.slope", 0.2, owner="a", backend_name="slope")
    offset = Parameter.dynamics("a.offset", 1.0, owner="a", fixed=True)
    cache = _cache((slope, offset))
    incoming = {
        "fixed": (slope, replace(offset, fixed=False)),
        "value": (slope, replace(offset, value=2.0)),
        "backend": (replace(slope, backend_name="other"), offset),
        "missing": (slope,),
        "added": (slope, offset, Parameter.dynamics("a.extra", 0.1, owner="a")),
    }[change]
    with pytest.raises(ValueError, match="inconsistent"):
        cache.check_parameters(incoming)


def test_cache_check_allows_changed_free_start_and_optimizer_settings():
    slope = Parameter.dynamics("a.slope", 0.2, owner="a", backend_name="slope")
    cache = _cache((slope,))
    cache.check_parameters((replace(slope, value=0.3, bounds=(0, 1), step=0.01),))


@pytest.mark.parametrize("chunk_size", [3, 100])
@pytest.mark.parametrize("normalize_zero", [False, True])
def test_fixed_zero_component_only_requires_positive_integral_when_normalized(
    chunk_size, normalize_zero,
):
    components = (
        AmplitudeComponent("a", _shape, 1.0),
        AmplitudeComponent(
            "zero", lambda data, pars: jnp.zeros_like(data["x"]),
            1.0, normalize_component=normalize_zero,
        ),
    )
    if normalize_zero:
        with pytest.raises(ValueError, match="positive diagonal"):
            _cache((), components=components, chunk_size=chunk_size)
    else:
        cache = _cache((), components=components, chunk_size=chunk_size)
        np.testing.assert_allclose(cache.normalization_matrix({}), [[1, 0], [0, 0]])
        assert jnp.all(jnp.isfinite(cache.intensity({})))


def test_minimizer_does_not_share_hessian_schedule_across_parameter_kinds():
    def objective(values):
        return values["x"] ** 2

    coefficient = Parameter.coefficient("x", 1.0)
    dynamic = Parameter.dynamics("x", 1.0, owner="a")
    first = Minimizer(objective, (coefficient,))._backend()
    second = Minimizer(objective, (dynamic,))._backend()
    # The dynamic backend must checkpoint and dispatch one HVP at a time.
    assert first[4] is not second[4]
    np.testing.assert_allclose(first[4](1.0), second[4](1.0))


def test_shared_quadrature_preserves_numpy_rule_and_rejects_mutation():
    nodes, weights = legendre_rule(17)
    expected = np.polynomial.legendre.leggauss(17)
    np.testing.assert_array_equal(nodes, expected[0])
    np.testing.assert_array_equal(weights, expected[1])
    # Exact polynomial moments up through the rule's degree of exactness.
    for power in range(34):
        target = 0.0 if power % 2 else 2.0 / (power + 1)
        np.testing.assert_allclose(weights @ nodes**power, target, atol=3e-15)
    for array in (nodes, weights):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_scaled_quadrature_cannot_modify_shared_rule():
    from jaxpwa.integration.gauss_legendre import _scaled_legendre

    nodes, weights = _scaled_legendre(17, 2, 5)
    nodes[:] = 0
    weights[:] = 0
    fresh_nodes, fresh_weights = _scaled_legendre(17, 2, 5)
    expected_nodes, expected_weights = np.polynomial.legendre.leggauss(17)
    np.testing.assert_array_equal(fresh_nodes, 3.5 + 1.5 * expected_nodes)
    np.testing.assert_array_equal(fresh_weights, 1.5 * expected_weights)


def test_worsened_polishing_cannot_replace_an_earlier_minuit_stage(monkeypatch):
    from iminuit import Minuit

    original = Minuit.migrad
    stage_values = []

    def displaced_polish(self, **kwargs):
        if stage_values:
            self.values["x"] = 200.0
        result = original(self, **kwargs)
        stage_values.append(float(self.fval))
        return result

    monkeypatch.setattr(Minuit, "migrad", displaced_polish)
    result = Minimizer(lambda p: p["x"] ** 2, (Parameter("x", 2.0),)).fit(
        strategy=1, hesse=False, ncall=1,
    )
    assert stage_values == [4.0, 40000.0]
    assert result.fval == 4.0
    assert result.values["x"] == 2.0
    assert not result.valid  # The fallback must not fabricate convergence.


@pytest.mark.parametrize("data_first", [False, True])
def test_model_reuses_data_executable_across_preparation_routes(data_first):
    from jaxpwa import DecayChannel, DecayModel, NonResonant

    model = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(1.0)], normalization_method="square-dalitz",
        normalization_resolution=9,
    )
    data_kernel = (
        model._compact_data_kernel(normalize_components=True) if data_first else None
    )
    plain = model._compact_prepare_kernel(
        normalize_components=True, has_efficiency=False,
    )
    accepted = model._compact_prepare_kernel(
        normalize_components=True, has_efficiency=True,
    )
    reused = model._compact_data_kernel(normalize_components=True)
    assert plain.data_kernel is accepted.data_kernel is reused
    if data_first:
        assert reused is data_kernel
    sample = model.generate_phase_space(12, seed=1)
    first = model.prepare_cache(sample)
    second = model.prepare_cache(sample)
    np.testing.assert_array_equal(first.data_components, second.data_components)


@pytest.mark.parametrize("reference", [[[0.0]], [[0.0], [1.0]]])
def test_kdtree_residuals_support_a_single_neighbour(reference):
    from jaxpwa.goodness_of_fit import kdtree_local_residuals

    observed = np.array([[0.0], [2.0]])
    pulls, counts, expected, radii = kdtree_local_residuals(
        observed, reference, k=1,
    )
    distances = abs(observed[:, None, 0] - np.asarray(reference)[None, :, 0])
    np.testing.assert_array_equal(radii, distances.min(axis=1))
    np.testing.assert_allclose(pulls, (counts - expected) / np.sqrt(expected))
    assert pulls.shape == (2,)
    assert np.isfinite(pulls).all()
