import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    DalitzAmplitude,
    DecayChannel,
    DecayModel,
    Parameter,
    PolarFormFactorSymNR,
    RealImag,
    enable_x64,
    model_from_spec,
    model_to_spec,
)

enable_x64()


def _laura_polar_ff_sym_nr(s, t, lam, no_interference=False):
    # LauPolarFormFactorSymNR::amplitude, written out independently.
    if no_interference:
        return 1.0 / (1.0 + min(s, t) / lam**2)
    return 1.0 / (1.0 + s / lam**2) + 1.0 / (1.0 + t / lam**2)


@pytest.mark.parametrize("no_interference", [False, True])
def test_matches_laura_formula_at_reference_points(no_interference):
    points = [(1.1, 9.0), (4.0, 4.0), (12.5, 2.3), (25.0, 0.98)]
    lam = 1.5845
    shape = PolarFormFactorSymNR(lam, no_interference=no_interference)
    data = {
        "s13": jnp.asarray([s for s, _ in points]),
        "s23": jnp.asarray([t for _, t in points]),
    }
    value = np.asarray(shape(data))
    expected = [_laura_polar_ff_sym_nr(s, t, lam, no_interference) for s, t in points]
    np.testing.assert_allclose(value.real, expected, rtol=1e-14)
    np.testing.assert_array_equal(value.imag, 0.0)


def test_symmetric_under_exchange_of_the_two_invariants():
    shape = PolarFormFactorSymNR(1.2)
    a = shape({"s13": jnp.asarray([1.5, 7.0]), "s23": jnp.asarray([7.0, 1.5])})
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(a[1]), rtol=1e-15)


def test_custom_invariants_select_the_data_keys():
    invariants = {"s12": 2.0, "s13": 3.0, "s23": 5.0}
    data = {key: jnp.asarray([value]) for key, value in invariants.items()}
    value = PolarFormFactorSymNR(1.0, invariants=("s12", "s23"))(data)
    np.testing.assert_allclose(np.asarray(value).real, [1 / 3 + 1 / 6], rtol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_": 0.0},
        {"lambda_": -1.0},
        {"lambda_": float("inf")},
        {"lambda_": float("nan")},
        {"invariants": ("s13", "s13")},
        {"invariants": ("s13", "m23")},
        {"no_interference": 1},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PolarFormFactorSymNR(**kwargs)


def _kkk_model(lam):
    return DecayModel(
        DecayChannel("B+", ("K+", "K+", "K-")),
        [DalitzAmplitude("NR", PolarFormFactorSymNR(lam), RealImag(1.0, 0.0))],
        normalization_resolution=30,
    )


def test_floating_lambda_changes_intensity_with_a_finite_gradient():
    lam = Parameter.dynamics("NR.lambda", 1.5845, owner="NR", bounds=(0.0, 10.0))
    model = _kkk_model(lam)
    assert "NR.lambda" in {parameter.name for parameter in model.parameters}

    data = model.generate_phase_space(500, seed=3).as_dict()
    for value in (1.5845, 2.5):
        # A normalized intensity: proportional to |A|^2 at this lambda.
        intensity = np.asarray(model.intensity(data, {"NR.lambda": value}))
        expected = np.abs(np.asarray(PolarFormFactorSymNR(value)(data))) ** 2
        ratio = intensity / expected
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)

    gradient = jax.grad(
        lambda value: jnp.sum(jnp.log(model.intensity(data, {"NR.lambda": value})))
    )(1.5845)
    assert np.isfinite(float(gradient)) and abs(float(gradient)) > 0.0


def test_compact_prepared_data_keeps_only_the_two_invariants():
    shape = PolarFormFactorSymNR(1.0)
    data = {"s12": 1.0, "s13": 2.0, "s23": 3.0, "extra": 4.0}
    assert shape.compact_prepared_data(data) == {"s13": 2.0, "s23": 3.0}


def test_generic_floating_lambda_cannot_be_silently_frozen_by_the_cache():
    with pytest.raises(ValueError, match="Parameter.dynamics"):
        PolarFormFactorSymNR(Parameter("NR.lambda", 1.2))
    assert PolarFormFactorSymNR(Parameter("NR.lambda", 1.2, fixed=True))


def test_model_io_round_trip():
    lam = Parameter.dynamics("NR.lambda", 1.5845, owner="NR", bounds=(0.0, 10.0))
    model = _kkk_model(lam)
    restored = model_from_spec(model_to_spec(model))
    data = model.generate_phase_space(300, seed=5).as_dict()
    assert jnp.allclose(model.intensity(data), restored.intensity(data))
    assert "NR.lambda" in {parameter.name for parameter in restored.parameters}


@pytest.mark.parametrize("no_interference", [False, True])
def test_public_parameter_resolution_and_boundary_gradient(no_interference):
    lam = Parameter.dynamics("NR.lambda", 1.0, owner="NR")
    shape = PolarFormFactorSymNR(lam, no_interference=no_interference)
    data = {"s13": jnp.asarray([2.0]), "s23": jnp.asarray([3.0])}
    expected = PolarFormFactorSymNR(2.0, no_interference=no_interference)(data)
    np.testing.assert_allclose(shape(data, {"NR.lambda": 2.0}), expected)
    np.testing.assert_allclose(
        shape(data), PolarFormFactorSymNR(1.0, no_interference)(data)
    )

    evaluate = jax.jit(
        jax.value_and_grad(lambda value: jnp.real(shape(data, {"NR.lambda": value})[0]))
    )
    for value in (0.0, 1e-200):
        magnitude, gradient = evaluate(value)
        assert float(magnitude) == 0.0
        slope = 1.0 if no_interference else 5.0 / 3.0
        np.testing.assert_allclose(float(gradient), slope * value, rtol=1e-14, atol=0.0)
