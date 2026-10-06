import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import enable_x64
from jaxpwa.likelihood import WeightedUnbinnedNLL
from jaxpwa.likelihood.weighted import (
    sandwich_covariance_from_score_outer,
    sweight_covariance_from_hessians,
)

enable_x64()


def _scaled_logpdf(data, parameters):
    return -0.5 * parameters["scale"] * data["x"] ** 2


def test_weighted_unbinned_nll_matches_manual_sum():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 0.5, -0.2])
    nll = WeightedUnbinnedNLL(_scaled_logpdf, data, weights)
    parameters = {"scale": 2.0}
    expected = -jnp.sum(weights * _scaled_logpdf(data, parameters))
    assert jnp.allclose(nll(parameters), expected, rtol=1e-12, atol=1e-12)


def test_weighted_unbinned_nll_is_differentiable():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 0.5, 0.2])
    nll = WeightedUnbinnedNLL(_scaled_logpdf, data, weights)
    grad = jax.grad(lambda parameters: nll(parameters))({"scale": 2.0})
    expected = -jnp.sum(weights * (-0.5 * data["x"] ** 2))
    assert jnp.allclose(grad["scale"], expected, rtol=1e-12, atol=1e-12)


def test_weighted_unbinned_nll_rejects_empty_data():
    with pytest.raises(ValueError, match="non-empty"):
        WeightedUnbinnedNLL(_scaled_logpdf, {}, jnp.array([]))


def test_weighted_unbinned_nll_rejects_mismatched_weight_shape():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="weights must have shape"):
        WeightedUnbinnedNLL(_scaled_logpdf, data, jnp.array([1.0, 1.0]))


def test_weighted_unbinned_nll_rejects_non_finite_weights():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="finite"):
        WeightedUnbinnedNLL(_scaled_logpdf, data, jnp.array([1.0, jnp.nan, 1.0]))


def test_weighted_unbinned_nll_zero_weight_masks_non_finite_logpdf():
    data = {"x": jnp.array([0.0, 2.0])}
    weights = jnp.array([0.0, 1.0])
    nll = WeightedUnbinnedNLL(lambda d, p: jnp.log(d["x"]), data, weights)
    assert float(nll({})) == pytest.approx(-np.log(2.0))


def test_weighted_unbinned_nll_zero_weight_gradient_needs_safe_logpdf():
    # The mask fixes the value, but autodiff still differentiates the
    # user's logpdf. A safe-log logpdf (as Jax-PWA's own signal densities
    # use) keeps the gradient of a zero-weight physical zero finite.
    data = {"x": jnp.array([0.0, 1.0, 2.0])}

    def safe_logpdf(d, p):
        intensity = (d["x"] * p["a"]) ** 2
        valid = intensity > 0
        log_intensity = jnp.where(
            valid, jnp.log(jnp.where(valid, intensity, 1.0)), -jnp.inf
        )
        return log_intensity - jnp.log(5.0 * p["a"] ** 2 + 1.0)

    nll = WeightedUnbinnedNLL(safe_logpdf, data, jnp.array([0.0, 1.0, -0.3]))
    reference = WeightedUnbinnedNLL(
        safe_logpdf, {"x": data["x"][1:]}, jnp.array([1.0, -0.3])
    )
    gradient = jax.grad(lambda a: nll({"a": a}))(1.2)
    assert np.isfinite(float(gradient))
    assert float(gradient) == pytest.approx(
        float(jax.grad(lambda a: reference({"a": a}))(1.2)), rel=1e-12
    )


def test_weighted_unbinned_nll_active_non_finite_logpdf_is_invalid():
    data = {"x": jnp.array([0.0, 2.0])}
    nll = WeightedUnbinnedNLL(
        lambda d, p: jnp.log(d["x"]), data, jnp.array([1.0, 0.0])
    )
    assert jnp.isposinf(nll({}))


def test_weighted_unbinned_nll_requires_real_weights_and_logpdf():
    data = {"x": jnp.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="weights must be real"):
        WeightedUnbinnedNLL(_scaled_logpdf, data, jnp.array([1.0 + 1.0j, 2.0]))
    nll = WeightedUnbinnedNLL(
        lambda d, p: jnp.asarray(d["x"], dtype=jnp.complex128),
        data,
        jnp.ones(2),
    )
    with pytest.raises(ValueError, match="logpdf must return real"):
        nll({})


def test_weighted_unbinned_nll_rejects_logpdf_shape_mismatch():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 1.0, 1.0])
    nll = WeightedUnbinnedNLL(lambda d, p: jnp.array([0.0, 0.0]), data, weights)
    with pytest.raises(ValueError, match="logpdf must return shape"):
        nll({})


def test_sweight_covariance_from_hessians_matches_matrix_formula():
    weighted_hessian = np.array([[5.0, 1.0], [1.0, 3.0]])
    squared_weight_hessian = np.array([[7.0, 0.5], [0.5, 4.0]])
    inverse = np.linalg.inv(weighted_hessian)
    expected = inverse @ squared_weight_hessian @ inverse
    actual = sweight_covariance_from_hessians(
        weighted_hessian,
        squared_weight_hessian,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sandwich_covariance_from_score_outer_matches_matrix_formula():
    weighted_hessian = np.array([[5.0, 1.0], [1.0, 3.0]])
    score_outer = np.array([[8.0, 1.5], [1.5, 6.0]])
    inverse = np.linalg.inv(weighted_hessian)
    expected = inverse @ score_outer @ inverse
    actual = sandwich_covariance_from_score_outer(weighted_hessian, score_outer)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sweight_covariance_from_hessians_rejects_bad_shapes():
    with pytest.raises(ValueError, match="square"):
        sweight_covariance_from_hessians(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        sweight_covariance_from_hessians(np.eye(2), np.eye(3))


def test_sweight_covariance_from_hessians_rejects_singular_hessian():
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        sweight_covariance_from_hessians(
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.eye(2),
        )


def test_sandwich_covariance_rejects_nonfinite_and_singular_inputs():
    with pytest.raises(ValueError, match="finite"):
        sandwich_covariance_from_score_outer(
            np.array([[np.nan, 0.0], [0.0, 1.0]]), np.eye(2)
        )
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        sandwich_covariance_from_score_outer(
            np.array([[1.0, 1.0], [1.0, 1.0]]), np.eye(2)
        )
