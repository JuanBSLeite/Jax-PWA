import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import crystalball

from jaxpwa import (
    BreitWigner1D,
    Chebyshev1D,
    CrystalBall1D,
    Exponential1D,
    FactorizedDensity,
    Gaussian1D,
    Histogram1D,
    LineshapeIntensity1D,
    Parameter,
    RelativisticBreitWigner,
    ResonanceContext,
    SumPDF1D,
)


def _trapz(y, x):
    return jnp.trapezoid(y, x)


def test_gaussian_is_normalized_on_finite_interval():
    pdf = Gaussian1D(mean=0.2, sigma=0.7, low=-2.0, high=2.5)
    x = jnp.linspace(-2.0, 2.5, 20001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_breit_wigner_is_normalized_on_finite_interval():
    pdf = BreitWigner1D(mean=5.279, width=0.030, low=5.15, high=5.40)
    x = jnp.linspace(5.15, 5.40, 40001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_relativistic_lineshape_intensity_uses_context_mass_range_and_is_normalized():
    context = ResonanceContext(
        parent_mass=5.27934,
        daughter_masses=(0.493677, 0.13957039),
        bachelor_mass=0.13957039,
        spin=1,
        pole_mass=0.8958,
        pole_width=0.0474,
        resonance_radius=4.0,
        parent_radius=4.0,
    )
    pdf = LineshapeIntensity1D.from_context(
        RelativisticBreitWigner(), context, quadrature_order=512
    )
    expected_low = 0.493677 + 0.13957039
    expected_high = 5.27934 - 0.13957039
    assert jnp.isclose(pdf.low, expected_low)
    assert jnp.isclose(pdf.high, expected_high)

    x = jnp.linspace(pdf.low, pdf.high, 60001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=8e-4, atol=8e-4)
    assert jnp.all(pdf(x) >= 0.0)


def test_crystal_ball_matches_scipy_reference_and_is_normalized():
    mean, sigma, alpha, n = 0.3, 1.2, 1.681, 11.0
    low, high = mean - 3 * sigma, mean + 2 * sigma
    pdf = CrystalBall1D(mean=mean, sigma=sigma, alpha=alpha, n=n, low=low, high=high)

    x = jnp.linspace(low, high, 40001)
    values = pdf(x)
    assert jnp.isclose(_trapz(values, x), 1.0, rtol=1e-4, atol=1e-4)

    reference = crystalball(alpha, n, loc=mean, scale=sigma)
    reference_norm = reference.cdf(high) - reference.cdf(low)
    reference_values = reference.pdf(np.asarray(x)) / reference_norm
    assert np.allclose(np.asarray(values), reference_values, rtol=1e-8, atol=1e-10)


def test_crystal_ball_gradients_are_finite():
    mean, sigma, alpha, n = 0.3, 1.2, 1.681, 11.0
    low, high = mean - 3 * sigma, mean + 2 * sigma
    x = jnp.linspace(low, high, 21)[1:-1]

    def log_density(mean_, sigma_, alpha_, n_):
        pdf = CrystalBall1D(
            mean=mean_, sigma=sigma_, alpha=alpha_, n=n_, low=low, high=high
        )
        return jnp.sum(jnp.log(pdf(x)))

    grads = jax.grad(log_density, argnums=(0, 1, 2, 3))(mean, sigma, alpha, n)
    assert all(bool(jnp.isfinite(g)) for g in grads)


def test_crystal_ball_gradients_match_finite_differences_beyond_tail_pole():
    # With small n the power-law pole b = n/alpha - alpha lies inside the core
    # (here b = 0.5): core points with z > b used to evaluate the discarded
    # tail branch at an infinite value, giving NaN gradients for every shape
    # parameter even though the density itself was correct.
    low, high = 5150.0, 5500.0
    x = jnp.linspace(low + 5.0, high - 5.0, 200)
    names = ("mean", "sigma", "alpha", "n")
    point = {"mean": 5280.0, "sigma": 17.0, "alpha": 1.5, "n": 3.0}

    def log_density(values):
        pdf = CrystalBall1D(*(values[name] for name in names), low=low, high=high)
        return jnp.sum(jnp.log(pdf(x)))

    grads = jax.grad(log_density)(point)
    for name in names:
        step = 1e-6 * max(1.0, abs(point[name]))
        up, down = dict(point), dict(point)
        up[name] += step
        down[name] -= step
        finite_difference = (log_density(up) - log_density(down)) / (2.0 * step)
        assert bool(jnp.isfinite(grads[name])), name
        assert np.isclose(float(grads[name]), float(finite_difference), rtol=1e-5), name


def test_crystal_ball_negative_alpha_is_the_mirrored_right_tail():
    # RooCBShape/LHCb convention: alpha < 0 puts the power-law tail on the
    # high side. Reference: scipy's left-tail Crystal Ball of -x.
    mean, sigma, alpha, n = 5280.0, 24.0, -2.2, 2.04
    low, high = 5150.0, 5500.0
    pdf = CrystalBall1D(mean=mean, sigma=sigma, alpha=alpha, n=n, low=low, high=high)
    x = jnp.linspace(low, high, 40001)
    values = pdf(x)
    assert jnp.isclose(_trapz(values, x), 1.0, rtol=1e-6, atol=1e-6)

    reference = crystalball(-alpha, n, loc=-mean, scale=sigma)
    reference_norm = reference.cdf(-low) - reference.cdf(-high)
    reference_values = reference.pdf(-np.asarray(x)) / reference_norm
    assert np.allclose(np.asarray(values), reference_values, rtol=1e-8, atol=1e-12)
    # The tail is on the right: far above the peak it decays slower than a Gaussian.
    assert float(pdf(jnp.asarray([mean + 5 * sigma]))[0]) > float(
        pdf(jnp.asarray([mean - 5 * sigma]))[0]
    )

    grads = jax.grad(
        lambda a: jnp.sum(
            jnp.log(CrystalBall1D(mean, sigma, a, n, low, high)(x[1:-1:400]))
        )
    )(alpha)
    assert bool(jnp.isfinite(grads))


def test_sum_pdf_uses_recursive_fractions_and_stays_normalized():
    low, high = 5150.0, 5500.0
    gauss = Gaussian1D(5280.0, 15.0, low, high)
    left = CrystalBall1D(5280.0, 34.0, 0.47, 2.34, low, high)
    right = CrystalBall1D(5280.0, 24.0, -2.23, 2.04, low, high)
    f_gauss, f_cb = 0.64, 0.09
    pdf = SumPDF1D((gauss, left, right), (f_gauss, f_cb))
    x = jnp.linspace(low, high, 40001)
    expected = (
        f_gauss * gauss(x)
        + (1 - f_gauss) * f_cb * left(x)
        + (1 - f_gauss) * (1 - f_cb) * right(x)
    )
    assert np.allclose(np.asarray(pdf(x)), np.asarray(expected), rtol=1e-12)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=1e-6, atol=1e-6)


def test_chebyshev_matches_numpy_and_is_normalized():
    low, high = 0.98, 1.05
    coefficients = (0.4, -0.2, 0.1, 0.05)
    pdf = Chebyshev1D(coefficients, low, high)
    x = jnp.linspace(low, high, 40001)
    values = pdf(x)
    assert jnp.isclose(_trapz(values, x), 1.0, rtol=1e-8, atol=1e-8)
    t = (2.0 * np.asarray(x) - low - high) / (high - low)
    raw = np.polynomial.chebyshev.chebval(t, (1.0, *coefficients))
    assert np.allclose(
        np.asarray(values), raw / np.trapezoid(raw, np.asarray(x)), rtol=1e-6
    )
    grad = jax.grad(
        lambda c: jnp.sum(jnp.log(Chebyshev1D((c, -0.2), low, high)(x[::1000])))
    )(0.4)
    assert bool(jnp.isfinite(grad))


@pytest.mark.parametrize("coefficients", [(2.0,), (1.6, 0.5), (0.0, 1.1, 0.0)])
def test_chebyshev_rejects_negative_regions_even_without_events_there(coefficients):
    # For (1.6, 0.5), both endpoints and t=0 are positive; its interior
    # minimum at t=-0.8 is negative. Testing only event values misses this.
    pdf = Chebyshev1D(coefficients, -1.0, 1.0)
    assert np.all(np.isnan(np.asarray(jax.jit(pdf)(jnp.asarray([0.9, 1.0])))))


@pytest.mark.parametrize(
    "coefficients",
    [(), (1.4, 0.5), (1.4, 0.5, 0.0), (1.4, 0.5, 0.01, 0.005), (0.0, 0.0, 0.0)],
)
def test_chebyshev_allows_positive_polynomials_and_degenerate_degree(coefficients):
    x = jnp.linspace(-1.0, 1.0, 20001)
    density = jax.jit(Chebyshev1D(coefficients, -1.0, 1.0))(x)
    assert float(_trapz(density, x)) == pytest.approx(1.0, rel=1e-8)


def test_chebyshev_physical_domain_check_preserves_gradient_and_hessian():
    coefficient = Parameter("c", 1.4)
    pdf = Chebyshev1D((coefficient, 0.5, 0.0), -1.0, 1.0)
    x = jnp.asarray([-0.5, 0.2, 0.7])

    def objective(c):
        return jnp.sum(jnp.log(pdf(x, {"c": c})))

    def reference(c):
        return jnp.sum(jnp.log((0.5 + c * x + x**2) / (5.0 / 3.0)))

    for derivative in (jax.grad, lambda f: jax.grad(jax.grad(f))):
        np.testing.assert_allclose(
            jax.jit(derivative(objective))(1.4), derivative(reference)(1.4), rtol=1e-12
        )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_sum_pdf_rejects_invalid_fractions(fraction):
    pdfs = (Gaussian1D(0, 1, -5, 5), Gaussian1D(1, 1, -5, 5))
    with pytest.raises(ValueError, match="fractions"):
        SumPDF1D(pdfs, (fraction,))
    mixture = SumPDF1D(pdfs, (Parameter("f", 0.5),))
    density = jax.jit(lambda f: mixture(jnp.asarray([0.0]), {"f": f}))(fraction)
    assert np.isnan(float(density[0]))


def test_invalid_chebyshev_cannot_hide_behind_a_positive_mixture_component():
    mixture = SumPDF1D(
        (Gaussian1D(0, 1, -1, 1), Chebyshev1D((1.6, 0.5), -1, 1)), (0.5,)
    )
    assert np.all(np.isnan(np.asarray(mixture(jnp.asarray([0.0, 0.9])))))


@pytest.mark.parametrize("alpha", [0.05, -0.05])
def test_crystalball_large_power_has_finite_density_and_shape_derivatives(alpha):
    # These are permitted bounds in the mass-fit notebooks, but the direct
    # (n/|alpha|)**n coefficient overflows in float64.
    x = jnp.linspace(-20.0, 20.0, 20001)
    pdf = CrystalBall1D(0.0, 1.0, alpha, 100.0, -20.0, 20.0)
    values = jax.jit(pdf)(x)
    assert bool(jnp.all(jnp.isfinite(values) & (values > 0)))
    assert float(_trapz(values, x)) == pytest.approx(1.0, rel=1e-7)

    def objective(a, n):
        density = CrystalBall1D(0.0, 1.0, a, n, -20.0, 20.0)(x[::2000])
        return jnp.sum(jnp.log(density))

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(alpha, 100.0)
    for index, gradient in enumerate(gradients):
        point = [alpha, 100.0]
        step = 1e-5 if index == 0 else 1e-2
        up, down = point.copy(), point.copy()
        up[index] += step
        down[index] -= step
        reference = (float(objective(*up)) - float(objective(*down))) / (2 * step)
        assert np.isfinite(float(gradient))
        assert float(gradient) == pytest.approx(reference, rel=1e-6, abs=1e-9)


def test_exponential_is_normalized_on_finite_interval():
    pdf = Exponential1D(slope=-1.3, low=0.0, high=1.0)
    x = jnp.linspace(0.0, 1.0, 20001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_histogram_is_normalized():
    pdf = Histogram1D(
        edges=jnp.array([0.0, 0.2, 0.5, 1.0]), values=jnp.array([1.0, 3.0, 2.0])
    )
    widths = jnp.diff(pdf.edges)
    assert jnp.isclose(jnp.sum(widths * pdf.values), 1.0)


def test_factorized_density_multiplies_independent_terms():
    def base(pars):
        return jnp.array([0.2, 0.8])

    mass_pdf = Gaussian1D(mean=5.28, sigma=0.02, low=5.2, high=5.35)
    density = FactorizedDensity(
        base_density=base,
        observables={"mass": jnp.array([5.28, 5.30])},
        pdfs={"mass": mass_pdf},
    )
    result = density({})
    expected = base({}) * mass_pdf(jnp.array([5.28, 5.30]), {})
    assert jnp.allclose(result, expected)
