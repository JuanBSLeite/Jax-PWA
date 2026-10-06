from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    BackgroundSpec,
    BinnedChi2Result,
    DecayChannel,
    DecayModel,
    FitSession,
    GaussianConstraint,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    PointToPointResult,
    RealImag,
    generate_signal_toy,
)


def _model():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(x, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def _data():
    return PhaseSpaceSample(
        s12=jnp.asarray([0.2, 0.3]),
        s13=jnp.asarray([0.4, 0.5]),
        s23=jnp.asarray([2.0, 1.8]),
        weights=jnp.ones(2),
    )


def test_fit_session_signal_only_collects_model_parameters():
    session = FitSession(_model(), _data())
    assert [p.name for p in session.parameters] == ["NR.x"]
    value = session.objective({"NR.x": 1.0})
    assert jnp.isfinite(value)


def test_fit_session_cold_fit_materializes_cache_before_jit():
    session = FitSession(_model(), _data())

    # This deliberately calls fit() without touching objective or signal_cache
    # first. The prepared cache must be built outside the Minimizer JIT trace.
    result = session.fit(
        {"NR.x": 0.9},
        simplex=False,
        ncall=100,
    )

    assert result.valid
    assert "signal_cache" in session.__dict__
    assert "acceptance_data" in session.__dict__


def test_fit_session_fit_update_model_returns_a_model_with_fitted_values():
    session = FitSession(_model(), _data())
    initial_x = next(p for p in session.model.parameters if p.name == "NR.x").value

    result, updated_model = session.fit(
        {"NR.x": 0.9}, simplex=False, ncall=100, update_model=True
    )

    assert result.valid
    fitted_x = next(p for p in updated_model.parameters if p.name == "NR.x").value
    assert fitted_x == pytest.approx(float(result.values["NR.x"]))
    # the session's own model is a frozen dataclass and is left untouched
    session_x = next(p for p in session.model.parameters if p.name == "NR.x")
    assert session_x.value == initial_x


def test_fit_session_cached_signal_matches_generic_pdf():
    session = FitSession(_model(), _data())
    values = {"NR.x": 1.3}
    cached = session._cached_signal_density(values)
    generic = session.signal_pdf(session.data.as_dict(), values)
    assert jnp.allclose(cached, generic, rtol=1e-12, atol=1e-12)


def test_fit_session_reuses_prepared_signal_cache():
    session = FitSession(_model(), _data())
    first = session.signal_cache
    second = session.signal_cache
    assert first is second
    assert first.data_components.shape[0] == session.data.size


def test_fit_session_reuses_projection_sample_for_same_size_and_seed():
    session = FitSession(_model(), _data())
    first = session._get_projection_sample(128, 1234)
    second = session._get_projection_sample(128, 1234)
    different = session._get_projection_sample(128, 1235)

    assert first is second
    assert different is not first


def test_fit_session_projection_prepared_density_matches_generic_pdf():
    session = FitSession(_model(), _data())
    sample = session._get_projection_sample(128, 5678)
    values = {"NR.x": 1.3}

    prepared = session._projection_signal_density(sample, values)
    generic = session.signal_pdf(sample.as_dict(), values)

    assert jnp.allclose(prepared, generic, rtol=1e-12, atol=1e-12)
    assert len(session._projection_prepared) == 1

    again = session._projection_signal_density(sample, values)
    assert jnp.allclose(again, prepared, rtol=1e-12, atol=1e-12)
    assert len(session._projection_prepared) == 1


def test_fit_session_automatically_normalizes_background_shape():
    model = _model()
    data = _data()
    fraction = Parameter("signal_fraction", 0.7, bounds=(0.0, 1.0))
    session = FitSession(
        model,
        data,
        backgrounds=(BackgroundSpec("comb", lambda d: jnp.ones_like(d["s12"])),),
        signal_fraction=fraction,
    )
    category = session.background_categories[0]
    expected = jnp.mean(model.normalization_sample.weights)
    assert jnp.allclose(category.normalization, expected)
    assert any(p.name == "signal_fraction" for p in session.parameters)
    assert jnp.isfinite(session.objective({"NR.x": 1.0, "signal_fraction": 0.7}))


def test_fit_session_adds_external_constraints():
    model = _model()
    constraint = GaussianConstraint(model.parameters[0], mean=1.0, sigma=0.2)
    base = FitSession(model, _data())
    constrained = base.with_constraint(constraint)
    assert jnp.allclose(
        constrained.objective({"NR.x": 1.2}) - base.objective({"NR.x": 1.2}),
        0.5,
    )


def test_fit_session_projection_weights_reproduce_expected_events():
    session = FitSession(_model(), _data())
    components = session._projection_components({"NR.x": 1.0})
    assert len(components) == 1
    assert jnp.allclose(
        jnp.sum(jnp.asarray(components[0][2])),
        session.data.size,
        rtol=1e-6,
    )


def test_sessions_reuse_normalization_without_acceptance(monkeypatch):
    model = _model()
    first = FitSession(model, model.generate_phase_space(16, seed=401))
    _ = first.signal_cache
    scales, matrix = model._fixed_normalization_templates[True]

    def unexpected_prepare(**kwargs):
        raise AssertionError('Normalization must not be prepared again')

    # A reused template only needs the data kernel, never the full norm kernel.
    monkeypatch.setattr(DecayModel, '_compact_prepare_kernel',
                        lambda self, **kwargs: unexpected_prepare(**kwargs))
    second = FitSession(model, model.generate_phase_space(16, seed=402))
    assert jnp.allclose(second.signal_cache.normalization_matrix_fixed, matrix)
    assert jnp.allclose(second.signal_cache.component_scales, scales)
    assert 'acceptance_normalization' not in second.__dict__
    assert jnp.allclose(second._cached_signal_density({'NR.x': 1.0}),
                        second.signal_pdf(second.data.as_dict(), {'NR.x': 1.0}))


def test_sessions_with_efficiency_or_veto_do_not_reuse_unweighted_template():
    model = _model()
    data = model.generate_phase_space(32, seed=403)
    unweighted = FitSession(model, data).signal_cache.normalization_matrix_fixed
    for kwargs in (
        {'efficiency': lambda d: 0.25 * jnp.ones_like(d['s12'])},
        {'veto': lambda d: d['s12'] > 1.0},
    ):
        session = FitSession(model, data, **kwargs)
        weighted = session.signal_cache.normalization_matrix_fixed
        assert not jnp.allclose(weighted, unweighted)
        assert jnp.allclose(session._cached_signal_density({'NR.x': 1.0}),
                            session.signal_pdf(data.as_dict(), {'NR.x': 1.0}),
                            atol=1e-12)


def _toy_session(n_events=400, seed=7):
    model = _model()
    data = generate_signal_toy(model, n_events, seed=seed)
    return FitSession(model, data)


def _toy_result(values=None):
    return SimpleNamespace(values=values or {"NR.x": 1.0}, fixed=None)


def test_goodness_of_fit_projection_default_free_parameters_counts_only_floating():
    session = _toy_session()
    result = _toy_result()
    gof = session.goodness_of_fit_projection(result, "s13", bins=10)
    assert isinstance(gof, BinnedChi2Result)
    assert gof.n_free_parameters == 1  # only "NR.x" is a floating Parameter
    assert gof.n_bins <= 10
    assert np.isfinite(gof.chi2)
    assert 0.0 <= gof.p_value_min <= gof.p_value_max <= 1.0
    assert np.isclose(np.sum(gof.observed), session.data.size)


def test_goodness_of_fit_projection_honors_explicit_free_parameters():
    session = _toy_session()
    gof = session.goodness_of_fit_projection(
        _toy_result(), "s13", bins=10, n_free_parameters=0
    )
    assert gof.n_free_parameters == 0
    assert gof.dof_max - gof.dof_min == 0


def test_goodness_of_fit_chi2_2d_matches_total_events():
    session = _toy_session()
    gof = session.goodness_of_fit_chi2(_toy_result(), bins=6)
    assert isinstance(gof, BinnedChi2Result)
    assert gof.pulls.ndim == 2
    assert np.isclose(np.sum(gof.observed), session.data.size)


def test_goodness_of_fit_chi2_square_dalitz_requires_mother_mass_and_masses():
    session = _toy_session()

    with pytest.raises(ValueError, match="square_dalitz"):
        session.goodness_of_fit_chi2(_toy_result(), square_dalitz=True)


def test_point_to_point_dissimilarity_returns_bounded_p_value():
    session = _toy_session(n_events=300)
    result = session.point_to_point_dissimilarity(
        _toy_result(), mc_size=600, n_permutations=40
    )
    assert isinstance(result, PointToPointResult)
    assert result.n_data == session.data.size
    assert result.n_reference == 600
    assert 0.0 <= result.p_value <= 1.0


def test_point_to_point_dissimilarity_respects_max_total_events_guard():
    session = _toy_session(n_events=300)

    with pytest.raises(ValueError, match="max_total_events"):
        session.point_to_point_dissimilarity(
            _toy_result(), mc_size=10_000, max_total_events=1_000
        )


def test_plot_projection_show_pulls_returns_shared_axes_pair():
    import matplotlib.pyplot as plt

    session = _toy_session()
    ax, ax_pulls = session.plot_projection(
        _toy_result(), "s13", bins=10, show_pulls=True
    )
    assert ax.get_shared_x_axes().joined(ax, ax_pulls)
    assert ax_pulls.get_xlabel()
    plt.close("all")


def test_plot_projection_default_return_unchanged_by_show_pulls_option():
    import matplotlib.pyplot as plt

    session = _toy_session()
    ax = session.plot_projection(_toy_result(), "s13", bins=10)
    assert not isinstance(ax, tuple)
    plt.close("all")


def test_plot_projection_show_pulls_rejects_explicit_ax():
    import matplotlib.pyplot as plt

    session = _toy_session()
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="show_pulls=True"):
        session.plot_projection(_toy_result(), "s13", show_pulls=True, ax=ax)
    plt.close("all")


@pytest.mark.parametrize("covariance", ["sweight", "sumw2"])
def test_fit_session_sumw2_excludes_fixed_constraint_from_variability(covariance):
    model = _model()
    constraint = GaussianConstraint(model.parameters[0], mean=1.0, sigma=0.2)
    session = FitSession(model, _data(), constraints=(constraint,))
    result = session.fit(
        {"NR.x": 0.8},
        weights=jnp.asarray([1.0, -0.25]),
        covariance=covariance,
        strategy=1,
        hesse=False,
        hessian="jax",
        ncall=100,
    )

    assert result.valid
    # hessian="jax" in sWeight mode is postfit-only: MIGRAD must not receive
    # the signed-weight Hessian as its search curvature.
    assert result.nhessian == 0
    assert float(result.values["NR.x"]) == pytest.approx(1.0, abs=1e-5)
    # The normalized one-component event PDF is independent of its global
    # coefficient. The fixed Gaussian penalty identifies the point and enters
    # A, but it is not an event fluctuation and therefore does not enter H_w2.
    assert float(result.covariance["NR.x", "NR.x"]) == pytest.approx(0.0, abs=1e-14)
    assert float(result.errors["NR.x"]) == pytest.approx(0.0, abs=1e-7)


def test_score_outer_masks_zero_weight_at_physical_pdf_zero():
    class ZeroSupportSession(FitSession):
        @property
        def parameters(self):
            return (Parameter("mu", 0.0, step=0.1),)

        def _cached_signal_logpdf(self, data, parameters):
            finite = -0.5 * (jnp.asarray(data["s12"]) - parameters["mu"]) ** 2
            return finite.at[0].set(-jnp.inf)

        def _weighted_objective(self, weights):
            from jaxpwa.likelihood import WeightedUnbinnedNLL

            return WeightedUnbinnedNLL(
                self._cached_signal_logpdf, self.data.as_dict(), weights
            )

    data = PhaseSpaceSample(
        s12=jnp.asarray([0.0, 1.0]),
        s13=jnp.asarray([0.2, 0.2]),
        s23=jnp.asarray([1.0, 1.0]),
        weights=jnp.ones(2),
    )
    session = ZeroSupportSession(model=None, data=data)
    objective = session._score_outer_objective(
        jnp.asarray([0.0, 1.0]), {"mu": 0.0}
    )
    assert float(objective({"mu": 0.0})) == 0.0
    assert float(jax.hessian(lambda mu: objective({"mu": mu}))(0.0)) == 1.0


def test_corrected_covariance_rejects_indefinite_matrix():
    from types import SimpleNamespace

    from jaxpwa.workflow import _install_minuit_covariance

    result = SimpleNamespace(parameters=("x", "y"))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _install_minuit_covariance(result, ("x", "y"), [[1.0, 2.0], [2.0, 1.0]])


def test_fit_session_sandwich_covariance_matches_weighted_mean_formula():
    class GaussianLocationSession(FitSession):
        @property
        def parameters(self):
            return (Parameter("mu", 0.0, step=0.1),)

        def _cached_signal_logpdf(self, data, parameters):
            return -0.5 * (jnp.asarray(data["s12"]) - parameters["mu"]) ** 2

        def _weighted_objective(self, weights):
            from jaxpwa.likelihood import WeightedUnbinnedNLL

            return WeightedUnbinnedNLL(
                self._cached_signal_logpdf,
                self.data.as_dict(),
                weights,
            )

    data = PhaseSpaceSample(
        s12=jnp.asarray([0.0, 1.0, 2.0]),
        s13=jnp.asarray([0.2, 0.2, 0.2]),
        s23=jnp.asarray([1.0, 1.0, 1.0]),
        weights=jnp.ones(3),
    )
    weights = jnp.asarray([1.0, 1.0, -0.2])
    session = GaussianLocationSession(model=None, data=data)
    result = session.fit(
        {"mu": 0.0},
        weights=weights,
        covariance="sandwich",
        strategy=1,
        hesse=False,
        hessian="jax",
        ncall=100,
    )

    expected_mu = float(jnp.sum(weights * data.s12) / jnp.sum(weights))
    score = np.asarray(data.s12) - expected_mu
    expected_variance = float(
        np.sum(np.asarray(weights) ** 2 * score**2) / float(jnp.sum(weights)) ** 2
    )

    assert result.valid
    assert result.nhessian == 0
    assert float(result.values["mu"]) == pytest.approx(expected_mu, abs=1e-7)
    assert float(result.covariance["mu", "mu"]) == pytest.approx(
        expected_variance, rel=1e-6
    )


@pytest.mark.parametrize("covariance", ["sandwich", "sumw2"])
def test_weighted_covariance_with_constraint_matches_closed_form(covariance):
    # NLL = sum_i w_i (x_i-mu)^2/2 + (mu-m0)^2/(2 s^2). The constraint enters
    # the sensitivity A = W + 1/s^2 but never the event variability B.
    mu = Parameter("mu", 0.0, step=0.1)
    m0, sigma = 0.5, 0.7
    constraint = GaussianConstraint(mu, mean=m0, sigma=sigma)

    class ConstrainedGaussianLocationSession(FitSession):
        @property
        def parameters(self):
            return (mu,)

        def _cached_signal_logpdf(self, data, parameters):
            return -0.5 * (jnp.asarray(data["s12"]) - parameters["mu"]) ** 2

        def _weighted_nll(self, weights):
            from jaxpwa.likelihood import WeightedUnbinnedNLL

            return WeightedUnbinnedNLL(
                self._cached_signal_logpdf, self.data.as_dict(), weights
            )

    x = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 0.8, -0.2, 0.5])
    data = PhaseSpaceSample(
        s12=jnp.asarray(x),
        s13=jnp.full(4, 0.2),
        s23=jnp.ones(4),
        weights=jnp.ones(4),
    )
    session = ConstrainedGaussianLocationSession(
        model=None, data=data, constraints=(constraint,)
    )
    result = session.fit(
        {"mu": 0.0},
        weights=jnp.asarray(w),
        covariance=covariance,
        strategy=1,
        hesse=False,
        hessian="jax",
        ncall=100,
    )

    sensitivity = w.sum() + 1.0 / sigma**2
    expected_mu = (np.sum(w * x) + m0 / sigma**2) / sensitivity
    if covariance == "sumw2":
        variability = np.sum(w**2)
    else:
        variability = np.sum(w**2 * (x - expected_mu) ** 2)

    assert result.valid
    assert float(result.values["mu"]) == pytest.approx(expected_mu, abs=1e-7)
    assert float(result.covariance["mu", "mu"]) == pytest.approx(
        variability / sensitivity**2, rel=1e-6
    )


def test_fit_session_sweight_covariance_requires_weights():
    session = FitSession(_model(), _data())
    with pytest.raises(ValueError, match="requires event weights"):
        session.fit(covariance="sweight")


@pytest.mark.parametrize("covariance", ["sandwich", "sumw2", "sweight"])
def test_invalid_weighted_fit_skips_postfit_covariance(monkeypatch, covariance):
    from types import SimpleNamespace

    from jaxpwa.fit import Minimizer

    failed = SimpleNamespace(valid=False, fval=123.0, covariance=None)
    monkeypatch.setattr(Minimizer, "fit", lambda *args, **kwargs: failed)

    def forbidden_hessian(*args, **kwargs):
        pytest.fail("An invalid fit must not evaluate postfit Hessians")

    monkeypatch.setattr(Minimizer, "jax_hessian", forbidden_hessian)
    session = FitSession(_model(), _data())
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = session.fit(weights=jnp.asarray([1.0, -0.25]), covariance=covariance)
    assert result is failed
    assert not result.valid
    assert result.covariance is None


@pytest.mark.parametrize("covariance", ["sandwich", "sumw2"])
def test_weighted_fit_returning_nesterov_endpoint_skips_postfit_covariance(
    monkeypatch, covariance
):
    from jaxpwa.fit import Minimizer
    from jaxpwa.fit.nesterov import NesterovResult

    # nesterov-minuit falls back to the Nesterov endpoint when the Minuit
    # continuation worsens the NLL. That endpoint has no Minuit parameter
    # list or covariance, so the session must warn instead of crashing.
    endpoint = NesterovResult(
        values={"NR.x": 0.9}, fval=1.0, valid=True, status="converged",
        converged=True, history=(), nfcn=1, errors={"NR.x": float("nan")},
    )
    monkeypatch.setattr(Minimizer, "fit", lambda *args, **kwargs: endpoint)

    def forbidden_hessian(*args, **kwargs):
        pytest.fail("A Nesterov endpoint must not evaluate postfit Hessians")

    monkeypatch.setattr(Minimizer, "jax_hessian", forbidden_hessian)
    session = FitSession(_model(), _data())
    with pytest.warns(RuntimeWarning, match="Nesterov endpoint"):
        result = session.fit(
            weights=jnp.asarray([1.0, -0.25]),
            covariance=covariance,
            method="nesterov-minuit",
        )
    assert result is endpoint
    assert result.covariance is None


def test_fit_session_weighted_fit_rejects_explicit_background_mixture():
    model = _model()
    session = FitSession(
        model,
        _data(),
        backgrounds=(BackgroundSpec("comb", lambda d: jnp.ones_like(d["s12"])),),
        signal_fraction=0.8,
    )
    with pytest.raises(ValueError, match="signal-only FitSession"):
        session.fit(weights=jnp.ones(session.data.size))


def test_fit_session_rejects_unknown_covariance_mode():
    session = FitSession(_model(), _data())
    with pytest.raises(ValueError, match="covariance must be"):
        session.fit(covariance="bootstrap")
