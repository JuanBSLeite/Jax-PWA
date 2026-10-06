from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    BinnedChi2Result,
    CPBackgroundSpec,
    CPFitSession,
    CPRealImag,
    DalitzAmplitude,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    PointToPointResult,
    PolarFormFactorSymNR,
    RealImag,
    generate_signal_toy,
)


def _models():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    dx = Parameter.coefficient("NR.dx", 0.1, bounds=(-0.5, 0.5), owner="NR")
    cp = CPRealImag(x, 0.0, dx, 0.0)
    plus = DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    minus = DecayModel(
        DecayChannel("B-", ("K-", "pi-", "pi+")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    return plus, minus


def _data(offset=0.0):
    return PhaseSpaceSample(
        s12=jnp.asarray([2.0 + offset, 2.3 + offset]),
        s13=jnp.asarray([1.2 + offset, 1.5 + offset]),
        s23=jnp.asarray([4.0 - offset, 3.7 - offset]),
        weights=jnp.ones(2),
    )


def test_cp_fit_session_collects_shared_parameters_once():
    plus, minus = _models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    assert [p.name for p in session.parameters] == ["NR.x", "NR.dx"]
    value = session.objective({"NR.x": 1.0, "NR.dx": 0.1})
    assert jnp.isfinite(value)


def test_cp_fit_session_fit_update_model_returns_a_model_pair_with_fitted_values():
    plus, minus = _models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))

    result, plus_model, minus_model = session.fit(
        {"NR.x": 1.0, "NR.dx": 0.1}, simplex=False, ncall=100, update_model=True
    )

    assert result.valid
    fitted_x = float(result.values["NR.x"])
    plus_x = next(p for p in plus_model.parameters if p.name == "NR.x")
    minus_x = next(p for p in minus_model.parameters if p.name == "NR.x")
    assert plus_x.value == pytest.approx(fitted_x)
    assert minus_x is plus_x
    # the session's own models are frozen dataclasses and are left untouched
    assert next(p for p in session.plus_model.parameters if p.name == "NR.x").value == 1.0


def test_cp_fit_session_automatically_builds_joint_background():
    plus, minus = _models()
    fraction = Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0))
    background = CPBackgroundSpec(
        "comb",
        lambda d: jnp.ones_like(d["s12"]),
    )
    session = CPFitSession(
        plus,
        minus,
        _data(),
        _data(0.02),
        backgrounds=(background,),
        signal_fraction=fraction,
    )
    category = session.background_categories[0]
    expected_plus = jnp.mean(plus.normalization_sample.weights)
    expected_minus = jnp.mean(minus.normalization_sample.weights)
    assert jnp.allclose(category.plus_normalization, expected_plus)
    assert jnp.allclose(category.minus_normalization, expected_minus)
    assert jnp.isfinite(
        session.objective({"NR.x": 1.0, "NR.dx": 0.1, "signal_fraction": 0.8})
    )


def test_cp_fit_session_symmetric_efficiency_is_convenient():
    plus, minus = _models()
    efficiency = lambda d: 0.7 + 0.0 * d["s12"]
    session = CPFitSession(plus, minus, _data(), _data(0.02)).with_efficiency(efficiency)
    assert session.plus_efficiency is efficiency
    assert session.minus_efficiency is efficiency
    assert jnp.allclose(session.plus_acceptance_data, 0.7)
    assert jnp.allclose(session.minus_acceptance_data, 0.7)


def test_cp_projection_weights_preserve_joint_event_count():
    plus, minus = _models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    values = {"NR.x": 1.0, "NR.dx": 0.1}
    plus_components = session._projection_components(values, "plus")
    minus_components = session._projection_components(values, "minus")
    total = jnp.sum(jnp.asarray(plus_components[0][2])) + jnp.sum(jnp.asarray(minus_components[0][2]))
    assert jnp.allclose(total, session.plus_data.size + session.minus_data.size, rtol=1e-6)


def _toy_cp_session(n_events=300, seed=11):
    plus, minus = _models()
    plus_data = generate_signal_toy(plus, n_events, seed=seed)
    minus_data = generate_signal_toy(minus, n_events, seed=seed + 1)
    return CPFitSession(plus, minus, plus_data, minus_data)


def _toy_cp_result():
    return SimpleNamespace(values={"NR.x": 1.0, "NR.dx": 0.1}, fixed=None)


def test_cp_goodness_of_fit_projection_returns_both_charges_by_default():
    session = _toy_cp_session()
    result = _toy_cp_result()
    gof = session.goodness_of_fit_projection(result, "s13", bins=8)
    assert set(gof) == {"plus", "minus"}
    for charge_result in gof.values():
        assert isinstance(charge_result, BinnedChi2Result)
        assert charge_result.n_free_parameters == 2
        assert np.isfinite(charge_result.chi2)


def test_cp_goodness_of_fit_projection_single_charge():
    session = _toy_cp_session()
    result = _toy_cp_result()
    plus_only = session.goodness_of_fit_projection(result, "s13", bins=8, charge="plus")
    assert isinstance(plus_only, BinnedChi2Result)
    with pytest.raises(ValueError, match="charge"):
        session.goodness_of_fit_projection(result, "s13", charge="neutral")


def test_cp_goodness_of_fit_chi2_2d_returns_both_charges():
    session = _toy_cp_session()
    result = _toy_cp_result()
    gof = session.goodness_of_fit_chi2(result, bins=5)
    assert set(gof) == {"plus", "minus"}
    assert gof["plus"].pulls.ndim == 2


def test_cp_point_to_point_dissimilarity_requires_charge():
    session = _toy_cp_session(n_events=200)
    result = _toy_cp_result()
    with pytest.raises(TypeError):
        session.point_to_point_dissimilarity(result)
    with pytest.raises(ValueError, match="charge"):
        session.point_to_point_dissimilarity(result, charge="neutral")

    plus_result = session.point_to_point_dissimilarity(
        result, charge="plus", mc_size=400, n_permutations=30
    )
    assert isinstance(plus_result, PointToPointResult)
    assert plus_result.n_data == session.plus_data.size
    assert 0.0 <= plus_result.p_value <= 1.0


def test_cp_plot_projection_show_pulls_returns_2x2_grid():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    grid = session.plot_projection(
        _toy_cp_result(), "s13", bins=10, show_pulls=True
    )
    assert grid.shape == (2, 2)
    assert grid[0, 0].get_shared_x_axes().joined(grid[0, 0], grid[1, 0])
    assert grid[0, 1].get_shared_x_axes().joined(grid[0, 1], grid[1, 1])
    plt.close("all")


def test_cp_plot_projection_default_return_unchanged_by_show_pulls_option():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    axes = session.plot_projection(_toy_cp_result(), "s13", bins=10)
    assert axes.shape == (2,)
    plt.close("all")


def test_cp_plot_projection_show_pulls_rejects_explicit_axes():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="show_pulls=True"):
        session.plot_projection(_toy_cp_result(), "s13", show_pulls=True, axes=axes)
    plt.close("all")


# --- Event-weighted (sWeight/COW) CP fits -----------------------------------


def _fixed_scale_models():
    """NR-only CP models with x fixed: only dx (the charge split) is fitted."""
    x = Parameter.coefficient("NR.x", 1.0, fixed=True, owner="NR")
    dx = Parameter.coefficient("NR.dx", 0.0, bounds=(-0.9, 0.9), owner="NR")
    cp = CPRealImag(x, 0.0, dx, 0.0)
    models = []
    for parent, daughters, charge in (
        ("B+", ("K+", "pi+", "pi-"), +1),
        ("B-", ("K-", "pi-", "pi+"), -1),
    ):
        models.append(DecayModel(
            DecayChannel(parent, daughters),
            [NonResonant(cp.for_charge(charge))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
            normalization_pair=(0, 2),
        ))
    return tuple(models)


def _charge_log_probabilities(dx):
    """log S_+/S_- up to an event-independent constant, for NR-only models.

    With identical B+/B- normalization grids, I_q = (1 + q dx)^2 I_0, so the
    jointly normalized density of a charge-q event is (1 + q dx)^2 / D up to
    the constant 1/I_0, with D = (1 + dx)^2 + (1 - dx)^2.
    """
    total = (1.0 + dx) ** 2 + (1.0 - dx) ** 2
    return np.log((1.0 + dx) ** 2 / total), np.log((1.0 - dx) ** 2 / total)


def _derivatives(function, dx, step=1e-4):
    plus0, minus0 = function(dx)
    plus_up, minus_up = function(dx + step)
    plus_down, minus_down = function(dx - step)
    first = ((plus_up - plus_down) / (2 * step), (minus_up - minus_down) / (2 * step))
    second = (
        (plus_up - 2 * plus0 + plus_down) / step**2,
        (minus_up - 2 * minus0 + minus_down) / step**2,
    )
    return first, second


PLUS_WEIGHTS = np.asarray([1.0, -0.3])
MINUS_WEIGHTS = np.asarray([0.9, 0.6])


def test_weighted_cp_joint_nll_matches_manual_weighted_sum():
    from jaxpwa.likelihood import CPJointNLL

    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    values = {"NR.x": 1.0, "NR.dx": 0.2}
    weighted = CPJointNLL(
        session.plus_cache, session.minus_cache,
        plus_weights=jnp.asarray(PLUS_WEIGHTS), minus_weights=jnp.asarray(MINUS_WEIGHTS),
    )
    unweighted = CPJointNLL(session.plus_cache, session.minus_cache)
    log_plus, log_minus = weighted.signal_log_densities(values)

    expected = -np.sum(PLUS_WEIGHTS * np.asarray(log_plus)) - np.sum(MINUS_WEIGHTS * np.asarray(log_minus))
    assert float(weighted(values)) == pytest.approx(expected, rel=1e-12)
    # Unit weights reproduce the ordinary (non-extended, signal-only) joint NLL.
    unit = CPJointNLL(
        session.plus_cache, session.minus_cache,
        plus_weights=jnp.ones(2), minus_weights=jnp.ones(2),
    )
    assert float(unit(values)) == pytest.approx(float(unweighted(values)), rel=1e-12)


@pytest.mark.parametrize("covariance", ["sandwich", "sumw2"])
def test_weighted_cp_fit_matches_weighted_bernoulli_closed_form(covariance):
    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    result = session.fit(
        {"NR.dx": 0.0},
        weights=(jnp.asarray(PLUS_WEIGHTS), jnp.asarray(MINUS_WEIGHTS)),
        covariance=covariance,
        strategy=1,
        hesse=False,
        hessian="jax",
        ncall=500,
        tolerance=1e-7,  # the closed-form comparison needs a tight EDM
    )

    # Joint normalization makes the charge a weighted Bernoulli variable:
    # p_+ = W+ / (W+ + W-)  =>  dx = (r - 1) / (r + 1), r = sqrt(W+ / W-).
    total_plus, total_minus = PLUS_WEIGHTS.sum(), MINUS_WEIGHTS.sum()
    ratio = np.sqrt(total_plus / total_minus)
    expected_dx = (ratio - 1.0) / (ratio + 1.0)

    (score_plus, score_minus), (curv_plus, curv_minus) = _derivatives(
        _charge_log_probabilities, expected_dx
    )
    estimating_hessian = -(total_plus * curv_plus + total_minus * curv_minus)
    if covariance == "sandwich":
        variability = (
            np.sum(PLUS_WEIGHTS**2) * score_plus**2
            + np.sum(MINUS_WEIGHTS**2) * score_minus**2
        )
    else:
        variability = -(
            np.sum(PLUS_WEIGHTS**2) * curv_plus + np.sum(MINUS_WEIGHTS**2) * curv_minus
        )
    expected_variance = variability / estimating_hessian**2

    assert result.valid
    assert result.nhessian == 0  # signed-weight Hessian is postfit-only
    assert float(result.values["NR.dx"]) == pytest.approx(expected_dx, abs=1e-6)
    assert float(result.covariance["NR.dx", "NR.dx"]) == pytest.approx(expected_variance, rel=1e-5)
    assert float(result.errors["NR.dx"]) == pytest.approx(np.sqrt(expected_variance), rel=1e-5)


def test_unit_weight_cp_fit_reproduces_unweighted_fit():
    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    start = {"NR.dx": 0.1}
    unweighted = session.fit(start, strategy=1, ncall=500)
    weighted = session.fit(
        start, weights=(jnp.ones(2), jnp.ones(2)), covariance="minuit", strategy=1, ncall=500,
    )
    assert weighted.valid and unweighted.valid
    assert float(weighted.values["NR.dx"]) == pytest.approx(float(unweighted.values["NR.dx"]), abs=1e-6)
    assert float(weighted.errors["NR.dx"]) == pytest.approx(float(unweighted.errors["NR.dx"]), rel=1e-4)


def test_weighted_cp_fit_rejects_background_configuration():
    plus, minus = _fixed_scale_models()
    session = CPFitSession(
        plus, minus, _data(), _data(0.02),
        backgrounds=(CPBackgroundSpec("comb", lambda d: jnp.ones_like(d["s12"])),),
        signal_fraction=0.8,
    )
    with pytest.raises(ValueError, match="signal-only CPFitSession"):
        session.fit(weights=(jnp.ones(2), jnp.ones(2)))


def test_weighted_cp_fit_validates_weight_pair():
    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    with pytest.raises(TypeError, match="pair"):
        session.fit(weights=jnp.ones(4))
    with pytest.raises(ValueError, match="minus weights must have shape"):
        session.fit(weights=(jnp.ones(2), jnp.ones(3)))
    with pytest.raises(ValueError, match="requires event weights"):
        session.fit(covariance="sandwich")
    with pytest.raises(ValueError, match="plus weights must be real"):
        session.fit(weights=(jnp.ones(2, dtype=complex), jnp.ones(2)))


def test_cp_joint_nll_rejects_one_sided_efficiency_array():
    from jaxpwa.likelihood import CPJointNLL

    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    with pytest.raises(ValueError, match="must be supplied together"):
        CPJointNLL(
            session.plus_cache,
            session.minus_cache,
            plus_efficiency=jnp.ones(2),
            minus_efficiency=None,
        )


def test_weighted_cp_joint_nll_rejects_non_finite_weights_and_background():
    from jaxpwa.likelihood import CPJointNLL

    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    with pytest.raises(ValueError, match="supplied together"):
        CPJointNLL(session.plus_cache, session.minus_cache, plus_weights=jnp.ones(2))
    with pytest.raises(ValueError, match="must be finite"):
        CPJointNLL(
            session.plus_cache, session.minus_cache,
            plus_weights=jnp.asarray([1.0, jnp.nan]), minus_weights=jnp.ones(2),
        )
    with pytest.raises(ValueError, match="signal-only"):
        CPJointNLL(
            session.plus_cache, session.minus_cache,
            plus_background=jnp.ones(2), minus_background=jnp.ones(2),
            plus_background_normalization=1.0, minus_background_normalization=1.0,
            signal_fraction=0.5,
            plus_weights=jnp.ones(2), minus_weights=jnp.ones(2),
        )


def test_with_event_weights_session_fits_and_projects_weighted_data():
    import matplotlib.pyplot as plt

    plus, minus = _fixed_scale_models()
    plain = CPFitSession(plus, minus, _data(), _data(0.02))
    weights = (jnp.asarray(PLUS_WEIGHTS), jnp.asarray(MINUS_WEIGHTS))
    session = plain.with_event_weights(*weights)
    values = {"NR.x": 1.0, "NR.dx": 0.2}

    # The session objective is the weighted one, and fit() uses the session
    # weights by default.
    assert float(session.objective(values)) == pytest.approx(
        float(plain._weighted_objective(weights)(values)), rel=1e-12
    )
    implicit = session.fit({"NR.dx": 0.0}, covariance="sandwich", strategy=1, ncall=500, tolerance=1e-7)
    explicit = plain.fit(
        {"NR.dx": 0.0}, weights=weights, covariance="sandwich", strategy=1, ncall=500, tolerance=1e-7,
    )
    assert float(implicit.values["NR.dx"]) == pytest.approx(float(explicit.values["NR.dx"]), abs=1e-8)
    assert float(implicit.errors["NR.dx"]) == pytest.approx(float(explicit.errors["NR.dx"]), rel=1e-6)

    # Signal-only projection: the model is scaled to the weighted sums, split
    # between charges by I_+/I_-; the data histograms are weighted.
    plus_components, minus_components = session._projection_components(values, "plus"), session._projection_components(values, "minus")
    model_total = sum(float(np.sum(w)) for _, _, w in plus_components) + sum(
        float(np.sum(w)) for _, _, w in minus_components
    )
    assert model_total == pytest.approx(PLUS_WEIGHTS.sum() + MINUS_WEIGHTS.sum(), rel=1e-10)
    axes = session.plot_projection(implicit, "s13", bins=4, range=(1.0, 1.8), projection_size=2000)
    data_heights = [float(np.nansum(line.get_ydata())) for line in axes[0].lines if line.get_linestyle() == "None"]
    assert PLUS_WEIGHTS.sum() == pytest.approx(sum(data_heights), rel=1e-12)
    plt.close("all")


def test_event_weighted_session_rejects_point_to_point_and_bad_weights():
    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02)).with_event_weights(jnp.ones(2), jnp.ones(2))
    result = session.fit({"NR.dx": 0.0}, strategy=1, ncall=200)
    with pytest.raises(NotImplementedError, match="event-weighted"):
        session.point_to_point_dissimilarity(result, charge="plus")
    with pytest.raises(ValueError, match="plus weights must have shape"):
        CPFitSession(plus, minus, _data(), _data(0.02)).with_event_weights(jnp.ones(3), jnp.ones(2))


def test_log_scale_projection_ignores_empty_bins_for_the_y_axis():
    import matplotlib.pyplot as plt

    plus, minus = _fixed_scale_models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    result = session.fit({"NR.dx": 0.0}, strategy=1, ncall=200)
    # The range extends far beyond phase space: those bins are empty for both
    # data and model, as bins inside a veto are.
    axes = session.plot_projection_from_toy(
        result, "s13", bins=20, range=(0.5, 40.0), log_scale=True,
        projection_size=2000, include_toy_uncertainty=True,
    )
    for ax in axes:
        assert ax.get_ylim()[0] > 1e-6
    plt.close("all")


def test_amplitude_components_add_up_to_signal_with_interference():
    import matplotlib.pyplot as plt
    from jaxpwa import RealImag, Resonance

    def models():
        out = []
        for parent, daughters in (("B+", ("K+", "pi+", "pi-")), ("B-", ("K-", "pi-", "pi+"))):
            out.append(DecayModel(
                DecayChannel(parent, daughters),
                [
                    NonResonant(RealImag(1.0, 0.0)),
                    Resonance("rho", (1, 2), RealImag(0.5, 0.3), mass=0.775, width=0.149, spin=1),
                ],
                normalization_method="square-dalitz", normalization_resolution=12, normalization_pair=(0, 2),
            ))
        return out

    plus, minus = models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    result = SimpleNamespace(values={})
    axes = session.plot_projection(
        result, "s23", bins=8, range=(0.3, 6.0), projection_size=4000,
        show_amplitude_components=True,
    )
    for ax in axes:
        curves = {
            artist.get_label(): artist.get_data()[0]
            for artist in ax.patches if hasattr(artist, "get_data")
        }
        assert {"signal", "NR", "rho", "interference"} <= set(curves)
        assert np.allclose(curves["NR"] + curves["rho"] + curves["interference"], curves["signal"], atol=1e-9)
        assert np.all(curves["NR"] >= 0) and np.all(curves["rho"] >= 0)
    plt.close("all")

    single = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    axes = single.plot_projection(
        SimpleNamespace(values={"NR.dx": 0.1}), "s13", bins=6, range=(0.5, 20.0),
        projection_size=2000, show_amplitude_components=True,
    )
    curves = {a.get_label(): a.get_data()[0] for a in axes[0].patches if hasattr(a, "get_data")}
    assert np.allclose(curves["NR"], curves["signal"])
    assert np.allclose(curves["interference"], 0.0, atol=1e-9)
    plt.close("all")


def test_legend_panel_moves_the_legend_to_a_separate_column():
    import matplotlib.pyplot as plt

    session = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    result = SimpleNamespace(values={"NR.dx": 0.1})
    grid = session.plot_projection(
        result, "s13", bins=6, range=(0.5, 20.0), projection_size=2000,
        show_pulls=True, legend_panel=True, extra_panel=True,
    )
    figure = grid[0][0].figure
    assert grid.shape == (2, 2)
    assert all(ax.get_legend() is None for ax in grid[0])
    labels = [text.get_text() for text in figure.legend_axes.get_legend().get_texts()]
    assert "data" in labels and "signal" in labels
    assert figure.extra_axes is not None and figure.extra_axes is not figure.legend_axes
    plt.close("all")

    axes = session.plot_projection(result, "s13", bins=6, range=(0.5, 20.0), projection_size=2000, legend_panel=True)
    assert axes[0].figure.extra_axes is None
    assert axes[0].figure.legend_axes.get_legend() is not None
    plt.close("all")
    with pytest.raises(ValueError, match="legend_panel"):
        _, existing = plt.subplots(1, 2)
        session.plot_projection(result, "s13", axes=existing, legend_panel=True)
    plt.close("all")


def test_share_y_gives_both_charges_one_y_axis_labelled_on_the_left():
    import matplotlib.pyplot as plt

    session = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    result = SimpleNamespace(values={"NR.dx": 0.3})
    for legend_panel in (False, True):
        grid = session.plot_projection(
            result, "s13", bins=6, range=(0.5, 20.0), projection_size=2000,
            show_pulls=True, share_y=True, legend_panel=legend_panel, log_scale=True,
        )
        for row in grid:
            assert row[0].get_ylim() == row[1].get_ylim()
            assert row[0].get_ylabel() != "" and row[1].get_ylabel() == ""
            assert not any(label.get_visible() and label.get_text() for label in row[1].get_yticklabels())
        plt.close("all")


def test_show_interference_false_hides_the_interference_curve():
    import matplotlib.pyplot as plt

    session = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    axes = session.plot_projection(
        SimpleNamespace(values={"NR.dx": 0.1}), "s13", bins=6, range=(0.5, 20.0),
        projection_size=2000, show_amplitude_components=True, show_interference=False,
    )
    labels = {artist.get_label() for artist in axes[0].patches}
    assert "NR" in labels and "interference" not in labels
    plt.close("all")


def test_component_cp_asymmetries_coefficient_only_matches_analytic():
    session = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    dx, sigma = 0.1, 0.03
    result = SimpleNamespace(values={"NR.dx": dx}, covariance=np.asarray([[sigma**2]]))
    asymmetries = session.component_cp_asymmetries(result)
    entry = asymmetries["NR"]
    # |c+|^2 = (1 + dx)^2, |c-|^2 = (1 - dx)^2 on identical integration grids.
    assert entry["value"] == pytest.approx(-2.0 * dx / (1.0 + dx**2), rel=1e-10)
    slope = -2.0 * (1.0 - dx**2) / (1.0 + dx**2) ** 2
    assert entry["error"] == pytest.approx(abs(slope) * sigma, rel=1e-8)


def test_component_cp_asymmetries_include_cp_violation_in_the_dynamics():
    # Coefficient fixed and charge-symmetric: the only CP violation is a
    # per-charge dynamics parameter, as for the nodes of a QMI S-wave.
    lambdas = {
        charge: Parameter.dynamics(f"NR.lambda_{charge}", value, owner="NR", bounds=(0.1, 10.0))
        for charge, value in (("plus", 1.2), ("minus", 1.8))
    }
    models = tuple(
        DecayModel(
            DecayChannel(parent, daughters),
            [DalitzAmplitude("NR", PolarFormFactorSymNR(lambdas[charge]), RealImag(1.0, 0.0))],
            normalize_components=False,
            normalization_resolution=30,
        )
        for parent, daughters, charge in (
            ("B+", ("K+", "K+", "K-"), "plus"),
            ("B-", ("K-", "K-", "K+"), "minus"),
        )
    )
    data = models[0].generate_phase_space(50, seed=1)
    session = CPFitSession(*models, data, models[1].generate_phase_space(50, seed=2))
    values = {"NR.lambda_plus": 1.2, "NR.lambda_minus": 1.8}
    covariance = np.diag([0.02**2, 0.05**2])  # ordered like the sorted free names
    names = sorted(values)
    entry = session.component_cp_asymmetries(
        SimpleNamespace(values=values, covariance=covariance)
    )["NR"]

    def direct(point):
        integrals = []
        for model, name in zip(models, ("NR.lambda_plus", "NR.lambda_minus"), strict=True):
            sample = model.normalization_sample
            amplitude = PolarFormFactorSymNR(point[name])(sample.as_dict())
            integrals.append(float(jnp.mean(sample.weights * jnp.abs(amplitude) ** 2)))
        plus, minus = integrals
        return (minus - plus) / (minus + plus)

    assert entry["value"] > 0.1  # a larger lambda gives the larger intensity
    assert entry["value"] == pytest.approx(direct(values), rel=1e-10)
    step = 1e-5
    gradient = np.asarray([
        (direct({**values, name: values[name] + step}) - direct({**values, name: values[name] - step}))
        / (2 * step)
        for name in names
    ])
    expected = float(np.sqrt(gradient @ covariance @ gradient))
    assert entry["error"] == pytest.approx(expected, rel=1e-5)


def test_component_cp_asymmetries_without_covariance_have_nan_errors():
    session = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    entry = session.component_cp_asymmetries(SimpleNamespace(values={"NR.dx": 0.1}, covariance=None))["NR"]
    assert np.isfinite(entry["value"]) and np.isnan(entry["error"])


def test_fixed_component_cp_asymmetries_with_only_yield_covariance():
    models = tuple(DecayModel(
        model.channel, [NonResonant(RealImag(1.0, 0.0))],
        normalization_resolution=12,
    ) for model in _fixed_scale_models())
    session = CPFitSession(*models, _data(), _data(0.02), extended=True,
                           signal_yield=Parameter("yield", 4.0))
    entry = session.component_cp_asymmetries(
        SimpleNamespace(values={"yield": 4.0}, covariance=np.asarray([[4.0]]))
    )["NR"]
    assert entry["value"] == pytest.approx(0.0, abs=1e-14)
    assert entry["error"] == 0.0


def test_event_weights_are_validated_before_building_caches():
    from dataclasses import replace

    plain = CPFitSession(*_fixed_scale_models(), _data(), _data(0.02))
    with pytest.raises(ValueError, match="must be finite"):
        plain.with_event_weights([1.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="supplied together"):
        replace(plain, plus_event_weights=[1.0, 1.0])
    background = plain.with_background("comb", lambda d: jnp.ones_like(d["s12"]))
    with pytest.raises(ValueError, match="signal-only"):
        background.with_event_weights([1.0, 1.0], [1.0, 1.0])
    assert "plus_cache" not in plain.__dict__


def test_amplitude_projection_retains_components_at_coherent_zeros():
    import matplotlib.pyplot as plt

    class Model:
        def component_amplitudes(self, data, values):
            return {"a": np.asarray([1.0, 1.0]), "b": np.asarray([-1.0, 1.0])}

    _, ax = plt.subplots()
    CPFitSession._draw_amplitude_components(
        ax, Model(), {}, _data(), np.asarray([0.0, 4.0]),
        np.asarray([True, True]), np.asarray([0.25, 0.75]),
        np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 4.0]), False,
        integration_weights=np.ones(2),
    )
    curves = {a.get_label(): a.get_data()[0] for a in ax.patches}
    np.testing.assert_allclose(curves["a"], [1.0, 1.0])
    np.testing.assert_allclose(curves["b"], [1.0, 1.0])
    np.testing.assert_allclose(curves["interference"], [-2.0, 2.0])
    plt.close("all")


def test_unit_weight_goodness_of_fit_uses_squared_weight_variance():
    plain = _toy_cp_session()
    result = _toy_cp_result()
    weighted = plain.with_event_weights(
        jnp.ones(plain.plus_data.size), jnp.ones(plain.minus_data.size)
    )
    for method, kwargs in (
        ("goodness_of_fit_projection", dict(variable="s13", bins=8)),
        ("goodness_of_fit_chi2", dict(bins=5)),
    ):
        reference = getattr(plain, method)(result, **kwargs)["plus"]
        unit = getattr(weighted, method)(result, **kwargs)["plus"]
        # Same histograms; unit weights give variance = observed counts
        # (Neyman), the unweighted test keeps the Pearson variance = expected.
        np.testing.assert_allclose(unit.observed, reference.observed)
        np.testing.assert_allclose(unit.expected, reference.expected, rtol=1e-12)
        used = (reference.expected > 0) & (reference.observed > 0)
        expected_chi2 = np.sum(
            (reference.observed[used] - reference.expected[used]) ** 2 / reference.observed[used]
        )
        assert unit.chi2 == pytest.approx(expected_chi2, rel=1e-12)
        assert reference.chi2 == pytest.approx(
            np.sum(
                (reference.observed[reference.expected > 0] - reference.expected[reference.expected > 0]) ** 2
                / reference.expected[reference.expected > 0]
            ),
            rel=1e-12,
        )


def test_weighted_goodness_of_fit_histograms_sum_weights():
    plain = _toy_cp_session()
    result = _toy_cp_result()
    rng = np.random.default_rng(3)
    plus_weights = rng.uniform(-0.5, 2.0, plain.plus_data.size)
    minus_weights = rng.uniform(-0.5, 2.0, plain.minus_data.size)
    weighted = plain.with_event_weights(jnp.asarray(plus_weights), jnp.asarray(minus_weights))
    gof = weighted.goodness_of_fit_projection(result, "s13", bins=6)["plus"]
    edges = gof.edges[0]
    values = np.asarray(plain.plus_data.s13)
    np.testing.assert_allclose(gof.observed, np.histogram(values, edges, weights=plus_weights)[0])
    variance = np.histogram(values, edges, weights=plus_weights**2)[0]
    used = (gof.expected > 0) & (variance > 0)
    np.testing.assert_allclose(
        gof.pulls[used], (gof.observed[used] - gof.expected[used]) / np.sqrt(variance[used])
    )
