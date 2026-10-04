"""Generated CP projections preserve fit yields, cuts and rendering options."""

import gc
import inspect
import weakref
from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxpwa import (
    CPFitSession,
    CPProjectionToy,
    CPRealImag,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    PhaseSpaceMC,
    YieldAsymmetry,
)


@pytest.fixture
def setup():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    normalization = PhaseSpaceMC(channel.parent_mass, channel.daughter_masses).generate(
        512,
        seed=7,
        include_momenta=False,
    )
    cp = CPRealImag(Parameter.coefficient("x", 1.0), 0.0, 0.2, 0.0)
    models = [
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_sample=normalization,
            normalize_components=False,
        )
        for q in (1, -1)
    ]
    data = normalization.take(jnp.arange(50))
    session = CPFitSession(
        *models,
        data,
        data,
        extended=True,
        signal_yield=YieldAsymmetry(100.0, 0.2),
    ).with_background(
        "comb",
        lambda d: jnp.ones_like(d["s12"]),
        minus_shape=lambda d: 3.0 * jnp.ones_like(d["s12"]),
        yield_=20.0,
    )
    return session, SimpleNamespace(values={"x": 1.0})


@pytest.fixture
def fake_generator(monkeypatch, setup):
    session, _ = setup
    calls = []

    def generate(model, size, **kwargs):
        calls.append(kwargs)
        source = session.plus_data
        return source.take(jnp.arange(size) % source.size)

    monkeypatch.setattr("jaxpwa.projection_toys.generate_toy", generate)
    return calls


def test_all_existing_projection_options_are_available():
    old = set(inspect.signature(CPFitSession.plot_projection).parameters)
    new = set(inspect.signature(CPFitSession.plot_projection_from_toy).parameters)
    assert old <= new


def test_standalone_charge_yields_acceptance_and_background_veto(setup, fake_generator):
    session, result = setup

    def efficiency(d):
        return jnp.full_like(d["s12"], 0.8)

    def veto(d):
        return d["s13"] > 2.0

    session = session.with_efficiency(efficiency).with_veto(veto)
    session = replace(
        session, backgrounds=(replace(session.backgrounds[0], apply_veto=False),)
    )
    toy = session.prepare_projection_toy(result, projection_size=101)
    assert isinstance(toy, CPProjectionToy)
    for components, yields in (
        (toy.plus_components, [40, 5]),
        (toy.minus_components, [60, 15]),
    ):
        assert sum(sample.size for _, sample, _ in components) == 101
        np.testing.assert_allclose(
            [weights.sum() for _, _, weights in components], yields
        )
        assert all(isinstance(sample.s12, np.ndarray) for _, sample, _ in components)
    assert all(
        c["efficiency"] is efficiency and c["veto"] is veto for c in fake_generator
    )
    assert all(c["include_momenta"] is False for c in fake_generator)
    assert all(
        not c["backgrounds"][0].apply_veto for c in fake_generator if c["backgrounds"]
    )


def test_shared_signal_yield_keeps_joint_amplitude_split(setup, fake_generator):
    session, result = setup
    session = replace(session, signal_yield=100.0)
    toy = session.prepare_projection_toy(result, projection_size=100)
    probability = 1.2**2 / (1.2**2 + 0.8**2)
    assert toy.plus_components[0][2].sum() == pytest.approx(100 * probability)
    assert toy.minus_components[0][2].sum() == pytest.approx(100 * (1 - probability))


def test_nonextended_mixture_uses_total_data_count(setup, fake_generator):
    session, result = setup
    session = replace(
        session,
        extended=False,
        signal_yield=None,
        signal_fraction=0.7,
        backgrounds=(replace(session.backgrounds[0], yield_=None),),
    )
    toy = session.prepare_projection_toy(result, projection_size=100)
    assert sum(
        w.sum() for c in (toy.plus_components, toy.minus_components) for _, _, w in c
    ) == pytest.approx(100)
    assert toy.plus_components[1][2].sum() == pytest.approx(7.5)
    assert toy.minus_components[1][2].sum() == pytest.approx(22.5)


def test_zero_signal_is_not_generated_and_zero_charge_plots(setup, fake_generator):
    session, result = setup
    session = replace(session, signal_yield=YieldAsymmetry(0.0, 0.0))
    toy = session.prepare_projection_toy(result, projection_size=50)
    assert all(c[0] == "comb" for c in toy.plus_components)
    assert len(fake_generator) == 2
    assert all(c["signal_fraction"] == 0 for c in fake_generator)
    zero = replace(session, backgrounds=())
    empty = zero.prepare_projection_toy(result, projection_size=50)
    assert empty.plus_components == empty.minus_components == ()
    axes = zero.plot_projection_from_toy(result, projection_toy=empty, show_pulls=True)
    np.testing.assert_array_equal(axes[0, 0].patches[0].get_data().values, 0)
    plt.close("all")


def test_reused_toy_fold_selection_weights_errors_and_axes(setup, fake_generator):
    session, result = setup
    toy = session.prepare_projection_toy(result, projection_size=100)
    calls_before = len(fake_generator)
    edges = np.array([0.0, 2.0, 5.0, 10.0, 30.0])

    def selection(d):
        return np.minimum(d["s13"], d["s23"]) > 2.0

    grid = session.plot_projection_from_toy(
        result,
        "s13",
        bins=edges,
        range=(0, 30),
        projection_toy=toy,
        folded=True,
        partner_variable="s23",
        fold_side="high",
        selection=selection,
        show_pulls=True,
        log_scale=True,
        show_components=True,
    )
    assert grid.shape == (2, 2)
    for col, components in enumerate((toy.plus_components, toy.minus_components)):
        expected = np.zeros(4)
        variance = np.zeros(4)
        full = 0
        for _, sample, weights in components:
            mask = selection(sample.as_dict())
            values = np.maximum(sample.s13, sample.s23)[mask]
            expected += np.histogram(values, edges, weights=weights[mask])[0]
            variance += np.histogram(values, edges, weights=weights[mask] ** 2)[0]
            full += weights.sum()
        ax = grid[0, col]
        np.testing.assert_allclose(ax.patches[-1].get_data().values, expected)
        assert expected.sum() < full  # selected region was not renormalized
        assert ax.get_yscale() == "log"
        assert len(ax.patches) == 3  # signal, comb, total
        data = session.plus_data
        observed = np.histogram(
            np.maximum(data.s13, data.s23)[selection(data.as_dict())], edges
        )[0]
        occupied = expected > 0
        pulls = (observed[occupied] - expected[occupied]) / np.sqrt(
            expected[occupied] + variance[occupied]
        )
        drawn = np.array([p.get_height() for p in grid[1, col].containers[0]])
        np.testing.assert_allclose(drawn[occupied], pulls)
    _, axes = plt.subplots(1, 2)
    returned = session.plot_projection_from_toy(
        result,
        projection_toy=toy,
        axes=axes,
        show_components=False,
        include_toy_uncertainty=False,
        folded=True,
        partner_variable="s23",
        fold_side="low",
    )
    assert returned is axes
    assert len(axes[0].patches) == 1
    assert len(fake_generator) == calls_before
    plt.close("all")


def test_reuse_rejects_stale_values_or_different_session(setup, fake_generator):
    session, result = setup
    toy = session.prepare_projection_toy(result, projection_size=50)
    with pytest.raises(ValueError, match="fitted values changed"):
        session.plot_projection_from_toy(
            SimpleNamespace(values={"x": 1.1}), projection_toy=toy
        )
    with pytest.raises(ValueError, match="different fit session"):
        replace(session).plot_projection_from_toy(result, projection_toy=toy)
    assert plt.get_fignums() == []


def test_reuse_rejects_generation_options(setup, fake_generator):
    session, result = setup
    toy = session.prepare_projection_toy(result, projection_size=50)
    with pytest.raises(TypeError, match="ignored with projection_toy"):
        session.plot_projection_from_toy(
            result, projection_toy=toy, inverse_resolution=100
        )
    assert plt.get_fignums() == []


def test_toy_does_not_keep_session_alive(setup, fake_generator):
    session, result = setup
    clone = replace(session)
    ref = weakref.ref(clone)
    toy = clone.prepare_projection_toy(result, projection_size=50)
    del clone
    gc.collect()
    assert ref() is None
    assert toy.plus_components[0][1].size > 0


@pytest.mark.parametrize(
    "options,match",
    [
        ({"projection_size": 0}, "positive integer"),
        ({"projection_size": True}, "positive integer"),
        ({"projection_size": 1}, "one toy"),
        ({"method": "bad"}, "method"),
        ({"pool_size": 100}, "accept-reject"),
        ({"unsupported": 1}, "unsupported toy options"),
    ],
)
def test_invalid_generation_options(setup, options, match):
    session, result = setup
    with pytest.raises((ValueError, TypeError), match=match):
        session.prepare_projection_toy(result, **options)


@pytest.mark.parametrize("method", ["inverse-transform", "accept-reject"])
def test_real_samplers_respect_signal_veto_and_projection_closure(setup, method):
    session, result = setup
    session = replace(session, backgrounds=())
    session = session.with_veto(lambda d: d["s13"] > 3.0)
    session = session.with_efficiency(lambda d: 0.2 + 0.02 * d["s13"])
    options = (
        {"inverse_resolution": 96}
        if method == "inverse-transform"
        else {"pool_size": 1024, "batch_size": 1024}
    )
    toy = session.prepare_projection_toy(
        result, projection_size=2000, method=method, **options
    )
    for components in (toy.plus_components, toy.minus_components):
        sample = components[0][1]
        assert sample.size == 2000
        assert sample.p1 is None
        assert np.all(sample.s13 > 3.0)
        assert np.unique(sample.s13).size == sample.size
    # Independent weighted integration sample predicts the generated s13 mean.
    model = session.plus_model
    reference = model.generate_phase_space(20_000, seed=22)
    density = np.asarray(session.plus_efficiency(reference.as_dict())) * (
        np.asarray(reference.s13) > 3.0
    )
    expected = np.average(
        reference.s13, weights=np.asarray(reference.weights) * density
    )
    generated = toy.plus_components[0][1].s13
    assert abs(np.mean(generated) - expected) < 6 * np.std(generated) / np.sqrt(
        generated.size
    )
    plt.close("all")
