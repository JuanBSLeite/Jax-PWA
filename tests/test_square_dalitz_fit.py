"""Fits with ``coordinates="square-dalitz"``: the events as points of (m', theta')."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import uproot

from jaxpwa import (
    QMI,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    FitSession,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    RelativisticBreitWigner,
    Resonance,
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
    ToyBackground,
    enable_x64,
    generate_toy,
    read_phase_space_sample,
    square_dalitz_log_jacobian,
)

enable_x64()

PLUS = DecayChannel("B+", ("pi-", "pi+", "pi+"))
MINUS = DecayChannel("B-", ("pi+", "pi-", "pi-"))
MOTHER = float(PLUS.parent_mass)
MASSES = tuple(float(m) for m in PLUS.daughter_masses)
PAIR = (1, 2)  # pi+ pi+: the Square-Dalitz pair of the models below


def _model(channel, rho, nr, *, method="square-dalitz", extra=()):
    return DecayModel(
        channel,
        [
            Resonance(
                "rho",
                (0, 1),
                rho,
                lineshape=RelativisticBreitWigner(),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(nr, name="nr"),
            *extra,
        ],
        normalize_components=True,
        normalization_method=method,
        normalization_pair=PAIR,
        normalization_resolution=120,
    )


def _single_model(method="square-dalitz"):
    nr = RealImag(
        Parameter.coefficient("nr.x", 0.6, owner="nr"),
        Parameter.coefficient("nr.y", -0.3, owner="nr"),
    )
    return _model(PLUS, RealImag(1.0, 0.0), nr, method=method)


def _cp_models(extra=lambda charge: ()):
    nr = CPRealImag(
        *(
            Parameter.coefficient(f"nr.{a}", v, owner="nr")
            for a, v in zip(
                ("x", "y", "dx", "dy"), (0.6, -0.3, 0.08, 0.04), strict=True
            )
        )
    )
    rho = CPRealImag(1.0, 0.0, Parameter.coefficient("rho.dx", 0.05, owner="rho"), 0.0)
    return tuple(
        _model(channel, rho.for_charge(q), nr.for_charge(q), extra=extra(q))
        for channel, q in ((PLUS, +1), (MINUS, -1))
    )


def _toy(model, size, seed):
    return generate_toy(
        model,
        size,
        seed=seed,
        method="inverse-transform",
        inverse_resolution=200,
        include_momenta=False,
    )


def _toy_with_efficiency_and_background(model, efficiency, background, size, seed):
    return generate_toy(
        model,
        size,
        seed=seed,
        method="accept-reject",
        efficiency=efficiency,
        signal_fraction=0.8,
        backgrounds=[ToyBackground("bkg", background)],
        include_momenta=False,
    )


def _assert_same_fit(dalitz, square, log_jacobian_sum):
    assert dalitz.valid and square.valid
    for name in dalitz.parameters:
        np.testing.assert_allclose(
            square.values[name], dalitz.values[name], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(square.errors[name], dalitz.errors[name], rtol=1e-4)
    np.testing.assert_allclose(dalitz.fval - square.fval, log_jacobian_sum, rtol=1e-9)


def test_sample_round_trips_through_square_dalitz_coordinates():
    mprime = jnp.asarray([0.1, 0.4, 0.75, 0.9])
    thetaprime = jnp.asarray([0.2, 0.6, 0.35, 0.95])
    sample = PhaseSpaceSample.from_square_dalitz(
        mprime, thetaprime, mother_mass=MOTHER, masses=MASSES, pair=PAIR
    )
    np.testing.assert_allclose(sample.weights, 1.0)
    back = sample.square_dalitz(mother_mass=MOTHER, masses=MASSES, pair=PAIR)
    np.testing.assert_allclose(back[0], mprime, atol=1e-10)
    np.testing.assert_allclose(back[1], thetaprime, atol=1e-10)
    np.testing.assert_allclose(
        sample.s12 + sample.s13 + sample.s23,
        MOTHER**2 + sum(m * m for m in MASSES),
        rtol=1e-12,
    )


def test_root_input_in_square_dalitz_branches(tmp_path):
    mprime = np.array([0.15, 0.5, 0.8])
    thetaprime = np.array([0.3, 0.7, 0.1])
    path = tmp_path / "square.root"
    with uproot.recreate(path) as root_file:
        root_file["events"] = {
            "mp": mprime,
            "thp": thetaprime,
            "w": np.array([1.0, 2.0, 0.5]),
        }
    sample = read_phase_space_sample(
        path,
        "events",
        mprime="mp",
        thetaprime="thp",
        weight="w",
        mother_mass=MOTHER,
        masses=MASSES,
        pair=PAIR,
    )
    expected = PhaseSpaceSample.from_square_dalitz(
        mprime, thetaprime, mother_mass=MOTHER, masses=MASSES, pair=PAIR
    )
    np.testing.assert_allclose(sample.s12, expected.s12)
    np.testing.assert_allclose(sample.s13, expected.s13)
    np.testing.assert_allclose(sample.weights, [1.0, 2.0, 0.5])
    with pytest.raises(ValueError, match="requires mprime, thetaprime"):
        read_phase_space_sample(path, "events", mprime="mp", thetaprime="thp")


def test_single_sample_fit_is_the_same_in_square_dalitz_coordinates():
    model = _single_model()
    data = _toy(model, 1500, seed=1)
    start = {"nr.x": 0.5, "nr.y": -0.2}
    fits = {
        coordinates: FitSession(model, data, coordinates=coordinates).fit(
            dict(start), strategy=1, hessian="jax"
        )
        for coordinates in ("dalitz", "square-dalitz")
    }
    _assert_same_fit(
        fits["dalitz"],
        fits["square-dalitz"],
        float(jnp.sum(square_dalitz_log_jacobian(model, data))),
    )


def test_qmi_objective_differs_only_by_the_jacobian_term():
    knots = (0.28, 0.6, 1.0, 1.6, 3.0, 5.14)
    nodes = [
        (
            Parameter.dynamics(f"S[{k}].re", 0.3 + 0.1 * k, owner="S"),
            Parameter.dynamics(f"S[{k}].im", -0.1 * k, owner="S"),
        )
        for k in range(len(knots))
    ]
    s_wave = Resonance(
        "S",
        (0, 1),
        RealImag(1.0, 0.0),
        lineshape=QMI(
            knots=knots,
            real_parts=tuple(n[0] for n in nodes),
            imaginary_parts=tuple(n[1] for n in nodes),
            interpolation="linear",
        ),
        mass=1.0,
        width=0.1,
        spin=0,
        normalize_component=False,
    )
    model = _model(PLUS, RealImag(1.0, 0.0), RealImag(0.5, 0.2), extra=(s_wave,))
    data = model.generate_phase_space(400, seed=4, include_momenta=False)
    dalitz = FitSession(model, data).objective
    square = FitSession(model, data, coordinates="square-dalitz").objective
    shift = float(jnp.sum(square_dalitz_log_jacobian(model, data)))
    for scale in (1.0, 1.3):
        values = {
            p.name: float(p.value) * scale for p in model.parameters if not p.fixed
        }
        np.testing.assert_allclose(
            float(dalitz(values)) - float(square(values)), shift, rtol=1e-10
        )
        gd, gs = jax.grad(dalitz)(values), jax.grad(square)(values)
        for name in values:
            np.testing.assert_allclose(gs[name], gd[name], rtol=1e-9, atol=1e-12)


def _efficiency_and_background():
    edges = tuple(np.linspace(0.0, 1.0, 6))
    rng = np.random.default_rng(7)
    efficiency = SquareDalitzHistogramEfficiency(
        edges,
        edges,
        0.5 + 0.4 * rng.random((5, 5)),
        mother_mass=MOTHER,
        masses=MASSES,
        pair=PAIR,
        interpolation="linear",
    )
    background = SquareDalitzHistogramBackground(
        edges,
        edges,
        0.2 + rng.random((5, 5)),
        mother_mass=MOTHER,
        masses=MASSES,
        pair=PAIR,
        divide_jacobian=True,
    )
    return efficiency, background


def test_fit_with_efficiency_and_background_is_the_same_in_square_dalitz_coordinates():
    model = _single_model()
    efficiency, background = _efficiency_and_background()
    data = _toy_with_efficiency_and_background(
        model, efficiency, background, 1500, seed=2
    )
    fraction = Parameter("signal_fraction", 0.75, bounds=(0.0, 1.0))
    start = {"nr.x": 0.5, "nr.y": -0.2, "signal_fraction": 0.75}
    fits = {}
    for coordinates in ("dalitz", "square-dalitz"):
        session = FitSession(
            model,
            data,
            efficiency=efficiency,
            signal_fraction=fraction,
            coordinates=coordinates,
        )
        session = session.with_background("bkg", background)
        fits[coordinates] = session.fit(dict(start), strategy=1, hessian="jax")
    _assert_same_fit(
        fits["dalitz"],
        fits["square-dalitz"],
        float(jnp.sum(square_dalitz_log_jacobian(model, data))),
    )


def test_cp_fit_with_efficiency_and_background_is_the_same_in_square_dalitz():
    plus_model, minus_model = _cp_models()
    efficiency, background = _efficiency_and_background()
    plus_data, minus_data = (
        _toy_with_efficiency_and_background(
            model, efficiency, background, 1200, seed=seed
        )
        for model, seed in ((plus_model, 5), (minus_model, 6))
    )
    fraction = Parameter("signal_fraction", 0.75, bounds=(0.0, 1.0))
    start = {
        "nr.x": 0.5,
        "nr.y": -0.2,
        "nr.dx": 0.0,
        "nr.dy": 0.0,
        "rho.dx": 0.0,
        "signal_fraction": 0.75,
    }
    fits = {}
    for coordinates in ("dalitz", "square-dalitz"):
        session = CPFitSession(
            plus_model,
            minus_model,
            plus_data,
            minus_data,
            signal_fraction=fraction,
            coordinates=coordinates,
        ).with_efficiency(efficiency)
        session = session.with_background("bkg", background)
        fits[coordinates] = session.fit(dict(start), strategy=1, hessian="jax")
    shift = float(
        jnp.sum(square_dalitz_log_jacobian(plus_model, plus_data))
        + jnp.sum(square_dalitz_log_jacobian(minus_model, minus_data))
    )
    _assert_same_fit(fits["dalitz"], fits["square-dalitz"], shift)


def test_weighted_cp_objective_uses_weighted_jacobian_term():
    plus_model, minus_model = _cp_models()
    plus_data = plus_model.generate_phase_space(300, seed=8, include_momenta=False)
    minus_data = minus_model.generate_phase_space(300, seed=9, include_momenta=False)
    rng = np.random.default_rng(3)
    weights = (jnp.asarray(0.5 + rng.random(300)), jnp.asarray(0.5 + rng.random(300)))
    sessions = {
        c: CPFitSession(
            plus_model, minus_model, plus_data, minus_data, coordinates=c
        ).with_event_weights(*weights)
        for c in ("dalitz", "square-dalitz")
    }
    shift = float(
        jnp.sum(weights[0] * square_dalitz_log_jacobian(plus_model, plus_data))
        + jnp.sum(weights[1] * square_dalitz_log_jacobian(minus_model, minus_data))
    )
    values = {
        p.name: float(p.value) + 0.05
        for p in plus_model.parameters + minus_model.parameters
        if not p.fixed
    }
    for objective in ("objective", "_weighted_objective"):

        def evaluate(c, v, objective=objective):
            session = sessions[c]
            nll = (
                session.objective
                if objective == "objective"
                else session._weighted_objective(weights)
            )
            return nll(v)

        np.testing.assert_allclose(
            float(evaluate("dalitz", values))
            - float(evaluate("square-dalitz", values)),
            shift,
            rtol=1e-10,
        )


def test_square_dalitz_coordinates_reject_inconsistent_inputs():
    data = _single_model().generate_phase_space(50, seed=1, include_momenta=False)
    with pytest.raises(ValueError, match="coordinates must be one of"):
        FitSession(_single_model(), data, coordinates="polar")
    with pytest.raises(ValueError, match="normalization_method='square-dalitz'"):
        FitSession(
            _single_model(method="gauss-legendre"), data, coordinates="square-dalitz"
        )
    edges = (0.0, 0.5, 1.0)
    background = SquareDalitzHistogramBackground(
        edges, edges, np.ones((2, 2)), mother_mass=MOTHER, masses=MASSES, pair=PAIR
    )
    with pytest.raises(ValueError, match="divide_jacobian=True"):
        FitSession(
            _single_model(), data, signal_fraction=0.8, coordinates="square-dalitz"
        ).with_background("bkg", background)
    efficiency = SquareDalitzHistogramEfficiency(
        edges,
        edges,
        np.ones((2, 2)),
        mother_mass=MOTHER,
        masses=MASSES,
        pair=PAIR,
        divide_jacobian=True,
    )
    with pytest.raises(ValueError, match="dimensionless"):
        FitSession(
            _single_model(), data, efficiency=efficiency, coordinates="square-dalitz"
        )
    plus_model, minus_model = _cp_models()
    with pytest.raises(ValueError, match="divide_jacobian=True"):
        CPFitSession(
            plus_model,
            minus_model,
            data,
            data,
            signal_fraction=0.8,
            coordinates="square-dalitz",
        ).with_background("bkg", background)
    # The Dalitz-plot default is untouched.
    FitSession(
        _single_model(method="gauss-legendre"), data, signal_fraction=0.8
    ).with_background("bkg", background)
