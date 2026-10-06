"""Physics and fit-infrastructure regressions for the initial four-body layer."""

from dataclasses import replace
from math import factorial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from jaxpwa import (
    AmplitudeComponent,
    CascadeChain,
    ConstantAmplitude,
    FitSession,
    FourBodyDecayModel,
    Isobar,
    NBodyDecayChannel,
    NBodyPhaseSpaceMC,
    NBodySample,
    PairChain,
    Parameter,
    Pole,
    RealImag,
    RelativisticBreitWigner,
    cascade_coordinates,
    pair_coordinates,
    pair_coordinates_to_momenta,
    weighted_resample,
)
from jaxpwa.dynamics.sequential import clebsch_gordan, wigner_d
from jaxpwa.kinematics.phase_space_mc import _boost_from_rest


@pytest.fixture(scope="module")
def sample():
    return NBodyPhaseSpaceMC(2.0, (0.1, 0.2, 0.3, 0.4)).generate(2048, seed=45)


@pytest.mark.parametrize(
    "mother,masses",
    [(1.0, (0.0,) * 4), (2.0, (0.1, 0.2, 0.3, 0.4)), (1.0, (0.24999,) * 4)],
)
def test_physical_conservation_and_reproducibility(mother, masses):
    generator = NBodyPhaseSpaceMC(mother, masses)
    sample = generator.generate(4096, seed=13)
    sample.validate_integration()
    sample.validate_physical(mother, masses)
    np.testing.assert_allclose(
        jnp.sum(sample.momenta, axis=1),
        np.tile([mother, 0, 0, 0], (sample.size, 1)),
        atol=2e-9,
    )
    np.testing.assert_array_equal(
        sample.momenta, generator.generate(4096, seed=13).momenta
    )
    np.testing.assert_array_equal(
        jax.jit(lambda s: s.mass_squared(0, 1))(sample), sample.mass_squared(0, 1)
    )


@pytest.mark.parametrize("nbody", [2, 3, 4, 5])
def test_massless_absolute_volume(nbody):
    mother = 1.7
    sample = NBodyPhaseSpaceMC(mother, (0.0,) * nbody).generate(80_000, seed=nbody)
    # Integrated labelled phase space including (2pi)^4 delta^4.
    expected = mother ** (2 * nbody - 4) / (
        2 * (4 * np.pi) ** (2 * nbody - 3) * factorial(nbody - 1) * factorial(nbody - 2)
    )
    w = np.asarray(sample.weights)
    assert abs(np.mean(w) - expected) < 6 * np.std(w) / np.sqrt(w.size) + 1e-14


def test_massive_volume_against_independent_pair_quadrature():
    masses, mother = (0.1, 0.2, 0.3, 0.4), 2.0
    nodes, weights = np.polynomial.legendre.leggauss(160)
    u, w = (nodes + 1) / 2, weights / 2
    lo, hi = sum(masses[:2]) ** 2, (mother - sum(masses[2:])) ** 2
    s1 = lo + (hi - lo) * u[:, None]
    lower = sum(masses[2:]) ** 2
    upper = (mother - np.sqrt(s1)) ** 2
    s2 = lower + (upper - lower) * u[None, :]

    def rho(s, a, b):
        return np.sqrt(np.maximum((s - a - b) ** 2 - 4 * a * b, 0)) / (8 * np.pi * s)

    integrand = (
        rho(mother**2, s1, s2)
        * rho(s1, masses[0] ** 2, masses[1] ** 2)
        * rho(s2, masses[2] ** 2, masses[3] ** 2)
        * (hi - lo)
        * (upper - lower)
        / (2 * np.pi) ** 2
    )
    expected = np.sum(w[:, None] * w[None, :] * integrand)
    sample = NBodyPhaseSpaceMC(mother, masses).generate(100_000, seed=68)
    observed = np.asarray(sample.weights)
    assert abs(observed.mean() - expected) < 5 * observed.std() / np.sqrt(observed.size)


def test_phase_space_symmetry_and_angular_moments():
    sample = NBodyPhaseSpaceMC(1.0, (0.0,) * 4).generate(100_000, seed=87)
    w = np.asarray(sample.weights)
    for name in ("s12", "s13", "s14", "s23", "s24", "s34"):
        x = np.asarray(sample.observable(name)) - 1 / 6
        assert abs(np.mean(w * x)) < 5 * np.std(w * x) / np.sqrt(sample.size)
    angles = pair_coordinates(sample.momenta)
    for x in (
        angles["cos_theta1"],
        angles["cos_theta2"],
        jnp.sin(angles["phi"]),
        jnp.cos(angles["phi"]),
        angles["cos_theta1"] ** 2 - 1 / 3,
    ):
        wx = w * np.asarray(x)
        assert abs(np.mean(wx)) < 5 * np.std(wx) / np.sqrt(sample.size)


def test_lorentz_invariance_and_parity(sample):
    beta = jnp.array([0.24, -0.17, 0.31])
    # Also rotate the event by a proper spatial rotation.
    p = sample.momenta.at[..., 1:].set(sample.momenta[..., (2, 3, 1)])
    transformed = replace(sample, momenta=_boost_from_rest(p[..., 0], p[..., 1:], beta))
    transformed.validate_physical(2.0, (0.1, 0.2, 0.3, 0.4))
    for key, value in sample.invariants().items():
        np.testing.assert_allclose(transformed.invariants()[key], value, atol=2e-12)
    for coordinates in (pair_coordinates, cascade_coordinates):
        original, boosted = (
            coordinates(sample.momenta),
            coordinates(transformed.momenta),
        )
        for key in original:
            if key == "phi":
                np.testing.assert_allclose(
                    jnp.exp(1j * original[key]), jnp.exp(1j * boosted[key]), atol=2e-11
                )
            else:
                np.testing.assert_allclose(boosted[key], original[key], atol=2e-11)
    parity = replace(sample, momenta=sample.momenta * jnp.array([1, -1, -1, -1]))
    np.testing.assert_allclose(
        parity.invariants()["orientation"],
        -sample.invariants()["orientation"],
        atol=1e-14,
    )
    for chain in (
        PairChain(Isobar(0.6, 0.15, 1), Isobar(0.85, 0.15, 1), orbital=1),
        CascadeChain(Isobar(1.4, 0.2, 1), Isobar(0.85, 0.15, 1)),
    ):
        original = chain({"momenta": sample.momenta})
        assert np.all(np.isfinite(original))
        np.testing.assert_allclose(
            chain({"momenta": transformed.momenta}), original, atol=2e-10, rtol=1e-9
        )


def test_wigner_and_clebsch_gordan_conventions():
    x = jnp.linspace(-0.9, 0.9, 13)
    np.testing.assert_allclose(wigner_d(1, 1, 0, x), -jnp.sqrt((1 - x * x) / 2))
    np.testing.assert_allclose(wigner_d(2, 0, 0, x), (3 * x * x - 1) / 2, atol=1e-14)
    for j in range(5):
        d = np.array(
            [[wigner_d(j, m, n, x) for n in range(-j, j + 1)] for m in range(-j, j + 1)]
        )
        np.testing.assert_allclose(
            np.einsum("mne,pne->mpe", d, d),
            np.repeat(np.eye(2 * j + 1)[..., None], x.size, -1),
            atol=3e-14,
        )
    for h in (-1, 0, 1):
        assert clebsch_gordan(1, h, 1, -h, 0, 0) == pytest.approx(
            (-1) ** (1 - h) / np.sqrt(3)
        )


def test_known_angular_limits(sample):
    pair = PairChain(Isobar(0.6, 0.15, 1), Isobar(0.85, 0.15, 1))
    angles = pair_coordinates(sample.momenta)
    x, y, phi = (angles[k] for k in ("cos_theta1", "cos_theta2", "phi"))
    expected = (x * y + jnp.sqrt((1 - x * x) * (1 - y * y)) * jnp.cos(phi)) / np.sqrt(3)
    np.testing.assert_allclose(
        pair._get_geometry(sample.as_dict())["angular"], expected, atol=2e-14
    )
    # Independent canonical-vector spin contraction: two spin-one objects
    # coupled to S=L=0 have no preferred production axis. In rotation-free
    # rest frames this is -a_hat dot c_hat / sqrt(3). It detects omission of
    # the particle-2 helicity rotation phase in the LS conversion.
    from jaxpwa import boost_to_rest_frame

    p = sample.momenta
    a = boost_to_rest_frame(p[:, 0], p[:, 0] + p[:, 1])[:, 1:]
    c = boost_to_rest_frame(p[:, 2], p[:, 2] + p[:, 3])[:, 1:]
    contraction = -jnp.sum(a * c, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(c, axis=-1) * jnp.sqrt(3.0)
    )
    np.testing.assert_allclose(expected, contraction, atol=2e-13)
    cascade = CascadeChain(Isobar(1.4, 0.2, 0), Isobar(0.85, 0.15, 1), orbital=1)
    angles = cascade_coordinates(sample.momenta)
    np.testing.assert_allclose(
        cascade._get_geometry(sample.as_dict())["angular"],
        -angles["cos_theta_S"],
        atol=2e-14,
    )
    scalar = PairChain(Isobar(0.6, 0.15), Isobar(0.85, 0.15))
    np.testing.assert_array_equal(scalar._get_geometry(sample.as_dict())["angular"], 1)


def test_five_coordinate_roundtrip_and_relative_chain_phase(sample):
    angles = pair_coordinates(sample.momenta)
    reconstructed = jax.jit(
        lambda a: pair_coordinates_to_momenta(
            2.0,
            (0.1, 0.2, 0.3, 0.4),
            a["s_ab"],
            a["s_cd"],
            a["cos_theta1"],
            a["cos_theta2"],
            a["phi"],
        )
    )(angles)
    rebuilt = NBodySample(reconstructed, sample.weights)
    rebuilt.validate_physical(2.0, (0.1, 0.2, 0.3, 0.4))
    for key, original in sample.invariants().items():
        np.testing.assert_allclose(rebuilt.invariants()[key], original, atol=5e-13)
    # Check the complex amplitude, not only |A|^2, across different topologies.
    for function in (
        PairChain(Isobar(0.6, 0.1, 1), Isobar(0.85, 0.1, 1), orbital=1),
        CascadeChain(Isobar(1.4, 0.1, 1), Isobar(0.85, 0.1, 1), orbital=1),
    ):
        np.testing.assert_allclose(
            function(rebuilt.as_dict()),
            function(sample.as_dict()),
            atol=1e-10,
            rtol=2e-9,
        )


def test_existing_running_width_lineshape(sample):
    function = PairChain(
        Isobar(0.6, 0.15, 1, lineshape=RelativisticBreitWigner()),
        Isobar(0.85, 0.2, 1, lineshape=RelativisticBreitWigner()),
    )
    result = jax.jit(function)(sample.as_dict())
    assert np.all(np.isfinite(result))
    with pytest.raises(ValueError, match="dedicated width"):
        CascadeChain(
            Isobar(1.4, 0.2, 1, lineshape=RelativisticBreitWigner()),
            Isobar(0.85, 0.15, 1),
        )


def test_isobar_from_particle_database_and_overrides():
    rho = Isobar.from_particle("rho(770)0")
    assert rho.particle_name == "rho(770)0"
    assert rho.mass == pytest.approx(0.77526)
    assert rho.width == pytest.approx(0.1474)
    assert rho.spin == 1
    assert isinstance(rho.lineshape, Pole)

    # EvtGen aliases use the same resolver as DecayChannel.
    assert Isobar.from_particle("rho0").mass == pytest.approx(rho.mass)

    mass = Parameter.dynamics("rho.mass", 0.77, owner="rho")
    lineshape = RelativisticBreitWigner()
    overridden = Isobar.from_particle(
        "rho(770)0",
        mass=mass,
        width=0.2,
        spin=2,
        lineshape=lineshape,
        radius=4.0,
    )
    assert overridden.mass is mass
    assert overridden.width == 0.2
    assert overridden.spin == 2
    assert overridden.lineshape is lineshape
    assert overridden.radius == 4.0

    with pytest.raises(ValueError, match="Could not resolve"):
        Isobar.from_particle("not-a-real-particle", mass=1.0, width=0.1, spin=0)
    with pytest.raises(ValueError, match="integer spin"):
        Isobar.from_particle("Delta(1232)++")


def _model(sample, *, dynamic=False, normalize=True):
    mass = (
        Parameter.dynamics("mass", 0.63, owner="pair", bounds=(0.5, 0.8))
        if dynamic
        else 0.63
    )
    width = (
        Parameter.dynamics("width", 0.15, owner="pair", bounds=(0.1, 0.25))
        if dynamic
        else 0.15
    )
    components = [
        AmplitudeComponent("nr", ConstantAmplitude(), RealImag(1.0, 0.0)),
        AmplitudeComponent(
            "pair",
            PairChain(Isobar(mass, width), Isobar(0.85, 0.18)),
            RealImag(Parameter.coefficient("x", 0.4), Parameter.coefficient("y", 0.2)),
        ),
    ]
    return FourBodyDecayModel(
        NBodyDecayChannel(2.0, (0.1, 0.2, 0.3, 0.4)),
        components,
        normalization_sample=sample,
        normalize_components=normalize,
        normalization_chunk_size=601,
    )


def test_four_body_model_with_fixed_parameters_is_uniform(sample):
    model = _model(sample, dynamic=True)
    original = {parameter.name: parameter for parameter in model.parameters}

    updated = model.with_fixed_parameters("mass", values={"x": 0.61})
    result = {parameter.name: parameter for parameter in updated.parameters}

    assert result["mass"].fixed
    assert result["mass"].value == original["mass"].value
    assert result["x"].fixed
    assert result["x"].value == pytest.approx(0.61)
    assert not result["width"].fixed
    assert not original["mass"].fixed
    assert not original["x"].fixed
    assert updated.normalization_sample is sample
    assert updated.normalize_components == model.normalize_components
    assert updated.symmetrize == model.symmetrize
    assert updated.normalization_chunk_size == model.normalization_chunk_size

    with pytest.raises(ValueError, match="unknown parameter"):
        model.with_fixed_parameters("does.not.exist")


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_cache_pdf_gradients_and_fit_session(sample, dynamic, normalize):
    model = _model(sample, dynamic=dynamic, normalize=normalize)
    data = sample.take(jnp.arange(31))
    session = FitSession(model, data).with_efficiency(lambda d: 0.5 + 0.1 * d["s12"])
    cache = session.signal_cache
    values = {p.name: p.value for p in model.parameters}
    values.update(x=0.31, y=-0.21)
    if dynamic:
        values.update(mass=0.67, width=0.17)
    expected = model.pdf(efficiency=session.efficiency)
    np.testing.assert_allclose(
        cache.intensity(values), model.intensity(data.as_dict(), values), rtol=2e-10
    )
    np.testing.assert_allclose(
        cache.normalization(values), expected.normalization(values), rtol=2e-10
    )
    objective = session.objective
    value, grad = jax.jit(jax.value_and_grad(lambda v: objective(v)))(values)
    assert np.isfinite(value)
    for name in values:
        plus, minus = dict(values), dict(values)
        plus[name] += 1e-5
        minus[name] -= 1e-5
        fd = (objective(plus) - objective(minus)) / (2e-5)
        assert np.isfinite(grad[name])
        np.testing.assert_allclose(grad[name], fd, rtol=2e-5, atol=2e-6)
    if dynamic:
        prepared = model.components[1].function.compact_prepared_data(data.as_dict())
        assert "momenta" not in prepared
    else:
        assert cache.is_compact
        assert cache.normalization_components is None


def test_bose_symmetry_and_owner_validation(sample):
    sample = NBodyPhaseSpaceMC(2.0, (0.2,) * 4).generate(64, seed=13)
    component = AmplitudeComponent(
        "r", PairChain(Isobar(0.6, 0.1), Isobar(0.8, 0.2)), 1.0
    )
    model = FourBodyDecayModel(
        NBodyDecayChannel(2.0, (0.2,) * 4, (1, 2, 1, 2)),
        [component],
        normalization_sample=sample,
    )
    swapped = sample.momenta[:, (2, 1, 0, 3)]
    np.testing.assert_allclose(
        model.intensity({"momenta": swapped}),
        model.intensity({"momenta": sample.momenta}),
        rtol=1e-12,
    )
    bad = replace(
        component,
        function=PairChain(
            Isobar(Parameter.dynamics("m", 0.6, owner="wrong"), 0.1), Isobar(0.8, 0.2)
        ),
    )
    with pytest.raises(ValueError, match="owner"):
        FourBodyDecayModel(model.channel, [bad], normalization_sample=sample)


def test_external_normalization_and_component_overrides(sample):
    base = _model(sample)
    components = list(base.components)
    components[1] = replace(components[1], normalize_component=False)
    model = FourBodyDecayModel(base.channel, components, normalization_sample=sample)
    selected = sample.select_for_integration(sample.mass_squared(0, 1) < 0.7)
    data = selected.take(jnp.arange(13))
    values = {"x": 0.3, "y": 0.1}
    cache = model.prepare_cache(data, selected)
    pdf = model.pdf(selected)
    np.testing.assert_allclose(
        cache.intensity(values) / cache.normalization(values),
        pdf(data.as_dict(), values),
        rtol=2e-12,
    )
    ff = cache.fit_fractions(values)
    interference = cache.interference_fractions(values)
    np.testing.assert_allclose(jnp.sum(ff) + jnp.sum(jnp.triu(interference, 1)), 1.0)


def test_integration_selection_resampling_and_plotting(sample):
    mask = sample.mass_squared(0, 1) < 0.5
    selected = sample.select_for_integration(mask)
    np.testing.assert_allclose(
        jnp.mean(selected.weights), jnp.mean(sample.weights * mask)
    )
    data = weighted_resample(jax.random.PRNGKey(3), sample, sample.weights, 80)
    assert isinstance(data, NBodySample)
    np.testing.assert_array_equal(data.weights, 1)
    model = _model(sample)
    session = FitSession(model, data)
    values = {p.name: p.value for p in model.parameters}
    import matplotlib.pyplot as plt

    for variable in ("s34", "s234", "cos_theta1", "phi"):
        ax = session.plot_projection(
            SimpleNamespace(values=values), variable, bins=12, projection_sample=sample
        )
        assert len(ax.lines) > 0
        plt.close(ax.figure)


def test_validation(sample):
    for masses in ((1.0,) * 4, (-0.1, 0.2), (np.nan, 0.1)):
        with pytest.raises(ValueError):
            NBodyPhaseSpaceMC(1.0, masses)
    with pytest.raises(ValueError, match="L=S"):
        PairChain(Isobar(0.6, 0.1, 1), Isobar(0.8, 0.1), orbital=0)
    with pytest.raises(ValueError, match="couple"):
        CascadeChain(Isobar(1.4, 0.1, 2), Isobar(0.8, 0.1), orbital=0)
    with pytest.raises(ValueError, match="spinless"):
        NBodyDecayChannel.from_particles("B0", ("p", "p~", "pi+", "pi-"))
    with pytest.raises(ValueError):
        replace(sample, weights=-sample.weights).validate_integration()
    with pytest.raises(ValueError, match="off shell"):
        replace(sample, momenta=sample.momenta * 1.1).validate_physical(
            2.0, (0.1, 0.2, 0.3, 0.4)
        )


def test_continuous_toy_minuit_closure():
    """Actual independent generation/normalization/fit, with the phase fixed."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/examples/four_body_closure.py"
    )
    spec = importlib.util.spec_from_file_location("four_body_closure", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    _, result, summary = example.run_closure(events=1500, normalization=60_000)
    assert result.valid
    assert max(abs(p) for p in summary["pulls"].values()) < 5
