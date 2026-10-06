import jax.numpy as jnp
import pytest

from jaxpwa import (
    DecayChannel,
    DecayModel,
    Isobar,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    Resonance,
    enable_x64,
)

enable_x64()


class ConstantLineshape:
    """Minimal test plugin proving DecayModel is lineshape-agnostic."""

    def __call__(self, mass, context):
        del context
        return jnp.ones_like(mass, dtype=jnp.complex128)


def _model(channel, components, **kwargs):
    return DecayModel(
        channel,
        components,
        normalization_method="square-dalitz",
        normalization_resolution=45,
        **kwargs,
    )


def _floating_rho_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass", 0.760, owner="rho", backend_name="mass", bounds=(0.72, 0.82)
    )
    width = Parameter.dynamics(
        "rho.width", 0.180, owner="rho", backend_name="width", bounds=(0.08, 0.25)
    )
    x = Parameter.coefficient("rho.x", 0.8, owner="rho")
    y = Parameter.coefficient("rho.y", 0.2, owner="rho")
    return _model(
        channel,
        [
            Resonance(
                "rho",
                pair=(0, 1),
                coefficient=RealImag(x, y),
                mass=mass,
                width=width,
                spin=1,
            ),
            NonResonant(RealImag(1.0, 0.0)),
        ],
    )


def test_decay_channel_resolves_particle_masses_in_gev():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    assert 1.86 < channel.parent_mass < 1.88
    assert all(0.139 < mass < 0.141 for mass in channel.daughter_masses)
    assert channel.final_state_ids[1] == channel.final_state_ids[2]
    assert channel.final_state_ids[0] != channel.final_state_ids[1]


def test_decay_channel_from_particles_matches_named_constructor():
    direct = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    assert DecayChannel.from_particles(
        "D+", ("pi-", "pi+", "pi+")
    ) == direct


def test_resonance_and_isobar_share_particle_resolution():
    coefficient = RealImag(1.0, 0.0)
    resonance = Resonance.from_particle(
        "rho(770)0",
        pair=(0, 1),
        coefficient=coefficient,
        name="rho_12",
    )
    isobar = Isobar.from_particle("rho(770)0")

    assert resonance.name == "rho_12"
    assert resonance.particle_name == isobar.particle_name == "rho(770)0"
    assert resonance.mass == pytest.approx(isobar.mass)
    assert resonance.width == pytest.approx(isobar.width)
    assert resonance.spin == isobar.spin == 1

    overridden = Resonance.from_particle(
        "rho(770)0",
        pair=(0, 1),
        coefficient=coefficient,
        mass=0.77,
        width=0.2,
        spin=2,
    )
    assert overridden.mass == 0.77
    assert overridden.width == 0.2
    assert overridden.spin == 2

    with pytest.raises(ValueError, match="Could not resolve"):
        Resonance.from_particle(
            "not-a-real-particle",
            pair=(0, 1),
            coefficient=coefficient,
            mass=1.0,
            width=0.1,
            spin=0,
        )
    with pytest.raises(ValueError, match="integer spin"):
        Resonance.from_particle(
            "Delta(1232)++",
            pair=(0, 1),
            coefficient=coefficient,
        )


def test_decay_model_with_fixed_parameters_is_immutable_and_selective():
    model = _floating_rho_model()
    original = {parameter.name: parameter for parameter in model.parameters}

    updated = model.with_fixed_parameters(
        "rho.mass", values={"rho.x": 0.91}
    )
    result = {parameter.name: parameter for parameter in updated.parameters}

    assert result["rho.mass"].fixed
    assert result["rho.mass"].value == original["rho.mass"].value
    assert result["rho.x"].fixed
    assert result["rho.x"].value == pytest.approx(0.91)
    assert not result["rho.width"].fixed
    assert not original["rho.mass"].fixed
    assert not original["rho.x"].fixed
    assert updated.normalization_method == model.normalization_method
    assert updated.normalization_resolution == model.normalization_resolution

    with pytest.raises(ValueError, match="unknown parameter"):
        model.with_fixed_parameters("does.not.exist")
    with pytest.raises(ValueError, match="at least one"):
        model.with_fixed_parameters()


def test_decay_model_with_fixed_parameters_preserves_toy_normalization_sample():
    base = _floating_rho_model()
    sample = base.generate_phase_space(250, seed=19)
    model = DecayModel(
        base.channel,
        base.components,
        normalization_sample=sample,
    )

    updated = model.with_fixed_parameters("rho.mass")

    assert updated.normalization_method == "toy-mc"
    assert updated.normalization_sample is sample


def test_unphysical_decay_channel_is_rejected():
    with pytest.raises(ValueError, match="parent mass must exceed"):
        DecayChannel("pi0", ("pi0", "pi0", "pi0"))


def test_decay_model_defaults_to_gauss_legendre_component_normalization():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(channel, [NonResonant(RealImag(1.0, 0.0))])
    assert model.normalize_components is True
    assert model.normalization_method == "gauss-legendre"
    assert model.normalization_resolution == 1000
    assert model._normalization_sample is None


def test_internal_normalization_grid_is_lazy_and_reused():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(channel, [NonResonant(RealImag(1.0, 0.0))])
    assert model._normalization_sample is None
    first = model.normalization_sample
    second = model.normalization_sample
    assert first is second
    assert first.size == 45**2
    assert bool(jnp.any(first.weights != first.weights[0]))


def test_decay_model_rejects_unknown_normalization_methods():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    components = [NonResonant(RealImag(1.0, 0.0))]
    for method in ("equal_area", "adaptive", "auto"):
        with pytest.raises(ValueError, match="gauss-legendre.*square-dalitz.*toy-mc"):
            DecayModel(channel, components, normalization_method=method)


def test_toy_mc_normalization_requires_sample():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    with pytest.raises(ValueError, match="requires normalization_sample"):
        DecayModel(
            channel,
            [NonResonant(RealImag(1.0, 0.0))],
            normalization_method="toy-mc",
        )


def test_external_toy_mc_sample_replaces_grid_for_all_normalization():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    toy = PhaseSpaceSample(
        s12=jnp.asarray([0.20, 0.30, 0.40, 0.50]),
        s13=jnp.asarray([0.60, 0.70, 0.80, 0.90]),
        s23=jnp.asarray([2.20, 2.00, 1.80, 1.60]),
        weights=jnp.asarray([0.5, 1.0, 1.5, 2.0]),
    )
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalize_components=False,
        normalization_sample=toy,
    )

    assert model.normalization_method == "toy-mc"
    assert model.normalization_sample is toy
    assert model.normalization_scheme == {
        "method": "toy-mc",
        "adaptive": False,
        "sample_size": 4,
        "weighted": True,
        "chunk_size": "auto",
    }

    data = PhaseSpaceSample(
        s12=toy.s12[:2],
        s13=toy.s13[:2],
        s23=toy.s23[:2],
        weights=jnp.ones((2,)),
    )
    cache = model.prepare_cache(data)
    _, normalization = cache.evaluate({})
    assert jnp.allclose(normalization, jnp.mean(toy.weights))

    normalized_model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_sample=toy,
    )
    component = normalized_model.amplitude_model.components[0]
    assert jnp.allclose(
        normalized_model._component_scale(component),
        1.0 / jnp.sqrt(jnp.mean(toy.weights)),
    )


def test_unweighted_toy_mc_sample_uses_unit_weights():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    toy = PhaseSpaceSample(
        s12=jnp.asarray([0.20, 0.30, 0.40]),
        s13=jnp.asarray([0.60, 0.70, 0.80]),
        s23=jnp.asarray([2.20, 2.00, 1.80]),
        weights=jnp.ones((3,)),
    )
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalize_components=False,
        normalization_method="toy-mc",
        normalization_sample=toy,
    )
    assert model.normalization_scheme["weighted"] is False
    cache = model.prepare_cache(toy)
    _, normalization = cache.evaluate({})
    assert jnp.allclose(normalization, 1.0)


def test_component_normalization_is_unit_diagonal_by_default():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(RealImag(0.3, -0.2)),
        ],
    )
    data = model.generate_phase_space(64, seed=11)
    cache = model.prepare_cache(data)
    diagonal = jnp.real(jnp.diag(cache.normalization_matrix_fixed))
    assert jnp.allclose(diagonal, jnp.ones_like(diagonal), rtol=1e-11, atol=1e-11)


def test_component_normalization_can_be_disabled():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalize_components=False,
    )
    data = model.generate_phase_space(32, seed=12)
    cache = model.prepare_cache(data)
    assert cache.normalize_components is False


def test_component_normalization_override_reaches_amplitude_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
                normalize_component=False,
            ),
            NonResonant(RealImag(0.3, -0.2)),
        ],
        normalize_components=True,
    )

    built = model.amplitude_model.components
    assert built[0].normalize_component is False
    assert built[1].normalize_component is None
    assert model._component_scale(built[0]) == 1.0
    assert not jnp.isclose(model._component_scale(built[1]), 1.0)


def test_decay_model_computes_and_prints_fit_fractions(capsys):
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            NonResonant(RealImag(1.0, 0.0), name="first"),
            NonResonant(RealImag(1.0, 0.0), name="second"),
        ],
    )

    fractions = model.fit_fractions()
    interference = model.interference_fractions()
    printed = model.print_fit_fractions(include_interference=True, precision=2)
    output = capsys.readouterr().out

    assert jnp.allclose(fractions, jnp.asarray([0.25, 0.25]))
    assert jnp.allclose(
        jnp.sum(fractions) + jnp.sum(jnp.triu(interference, k=1)),
        1.0,
    )
    # include_interference=True switches the return value from the plain
    # {component: fraction} dict to a wrapping dict that also carries the
    # interference fractions.
    assert printed == {
        "fractions": {"first": 0.25, "second": 0.25},
        "interference": {("first", "second"): 0.5},
    }
    assert "Fit fractions (physical)" in output
    assert "first x second" in output
    assert "50.00" in output

    plain = model.print_fit_fractions(precision=2)
    capsys.readouterr()
    assert plain == {"first": 0.25, "second": 0.25}


def test_decay_model_acceptance_weighted_fit_fractions_validate_efficiency():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0), name="NR")],
    )

    fractions = model.fit_fractions(
        efficiency=lambda data: 0.5 + 0.0 * data["s12"]
    )
    assert jnp.allclose(fractions, jnp.ones((1,)))

    with pytest.raises(ValueError, match="one value per normalization point"):
        model.fit_fractions(efficiency=jnp.ones((3,)))


def test_decay_model_rejects_duplicate_component_names():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    with pytest.raises(ValueError, match="component names must be unique"):
        _model(
            channel,
            [
                NonResonant(RealImag(1.0, 0.0), name="same"),
                NonResonant(RealImag(0.2, 0.1), name="same"),
            ],
        )


def test_decay_model_rejects_ambiguous_dynamics_backend_names():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.775,
        owner="rho",
        backend_name="same_backend",
        bounds=(0.70, 0.85),
    )
    width = Parameter.dynamics(
        "rho.width",
        0.149,
        owner="rho",
        backend_name="same_backend",
        bounds=(0.05, 0.30),
    )
    with pytest.raises(ValueError, match="map to the same backend key"):
        _model(
            channel,
            [
                Resonance(
                    "rho",
                    pair=(0, 1),
                    coefficient=RealImag(1.0, 0.0),
                    mass=mass,
                    width=width,
                    spin=1,
                )
            ],
        )


def test_decay_model_rejects_unphysical_core_parameter_bounds():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.775,
        owner="rho",
        bounds=(-0.1, 1.0),
    )
    with pytest.raises(ValueError, match="lower bound for rho.mass must be positive"):
        Resonance(
            "rho",
            pair=(0, 1),
            coefficient=RealImag(1.0, 0.0),
            mass=mass,
            width=0.149,
            spin=1,
        )


def test_decay_model_can_generate_compact_phase_space():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(channel, [NonResonant(RealImag(1.0, 0.0))])

    full = model.generate_phase_space(128, seed=16)
    compact = model.generate_phase_space(128, seed=16, include_momenta=False)

    assert compact.p1 is None and compact.p2 is None and compact.p3 is None
    assert compact.size == full.size == 128
    invariant_sum = compact.s12 + compact.s13 + compact.s23
    expected = channel.parent_mass**2 + sum(
        mass**2 for mass in channel.daughter_masses
    )
    assert jnp.allclose(invariant_sum, expected, rtol=0.0, atol=1e-12)
    assert bool(jnp.all(compact.weights > 0.0))
    assert compact.nbytes * 4 == full.nbytes


def test_decay_model_accepts_custom_lineshape_plugin():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                lineshape=ConstantLineshape(),
                mass=0.77526,
                width=0.1491,
                spin=1,
            )
        ],
    )
    sample = model.generate_phase_space(64, seed=19)
    values = model.intensity(sample.as_dict())
    assert values.shape == (64,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values >= 0.0))


def test_decay_model_exposes_and_resolves_dynamic_parameters():
    model = _floating_rho_model()
    assert {parameter.name for parameter in model.parameters} == {
        "rho.mass",
        "rho.width",
        "rho.x",
        "rho.y",
    }

    sample = model.generate_phase_space(128, seed=31)
    default_values = model.intensity(sample.as_dict())
    shifted_values = model.intensity(
        sample.as_dict(),
        {
            "rho.mass": 0.790,
            "rho.width": 0.120,
            "rho.x": 1.1,
            "rho.y": -0.3,
        },
    )
    assert not jnp.allclose(default_values, shifted_values)


def test_prepared_cache_recomputes_floating_dynamics():
    model = _floating_rho_model()
    data = model.generate_phase_space(96, seed=41)
    cache = model.prepare_cache(data)

    initial = {
        "rho.mass": 0.760,
        "rho.width": 0.180,
        "rho.x": 0.8,
        "rho.y": 0.2,
    }
    shifted = {
        "rho.mass": 0.790,
        "rho.width": 0.120,
        "rho.x": 0.8,
        "rho.y": 0.2,
    }
    assert cache.data is not None
    assert cache.normalization_data is not None
    assert "p1" not in cache.data and "p2" not in cache.data and "p3" not in cache.data
    assert (
        "p1" not in cache.normalization_data
        and "p2" not in cache.normalization_data
        and "p3" not in cache.normalization_data
    )

    intensity_initial, norm_initial = cache.evaluate(initial)
    intensity_shifted, norm_shifted = cache.evaluate(shifted)
    assert not jnp.allclose(intensity_initial, intensity_shifted)
    assert not jnp.allclose(norm_initial, norm_shifted)


def test_prepared_cache_matches_direct_model_for_floating_dynamics():
    model = _floating_rho_model()
    data = model.generate_phase_space(128, seed=51)
    cache = model.prepare_cache(data)
    norm = model.normalization_sample

    points = (
        {"rho.mass": 0.735, "rho.width": 0.095, "rho.x": -0.3, "rho.y": 1.1},
        {"rho.mass": 0.775, "rho.width": 0.149, "rho.x": 0.8, "rho.y": 0.2},
        {"rho.mass": 0.815, "rho.width": 0.235, "rho.x": 1.4, "rho.y": -0.7},
    )
    for values in points:
        cached_intensity, cached_norm = cache.evaluate(values)
        direct_intensity = model.intensity(data.as_dict(), values)
        direct_norm = jnp.mean(
            norm.weights * model.intensity(norm.as_dict(), values)
        )
        assert jnp.allclose(
            cached_intensity,
            direct_intensity,
            rtol=1e-11,
            atol=1e-12,
        )
        assert jnp.allclose(cached_norm, direct_norm, rtol=1e-11, atol=1e-12)


def test_decay_model_builds_normalized_pdf_with_internal_grid():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(channel, [NonResonant(RealImag(1.0, 0.0))])
    pdf = model.pdf()
    norm = model.normalization_sample
    values = pdf(norm.as_dict(), {})
    assert values.shape == (norm.size,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values > 0.0))


def test_amplitude_model_is_built_once_and_reused():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(RealImag(0.2, -0.1)),
        ],
    )
    first = model.amplitude_model
    second = model.amplitude_model
    assert first is second
    assert first.components[0] is second.components[0]


def test_normalization_chunk_size_is_configurable_and_validated():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    components = [NonResonant(RealImag(1.0, 0.0), name="NR")]
    model = DecayModel(
        channel,
        components,
        normalization_method="square-dalitz",
        normalization_resolution=20,
        normalization_chunk_size=37,
    )
    kernel = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=False,
    )
    assert kernel.normalization_kernel.chunk_size == 37
    assert DecayModel(channel, components).normalization_chunk_size == "auto"

    for invalid in (0, -1, True, 1.5, "big"):
        with pytest.raises(ValueError, match="normalization_chunk_size"):
            DecayModel(channel, components, normalization_chunk_size=invalid)


def test_compact_prepare_kernel_is_reused_by_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0), name="NR")],
    )
    first = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=False,
    )
    second = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=False,
    )
    efficient = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=True,
    )
    assert first is second
    assert efficient is not first

    first_data = model.generate_phase_space(32, seed=101)
    second_data = model.generate_phase_space(32, seed=102)
    first_cache = model.prepare_cache(first_data)
    second_cache = model.prepare_cache(second_data)
    assert first_cache.is_compact
    assert second_cache.is_compact
    assert len(model._compact_prepare_kernels) == 2


def test_gauss_legendre_automatically_adapts_to_symmetrized_narrow_bands():
    channel = DecayChannel("B+", ("pi+", "pi+", "pi-"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "omega782",
                pair=(0, 2),
                coefficient=RealImag(0.09, -0.01),
                mass=0.78265,
                width=0.00849,
                spin=1,
            )
        ],
        normalization_method="gauss-legendre",
        normalization_bin_width=0.05,
        normalization_binning_factor=5.0,
    )

    scheme = model.normalization_scheme
    assert scheme["method"] == "gauss-legendre"
    assert scheme["adaptive"] is True
    assert scheme["internal_coordinates"] == "m13-m23"
    narrow = scheme["narrow_resonances"]
    assert narrow["m13"] == ((0.78265, 0.00849),)
    assert narrow["m23"] == ((0.78265, 0.00849),)
    assert any(segment.narrow for segment in scheme["m13_segments"])
    assert any(segment.narrow for segment in scheme["m23_segments"])


def test_gauss_legendre_diagonal_narrow_band_uses_square_coordinates_internally():
    channel = DecayChannel("D+", ("pi-", "pi+", "K+"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "narrow_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.77,
                width=0.010,
                spin=0,
            )
        ],
        normalization_method="gauss-legendre",
        normalization_resolution=32,
        normalization_binning_factor=4.0,
    )

    scheme = model.normalization_scheme
    assert scheme["method"] == "gauss-legendre"
    assert scheme["adaptive"] is True
    assert scheme["internal_coordinates"] == "square-dalitz"
    assert scheme["pair"] == (0, 1)
    assert scheme["narrow_resonances"]["m12"] == ((0.77, 0.01),)


def test_square_dalitz_automatically_adapts_and_prints(capsys):
    channel = DecayChannel("D+", ("pi-", "K+", "pi+"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "narrow",
                pair=(0, 2),
                coefficient=RealImag(1.0, 0.0),
                mass=0.90,
                width=0.010,
                spin=0,
            )
        ],
        normalization_method="square-dalitz",
        normalization_pair=(0, 2),
        normalization_resolution=16,
        normalization_binning_factor=2.0,
    )

    scheme = model.normalization_scheme
    assert scheme["method"] == "square-dalitz"
    assert scheme["adaptive"] is True
    assert scheme["narrow_resonances"]["m13"] == ((0.9, 0.01),)
    assert scheme["adaptive_axis"] == "mprime"
    assert scheme["mprime_nodes"] > 16
    assert scheme["estimated_points"] == scheme["mprime_nodes"] * 16

    sample = model.normalization_sample
    output = capsys.readouterr().out
    assert "narrow resonance band(s) detected" in output
    assert "adaptive Square-Dalitz refinement only along the mass axis" in output
    assert sample.size > 16**2


def test_automatic_adaptation_uses_nominal_values_for_floating_dynamics():
    channel = DecayChannel("D+", ("pi-", "K+", "pi+"))
    mass = Parameter.dynamics(
        "narrow.mass",
        0.90,
        owner="narrow",
        backend_name="mass",
        bounds=(0.85, 0.95),
    )
    width = Parameter.dynamics(
        "narrow.width",
        0.010,
        owner="narrow",
        backend_name="width",
        bounds=(0.005, 0.030),
    )
    model = DecayModel(
        channel,
        [
            Resonance(
                "narrow",
                pair=(0, 2),
                coefficient=RealImag(1.0, 0.0),
                mass=mass,
                width=width,
                spin=0,
            )
        ],
        normalization_method="gauss-legendre",
        normalization_bin_width=0.05,
        normalization_binning_factor=5.0,
    )

    scheme = model.normalization_scheme
    assert scheme["narrow_resonances"]["m13"] == ((0.9, 0.01),)
