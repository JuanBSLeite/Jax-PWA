import jax.numpy as jnp
import pytest

from jaxpwa import (
    LASS,
    QMI,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    FitSession,
    Flatte,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    cp_models_from_spec,
    cp_models_to_spec,
    cp_models_with_fitted_values,
    enable_x64,
    export_cp_models,
    export_model,
    import_cp_models,
    import_model,
    model_from_spec,
    model_to_spec,
    model_with_fitted_values,
)

enable_x64()


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
    return DecayModel(
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
        normalization_method="square-dalitz",
        normalization_resolution=45,
    )


def _sample(model, size=2000):
    return model.generate_phase_space(size)


def test_model_round_trip_reproduces_intensity():
    model = _floating_rho_model()
    sample = _sample(model)

    spec = model_to_spec(model)
    restored = model_from_spec(spec)

    original = model.intensity(sample.as_dict())
    reloaded = restored.intensity(sample.as_dict())
    assert jnp.allclose(original, reloaded)


def test_model_round_trip_preserves_parameters():
    model = _floating_rho_model()
    restored = model_from_spec(model_to_spec(model))

    original = {p.name: p for p in model.parameters}
    reloaded = {p.name: p for p in restored.parameters}
    assert original == reloaded


def test_model_round_trip_preserves_particle_backed_resonance_identity():
    model = DecayModel(
        DecayChannel.from_particles("D+", ("pi-", "pi+", "pi+")),
        [
            Resonance.from_particle(
                "rho(770)0",
                name="rho_12",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
            )
        ],
        normalization_method="square-dalitz",
        normalization_resolution=20,
    )

    restored = model_from_spec(model_to_spec(model))
    resonance = restored.components[0]
    assert resonance.name == "rho_12"
    assert resonance.particle_name == "rho(770)0"
    assert resonance.mass == pytest.approx(0.77526)
    assert resonance.width == pytest.approx(0.1474)
    assert resonance.spin == 1


def test_model_round_trip_preserves_normalization_chunk_size():
    spec = model_to_spec(_floating_rho_model())
    assert spec["normalization_chunk_size"] == "auto"
    spec["normalization_chunk_size"] = 12_345
    assert model_from_spec(spec).normalization_chunk_size == 12_345


def test_old_model_spec_without_chunk_size_uses_auto_and_ignores_removed_options():
    spec = model_to_spec(_floating_rho_model())
    del spec["normalization_chunk_size"]
    spec["dynamics_microbatch_size"] = 12_345  # option removed from the package
    restored = model_from_spec(spec)
    assert restored.normalization_chunk_size == "auto"
    assert not hasattr(restored, "dynamics_microbatch_size")


def test_export_import_file_round_trip(tmp_path):
    model = _floating_rho_model()
    path = tmp_path / "model.json"
    export_model(model, path)
    restored = import_model(path)

    sample = _sample(model)
    original = model.intensity(sample.as_dict())
    reloaded = restored.intensity(sample.as_dict())
    assert jnp.allclose(original, reloaded)


def test_shared_parameter_identity_is_preserved_via_registry():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    shared_radius = Parameter.meson_radius(
        "shared.parent_radius", 5.0, owner="rho", fixed=True
    )
    model = DecayModel(
        channel,
        [
            Resonance(
                "rho",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                parent_radius=shared_radius,
                spin=1,
            ),
            NonResonant(RealImag(1.0, 0.0)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=45,
    )
    spec = model_to_spec(model)
    registry = {shared_radius.name: shared_radius}
    restored = model_from_spec(spec, parameter_registry=registry)
    restored_radius = next(
        p for p in restored.parameters if p.name == shared_radius.name
    )
    assert restored_radius is shared_radius


def test_toy_mc_normalization_cannot_be_exported():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    base = DecayModel(
        channel,
        [Resonance("rho", pair=(0, 1), coefficient=RealImag(1.0, 0.0), spin=1)],
        normalization_method="square-dalitz",
        normalization_resolution=45,
    )
    toy_sample = base.generate_phase_space(500)
    model = DecayModel(
        channel,
        [Resonance("rho", pair=(0, 1), coefficient=RealImag(1.0, 0.0), spin=1)],
        normalization_sample=toy_sample,
    )
    with pytest.raises(ValueError, match="toy-mc"):
        model_to_spec(model)


def test_generic_components_lineshapes_and_coefficients_round_trip():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    knots = tuple(0.28 + 0.02 * i for i in range(6))
    model = DecayModel(
        channel,
        [
            Resonance(
                "kpi_lass",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                lineshape=LASS(),
                mass=0.8,
                width=0.05,
                spin=0,
            ),
            Resonance(
                "f0_980",
                pair=(0, 1),
                coefficient=CPRealImag(0.5, -0.2, dx=0.01, dy=0.02),
                lineshape=Flatte.f0_980(),
                mass=0.965,
                width=0.05,
                spin=0,
            ),
            Resonance(
                "qmi_swave",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                lineshape=QMI(
                    knots=knots,
                    magnitudes=tuple(1.0 for _ in knots),
                    phases=tuple(0.1 * i for i in range(len(knots))),
                    interpolation="linear",
                ),
                mass=0.9,
                width=0.1,
                spin=0,
            ),
            NonResonant(RealImag(0.3, 0.1)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=45,
    )
    restored = model_from_spec(model_to_spec(model))

    sample = model.generate_phase_space(1500)
    original = model.intensity(sample.as_dict())
    reloaded = restored.intensity(sample.as_dict())
    assert jnp.allclose(original, reloaded)


def test_model_to_spec_accepts_a_fit_session_directly():
    model = _floating_rho_model()
    sample = model.generate_phase_space(500)
    session = FitSession(model, sample)

    restored = model_from_spec(model_to_spec(session))

    original = model.intensity(sample.as_dict())
    reloaded = restored.intensity(sample.as_dict())
    assert jnp.allclose(original, reloaded)


def test_export_model_rejects_a_cp_fit_session():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    cp = CPRealImag(1.0, 0.0)
    plus, minus = (
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        for q in (1, -1)
    )
    session = CPFitSession(
        plus, minus, plus.generate_phase_space(50), minus.generate_phase_space(40)
    )
    with pytest.raises(TypeError, match="export_cp_models"):
        model_to_spec(session)


def test_cp_models_round_trip_preserves_shared_coefficient_identity():
    dx = Parameter.coefficient("dx", 0.05, owner="NR", bounds=(-0.8, 0.8))
    cp = CPRealImag(1.0, 0.0, dx, 0.0)
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    plus, minus = (
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        for q in (1, -1)
    )

    spec = cp_models_to_spec(plus, minus)
    restored_plus, restored_minus = cp_models_from_spec(spec)

    plus_dx = next(p for p in restored_plus.parameters if p.name == "dx")
    minus_dx = next(p for p in restored_minus.parameters if p.name == "dx")
    assert plus_dx is minus_dx

    plus_sample = plus.generate_phase_space(500)
    minus_sample = minus.generate_phase_space(500)
    assert jnp.allclose(
        plus.intensity(plus_sample.as_dict()),
        restored_plus.intensity(plus_sample.as_dict()),
    )
    assert jnp.allclose(
        minus.intensity(minus_sample.as_dict()),
        restored_minus.intensity(minus_sample.as_dict()),
    )


def test_export_import_cp_models_file_round_trip(tmp_path):
    dx = Parameter.coefficient("dx", 0.05, owner="NR", bounds=(-0.8, 0.8))
    cp = CPRealImag(1.0, 0.0, dx, 0.0)
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    plus, minus = (
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        for q in (1, -1)
    )
    data = plus.generate_phase_space(60, seed=1), minus.generate_phase_space(40, seed=2)
    session = CPFitSession(plus, minus, *data)

    path = tmp_path / "cp_models.json"
    export_cp_models(session, path)
    restored_plus, restored_minus = import_cp_models(path)
    restored_session = CPFitSession(restored_plus, restored_minus, *data)

    result = restored_session.fit()
    assert result.valid


def test_export_model_after_a_fit_still_writes_initial_values():
    """Parameter/DecayModel are frozen: fitting never mutates model.parameters."""
    model = _floating_rho_model()
    initial = {p.name: p.value for p in model.parameters}

    sample = model.generate_phase_space(300, seed=3)
    session = FitSession(model, sample)
    session.fit(ncall=5)  # a couple of steps is enough to move values off their start

    assert {p.name: p.value for p in model.parameters} == initial
    spec = model_to_spec(model)
    resonance_spec = next(c for c in spec["components"] if c["type"] == "Resonance")
    coefficient_fields = resonance_spec["fields"]["coefficient"]["fields"]
    exported_x = coefficient_fields["x"]["parameter"]["value"]
    assert exported_x == pytest.approx(initial["rho.x"])


def test_model_with_fitted_values_bakes_in_values_not_yet_on_the_model():
    model = _floating_rho_model()
    fitted = {
        "rho.mass": 0.771234,
        "rho.width": 0.149876,
        "rho.x": 0.95,
        "rho.y": -0.1,
    }
    initial = {p.name: p.value for p in model.parameters}
    for name, value in fitted.items():
        assert initial[name] != pytest.approx(value)

    updated = model_with_fitted_values(model, fitted)

    updated_by_name = {p.name: p.value for p in updated.parameters}
    for name, value in fitted.items():
        assert updated_by_name[name] == pytest.approx(value)
    # the original model instance is untouched
    assert {p.name: p.value for p in model.parameters} == initial

    restored = model_from_spec(model_to_spec(updated))
    for name, value in fitted.items():
        restored_param = next(p for p in restored.parameters if p.name == name)
        assert restored_param.value == pytest.approx(value)


def test_model_with_fitted_values_can_fix_the_baked_in_parameters():
    model = _floating_rho_model()
    updated = model_with_fitted_values(model, {"rho.mass": 0.772}, fix=True)
    mass_param = next(p for p in updated.parameters if p.name == "rho.mass")
    assert mass_param.fixed
    assert mass_param.value == pytest.approx(0.772)


def test_model_with_fitted_values_accepts_a_fit_session():
    model = _floating_rho_model()
    sample = model.generate_phase_space(300, seed=4)
    session = FitSession(model, sample)
    updated = model_with_fitted_values(session, {"rho.x": 0.5})
    updated_x = next(p for p in updated.parameters if p.name == "rho.x")
    assert updated_x.value == pytest.approx(0.5)


def test_cp_models_with_fitted_values_updates_the_shared_coefficient():
    dx = Parameter.coefficient("dx", 0.05, owner="NR", bounds=(-0.8, 0.8))
    cp = CPRealImag(1.0, 0.0, dx, 0.0)
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    plus, minus = (
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        for q in (1, -1)
    )

    updated_plus, updated_minus = cp_models_with_fitted_values(
        plus, {"dx": 0.3}, minus
    )
    plus_dx = next(p for p in updated_plus.parameters if p.name == "dx")
    minus_dx = next(p for p in updated_minus.parameters if p.name == "dx")
    assert plus_dx.value == pytest.approx(0.3)
    assert minus_dx is plus_dx
