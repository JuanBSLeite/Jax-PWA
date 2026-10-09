import jax.numpy as jnp

from jaxpwa import (
    DalitzAmplitude,
    DecayChannel,
    DecayModel,
    Parameter,
    QMIPixel,
    RealImag,
    enable_x64,
    physical_bin_mask,
)


enable_x64()


def _data(s12, s13):
    return {"s12": jnp.asarray(s12), "s13": jnp.asarray(s13)}


def test_qmi_pixel_none_is_piecewise_constant_per_bin():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="none",
    )
    mag, phase = model.interpolated_magnitude_phase(
        _data([0.2, 0.2, 1.2, 1.2], [0.2, 1.2, 0.2, 1.2])
    )
    assert bool(jnp.allclose(mag, jnp.asarray([1.0, 2.0, 3.0, 4.0])))
    assert bool(jnp.allclose(phase, jnp.asarray([0.0, 0.1, 0.2, 0.3])))


def test_qmi_pixel_linear_interpolates_between_bin_centers():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 3.0), (3.0, 5.0)),
        phases=((0.0, 0.2), (0.2, 0.4)),
        interpolation="linear",
    )
    mag, phase = model.interpolated_magnitude_phase(_data([1.0], [1.0]))
    assert abs(float(mag[0]) - 3.0) < 1e-12
    assert abs(float(phase[0]) - 0.2) < 1e-12


def test_qmi_pixel_cubic_reproduces_all_bin_center_values():
    edges = (0.0, 1.0, 2.0, 3.0, 4.0)
    magnitudes = tuple(tuple(float(i + 2 * j + 1) for j in range(4)) for i in range(4))
    phases = tuple(tuple(float(0.1 * i - 0.05 * j) for j in range(4)) for i in range(4))
    model = QMIPixel(
        s12_edges=edges,
        s13_edges=edges,
        magnitudes=magnitudes,
        phases=phases,
        interpolation="cubic",
    )
    centers = jnp.asarray([0.5, 1.5, 2.5, 3.5])
    xx, yy = jnp.meshgrid(centers, centers, indexing="ij")
    mag, phase = model.interpolated_magnitude_phase(_data(xx.ravel(), yy.ravel()))
    assert bool(jnp.allclose(mag.reshape(4, 4), jnp.asarray(magnitudes), atol=1e-12))
    assert bool(jnp.allclose(phase.reshape(4, 4), jnp.asarray(phases), atol=1e-12))


def test_qmi_pixel_caches_fixed_interpolation_geometry():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="linear",
        active_mask=((True, False), (True, True)),
    )
    first_x = model._x_centers_fixed
    first_y = model._y_centers_fixed
    first_sources = model._ghost_sources
    model.interpolated_magnitude_phase(_data([0.25, 1.25], [0.25, 1.25]))
    assert model._x_centers_fixed is first_x
    assert model._y_centers_fixed is first_y
    assert model._ghost_sources is first_sources
    assert first_sources == (0, 0, 2, 3)


def test_qmi_pixel_folded_is_symmetric_under_s12_s13_exchange():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.2), (0.4, 0.6)),
        folded=True,
    )
    first = model(_data([0.2], [1.2]))
    second = model(_data([1.2], [0.2]))
    assert bool(jnp.allclose(first, second, atol=1e-12))


def test_physical_bin_mask_keeps_endpoint_bins_and_rejects_external_cells():
    channel = DecayChannel("D_s+", ("pi-", "pi+", "pi+"))
    m1, m2, m3 = channel.daughter_masses
    smin = (m1 + m2) ** 2
    smax = (channel.parent_mass - m3) ** 2
    edges = tuple(float(v) for v in jnp.linspace(smin, smax, 9))
    mask = physical_bin_mask(
        edges,
        edges,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        folded=True,
        samples_per_bin=257,
    )
    assert any(row[-1] for row in mask)
    assert not mask[-1][-1]


def test_qmi_pixel_active_mask_zeroes_inactive_piecewise_bins():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="none",
        active_mask=((True, True), (False, True)),
    )
    mag, phase = model.interpolated_magnitude_phase(_data([1.2], [0.2]))
    assert float(mag[0]) == 0.0
    assert float(phase[0]) == 0.0


def test_qmi_pixel_parameters_are_collected_and_change_decay_intensity():
    owner = "qmi_pixel"
    a00 = Parameter.dynamics("qmi_pixel.a00", 1.0, owner=owner, bounds=(0.0, None))
    p00 = Parameter.dynamics("qmi_pixel.p00", 0.0, owner=owner)
    field = QMIPixel(
        s12_edges=(0.0, 2.0, 4.0),
        s13_edges=(0.0, 2.0, 4.0),
        magnitudes=((a00, 1.2), (0.8, 1.1)),
        phases=((p00, 0.2), (0.4, 0.6)),
        interpolation="linear",
        folded=True,
    )
    decay = DecayModel(
        DecayChannel("D_s+", ("pi-", "pi+", "pi+")),
        [DalitzAmplitude(owner, field, RealImag(1.0, 0.0))],
        normalization_resolution=30,
    )
    names = {parameter.name for parameter in decay.parameters}
    assert {"qmi_pixel.a00", "qmi_pixel.p00"}.issubset(names)

    data = decay.normalization_sample.as_dict()
    nominal = decay.intensity(data, {"qmi_pixel.a00": 1.0, "qmi_pixel.p00": 0.0})
    shifted = decay.intensity(data, {"qmi_pixel.a00": 1.8, "qmi_pixel.p00": 0.5})
    assert bool(jnp.any(jnp.abs(nominal - shifted) > 1e-10))


def test_qmi_pixel_folded_rejects_mismatched_axis_edges():
    # `_coordinates()` looks up min(s12,s13) on the s12 grid and max(s12,s13)
    # on the s13 grid; if the two grids have different ranges this silently
    # clamps whichever physical value happens to be smaller/larger to the
    # narrower grid's boundary instead of raising, distorting the field near
    # and beyond that boundary. folded=True is documented (docs/lineshapes.md)
    # to always be used with identical s12_edges/s13_edges, so a mismatch is
    # rejected outright rather than silently misinterpreted.
    try:
        QMIPixel(
            s12_edges=(0.0, 1.0, 2.0),
            s13_edges=(0.0, 1.0, 2.0, 3.0),
            magnitudes=((1.0, 2.0), (3.0, 4.0)),
            phases=((0.0, 0.2), (0.4, 0.6)),
            folded=True,
        )
    except ValueError as exc:
        assert "folded" in str(exc)
    else:
        raise AssertionError("QMIPixel accepted folded=True with mismatched axis edges")


def test_qmi_pixel_rejects_unknown_interpolation_mode():
    try:
        QMIPixel(
            s12_edges=(0.0, 1.0),
            s13_edges=(0.0, 1.0),
            magnitudes=((1.0,),),
            phases=((0.0,),),
            interpolation="spline",
        )
    except ValueError as exc:
        assert "interpolation" in str(exc)
    else:
        raise AssertionError("QMIPixel accepted an unknown interpolation mode")


def test_qmi_pixel_compact_prepared_data_drops_unrelated_state():
    model = QMIPixel(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="linear",
    )
    data = _data([1.2], [0.4])
    shared = dict(data)
    shared["__kin_p1_p2_p3_mass"] = jnp.asarray([1.5])
    shared["unrelated_component_table"] = jnp.asarray([[1.0, 2.0]])

    compact = model.compact_prepared_data(shared)

    assert set(compact) == {"s12", "s13"}
    assert bool(jnp.allclose(model(compact), model(shared)))


def test_qmi_pixel_compact_prepared_data_falls_back_without_coordinates():
    model = QMIPixel(
        s12_edges=(0.0, 1.0),
        s13_edges=(0.0, 1.0),
        magnitudes=((1.0,),),
        phases=((0.0,),),
    )
    data = {"other": jnp.asarray([1.0])}
    assert model.compact_prepared_data(data) == data


def test_resolved_direct_dynamics_delegates_compaction_to_qmi_pixel():
    owner = "qmi_pixel"
    field = QMIPixel(
        s12_edges=(0.0, 2.0, 4.0),
        s13_edges=(0.0, 2.0, 4.0),
        magnitudes=((1.0, 1.2), (0.8, 1.1)),
        phases=((Parameter.dynamics("qmi_pixel.p00", 0.0, owner=owner), 0.2), (0.4, 0.6)),
        interpolation="linear",
    )
    decay = DecayModel(
        DecayChannel("D_s+", ("pi-", "pi+", "pi+")),
        [DalitzAmplitude(owner, field, RealImag(1.0, 0.0))],
        normalization_resolution=30,
    )
    (component,) = decay.amplitude_model.components

    data = _data([1.2], [0.4])
    shared = dict(data)
    shared["unrelated_component_table"] = jnp.asarray([[1.0, 2.0]])
    compact = component.function.compact_prepared_data(shared)

    assert set(compact) == {"s12", "s13"}
