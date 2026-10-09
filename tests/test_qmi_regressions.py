"""Regressions for QMI preparation isolation and parameter validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    QMI,
    DecayChannel,
    DecayModel,
    Parameter,
    QMIPixel,
    RealImag,
    Resonance,
    enable_x64,
)

enable_x64()


@pytest.mark.parametrize("interpolation", ["linear", "cubic", "hermite", "natural"])
@pytest.mark.parametrize("floating", [False, True])
def test_distinct_qmi_grids_are_independent_of_component_order(interpolation, floating):
    components = []
    for name, knots in [("q1", (0.3, 0.7, 1.6)), ("q2", (0.3, 0.6, 1.2, 1.6))]:
        real = (1.0, 0.2, 0.8) if name == "q1" else (0.9, -0.2, 0.4, 0.7)
        imaginary = (0.0, 0.3, -0.2) if name == "q1" else (0.2, 0.1, -0.3, 0.2)
        first = Parameter.dynamics(f"{name}.x", real[0], owner=name, fixed=not floating)
        components.append(
            Resonance(
                name,
                (0, 1),
                RealImag(1.0, 0.0),
                mass=1.0,
                width=0.0,
                spin=0,
                lineshape=QMI(
                    knots=knots,
                    real_parts=(first, *real[1:]),
                    imaginary_parts=imaginary,
                    interpolation=interpolation,
                ),
                normalize_component=False,
            )
        )
    reference = None
    for ordered in (components, components[::-1]):
        model = DecayModel(
            DecayChannel("D+", ("pi-", "pi+", "pi+")),
            ordered,
            normalize_components=False,
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        data = model.generate_phase_space(30, seed=46)
        cache = model.prepare_cache(data)
        values = {"q1.x": 1.0, "q2.x": 0.9}
        actual, norm = cache.evaluate(values)
        expected = model.intensity(data.as_dict(), values)
        expected_norm = jnp.mean(
            model.normalization_sample.weights
            * model.intensity(model.normalization_sample.as_dict(), values)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(norm, expected_norm, rtol=1e-11, atol=1e-11)
        if reference is not None:
            np.testing.assert_allclose(actual, reference[0], rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(norm, reference[1], rtol=1e-11, atol=1e-11)
        reference = actual, norm
        if floating:

            def objective(x, cache=cache, values=values, size=data.size):
                intensity, normalization = cache.evaluate({**values, "q2.x": x})
                return -jnp.sum(jnp.log(intensity)) + size * jnp.log(normalization)

            gradient = jax.grad(objective)(0.9)
            step = 1e-5
            finite = (objective(0.9 + step) - objective(0.9 - step)) / (2 * step)
            np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_qmi_rejects_nonfinite_knots_and_qmi_pixel_edges(bad):
    with pytest.raises(ValueError, match="finite"):
        QMI(
            knots=(0.3, bad, 1.6),
            real_parts=(1.0, 2.0, 3.0),
            imaginary_parts=(0.0, 0.0, 0.0),
        )
    for xedges, yedges in [((0.2, bad), (0.2, 1.0)), ((0.2, 1.0), (0.2, bad))]:
        with pytest.raises(ValueError, match="finite"):
            QMIPixel(xedges, yedges, ((1.0,),), ((0.0,),))


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("second_group", [False, True])
def test_qmi_rejects_generic_free_parameters_in_both_groups(polar, second_group):
    groups = [(1.0, 0.5), (0.0, 0.2)]
    groups[int(second_group)] = (Parameter("node", 0.4, owner="qmi"), 0.2)
    kwargs = (
        dict(magnitudes=groups[0], phases=groups[1])
        if polar
        else dict(real_parts=groups[0], imaginary_parts=groups[1])
    )
    with pytest.raises(ValueError, match=r"Parameter.dynamics"):
        QMI(knots=(0.3, 1.6), **kwargs)


def test_qmi_pixel_rejects_generic_free_parameters():
    for mag, phase in [
        (((Parameter("a", 1.0),),), ((0.0,),)),
        (((1.0,),), ((Parameter("phi", 0.0),),)),
    ]:
        with pytest.raises(ValueError, match=r"Parameter.dynamics"):
            QMIPixel((0.2, 1.0), (0.2, 1.0), mag, phase)


def test_fixed_generic_nodes_remain_supported():
    fixed = Parameter("fixed", 1.0, fixed=True)
    QMI((0.3, 1.6), magnitudes=(fixed, 0.5), phases=(0.0, 0.2))
    QMIPixel((0.2, 1.0), (0.2, 1.0), ((fixed,),), ((0.0,),))
