import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import QMIPixel, Parameter, enable_x64, physical_bin_mask
from jaxpwa.dynamics.qmi_pixel import _catmull_rom

enable_x64()


@pytest.mark.parametrize("phase", [False, True])
def test_inactive_cell_rejects_free_nodes(phase):
    free = Parameter.dynamics("dead", 1.0, owner="q")
    grid = ((1.0, free), (1.0, 1.0))
    fixed = ((1.0, 1.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="inactive cell"):
        QMIPixel(
            (0.0, 1.0, 2.0),
            (0.0, 1.0, 2.0),
            fixed if phase else grid,
            grid if phase else fixed,
            interpolation="none",
            active_mask=((True, False), (True, True)),
        )
    QMIPixel(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        ((1.0, Parameter("fixed", 1.0, fixed=True)), (1.0, 1.0)),
        fixed,
        interpolation="none",
        active_mask=((True, False), (True, True)),
    )


def test_mask_clips_external_bins_and_zero_edges():
    args = dict(mother_mass=2.0, masses=(0.1, 0.1, 0.1))
    assert physical_bin_mask((10.0, 11.0), (-100.0, 100.0), **args) == ((False,),)
    assert physical_bin_mask((-2.0, 0.0), (-100.0, 100.0), **args) == ((False,),)
    assert physical_bin_mask((0.0, 1.0), (0.0, 1.0), **args) == ((True,),)
    assert physical_bin_mask((0.0, 10.0), (-100.0, 100.0), **args) == ((True,),)
    assert physical_bin_mask(
        (0.0, 1.0), (0.0, 1.0), mother_mass=2.0, masses=(0.0, 0.0, 0.0)
    ) == ((True,),)


@pytest.mark.parametrize("edges", [(0.0, float("nan")), (1.0, 0.0), (1.0,)])
def test_mask_rejects_invalid_edges(edges):
    with pytest.raises(ValueError, match="edges"):
        physical_bin_mask(edges, (0.0, 1.0), mother_mass=2.0, masses=(0.1, 0.1, 0.1))


def _field(edges, values):
    return QMIPixel(edges, edges, values, ((0.0,) * 4,) * 4, interpolation="cubic")


def test_nonuniform_cubic_has_continuous_coordinate_derivatives():
    edges = (0.0, 1.0, 3.0, 6.0, 10.0)
    values = tuple(
        tuple(float((i + 1) ** 2 + j**2) for j in range(4)) for i in range(4)
    )
    field = _field(edges, values)
    for center in (2.0, 4.5):
        for axis in ("s12", "s13"):
            other = "s13" if axis == "s12" else "s12"

            def function(x, axis=axis, other=other):
                return jnp.real(field({axis: x, other: jnp.array(3.0)}))

            np.testing.assert_allclose(
                jax.grad(function)(center - 1e-7),
                jax.grad(function)(center + 1e-7),
                rtol=1e-6,
                atol=1e-6,
            )
    centers = jnp.array([0.5, 2.0, 4.5, 8.0])
    x, y = jnp.meshgrid(centers, centers, indexing="ij")
    np.testing.assert_allclose(
        jnp.real(field({"s12": x, "s13": y})), values, rtol=1e-12, atol=1e-12
    )


def test_uniform_cubic_preserves_previous_values_and_parameter_gradient():
    values = jnp.arange(16.0, dtype=float).reshape(4, 4) ** 2 / 30
    x, y = jnp.array([0.6, 1.8, 3.4]), jnp.array([0.7, 2.1, 3.3])

    def previous_one(xx, yy):
        i = jnp.clip(jnp.floor(xx - 0.5).astype(int), 0, 2)
        j = jnp.clip(jnp.floor(yy - 0.5).astype(int), 0, 2)
        ix = jnp.clip(jnp.arange(-1, 3) + i, 0, 3)
        iy = jnp.clip(jnp.arange(-1, 3) + j, 0, 3)
        patch = values[ix[:, None], iy[None, :]]
        along = jax.vmap(lambda c: _catmull_rom(*c, xx - (i + 0.5)), in_axes=1)(patch)
        return _catmull_rom(*along, yy - (j + 0.5))

    field = _field((0.0, 1.0, 2.0, 3.0, 4.0), values)
    np.testing.assert_allclose(
        jnp.real(field({"s12": x, "s13": y})), jax.vmap(previous_one)(x, y), atol=1e-12
    )

    def objective(v):
        changed = _field((0.0, 1.0, 3.0, 6.0, 10.0), values.at[1, 1].set(v))
        return jnp.sum(jnp.abs(changed({"s12": x, "s13": y})) ** 2)

    v = float(values[1, 1])
    h = 1e-5
    np.testing.assert_allclose(
        jax.grad(objective)(v),
        (objective(v + h) - objective(v - h)) / (2 * h),
        rtol=1e-6,
        atol=1e-7,
    )
