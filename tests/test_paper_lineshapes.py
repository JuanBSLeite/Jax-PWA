from dataclasses import replace

import jax.numpy as jnp

from jaxpwa import (
    GounarisSakurai,
    PipiKKRescattering,
    ResonanceContext,
    RhoOmegaMixing,
    SigmaPole,
    enable_x64,
)


enable_x64()


def _context(spin=0):
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=5.27934,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=spin,
        pole_mass=0.563,
        pole_width=0.350,
        resonance_radius=4.0,
        parent_radius=4.0,
    )


def test_sigma_pole_uses_the_paper_complex_pole_convention():
    context = _context()
    mass = jnp.asarray(0.8)
    pole = context.pole_mass - 1j * context.pole_width
    expected = 1.0 / (pole**2 - mass**2)

    value = SigmaPole()(mass, context)

    assert abs(complex(value) - expected) < 1e-14


def test_pipi_kk_rescattering_is_cut_to_the_paper_mass_window():
    model = PipiKKRescattering()
    context = _context()

    assert abs(complex(model(jnp.asarray(0.999), context))) == 0.0
    assert abs(complex(model(jnp.asarray(1.0), context))) > 0.0
    assert abs(complex(model(jnp.asarray(1.4), context))) > 0.0
    assert abs(complex(model(jnp.asarray(1.5), context))) == 0.0
    assert abs(complex(model(jnp.asarray(1.501), context))) == 0.0


def test_pipi_kk_rescattering_rejects_non_scalar_context():
    try:
        PipiKKRescattering()(jnp.asarray(1.2), _context(spin=1))
    except ValueError as exc:
        assert "spin-0" in str(exc)
    else:
        raise AssertionError("PipiKKRescattering accepted a non-scalar context")


def test_rescattering_laura_complex_reference_values():
    # Independent scalar evaluation of LauRescatteringRes::amplitude:
    # delta=atan(1/cotdelta), tau=(cos(2delta), sin(2delta)),
    # returned value=(-tauIm, tauRe)*NR1*NR2. Includes the phase crossing.
    mass = jnp.asarray([1.0, 1.1, 1.32, 1.49])
    expected = jnp.asarray([
        -0.14615559493806848 + 0.014175425798470658j,
        -0.10852898212231951 - 0.07046283953507726j,
        -0.0437459779265945j,
        0.004947864569887254 - 0.002740883023572237j,
    ])
    # The reference values were computed with kaon_mass=0.494 (the former
    # default); the class now defaults to the PDG K+ mass.
    actual = PipiKKRescattering(convention="laura", kaon_mass=0.494)(mass, _context())
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_rho_omega_mixing_returns_the_two_effective_terms():
    context = replace(
        _context(spin=1),
        pole_mass=0.7708,
        pole_width=0.1534,
    )
    model = RhoOmegaMixing(
        rho_mass=0.7708,
        rho_width=0.1534,
        omega_mass=0.78265,
        omega_width=0.00849,
    )
    rho_context = replace(context, pole_mass=model.rho_mass, pole_width=model.rho_width)
    rho = GounarisSakurai()(jnp.asarray(0.79), rho_context)
    omega = 1.0 / (
        model.omega_mass**2
        - 0.79**2
        - 1j * model.omega_mass * model.omega_width
    )
    delta = model.mixing_delta * (model.rho_mass + model.omega_mass)
    denominator = 1.0 - delta**2 * rho * omega

    rho_term = RhoOmegaMixing(
        component="rho",
        rho_mass=model.rho_mass,
        rho_width=model.rho_width,
        omega_mass=model.omega_mass,
        omega_width=model.omega_width,
        mixing_delta=model.mixing_delta,
    )(jnp.asarray(0.79), context)
    omega_term = replace(model, component="omega")(jnp.asarray(0.79), context)

    assert abs(complex(rho_term) - rho / denominator) < 1e-12
    assert abs(complex(omega_term) - delta * rho * omega / denominator) < 1e-12
