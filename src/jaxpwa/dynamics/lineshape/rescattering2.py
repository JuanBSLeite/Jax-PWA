"""Alternative Laura++ two-region rescattering S-wave."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ...particle_properties import mass_gev
from ..context import ResonanceContext


def _chebyshev_series(x, coefficients):
    """Evaluate sum_n c_n T_n(x) with a short stable recurrence."""

    x = jnp.asarray(x)
    coeffs = tuple(coefficients)
    if not coeffs:
        return jnp.zeros_like(x)
    result = jnp.asarray(coeffs[0]) * jnp.ones_like(x)
    if len(coeffs) == 1:
        return result

    t_nm2 = jnp.ones_like(x)
    t_nm1 = x
    result = result + jnp.asarray(coeffs[1]) * t_nm1
    for coefficient in coeffs[2:]:
        t_n = 2.0 * x * t_nm1 - t_nm2
        result = result + jnp.asarray(coefficient) * t_n
        t_nm2, t_nm1 = t_nm1, t_n
    return result


@dataclass(frozen=True)
class Rescattering2:
    """Port of the Laura++ LauRescattering2Res alternative rescattering model.

    The amplitude is

        A(m) = g_00(m) exp(i phi_00(m)) / (1 + m^2/Lambda^2),

    with separate Chebyshev expansions below and above 1.47 GeV, and
    g_00(m) = 0 below the charged-kaon threshold 2*m_K (``LauRescattering2Res::
    resAmp`` sets ``mag = 0`` there). The Chebyshev coefficients B/C/D/F are
    evaluated in degrees, exactly as in the Laura++ source; the whole phase
    phi_00(m) is converted to radians once, after evaluation, matching
    ``resAmp``'s ``phi00(...) * pi/180``. The default coefficients are those
    hard-coded in Laura++, including the ``1/(1+m^2/Lambda^2)`` suppression
    with ``Lambda = 1`` GeV. The first interval starts at the charged-kaon
    threshold, 2*m_K, and the second ends at 2.0 GeV. ``kaon_mass=None``
    (default) takes the charged-kaon mass from the PDG table of the
    ``particle`` package.

    The Laura++ source appears to contain a typo in initialise(): C0 and F0
    are formed from phi00(sqr_tmax[1]*sqr_tmax[1], 1) and
    g00(sqr_tmax[1]*sqr_tmax[1], 1), even though sqr_tmax stores masses and
    resAmp passes mass (not mass squared) to phi00/g00. Here we use the
    dimensionally and internally consistent transition mass itself, 1.47 GeV.
    This also makes the two Chebyshev regions continuous at the transition,
    which the literal squared-mass anchor does not.
    """

    B1: object = 23.6
    B2: object = 29.4
    B3: object = 0.6

    C1: object = 34.39
    C2: object = 4.4
    C3: object = -32.9
    C4: object = -16.0
    C5: object = 7.4

    D0: object = 0.59
    D1: object = -0.38
    D2: object = 0.12
    D3: object = -0.09

    F1: object = -0.043
    F2: object = -0.008
    F3: object = -0.028
    F4: object = 0.026

    lambda_scale: float = 1.0

    kaon_mass: float | None = None
    transition_mass: float = 1.47
    maximum_mass: float = 2.0
    threshold_phase_degrees: float = 226.5

    def __post_init__(self) -> None:
        if self.kaon_mass is None:
            object.__setattr__(self, "kaon_mass", mass_gev("K+"))
        threshold = 2.0 * float(self.kaon_mass)
        if threshold >= float(self.transition_mass):
            raise ValueError(
                "Rescattering2 requires 2*kaon_mass < transition_mass"
            )
        if float(self.transition_mass) >= float(self.maximum_mass):
            raise ValueError(
                "Rescattering2 requires transition_mass < maximum_mass"
            )

    @property
    def threshold_mass(self) -> float:
        """Charged-kaon threshold, 2*kaon_mass, below which the amplitude is zero."""
        return 2.0 * float(self.kaon_mass)

    def _scaled_mass(self, mass, lower: float, upper: float):
        return 2.0 * (jnp.asarray(mass) - lower) / (upper - lower) - 1.0

    def _low_phase_coefficients(self):
        # B0 and the phase Chebyshev series stay in degrees here, matching
        # Laura++'s ``B0_ = 226.5 + B1 - B2 + B3``. The degrees-to-radians
        # conversion is applied once, to the full phi_00(m), in ``phase()``.
        b0 = float(self.threshold_phase_degrees) + self.B1 - self.B2 + self.B3
        return (b0, self.B1, self.B2, self.B3)

    def _low_phase(self, mass):
        x_low = self._scaled_mass(
            mass, self.threshold_mass, float(self.transition_mass)
        )
        return _chebyshev_series(x_low, self._low_phase_coefficients())

    def _high_phase_coefficients(self):
        # Laura++ uses sqr_tmax[1]*sqr_tmax[1] here, but sqr_tmax stores
        # masses. Use the transition mass itself for a consistent definition.
        c0 = (
            self._low_phase(float(self.transition_mass))
            + self.C1
            - self.C2
            + self.C3
            - self.C4
            + self.C5
        )
        return (c0, self.C1, self.C2, self.C3, self.C4, self.C5)

    def _low_magnitude_coefficients(self):
        return (self.D0, self.D1, self.D2, self.D3)

    def _low_magnitude(self, mass):
        x_low = self._scaled_mass(
            mass, self.threshold_mass, float(self.transition_mass)
        )
        return _chebyshev_series(x_low, self._low_magnitude_coefficients())

    def _high_magnitude_coefficients(self):
        # Same correction as for C0 above.
        f0 = (
            self._low_magnitude(float(self.transition_mass))
            + self.F1
            - self.F2
            + self.F3
            - self.F4
        )
        return (f0, self.F1, self.F2, self.F3, self.F4)

    def phase(self, mass):
        """Return the Laura++ phase phi_00(m), in radians."""

        m = jnp.asarray(mass)
        x_high = self._scaled_mass(
            m, float(self.transition_mass), float(self.maximum_mass)
        )
        low = self._low_phase(m)
        high = _chebyshev_series(x_high, self._high_phase_coefficients())
        degrees = jnp.where(m <= self.transition_mass, low, high)
        return jnp.deg2rad(degrees)

    def magnitude(self, mass):
        """Return the signed Laura++ function g_00(m).

        This is the raw Chebyshev magnitude only: it does not apply the
        charged-kaon-threshold cut or the ``1/(1+m^2/Lambda^2)`` suppression
        that ``resAmp``/``__call__`` add on top of it.
        """

        m = jnp.asarray(mass)
        x_high = self._scaled_mass(
            m, float(self.transition_mass), float(self.maximum_mass)
        )
        low = self._low_magnitude(m)
        high = _chebyshev_series(x_high, self._high_magnitude_coefficients())
        return jnp.where(m <= self.transition_mass, low, high)

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("Rescattering2 is defined for a spin-0 S-wave")

        m = jnp.asarray(mass)
        magnitude = jnp.where(m < self.threshold_mass, 0.0, self.magnitude(m))
        suppression = 1.0 / (1.0 + (m / float(self.lambda_scale)) ** 2)
        value = magnitude * suppression * jnp.exp(1j * self.phase(m))
        inside = m < self.maximum_mass
        return jnp.where(inside, value, 0.0j)


__all__ = ["Rescattering2"]
