"""The pi-pi <-> K-Kbar rescattering shape used in the LHCb isobar model."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ...particle_properties import mass_gev
from ..context import ResonanceContext


@dataclass(frozen=True)
class PipiKKRescattering:
    """LHCb's phenomenological pi-pi to K-Kbar S-wave amplitude.

    The shape follows Eqs. (17)--(21) of Phys. Rev. D 101, 012006.  It is
    defined only in the 1.0--1.5 GeV mass window used by the isobar fit and is
    zero outside that window.

    ``convention='laura'`` uses the source factor in ``m**2`` and the
    overall phase ``i`` of Laura++ 3.8 ``LauRescatteringRes::amplitude``.
    The default ``'paper'`` retains the literal printed Eqs. (17)--(21).
    In either convention the explicit mass window is retained; Laura++ itself
    starts at the KK threshold, so set mass_min accordingly to reproduce it.
    The delta_*_squared parameters denote the source denominators: they have
    units GeV for 'paper' and GeV**2 (lambda**2) for 'laura'.

    Within the default 1.0--1.5 GeV window this reproduces
    ``LauRescatteringRes::amplitude`` (Laura++ 3.8) to float64 precision
    (checked against a literal line-by-line port, see
    ``docs/reviews/20260914_pipi_kk_rescattering.md``). ``convention='laura'``
    additionally reproduces two of its behaviours that only matter outside
    that window and are therefore invisible at the defaults: the eta0=1
    override below the 2*kaon_mass threshold and above ``m_prime`` (Laura++
    forces the inelasticity to zero there rather than trusting the continued
    formula), and safe handling of the eta0/cot(delta0) denominators that
    vanish exactly at ``m == 2*kaon_mass`` (Laura++'s own C++ produces NaN at
    that single point via ``0.0 * cos(NaN)``; this returns the well-defined
    zero that the formula's two-sided limit gives instead -- a deliberate,
    justified deviation at one measure-zero point, not a physics difference).
    The ``'paper'`` convention keeps the same safe denominators (avoiding the
    NaN is not convention-specific) but does not apply the eta0=1 override,
    since that is a Laura++ implementation choice absent from the printed
    equations.

    ``kaon_mass=None`` (default) takes the charged-kaon mass from the PDG
    table of the ``particle`` package, as Laura++ does with
    ``LauConstants::mK``.
    """

    mass_min: float = 1.0
    mass_max: float = 1.5
    delta_pipi_squared: float = 1.0
    delta_kk_squared: float = 1.0
    kaon_mass: float | None = None
    epsilon1: float = 2.4
    epsilon2: float = -5.5
    m_prime: float = 1.5
    m_f: float = 1.32
    m_s: float = 0.92
    c0: float = 1.3
    convention: str = "paper"

    def __post_init__(self) -> None:
        if self.kaon_mass is None:
            object.__setattr__(self, "kaon_mass", mass_gev("K+"))
        if self.convention not in {"paper", "laura"}:
            raise ValueError("rescattering convention must be 'paper' or 'laura'")
        if self.mass_min >= self.mass_max:
            raise ValueError("PipiKKRescattering requires mass_min < mass_max")
        if self.delta_pipi_squared <= 0.0 or self.delta_kk_squared <= 0.0:
            raise ValueError("rescattering source scales must be positive")

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("PipiKKRescattering is defined for a spin-0 S-wave")

        m = jnp.asarray(mass)
        tiny = jnp.finfo(m.dtype).tiny
        s = m**2
        s_safe = jnp.maximum(s, tiny)
        m_safe = jnp.maximum(m, tiny)

        # Laura++'s raw (signed) k2^2 = (s - 4*kaon_mass^2)/4 -- kept
        # unclamped, like LauRescatteringRes::amplitude's k2Square_s, so eta
        # uses the same continued value Laura does; k2_abs is |k2^2|**0.5,
        # matching its k2Abs_s (always real, even below threshold).
        k2_squared_signed = 0.25 * s - self.kaon_mass**2
        k2_abs = jnp.sqrt(jnp.abs(k2_squared_signed))
        # Guarded only to avoid an exact 0/0 at m == 2*kaon_mass (k2_abs is
        # also 0 there, so this never changes a nonzero result).
        k2_denominator = jnp.where(k2_squared_signed == 0.0, tiny, k2_squared_signed)

        eta = 1.0 - (
            self.epsilon1 * k2_abs / m_safe
            + self.epsilon2 * k2_squared_signed / s_safe
        ) * (self.m_prime**2 - s_safe) / s_safe
        if self.convention == "laura":
            below_threshold_or_above_mprime = (m < 2.0 * self.kaon_mass) | (m > self.m_prime)
            eta = jnp.where(below_threshold_or_above_mprime, 1.0, eta)
        cot_delta = self.c0 * (
            (s_safe - self.m_s**2) * (self.m_f**2 - s_safe) * k2_abs
        ) / (self.m_f**2 * m_safe * k2_denominator)
        exp_2i_delta = (cot_delta + 1j) / (cot_delta - 1j)
        scattering = jnp.sqrt(jnp.maximum(1.0 - eta**2, 0.0)) * exp_2i_delta
        source_variable = s if self.convention == "laura" else m
        source = 1.0 / (1.0 + source_variable / self.delta_pipi_squared) / (
            1.0 + source_variable / self.delta_kk_squared
        )
        value = source * scattering
        if self.convention == "laura":
            value = 1j * value
        inside = (m >= self.mass_min) & (m <= self.mass_max)
        return jnp.where(inside, value, 0.0j)


__all__ = ["PipiKKRescattering"]
