"""The coherent rho--omega mixing model used in the LHCb isobar fit."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

from ...particle_properties import mass_gev, width_gev
from ..context import ResonanceContext
from .gounaris_sakurai import GounarisSakurai


@dataclass(frozen=True)
class RhoOmegaMixing:
    """Effective rho or omega line shape from the paper's Eq. (15).

    The full mixed amplitude is

    ``Rrho * (1 + Romega * Delta * zeta) / (1 - Delta**2 * Rrho * Romega)``.

    For fit fractions the paper separates it into two effective shapes.  This
    class returns ``Rrho / denominator`` for ``component='rho'`` and
    ``Delta * Rrho * Romega / denominator`` for ``component='omega'``.  The
    published Cartesian coefficients then multiply the two shapes separately.

    Masses and widths left at ``None`` (default) are the PDG values of
    ``rho(770)0`` and ``omega(782)`` from the ``particle`` package; pass them
    explicitly to reproduce a publication that used other values.
    """

    component: str = "rho"
    rho_mass: float | None = None
    rho_width: float | None = None
    omega_mass: float | None = None
    omega_width: float | None = None
    mixing_delta: float = 0.00215

    def __post_init__(self) -> None:
        for field_name, lookup, particle in (
            ("rho_mass", mass_gev, "rho(770)0"),
            ("rho_width", width_gev, "rho(770)0"),
            ("omega_mass", mass_gev, "omega(782)"),
            ("omega_width", width_gev, "omega(782)"),
        ):
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, lookup(particle))
        if self.component not in {"rho", "omega"}:
            raise ValueError("RhoOmegaMixing component must be 'rho' or 'omega'")

    def __call__(self, mass, context: ResonanceContext):
        m = jnp.asarray(mass)
        rho_context = replace(
            context,
            spin=1,
            pole_mass=self.rho_mass,
            pole_width=self.rho_width,
        )
        rho = GounarisSakurai()(m, rho_context)
        # Laura++ fixes the omega width to its pole value and disables its
        # individual barrier/spin factors before inserting it in the mixing
        # propagator (LauRhoOmegaMix::initialiseOmega).
        omega = 1.0 / (
            self.omega_mass**2
            - m**2
            - 1j * self.omega_mass * self.omega_width
        )
        delta = self.mixing_delta * (self.rho_mass + self.omega_mass)
        denominator = 1.0 - delta**2 * rho * omega
        if self.component == "rho":
            return rho / denominator
        return delta * rho * omega / denominator


__all__ = ["RhoOmegaMixing"]
