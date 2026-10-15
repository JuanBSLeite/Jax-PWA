"""BaBar Flatte convention used for f0(980)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ...particle_properties import mass_gev
from ..context import ResonanceContext


@dataclass(frozen=True)
class BaBarFlatte:
    """Flatte form used in BaBar B± -> K± pi∓ pi± (arXiv:0803.4451).

    The paper uses

    ``Gamma_pipi = g_pi * [1/3 rho(pi0 pi0) + 2/3 rho(pi+ pi-)]``
    ``Gamma_KK   = g_K  * [1/2 rho(K+ K-) + 1/2 rho(K0 K0)]``

    with ``rho(h h) = sqrt(1 - 4 m_h^2 / m^2)`` analytically continued below
    threshold. The defaults are the BES couplings used by BaBar,
    ``g_pi = 0.165 GeV`` and ``g_K = 4.21*g_pi``. Daughter masses left at
    ``None`` (default) are taken from the PDG table of the ``particle``
    package.
    """

    g_pi: float = 0.165
    g_k: float = 4.21 * 0.165
    mpi0: float | None = None
    mpip: float | None = None
    mkp: float | None = None
    mk0: float | None = None

    def __post_init__(self) -> None:
        for field_name, particle in (
            ("mpi0", "pi0"), ("mpip", "pi+"), ("mkp", "K+"), ("mk0", "K0"),
        ):
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, mass_gev(particle))
        if self.g_pi < 0.0 or self.g_k < 0.0:
            raise ValueError("BaBarFlatte couplings must be non-negative")
        if min(self.mpi0, self.mpip, self.mkp, self.mk0) <= 0.0:
            raise ValueError("BaBarFlatte daughter masses must be positive")

    @staticmethod
    def _rho(mass, daughter_mass):
        m = jnp.asarray(mass)
        argument = (1.0 - 4.0 * daughter_mass**2 / m**2).astype(jnp.complex128)
        return jnp.sqrt(argument)

    def widths(self, mass):
        """Return the (Gamma_pipi, Gamma_KK) coupled-channel widths at each mass."""
        gamma_pi = self.g_pi * (
            (1.0 / 3.0) * self._rho(mass, self.mpi0)
            + (2.0 / 3.0) * self._rho(mass, self.mpip)
        )
        gamma_k = self.g_k * (
            0.5 * self._rho(mass, self.mkp)
            + 0.5 * self._rho(mass, self.mk0)
        )
        return gamma_pi, gamma_k

    def __call__(self, mass, context: ResonanceContext):
        gamma_pi, gamma_k = self.widths(mass)
        m = jnp.asarray(mass)
        m0 = jnp.asarray(context.pole_mass)
        return 1.0 / (m0**2 - m**2 - 1j * m0 * (gamma_pi + gamma_k))


__all__ = ["BaBarFlatte"]
