"""Tensor-product Gauss--Legendre integration on the Dalitz plot.

The quadrature is applied to the masses ``m13`` and ``m23`` rather than
directly to the squared invariant masses. The change of variables

    ds13 ds23 = 4 m13 m23 dm13 dm23

is included in the quadrature weights, and only nodes inside the physical
Dalitz boundary are retained.

The returned :class:`~jaxpwa.kinematics.PhaseSpaceSample` follows the
package-wide convention ``mean(sample.weights * f)``.  Therefore each retained
quadrature weight is multiplied by the number of retained points so that this
mean is exactly the Gauss--Legendre weighted sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jaxpwa._quadrature import legendre_rule
from jaxpwa.kinematics import PhaseSpaceSample


def _scaled_legendre(
    order: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes and weights scaled to ``[low, high]``."""

    nodes, weights = legendre_rule(order)
    half = 0.5 * (high - low)
    mean = 0.5 * (high + low)
    return mean + half * nodes, half * weights


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _s13_limits_numpy(
    s12: np.ndarray,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exact boundary in NumPy float64 before conversion to JAX."""

    m1, m2, m3 = masses
    root_s12 = np.sqrt(s12)
    e1 = (s12 + m1**2 - m2**2) / (2.0 * root_s12)
    e3 = (mother_mass**2 - s12 - m3**2) / (2.0 * root_s12)
    q = np.sqrt(np.maximum(_kallen(s12, m1**2, m2**2), 0.0)) / (
        2.0 * root_s12
    )
    p = np.sqrt(
        np.maximum(_kallen(mother_mass**2, s12, m3**2), 0.0)
    ) / (2.0 * root_s12)
    common = m1**2 + m3**2 + 2.0 * e1 * e3
    spread = 2.0 * q * p
    return common - spread, common + spread


@dataclass(frozen=True)
class DalitzGaussLegendreGrid:
    """Tensor-product Gauss--Legendre grid in ``m13`` and ``m23``.

    Parameters
    ----------
    mother_mass:
        Parent mass in GeV.
    masses:
        Daughter masses ``(m1, m2, m3)`` in GeV.
    bin_width:
        Target mass bin width in GeV. The default is 0.005 GeV (5 MeV).
    order_m13, order_m23:
        Optional explicit quadrature orders.  If omitted, each order is the
        corresponding kinematic mass range divided by ``bin_width`` (rounded
        upward).
    """

    mother_mass: float
    masses: tuple[float, float, float]
    bin_width: float = 0.005
    order_m13: int | None = None
    order_m23: int | None = None

    def __post_init__(self) -> None:
        if len(self.masses) != 3:
            raise ValueError(
                "DalitzGaussLegendreGrid requires exactly three daughter masses"
            )
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        if self.bin_width <= 0.0:
            raise ValueError("bin_width must be positive")
        if self.order_m13 is not None and self.order_m13 < 2:
            raise ValueError("order_m13 must be at least 2")
        if self.order_m23 is not None and self.order_m23 < 2:
            raise ValueError("order_m23 must be at least 2")

    @property
    def mass_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the physical bounding-box ranges for ``m13`` and ``m23``."""

        m1, m2, m3 = self.masses
        return (
            (m1 + m3, self.mother_mass - m2),
            (m2 + m3, self.mother_mass - m1),
        )

    @property
    def orders(self) -> tuple[int, int]:
        """Return the quadrature orders used on the two mass axes."""

        (m13_low, m13_high), (m23_low, m23_high) = self.mass_ranges
        n13 = self.order_m13 or max(2, math.ceil((m13_high - m13_low) / self.bin_width))
        n23 = self.order_m23 or max(2, math.ceil((m23_high - m23_low) / self.bin_width))
        return int(n13), int(n23)

    def sample(self) -> PhaseSpaceSample:
        """Build the physical Gauss--Legendre quadrature sample."""

        m1, m2, m3 = self.masses
        (m13_low, m13_high), (m23_low, m23_high) = self.mass_ranges
        n13, n23 = self.orders

        m13_axis, w13_axis = _scaled_legendre(n13, m13_low, m13_high)
        m23_axis, w23_axis = _scaled_legendre(n23, m23_low, m23_high)

        m13, m23 = np.meshgrid(m13_axis, m23_axis, indexing="ij")
        w13, w23 = np.meshgrid(w13_axis, w23_axis, indexing="ij")

        s13 = m13 * m13
        s23 = m23 * m23
        invariant_sum = self.mother_mass**2 + m1**2 + m2**2 + m3**2
        s12 = invariant_sum - s13 - s23

        s12_min = (m1 + m2) ** 2
        s12_max = (self.mother_mass - m3) ** 2
        s12_safe = np.clip(s12, s12_min, s12_max)
        low13, high13 = _s13_limits_numpy(
            s12_safe,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )

        scale = max(1.0, self.mother_mass**2)
        tol = 64.0 * np.finfo(float).eps * scale
        physical = (
            (s12 >= s12_min - tol)
            & (s12 <= s12_max + tol)
            & (s13 >= low13 - tol)
            & (s13 <= high13 + tol)
        )

        # Convert dm13 dm23 quadrature to ds13 ds23.
        quadrature_weights = 4.0 * m13 * m23 * w13 * w23

        s12 = s12[physical].reshape(-1)
        s13 = s13[physical].reshape(-1)
        s23 = s23[physical].reshape(-1)
        quadrature_weights = quadrature_weights[physical].reshape(-1)
        if s12.size == 0:
            raise RuntimeError(
                "Gauss--Legendre quadrature produced no physical Dalitz points"
            )

        # GridIntegrator computes mean(weights * f); multiplying raw quadrature
        # weights by N converts that convention into the required weighted sum.
        estimator_weights = quadrature_weights * float(s12.size)

        return PhaseSpaceSample(
            s12=jnp.asarray(s12),
            s13=jnp.asarray(s13),
            s23=jnp.asarray(s23),
            weights=jnp.asarray(estimator_weights),
        )


__all__ = ["DalitzGaussLegendreGrid"]
