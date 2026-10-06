"""Mass-axis adaptive quadrature in Square-Dalitz coordinates.

The Square-Dalitz transformation uses one invariant mass as the m-prime
coordinate and a helicity angle as theta-prime. Narrow-resonance refinement is
therefore applied only along m-prime. The theta-prime axis keeps the ordinary
fixed Gauss-Legendre resolution.

This avoids the expensive 2D quadtree refinement previously used here while
retaining fine sampling for narrow structures aligned with the selected
Square-Dalitz mass pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jaxpwa._quadrature import legendre_rule
from jaxpwa.kinematics import (
    PhaseSpaceSample,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)


def _sorted_pair(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(pair))


@dataclass(frozen=True)
class AdaptiveSquareDalitzGrid:
    """Square-Dalitz quadrature with adaptive refinement only in mass."""

    mother_mass: float
    masses: tuple[float, float, float]
    narrow_resonances: tuple[tuple[tuple[int, int], float, float], ...]
    resolution: int = 1000
    pair: tuple[int, int] = (0, 1)
    window_n_widths: float = 5.0
    binning_factor: float = 100.0
    cell_order: int = 8
    max_depth: int = 12

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("resolution must be at least 2")
        if len(self.masses) != 3:
            raise ValueError("exactly three daughter masses are required")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        i, j = self.pair
        if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
            raise ValueError("pair must contain two distinct indices from 0, 1, 2")
        if self.window_n_widths <= 0.0:
            raise ValueError("window_n_widths must be positive")
        if self.binning_factor <= 0.0:
            raise ValueError("binning_factor must be positive")
        if self.cell_order < 2:
            raise ValueError("cell_order must be at least 2")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    @property
    def aligned_narrow_resonances(
        self,
    ) -> tuple[tuple[tuple[int, int], float, float], ...]:
        selected = _sorted_pair(self.pair)
        return tuple(
            (pair, float(mass), float(width))
            for pair, mass, width in self.narrow_resonances
            if _sorted_pair(pair) == selected and float(width) > 0.0
        )

    @property
    def crossed_narrow_resonances(
        self,
    ) -> tuple[tuple[tuple[int, int], float, float], ...]:
        selected = _sorted_pair(self.pair)
        return tuple(
            (pair, float(mass), float(width))
            for pair, mass, width in self.narrow_resonances
            if _sorted_pair(pair) != selected and float(width) > 0.0
        )

    @property
    def mass_range(self) -> tuple[float, float]:
        i, j = self.pair
        bachelor = next(index for index in range(3) if index not in self.pair)
        return (
            self.masses[i] + self.masses[j],
            self.mother_mass - self.masses[bachelor],
        )

    def _mass_from_mprime(self, mprime: float | np.ndarray) -> np.ndarray:
        m_min, m_max = self.mass_range
        mp = np.asarray(mprime, dtype=np.float64)
        return m_min + 0.5 * (m_max - m_min) * (1.0 + np.cos(np.pi * mp))

    def _needs_refinement(self, mp_low: float, mp_high: float) -> bool:
        endpoint_masses = self._mass_from_mprime(np.asarray([mp_low, mp_high]))
        mass_low = float(np.min(endpoint_masses))
        mass_high = float(np.max(endpoint_masses))
        mass_span = mass_high - mass_low

        for _, pole_mass, pole_width in self.aligned_narrow_resonances:
            window_low = pole_mass - self.window_n_widths * pole_width
            window_high = pole_mass + self.window_n_widths * pole_width
            if mass_high < window_low or mass_low > window_high:
                continue

            allowed_span = self.cell_order * pole_width / self.binning_factor
            if mass_span > allowed_span:
                return True

        return False

    def _mprime_cells(self) -> tuple[tuple[float, float], ...]:
        base_cells = max(1, int(math.ceil(self.resolution / self.cell_order)))
        edges = np.linspace(0.0, 1.0, base_cells + 1)

        stack: list[tuple[float, float, int]] = [
            (float(edges[index]), float(edges[index + 1]), 0)
            for index in range(base_cells)
        ]
        leaves: list[tuple[float, float]] = []
        unresolved: list[tuple[float, float]] = []

        while stack:
            low, high, depth = stack.pop()
            needs_more = self._needs_refinement(low, high)
            if depth < self.max_depth and needs_more:
                middle = 0.5 * (low + high)
                next_depth = depth + 1
                stack.append((low, middle, next_depth))
                stack.append((middle, high, next_depth))
            else:
                leaves.append((low, high))
                if needs_more:
                    unresolved.append((low, high))

        if unresolved:
            low, high = unresolved[0]
            raise ValueError(
                "AdaptiveSquareDalitzGrid could not refine "
                f"{len(unresolved)} m' cell(s) (e.g. [{low:.6g}, {high:.6g}]) down to "
                "the target resolution within max_depth="
                f"{self.max_depth}: the aligned narrow resonance(s) are too narrow "
                "relative to binning_factor/max_depth for this grid to resolve, and "
                "the normalization integral would otherwise be silently inaccurate "
                "there. Increase max_depth or cell_order, or decrease binning_factor, "
                "by constructing AdaptiveSquareDalitzGrid directly and passing its "
                "sample() as DecayModel(normalization_sample=...)."
            )

        leaves.sort()
        return tuple(leaves)

    @property
    def mprime_cell_count(self) -> int:
        return len(self._mprime_cells())

    @property
    def mprime_node_count(self) -> int:
        return self.cell_order * self.mprime_cell_count

    @property
    def estimated_points(self) -> int:
        return self.mprime_node_count * self.resolution

    def sample(self) -> PhaseSpaceSample:
        x, w = legendre_rule(self.cell_order)

        mp_parts: list[np.ndarray] = []
        mp_weight_parts: list[np.ndarray] = []
        for low, high in self._mprime_cells():
            mp_parts.append(0.5 * (high - low) * x + 0.5 * (high + low))
            mp_weight_parts.append(0.5 * (high - low) * w)

        mprime_axis = np.concatenate(mp_parts)
        mprime_weights = np.concatenate(mp_weight_parts)

        theta_x, theta_w = legendre_rule(self.resolution)
        theta_axis = 0.5 * (theta_x + 1.0)
        theta_weights = 0.5 * theta_w

        mprime, thetaprime = np.meshgrid(
            mprime_axis,
            theta_axis,
            indexing="ij",
        )
        wm, wt = np.meshgrid(
            mprime_weights,
            theta_weights,
            indexing="ij",
        )

        mp = jnp.asarray(mprime.reshape(-1))
        tp = jnp.asarray(thetaprime.reshape(-1))

        s12, s13, s23 = square_dalitz_to_invariants(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )

        raw_weights = jnp.asarray((wm * wt).reshape(-1))
        weights = jacobian * raw_weights * float(mp.size)

        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=weights,
        )


__all__ = ["AdaptiveSquareDalitzGrid"]
