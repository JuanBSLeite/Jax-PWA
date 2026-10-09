"""Generic weighted three-body event container."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import jax.numpy as jnp
from jax import Array


class EventSample(Protocol):
    """Minimal dimension-independent interface used by integration and fitting."""

    weights: Array

    @property
    def size(self) -> int: ...

    def as_dict(self) -> dict[str, Array]: ...

    def take(self, indices: Array) -> EventSample: ...

    def validate_integration(self) -> None: ...


@dataclass(frozen=True)
class PhaseSpaceSample:
    """Three-body events, invariants, four-momenta and Monte Carlo weights."""

    s12: Array
    s13: Array
    s23: Array
    weights: Array
    p1: Array | None = None
    p2: Array | None = None
    p3: Array | None = None

    @property
    def size(self) -> int:
        """Number of events in the sample."""

        return int(self.s12.shape[0])

    def as_dict(self) -> dict[str, Array]:
        """Return invariants (and four-momenta, if present) as a flat dict."""

        data = {"s12": self.s12, "s13": self.s13, "s23": self.s23}
        if self.p1 is not None and self.p2 is not None and self.p3 is not None:
            data.update({"p1": self.p1, "p2": self.p2, "p3": self.p3})
        return data

    @classmethod
    def from_square_dalitz(
        cls,
        mprime,
        thetaprime,
        *,
        mother_mass: float,
        masses: tuple[float, float, float],
        pair: tuple[int, int],
        weights=None,
    ) -> PhaseSpaceSample:
        """Build a sample from Laura++ Square-Dalitz coordinates ``(m', theta')``.

        ``pair`` is the ordered daughter pair defining ``m'`` (the
        ``normalization_pair`` of a ``square-dalitz`` model). The invariants
        are what amplitudes are evaluated from; ``weights`` default to one.
        """
        from .square_dalitz import square_dalitz_to_invariants

        s12, s13, s23 = square_dalitz_to_invariants(
            jnp.asarray(mprime), jnp.asarray(thetaprime),
            mother_mass=mother_mass, masses=masses, pair=pair,
        )
        if weights is None:
            weights = jnp.ones_like(s12)
        return cls(s12=s12, s13=s13, s23=s23, weights=jnp.asarray(weights))

    def square_dalitz(
        self,
        *,
        mother_mass: float,
        masses: tuple[float, float, float],
        pair: tuple[int, int],
    ) -> tuple[Array, Array]:
        """Return the events' Laura++ Square-Dalitz coordinates ``(m', theta')``."""
        from .square_dalitz import invariants_to_square_dalitz

        return invariants_to_square_dalitz(
            self.s12, self.s13, self.s23,
            mother_mass=mother_mass, masses=masses, pair=pair,
        )

    def momentum_dict(self) -> dict[str, Array]:
        """Return ``{"p1", "p2", "p3"}`` four-momenta; raises if not stored."""

        if self.p1 is None or self.p2 is None or self.p3 is None:
            raise ValueError("This sample does not contain four-momenta")
        return {"p1": self.p1, "p2": self.p2, "p3": self.p3}

    @property
    def nbytes(self) -> int:
        """Approximate bytes occupied by the sample's array payloads."""

        arrays = (self.s12, self.s13, self.s23, self.weights, self.p1, self.p2, self.p3)
        return sum(
            int(jnp.asarray(array).size * jnp.asarray(array).dtype.itemsize)
            for array in arrays
            if array is not None
        )

    def without_momenta(self) -> PhaseSpaceSample:
        """Return a compact view containing only invariants and event weights."""

        if self.p1 is None and self.p2 is None and self.p3 is None:
            return self
        return PhaseSpaceSample(
            s12=self.s12,
            s13=self.s13,
            s23=self.s23,
            weights=self.weights,
        )

    def validate_integration(self) -> None:
        """Validate an ordinary, non-negative integration sample on the host.

        Proposal correctness is the caller's responsibility; it cannot be
        inferred from the observed points or unit weights.
        """
        if jnp.asarray(self.s12).ndim != 1 or self.size < 1:
            raise ValueError(
                "normalization_sample must contain a non-empty event vector"
            )
        for name in ("s12", "s13", "s23", "weights"):
            array = jnp.asarray(getattr(self, name))
            if array.shape != (self.size,):
                raise ValueError(
                    f"normalization_sample.{name} must have shape ({self.size},)"
                )
            if not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError(f"normalization_sample.{name} must be finite")
        weights = jnp.asarray(self.weights)
        if bool(jnp.any(weights < 0)) or not bool(jnp.any(weights > 0)):
            raise ValueError(
                "integration weights must be non-negative with positive support"
            )
        momenta = (self.p1, self.p2, self.p3)
        if any(p is not None for p in momenta):
            for p in momenta:
                if p is None or jnp.asarray(p).shape != (self.size, 4):
                    raise ValueError("integration momenta must all have shape (N, 4)")
                if not bool(jnp.all(jnp.isfinite(p))):
                    raise ValueError("integration momenta must be finite")

    def with_importance_weights(self, proposal_density: Array) -> PhaseSpaceSample:
        """Replace weights by 1/q for draws from the supplied proposal q.

        q must be normalized in the desired integration measure and have
        support everywhere the integrand contributes. Relative q gives only
        relative integrals; neither condition can be checked from these draws.
        Existing weights are replaced, not multiplied (avoids double weighting).
        """
        q = jnp.asarray(proposal_density)
        if q.shape != (self.size,) or not bool(jnp.all(jnp.isfinite(q) & (q > 0))):
            raise ValueError(
                "proposal_density must be finite, positive and have shape (N,)"
            )
        result = replace(self, weights=1.0 / q)
        result.validate_integration()
        return result

    def select_for_integration(self, mask: Array) -> PhaseSpaceSample:
        """Filter while preserving mean(weights * integrand * mask).

        Unlike data selection via take(), retained weights include N_kept/N.
        An empty selected domain cannot normalize a PDF and is rejected.
        """
        self.validate_integration()
        mask = jnp.asarray(mask)
        if mask.shape != (self.size,) or mask.dtype != jnp.bool_:
            raise ValueError(
                "integration selection must be a boolean vector of shape (N,)"
            )
        selected = self.take(jnp.flatnonzero(mask))
        if selected.size == 0:
            raise ValueError("integration selection retained no events")
        result = replace(
            selected, weights=selected.weights * (selected.size / self.size)
        )
        result.validate_integration()
        return result

    def take(self, indices: Array) -> PhaseSpaceSample:
        """Return the subsample at ``indices`` (invariants, weights, momenta)."""

        indices = jnp.asarray(indices)
        return PhaseSpaceSample(
            s12=self.s12[indices],
            s13=self.s13[indices],
            s23=self.s23[indices],
            weights=self.weights[indices],
            p1=None if self.p1 is None else self.p1[indices],
            p2=None if self.p2 is None else self.p2[indices],
            p3=None if self.p3 is None else self.p3[indices],
        )
