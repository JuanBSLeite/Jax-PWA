"""Polar form-factor nonresonant amplitude for symmetric Dalitz plots."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import jax.numpy as jnp

from jaxpwa.fit.parameters import Parameter, ParameterKind

_INVARIANTS = ("s12", "s13", "s23")


@dataclass(frozen=True)
class PolarFormFactorSymNR:
    """Nonresonant polar form factor, ported from Laura++ ``LauPolarFormFactorSymNR``.

    Real amplitude for a Dalitz plot with two identical particles, depending on
    the two invariants ``s`` and ``t`` that pair each identical particle with
    the third one (GeV^2), and on ``lambda_`` (GeV)::

        PolarFFSymNR:        1/(1 + s/lambda^2) + 1/(1 + t/lambda^2)
        PolarFFSymNRNoInter: 1/(1 + min(s, t)/lambda^2)

    ``no_interference=True`` selects the second form. The model follows
    Nogueira et al., PRD 92, 054010 (2015), modified for symmetric Dalitz plots
    as in Laura++, and, as there, carries no ``1/sqrt(2)`` or other overall
    normalization. Laura++ evaluates ``s = m13^2`` and ``t = m23^2`` with the
    identical particles as tracks 1 and 2, which is the default
    ``invariants=("s13", "s23")`` for a channel ordered like
    ``("K+", "K+", "K-")``; for another ordering pass the two invariants that
    contain the non-identical particle. Laura++'s default ``lambda`` is 1.0
    GeV, within a [0, 10] range. A floating ``lambda_`` must use
    ``Parameter.dynamics(..., owner=component_name)`` so prepared caches
    recompute this component when it changes.

    We evaluate the algebraically equivalent ``lambda^2/(lambda^2+s)``
    terms to preserve finite derivatives in the zero-scale limit for
    positive physical invariants. Numeric initial scales must be positive
    and finite; a zero scale cannot provide a unit-normalized component.

    Attach it through ``DalitzAmplitude``: it is a function of two Dalitz
    invariants, not a one-dimensional lineshape, and the sum of both terms
    already includes the identical-particle symmetrization.
    """

    lambda_: object = 1.0
    no_interference: bool = False
    invariants: tuple[str, str] = ("s13", "s23")

    def __post_init__(self) -> None:
        if (
            isinstance(self.lambda_, Parameter)
            and not self.lambda_.fixed
            and self.lambda_.kind is not ParameterKind.DYNAMICS
        ):
            raise ValueError(
                "floating lambda_ must use Parameter.dynamics with the component owner"
            )
        invariants = tuple(self.invariants)
        if (
            len(invariants) != 2
            or invariants[0] == invariants[1]
            or any(name not in _INVARIANTS for name in invariants)
        ):
            raise ValueError(
                "invariants must be two distinct names among 's12', 's13', 's23'"
            )
        object.__setattr__(self, "invariants", invariants)
        if not isinstance(self.no_interference, bool):
            raise ValueError("no_interference must be a boolean")
        if isinstance(self.lambda_, Real) and (
            not isfinite(self.lambda_) or not self.lambda_ > 0
        ):
            raise ValueError("lambda_ must be positive and finite")

    def __call__(self, data, parameters=None):
        resolver = getattr(self.lambda_, "resolve", None)
        lam = self.lambda_ if resolver is None else resolver(parameters)
        lambda_sq = jnp.asarray(lam) ** 2
        s = jnp.asarray(data[self.invariants[0]])
        t = jnp.asarray(data[self.invariants[1]])
        if self.no_interference:
            magnitude = lambda_sq / (lambda_sq + jnp.minimum(s, t))
        else:
            magnitude = lambda_sq / (lambda_sq + s) + lambda_sq / (lambda_sq + t)
        return magnitude + 0.0j

    def compact_prepared_data(self, data):
        """Retain only the two invariants this amplitude is evaluated from."""

        if any(name not in data for name in self.invariants):
            return dict(data)
        return {name: data[name] for name in self.invariants}


__all__ = ["PolarFormFactorSymNR"]
