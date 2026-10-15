"""Likelihoods expressed as densities in Square-Dalitz coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jaxpwa.kinematics.dalitz_grid import dalitz_s13_limits
from jaxpwa.kinematics.square_dalitz import (
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
)


def square_dalitz_pair(model, pair=None) -> tuple[int, int]:
    """The ordered daughter pair defining ``(m', theta')`` for ``model``.

    An explicit ``pair`` wins; otherwise a ``square-dalitz``-normalized model
    supplies its ``normalization_pair``. Any other model needs ``pair``.
    """
    if pair is not None:
        pair = tuple(int(index) for index in pair)
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= {0, 1, 2}:
            raise ValueError(
                "square_dalitz_pair must contain two distinct indices from 0, 1, 2"
            )
        return pair
    if getattr(model, "normalization_method", None) == "square-dalitz":
        return tuple(model.normalization_pair)
    raise ValueError(
        "Square-Dalitz coordinates need the ordered daughter pair defining m': pass "
        "square_dalitz_pair=(i, j), or use a model with normalization_method="
        "'square-dalitz', whose normalization_pair is then used"
    )


def square_dalitz_log_jacobian(model, data, *, pair=None) -> Array:
    r"""Per-event ``log |J(m', theta')|`` of ``model``'s Square-Dalitz map.

    ``|J|`` is the Jacobian of ``(m', theta') -> (s_a, s_b)`` for the ordered
    daughter ``pair`` (default: the ``normalization_pair`` of a
    ``square-dalitz``-normalized model, see :func:`square_dalitz_pair`),
    evaluated at the events of ``data`` (a ``PhaseSpaceSample`` or a mapping
    with ``s12``, ``s13``, ``s23``). A density in ``(m', theta')`` is the
    Dalitz-plot density times ``|J|``. Events on or outside the Square-Dalitz
    boundary (within floating-point roundoff) raise ``ValueError``. The
    invariants must be finite and the two that define ``(m', theta')``
    (``s_ij`` of ``pair`` and ``s_ik``) must lie strictly inside the physical
    region; validation happens before the coordinate transform clips its
    cosines. The invariant sum is not checked, so stored single-precision
    ``s12``/``s13``/``s23`` branches are accepted as on the Dalitz path.
    """
    pair = square_dalitz_pair(model, pair)
    values = data.as_dict() if hasattr(data, "as_dict") else data
    channel = model.channel
    mother_mass = float(channel.parent_mass)
    masses = tuple(float(m) for m in channel.daughter_masses)
    invariants = {
        key: jnp.asarray(values[key]) for key in ("s12", "s13", "s23")
    }
    if (
        len({v.shape for v in invariants.values()}) != 1
        or any(jnp.iscomplexobj(v) for v in invariants.values())
    ):
        raise ValueError("Dalitz invariants must be real arrays with matching shapes")
    i, j = pair
    k = next(index for index in range(3) if index not in pair)

    def invariant(a, b):
        a, b = sorted((a, b))
        return invariants[f"s{a + 1}{b + 1}"]

    s_ij, s_ik = invariant(i, j), invariant(i, k)
    low, high = dalitz_s13_limits(
        s_ij, mother_mass=mother_mass, masses=(masses[i], masses[j], masses[k])
    )
    total = mother_mass**2 + sum(m * m for m in masses)
    # Boundary arithmetic loses a few ulps. Reject that roundoff-sized layer
    # rather than relying on sin(pi) being exactly zero.
    tolerance = 32 * jnp.finfo(low.dtype).eps * total
    physical = (
        (s_ij > (masses[i] + masses[j])**2 + tolerance)
        & (s_ij < (mother_mass - masses[k])**2 - tolerance)
        & (s_ik > low + tolerance)
        & (s_ik < high - tolerance)
    )
    for value in invariants.values():
        physical = physical & jnp.isfinite(value)
    if not bool(jnp.all(physical)):
        raise ValueError(
            "events must have finite invariants strictly inside the physical "
            "Square-Dalitz boundary"
        )
    mprime, thetaprime = invariants_to_square_dalitz(
        values["s12"],
        values["s13"],
        values["s23"],
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )
    jacobian = square_dalitz_jacobian(
        mprime,
        thetaprime,
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )
    if not bool(jnp.all(jnp.isfinite(jacobian) & (jacobian > 0.0))):
        raise ValueError(
            "events on or outside the Square-Dalitz boundary have a zero Jacobian "
            "and no density in (m', theta')"
        )
    return jnp.log(jacobian)


def _square_dalitz_log_jacobian_sum(model, data, *, pair=None, weights=None):
    """Prepare the coordinate shift, ignoring zero-weight events before validation."""
    if weights is not None:
        weights = jnp.asarray(weights)
        active = weights != 0
        values = data.as_dict() if hasattr(data, "as_dict") else data
        data = {key: jnp.asarray(values[key])[active] for key in ("s12", "s13", "s23")}
        weights = weights[active]
    log_jacobian = square_dalitz_log_jacobian(model, data, pair=pair)
    if weights is not None:
        log_jacobian = weights * log_jacobian
    return jnp.sum(log_jacobian)


@dataclass(frozen=True)
class SquareDalitzNLL:
    r"""Express a Dalitz-plot NLL as an NLL in Square-Dalitz coordinates.

    With every event density written per ``dm' dtheta'`` instead of per
    ``ds_a ds_b``, each event's density gains the factor ``|J_i|``, so

    ``NLL_square(theta) = NLL_dalitz(theta) - sum_i w_i log |J_i|``,

    where ``w_i`` are the event weights of a weighted (sWeight/COW) NLL and 1
    otherwise; ``log_jacobian_sum`` is that precomputed sum. This is exact for
    mixtures (signal plus backgrounds, extended or not) as long as every
    component density is a Dalitz-plot density, which is the package
    convention (Square-Dalitz histogram backgrounds with
    ``divide_jacobian=True``). The Jacobian does not depend on the fit
    parameters, so minimum, Hessian and covariance are those of the wrapped
    NLL; only the NLL value changes.
    """

    nll: object
    log_jacobian_sum: object

    def __call__(self, parameters: Mapping[str, object]) -> Array:
        return jnp.asarray(self.nll(parameters)) - self.log_jacobian_sum


COORDINATES = ("dalitz", "square-dalitz")


def validate_session_coordinates(
    coordinates, *, models, efficiencies=(), background_shapes=(), pair=None
):
    """Check a session's ``coordinates`` and, for Square-Dalitz, its inputs.

    In Square-Dalitz coordinates every component density gains the same
    ``|J|``, which is exact only if efficiencies are dimensionless and
    background shapes are densities in the ordinary Dalitz measure. A
    Square-Dalitz histogram background is a density per ``dm' dtheta'`` and
    must therefore set ``divide_jacobian=True`` (``b / |J|``, normalized to
    ``int b dm' dtheta'``); a Square-Dalitz histogram efficiency must not.
    """
    if coordinates not in COORDINATES:
        raise ValueError(
            f"coordinates must be one of {COORDINATES}, got {coordinates!r}"
        )
    if coordinates == "dalitz":
        if pair is not None:
            raise ValueError(
                "square_dalitz_pair is only used with coordinates='square-dalitz'"
            )
        return
    from jaxpwa.square_histograms import (
        SquareDalitzHistogramBackground,
        SquareDalitzHistogramEfficiency,
    )

    for model in models:
        square_dalitz_pair(model, pair)
    for efficiency in efficiencies:
        if (
            isinstance(efficiency, SquareDalitzHistogramEfficiency)
            and efficiency.divide_jacobian
        ):
            raise ValueError(
                "a SquareDalitzHistogramEfficiency with divide_jacobian=True is not a "
                "dimensionless efficiency; build it with divide_jacobian=False"
            )
    for shape in background_shapes:
        if (
            isinstance(shape, SquareDalitzHistogramBackground)
            and not shape.divide_jacobian
        ):
            raise ValueError(
                "in Square-Dalitz coordinates a SquareDalitzHistogramBackground "
                "must set divide_jacobian=True: its histogram is a density per "
                "dm' dtheta', and the session treats every background shape as a "
                "Dalitz-plot density (b / |J|), so that signal and backgrounds "
                "share the same |J| factor"
            )


__all__ = ["SquareDalitzNLL", "square_dalitz_log_jacobian", "square_dalitz_pair"]
