"""Weighted unbinned likelihood objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

Parameters = Mapping[str, Array | float]
LogPDF = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class WeightedUnbinnedNLL:
    """Weighted objective ``-sum_i w_i log p(x_i)``.

    Negative finite weights are permitted, which is useful for sWeight/COW and
    other externally weighted objectives. The low-level class defines only the
    weighted estimating objective. High-level :class:`jaxpwa.workflow.FitSession`
    can apply the fixed-weight Godambe score-outer-product covariance with
    ``fit(weights=..., covariance="sandwich")`` or the historical squared-weight
    Hessian correction with ``covariance="sumw2"`` (``"sweight"`` alias).

    A zero-weight event contributes exactly zero, even where ``logpdf`` is
    ``-inf``. The mask cannot reach inside ``logpdf``'s autodiff, so a gradient
    stays finite there only if ``logpdf`` itself uses a safe logarithm
    (``log(where(p > 0, p, 1))``), as Jax-PWA's own signal densities do.
    """

    logpdf: LogPDF
    data: dict[str, Array]
    weights: Array

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("weighted likelihood data must be non-empty")
        size = int(jnp.asarray(next(iter(self.data.values()))).shape[0])
        weights = jnp.asarray(self.weights)
        if weights.shape != (size,):
            raise ValueError(f"weights must have shape ({size},), got {weights.shape}")
        if jnp.iscomplexobj(weights):
            raise ValueError("weights must be real")
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("weights must be finite")
        object.__setattr__(self, "weights", weights)

    def __call__(self, parameters: Parameters) -> Array:
        values = jnp.asarray(self.logpdf(self.data, parameters))
        if values.shape != self.weights.shape:
            raise ValueError(
                f"logpdf must return shape {self.weights.shape}, got {values.shape}"
            )
        if jnp.iscomplexobj(values):
            raise ValueError("logpdf must return real values")
        # A zero-weight event contributes exactly zero, including at a physical
        # PDF zero where log(p)=-inf. Mask before multiplication to avoid 0*inf.
        active = self.weights != 0
        finite = jnp.isfinite(values)
        valid = jnp.all(finite | ~active)
        terms = jnp.where(
            active & finite,
            self.weights * jnp.where(finite, values, 0.0),
            0.0,
        )
        total = -jnp.sum(terms)
        return jnp.where(valid & jnp.isfinite(total), total, jnp.inf)


def _sandwich_covariance(
    estimating_hessian,
    variability_matrix,
    *,
    label: str,
) -> np.ndarray:
    """Return ``A^-1 B A^-1`` with validation and symmetric solves."""

    hessian = np.asarray(estimating_hessian, dtype=float)
    variability = np.asarray(variability_matrix, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("estimating_hessian must be a square matrix")
    if variability.shape != hessian.shape:
        raise ValueError(
            "variability_matrix must have the same shape as estimating_hessian"
        )
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(variability)):
        raise ValueError(f"{label} covariance matrices must be finite")

    hessian = 0.5 * (hessian + hessian.T)
    variability = 0.5 * (variability + variability.T)
    try:
        left = np.linalg.solve(hessian, variability)
        covariance = np.linalg.solve(hessian, left.T).T
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"estimating Hessian is singular; cannot compute {label} covariance"
        ) from exc
    return 0.5 * (covariance + covariance.T)


def sandwich_covariance_from_score_outer(
    weighted_hessian,
    score_outer,
) -> np.ndarray:
    r"""Return the Godambe/sandwich covariance for a weighted score equation.

    Let

    .. math::

       U(\\theta)=\\sum_i w_i s_i(\\theta), \\qquad
       s_i(\\theta)=\\partial_\\theta \\log p_i(\\theta),

    and define

    .. math::

       A=-\\partial_\\theta U
        =-\\sum_i w_i\\,\\partial_\\theta^2\\log p_i,
       \\qquad
       B=\\sum_i w_i^2 s_i s_i^T.

    This function returns

    .. math::

       C_{\\rm sandwich}=A^{-1}BA^{-1}.

    ``score_outer`` is the empirical variability matrix ``B`` evaluated at the
    fitted point. This is the asymptotically correct M-estimator/Godambe
    covariance for fixed event weights under the usual regularity conditions.
    It does not by itself propagate uncertainty from the procedure that
    estimated the event weights. See C. Langenbruch, Eur. Phys. J. C 82
    (2022) 393, arXiv:1911.01303, especially Eqs. (18--21), and H. Dembinski
    et al., arXiv:2112.04574 for the COW/sWeight setting.
    """

    return _sandwich_covariance(
        weighted_hessian,
        score_outer,
        label="sandwich",
    )


def sweight_covariance_from_hessians(
    weighted_hessian,
    squared_weight_hessian,
) -> np.ndarray:
    r"""Return the squared-weight Hessian covariance correction.

    For a weighted log-likelihood, define

    .. math::

       H_w = -\\sum_i w_i\\,\\partial^2 \\log p_i, \\qquad
       H_{w^2} = -\\sum_i w_i^2\\,\\partial^2 \\log p_i.

    This function returns the commonly used SumW2/RooFit-style correction

    .. math::

       C = H_w^{-1} H_{w^2} H_w^{-1}.

    The two Hessians must be evaluated at the same fitted parameter point.
    This correction is widely used for sWeight fits, but is not the most
    general asymptotically-correct covariance when the score/Hessian identity
    fails, and it does not propagate uncertainty from the determination of the
    event weights themselves. See C. Langenbruch, Eur. Phys. J. C 82 (2022)
    393, arXiv:1911.01303, Eqs. (20--21).
    """

    hessian = np.asarray(weighted_hessian, dtype=float)
    squared = np.asarray(squared_weight_hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("weighted_hessian must be a square matrix")
    if squared.shape != hessian.shape:
        raise ValueError(
            "squared_weight_hessian must have the same shape as weighted_hessian"
        )
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(squared)):
        raise ValueError("sWeight covariance Hessians must be finite")
    return _sandwich_covariance(
        hessian,
        squared,
        label="sWeight SumW2",
    )
