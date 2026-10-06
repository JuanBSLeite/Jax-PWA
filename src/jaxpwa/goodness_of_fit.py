"""Goodness-of-fit tests for unbinned amplitude fits.

Two independent methods are provided, both following M. Williams, "How good
are your fits? Unbinned multivariate goodness-of-fit tests in high energy
physics" (arXiv:1006.3019), which studies exactly the Dalitz-plot-analysis use
case this module targets:

* :func:`chi2_from_histograms` -- the ordinary binned Pearson chi2 test
  (Williams Sec. 3.1, Eq. 3.2), applied to data/model histograms the caller
  has already built (e.g. via a reweighted-MC projection).
* :func:`point_to_point_dissimilarity` -- the unbinned point-to-point
  dissimilarity (PPD) test (Williams Sec. 3.3, Eqs. 3.12-3.13), with a
  permutation-test p-value (Williams Appendix C).

Both are physics-agnostic: they consume plain arrays, not any
:class:`~jaxpwa.workflow.FitSession` object. The session-level
convenience methods that build those arrays from a fit live on
:class:`~jaxpwa.workflow.FitSession` and
:class:`~jaxpwa.cp_workflow.CPFitSession`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc


def kdtree_local_residuals(observed_points, expected_points, expected_weights=None, *, k=50):
    """Compute adaptive k-neighbour residuals for DP diagnostics.

    For every observed point, a radius is chosen from its ``k``th nearest
    expected point. The observed and weighted expected populations inside
    that ball are compared with a Pearson pull. Works in any coordinate pair,
    including ``(s13,s23)`` and ``(m',theta')``.
    """
    from scipy.spatial import cKDTree
    observed = np.asarray(observed_points, dtype=float)
    expected = np.asarray(expected_points, dtype=float)
    if observed.ndim != 2 or expected.ndim != 2 or observed.shape[1] != expected.shape[1]:
        raise ValueError("point arrays must have shape (N, D) with matching D")
    if not observed.shape[0] or not expected.shape[0]:
        raise ValueError("point arrays must be non-empty")
    k = min(max(int(k), 1), expected.shape[0])
    weights = np.ones(expected.shape[0]) if expected_weights is None else np.asarray(expected_weights, dtype=float)
    if weights.shape != (expected.shape[0],) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("expected_weights must be finite and non-negative")
    tree_e, tree_o = cKDTree(expected), cKDTree(observed)
    distances = tree_e.query(observed, k=k)[0]
    # SciPy squeezes the neighbour axis for k=1, including when the reference
    # sample itself contains only one point and caps the requested k above.
    radii = np.asarray(distances) if k == 1 else distances[:, -1]
    observed_count = np.asarray([len(tree_o.query_ball_point(p, r)) for p, r in zip(observed, radii)], dtype=float)
    expected_count = np.asarray([weights[idx].sum() for p, r in zip(observed, radii) for idx in [tree_e.query_ball_point(p, r)]], dtype=float)
    expected_count *= observed.shape[0] / max(float(weights.sum()), np.finfo(float).tiny)
    expected_count = np.maximum(expected_count, np.finfo(float).tiny)
    return (observed_count - expected_count) / np.sqrt(expected_count), observed_count, expected_count, radii


@dataclass(frozen=True)
class BinnedChi2Result:
    """Binned Pearson chi2 goodness-of-fit result.

    When the model's free parameters were obtained from an *unbinned*
    maximum-likelihood fit (always the case for a
    :class:`~jaxpwa.workflow.FitSession` fit) rather than by
    minimizing this chi2 itself, the exact number of degrees of freedom is not
    known. Williams (Sec. 3.1) shows only the bound
    ``chi2(dof=n_bins-n_free_parameters-1) <= chi2 <= chi2(dof=n_bins-1)``
    holds, so this result reports both bounding degrees of freedom and their
    corresponding p-values rather than a single (potentially wrong) dof/p-value
    pair. ``p_value_min`` (the smaller p-value, from ``dof_min``) and
    ``p_value_max`` (from ``dof_max``) bracket the true p-value: if even
    ``p_value_max`` is small the fit can be safely rejected, and if even
    ``p_value_min`` is large it can be safely retained.
    """

    chi2: float
    n_bins: int
    n_free_parameters: int
    dof_min: int
    dof_max: int
    p_value_min: float
    p_value_max: float
    observed: np.ndarray
    expected: np.ndarray
    pulls: np.ndarray
    edges: tuple[np.ndarray, ...]


def _chi2_survival(chi2: float, dof: int) -> float:
    """Upper-tail chi2 p-value via the regularized incomplete gamma function.

    ``gammaincc(dof/2, chi2/2)`` is exactly the chi2 survival function; using
    ``jax.scipy.special`` avoids adding a ``scipy`` dependency for this one
    call (the project already uses ``jax.scipy.special`` elsewhere, e.g.
    ``discriminants.py``, ``resolution/convolution.py``).
    """

    if dof <= 0:
        return float("nan")
    return float(gammaincc(0.5 * dof, 0.5 * chi2))


def chi2_from_histograms(
    observed,
    expected,
    *,
    n_free_parameters: int = 0,
    edges: tuple[np.ndarray, ...] | None = None,
    observed_variance=None,
) -> BinnedChi2Result:
    """Binned Pearson chi2 test between observed and expected bin counts.

    ``observed``/``expected`` may be 1D or 2D (any shape, as long as they
    match); ``edges`` should be the corresponding ``np.histogram``/
    ``np.histogram2d`` bin edges for downstream plotting and is stored as-is.

    Bins with ``expected <= 0`` cannot contribute a finite Pearson term and
    are dropped from the chi2 sum and from ``n_bins`` (their ``pulls`` entry
    is ``nan``); a model that predicts zero density anywhere real data is
    observed is separately a sign of a badly-chosen model or binning, not
    something this function can average over.

    ``observed_variance`` (same shape) replaces the Poisson variance
    ``expected`` in each bin's denominator. For an event-weighted histogram
    (sWeights/COW) pass the per-bin sum of squared weights: ``observed`` is
    then the per-bin sum of weights and the statistic is
    ``sum (observed - expected)^2 / sum(w^2)``. That variance is an estimate
    from the bin's own events, so the chi2 distribution of the statistic is
    asymptotic and needs well-populated bins; bins with a non-positive
    variance are dropped like bins with ``expected <= 0``.
    """

    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape")
    if n_free_parameters < 0:
        raise ValueError("n_free_parameters must be non-negative")

    if observed_variance is None:
        variance = expected
    else:
        variance = np.asarray(observed_variance, dtype=float)
        if variance.shape != observed.shape:
            raise ValueError("observed_variance must have the shape of observed")
    occupied = (expected > 0) & (variance > 0)
    n_bins = int(np.sum(occupied))
    if n_bins == 0:
        raise ValueError("no bins with positive expected counts")

    residual = observed - expected
    chi2 = float(np.sum(residual[occupied] ** 2 / variance[occupied]))

    pulls = np.full(observed.shape, np.nan)
    pulls[occupied] = residual[occupied] / np.sqrt(variance[occupied])

    dof_min = n_bins - n_free_parameters - 1
    dof_max = n_bins - 1
    p_value_min = _chi2_survival(chi2, dof_min)
    p_value_max = _chi2_survival(chi2, dof_max)

    return BinnedChi2Result(
        chi2=chi2,
        n_bins=n_bins,
        n_free_parameters=int(n_free_parameters),
        dof_min=dof_min,
        dof_max=dof_max,
        p_value_min=p_value_min,
        p_value_max=p_value_max,
        observed=observed,
        expected=expected,
        pulls=pulls,
        edges=tuple(np.asarray(e) for e in edges) if edges is not None else (),
    )


@dataclass(frozen=True)
class PointToPointResult:
    """Point-to-point dissimilarity (PPD) goodness-of-fit result.

    ``p_value`` is estimated by the permutation test of Williams Appendix C
    and is therefore itself a Monte Carlo estimate: ``p_value == 0.0`` only
    means no permutation reached ``statistic`` among ``n_permutations``
    trials, not that the true p-value is exactly zero. Increase
    ``n_permutations`` for a tighter estimate near small p-values.
    """

    statistic: float
    p_value: float
    n_permutations: int
    sigma_bar: float
    n_data: int
    n_reference: int


def point_to_point_dissimilarity(
    data_xy,
    reference_xy,
    data_density,
    reference_density,
    *,
    sigma_bar: float = 0.01,
    phase_space_area: float,
    n_permutations: int = 200,
    seed: int = 0,
    max_total_events: int = 5_000,
) -> PointToPointResult:
    """Unbinned point-to-point dissimilarity test (Williams Eqs. 3.12-3.13).

    ``data_xy``/``reference_xy`` are ``(n_data, 2)``/``(n_reference, 2)``
    coordinate arrays; ``reference_xy`` should be a large (``n_reference >>
    n_data``) sample drawn from the fitted density ``f0``, and
    ``data_density``/``reference_density`` must be ``f0`` evaluated at the
    corresponding points (any consistent overall normalization of ``f0`` is
    fine: it is absorbed into the tunable nuisance parameter ``sigma_bar``,
    exactly as Williams intends -- see ``docs/goodness_of_fit.md``).

    The test statistic is

    .. code-block:: text

        T = (1/n_d^2) * sum_{i<j in data} psi(x_i, x_j)
            - (1/(n_d*n_mc)) * sum_{i in data, j in reference} psi(x_i, x_j)

        psi(x_i, x_j) = exp(-|x_i-x_j|^2 / (2*sigma(x_i)*sigma(x_j)))
        sigma(x) = sigma_bar / (f0(x) * phase_space_area)

    Larger ``T`` means worse agreement between data and ``f0``. Because
    ``T``'s null distribution is not known analytically, its p-value is
    estimated with the permutation test of Williams Appendix C: the pooled
    data+reference sample is repeatedly relabeled into a random size-``n_data``
    "data" subset, and the p-value is the fraction of those relabelings whose
    recomputed statistic is at least as large as the observed one.

    Since ``sigma(x_i)`` depends only on position (through ``f0``), not on
    the data/reference label, the full pairwise kernel matrix is built once
    and every permutation reduces to one matrix-vector product against it --
    this is the JAX-idiomatic reformulation of the permutation test, not a
    change to the statistic itself.

    This is ``O(n^2)`` in memory and compute, where
    ``n = n_data + n_reference`` -- the default ``max_total_events=5_000``
    already means multiple dense ``n x n`` float64 arrays of order 200 MB
    each, so raising it is a real memory commitment, not a formality. It
    raises a clear error rather than attempting a huge dense array; lower
    ``mc_size``/subsample ``data`` if this is hit -- see
    ``docs/goodness_of_fit.md``.
    """

    data_xy = np.asarray(data_xy, dtype=float)
    reference_xy = np.asarray(reference_xy, dtype=float)
    data_density = np.asarray(data_density, dtype=float)
    reference_density = np.asarray(reference_density, dtype=float)
    if (
        data_xy.ndim != 2
        or reference_xy.ndim != 2
        or data_xy.shape[1] != reference_xy.shape[1]
    ):
        raise ValueError(
            "data_xy and reference_xy must be (N, D) arrays with matching D"
        )
    n_data = data_xy.shape[0]
    n_reference = reference_xy.shape[0]
    if data_density.shape != (n_data,) or reference_density.shape != (n_reference,):
        raise ValueError("density arrays must have one value per event")
    if n_data < 2 or n_reference < 1:
        raise ValueError("point_to_point_dissimilarity requires at least 2 data events")
    if phase_space_area <= 0 or not np.isfinite(phase_space_area):
        raise ValueError("phase_space_area must be finite and positive")
    if sigma_bar <= 0:
        raise ValueError("sigma_bar must be positive")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    n_total = n_data + n_reference
    if n_total > max_total_events:
        raise ValueError(
            f"pooled sample size {n_total} exceeds "
            f"max_total_events={max_total_events}; point_to_point_dissimilarity is "
            "O(n^2) in memory -- lower mc_size or subsample data, or raise "
            "max_total_events if you accept the cost"
        )
    if np.any(data_density <= 0) or np.any(reference_density <= 0):
        raise ValueError("f0 density must be strictly positive at every event")

    coordinates = jnp.asarray(np.concatenate([data_xy, reference_xy], axis=0))
    density = jnp.asarray(np.concatenate([data_density, reference_density]))
    sigma = sigma_bar / (density * phase_space_area)

    # |x_i - x_j|^2 via norms + one matmul, avoiding an (n, n, D) intermediate.
    squared_norm = jnp.sum(coordinates * coordinates, axis=1)
    cross_term = coordinates @ coordinates.T
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * cross_term
    kernel = jnp.exp(-squared_distance / (2.0 * sigma[:, None] * sigma[None, :]))
    kernel = kernel * (1.0 - jnp.eye(n_total))
    row_sum = jnp.sum(kernel, axis=1)

    def _statistic(label: jnp.ndarray) -> jnp.ndarray:
        quadratic = jnp.dot(label, kernel @ label)
        linear = jnp.dot(label, row_sum)
        data_data_term = quadratic / (2.0 * n_data**2)
        data_reference_term = (linear - quadratic) / (n_data * n_reference)
        return data_data_term - data_reference_term

    observed_label = jnp.concatenate([jnp.ones(n_data), jnp.zeros(n_reference)])
    statistic = float(_statistic(observed_label))

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_permutations)

    def _permuted_label(permutation_key):
        order = jax.random.permutation(permutation_key, n_total)
        selected = order[:n_data]
        return jnp.zeros(n_total).at[selected].set(1.0)

    labels = jax.vmap(_permuted_label)(keys)
    permuted_statistics = jax.vmap(_statistic)(labels)
    p_value = float(jnp.mean(permuted_statistics >= statistic))

    return PointToPointResult(
        statistic=statistic,
        p_value=p_value,
        n_permutations=int(n_permutations),
        sigma_bar=float(sigma_bar),
        n_data=int(n_data),
        n_reference=int(n_reference),
    )


__all__ = [
    "BinnedChi2Result",
    "PointToPointResult",
    "chi2_from_histograms",
    "point_to_point_dissimilarity",
]
