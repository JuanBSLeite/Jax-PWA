"""Generic one-dimensional PDF convolution for detector-resolution effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erf

from jaxpwa._quadrature import legendre_rule

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters | None):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def _gauss_legendre_nodes(low: float, high: float, order: int) -> tuple[Array, Array]:
    if high <= low:
        raise ValueError("integration interval requires low < high")
    if order < 2:
        raise ValueError("Gauss-Legendre order must be at least two")
    nodes, weights = legendre_rule(order)
    scale = 0.5 * (high - low)
    shift = 0.5 * (high + low)
    return jnp.asarray(scale * nodes + shift), jnp.asarray(scale * weights)


def _gauss_hermite_nodes(order: int) -> tuple[Array, Array]:
    if order < 2:
        raise ValueError("Gauss-Hermite order must be at least two")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return jnp.asarray(nodes), jnp.asarray(weights)


@dataclass(frozen=True)
class GaussianResolution1D:
    """Gaussian conditional resolution density ``R(x_obs | x_true)``.

    ``bias`` shifts the reconstructed mean relative to the true value and both
    ``bias`` and ``sigma`` may be ordinary numbers or fit ``Parameter`` objects.
    The kernel itself is normalized on the full real line; ``interval_probability``
    supplies the exact probability retained by a finite observed fit window.

    For convolution numerators this kernel provides a Gauss--Hermite path local
    to each observed point. That avoids the poor resolution of a narrow Gaussian
    kernel on a single global true-variable quadrature grid.
    """

    sigma: object
    bias: object = 0.0
    floor: float = 0.0

    def __call__(
        self,
        observed: Array,
        true: Array,
        parameters: Parameters | None = None,
    ) -> Array:
        observed = jnp.asarray(observed)
        true = jnp.asarray(true)
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        bias = jnp.asarray(_resolve(self.bias, parameters))
        z = (observed - (true + bias)) / sigma
        raw = jnp.exp(-0.5 * z**2) / (jnp.sqrt(2.0 * jnp.pi) * sigma)
        valid = sigma > 0.0
        if self.floor > 0.0:
            raw = jnp.clip(raw, min=self.floor)
        return jnp.where(valid, raw, 0.0)

    def interval_probability(
        self,
        low: float,
        high: float,
        true: Array,
        parameters: Parameters | None = None,
    ) -> Array:
        """Probability to reconstruct an event at ``true`` inside ``[low, high]``."""

        if high <= low:
            raise ValueError("observed interval requires low < high")
        true = jnp.asarray(true)
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        bias = jnp.asarray(_resolve(self.bias, parameters))
        root2_sigma = jnp.sqrt(2.0) * sigma
        mean = true + bias
        upper = (high - mean) / root2_sigma
        lower = (low - mean) / root2_sigma
        probability = 0.5 * (erf(upper) - erf(lower))
        return jnp.where(sigma > 0.0, probability, 0.0)

    def convolve_pdf(
        self,
        pdf,
        observed: Array,
        *,
        true_low: float,
        true_high: float,
        nodes: Array,
        weights: Array,
        parameters: Parameters | None = None,
    ) -> Array:
        """Evaluate the raw Gaussian convolution with Gauss--Hermite quadrature."""

        observed = jnp.asarray(observed)
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        bias = jnp.asarray(_resolve(self.bias, parameters))
        true = observed[:, None] - bias + jnp.sqrt(2.0) * sigma * nodes[None, :]
        density = jnp.asarray(pdf(true, parameters))
        inside = (true >= true_low) & (true <= true_high)
        numerator = jnp.sum(
            weights[None, :] * jnp.where(inside, density, 0.0),
            axis=1,
        ) / jnp.sqrt(jnp.pi)
        return jnp.where(sigma > 0.0, numerator, 0.0)


@dataclass(frozen=True)
class ConvolvedPDF1D:
    """Numerically convolve a normalized 1D PDF with a resolution kernel.

    The true PDF must follow the existing discriminant convention
    ``pdf(x, parameters)``. The kernel must provide both
    ``kernel(observed, true, parameters)`` and
    ``kernel.interval_probability(low, high, true, parameters)``.

    A generic kernel numerator uses Gauss--Legendre quadrature over the true
    interval. Kernels may provide a specialized ``convolve_pdf`` method; the
    Gaussian kernel uses Gauss--Hermite quadrature centered on each observed
    value, which remains accurate for resolutions much narrower than the full
    true-variable interval.

    ``quadrature_order`` is purely numerical: it sets the number of nodes used
    by both the Gauss--Legendre normalization grid and the specialized
    Gauss--Hermite Gaussian-convolution grid.

    The result is normalized on the finite observed fit range. This
    normalization is parameter dependent whenever probability migrates across
    the observed boundaries.
    """

    pdf: object
    kernel: object
    true_low: float
    true_high: float
    observed_low: float
    observed_high: float
    quadrature_order: int = 96
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.true_high <= self.true_low:
            raise ValueError("ConvolvedPDF1D requires true_low < true_high")
        if self.observed_high <= self.observed_low:
            raise ValueError("ConvolvedPDF1D requires observed_low < observed_high")
        nodes, weights = _gauss_legendre_nodes(
            self.true_low, self.true_high, self.quadrature_order
        )
        hermite_nodes, hermite_weights = _gauss_hermite_nodes(self.quadrature_order)
        object.__setattr__(self, "_true_nodes", nodes)
        object.__setattr__(self, "_true_weights", weights)
        object.__setattr__(self, "_hermite_nodes", hermite_nodes)
        object.__setattr__(self, "_hermite_weights", hermite_weights)

    def normalization(self, parameters: Parameters | None = None) -> Array:
        """Return the retained-probability normalization on the observed range.

        Integrates ``pdf(true) * P(observed in range | true)`` over the true
        interval by Gauss-Legendre quadrature; parameter dependent whenever
        probability migrates across the observed boundaries.
        """

        nodes = self._true_nodes
        true_density = jnp.asarray(self.pdf(nodes, parameters))
        retained = jnp.asarray(
            self.kernel.interval_probability(
                self.observed_low,
                self.observed_high,
                nodes,
                parameters,
            )
        )
        return jnp.sum(self._true_weights * true_density * retained)

    def _numerator(self, x_flat: Array, parameters: Parameters | None) -> Array:
        specialized = getattr(self.kernel, "convolve_pdf", None)
        if specialized is not None:
            return jnp.asarray(
                specialized(
                    self.pdf,
                    x_flat,
                    true_low=self.true_low,
                    true_high=self.true_high,
                    nodes=self._hermite_nodes,
                    weights=self._hermite_weights,
                    parameters=parameters,
                )
            )

        nodes = self._true_nodes
        true_density = jnp.asarray(self.pdf(nodes, parameters))
        response = jnp.asarray(
            self.kernel(x_flat[:, None], nodes[None, :], parameters)
        )
        return jnp.sum(
            response * (self._true_weights * true_density)[None, :],
            axis=1,
        )

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        x_flat = jnp.ravel(x)
        numerator = self._numerator(x_flat, parameters)
        norm = self.normalization(parameters)
        inside = (
            (x_flat >= self.observed_low)
            & (x_flat <= self.observed_high)
            & (norm > 0.0)
        )
        values = jnp.where(
            inside,
            jnp.clip(numerator / norm, min=self.floor),
            0.0,
        )
        return jnp.reshape(values, x.shape)


__all__ = ["ConvolvedPDF1D", "GaussianResolution1D"]
