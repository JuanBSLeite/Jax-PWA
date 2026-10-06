"""Basic normalized PDFs for discriminating variables beyond the Dalitz plot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxpwa._quadrature import legendre_rule
from jaxpwa.dynamics.context import resolve_value

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters | None):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def jax_erf(x: Array) -> Array:
    from jax.scipy.special import erf as _erf

    return _erf(x)


def _gauss_legendre_nodes(low: float, high: float, order: int) -> tuple[Array, Array]:
    if high <= low:
        raise ValueError("integration interval requires low < high")
    if order < 2:
        raise ValueError("Gauss-Legendre order must be at least two")
    nodes, weights = legendre_rule(order)
    scale = 0.5 * (high - low)
    shift = 0.5 * (high + low)
    return jnp.asarray(scale * nodes + shift), jnp.asarray(scale * weights)


def _chebyshev_value(coefficients, t):
    value = jnp.ones_like(t)
    previous, current = jnp.ones_like(t), t
    for k, c in enumerate(coefficients, start=1):
        if k > 1:
            previous, current = current, 2.0 * t * current - previous
        value = value + c * current
    return value


def _chebyshev_nonnegative(coefficients):
    """Check the endpoints and all stationary points, independent of the data.

    Root finding only decides physical validity; derivatives of a valid PDF
    still use the ordinary polynomial and analytic normalization. Branching
    on the actual degree handles trailing zero coefficients under JIT.
    """
    coefficients = jax.lax.stop_gradient(coefficients)
    count = coefficients.size
    if count == 0:
        return jnp.asarray(True)
    degree = jnp.max(jnp.where(coefficients != 0, jnp.arange(1, count + 1), 0))

    def endpoints(c):
        return jnp.min(_chebyshev_value(c, jnp.asarray([-1.0, 1.0]))) >= 0

    branches = [endpoints, endpoints]
    for order in range(2, count + 1):
        # Fixed Chebyshev -> monomial derivative transform. Only this small
        # geometry constant is constructed with NumPy, never fitted values.
        columns = []
        for row in np.eye(order + 1):
            powers = np.polynomial.chebyshev.cheb2poly(row)
            padded = np.pad(powers, (0, order + 1 - len(powers)))
            columns.append(np.polynomial.polynomial.polyder(padded))
        transform = np.stack(columns, axis=1)

        def stationary(c, order=order, transform=transform):
            derivative = jnp.asarray(transform) @ jnp.r_[1.0, c[:order]]
            roots = jnp.roots(derivative[::-1], strip_zeros=False)
            inside = (jnp.abs(roots.imag) < 1e-10) & (jnp.abs(roots.real) <= 1)
            t = jnp.where(inside, roots.real, 0.0)
            values = _chebyshev_value(c, t)
            return endpoints(c) & jnp.all(jnp.where(inside, values >= 0, True))

        branches.append(stationary)
    # sum |c_k| <= 1 guarantees positivity and avoids root finding in the
    # common fit domain, since |T_k| <= 1 on the support.
    return jnp.all(jnp.isfinite(coefficients)) & jax.lax.cond(
        jnp.sum(jnp.abs(coefficients)) <= 1,
        lambda c: jnp.asarray(True),
        lambda c: jax.lax.switch(degree, branches, c),
        coefficients,
    )


@dataclass(frozen=True)
class Gaussian1D:
    """Gaussian PDF normalized on a finite interval."""

    mean: object
    sigma: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("Gaussian1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mu = jnp.asarray(_resolve(self.mean, parameters))
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        z = (x - mu) / sigma
        raw = jnp.exp(-0.5 * z**2) / (jnp.sqrt(2.0 * jnp.pi) * sigma)
        a = (self.low - mu) / (jnp.sqrt(2.0) * sigma)
        b = (self.high - mu) / (jnp.sqrt(2.0) * sigma)
        norm = 0.5 * (jax_erf(b) - jax_erf(a))
        inside = (x >= self.low) & (x <= self.high) & (sigma > 0.0) & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class BreitWigner1D:
    """Constant-width Breit-Wigner PDF normalized on a finite mass interval."""

    mean: object
    width: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("BreitWigner1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mean = jnp.asarray(_resolve(self.mean, parameters))
        width = jnp.asarray(_resolve(self.width, parameters))
        gamma = 0.5 * width
        raw = gamma / ((x - mean) ** 2 + gamma**2)
        norm = jnp.arctan((self.high - mean) / gamma) - jnp.arctan(
            (self.low - mean) / gamma
        )
        inside = (x >= self.low) & (x <= self.high) & (width > 0.0) & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class CrystalBall1D:
    """Crystal Ball PDF normalized on a finite interval.

    Follows the standard convention (matching ``scipy.stats.crystalball``):
    a Gaussian core for ``z = (x-mean)/sigma >= -alpha`` and a power-law tail
    on the left, ``alpha > 0`` and ``n > 1``. A negative ``alpha`` puts the
    tail on the right instead (the RooFit ``RooCBShape`` / LHCb convention
    in which the sign of ``alpha`` selects the side): the density is then
    the mirror image ``z -> -z`` with ``|alpha|``. Normalization is closed-form —
    the Gaussian half via ``erf`` and the power-law half via its elementary
    antiderivative — rather than numerical quadrature, so it stays cheap
    under a JAX-compiled unbinned likelihood with many events.
    """

    mean: object
    sigma: object
    alpha: object
    n: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("CrystalBall1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mu = jnp.asarray(_resolve(self.mean, parameters))
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        alpha = jnp.asarray(_resolve(self.alpha, parameters))
        n = jnp.asarray(_resolve(self.n, parameters))

        valid_shape = (
            jnp.isfinite(sigma)
            & jnp.isfinite(alpha)
            & jnp.isfinite(n)
            & (sigma > 0.0)
            & (alpha != 0.0)
            & (n > 1.0)
        )
        safe_sigma = jnp.where(valid_shape, sigma, 1.0)
        # side = +1: low-mass tail; side = -1: high-mass tail (mirror z -> -z).
        side = jnp.where(alpha < 0.0, -1.0, 1.0)
        safe_alpha = jnp.where(valid_shape, jnp.abs(alpha), 1.0)
        safe_n = jnp.where(valid_shape, n, 2.0)

        z_split = -safe_alpha

        def tail_log_ratio(z):
            # (b-z)/(n/alpha) = 1-alpha*(alpha+z)/n in the tail.
            # Combining the powers before evaluation avoids overflow in
            # (n/alpha)**n even for allowed fits, e.g. alpha=.05, n=100.
            tail_z = jnp.minimum(z, z_split)
            return jnp.log1p(-safe_alpha * (safe_alpha + tail_z) / safe_n)

        def gauss_antideriv(z: Array) -> Array:
            return jnp.sqrt(0.5 * jnp.pi) * jax_erf(z / jnp.sqrt(2.0))

        def power_antideriv(z: Array) -> Array:
            # Clamp into the tail (z <= -alpha, where b - z >= n/alpha > 0)
            # before the power: an out-of-tail argument would otherwise give
            # an infinite value whose NaN gradient survives the jnp.where below.
            exponent = -0.5 * safe_alpha**2 + (1.0 - safe_n) * tail_log_ratio(z)
            return jnp.exp(exponent) * safe_n / (safe_alpha * (safe_n - 1.0))

        # Interval in the mirrored variable t = side * z, where the tail is
        # always on the low side of t.
        t_a = side * (self.low - mu) / safe_sigma
        t_b = side * (self.high - mu) / safe_sigma
        z_low = jnp.minimum(t_a, t_b)
        z_high = jnp.maximum(t_a, t_b)

        power_hi = jnp.minimum(z_high, z_split)
        power_norm = jnp.where(
            power_hi > z_low, power_antideriv(power_hi) - power_antideriv(z_low), 0.0
        )
        gauss_lo = jnp.maximum(z_low, z_split)
        gauss_norm = jnp.where(
            z_high > gauss_lo, gauss_antideriv(z_high) - gauss_antideriv(gauss_lo), 0.0
        )
        norm = power_norm + gauss_norm

        z = side * (x - mu) / safe_sigma
        # Same clamp for the per-event tail: core events (z > -alpha) must not
        # evaluate the power law outside its domain (inf value -> NaN gradient).
        power_pdf = jnp.exp(-0.5 * safe_alpha**2 - safe_n * tail_log_ratio(z))
        gauss_pdf = jnp.exp(-0.5 * z**2)
        raw = jnp.where(z >= z_split, gauss_pdf, power_pdf) / safe_sigma

        valid = valid_shape & jnp.isfinite(norm) & (norm > 0.0)
        safe_norm = jnp.where(valid, norm, 1.0)
        inside = (x >= self.low) & (x <= self.high) & valid
        return jnp.where(inside, jnp.clip(raw / safe_norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class SumPDF1D:
    """Sum of normalized 1D PDFs with recursive fractions.

    For components ``p_1..p_k`` and fractions ``f_1..f_{k-1}``,

    ``P = f_1 p_1 + (1-f_1) f_2 p_2 + ... + (1-f_1)...(1-f_{k-1}) p_k``

    (the RooFit ``RooAddPdf`` recursive convention), so ``P`` integrates to
    one whenever every component does and each ``f_i`` lies in ``[0, 1]``.
    E.g. the LHCb B -> hhh signal model ``fG G + (1-fG) fCB CB1 +
    (1-fG)(1-fCB) CB2`` is ``SumPDF1D((G, CB1, CB2), (fG, fCB))``. Fractions
    may be numbers or fit ``Parameter`` objects.
    """

    pdfs: tuple
    fractions: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "pdfs", tuple(self.pdfs))
        object.__setattr__(self, "fractions", tuple(self.fractions))
        if len(self.pdfs) < 2:
            raise ValueError("SumPDF1D requires at least two component PDFs")
        if len(self.fractions) != len(self.pdfs) - 1:
            raise ValueError("SumPDF1D requires one fraction fewer than components")
        if any(not callable(pdf) for pdf in self.pdfs):
            raise TypeError("SumPDF1D component PDFs must be callable")
        for fraction in self.fractions:
            if isinstance(fraction, Real) and (
                not isfinite(fraction) or not 0 <= fraction <= 1
            ):
                raise ValueError("SumPDF1D fractions must be finite and in [0, 1]")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        total = 0.0
        remainder = 1.0
        valid = jnp.asarray(True)
        for pdf, fraction in zip(self.pdfs[:-1], self.fractions, strict=True):
            f = jnp.asarray(_resolve(fraction, parameters))
            physical = jnp.isfinite(f) & (f >= 0) & (f <= 1)
            valid = valid & physical
            f = jnp.where(physical, f, 0.0)
            total = total + remainder * f * pdf(x, parameters)
            remainder = remainder * (1.0 - f)
        total = total + remainder * self.pdfs[-1](x, parameters)
        return jnp.where(valid, total, jnp.nan)


@dataclass(frozen=True)
class LineshapeIntensity1D:
    """Normalize the intensity of an existing complex resonance lineshape.

    The wrapped lineshape follows ``lineshape(mass, ResonanceContext)`` and the
    returned PDF is ``|lineshape(m)|^2`` normalized on ``[low, high]``.
    ``from_context`` infers the full physical pair-mass interval,
    ``m1 + m2 <= m <= M_parent - m_bachelor``.

    ``quadrature_order`` is purely numerical and controls the Gauss--Legendre
    normalization grid. For a coherent model with interfering amplitudes,
    detector resolution must act on the full coherent intensity rather than on
    individual component intensities.
    """

    lineshape: object
    context: object
    low: float
    high: float
    quadrature_order: int = 256
    floor: float = 1e-300

    def __post_init__(self) -> None:
        nodes, weights = _gauss_legendre_nodes(
            self.low, self.high, self.quadrature_order
        )
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_weights", weights)

    @classmethod
    def from_context(
        cls,
        lineshape: object,
        context: object,
        *,
        quadrature_order: int = 256,
        floor: float = 1e-300,
        parameters: Parameters | None = None,
    ) -> LineshapeIntensity1D:
        """Use the full kinematic mass interval encoded by ``ResonanceContext``."""

        resolved = context.resolve(parameters)
        daughter1, daughter2 = resolved.daughter_masses
        low = float(daughter1 + daughter2)
        high = float(resolved.parent_mass - resolved.bachelor_mass)
        if high <= low:
            raise ValueError(
                "ResonanceContext has no physical two-body mass interval: "
                "parent_mass - bachelor_mass must exceed daughter mass sum"
            )
        return cls(
            lineshape=lineshape,
            context=context,
            low=low,
            high=high,
            quadrature_order=quadrature_order,
            floor=floor,
        )

    def _intensity(self, x: Array, parameters: Parameters | None = None) -> Array:
        context = self.context.resolve(parameters)
        lineshape = resolve_value(self.lineshape, parameters)
        amplitude = jnp.asarray(lineshape(jnp.asarray(x), context))
        return jnp.real(amplitude * jnp.conj(amplitude))

    def normalization(self, parameters: Parameters | None = None) -> Array:
        """Integral of ``|lineshape|^2`` over ``[low, high]`` by quadrature."""

        values = self._intensity(self._nodes, parameters)
        return jnp.sum(self._weights * values)

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        norm = self.normalization(parameters)
        intensity = self._intensity(x, parameters)
        inside = (x >= self.low) & (x <= self.high) & (norm > 0.0)
        return jnp.where(
            inside,
            jnp.clip(intensity / norm, min=self.floor),
            0.0,
        )


@dataclass(frozen=True)
class Exponential1D:
    """Exponential PDF ``exp(slope*x)`` normalized on a finite interval."""

    slope: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("Exponential1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        slope = jnp.asarray(_resolve(self.slope, parameters))
        x = jnp.asarray(x)
        span = self.high - self.low
        near_zero = jnp.abs(slope) < 1e-10
        norm = jnp.where(
            near_zero,
            span,
            (jnp.exp(slope * self.high) - jnp.exp(slope * self.low)) / slope,
        )
        raw = jnp.exp(slope * x)
        inside = (x >= self.low) & (x <= self.high) & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class Chebyshev1D:
    """Chebyshev-polynomial PDF normalized on a finite interval.

    ``f(x) ∝ 1 + sum_k c_k T_k(t)`` with ``t = (2x - low - high)/(high - low)``
    in ``[-1, 1]`` and ``k = 1..len(coefficients)`` (the ``RooChebychev``
    convention). The normalization is closed-form, ``int T_k dt = 0`` for odd
    ``k`` and ``2/(1-k^2)`` for even ``k``. The polynomial is not positive by
    construction: coefficients that make it negative anywhere on the support
    return NaN everywhere, rejecting that parameter point even in a mixture. Clipping
    a negative polynomial would invalidate its analytic normalization.
    Coefficients may be numbers or fit ``Parameter`` objects.
    """

    coefficients: tuple
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if self.high <= self.low:
            raise ValueError("Chebyshev1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        t = (2.0 * x - self.low - self.high) / (self.high - self.low)
        coefficients = jnp.asarray(
            [_resolve(coefficient, parameters) for coefficient in self.coefficients],
            dtype=float,
        )
        value = _chebyshev_value(coefficients, t)
        integral = 2.0  # integral of T_0 = 1 over t in [-1, 1]
        for k, c in enumerate(coefficients, start=1):
            if k % 2 == 0:
                integral = integral + c * 2.0 / (1.0 - k**2)
        norm = 0.5 * (self.high - self.low) * integral
        valid = jnp.isfinite(norm) & (norm > 0) & _chebyshev_nonnegative(coefficients)
        safe_norm = jnp.where(valid, norm, 1.0)
        inside = (x >= self.low) & (x <= self.high) & valid
        density = jnp.where(inside, jnp.clip(value / safe_norm, min=self.floor), 0.0)
        return jnp.where(valid, density, jnp.nan)


@dataclass(frozen=True)
class Histogram1D:
    """Piecewise-constant normalized histogram PDF."""

    edges: Array
    values: Array

    def __post_init__(self) -> None:
        edges = jnp.asarray(self.edges)
        values = jnp.asarray(self.values)
        if edges.ndim != 1 or values.ndim != 1 or edges.size != values.size + 1:
            raise ValueError("Histogram1D requires len(edges)=len(values)+1")
        if bool(jnp.any(jnp.diff(edges) <= 0.0)) or bool(jnp.any(values < 0.0)):
            raise ValueError(
                "Histogram1D requires increasing edges and non-negative values"
            )
        widths = jnp.diff(edges)
        norm = jnp.sum(widths * values)
        if not bool(jnp.isfinite(norm)) or bool(norm <= 0.0):
            raise ValueError("Histogram1D integral must be positive and finite")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "values", values / norm)

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        idx = jnp.searchsorted(self.edges, x, side="right") - 1
        valid = (idx >= 0) & (idx < self.values.size)
        idx = jnp.clip(idx, 0, self.values.size - 1)
        return jnp.where(valid, self.values[idx], 0.0)


@dataclass(frozen=True)
class FactorizedDensity:
    """Multiply a base event density by independent discriminant PDFs."""

    base_density: object
    observables: Mapping[str, Array]
    pdfs: Mapping[str, object]

    def __post_init__(self) -> None:
        if set(self.observables) != set(self.pdfs):
            raise ValueError(
                "FactorizedDensity observables and pdfs must have identical keys"
            )

    def __call__(self, parameters: Parameters) -> Array:
        base = jnp.asarray(self.base_density(parameters))
        result = base
        for name, values in self.observables.items():
            factor = jnp.asarray(self.pdfs[name](values, parameters))
            if factor.shape != base.shape:
                raise ValueError(f"discriminant {name!r} shape must match base density")
            result = result * factor
        return result


__all__ = [
    "BreitWigner1D",
    "Chebyshev1D",
    "CrystalBall1D",
    "SumPDF1D",
    "Exponential1D",
    "FactorizedDensity",
    "Gaussian1D",
    "Histogram1D",
    "LineshapeIntensity1D",
]
