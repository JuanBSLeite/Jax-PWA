"""Joint charge-Dalitz likelihoods for direct-CP amplitude fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from jaxpwa.amplitude import PreparedAmplitudeCache
from jaxpwa.background import CPBackgroundCategory

Parameters = Mapping[str, Array | float]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def _signal_yield_pair(value: object, parameters: Parameters) -> tuple[Array, Array, bool]:
    """Resolve an extended-fit signal yield into ``(n_plus, n_minus, standalone)``.

    ``value`` is either a plain number/``Parameter`` shared by both charges
    (the historical behaviour: ``standalone=False``, the same value multiplies
    the *jointly*-normalized ``signal_plus``/``signal_minus`` shapes, so the
    charge split comes entirely from the amplitude's own interference), or a
    ``YieldAsymmetry`` (``standalone=True``): ``n_plus``/``n_minus`` are then
    literal expected per-charge counts that must multiply *standalone*
    (per-charge) normalized shapes instead, since a jointly-normalized shape
    would silently reintroduce the amplitude's own charge split on top of the
    independently-fitted one.
    """
    if isinstance(value, YieldAsymmetry):
        return jnp.asarray(value.plus(parameters)), jnp.asarray(value.minus(parameters)), True
    shared = jnp.asarray(_resolve(value, parameters))
    return shared, shared, False


@dataclass(frozen=True)
class YieldAsymmetry:
    """Total extended-fit signal yield split into a charge asymmetry.

    ``total`` (``N_s``) and ``asymmetry`` (the yield asymmetry, distinct from
    the coefficient-level ``A_CP`` of ``CPRealImag``) parameterize

    ``N_plus  = N_s (1 - asymmetry) / 2``
    ``N_minus = N_s (1 + asymmetry) / 2``,

    so ``N_plus + N_minus == N_s`` for any value of ``asymmetry``. ``N_plus``
    and ``N_minus`` are literal expected event counts for each charge:
    passing a ``YieldAsymmetry`` to ``CPJointNLL.signal_yield`` makes each
    charge's signal density its own independently-normalized isobar PDF,
    ``p_q(phi) = |A_q(phi)|^2 / I_q``, scaled by that charge's own yield. The
    two amplitudes ``A_plus``/``A_minus`` are still coherent sums of all
    interfering components, so the *shape* of direct CP violation within
    each charge's own Dalitz plot is unaffected; what changes is that the
    *relative rate* between charges is now set directly by ``asymmetry``
    instead of by the amplitude's own ``I_plus``/``I_minus`` (see
    ``docs/cp_coefficients.md``'s "central invariant" discussion of
    ``CPJointNLL``'s default joint normalization, which this deliberately
    departs from). Passing a plain number/``Parameter`` as
    ``CPJointNLL.signal_yield`` instead of a ``YieldAsymmetry`` keeps the
    historical shared-yield behaviour exactly, where the charge split comes
    entirely from ``I_plus``/``I_minus``.

    ``total`` and ``asymmetry`` may be ordinary numbers or fit ``Parameter``
    objects.
    """

    total: object
    asymmetry: object = 0.0

    @property
    def parameters(self) -> tuple[object, ...]:
        """The fit Parameter objects among total/asymmetry, skipping plain numbers."""
        return tuple(value for value in (self.total, self.asymmetry) if hasattr(value, "resolve"))

    def plus(self, parameters: Parameters) -> Array:
        """Expected B+ signal yield, ``N_plus = N_s (1 - asymmetry) / 2``."""
        return 0.5 * jnp.asarray(_resolve(self.total, parameters)) * (1.0 - jnp.asarray(_resolve(self.asymmetry, parameters)))

    def minus(self, parameters: Parameters) -> Array:
        """Expected B- signal yield, ``N_minus = N_s (1 + asymmetry) / 2``."""
        return 0.5 * jnp.asarray(_resolve(self.total, parameters)) * (1.0 + jnp.asarray(_resolve(self.asymmetry, parameters)))


@dataclass(frozen=True)
class CPJointNLL:
    """Unbinned likelihood over the joint ``(Dalitz, charge)`` sample space.

    The signal density is always normalized jointly over both charges,

    ``S_q(phi) = eps_q(phi) |A_q(phi)|^2 / (I_+ + I_-)``.

    Arbitrary named background categories can be supplied through
    ``background_categories``.  Each category is also normalized jointly over
    charge.  The legacy single-background arguments remain supported.

    In extended mode, ``signal_yield`` may be a plain number/``Parameter``
    (shared ``N_sig`` for both charges, the historical behaviour) or a
    ``YieldAsymmetry`` instance parameterizing independent ``N_plus``/
    ``N_minus`` through a total yield and a yield asymmetry.

    ``plus_weights``/``minus_weights`` (supplied together) turn the objective
    into the signal-only weighted estimating function

    ``-sum_i w_i log S_+(phi_i) - sum_j w_j log S_-(phi_j)``,

    e.g. for sWeight/COW background subtraction. The signal density keeps
    the joint charge normalization above, so the charge asymmetry is still
    measured through ``I_+``/``I_-``: the weighted B+/B- sums play the role
    of the per-charge event counts. Weighted mode is signal-only (no
    background, ``signal_fraction`` or extended terms). Events with zero
    weight do not contribute, even where the signal density vanishes.
    """

    plus_cache: PreparedAmplitudeCache
    minus_cache: PreparedAmplitudeCache
    plus_efficiency: Array | None = None
    minus_efficiency: Array | None = None
    plus_background: Array | None = None
    minus_background: Array | None = None
    plus_background_normalization: Array | float | None = None
    minus_background_normalization: Array | float | None = None
    background_categories: tuple[CPBackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    background_yield: object | None = None
    plus_weights: Array | None = None
    minus_weights: Array | None = None

    def __post_init__(self) -> None:
        if (self.plus_efficiency is None) != (self.minus_efficiency is None):
            raise ValueError("plus_efficiency and minus_efficiency must be supplied together")
        if (self.plus_weights is None) != (self.minus_weights is None):
            raise ValueError("plus_weights and minus_weights must be supplied together")
        legacy_fields = (
            self.plus_background,
            self.minus_background,
            self.plus_background_normalization,
            self.minus_background_normalization,
        )
        legacy_supplied = tuple(value is not None for value in legacy_fields)
        if any(legacy_supplied) and not all(legacy_supplied):
            raise ValueError("background requires plus/minus background values and plus/minus normalizations")
        if any(legacy_supplied) and self.background_categories:
            raise ValueError("use either the legacy single-background arguments or background_categories, not both")
        names = [category.name for category in self.background_categories]
        if len(set(names)) != len(names):
            raise ValueError("CP background category names must be unique")
        if self.extended:
            if self.signal_fraction is not None:
                raise ValueError("signal_fraction is not used in extended mode")
            if self.signal_yield is None:
                raise ValueError("extended mode requires signal_yield")
            if self.background_categories:
                if self.background_yield is not None:
                    raise ValueError("background_yield belongs to the legacy single-background interface")
                if any(category.fraction is not None for category in self.background_categories):
                    raise ValueError("background fractions are not used in extended mode")
                if any(category.yield_ is None for category in self.background_categories):
                    raise ValueError("every extended CP background category requires yield_")
            elif self.has_legacy_background:
                if self.background_yield is None:
                    raise ValueError("extended single-background fits require background_yield")
            elif self.background_yield is not None:
                raise ValueError("background_yield requires a background model")
        else:
            if self.signal_yield is not None or self.background_yield is not None:
                raise ValueError("signal_yield/background_yield require extended=True")
            if self.has_background and self.signal_fraction is None:
                raise ValueError("non-extended background fits require signal_fraction")
            if not self.has_background and self.signal_fraction is not None:
                raise ValueError("signal_fraction requires a background model")
            if self.background_categories:
                if any(category.yield_ is not None for category in self.background_categories):
                    raise ValueError("background yields require extended=True")
                if len(self.background_categories) > 1:
                    if any(c.fraction is None for c in self.background_categories[:-1]):
                        raise ValueError("all CP background categories except the last require a relative fraction")
                    if self.background_categories[-1].fraction is not None:
                        raise ValueError("the last CP background category is the remainder and must not define fraction")
                elif len(self.background_categories) == 1 and self.background_categories[0].fraction is not None:
                    raise ValueError("a single CP background category does not need a relative fraction")

        for charge, cache in (("plus", self.plus_cache), ("minus", self.minus_cache)):
            size = cache.data_components.shape[0]
            for label in ("efficiency", "background"):
                name = f"{charge}_{label}"
                value = getattr(self, name)
                if value is not None:
                    array = jnp.asarray(value)
                    if label == "efficiency" and array.ndim == 0:
                        array = jnp.full((size,), array)
                    if array.shape != (size,):
                        raise ValueError(f"{name} must have shape ({size},)")
                    if bool(jnp.any(~jnp.isfinite(array) | (array < 0))):
                        raise ValueError(f"{name} must be finite and non-negative")
                    object.__setattr__(self, name, array)
            for category in self.background_categories:
                if getattr(category, f"{charge}_values").shape != (size,):
                    raise ValueError(f"{category.name} {charge} background size mismatch")
            if self.weighted:
                weights = jnp.asarray(getattr(self, f"{charge}_weights"))
                if weights.shape != (size,):
                    raise ValueError(f"{charge}_weights must have shape ({size},), got {weights.shape}")
                if not bool(jnp.all(jnp.isfinite(weights))):
                    raise ValueError(f"{charge}_weights must be finite")
                object.__setattr__(self, f"{charge}_weights", weights)
        if self.weighted and (self.has_background or self.extended or self.signal_fraction is not None):
            raise ValueError(
                "event weights are incompatible with background, signal_fraction or "
                "extended terms; a weighted CPJointNLL is signal-only"
            )
        if self.has_legacy_background:
            for value in (self.plus_background_normalization, self.minus_background_normalization):
                array = jnp.asarray(value)
                if array.ndim != 0 or not bool(jnp.isfinite(array) & (array > 0)):
                    raise ValueError("Background normalizations must be positive finite scalars")
        if not bool(self._physical_parameters({})):
            raise ValueError("Initial CP yields/fractions must be finite and physical")

    def _physical_parameters(self, parameters):
        def valid_scalar(value, upper=None):
            resolved = jnp.asarray(_resolve(value, parameters))
            if resolved.ndim != 0:
                raise ValueError("CP yields and fractions must be scalars")
            valid = jnp.isfinite(resolved) & (resolved >= 0)
            return valid if upper is None else valid & (resolved <= upper)

        def valid_yield_pair(value):
            n_plus, n_minus, _ = _signal_yield_pair(value, parameters)
            if n_plus.ndim != 0 or n_minus.ndim != 0:
                raise ValueError("CP yields and fractions must be scalars")
            return jnp.isfinite(n_plus) & (n_plus >= 0) & jnp.isfinite(n_minus) & (n_minus >= 0)

        valid = jnp.asarray(True)
        if self.extended:
            valid = valid & valid_yield_pair(self.signal_yield)
            if self.has_legacy_background:
                valid = valid & valid_scalar(self.background_yield)
            for category in self.background_categories:
                valid = valid & valid_scalar(category.yield_)
        elif self.has_background:
            valid = valid & valid_scalar(self.signal_fraction, 1)
            for category in self.background_categories[:-1]:
                valid = valid & valid_scalar(category.fraction, 1)
            weights = self.background_weights(parameters)
            valid = valid & jnp.all(jnp.isfinite(weights) & (weights >= 0))
        return valid

    @property
    def weighted(self) -> bool:
        """Whether per-event plus_weights/minus_weights are set."""
        return self.plus_weights is not None

    @property
    def has_legacy_background(self) -> bool:
        """Whether the legacy plus_background/minus_background arguments are set."""
        return self.plus_background is not None

    @property
    def has_background(self) -> bool:
        """Whether any background, legacy or category-based, is configured."""
        return self.has_legacy_background or bool(self.background_categories)

    def _signal_densities(self, parameters: Parameters) -> tuple[Array, Array, Array, Array]:
        intensity_plus, integral_plus = self.plus_cache.evaluate(parameters)
        intensity_minus, integral_minus = self.minus_cache.evaluate(parameters)
        if self.plus_efficiency is not None:
            intensity_plus = jnp.asarray(self.plus_efficiency) * intensity_plus
            intensity_minus = jnp.asarray(self.minus_efficiency) * intensity_minus
        total_integral = integral_plus + integral_minus
        valid = (jnp.isfinite(integral_plus) & jnp.isfinite(integral_minus)
                 & (integral_plus >= 0) & (integral_minus >= 0) & (total_integral > 0))
        total_integral = jnp.where(valid, total_integral, jnp.nan)
        return intensity_plus / total_integral, intensity_minus / total_integral, integral_plus, integral_minus

    def _legacy_background_densities(self) -> tuple[Array, Array]:
        if not self.has_legacy_background:
            raise RuntimeError("legacy background densities requested without legacy background")
        total = jnp.asarray(self.plus_background_normalization) + jnp.asarray(self.minus_background_normalization)
        return jnp.asarray(self.plus_background) / total, jnp.asarray(self.minus_background) / total

    def _background_densities(self) -> tuple[Array, Array]:
        """Backward-compatible single-background density helper."""
        return self._legacy_background_densities()

    def signal_log_densities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Per-event ``(log S_+, log S_-)`` of the jointly normalized signal.

        ``-inf`` where the density is zero or the joint normalization is
        invalid; gradients stay finite (safe logarithm arguments).
        """
        signal_plus, signal_minus, _, _ = self._signal_densities(parameters)

        def safe_log(values):
            valid = jnp.isfinite(values) & (values > 0)
            return jnp.where(valid, jnp.log(jnp.where(valid, values, 1.0)), -jnp.inf)

        return safe_log(signal_plus), safe_log(signal_minus)

    def background_weights(self, parameters: Parameters) -> Array:
        """Resolve each non-extended background category's relative fraction.

        The last category's weight is the remainder, ``1 - sum(others)``.
        """
        n = len(self.background_categories)
        if n == 0:
            return jnp.empty((0,), dtype=jnp.float64)
        if n == 1:
            return jnp.ones((1,), dtype=jnp.float64)
        explicit = jnp.asarray([_resolve(category.fraction, parameters) for category in self.background_categories[:-1]], dtype=jnp.float64)
        remainder = 1.0 - jnp.sum(explicit)
        return jnp.concatenate((explicit, jnp.asarray([remainder])))

    def component_densities(self, parameters: Parameters):
        """Return unweighted component densities for diagnostics/legacy tests."""
        signal_plus, signal_minus, _, _ = self._signal_densities(parameters)
        if self.background_categories:
            weights = self.background_weights(parameters) if not self.extended else None
            categories = []
            for index, category in enumerate(self.background_categories):
                if self.extended:
                    categories.append((category.plus_density, category.minus_density))
                else:
                    categories.append((weights[index] * category.plus_density, weights[index] * category.minus_density))
            return (signal_plus, signal_minus), tuple(categories)
        if self.has_legacy_background:
            return (signal_plus, signal_minus), self._legacy_background_densities()
        return (signal_plus, signal_minus), None

    def densities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return the (plus, minus) per-event densities the NLL takes the log of.

        In extended mode these are total expected densities (signal yield(s)
        plus any background yield(s)); otherwise they are the
        signal_fraction-weighted signal/background mixture.
        """
        signal_plus, signal_minus, integral_plus, integral_minus = self._signal_densities(parameters)
        if self.extended:
            n_plus, n_minus, standalone = _signal_yield_pair(self.signal_yield, parameters)
            if standalone:
                total_integral = integral_plus + integral_minus
                total_plus = n_plus * signal_plus * total_integral / integral_plus
                total_minus = n_minus * signal_minus * total_integral / integral_minus
            else:
                total_plus = n_plus * signal_plus
                total_minus = n_minus * signal_minus
            if self.background_categories:
                for category in self.background_categories:
                    n_background = jnp.asarray(_resolve(category.yield_, parameters))
                    total_plus = total_plus + n_background * category.plus_density
                    total_minus = total_minus + n_background * category.minus_density
            elif self.has_legacy_background:
                n_background = jnp.asarray(_resolve(self.background_yield, parameters))
                background_plus, background_minus = self._legacy_background_densities()
                total_plus = total_plus + n_background * background_plus
                total_minus = total_minus + n_background * background_minus
            return total_plus, total_minus
        if not self.has_background:
            return signal_plus, signal_minus
        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        if self.background_categories:
            weights = self.background_weights(parameters)
            background_plus = jnp.zeros_like(signal_plus)
            background_minus = jnp.zeros_like(signal_minus)
            for weight, category in zip(weights, self.background_categories):
                background_plus = background_plus + weight * category.plus_density
                background_minus = background_minus + weight * category.minus_density
        else:
            background_plus, background_minus = self._legacy_background_densities()
        return f_signal * signal_plus + (1.0 - f_signal) * background_plus, f_signal * signal_minus + (1.0 - f_signal) * background_minus

    def expected_events(self, parameters: Parameters) -> Array:
        """Total expected events (signal + background) for the extended Poisson term."""
        if not self.extended:
            raise RuntimeError("expected_events is only defined in extended mode")
        if isinstance(self.signal_yield, YieldAsymmetry):
            total = jnp.asarray(_resolve(self.signal_yield.total, parameters))
        else:
            total = jnp.asarray(_resolve(self.signal_yield, parameters))
        if self.background_categories:
            for category in self.background_categories:
                total = total + jnp.asarray(_resolve(category.yield_, parameters))
        elif self.has_legacy_background:
            total = total + jnp.asarray(_resolve(self.background_yield, parameters))
        return total

    def _weighted_nll(self, parameters: Parameters) -> Array:
        total = jnp.asarray(0.0, dtype=self.plus_cache.data_components.real.dtype)
        valid = jnp.asarray(True)
        log_densities = self.signal_log_densities(parameters)
        for weights, log_density in zip((self.plus_weights, self.minus_weights), log_densities, strict=True):
            active = weights != 0
            finite = jnp.isfinite(log_density)
            valid = valid & jnp.all(finite | ~active)
            total = total - jnp.sum(jnp.where(active & finite, weights * jnp.where(finite, log_density, 0.0), 0.0))
        return jnp.where(valid & jnp.isfinite(total), total, jnp.inf)

    def __call__(self, parameters: Parameters) -> Array:
        """Return +inf outside the physical domain, including during JIT fits."""
        if self.weighted:
            return self._weighted_nll(parameters)

        def evaluate(_):
            plus, minus = self.densities(parameters)
            valid = (jnp.all(jnp.isfinite(plus) & (plus > 0))
                     & jnp.all(jnp.isfinite(minus) & (minus > 0)))
            # Safe arguments keep invalid log branches out of the gradient.
            nll = -jnp.sum(jnp.log(jnp.where(plus > 0, plus, 1.0)))
            nll -= jnp.sum(jnp.log(jnp.where(minus > 0, minus, 1.0)))
            if self.extended:
                nll += self.expected_events(parameters)
            return jnp.where(valid & jnp.isfinite(nll), nll, jnp.inf)

        return jax.lax.cond(self._physical_parameters(parameters), evaluate,
                            lambda _: jnp.asarray(jnp.inf, dtype=self.plus_cache.data_components.real.dtype),
                            operand=None)

    def charge_probabilities(self, parameters: Parameters) -> tuple[Array, Array]:
        """The model's predicted (B+, B-) event-count fractions, signal+background."""
        integral_plus = self.plus_cache.normalization(parameters)
        integral_minus = self.minus_cache.normalization(parameters)
        signal_total = integral_plus + integral_minus
        signal_plus = integral_plus / signal_total
        signal_minus = integral_minus / signal_total
        if self.extended and isinstance(self.signal_yield, YieldAsymmetry):
            # A YieldAsymmetry's n_plus/n_minus already are the expected
            # per-charge signal counts; unlike the shared-yield branch below,
            # they must not be reweighted by the amplitude-driven
            # signal_plus/signal_minus fractions (see YieldAsymmetry).
            numerator_plus = jnp.asarray(self.signal_yield.plus(parameters))
            numerator_minus = jnp.asarray(self.signal_yield.minus(parameters))
            total_yield = numerator_plus + numerator_minus
            if self.background_categories:
                for category in self.background_categories:
                    n_background = jnp.asarray(_resolve(category.yield_, parameters))
                    numerator_plus = numerator_plus + n_background * category.plus_probability
                    numerator_minus = numerator_minus + n_background * category.minus_probability
                    total_yield = total_yield + n_background
            elif self.has_legacy_background:
                n_background = jnp.asarray(_resolve(self.background_yield, parameters))
                bplus = jnp.asarray(self.plus_background_normalization)
                bminus = jnp.asarray(self.minus_background_normalization)
                btotal = bplus + bminus
                numerator_plus = numerator_plus + n_background * bplus / btotal
                numerator_minus = numerator_minus + n_background * bminus / btotal
                total_yield = total_yield + n_background
            return numerator_plus / total_yield, numerator_minus / total_yield
        if not self.has_background:
            return signal_plus, signal_minus
        if self.extended:
            n_signal = jnp.asarray(_resolve(self.signal_yield, parameters))
            numerator_plus = n_signal * signal_plus
            numerator_minus = n_signal * signal_minus
            total_yield = numerator_plus + numerator_minus
            if self.background_categories:
                for category in self.background_categories:
                    n_background = jnp.asarray(_resolve(category.yield_, parameters))
                    numerator_plus = numerator_plus + n_background * category.plus_probability
                    numerator_minus = numerator_minus + n_background * category.minus_probability
                    total_yield = total_yield + n_background
            else:
                n_background = jnp.asarray(_resolve(self.background_yield, parameters))
                bplus = jnp.asarray(self.plus_background_normalization)
                bminus = jnp.asarray(self.minus_background_normalization)
                btotal = bplus + bminus
                numerator_plus = numerator_plus + n_background * bplus / btotal
                numerator_minus = numerator_minus + n_background * bminus / btotal
                total_yield = total_yield + n_background
            return numerator_plus / total_yield, numerator_minus / total_yield
        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        if self.background_categories:
            weights = self.background_weights(parameters)
            background_plus = jnp.asarray(0.0)
            background_minus = jnp.asarray(0.0)
            for weight, category in zip(weights, self.background_categories):
                background_plus = background_plus + weight * category.plus_probability
                background_minus = background_minus + weight * category.minus_probability
        else:
            bplus = jnp.asarray(self.plus_background_normalization)
            bminus = jnp.asarray(self.minus_background_normalization)
            btotal = bplus + bminus
            background_plus = bplus / btotal
            background_minus = bminus / btotal
        return f_signal * signal_plus + (1.0 - f_signal) * background_plus, f_signal * signal_minus + (1.0 - f_signal) * background_minus


__all__ = ["CPJointNLL", "YieldAsymmetry"]
