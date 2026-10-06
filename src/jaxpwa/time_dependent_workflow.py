"""User-facing high-level workflow for time-dependent tagged Dalitz fits.

Composes ``TimeDependentDalitzNLL``/``NeutralMesonMixing`` the same way
``FitSession``/``CPFitSession`` compose their lower-level likelihoods (see
``docs/user_friendly_api.md``, "Design principle"): this module adds no new
physics, only less boilerplate around building the shared A/Abar
``PreparedAmplitudeCache`` and collecting fit parameters. See
``docs/time_dependent.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property

import jax.numpy as jnp
import numpy as np

from jaxpwa.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from jaxpwa.background import BackgroundCategory
from jaxpwa.constraints import ConstrainedNLL
from jaxpwa.decay import DecayModel
from jaxpwa.fit import Minimizer, Parameter, ParameterKind
from jaxpwa.io import model_with_fitted_values
from jaxpwa.kinematics import PhaseSpaceSample
from jaxpwa.likelihood.mixture import _resolve
from jaxpwa.likelihood.time_dependent import (
    NeutralMesonMixing,
    TimeDependentDalitzNLL,
)
from jaxpwa.likelihood.time_dependent_mixture import (
    TimeDependentBackgroundCategory,
    TimeDependentMixtureNLL,
)


@dataclass(frozen=True)
class TimeDependentBackgroundSpec:
    """Factorized observed background: Dalitz shape times normalized time PDF.

    ``shape(data)`` is fixed and normalized automatically over the model's
    physical phase space, separately for each tag; it may inspect ``tag``.
    ``time_pdf(data, parameters)`` must integrate to one over the session's
    selected observed-time range for each tag/sigma_t. Data contains ``t``,
    ``tag`` and ``sigma_t`` if supplied. This PDF includes its own acceptance
    and resolution; the signal response/efficiency is never applied to it.
    Declare any time-shape parameters in ``parameters`` (or on the callable).
    Use TimeDependentBackgroundCategory for a correlated Dalitz-time PDF.
    """

    name: str
    shape: object
    time_pdf: object
    fraction: object = None
    yield_: object = None
    normalization_sample: PhaseSpaceSample | None = None
    apply_veto: bool = True
    tag_fraction: object = None
    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self):
        if not self.name:
            raise ValueError("background name must be non-empty")
        if not callable(self.shape) or not callable(self.time_pdf):
            raise TypeError("background shape and time_pdf must be callable")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background cannot define both fraction and yield")


def _collect_parameters(value: object) -> tuple[Parameter, ...]:
    if isinstance(value, Parameter):
        return (value,)
    if value is None:
        return ()
    if is_dataclass(value) and not isinstance(value, type):
        found: list[Parameter] = []
        for field in fields(value):
            found.extend(_collect_parameters(getattr(value, field.name)))
        return tuple(found)
    parameters = getattr(value, "parameters", None)
    if parameters is not None and not callable(parameters):
        return _collect_parameters(parameters)
    if isinstance(value, Mapping):
        return tuple(
            parameter
            for item in value.values()
            for parameter in _collect_parameters(item)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            parameter for item in value for parameter in _collect_parameters(item)
        )
    return ()


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    """Evaluate the parameter-independent event acceptance once."""

    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    values = jnp.ones((size,), dtype=jnp.float64)
    for label, function in (("efficiency", efficiency), ("veto", veto)):
        if function is not None:
            array = jnp.asarray(function(data))
            if array.ndim == 0:
                array = jnp.full((size,), array)
            if array.shape != (size,):
                raise ValueError(f"{label} must have shape ({size},)")
            if bool(jnp.any(~jnp.isfinite(array) | (array < 0))):
                raise ValueError(f"{label} must be finite and non-negative")
            values = values * array
    return values


@dataclass(frozen=True)
class _ObservedTagFraction:
    """Observed positive-tag fraction from production and mistag inputs."""

    production_fraction: object
    wrong_tag: object

    @property
    def parameters(self):
        return _collect_parameters((self.production_fraction, self.wrong_tag))

    def resolve(self, values=None):
        production = _resolve(self.production_fraction, values)
        wrong = _resolve(self.wrong_tag, values)
        return production * (1 - wrong) + (1 - production) * wrong


@dataclass(frozen=True)
class _ReflectedAmplitude:
    """No-direct-CPV default: Abar(s12,s13) = A(s13,s12), same s23.

    Passes only raw reflected invariants to the wrapped component's dynamics
    function -- any *prepared* kinematic arrays from the unreflected point
    would silently evaluate the wrong amplitude (see docs/time_dependent.md,
    "Preparing A and Abar").
    """

    function: object

    def __call__(self, data, parameters=None):
        reflected = {"s12": data["s13"], "s13": data["s12"], "s23": data["s23"]}
        return self.function(reflected, parameters)


@dataclass(frozen=True)
class TimeDependentFitSession:
    """Tagged Dalitz-time session with optional multiple backgrounds and yields.

    A composition layer over ``TimeDependentDalitzNLL``: it builds the single
    shared A+Abar ``PreparedAmplitudeCache`` (same final-state coordinates,
    same integration sample, per ``docs/time_dependent.md``) and collects fit
    parameters from ``model`` (and ``abar_model``, if given) plus ``mixing``.
    It does not replace the low-level classes; use them directly for
    workflows this session does not cover.

    ``abar_model`` is optional. When ``None`` (the default, no-direct-CPV
    convention used by the Belle-style reproduction in
    ``notebooks/benchmark/belle_2014_d0_kspipi_time_dependent.ipynb``), the
    D0bar amplitude is derived by reflecting ``model``'s own components:
    ``Abar(s12,s13) = A(s13,s12)``. Pass an independently built
    ``abar_model`` for direct CP violation; reference phases/scales must be
    fixed by the caller to remove unidentifiable directions, exactly as
    documented for the low-level classes. Only ``abar_model``'s dynamics and
    coefficients are used -- its own ``channel``/``normalization_sample`` are
    ignored, since the cache is built once on ``model``'s ``data``/
    ``normalization_sample`` for both flavours (``__post_init__`` checks that
    the two channels at least share daughter masses).

    ``signal_objective`` retains the signal-only time-dependent likelihood.
    With backgrounds or extended yields, ``base_objective`` composes it with
    ``TimeDependentMixtureNLL``. Fractions describe the same conditional
    composition for both tags; extended yields count both tags, with explicit
    component tag fractions (defaulting to the observed tag proportions).
    ``plot_time_projection`` overlays each tag's decay-time histogram against
    the exact Dalitz-integrated curve (unit acceptance/perfect resolution
    only); ``plot_projection`` is the Dalitz-variable analogue, time-integrated
    over ``time_range``. See ``docs/time_dependent.md``.
    """

    model: DecayModel
    data: PhaseSpaceSample
    times: object
    tags: object
    mixing: NeutralMesonMixing
    abar_model: DecayModel | None = None
    efficiency: object | None = None
    veto: object | None = None
    wrong_tag: object = 0.0
    time_range: tuple = (0.0, np.inf)
    time_nodes: object = None
    time_weights: object = None
    time_acceptance: object = None
    sigma_t: object = None
    constraints: tuple[object, ...] = ()
    backgrounds: tuple[
        TimeDependentBackgroundSpec
        | TimeDependentBackgroundCategory
        | BackgroundCategory,
        ...,
    ] = ()
    signal_fraction: object = None
    extended: bool = False
    signal_yield: object = None
    signal_tag_fraction: object = None
    production_fraction: object = None

    def __post_init__(self) -> None:
        if self.production_fraction is not None and not self.extended:
            raise ValueError("production_fraction requires extended=True")
        if (
            self.production_fraction is not None
            and self.signal_tag_fraction is not None
        ):
            raise ValueError(
                "provide either production_fraction or signal_tag_fraction, not both"
            )
        if self.abar_model is not None:
            own = tuple(float(m) for m in self.model.channel.daughter_masses)
            other = tuple(float(m) for m in self.abar_model.channel.daughter_masses)
            if own != other:
                raise ValueError(
                    "abar_model must share model's daughter masses: the cache "
                    "is built once on model's data/normalization_sample for "
                    "both flavours, so an unrelated channel would silently "
                    "evaluate abar_model's dynamics at the wrong kinematics"
                )

    def with_background(
        self,
        name,
        shape,
        *,
        time_pdf,
        fraction=None,
        yield_=None,
        normalization_sample=None,
        apply_veto=True,
        tag_fraction=None,
        parameters=(),
    ):
        """Append a factorized background, following FitSession's mixture API.

        ``time_pdf(data, parameters)`` must be normalized over ``time_range``.
        Its observed-time resolution and acceptance belong to the background.
        """
        spec = TimeDependentBackgroundSpec(
            name,
            shape,
            time_pdf,
            fraction,
            yield_,
            normalization_sample,
            apply_veto,
            tag_fraction,
            tuple(parameters),
        )
        return replace(self, backgrounds=self.backgrounds + (spec,))

    @cached_property
    def _default_tag_fraction(self):
        return float(np.mean(np.asarray(self.tags) == 1))

    @property
    def _signal_tag_fraction(self):
        if self.production_fraction is not None:
            return _ObservedTagFraction(self.production_fraction, self.wrong_tag)
        return (
            self._default_tag_fraction
            if self.signal_tag_fraction is None
            else self.signal_tag_fraction
        )

    def _event_data(self):
        data = dict(
            self.data.as_dict(), t=jnp.asarray(self.times), tag=jnp.asarray(self.tags)
        )
        if self.sigma_t is not None:
            data["sigma_t"] = jnp.broadcast_to(
                jnp.asarray(self.sigma_t), jnp.shape(self.times)
            )
        return data

    def _build_background(self, source):
        if isinstance(source, BackgroundCategory):
            source = TimeDependentBackgroundCategory(
                source.name,
                source.density,
                source.fraction,
                source.yield_,
            )
        if isinstance(source, TimeDependentBackgroundCategory):
            return replace(
                source,
                tag_fraction=(
                    self._default_tag_fraction
                    if source.tag_fraction is None
                    else source.tag_fraction
                ),
            )
        if not isinstance(source, TimeDependentBackgroundSpec):
            raise TypeError(
                "backgrounds require TimeDependentBackgroundSpec or a category"
            )
        if any(not p.fixed for p in _collect_parameters(source.shape)):
            raise ValueError(
                "factorized Dalitz background shape parameters must be fixed; "
                "use TimeDependentBackgroundCategory for a floating joint PDF"
            )
        sample = (
            self.model.normalization_sample
            if source.normalization_sample is None
            else source.normalization_sample
        )
        sample.validate_integration()

        def shape(data):
            size = len(data["s12"])
            array = jnp.broadcast_to(jnp.asarray(source.shape(data)), (size,))
            if source.apply_veto and self.veto is not None:
                array = array * jnp.asarray(self.veto(data))
            return array

        integrals = []
        for tag in (1, -1):
            data = dict(sample.as_dict(), tag=jnp.full(sample.size, tag))
            array = shape(data)
            if bool(jnp.any(~jnp.isfinite(array) | (array < 0))):
                raise ValueError("background shape must be finite and nonnegative")
            norm = jnp.mean(sample.weights * array)
            if not bool(jnp.isfinite(norm) & (norm > 0)):
                raise ValueError("background normalization must be finite and positive")
            integrals.append(norm)

        def dalitz_density(data, parameters):
            norm = jnp.where(jnp.asarray(data["tag"]) == 1, integrals[0], integrals[1])
            return shape(data) / norm

        data = self._event_data()
        dalitz = dalitz_density(data, {})
        if bool(jnp.any(~jnp.isfinite(dalitz) | (dalitz < 0))):
            raise ValueError("background data values must be finite and nonnegative")

        def density(parameters):
            return dalitz * jnp.asarray(source.time_pdf(data, parameters))

        return TimeDependentBackgroundCategory(
            source.name,
            density,
            source.fraction,
            source.yield_,
            self._default_tag_fraction
            if source.tag_fraction is None
            else source.tag_fraction,
            dalitz_density,
            source.time_pdf,
            source.parameters,
        )

    @cached_property
    def background_categories(self):
        return tuple(self._build_background(b) for b in self.backgrounds)

    def with_efficiency(self, efficiency) -> TimeDependentFitSession:
        """Return a copy with a data/normalization-sample efficiency set."""
        return replace(self, efficiency=efficiency)

    def with_veto(self, veto) -> TimeDependentFitSession:
        """Return a copy with a data/normalization-sample veto set."""
        return replace(self, veto=veto)

    def with_constraint(self, constraint) -> TimeDependentFitSession:
        """Return a copy with an added constraint applied to the objective."""
        return replace(self, constraints=self.constraints + (constraint,))

    @cached_property
    def acceptance_data(self) -> jnp.ndarray:
        return _acceptance(self.efficiency, self.veto, self.data.as_dict())

    @cached_property
    def acceptance_normalization(self) -> jnp.ndarray:
        sample = self.model.normalization_sample
        return _acceptance(self.efficiency, self.veto, sample.as_dict())

    @cached_property
    def _abar_components(self) -> tuple[AmplitudeComponent, ...]:
        model = self.model if self.abar_model is None else self.abar_model
        return tuple(
            AmplitudeComponent(
                "bar_" + c.name,
                (
                    _ReflectedAmplitude(c.function)
                    if self.abar_model is None
                    else c.function
                ),
                c.coefficient,
                model.normalize_components
                if c.normalize_component is None
                else c.normalize_component,
            )
            for c in model.amplitude_model.components
        )

    @cached_property
    def _cache_parameters(self) -> tuple[Parameter, ...]:
        # A shared fit value may drive two separately named cache components.
        # Preserve its public name/backend name and remap only the owner.
        model = self.model if self.abar_model is None else self.abar_model
        return tuple(self.model.parameters) + tuple(
            replace(p, owner="bar_" + p.owner)
            for p in model.parameters
            if p.kind is ParameterKind.DYNAMICS and p.owner is not None
        )

    @cached_property
    def n_particle_components(self) -> int:
        return len(self.model.amplitude_model.components)

    @cached_property
    def cache(self) -> PreparedAmplitudeCache:
        return self._prepare_cache(self.data)

    def _prepare_cache(self, data: PhaseSpaceSample) -> PreparedAmplitudeCache:
        components = self.model.amplitude_model.components + self._abar_components
        sample = self.model.normalization_sample
        return PreparedAmplitudeCache.prepare(
            components,
            data=data.as_dict(),
            normalization_data=sample.as_dict(),
            normalization_weights=sample.weights,
            efficiency_normalization=(
                None
                if self.efficiency is None and self.veto is None
                else self.acceptance_normalization
            ),
            normalize_components=self.model.normalize_components,
            parameters=self._cache_parameters,
        )

    @cached_property
    def signal_objective(self) -> TimeDependentDalitzNLL:
        return TimeDependentDalitzNLL(
            self.cache,
            self.n_particle_components,
            self.times,
            self.tags,
            self.mixing,
            efficiency=(
                1.0
                if self.efficiency is None and self.veto is None
                else self.acceptance_data
            ),
            wrong_tag=self.wrong_tag,
            time_range=self.time_range,
            time_nodes=self.time_nodes,
            time_weights=self.time_weights,
            time_acceptance=self.time_acceptance,
            sigma_t=self.sigma_t,
        )

    @cached_property
    def base_objective(self):
        signal = self.signal_objective
        if (
            not self.backgrounds
            and not self.extended
            and self.signal_fraction is None
            and self.signal_yield is None
            and self.signal_tag_fraction is None
            and self.production_fraction is None
        ):
            return signal
        if not self.extended and (
            self.signal_tag_fraction is not None
            or any(
                getattr(b, "tag_fraction", None) is not None for b in self.backgrounds
            )
        ):
            raise ValueError("tag_fraction requires extended=True")
        return TimeDependentMixtureNLL(
            signal_density=signal.densities,
            backgrounds=self.background_categories,
            signal_fraction=self.signal_fraction,
            extended=self.extended,
            signal_yield=self.signal_yield,
            tags=self.tags,
            signal_tag_fraction=self._signal_tag_fraction,
            signal_validity=signal._physical_parameters,
        )

    @cached_property
    def objective(self):
        return (
            ConstrainedNLL(self.base_objective, *self.constraints)
            if self.constraints
            else self.base_objective
        )

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """All fit parameters, deduplicated across model(s) and mixing.

        Raises if two sources give conflicting definitions for the same name.
        """
        candidates = list(self.model.parameters)
        if self.abar_model is not None:
            candidates += list(self.abar_model.parameters)
        candidates += list(self.mixing.parameters)
        candidates += _collect_parameters(self.wrong_tag)
        candidates += _collect_parameters(self.time_acceptance)
        candidates += _collect_parameters(self.constraints)
        candidates += _collect_parameters(self.backgrounds)
        candidates += _collect_parameters(self.signal_fraction)
        candidates += _collect_parameters(self.signal_yield)
        candidates += _collect_parameters(self.signal_tag_fraction)
        candidates += _collect_parameters(self.production_fraction)
        unique: dict[str, Parameter] = {}
        for p in candidates:
            if p.name in unique and unique[p.name] != p:
                raise ValueError(
                    f"conflicting definitions for fit parameter {p.name!r}"
                )
            unique[p.name] = p
        return tuple(unique.values())

    def minimizer(
        self, *, tolerance: float = 1e-4, verbose: int = 0,
        hessian: str = "numerical",
    ) -> Minimizer:
        """Build a Minimizer over the (optionally constrained) objective."""
        return Minimizer(
            self.objective, self.parameters,
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        )

    def fit(
        self, start_values=None, *, simplex: bool = False, ncall=None,
        strategy: int = 2, hesse: bool = True, tolerance: float = 1e-4,
        verbose: int = 0, hessian: str = "numerical",
        method: str = "minuit", nesterov_max_iter: int = 1000,
        nesterov_gtol: float = 1e-4, update_model: bool = False,
    ):
        """Fit the tagged time-dependent likelihood.

        ``self.model``/``self.abar_model`` are frozen and never mutated by
        this call. Pass ``update_model=True`` to also get a model (or a
        ``(model, abar_model)`` pair, if ``abar_model`` was given) with every
        free parameter's ``.value`` set to its fitted result (via
        ``model_with_fitted_values``) -- the return value then becomes
        ``(result, model)`` / ``(result, model, abar_model)`` instead of
        plain ``result``.
        """
        result = self.minimizer(
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        ).fit(
            start_values=start_values, simplex=simplex, ncall=ncall,
            strategy=strategy, hesse=hesse, method=method,
            nesterov_max_iter=nesterov_max_iter, nesterov_gtol=nesterov_gtol,
        )
        if not update_model:
            return result
        values = self.result_values(result)
        fitted_model = model_with_fitted_values(self.model, values)
        if self.abar_model is None:
            return result, fitted_model
        return result, fitted_model, model_with_fitted_values(self.abar_model, values)

    def fit_multistart(
        self, n_starts: int = 20, *, seed=None, include_default: bool = False,
        simplex: bool = False, strategy: int = 1, tolerance: float = 1e-4,
        verbose: int = 0, hessian: str = "numerical",
    ):
        """Fit from multiple random starts, keep the best fit."""
        return self.minimizer(
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        ).fit_multistart(
            n_starts=n_starts, seed=seed, include_default=include_default,
            simplex=simplex, strategy=strategy,
        )

    def result_values(self, result) -> dict[str, float]:
        """Map each parameter name to its fitted value (fixed value if not floated)."""
        return {
            p.name: (float(p.value) if p.fixed else float(result.values[p.name]))
            for p in self.parameters
        }

    def print_result(self, result, *, precision: int = 6) -> dict[str, float]:
        """Print validity, NLL and a value/error table, and return `result_values`."""
        if precision < 0:
            raise ValueError("precision must be non-negative")
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for p in self.parameters:
            error = 0.0 if p.fixed else float(result.errors[p.name])
            print(
                f"{p.name:24s} {values[p.name]:16.{precision}g} "
                f"{error:16.{precision}g}"
            )
        return values

    def print_fit_fractions(
        self, result, *, acceptance_weighted: bool = False,
        include_interference: bool = False, precision: int = 3,
        with_errors: bool = False,
    ) -> dict[str, float]:
        """Print and return the A-model's (D0's) fit fractions at `result`'s values.

        Delegates to ``self.model.print_fit_fractions`` -- the same c^dagger M
        c convention used everywhere else in the package, evaluated on
        ``model``'s own normalization_sample/method, independent of the
        time-dependent likelihood wrapping it. ``with_errors=True`` -- the
        same flag ``DecayModel.print_fit_fractions`` exposes -- also prints
        each fraction's (and, with ``include_interference=True``, each
        interference term's) delta-method standard error (see
        ``fit_fraction_errors()``) as an extra column.
        """
        return self.model.print_fit_fractions(
            self.result_values(result),
            efficiency=self.efficiency if acceptance_weighted else None,
            include_interference=include_interference,
            precision=precision,
            with_errors=with_errors,
            covariance=result.covariance if with_errors else None,
        )

    def fit_fraction_errors(
        self, result, *, acceptance_weighted: bool = False,
    ) -> dict[str, float]:
        """Delta-method standard errors for `print_fit_fractions()`'s central values.

        See ``DecayModel.fit_fraction_errors`` and
        ``jaxpwa.observables.delta_method_errors`` for the
        propagation itself.
        """
        return self.model.fit_fraction_errors(
            self.result_values(result),
            result.covariance,
            efficiency=self.efficiency if acceptance_weighted else None,
        )

    def report(
        self, result, *, include_fit_fractions: bool = True,
        acceptance_weighted_fractions: bool = False,
        include_correlation: bool = True,
    ) -> dict[str, object]:
        """Assemble a summary dict of fit results: validity, NLL, EDM, values, errors.

        Optionally includes the A-model's fit fractions
        (`include_fit_fractions`) and the free-parameter correlation matrix
        (`include_correlation`, only if a covariance is available).
        """
        values = self.print_result(result)
        errors = {
            p.name: (0.0 if p.fixed else float(result.errors[p.name]))
            for p in self.parameters
        }
        report: dict[str, object] = {
            "valid": bool(result.valid),
            "nll": float(result.fval),
            "edm": float(result.fmin.edm),
            "nfcn": int(result.nfcn),
            "values": values,
            "errors": errors,
        }
        if include_fit_fractions:
            report["fit_fractions"] = self.print_fit_fractions(
                result, acceptance_weighted=acceptance_weighted_fractions,
            )
        if include_correlation and getattr(result, "covariance", None) is not None:
            correlation = result.covariance.correlation()
            free = [p.name for p in self.parameters if not p.fixed]
            report["correlation"] = {
                first: {second: float(correlation[first, second]) for second in free}
                for first in free
            }
        return report

    def _projection_scales(self, values, tag):
        """Expected per-tag counts in the full selection, before plot cuts."""
        if self.extended:

            def scale(yield_, fraction):
                probability = float(_resolve(fraction, values))
                return float(_resolve(yield_, values)) * (
                    probability if tag == 1 else 1 - probability
                )

            return (
                scale(self.signal_yield, self._signal_tag_fraction),
                [scale(c.yield_, c.tag_fraction) for c in self.background_categories],
            )
        n = int(np.sum(np.asarray(self.tags) == tag))
        if not self.backgrounds:
            return float(n), []
        fraction = float(_resolve(self.signal_fraction, values))
        weights = np.asarray(self.base_objective.background_weights(values))
        return n * fraction, list(n * (1 - fraction) * weights)

    def _background_projection_data(self, data, tag):
        data = dict(data)
        size = len(next(iter(data.values())))
        data["tag"] = jnp.full(size, tag)
        if self.sigma_t is not None:
            if jnp.ndim(self.sigma_t) != 0:
                raise ValueError("background projections require a scalar sigma_t")
            data["sigma_t"] = jnp.full(size, self.sigma_t)
        return data

    @staticmethod
    def _background_marginal(category, kind, data, values):
        callback = getattr(category, kind)
        if callback is None:
            raise ValueError(
                f"background {category.name!r} requires {kind} for this projection"
            )
        size = len(next(iter(data.values())))
        density = np.broadcast_to(np.asarray(callback(data, values)), (size,))
        if np.any(~np.isfinite(density) | (density < 0)):
            raise ValueError("background marginal must be finite and nonnegative")
        return density

    def plot_time_projection(
        self, result, *, bins: int = 60, range: tuple[float, float] | None = None,
        curve_points: int = 400, time_unit: str = "", log_scale: bool = False,
        ax=None,
    ):
        """Overlay each tag's observed decay-time histogram against the
        model's exact Dalitz-integrated prediction.

        The curve comes from
        ``TimeDependentDalitzNLL.dalitz_integrated_time_pdf`` -- an analytic
        Dalitz marginal, not a rendering MC sample -- so it shares that
        method's scope: requires unit temporal acceptance and perfect
        resolution (no ``time_acceptance``/``sigma_t``) and a scalar
        ``wrong_tag``. A tag with zero observed events is skipped. Projects
        onto decay time only; see ``plot_projection`` for a Dalitz variable.
        """
        import matplotlib.pyplot as plt

        from jaxpwa.plotting import _bin_width_label, binned_data

        values = self.result_values(result)
        times = np.asarray(self.times)
        tags = np.asarray(self.tags)
        if range is not None:
            hist_range = range
        elif np.isfinite(self.time_range[0]) and np.isfinite(self.time_range[1]):
            hist_range = self.time_range
        else:
            hist_range = (float(times.min()), float(times.max()))
        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        bin_width = edges[1] - edges[0]
        curve_times = np.linspace(hist_range[0], hist_range[1], curve_points)
        plus_curve, minus_curve = self.signal_objective.dalitz_integrated_time_pdf(
            jnp.asarray(curve_times), values,
        )
        plus_curve = np.asarray(plus_curve)
        minus_curve = np.asarray(minus_curve)

        if ax is None:
            _, ax = plt.subplots()
        # Each tag gets its own colour, shared between its data points and its
        # fit curve, so the two tags (both otherwise plotted as circular
        # markers) stay visually distinguishable -- unlike plot_binned_data's
        # single-dataset convention of always-black data points.
        for color, tag, label, curve in (
            ("C0", 1, "D0", plus_curve), ("C1", -1, "D0bar", minus_curve),
        ):
            mask = tags == tag
            n = int(mask.sum())
            if n == 0 and not self.extended:
                continue
            centers, counts, errors, _ = binned_data(times[mask], bins=edges)
            ax.errorbar(
                centers, counts, yerr=errors, fmt="o", color=color, ecolor=color,
                markersize=4.5, linestyle="none", label=f"data ({label})", zorder=10,
            )
            signal_scale, background_scales = self._projection_scales(values, tag)
            total = curve * signal_scale * bin_width
            if self.backgrounds:
                ax.plot(
                    curve_times,
                    total,
                    color=color,
                    linestyle="--",
                    label=f"signal ({label})",
                )
            projection_data = self._background_projection_data({"t": curve_times}, tag)
            for category, scale in zip(
                self.background_categories, background_scales, strict=True
            ):
                if scale == 0:
                    continue
                marginal = self._background_marginal(
                    category,
                    "time_density",
                    projection_data,
                    values,
                )
                component = marginal * scale * bin_width
                total = total + component
                ax.plot(
                    curve_times,
                    component,
                    linestyle=":",
                    label=f"{category.name} ({label})",
                )
            ax.plot(curve_times, total, color=color, label=f"fit ({label})")
        ax.set_ylabel(_bin_width_label(edges, time_unit))
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("t" + (f" [{time_unit}]" if time_unit else ""))
        ax.legend()
        return ax

    def plot_projection(
        self, result, variable: str = "s13", *, bins: int = 60,
        range: tuple[float, float] | None = None, show_pulls: bool = False,
        log_scale: bool = False, projection_size: int = 100_000,
        projection_seed: int = 20260924, axes=None,
    ):
        """Plot each observed tag's Dalitz-variable histogram against the
        model's time-integrated, tag-conditional prediction, one subplot per
        tag (D0 | D0bar), matching ``CPFitSession.plot_projection``'s
        side-by-side layout for its two populations.

        A weighted phase-space MC sample is used only to render the
        one-dimensional projection; the density itself comes from
        ``TimeDependentDalitzNLL.tag_marginal_density``, time-integrated over
        ``self.time_range`` using the same mixing-integral machinery the fit
        itself normalizes against -- not a time-binned approximation. Shares
        that method's scope: requires scalar ``wrong_tag`` and, if supplied,
        scalar ``sigma_t``. The fitted amplitude parameters, component scales,
        efficiency and veto are applied at the rendering points.

        ``show_pulls=True`` adds a ``(observed-expected)/sqrt(expected)``
        panel below each tag's histogram, sharing that column's x axis. It
        builds its own 2x2 figure and therefore requires ``axes=None``; the
        return value is then the full 2x2 axes grid (row 0 the histograms,
        row 1 the pulls) instead of the usual length-2 list.
        """
        import matplotlib.pyplot as plt

        from jaxpwa.plotting import _draw_pulls_1d, plot_binned_data
        from jaxpwa.workflow import _scaled_projection_weights

        if show_pulls and axes is not None:
            raise ValueError(
                "show_pulls=True builds its own figure layout; pass axes=None"
            )
        values = self.result_values(result)
        tags = np.asarray(self.tags)
        data_values = np.asarray(getattr(self.data, variable))
        hist_range = range or (float(data_values.min()), float(data_values.max()))
        edges = np.histogram_bin_edges(data_values, bins=bins, range=hist_range)

        sample = self.model.generate_phase_space(
            projection_size, seed=projection_seed, include_momenta=False,
        )
        sample_data = sample.as_dict()
        cache = self._prepare_cache(sample)
        particle = jnp.arange(len(cache.components)) < self.n_particle_components
        amplitudes, _ = cache.coherent_groups(
            values, jnp.stack((particle, ~particle), axis=1),
        )
        density_plus, density_minus = self.signal_objective.tag_marginal_density(
            amplitudes[:, 0], amplitudes[:, 1], values,
            efficiency=_acceptance(self.efficiency, self.veto, sample_data),
        )
        sample_values = np.asarray(getattr(sample, variable))

        grid = None
        pulls_axes = (None, None)
        if axes is None:
            # Two side-by-side tag panels (and, with pulls, a second row
            # below each) at the active style's own base size
            # (`plt.style.use(...)`, e.g. mplhep), scaled by panel count
            # rather than a hardcoded absolute figsize.
            base_w, base_h = plt.rcParams["figure.figsize"]
            if show_pulls:
                _, grid = plt.subplots(
                    2, 2, figsize=(base_w * 2, base_h * 1.2), sharex="col",
                    gridspec_kw={"height_ratios": (3, 1)}, constrained_layout=True,
                )
                axes, pulls_axes = grid[0], grid[1]
            else:
                _, axes = plt.subplots(
                    1, 2, figsize=(base_w * 2, base_h), constrained_layout=True
                )

        unit = r"GeV$^2$" if variable in ("s12", "s13", "s23") else ""
        axis_label = rf"${variable}$" + (" [GeV$^2$]" if unit else "")
        for ax, ax_pulls, tag, label, density in zip(
            axes, pulls_axes, (1, -1), ("D0", "D0bar"), (density_plus, density_minus),
            strict=True,
        ):
            mask = tags == tag
            _, observed, _, _ = plot_binned_data(
                data_values[mask], bins=edges, ax=ax, label=f"data ({label})",
                unit=unit, log_scale=log_scale,
            )
            signal_scale, background_scales = self._projection_scales(values, tag)
            weights = _scaled_projection_weights(
                sample, np.asarray(density), signal_scale
            )
            total, _ = np.histogram(sample_values, bins=edges, weights=weights)
            if self.backgrounds:
                ax.stairs(total, edges, label=f"signal ({label})", linestyle="--")
            projection_data = self._background_projection_data(sample_data, tag)
            for category, scale in zip(
                self.background_categories, background_scales, strict=True
            ):
                if scale == 0:
                    continue
                marginal = self._background_marginal(
                    category,
                    "dalitz_density",
                    projection_data,
                    values,
                )
                bg_weights = _scaled_projection_weights(sample, marginal, scale)
                component, _ = np.histogram(
                    sample_values, bins=edges, weights=bg_weights
                )
                total = total + component
                ax.stairs(
                    component, edges, label=f"{category.name} ({label})", linestyle=":"
                )
            ax.stairs(total, edges, label=f"fit ({label})", linewidth=2.0)
            ax.set_title(label)
            ax.legend()
            if ax_pulls is None:
                ax.set_xlabel(axis_label)
                continue
            occupied = total > 0
            pulls = np.full(bins, np.nan)
            pulls[occupied] = (observed[occupied] - total[occupied]) / np.sqrt(
                total[occupied]
            )
            _draw_pulls_1d(ax_pulls, edges, pulls)
            ax_pulls.set_xlabel(axis_label)
        return grid if show_pulls else axes


__all__ = ["TimeDependentBackgroundSpec", "TimeDependentFitSession"]
