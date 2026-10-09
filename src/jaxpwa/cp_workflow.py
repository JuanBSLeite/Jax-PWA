"""High-level workflow for simultaneous direct-CP Dalitz fits."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa.background import CPBackgroundCategory
from jaxpwa.constraints import ConstrainedNLL
from jaxpwa.fit import Minimizer, Parameter
from jaxpwa.goodness_of_fit import (
    BinnedChi2Result,
    PointToPointResult,
    chi2_from_histograms,
)
from jaxpwa.goodness_of_fit import (
    point_to_point_dissimilarity as _point_to_point_dissimilarity,
)
from jaxpwa.io import cp_models_with_fitted_values, read_phase_space_sample
from jaxpwa.kinematics import (
    PhaseSpaceSample,
    SquareDalitzGrid,
    fold_thetaprime,
    invariants_to_square_dalitz,
)
from jaxpwa.likelihood import CPJointNLL
from jaxpwa.likelihood.square_dalitz import (
    SquareDalitzNLL,
    square_dalitz_log_jacobian,
    validate_session_coordinates,
)
from jaxpwa.likelihood.cp import _signal_yield_pair
from jaxpwa.observables.errors import _covariance_matrix
from jaxpwa.plotting import _draw_pulls_1d, plot_binned_data
from jaxpwa.sampling import weighted_resample
from jaxpwa.workflow import _fit_with_optional_weights


def _collect_parameters(value: object) -> tuple[Parameter, ...]:
    if isinstance(value, Parameter):
        return (value,)
    if value is None:
        return ()
    parameters = getattr(value, "parameters", None)
    if parameters is not None and not callable(parameters):
        try:
            return tuple(item for item in parameters if isinstance(item, Parameter))
        except TypeError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        found: list[Parameter] = []
        for field in fields(value):
            found.extend(_collect_parameters(getattr(value, field.name)))
        return tuple(found)
    if isinstance(value, Mapping):
        return tuple(parameter for item in value.values() for parameter in _collect_parameters(item))
    if isinstance(value, (tuple, list)):
        return tuple(parameter for item in value for parameter in _collect_parameters(item))
    return ()


def _resolve(value: object, parameters: Mapping[str, object]):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    values = jnp.ones((size,), dtype=jnp.float64)
    for label, function in (("efficiency", efficiency), ("veto", veto)):
        if function is not None:
            array = jnp.asarray(function(data))
            if array.ndim == 0:
                array = jnp.full((size,), array)
            if array.shape != (size,):
                raise ValueError(f"CP {label} must have shape ({size},)")
            if bool(jnp.any(~jnp.isfinite(array) | (array < 0))):
                raise ValueError(f"CP {label} must be finite and non-negative")
            values = values * array
    return values


def _joint_scaled_weights(plus_sample, plus_density, minus_sample, minus_density, scale: float):
    if scale == 0:
        return np.zeros(plus_sample.size), np.zeros(minus_sample.size)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("projection yield must be finite and non-negative")
    if plus_sample.size == 0 or minus_sample.size == 0:
        raise ValueError("projection samples must be non-empty")
    plus_raw = np.asarray(plus_sample.weights, dtype=float) * np.asarray(plus_density, dtype=float) / plus_sample.size
    minus_raw = np.asarray(minus_sample.weights, dtype=float) * np.asarray(minus_density, dtype=float) / minus_sample.size
    total = float(np.sum(plus_raw) + np.sum(minus_raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("joint projection density has non-positive or non-finite integral")
    factor = float(scale) / total
    return factor * plus_raw, factor * minus_raw


@dataclass(frozen=True)
class CPBackgroundSpec:
    name: str
    plus_shape: object
    minus_shape: object | None = None
    fraction: object | None = None
    yield_: object | None = None
    plus_normalization_sample: PhaseSpaceSample | None = None
    minus_normalization_sample: PhaseSpaceSample | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CP background name must be non-empty")
        if not callable(self.plus_shape):
            raise TypeError("plus_shape must be callable on an event-data mapping")
        if self.minus_shape is not None and not callable(self.minus_shape):
            raise TypeError("minus_shape must be callable on an event-data mapping")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a CP background cannot define both fraction and yield")

    @property
    def resolved_minus_shape(self):
        """The B- background shape, falling back to plus_shape when unset."""
        return self.plus_shape if self.minus_shape is None else self.minus_shape


@dataclass(frozen=True)
class CPFitSession:
    plus_model: object
    minus_model: object
    plus_data: PhaseSpaceSample
    minus_data: PhaseSpaceSample
    plus_efficiency: object | None = None
    minus_efficiency: object | None = None
    plus_veto: object | None = None
    minus_veto: object | None = None
    backgrounds: tuple[CPBackgroundSpec | CPBackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    constraints: tuple[object, ...] = ()
    plus_event_weights: object | None = None
    minus_event_weights: object | None = None
    # "square-dalitz": fit B+/B- events as points of each model's Square-Dalitz
    # plane (m', theta'); see FitSession and SquareDalitzNLL. The joint B+/B-
    # normalization is unchanged.
    coordinates: str = "dalitz"

    def __post_init__(self):
        validate_session_coordinates(
            self.coordinates,
            models=(self.plus_model, self.minus_model),
            efficiencies=(self.plus_efficiency, self.minus_efficiency),
            background_shapes=tuple(
                shape
                for background in self.backgrounds
                for shape in (
                    getattr(background, "plus_shape", None),
                    getattr(background, "minus_shape", None),
                )
            ),
        )
        if self.event_weights is not None and (
            self.backgrounds or self.extended or self.signal_fraction is not None
            or self.signal_yield is not None
        ):
            raise ValueError(
                "event weights require a signal-only CPFitSession without "
                "background/fraction/yield configuration"
            )

    @classmethod
    def from_root(cls, plus_model, minus_model, plus_file, plus_tree, minus_file, minus_tree, *, plus_root_kwargs=None, minus_root_kwargs=None, **session_kwargs):
        """Build a CPFitSession, reading plus/minus samples from separate ROOT files."""
        plus_data = read_phase_space_sample(plus_file, plus_tree, **({} if plus_root_kwargs is None else dict(plus_root_kwargs)))
        minus_data = read_phase_space_sample(minus_file, minus_tree, **({} if minus_root_kwargs is None else dict(minus_root_kwargs)))
        return cls(plus_model, minus_model, plus_data, minus_data, **session_kwargs)

    def with_efficiency(self, plus_efficiency, minus_efficiency=None):
        """Return a copy with plus/minus efficiencies set, defaulting minus to plus."""
        return replace(self, plus_efficiency=plus_efficiency, minus_efficiency=plus_efficiency if minus_efficiency is None else minus_efficiency)

    def with_veto(self, plus_veto, minus_veto=None):
        """Return a copy with plus/minus vetoes set, defaulting minus to plus."""
        return replace(self, plus_veto=plus_veto, minus_veto=plus_veto if minus_veto is None else minus_veto)

    def with_background(self, name, plus_shape, *, minus_shape=None, fraction=None, yield_=None, plus_normalization_sample=None, minus_normalization_sample=None, apply_veto=True):
        """Return a copy with an added charge-aware background component."""
        return replace(self, backgrounds=self.backgrounds + (CPBackgroundSpec(name, plus_shape, minus_shape, fraction, yield_, plus_normalization_sample, minus_normalization_sample, apply_veto),))

    def with_constraint(self, constraint):
        """Return a copy with an added constraint applied to the joint objective."""
        return replace(self, constraints=self.constraints + (constraint,))

    def with_event_weights(self, plus_weights, minus_weights):
        """Return a signal-only copy whose data carry per-event signal weights.

        For sWeight/COW background subtraction: ``objective`` becomes the
        weighted joint objective, ``fit()`` uses these weights unless others
        are passed, and projections draw weighted data histograms
        (``sqrt(sum w^2)`` errors) against a signal model scaled to the
        weighted sums. Binned/unbinned goodness-of-fit tests are not defined
        for weighted data and raise. See ``fit``.
        """
        return replace(self, plus_event_weights=plus_weights, minus_event_weights=minus_weights)

    @cached_property
    def event_weights(self):
        """The ``(plus, minus)`` session event weights, or None if unweighted."""
        if (self.plus_event_weights is None) != (self.minus_event_weights is None):
            raise ValueError("plus_event_weights and minus_event_weights must be supplied together")
        if self.plus_event_weights is None:
            return None
        return self._split_weights((self.plus_event_weights, self.minus_event_weights))

    def _data_weights(self, charge):
        """Per-event data weights of one charge (ones when unweighted)."""
        weights = self.event_weights
        sample = self.plus_data if charge == "plus" else self.minus_data
        if weights is None:
            return np.ones(sample.size)
        return np.asarray(weights[0] if charge == "plus" else weights[1], dtype=float)

    def _total_data_events(self):
        """N_plus + N_minus, or the weighted sum for an event-weighted session."""
        weights = self.event_weights
        if weights is None:
            return self.plus_data.size + self.minus_data.size
        return float(jnp.sum(weights[0]) + jnp.sum(weights[1]))

    def _require_unweighted(self, method):
        if self.event_weights is not None:
            raise NotImplementedError(
                f"{method} assumes unweighted (Poisson) data and is not defined for an "
                "event-weighted (sWeight/COW) CPFitSession"
            )

    @cached_property
    def plus_acceptance_data(self):
        return _acceptance(self.plus_efficiency, self.plus_veto, self.plus_data.as_dict())

    @cached_property
    def minus_acceptance_data(self):
        return _acceptance(self.minus_efficiency, self.minus_veto, self.minus_data.as_dict())

    @cached_property
    def plus_acceptance_normalization(self):
        s = self.plus_model.normalization_sample
        return _acceptance(self.plus_efficiency, self.plus_veto, s.as_dict())

    @cached_property
    def minus_acceptance_normalization(self):
        s = self.minus_model.normalization_sample
        return _acceptance(self.minus_efficiency, self.minus_veto, s.as_dict())

    @cached_property
    def plus_cache(self):
        return self.plus_model.prepare_cache(self.plus_data, self.plus_model.normalization_sample, efficiency_normalization=(None if self.plus_efficiency is None and self.plus_veto is None else self.plus_acceptance_normalization))

    @cached_property
    def minus_cache(self):
        return self.minus_model.prepare_cache(self.minus_data, self.minus_model.normalization_sample, efficiency_normalization=(None if self.minus_efficiency is None and self.minus_veto is None else self.minus_acceptance_normalization))

    @staticmethod
    def _evaluate_shape(shape, data):
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(f"background shape must return one value per event, got {values.shape} for {size} events")
        return values

    def _build_background(self, background):
        if isinstance(background, CPBackgroundCategory):
            return background
        plus_norm = self.plus_model.normalization_sample if background.plus_normalization_sample is None else background.plus_normalization_sample
        minus_norm = self.minus_model.normalization_sample if background.minus_normalization_sample is None else background.minus_normalization_sample
        plus_shape = background.plus_shape
        minus_shape = background.resolved_minus_shape
        pd, md, pn, mn = self.plus_data.as_dict(), self.minus_data.as_dict(), plus_norm.as_dict(), minus_norm.as_dict()
        pv, mv = self._evaluate_shape(plus_shape, pd), self._evaluate_shape(minus_shape, md)
        pnv, mnv = self._evaluate_shape(plus_shape, pn), self._evaluate_shape(minus_shape, mn)
        if background.apply_veto:
            if self.plus_veto is not None:
                pv *= jnp.asarray(self.plus_veto(pd)); pnv *= jnp.asarray(self.plus_veto(pn))
            if self.minus_veto is not None:
                mv *= jnp.asarray(self.minus_veto(md)); mnv *= jnp.asarray(self.minus_veto(mn))
        return CPBackgroundCategory(background.name, pv, mv, jnp.mean(plus_norm.weights * pnv), jnp.mean(minus_norm.weights * mnv), background.fraction, background.yield_)

    @cached_property
    def background_categories(self):
        return tuple(self._build_background(b) for b in self.backgrounds)

    @cached_property
    def base_objective(self):
        if self.event_weights is not None:
            return self._weighted_nll(self.event_weights)
        return CPJointNLL(self.plus_cache, self.minus_cache, plus_efficiency=self.plus_acceptance_data, minus_efficiency=self.minus_acceptance_data, background_categories=self.background_categories, signal_fraction=self.signal_fraction, extended=self.extended, signal_yield=self.signal_yield)

    def _in_coordinates(self, nll, weights=None):
        """Wrap a joint event NLL for this session's ``coordinates``."""
        if self.coordinates == "dalitz":
            return nll
        total = 0.0
        for index, (model, data) in enumerate(
            ((self.plus_model, self.plus_data), (self.minus_model, self.minus_data))
        ):
            log_jacobian = square_dalitz_log_jacobian(model, data)
            if weights is not None:
                log_jacobian = jnp.asarray(weights[index]) * log_jacobian
            total = total + jnp.sum(log_jacobian)
        return SquareDalitzNLL(nll, total)

    @cached_property
    def objective(self):
        nll = self._in_coordinates(self.base_objective, self.event_weights)
        return ConstrainedNLL(nll, *self.constraints) if self.constraints else nll

    def _split_weights(self, weights):
        """Validate ``weights=(plus_weights, minus_weights)`` against the data."""
        if not isinstance(weights, (tuple, list)) or len(weights) != 2:
            raise TypeError("CP event weights must be a (plus_weights, minus_weights) pair")
        plus_weights, minus_weights = (jnp.asarray(w) for w in weights)
        for label, array, sample in (("plus", plus_weights, self.plus_data), ("minus", minus_weights, self.minus_data)):
            if array.shape != (sample.size,):
                raise ValueError(f"{label} weights must have shape ({sample.size},), got {array.shape}")
            if jnp.iscomplexobj(array):
                raise ValueError(f"{label} weights must be real")
            if not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError(f"{label} weights must be finite")
        return plus_weights, minus_weights

    def _weighted_nll(self, weights):
        """Signal-only weighted ``CPJointNLL`` (no constraints) for sWeight/COW CP fits."""
        if self.backgrounds or self.extended or self.signal_fraction is not None or self.signal_yield is not None:
            raise ValueError(
                "event weights are incompatible with explicit background/fraction/"
                "yield configuration; build a signal-only CPFitSession for an sWeight/COW fit"
            )
        plus_weights, minus_weights = self._split_weights(weights)
        return CPJointNLL(
            self.plus_cache, self.minus_cache,
            plus_efficiency=self.plus_acceptance_data, minus_efficiency=self.minus_acceptance_data,
            plus_weights=plus_weights, minus_weights=minus_weights,
        )

    def _weighted_objective(self, weights):
        """Weighted joint NLL plus this session's constraints."""
        nll = self._in_coordinates(
            self._weighted_nll(weights), self._split_weights(weights)
        )
        return ConstrainedNLL(nll, *self.constraints) if self.constraints else nll

    def _score_outer_objective(self, weights, reference_parameters):
        """Scalar whose Hessian at the reference is ``sum_i w_i^2 s_i s_i^T`` over both charges.

        Same construction as ``FitSession._score_outer_objective``: every
        residual ``log p_i(theta) - log p_i(theta_hat)`` vanishes at
        ``theta_hat``, so the Hessian of ``0.5 sum_i w_i^2 delta_i^2`` there is
        exactly the weighted score outer product. Charges are summed, since
        both share one jointly normalized density.
        """
        nll = self._weighted_nll(weights)
        plus_weights, minus_weights = nll.plus_weights, nll.minus_weights
        reference = tuple(
            jax.lax.stop_gradient(values)
            for values in nll.signal_log_densities(reference_parameters)
        )
        squared = (jnp.square(plus_weights), jnp.square(minus_weights))

        def objective(parameters):
            total = 0.0
            for weight2, current, ref in zip(squared, nll.signal_log_densities(parameters), reference, strict=True):
                active = weight2 != 0
                delta = jnp.where(active, current - jnp.where(active, ref, 0.0), 0.0)
                total = total + 0.5 * jnp.sum(weight2 * jnp.square(delta))
            return total

        return objective

    @property
    def parameters(self):
        """All fit parameters, deduplicated across both models and backgrounds.

        Raises if two sources give conflicting definitions for the same name.
        """
        candidates = list(getattr(self.plus_model, "parameters", ())) + list(getattr(self.minus_model, "parameters", ()))
        candidates.extend(_collect_parameters(self.signal_fraction)); candidates.extend(_collect_parameters(self.signal_yield)); candidates.extend(_collect_parameters(self.backgrounds)); candidates.extend(_collect_parameters(self.constraints))
        unique = {}
        for p in candidates:
            if p.name in unique and unique[p.name] != p:
                raise ValueError(f"conflicting definitions for fit parameter {p.name!r}")
            unique[p.name] = p
        return tuple(unique.values())

    def minimizer(
        self, *, tolerance=1e-4, verbose=0, hessian="numerical",
    ):
        """Build a Minimizer over the joint objective and this session's parameters."""
        return Minimizer(
            self.objective, self.parameters,
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        )

    def fit(
        self, start_values=None, *, weights=None, covariance="minuit",
        simplex=False, ncall=None, strategy=2,
        hesse=True, tolerance=1e-4, verbose=0, hessian="numerical",
        method="minuit", nesterov_max_iter=1000, nesterov_gtol=1e-4,
        update_model=False,
    ):
        """Fit the joint B+/B- likelihood.

        Pass ``weights=(plus_weights, minus_weights)`` (e.g. sWeights/COWs,
        one array per charge, aligned with ``plus_data``/``minus_data``) to
        minimize the signal-only weighted joint objective
        ``-sum_i w_i log S_q(phi_i)``, with the same joint charge
        normalization as the unweighted fit (see ``CPJointNLL``).
        ``covariance`` has the same meaning as in ``FitSession.fit``:
        ``"sandwich"`` (recommended for signed weights) installs the Godambe
        covariance ``H_w^-1 (sum_i w_i^2 s_i s_i^T) H_w^-1``, ``"sumw2"``
        (alias ``"sweight"``) ``H_w^-1 H_w2 H_w^-1``, and ``"minuit"`` keeps
        Minuit's uncorrected HESSE. Both corrections treat the weights as
        fixed: they do not propagate the uncertainty of the fit that
        determined them. Weighted fits must be signal-only: no background,
        ``signal_fraction``, ``signal_yield`` or ``extended``. A session built
        with ``with_event_weights`` uses its own weights when ``weights`` is
        omitted.

        ``self.plus_model``/``self.minus_model`` are frozen and never mutated
        by this call. Pass ``update_model=True`` to also get a
        ``(plus_model, minus_model)`` pair with every free parameter's
        ``.value`` set to its fitted result (via
        ``cp_models_with_fitted_values``, which keeps a coefficient shared
        between charges shared in the returned models too) -- the return
        value then becomes ``(result, plus_model, minus_model)`` instead of
        plain ``result``.
        """
        result = _fit_with_optional_weights(
            unweighted_minimizer=lambda: self.minimizer(
                tolerance=tolerance, verbose=verbose, hessian=hessian,
            ),
            weighted_objective=self._weighted_objective,
            event_weighted_objective=self._weighted_nll,
            score_outer_objective=self._score_outer_objective,
            parameters=self.parameters,
            weights=self.event_weights if weights is None else tuple(weights),
            covariance=covariance,
            start_values=start_values,
            simplex=simplex,
            ncall=ncall,
            strategy=strategy,
            hesse=hesse,
            tolerance=tolerance,
            verbose=verbose,
            hessian=hessian,
            method=method,
            nesterov_max_iter=nesterov_max_iter,
            nesterov_gtol=nesterov_gtol,
        )
        if not update_model:
            return result
        plus_model, minus_model = cp_models_with_fitted_values(
            self.plus_model, self.result_values(result), self.minus_model
        )
        return result, plus_model, minus_model

    def fit_multistart(
        self, n_starts=20, *, seed=None, include_default=False, simplex=False,
        strategy=1, tolerance=1e-4, verbose=0, hessian="numerical",
    ):
        """Fit the joint likelihood from multiple random starts, keep the best fit."""
        return self.minimizer(
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        ).fit_multistart(
            n_starts=n_starts,
            seed=seed,
            include_default=include_default,
            simplex=simplex,
            strategy=strategy,
        )

    def result_values(self, result):
        """Map each parameter name to its fitted (or fixed) value from a fit result."""
        return {p.name: (float(p.value) if p.fixed else float(result.values[p.name])) for p in self.parameters}

    def print_result(self, result, *, precision=6):
        """Print fitted parameter values/errors and predicted B+/B- charge fractions.

        Returns the values dict.
        """
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for p in self.parameters:
            print(f"{p.name:24s} {values[p.name]:16.{precision}g} {(0.0 if p.fixed else float(result.errors[p.name])):16.{precision}g}")
        pplus, pminus = self.base_objective.charge_probabilities(values)
        print(f"predicted charge fractions: B+={float(pplus):.6f}  B-={float(pminus):.6f}")
        return values

    def fit_fraction_errors(self, result, *, acceptance_weighted=False):
        """Delta-method standard errors for print_fit_fractions()'s central values.

        Propagates the *joint* postfit covariance through both charges' fit
        fractions in a single Jacobian, not two independent ones -- plus_model
        and minus_model share almost every fit parameter (every CPRealImag
        coefficient and every dynamics parameter), so their fit fractions are
        correlated. That correlation is exactly what "mean" below needs:
        Var(mean) = 0.25*(Var(plus)+Var(minus)+2*Cov(plus,minus)), not a naive
        quadrature sum of the two charges' errors (see docs/cp_coefficients.md
        on why B+/B- are not independently normalized here). See
        jaxpwa.observables.delta_method_errors for the propagation
        itself.
        """
        values = self.result_values(result)
        plus_cache = self.plus_model._fraction_cache(None, self.plus_efficiency if acceptance_weighted else None)
        minus_cache = self.minus_model._fraction_cache(None, self.minus_efficiency if acceptance_weighted else None)
        plus_names = [component.name for component in plus_cache.components]
        minus_names = [component.name for component in minus_cache.components]
        if plus_names != minus_names:
            raise ValueError("plus_model and minus_model must declare the same components in the same order")

        parameter_names = sorted({
            parameter.name
            for parameter in (*self.plus_model.parameters, *self.minus_model.parameters)
            if not parameter.fixed
        })

        # Differentiate each charge's integration graph only for its own rows.
        # Stack columns in the SAME union-of-parameters order, then propagate
        # the joint covariance once, including every cross-charge term.
        jacobian = jnp.concatenate([
            self.plus_model._fraction_jacobian(plus_cache, values, parameter_names),
            self.minus_model._fraction_jacobian(minus_cache, values, parameter_names),
        ])
        parameter_covariance = _covariance_matrix(result.covariance, parameter_names)
        covariance = np.asarray(jacobian @ parameter_covariance @ jacobian.T)
        n = len(plus_names)
        variance_plus = np.clip(np.diag(covariance)[:n], 0.0, None)
        variance_minus = np.clip(np.diag(covariance)[n:], 0.0, None)
        cross = np.diag(covariance[:n, n:])
        variance_mean = np.clip(0.25 * (variance_plus + variance_minus + 2.0 * cross), 0.0, None)

        return {
            "plus": dict(zip(plus_names, (float(v) for v in np.sqrt(variance_plus)))),
            "minus": dict(zip(minus_names, (float(v) for v in np.sqrt(variance_minus)))),
            "mean": dict(zip(plus_names, (float(v) for v in np.sqrt(variance_mean)))),
        }

    def interference_fraction_errors(self, result, *, acceptance_weighted=False):
        """Delta-method standard errors for interference_fractions(), per charge.

        Same joint-covariance propagation as ``fit_fraction_errors()`` --
        including the plus/minus cross-term needed for a "mean" error -- but
        for the pairwise interference terms instead of the per-component
        fractions. Each returned dict is keyed by ``(name_i, name_j)``.
        """
        values = self.result_values(result)
        plus_cache = self.plus_model._fraction_cache(None, self.plus_efficiency if acceptance_weighted else None)
        minus_cache = self.minus_model._fraction_cache(None, self.minus_efficiency if acceptance_weighted else None)
        plus_names = [component.name for component in plus_cache.components]
        minus_names = [component.name for component in minus_cache.components]
        if plus_names != minus_names:
            raise ValueError("plus_model and minus_model must declare the same components in the same order")

        parameter_names = sorted({
            parameter.name
            for parameter in (*self.plus_model.parameters, *self.minus_model.parameters)
            if not parameter.fixed
        })

        plus_jacobian, pairs = self.plus_model._interference_jacobian(plus_cache, values, parameter_names)
        minus_jacobian, _ = self.minus_model._interference_jacobian(minus_cache, values, parameter_names)

        jacobian = jnp.concatenate([plus_jacobian, minus_jacobian])
        parameter_covariance = _covariance_matrix(result.covariance, parameter_names)
        covariance = np.asarray(jacobian @ parameter_covariance @ jacobian.T)
        n = len(pairs)
        variance_plus = np.clip(np.diag(covariance)[:n], 0.0, None)
        variance_minus = np.clip(np.diag(covariance)[n:], 0.0, None)
        cross = np.diag(covariance[:n, n:])
        variance_mean = np.clip(0.25 * (variance_plus + variance_minus + 2.0 * cross), 0.0, None)

        pair_names = [(plus_cache.components[i].name, plus_cache.components[j].name) for i, j in pairs]
        return {
            "plus": dict(zip(pair_names, (float(v) for v in np.sqrt(variance_plus)))),
            "minus": dict(zip(pair_names, (float(v) for v in np.sqrt(variance_minus)))),
            "mean": dict(zip(pair_names, (float(v) for v in np.sqrt(variance_mean)))),
        }

    def print_fit_fractions(self, result, *, acceptance_weighted=False, include_interference=False, precision=3, with_errors=False):
        """Print and return each charge's per-component fit fractions.

        ``with_errors=True`` -- the same flag ``DecayModel.print_fit_fractions``
        exposes -- also prints each charge's delta-method standard errors as
        an extra column in each table, including for the interference terms
        when ``include_interference=True``. Each charge's errors are
        propagated from its own restriction of the shared postfit
        ``result.covariance`` (equivalent to the "plus"/"minus" entries of
        ``fit_fraction_errors()``/``interference_fraction_errors()``, which
        remain the way to also get the cross-charge "mean" error).
        """
        values = self.result_values(result)
        covariance = result.covariance if with_errors else None
        print("B+ fit fractions")
        plus = self.plus_model.print_fit_fractions(
            values,
            efficiency=self.plus_efficiency if acceptance_weighted else None,
            include_interference=include_interference,
            precision=precision,
            with_errors=with_errors,
            covariance=covariance,
        )
        print("\nB- fit fractions")
        minus = self.minus_model.print_fit_fractions(
            values,
            efficiency=self.minus_efficiency if acceptance_weighted else None,
            include_interference=include_interference,
            precision=precision,
            with_errors=with_errors,
            covariance=covariance,
        )
        return {"plus": plus, "minus": minus}

    def component_cp_asymmetries(self, result, *, acceptance_weighted=False):
        """Integrated CP asymmetry of each amplitude component, with delta-method errors.

        ``A_CP(k) = (I_k^- - I_k^+) / (I_k^- + I_k^+)`` with
        ``I_k^q = integral |c_k^q A_k^q|^2 dPhi`` (``component_intensities``),
        i.e. the per-charge numerators of the fit fractions on the common
        integration scale both charges share. Unlike the coefficient-level
        ``A_CP`` of ``CPRealImag``, this includes CP violation carried by the
        component's dynamics, e.g. the per-node ``dx``/``dy`` of a QMI S-wave
        whose global coefficient is fixed. Physical (efficiency excluded) by
        default; ``acceptance_weighted=True`` weights both integrals by each
        charge's efficiency. Errors propagate the joint postfit covariance
        through both charges in one Jacobian (NaN without a covariance).

        Returns ``{name: {"value", "error", "intensity_plus", "intensity_minus"}}``.
        """
        values = self.result_values(result)
        plus_cache = self.plus_model._fraction_cache(None, self.plus_efficiency if acceptance_weighted else None)
        minus_cache = self.minus_model._fraction_cache(None, self.minus_efficiency if acceptance_weighted else None)
        names = [component.name for component in plus_cache.components]
        if names != [component.name for component in minus_cache.components]:
            raise ValueError("plus_model and minus_model must declare the same components in the same order")

        plus = np.asarray(plus_cache.component_intensities(values), dtype=float)
        minus = np.asarray(minus_cache.component_intensities(values), dtype=float)
        total = plus + minus
        with np.errstate(divide="ignore", invalid="ignore"):
            asymmetry = np.where(total > 0, (minus - plus) / total, np.nan)

        errors = np.full(len(names), np.nan)
        if getattr(result, "covariance", None) is not None:
            parameter_names = sorted({
                parameter.name
                for parameter in (*self.plus_model.parameters, *self.minus_model.parameters)
                if not parameter.fixed
            })
            if not parameter_names:
                errors = np.where(total > 0, 0.0, np.nan)
            else:
                plus_jacobian = np.asarray(self.plus_model._fraction_jacobian(
                    plus_cache, values, parameter_names, "component_intensities"))
                minus_jacobian = np.asarray(self.minus_model._fraction_jacobian(
                    minus_cache, values, parameter_names, "component_intensities"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    d_minus = np.where(total > 0, 2.0 * plus / total**2, 0.0)
                    d_plus = np.where(total > 0, -2.0 * minus / total**2, 0.0)
                jacobian = d_minus[:, None] * minus_jacobian + d_plus[:, None] * plus_jacobian
                covariance = np.asarray(_covariance_matrix(result.covariance, parameter_names))
                variance = np.einsum("kp,pq,kq->k", jacobian, covariance, jacobian)
                errors = np.where(total > 0, np.sqrt(np.clip(variance, 0.0, None)), np.nan)

        return {
            name: {
                "value": float(asymmetry[index]),
                "error": float(errors[index]),
                "intensity_plus": float(plus[index]),
                "intensity_minus": float(minus[index]),
            }
            for index, name in enumerate(names)
        }

    def report(self, result, *, include_fit_fractions=True, acceptance_weighted_fractions=False, include_correlation=True):
        """Assemble a dict summary of the fit.

        Includes validity, NLL, EDM, parameter values/errors, charge
        probabilities, and optionally fit fractions/correlation matrix.
        """
        values = self.print_result(result)
        errors = {p.name: (0.0 if p.fixed else float(result.errors[p.name])) for p in self.parameters}
        pplus, pminus = self.base_objective.charge_probabilities(values)
        out = {"valid": bool(result.valid), "nll": float(result.fval), "edm": float(result.fmin.edm), "nfcn": int(result.nfcn), "values": values, "errors": errors, "charge_probabilities": {"plus": float(pplus), "minus": float(pminus)}}
        if include_fit_fractions:
            out["fit_fractions"] = self.print_fit_fractions(result, acceptance_weighted=acceptance_weighted_fractions)
        if include_correlation and getattr(result, "covariance", None) is not None:
            c = result.covariance.correlation(); free = [p.name for p in self.parameters if not p.fixed]
            out["correlation"] = {a: {b: float(c[a,b]) for b in free} for a in free}
        return out

    def _projection_components_pair(self, values, plus_sample, minus_sample):
        from jaxpwa.workflow import _scaled_projection_weights

        total_events = self._total_data_events()
        standalone = False
        if self.extended:
            plus_yield, minus_yield, standalone = _signal_yield_pair(self.signal_yield, values)
            plus_yield, minus_yield = float(plus_yield), float(minus_yield)
        elif self.background_categories:
            plus_yield = minus_yield = total_events * float(_resolve(self.signal_fraction, values))
        else:
            plus_yield = minus_yield = float(total_events)
        plus_w, minus_w = np.zeros(plus_sample.size), np.zeros(minus_sample.size)
        if plus_yield or minus_yield:
            _, integral_plus = self.plus_cache.evaluate(values)
            _, integral_minus = self.minus_cache.evaluate(values)
            norm = float(integral_plus + integral_minus)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("signal projection requires positive finite joint integral")
            for sample, model, efficiency, veto, integral, signal_yield_value, target in (
                (plus_sample, self.plus_model, self.plus_efficiency, self.plus_veto, integral_plus, plus_yield, plus_w),
                (minus_sample, self.minus_model, self.minus_efficiency, self.minus_veto, integral_minus, minus_yield, minus_w),
            ):
                # A YieldAsymmetry's n_plus/n_minus are already literal per-charge
                # counts (see YieldAsymmetry docstring); a shared yield still
                # needs the amplitude-driven integral_q/norm split.
                scale = signal_yield_value if standalone else signal_yield_value * float(integral) / norm
                if scale:
                    density = _acceptance(efficiency, veto, sample.as_dict()) * model.intensity(sample.as_dict(), values)
                    target[:] = _scaled_projection_weights(sample, density, scale)
        plus_components = [("signal", plus_sample, plus_w)]
        minus_components = [("signal", minus_sample, minus_w)]
        if not self.background_categories:
            return plus_components, minus_components
        if self.extended:
            bg_scales = [float(_resolve(c.yield_, values)) for c in self.background_categories]
        else:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            bw = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales = [bg_total * float(w) for w in bw]
        for source, category, scale in zip(self.backgrounds, self.background_categories, bg_scales):
            if scale == 0:
                continue
            if not isinstance(source, CPBackgroundSpec):
                raise ValueError("plotting a precomputed CP background requires a CPBackgroundSpec with evaluable shapes")
            pr = jnp.asarray(source.plus_shape(plus_sample.as_dict())) if float(category.plus_probability) else jnp.zeros(plus_sample.size)
            mr = jnp.asarray(source.resolved_minus_shape(minus_sample.as_dict())) if float(category.minus_probability) else jnp.zeros(minus_sample.size)
            if source.apply_veto:
                if self.plus_veto is not None:
                    pr *= jnp.asarray(self.plus_veto(plus_sample.as_dict()))
                if self.minus_veto is not None:
                    mr *= jnp.asarray(self.minus_veto(minus_sample.as_dict()))
            pw = _scaled_projection_weights(plus_sample, pr, scale * float(category.plus_probability))
            mw = _scaled_projection_weights(minus_sample, mr, scale * float(category.minus_probability))
            plus_components.append((category.name, plus_sample, pw))
            minus_components.append((category.name, minus_sample, mw))
        return plus_components, minus_components

    def _projection_components(self, values, charge: str):
        """Compatibility wrapper using deterministic normalization samples."""
        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        plus, minus = self._projection_components_pair(
            values,
            self.plus_model.normalization_sample,
            self.minus_model.normalization_sample,
        )
        return plus if charge == "plus" else minus

    def plot_projection(
        self, result, variable="s13", *, bins=60, range=None,
        show_components=True, show_pulls=False, log_scale=False,
        projection_size=250_000, projection_seed=20260901, folded=False,
        partner_variable=None, fold_side="low", selection=None, axes=None,
        show_amplitude_components=False, legend_panel=False, extra_panel=False,
        share_y=False, show_interference=True,
    ):
        """Plot smooth B+/B- projections without histogramming quadrature nodes.

        Two weighted phase-space MC samples are used only for rendering. Their
        component weights are normalized jointly across charges, preserving the
        integrated charge asymmetry of the fitted model.

        ``folded=True`` projects onto ``s_low = min(variable, partner_variable)``
        (``fold_side="low"``, default) or ``s_high = max(...)``
        (``fold_side="high"``), event by event, *within each charge's own
        subplot* (B+ and B- are never mixed); see
        ``FitSession.plot_projection`` for the identical-daughter convention
        this exploits.

        ``selection(data)`` may return a boolean vector selecting a Dalitz
        region from the sample's invariant/momentum dictionary. The same cut
        is applied to each charge's data and rendering MC. Component weights
        are normalized on the full MC sample before selection: regional plots
        retain their predicted yields, rather than being rescaled to the
        selected data count. ``range`` only limits the projected x axis.

        ``show_pulls=True`` adds a ``(observed-expected)/sqrt(expected)`` panel
        below each charge's histogram, sharing that column's x axis. It builds
        its own 2x2 figure and therefore requires ``axes=None``; the return
        value is then the full 2x2 axes grid (row 0 the histograms, row 1 the
        pulls) instead of the usual length-2 list.

        ``show_amplitude_components=True`` also draws each amplitude
        component of the signal model separately: the incoherent
        contribution ``|c_k A_k|^2`` of component ``k`` (per rendering event,
        the signal weight times ``|c_k A_k|^2 / |sum_j c_j A_j|^2``), on the
        same scale as the signal. These do not add up to the signal; on a
        linear y axis the remainder is drawn as ``interference`` unless
        ``show_interference=False``.

        ``legend_panel=True`` (requires ``axes=None``) moves the legend out of
        the histograms into a separate column on the right of the figure,
        shared by both charges (the data entry is labelled ``data``).
        ``extra_panel=True`` additionally splits that column into the legend
        (top) and an empty panel (bottom) for the caller, e.g. a Dalitz-plot
        map of the projected region. They are available as
        ``figure.legend_axes`` and ``figure.extra_axes``
        (``axes[0].figure``); the return value is unchanged.

        ``share_y=True`` gives the B+ and B- histograms (and pull panels) one
        common y axis, with its label and tick labels only on the left panel.
        """
        return self._plot_projection(
            result, variable, bins=bins, range=range,
            show_components=show_components, show_pulls=show_pulls,
            log_scale=log_scale, projection_size=projection_size,
            projection_seed=projection_seed, folded=folded,
            partner_variable=partner_variable, fold_side=fold_side,
            selection=selection, axes=axes,
            show_amplitude_components=show_amplitude_components,
            legend_panel=legend_panel, extra_panel=extra_panel, share_y=share_y,
            show_interference=show_interference,
        )

    def prepare_projection_toy(
        self, result, *, projection_size=250_000, projection_seed=20260901,
        method="inverse-transform", **toy_options,
    ):
        """Generate reusable signal/background toys at the fitted parameters.

        ``projection_size`` is the total toy count per nonzero charge, allocated
        across signal/background categories. Their constant histogram weights
        preserve fitted yields and the joint CP charge split (or the literal
        per-charge yields of ``YieldAsymmetry``). Signal toys include efficiency
        and veto; backgrounds respect their own ``apply_veto`` setting.

        ``method`` accepts ``'inverse-transform'`` or ``'accept-reject'``.
        ``toy_options`` accepts the corresponding resolution/envelope settings
        from ``generate_toy`` and ``include_momenta`` (False by default).
        Returns ``CPProjectionToy`` with host arrays, reusable for any binning,
        folding or selection at these fitted values. It retains no fit session.
        """
        from jaxpwa.projection_toys import _prepare_cp_projection_toy

        return _prepare_cp_projection_toy(
            self, result, projection_size, projection_seed, method, toy_options,
        )

    def plot_projection_from_toy(
        self, result, variable="s13", *, bins=60, range=None,
        show_components=True, show_pulls=False, log_scale=False,
        projection_size=250_000, projection_seed=20260901, folded=False,
        partner_variable=None, fold_side="low", selection=None, axes=None,
        method="inverse-transform", projection_toy=None,
        include_toy_uncertainty=True, show_amplitude_components=False,
        legend_panel=False, extra_panel=False, share_y=False, show_interference=True,
        **toy_options,
    ):
        """Plot generated model toys with the same options as ``plot_projection``.

        Pass a ``prepare_projection_toy`` result as ``projection_toy`` to avoid
        regenerating events for each plot. Generation settings are used only
        when it is omitted; the supplied toy must match this session and result.
        ``show_components`` displays coherent signal and background categories,
        just as in ``plot_projection`` (not incoherent resonance toys).

        With ``include_toy_uncertainty=True``, show a one-sigma MC band and use
        ``(data-model)/sqrt(model + MC_variance)`` for pulls. The MC variance is
        the sum of squared histogram weights (Poisson approximation); it does
        not include fit-parameter uncertainty. False gives the original pull
        convention. Regional selections never renormalize the toy yields.
        The axes/2x2 pull-grid return convention is unchanged.
        ``show_amplitude_components`` is as in ``plot_projection``: the toy
        signal events are reweighted by ``|c_k A_k|^2 / |sum_j c_j A_j|^2``.
        ``legend_panel``/``extra_panel``/``share_y``/``show_interference`` are as
        in ``plot_projection``.
        """
        return self._plot_projection(
            result, variable, bins=bins, range=range,
            show_components=show_components, show_pulls=show_pulls,
            log_scale=log_scale, projection_size=projection_size,
            projection_seed=projection_seed, folded=folded,
            partner_variable=partner_variable, fold_side=fold_side,
            selection=selection, axes=axes, use_toy=True,
            projection_toy=projection_toy, method=method,
            include_toy_uncertainty=include_toy_uncertainty,
            toy_options=toy_options,
            show_amplitude_components=show_amplitude_components,
            legend_panel=legend_panel, extra_panel=extra_panel, share_y=share_y,
            show_interference=show_interference,
        )

    def _plot_projection(
        self, result, variable="s13", *, bins=60, range=None,
        show_components=True, show_pulls=False, log_scale=False,
        projection_size=250_000, projection_seed=20260901, folded=False,
        partner_variable=None, fold_side="low", selection=None, axes=None,
        use_toy=False, projection_toy=None, method="inverse-transform",
        include_toy_uncertainty=False, toy_options=None,
        show_amplitude_components=False, legend_panel=False, extra_panel=False,
        share_y=False, show_interference=True,
    ):
        import matplotlib.pyplot as plt
        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        if show_pulls and axes is not None:
            raise ValueError(
                "show_pulls=True builds its own figure layout; pass axes=None"
            )
        if (legend_panel or extra_panel) and axes is not None:
            raise ValueError(
                "legend_panel/extra_panel build their own figure layout; pass axes=None"
            )
        if extra_panel:
            legend_panel = True
        fold_fn = np.minimum if fold_side == "low" else np.maximum
        selection_masks = {}

        def _selection_mask(sample):
            key = id(sample)
            if key not in selection_masks:
                mask = (
                    np.ones(sample.size, dtype=bool) if selection is None
                    else np.asarray(selection(sample.as_dict()))
                )
                if mask.shape != (sample.size,) or mask.dtype != np.bool_:
                    raise ValueError(
                        "projection selection must return a boolean vector "
                        f"of shape ({sample.size},)"
                    )
                selection_masks[key] = mask
            return selection_masks[key]

        def _folded_values(sample):
            values_ = np.asarray(getattr(sample, variable))
            if not folded:
                return values_
            partner_values = np.asarray(getattr(sample, partner_variable))
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        combined = np.concatenate([
            _folded_values(d)[_selection_mask(d)]
            for d in (self.plus_data, self.minus_data)
        ])
        if range is None and combined.size == 0:
            raise ValueError("provide range when both charge datasets are empty")
        hist_range = range if range is not None else (float(np.min(combined)), float(np.max(combined)))
        edges = np.histogram_bin_edges(combined, bins=bins, range=hist_range)
        if use_toy:
            if projection_toy is None:
                projection_toy = self.prepare_projection_toy(
                    result, projection_size=projection_size,
                    projection_seed=projection_seed, method=method,
                    **(toy_options or {}),
                )
            elif toy_options:
                # Generation options cannot apply to an existing toy; refuse
                # them rather than silently dropping a mistyped keyword.
                raise TypeError(
                    f"toy options {sorted(toy_options)} are ignored with projection_toy; "
                    "pass them to prepare_projection_toy instead"
                )
            projection_toy._check(self, values)
            plus_components = projection_toy.plus_components
            minus_components = projection_toy.minus_components
        else:
            plus_sample = self.plus_model.generate_phase_space(
                projection_size, seed=projection_seed,
            )
            minus_sample = self.minus_model.generate_phase_space(
                projection_size, seed=projection_seed + 1,
            )
            plus_components, minus_components = self._projection_components_pair(
                values, plus_sample, minus_sample,
            )
        grid = None
        pulls_axes = (None, None)
        if axes is None:
            # Two side-by-side charge panels (and, with pulls, a second row
            # below each) at the active style's own base size
            # (`plt.style.use(...)`, e.g. mplhep), scaled by panel count
            # rather than a hardcoded absolute figsize.
            base_w, base_h = plt.rcParams["figure.figsize"]
            if not legend_panel:
                if show_pulls:
                    _, grid = plt.subplots(
                        2, 2, figsize=(base_w * 2, base_h * 1.2), sharex="col",
                        gridspec_kw={"height_ratios": (3, 1)},
                        constrained_layout=True,
                    )
                    axes, pulls_axes = grid[0], grid[1]
                else:
                    _, axes = plt.subplots(
                        1, 2, figsize=(base_w * 2, base_h), constrained_layout=True
                    )
            else:
                # Same charge panels plus a right-hand column for the legend
                # (and optionally an extra panel below it).
                side_ratio = 0.6
                rows = 2 if show_pulls else 1
                figure = plt.figure(
                    figsize=(base_w * (2 + side_ratio), base_h * (1.2 if show_pulls else 1.0)),
                    constrained_layout=True,
                )
                spec = figure.add_gridspec(
                    rows, 3, width_ratios=(1, 1, side_ratio),
                    height_ratios=(3, 1) if show_pulls else None,
                )
                top = [figure.add_subplot(spec[0, col]) for col in (0, 1)]
                if show_pulls:
                    bottom = [figure.add_subplot(spec[1, col], sharex=top[col]) for col in (0, 1)]
                    grid = np.empty((2, 2), dtype=object)
                    grid[0, :], grid[1, :] = top, bottom
                    axes, pulls_axes = grid[0], grid[1]
                else:
                    axes = np.asarray(top, dtype=object)
                if extra_panel:
                    side = spec[:, 2].subgridspec(2, 1, height_ratios=(1, 1.2))
                    figure.legend_axes = figure.add_subplot(side[0])
                    figure.extra_axes = figure.add_subplot(side[1])
                else:
                    figure.legend_axes = figure.add_subplot(spec[:, 2])
                    figure.extra_axes = None
                figure.legend_axes.axis("off")
        label = (
            rf"$s_{{\mathrm{{{fold_side}}}}}$" if folded else rf"${variable}$"
        )
        log_bottoms = []
        if share_y:
            axes[1].sharey(axes[0])
            if pulls_axes[0] is not None:
                pulls_axes[1].sharey(pulls_axes[0])
        for ax, ax_pulls, charge, data, components in zip(
            axes, pulls_axes, ("plus", "minus"),
            (self.plus_data, self.minus_data), (plus_components, minus_components),
        ):
            dv = _folded_values(data)[_selection_mask(data)]
            data_weights = (
                None if self.event_weights is None
                else self._data_weights(charge)[_selection_mask(data)]
            )
            unit = r"GeV$^2$" if variable in ("s12","s13","s23") else ""
            _, observed, observed_error, _ = plot_binned_data(
                dv, bins=edges, ax=ax, weights=data_weights,
                label=f"B{'+' if charge=='plus' else '-'} data"
                + ("" if data_weights is None else " (weighted)"),
                unit=unit, log_scale=log_scale,
            )
            total = np.zeros(len(edges) - 1)
            mc_variance = np.zeros_like(total)
            model = self.plus_model if charge == "plus" else self.minus_model
            for name, sample, weights in components:
                mask = _selection_mask(sample)
                cv = _folded_values(sample)[mask]
                counts, _ = np.histogram(
                    cv, bins=edges, weights=np.asarray(weights)[mask]
                )
                if show_amplitude_components and name == "signal":
                    integration_weights = None
                    if not use_toy:
                        efficiency = (self.plus_efficiency if charge == "plus"
                                      else self.minus_efficiency)
                        veto = self.plus_veto if charge == "plus" else self.minus_veto
                        integration_weights = np.asarray(sample.weights) * np.asarray(
                            _acceptance(efficiency, veto, sample.as_dict())
                        )
                    self._draw_amplitude_components(
                        ax, model, values, sample, np.asarray(weights), mask, cv, edges, counts, log_scale,
                        show_interference, integration_weights=integration_weights,
                    )
                total += counts
                if use_toy and include_toy_uncertainty:
                    mc_variance += np.histogram(
                        cv, bins=edges, weights=np.asarray(weights)[mask]**2,
                    )[0]
                if show_components:
                    ax.stairs(counts, edges, label=name)
            ax.stairs(total, edges, label="total fit", linewidth=2.0)
            if use_toy and include_toy_uncertainty:
                error = np.sqrt(mc_variance)
                if log_scale:
                    # Non-positive lower edges (e.g. bins emptied by a veto)
                    # are left out instead of clipped to a tiny positive
                    # number, which would stretch the log axis to ~1e-308.
                    lower = np.where(total - error > 0, total - error, np.nan)
                else:
                    lower = np.maximum(total - error, 0.0)
                ax.fill_between(
                    edges, np.r_[lower, lower[-1]],
                    np.r_[total + error, (total + error)[-1]],
                    step="post", alpha=0.2, color="grey", label="toy MC uncertainty",
                )
            if log_scale:
                # Bottom from the smallest positive model/data bin, so empty
                # (vetoed) bins and negative weighted bins do not set the scale.
                positive = np.concatenate([total[total > 0], np.asarray(observed)[np.asarray(observed) > 0]])
                if positive.size:
                    ax.set_ylim(bottom=0.5 * float(np.min(positive)))
                    log_bottoms.append(0.5 * float(np.min(positive)))
            axis_label = label + (" [GeV$^2$]" if unit else "")
            if legend_panel:
                pass  # one shared legend in figure.legend_axes, drawn below
            elif show_amplitude_components:
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.legend()
            if ax_pulls is None:
                ax.set_xlabel(axis_label)
                continue
            # Unweighted data: Poisson variance from the expectation. Weighted
            # (sWeight/COW) data: the observed sum w^2 per bin instead.
            data_variance = total if data_weights is None else np.square(observed_error)
            occupied = (total > 0) & (data_variance + mc_variance > 0)
            pulls = np.full(total.shape, np.nan)
            pulls[occupied] = (
                (observed[occupied] - total[occupied])
                / np.sqrt(data_variance[occupied] + mc_variance[occupied])
            )
            _draw_pulls_1d(ax_pulls, edges, pulls)
            if data_weights is not None:
                ax_pulls.set_ylabel(r"pull $(o-e)/\sqrt{\sum w^2+\sigma^2_{MC}}$")
            elif use_toy and include_toy_uncertainty:
                ax_pulls.set_ylabel(r"pull $(o-e)/\sqrt{e+\sigma^2_{MC}}$")
            ax_pulls.set_xlabel(axis_label)
        if share_y:
            # One y axis for both charges: common limits, label on the left only.
            if log_bottoms:
                axes[0].set_ylim(bottom=min(log_bottoms))
            for right in (axes[1], pulls_axes[1]):
                if right is not None:
                    right.set_ylabel("")
                    right.tick_params(labelleft=False)
        if legend_panel and axes is not None:
            handles, labels = axes[0].get_legend_handles_labels()
            labels = [text.replace("B+ data", "data", 1) for text in labels]
            axes[0].figure.legend_axes.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)
        return grid if show_pulls else axes

    @staticmethod
    def _draw_amplitude_components(ax, model, values, sample, weights, mask, folded_values, edges, signal_counts, log_scale,
                                   show_interference=True, *, integration_weights=None):
        """Draw |c_k A_k|^2 of every signal component (and, linear y, the interference)."""
        data = sample.as_dict()
        amplitudes = {name: np.asarray(value) for name, value in model.component_amplitudes(data, values).items()}
        coherent = np.abs(sum(amplitudes.values())) ** 2
        if integration_weights is None:
            scale = np.divide(weights, coherent, out=np.zeros_like(weights, dtype=float), where=coherent > 0)
        else:
            # Phase-space projections know their integration measure. Keep
            # individual contributions even where destructive interference
            # makes the coherent signal exactly zero (0/0 loses them).
            integral = float(np.sum(integration_weights * coherent))
            scale = (np.sum(weights) / integral * integration_weights
                     if integral > 0 else np.zeros_like(weights, dtype=float))
        incoherent_sum = np.zeros(len(edges) - 1)
        for name, amplitude in amplitudes.items():
            component_weights = (scale * np.abs(amplitude) ** 2)[mask]
            counts, _ = np.histogram(folded_values, bins=edges, weights=component_weights)
            incoherent_sum += counts
            ax.stairs(counts, edges, label=name, linewidth=1.0)
        if show_interference and not log_scale:
            ax.stairs(signal_counts - incoherent_sum, edges, label="interference",
                      linewidth=1.0, linestyle="-.", color="magenta")

    def _projection_signal_density(self, sample, values, charge):
        """Normalized per-charge signal density at arbitrary points.

        Unlike ``FitSession``, ``CPFitSession`` has no compact-cache-optimized
        projection path; this mirrors what ``_projection_components_pair``
        already does to render MC projections (``model.intensity`` evaluated
        directly), just normalized by the charge's own fitted integral so the
        result integrates to one over that charge's Dalitz plane alone.
        """

        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        model = self.plus_model if charge == "plus" else self.minus_model
        cache = self.plus_cache if charge == "plus" else self.minus_cache
        efficiency = self.plus_efficiency if charge == "plus" else self.minus_efficiency
        veto = self.plus_veto if charge == "plus" else self.minus_veto
        _, integral = cache.evaluate(values)
        acceptance = _acceptance(efficiency, veto, sample.as_dict())
        return acceptance * model.intensity(sample.as_dict(), values) / integral

    def _total_density(self, sample, values, charge):
        """Fraction-weighted, unit-integral total fitted density f0(x|charge).

        Mirrors ``_projection_components_pair``'s extended/signal_fraction/
        plain branch structure and its amplitude-driven charge split
        (``integral_q/(integral_plus+integral_minus)``), but returns
        *fractions* (summing to one, conditioned on this one charge) rather
        than absolute event-count scales -- what the point-to-point
        dissimilarity test's ``f0`` requires. See
        ``FitSession._total_density`` for why this cannot reuse
        ``_scaled_projection_weights`` directly.
        """

        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        total_events = self.plus_data.size + self.minus_data.size
        standalone = False
        if self.extended:
            plus_yield, minus_yield, standalone = _signal_yield_pair(self.signal_yield, values)
            signal_yield_nominal = float(plus_yield if charge == "plus" else minus_yield)
        elif self.background_categories:
            signal_yield_nominal = total_events * float(_resolve(self.signal_fraction, values))
        else:
            signal_yield_nominal = float(total_events)

        signal_scale = 0.0
        if signal_yield_nominal:
            if standalone:
                signal_scale = signal_yield_nominal
            else:
                _, integral_plus = self.plus_cache.evaluate(values)
                _, integral_minus = self.minus_cache.evaluate(values)
                norm = float(integral_plus + integral_minus)
                integral = integral_plus if charge == "plus" else integral_minus
                signal_scale = signal_yield_nominal * float(integral) / norm

        if self.extended:
            bg_scales_total = [
                float(_resolve(category.yield_, values))
                for category in self.background_categories
            ]
        elif self.background_categories:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            weights = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales_total = [bg_total * float(weight) for weight in weights]
        else:
            bg_scales_total = []

        probability_attr = "plus_probability" if charge == "plus" else "minus_probability"
        bg_scales = [
            scale * float(getattr(category, probability_attr))
            for scale, category in zip(bg_scales_total, self.background_categories)
        ]

        total_scale = signal_scale + sum(bg_scales)
        if total_scale <= 0:
            raise ValueError("total expected yield for this charge must be positive")

        density = (signal_scale / total_scale) * self._projection_signal_density(
            sample, values, charge
        )
        veto = self.plus_veto if charge == "plus" else self.minus_veto
        normalization_attr = (
            "plus_normalization" if charge == "plus" else "minus_normalization"
        )
        for source, category, scale in zip(
            self.backgrounds, self.background_categories, bg_scales
        ):
            if scale == 0:
                continue
            if not isinstance(source, CPBackgroundSpec):
                raise ValueError(
                    "goodness-of-fit density requires a CPBackgroundSpec with "
                    "evaluable shapes"
                )
            shape = source.plus_shape if charge == "plus" else source.resolved_minus_shape
            raw = jnp.asarray(shape(sample.as_dict()))
            if source.apply_veto and veto is not None:
                raw = raw * jnp.asarray(veto(sample.as_dict()))
            normalization = getattr(category, normalization_attr)
            density = density + (scale / total_scale) * (raw / normalization)
        return density

    def _default_free_parameters(self) -> int:
        return sum(1 for parameter in self.parameters if not parameter.fixed)

    def goodness_of_fit_projection(
        self,
        result,
        variable="s13",
        *,
        charge=None,
        bins=60,
        range=None,
        folded=False,
        partner_variable=None,
        fold_side="low",
        projection_size=250_000,
        projection_seed=20260901,
        n_free_parameters=None,
    ):
        """Binned Pearson chi2 goodness-of-fit test on a 1D projection.

        Returns a ``{"plus": ..., "minus": ...}`` dict of
        :class:`~jaxpwa.goodness_of_fit.BinnedChi2Result` by
        default, or a single result when ``charge`` is given. Uses the same
        reweighted-MC-projection histogram as ``plot_projection`` for the
        expected counts. For an event-weighted (sWeight/COW) session the data
        histogram is the per-bin sum of weights and each bin's variance is its
        sum of squared weights (``chi2_from_histograms(observed_variance=)``).
        """
        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        if charge is not None and charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        fold_fn = np.minimum if fold_side == "low" else np.maximum

        def _folded_values(sample):
            values_ = np.asarray(getattr(sample, variable))
            if not folded:
                return values_
            partner_values = np.asarray(getattr(sample, partner_variable))
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        combined = np.concatenate(
            [_folded_values(d) for d in (self.plus_data, self.minus_data)]
        )
        if range is None and combined.size == 0:
            raise ValueError("provide range when both charge datasets are empty")
        hist_range = range if range is not None else (
            float(np.min(combined)), float(np.max(combined))
        )
        edges = np.histogram_bin_edges(combined, bins=bins, range=hist_range)

        plus_sample = self.plus_model.generate_phase_space(projection_size, seed=projection_seed)
        minus_sample = self.minus_model.generate_phase_space(
            projection_size, seed=projection_seed + 1
        )
        plus_components, minus_components = self._projection_components_pair(
            values, plus_sample, minus_sample
        )

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()

        results = {}
        for name, data, components in (
            ("plus", self.plus_data, plus_components),
            ("minus", self.minus_data, minus_components),
        ):
            data_weights = None if self.event_weights is None else self._data_weights(name)
            observed, _ = np.histogram(_folded_values(data), bins=edges, weights=data_weights)
            variance = (
                None if data_weights is None
                else np.histogram(_folded_values(data), bins=edges, weights=data_weights**2)[0]
            )
            expected = np.zeros(len(edges) - 1, dtype=float)
            for _, component_sample, weights in components:
                component_values = _folded_values(component_sample)
                counts, _ = np.histogram(
                    component_values, bins=edges, weights=np.asarray(weights)
                )
                expected += counts
            results[name] = chi2_from_histograms(
                observed, expected, n_free_parameters=n_free_parameters, edges=(edges,),
                observed_variance=variance,
            )
        return results[charge] if charge is not None else results

    def goodness_of_fit_chi2(
        self,
        result,
        x="s13",
        y="s23",
        *,
        charge=None,
        bins=25,
        range=None,
        folded=False,
        square_dalitz=False,
        mother_mass=None,
        masses=None,
        pair=(0, 1),
        projection_size=250_000,
        projection_seed=20260901,
        n_free_parameters=None,
    ):
        """Binned Pearson chi2 goodness-of-fit test on the Dalitz plane.

        Returns a ``{"plus": ..., "minus": ...}`` dict of
        :class:`~jaxpwa.goodness_of_fit.BinnedChi2Result` by
        default, or a single result when ``charge`` is given. See
        ``FitSession.goodness_of_fit_chi2`` for the ``square_dalitz``/
        ``folded`` conventions. Event-weighted sessions use the per-bin sum of
        weights with a sum-of-squared-weights variance, as in
        ``goodness_of_fit_projection``.
        """
        if square_dalitz and (mother_mass is None or masses is None):
            raise ValueError("square_dalitz=True requires mother_mass and masses")
        if charge is not None and charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")

        def _coordinates(sample):
            if square_dalitz:
                data = sample.as_dict()
                mp, tp = invariants_to_square_dalitz(
                    data["s12"], data["s13"], data["s23"],
                    mother_mass=mother_mass,
                    masses=masses,
                    pair=pair,
                )
                if folded:
                    tp = fold_thetaprime(tp)
                return np.asarray(mp), np.asarray(tp)
            x_values = np.asarray(getattr(sample, x))
            y_values = np.asarray(getattr(sample, y))
            if folded:
                x_values, y_values = (
                    np.minimum(x_values, y_values),
                    np.maximum(x_values, y_values),
                )
            return x_values, y_values

        if range is not None:
            hist_range = range
        elif square_dalitz:
            hist_range = ((0.0, 1.0), (0.0, 0.5) if folded else (0.0, 1.0))
        else:
            hist_range = None

        values = self.result_values(result)
        if square_dalitz:
            # A regular Square-Dalitz quadrature removes the artificial MC
            # noise from the residual map and uses the same coordinate-area
            # measure as Laura++ (the sample weights contain |J|).
            resolution = max(2, int(np.sqrt(projection_size)))
            plus_sample = SquareDalitzGrid(
                mother_mass=mother_mass, masses=tuple(masses),
                pair=pair, resolution=resolution,
            ).sample()
            minus_sample = plus_sample
        else:
            plus_sample = self.plus_model.generate_phase_space(
                projection_size, seed=projection_seed
            )
            minus_sample = self.minus_model.generate_phase_space(
                projection_size, seed=projection_seed + 1
            )
        plus_components, minus_components = self._projection_components_pair(
            values, plus_sample, minus_sample
        )

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()

        results = {}
        for name, data, components in (
            ("plus", self.plus_data, plus_components),
            ("minus", self.minus_data, minus_components),
        ):
            data_x, data_y = _coordinates(data)
            data_weights = None if self.event_weights is None else self._data_weights(name)
            observed, x_edges, y_edges = np.histogram2d(
                data_x, data_y, bins=bins, range=hist_range, weights=data_weights
            )
            variance = (
                None if data_weights is None
                else np.histogram2d(data_x, data_y, bins=[x_edges, y_edges], weights=data_weights**2)[0]
            )
            expected = np.zeros_like(observed)
            for _, component_sample, weights in components:
                component_x, component_y = _coordinates(component_sample)
                counts, _, _ = np.histogram2d(
                    component_x,
                    component_y,
                    bins=[x_edges, y_edges],
                    weights=np.asarray(weights),
                )
                expected += counts
            results[name] = chi2_from_histograms(
                observed,
                expected,
                n_free_parameters=n_free_parameters,
                edges=(x_edges, y_edges),
                observed_variance=variance,
            )
        return results[charge] if charge is not None else results

    def point_to_point_dissimilarity(
        self,
        result,
        *,
        charge,
        x="s13",
        y="s23",
        sigma_bar=0.01,
        mc_size=None,
        n_permutations=200,
        seed=20260901,
        max_total_events=5_000,
        candidate_pool_size=200_000,
    ):
        """Unbinned point-to-point dissimilarity test for one charge.

        Run once per charge (``charge="plus"``/``"minus"``): the underlying
        statistic (Williams, arXiv:1006.3019) compares one data sample
        against one reference sample from one density, so a "joint" PPD
        across both charges is not a documented statistic and is not
        offered here. See ``FitSession.point_to_point_dissimilarity``.
        """

        self._require_unweighted("point_to_point_dissimilarity")
        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        values = self.result_values(result)
        data = self.plus_data if charge == "plus" else self.minus_data
        model = self.plus_model if charge == "plus" else self.minus_model
        n_data = data.size
        if mc_size is None:
            mc_size = min(10 * n_data, max(max_total_events - n_data, 1))

        candidate = model.generate_phase_space(candidate_pool_size, seed=seed)
        candidate_density = self._total_density(candidate, values, charge)
        target_weights = candidate.weights * candidate_density
        replica = weighted_resample(
            jax.random.PRNGKey(seed), candidate, target_weights, mc_size
        )

        data_density = self._total_density(data, values, charge)
        replica_density = self._total_density(replica, values, charge)
        phase_space_area = float(jnp.mean(model.normalization_sample.weights))

        data_xy = np.column_stack(
            [np.asarray(getattr(data, x)), np.asarray(getattr(data, y))]
        )
        replica_xy = np.column_stack(
            [np.asarray(getattr(replica, x)), np.asarray(getattr(replica, y))]
        )

        return _point_to_point_dissimilarity(
            data_xy,
            replica_xy,
            np.asarray(data_density),
            np.asarray(replica_density),
            sigma_bar=sigma_bar,
            phase_space_area=phase_space_area,
            n_permutations=n_permutations,
            seed=seed + 1,
            max_total_events=max_total_events,
        )


__all__ = ["CPBackgroundSpec", "CPFitSession"]
