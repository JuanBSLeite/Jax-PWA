"""User-facing high-level workflow helpers.

The low-level PDF, likelihood, cache and minimizer classes remain available.
This module only composes them for common analysis workflows with less boilerplate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa.amplitude import PreparedAmplitudeCache
from jaxpwa.background import BackgroundCategory
from jaxpwa.constraints import ConstrainedNLL
from jaxpwa.efficiency import UnityEfficiency
from jaxpwa.fit import Minimizer, Parameter
from jaxpwa.fit.nesterov import NesterovResult
from jaxpwa.goodness_of_fit import (
    BinnedChi2Result,
    PointToPointResult,
    chi2_from_histograms,
)
from jaxpwa.goodness_of_fit import (
    point_to_point_dissimilarity as _point_to_point_dissimilarity,
)
from jaxpwa.integration import GridIntegrator
from jaxpwa.io import model_with_fitted_values, read_phase_space_sample
from jaxpwa.kinematics import (
    EventSample,
    PhaseSpaceSample,
    fold_thetaprime,
    invariants_to_square_dalitz,
)
from jaxpwa.likelihood import MultiBackgroundNLL, UnbinnedNLL, WeightedUnbinnedNLL
from jaxpwa.likelihood.square_dalitz import (
    SquareDalitzNLL,
    square_dalitz_log_jacobian,
    validate_session_coordinates,
)
from jaxpwa.likelihood.weighted import (
    sandwich_covariance_from_score_outer,
    sweight_covariance_from_hessians,
)
from jaxpwa.pdf import SignalPDF
from jaxpwa.plotting import _draw_pulls_1d, _values, plot_binned_data
from jaxpwa.sampling import weighted_resample


@dataclass(frozen=True)
class BackgroundSpec:
    """Background shape that is normalized automatically on the fit measure."""

    name: str
    shape: object
    fraction: object | None = None
    yield_: object | None = None
    normalization_sample: PhaseSpaceSample | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BackgroundSpec name must be non-empty")
        if not callable(self.shape):
            raise TypeError("BackgroundSpec shape must be callable on an event-data mapping")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background cannot define both fraction and yield")


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


def _resolve(value: object, parameters: Mapping[str, object]):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


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


def _scaled_projection_weights(
    sample: PhaseSpaceSample,
    density,
    scale: float,
) -> np.ndarray:
    """Normalize MC projection weights to the requested component yield."""

    if not np.isfinite(scale) or scale < 0:
        raise ValueError("projection yield must be finite and non-negative")
    if scale == 0:
        return np.zeros(sample.size)
    raw = np.asarray(sample.weights, dtype=float) * np.asarray(density, dtype=float)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("projection density has non-positive or non-finite integral")
    return float(scale) * raw / total


def _install_minuit_covariance(result, names: Sequence[str], covariance) -> None:
    """Replace Minuit's reported covariance/errors with an external estimate."""

    from iminuit.util import Matrix

    names = tuple(names)
    if tuple(result.parameters) != names:
        raise RuntimeError(
            "corrected covariance parameter order does not match Minuit result"
        )
    values = np.asarray(covariance, dtype=float)
    if values.shape != (len(names), len(names)):
        raise ValueError("corrected covariance has incompatible shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("corrected covariance must be finite")
    values = 0.5 * (values + values.T)
    diagonal = np.diag(values)
    scale = max(1.0, float(np.max(np.abs(diagonal), initial=0.0)))
    if np.any(diagonal < -1e-10 * scale):
        raise ValueError("corrected covariance has a negative diagonal element")
    eigenvalues = np.linalg.eigvalsh(values)
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if np.any(eigenvalues < -1e-10 * eigenvalue_scale):
        raise ValueError("corrected covariance must be positive semidefinite")

    matrix = Matrix(names)
    matrix[:] = values
    # Minuit exposes covariance read-only, so update its stored Matrix and the
    # writable ErrorView together. Downstream Jax-PWA APIs then consume the
    # corrected covariance transparently.
    result._covariance = matrix
    result.errors = np.sqrt(np.clip(diagonal, 0.0, None))


_COVARIANCE_MODES = ("minuit", "sweight", "sumw2", "sandwich")


def _square_weights(weights):
    """Element-wise square of one weight array or of a (plus, minus) pair."""
    if isinstance(weights, tuple):
        return tuple(jnp.square(jnp.asarray(w)) for w in weights)
    return jnp.square(jnp.asarray(weights))


def _fit_with_optional_weights(
    *,
    unweighted_minimizer,
    weighted_objective,
    event_weighted_objective,
    score_outer_objective,
    parameters,
    weights,
    covariance: str,
    start_values,
    simplex: bool,
    ncall,
    strategy: int,
    hesse: bool,
    tolerance: float,
    verbose: int,
    hessian: str,
    method: str,
    nesterov_max_iter: int,
    nesterov_gtol: float,
):
    """Shared fit driver of ``FitSession.fit`` and ``CPFitSession.fit``.

    ``unweighted_minimizer()`` builds the ordinary minimizer used when
    ``weights is None``. ``weighted_objective(weights)`` returns the
    signal-only objective ``-sum_i w_i log p_i`` plus any constraints,
    ``event_weighted_objective(weights)`` returns the event term alone, and
    ``score_outer_objective(weights, fitted)`` a scalar whose Hessian at
    ``fitted`` is ``sum_i w_i^2 s_i s_i^T``. With
    ``covariance="sandwich"|"sumw2"|"sweight"`` the result's covariance and
    errors are replaced by the corrected weighted covariance.
    """
    if covariance not in _COVARIANCE_MODES:
        raise ValueError(
            "covariance must be 'minuit', 'sweight', 'sumw2', or 'sandwich'"
        )
    corrected_weight_covariance = covariance in ("sweight", "sumw2", "sandwich")
    if corrected_weight_covariance and weights is None:
        raise ValueError(f"covariance={covariance!r} requires event weights")
    if corrected_weight_covariance and method == "nesterov":
        raise ValueError(
            f"covariance={covariance!r} requires a Minuit-refined fit result"
        )

    if weights is None:
        minimizer = unweighted_minimizer()
    else:
        # A signed-weight likelihood is not guaranteed to have positive
        # curvature away from the minimum. Supplying its exact Hessian to
        # MIGRAD can therefore make Minuit's seed/error matrix indefinite
        # even when the objective and gradient are perfectly finite.
        #
        # In corrected weighted covariance modes, hessian="jax" keeps the
        # ordinary JAX gradient during minimization but not the Hessian.
        # Minuit estimates its search curvature numerically; the exact
        # memory-aware JAX matrices are evaluated only at the fitted point.
        # The low-level Minimizer and ordinary covariance="minuit" fits
        # retain the historical direct-to-MIGRAD JAX Hessian behavior.
        minimization_hessian = (
            "numerical"
            if corrected_weight_covariance and hessian == "jax"
            else hessian
        )
        minimizer = Minimizer(
            weighted_objective(weights),
            parameters,
            tolerance=tolerance,
            verbose=verbose,
            hessian=minimization_hessian,
        )

    result = minimizer.fit(
        start_values=start_values,
        simplex=simplex,
        ncall=ncall,
        strategy=strategy,
        hesse=hesse,
        method=method,
        nesterov_max_iter=nesterov_max_iter,
        nesterov_gtol=nesterov_gtol,
    )

    if not corrected_weight_covariance:
        return result
    if isinstance(result, NesterovResult):
        warnings.warn(
            "The Minuit continuation of the Nesterov prefit was rejected "
            "(non-finite or higher NLL), so the Nesterov endpoint is returned; "
            "skipping corrected covariance. It has no EDM or covariance check; "
            "do not report uncertainties from it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return result
    if not bool(result.valid) or not np.isfinite(float(result.fval)):
        warnings.warn(
            "Weighted fit did not converge; skipping corrected covariance. "
            "Inspect the objective and gradients before reporting uncertainties. "
            "The returned result retains its invalid fit status and any "
            "uncorrected optimizer covariance.",
            RuntimeWarning,
            stacklevel=3,
        )
        return result

    fitted = {name: float(result.values[name]) for name in result.parameters}
    names, weighted_hessian = minimizer.jax_hessian(fitted)
    if covariance in ("sweight", "sumw2"):
        # Fixed constraints contribute to the sensitivity matrix above, but
        # are not event fluctuations and do not belong in H_w2.
        second_objective = event_weighted_objective(_square_weights(weights))
        label = "weighted and squared-weight Hessians"
    else:
        second_objective = score_outer_objective(weights, fitted)
        label = "weighted Hessian and score-outer matrix"
    second_names, variability = Minimizer(
        second_objective,
        parameters,
        tolerance=tolerance,
        verbose=verbose,
        hessian="jax",
    ).jax_hessian(fitted)
    if second_names != names:
        raise RuntimeError(f"{label} use different parameters")
    if covariance in ("sweight", "sumw2"):
        corrected = sweight_covariance_from_hessians(weighted_hessian, variability)
    else:
        corrected = sandwich_covariance_from_score_outer(weighted_hessian, variability)
    _install_minuit_covariance(result, names, corrected)
    return result


@dataclass(frozen=True)
class FitSession:
    """Compose a common single-sample amplitude fit in a few lines.

    The likelihood path uses :class:`PreparedAmplitudeCache`: fixed component
    dynamics and the fixed normalization matrix are prepared once, while only
    components owning floating dynamical parameters are reevaluated during the
    fit. Efficiency and veto values are likewise cached on the data and
    normalization samples.

    ``coordinates="square-dalitz"`` fits the events as points of the
    Square-Dalitz plane ``(m', theta')`` of the ordered daughter pair
    ``square_dalitz_pair`` instead of ``(s_a, s_b)``: every event density is
    per ``dm' dtheta'``, i.e. the NLL becomes :class:`~jaxpwa.SquareDalitzNLL`
    of the Dalitz-plot one. The normalization integral is the same in any
    coordinates, so any normalization method works; ``square_dalitz_pair``
    defaults to the ``normalization_pair`` of a ``square-dalitz``-normalized
    model and is required otherwise. The Jacobian is parameter-independent,
    so fitted values and errors are those of the Dalitz-plot fit; only the NLL
    value changes. Efficiencies stay
    dimensionless and background shapes stay Dalitz-plot densities
    (Square-Dalitz histogram backgrounds with ``divide_jacobian=True``),
    which is checked on construction.
    """

    model: object
    data: EventSample
    efficiency: object | None = None
    veto: object | None = None
    backgrounds: tuple[BackgroundSpec | BackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    constraints: tuple[object, ...] = ()
    coordinates: str = "dalitz"
    square_dalitz_pair: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        validate_session_coordinates(
            self.coordinates,
            pair=self.square_dalitz_pair,
            models=(self.model,),
            efficiencies=(self.efficiency,),
            background_shapes=tuple(
                getattr(background, "shape", None) for background in self.backgrounds
            ),
        )

    @classmethod
    def from_root(
        cls,
        model: object,
        file_path: str | Path,
        tree: str,
        *,
        efficiency: object | None = None,
        veto: object | None = None,
        backgrounds: Sequence[BackgroundSpec | BackgroundCategory] = (),
        signal_fraction: object | None = None,
        extended: bool = False,
        signal_yield: object | None = None,
        constraints: Sequence[object] = (),
        coordinates: str = "dalitz",
        square_dalitz_pair: tuple[int, int] | None = None,
        **root_kwargs,
    ) -> "FitSession":
        """Build a `FitSession` reading its data sample from a ROOT tree.

        ``root_kwargs`` go to :func:`~jaxpwa.read_phase_space_sample` (e.g.
        ``mprime``/``thetaprime`` branches for Square-Dalitz input).
        """
        data = read_phase_space_sample(file_path, tree, **root_kwargs)
        return cls(
            model=model,
            data=data,
            efficiency=efficiency,
            veto=veto,
            backgrounds=tuple(backgrounds),
            signal_fraction=signal_fraction,
            extended=extended,
            signal_yield=signal_yield,
            constraints=tuple(constraints),
            coordinates=coordinates,
            square_dalitz_pair=square_dalitz_pair,
        )

    def with_efficiency(self, efficiency: object | None) -> "FitSession":
        """Return a copy of this session with `efficiency` replaced."""
        return replace(self, efficiency=efficiency)

    def with_veto(self, veto: object | None) -> "FitSession":
        """Return a copy of this session with `veto` replaced."""
        return replace(self, veto=veto)

    def with_background(
        self,
        name: str,
        shape: object,
        *,
        fraction: object | None = None,
        yield_: object | None = None,
        normalization_sample: PhaseSpaceSample | None = None,
        apply_veto: bool = True,
    ) -> "FitSession":
        """Return a copy of this session with a `BackgroundSpec` appended."""
        spec = BackgroundSpec(
            name=name,
            shape=shape,
            fraction=fraction,
            yield_=yield_,
            normalization_sample=normalization_sample,
            apply_veto=apply_veto,
        )
        return replace(self, backgrounds=self.backgrounds + (spec,))

    def with_constraint(self, constraint: object) -> "FitSession":
        """Return a copy of this session with `constraint` appended."""
        return replace(self, constraints=self.constraints + (constraint,))

    @cached_property
    def acceptance_data(self) -> jnp.ndarray:
        return _acceptance(self.efficiency, self.veto, self.data.as_dict())

    @cached_property
    def acceptance_normalization(self) -> jnp.ndarray:
        sample = self.model.normalization_sample
        return _acceptance(self.efficiency, self.veto, sample.as_dict())

    @cached_property
    def signal_cache(self):
        """Prepared amplitude cache used by every likelihood evaluation."""

        return self.model.prepare_cache(
            self.data,
            self.model.normalization_sample,
            efficiency_normalization=(
                None
                if self.efficiency is None and self.veto is None
                else self.acceptance_normalization
            ),
        )

    @cached_property
    def signal_pdf(self) -> SignalPDF:
        """Generic PDF retained for diagnostics and arbitrary projection samples.

        The fit objective itself does not use this slower generic path; it uses
        ``signal_cache`` so repeated likelihood evaluations do not recompute
        fixed dynamics or the full normalization integral.
        """

        sample = self.model.normalization_sample

        def intensity(data, parameters):
            return self.model.intensity(data, parameters)

        return SignalPDF(
            intensity=intensity,
            integrator=GridIntegrator(sample),
            efficiency=UnityEfficiency() if self.efficiency is None else self.efficiency,
            veto=self.veto,
        )

    @cached_property
    def _projection_samples(self) -> dict[tuple[int, int | None], PhaseSpaceSample]:
        """Phase-space samples reused by repeated projection calls."""

        return {}

    @cached_property
    def _projection_prepared(self) -> dict[int, tuple[PhaseSpaceSample, object | None, jnp.ndarray]]:
        """Prepared amplitudes/acceptance keyed by live projection-sample identity."""

        return {}

    def _get_projection_sample(
        self,
        size: int,
        seed: int | None,
    ) -> PhaseSpaceSample:
        size = int(size)
        if size < 1:
            raise ValueError("projection_size must be positive")
        resolved_seed = None if seed is None else int(seed)
        key = (size, resolved_seed)
        sample = self._projection_samples.get(key)
        if sample is None:
            sample = self.model.generate_phase_space(size, seed=resolved_seed)
            self._projection_samples[key] = sample
        return sample

    def _prepare_projection_sample(
        self,
        sample: PhaseSpaceSample,
    ) -> tuple[object | None, jnp.ndarray]:
        """Prepare projection amplitudes once and reuse the fit normalization.

        Coefficient-only fits already have the exact fixed normalization matrix
        and component scales in ``signal_cache``.  Projection rendering only
        needs component values on a new phase-space sample, so rebuilding the
        normalization integral (the old ``signal_pdf`` path) is unnecessary.
        """

        key = id(sample)
        prepared = self._projection_prepared.get(key)
        if prepared is not None and prepared[0] is sample:
            return prepared[1], prepared[2]

        acceptance = _acceptance(self.efficiency, self.veto, sample.as_dict())
        template = self.signal_cache
        projection_cache = None
        if template.is_compact:
            if template.component_scales is None:
                raise RuntimeError("compact signal cache is missing component scales")
            compact_data_kernel = None
            kernel_builder = getattr(self.model, "_compact_data_kernel", None)
            if callable(kernel_builder):
                compact_data_kernel = kernel_builder(
                    normalize_components=template.normalize_components
                )
            projection_cache = PreparedAmplitudeCache.prepare_from_fixed_normalization(
                template.components,
                data=sample.as_dict(),
                normalization_weights=template.normalization_weights,
                parameters=template.parameters,
                normalization_matrix_fixed=template.normalization_matrix_fixed,
                component_scales=template.component_scales,
                normalize_components=template.normalize_components,
                compact_data_kernel=compact_data_kernel,
            )

        self._projection_prepared[key] = (sample, projection_cache, acceptance)
        return projection_cache, acceptance

    def _projection_signal_density(
        self,
        sample: PhaseSpaceSample,
        values: Mapping[str, object],
    ) -> jnp.ndarray:
        projection_cache, acceptance = self._prepare_projection_sample(sample)
        if projection_cache is not None:
            intensity, normalization = projection_cache.evaluate(values)
            return acceptance * intensity / normalization

        # Floating-dynamics caches cannot yet be cloned data-only. Keep the
        # generic path for correctness until partial dynamic preparation exists.
        return jnp.asarray(self.signal_pdf(sample.as_dict(), values))

    def _cached_signal_density(self, parameters: Mapping[str, object]) -> jnp.ndarray:
        intensity, normalization = self.signal_cache.evaluate(parameters)
        return self.acceptance_data * intensity / normalization

    def _cached_signal_logpdf(
        self,
        data: dict[str, object],
        parameters: Mapping[str, object],
    ) -> jnp.ndarray:
        del data
        intensity, normalization = self.signal_cache.evaluate(parameters)
        numerator = self.acceptance_data * intensity
        from jaxpwa.pdf.signal import _normalized_log_density

        return _normalized_log_density(numerator, normalization)

    @staticmethod
    def _evaluate_shape(shape: object, data: dict) -> jnp.ndarray:
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(
                f"background shape must return one value per event, got {values.shape} "
                f"for {size} events"
            )
        return values

    def _build_background(
        self,
        background: BackgroundSpec | BackgroundCategory,
    ) -> BackgroundCategory:
        if isinstance(background, BackgroundCategory):
            return background
        data_dict = self.data.as_dict()
        norm_sample = (
            self.model.normalization_sample
            if background.normalization_sample is None
            else background.normalization_sample
        )
        norm_dict = norm_sample.as_dict()
        data_values = self._evaluate_shape(background.shape, data_dict)
        norm_values = self._evaluate_shape(background.shape, norm_dict)
        if self.veto is not None and background.apply_veto:
            data_values = data_values * jnp.asarray(
                self.veto(data_dict), dtype=data_values.dtype
            )
            norm_values = norm_values * jnp.asarray(
                self.veto(norm_dict), dtype=norm_values.dtype
            )
        normalization = jnp.mean(jnp.asarray(norm_sample.weights) * norm_values)
        return BackgroundCategory(
            name=background.name,
            values=data_values,
            normalization=normalization,
            fraction=background.fraction,
            yield_=background.yield_,
        )

    @cached_property
    def background_categories(self) -> tuple[BackgroundCategory, ...]:
        return tuple(
            self._build_background(background) for background in self.backgrounds
        )

    @cached_property
    def base_objective(self):
        # Materialize all parameter-independent signal arrays before the
        # minimizer JIT traces the objective. Creating a cached_property while
        # tracing would otherwise store JAX tracers in the session and makes
        # host-side cache validation illegal inside the traced function.
        _ = self.signal_cache
        _ = self.acceptance_data

        data = self.data.as_dict()
        if not self.background_categories and not self.extended:
            return UnbinnedNLL(self._cached_signal_logpdf, data)
        return MultiBackgroundNLL(
            signal_density=self._cached_signal_density,
            backgrounds=self.background_categories,
            signal_fraction=self.signal_fraction,
            extended=self.extended,
            signal_yield=self.signal_yield,
        )

    def _in_coordinates(self, nll, weights=None):
        """Wrap an event NLL for this session's ``coordinates``."""
        if self.coordinates == "dalitz":
            return nll
        log_jacobian = square_dalitz_log_jacobian(
            self.model, self.data, pair=self.square_dalitz_pair
        )
        if weights is not None:
            log_jacobian = jnp.asarray(weights) * log_jacobian
        return SquareDalitzNLL(nll, jnp.sum(log_jacobian))

    @cached_property
    def objective(self):
        nll: object = self._in_coordinates(self.base_objective)
        if self.constraints:
            nll = ConstrainedNLL(nll, *self.constraints)
        return nll

    def _weighted_nll(self, weights):
        """Signal-only event term used for sWeight/sPlot Dalitz fits."""

        if (
            self.backgrounds
            or self.extended
            or self.signal_fraction is not None
            or self.signal_yield is not None
        ):
            raise ValueError(
                "event weights are incompatible with explicit background/fraction/"
                "yield configuration; build a signal-only FitSession for an sWeight fit"
            )

        # Materialize cached properties before JAX traces the weighted objective.
        _ = self.signal_cache
        _ = self.acceptance_data
        return WeightedUnbinnedNLL(
            self._cached_signal_logpdf,
            self.data.as_dict(),
            weights,
        )

    def _weighted_objective(self, weights):
        """Weighted signal event term plus this session's constraints."""
        nll: object = self._in_coordinates(self._weighted_nll(weights), weights)
        if self.constraints:
            nll = ConstrainedNLL(nll, *self.constraints)
        return nll

    def _score_outer_objective(
        self,
        weights,
        reference_parameters: Mapping[str, float],
    ):
        r"""Return a scalar whose Hessian is the weighted score outer product.

        At the reference point ``theta_hat``, define

        ``R(theta) = 0.5 * sum_i w_i^2 [log p_i(theta)-log p_i(theta_hat)]^2``.

        Every residual vanishes at ``theta_hat``, so
        ``H_R(theta_hat) = sum_i w_i^2 s_i s_i^T`` exactly. This lets the
        memory-aware JAX Hessian/HVP backend compute the Godambe variability
        matrix without materializing the event-by-parameter score Jacobian.
        """

        _ = self._weighted_objective(weights)
        data = self.data.as_dict()
        weights_array = jnp.asarray(weights)
        reference = jax.lax.stop_gradient(
            jnp.asarray(self._cached_signal_logpdf(data, reference_parameters))
        )

        def objective(parameters):
            current = jnp.asarray(self._cached_signal_logpdf(data, parameters))
            active = weights_array != 0
            delta = jnp.where(
                active,
                current - jnp.where(active, reference, 0.0),
                0.0,
            )
            return 0.5 * jnp.sum(jnp.square(weights_array) * jnp.square(delta))

        return objective

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """All fit `Parameter`s from model, yield/fraction, backgrounds, constraints.

        Deduplicated by name; raises if the same name resolves to conflicting
        `Parameter` definitions across those sources.
        """
        candidates: list[Parameter] = list(getattr(self.model, "parameters", ()))
        candidates.extend(_collect_parameters(self.signal_fraction))
        candidates.extend(_collect_parameters(self.signal_yield))
        candidates.extend(_collect_parameters(self.backgrounds))
        candidates.extend(_collect_parameters(self.constraints))
        unique: dict[str, Parameter] = {}
        for parameter in candidates:
            previous = unique.get(parameter.name)
            if previous is not None and previous != parameter:
                raise ValueError(
                    f"conflicting definitions for fit parameter {parameter.name!r}"
                )
            unique[parameter.name] = parameter
        return tuple(unique.values())

    def minimizer(
        self,
        *,
        tolerance: float = 1e-4,
        verbose: int = 0,
        hessian: str = "numerical",
    ) -> Minimizer:
        """Build a `Minimizer` over `self.objective` and `self.parameters`."""
        return Minimizer(
            self.objective,
            self.parameters,
            tolerance=tolerance,
            verbose=verbose,
            hessian=hessian,
        )

    def fit(
        self,
        start_values: Mapping[str, float] | None = None,
        *,
        weights: object | None = None,
        covariance: str = "minuit",
        simplex: bool = False,
        ncall: int | None = None,
        strategy: int = 2,
        hesse: bool = True,
        tolerance: float = 1e-4,
        verbose: int = 0,
        hessian: str = "numerical",
        method: str = "minuit",
        nesterov_max_iter: int = 1000,
        nesterov_gtol: float = 1e-4,
        update_model: bool = False,
    ):
        """Fit with optional event weights and corrected weighted covariance.

        Pass ``weights=sweights`` to minimize the signal-only weighted objective
        :math:`-\\sum_i w_i \\log p(x_i)`. ``covariance="sandwich"`` uses the
        Godambe form :math:`H_w^{-1}(\\sum_i w_i^2 s_i s_i^T)H_w^{-1}` and is
        the recommended fixed-weight signed-weight covariance.
        ``covariance="sumw2"`` uses
        :math:`H_w^{-1}H_{w^2}H_w^{-1}`; ``"sweight"`` is retained as a
        backwards-compatible alias for ``"sumw2"``.

        In all corrected weighted modes, ``hessian="jax"`` deliberately does
        not inject the signed-weight Hessian into MIGRAD\'s search metric:
        Minuit uses numerical curvature during minimization and the exact JAX
        matrices are evaluated only at the fitted point. ``covariance="minuit"``
        preserves Minuit\'s ordinary weighted-HESSE covariance and the usual
        ``hessian`` behavior.

        Weighted fits are intentionally signal-only: do not also configure an
        explicit Dalitz background mixture, signal fraction/yield, or extended
        likelihood in the same ``FitSession``.

        The ``ncall`` limit applies separately to SIMPLEX, each MIGRAD call and
        HESSE, not to the whole fit. Strategy 2 runs MIGRAD twice.

        ``self.model`` is a frozen ``DecayModel`` and is never mutated by this
        call, regardless of ``update_model``: fitting always reports its
        result separately (see ``result_values``). Pass ``update_model=True``
        to also get a new ``DecayModel`` with every free parameter's
        ``.value`` set to its fitted result (via ``model_with_fitted_values``)
        -- the return value then becomes ``(result, updated_model)`` instead
        of plain ``result``.
        """
        result = _fit_with_optional_weights(
            unweighted_minimizer=lambda: self.minimizer(
                tolerance=tolerance,
                verbose=verbose,
                hessian=hessian,
            ),
            weighted_objective=self._weighted_objective,
            event_weighted_objective=self._weighted_nll,
            score_outer_objective=self._score_outer_objective,
            parameters=self.parameters,
            weights=weights,
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
        return result, model_with_fitted_values(self.model, self.result_values(result))

    def fit_multistart(
        self,
        n_starts: int = 20,
        *,
        seed: int | None = None,
        include_default: bool = False,
        simplex: bool = False,
        strategy: int = 1,
        tolerance: float = 1e-4,
        verbose: int = 0,
        hessian: str = "numerical",
    ):
        """Fit from `n_starts` randomized initial values, keeping the best result."""
        return self.minimizer(
            tolerance=tolerance,
            verbose=verbose,
            hessian=hessian,
        ).fit_multistart(
            n_starts=n_starts,
            seed=seed,
            include_default=include_default,
            simplex=simplex,
            strategy=strategy,
        )

    def result_values(self, result) -> dict[str, float]:
        """Map each parameter name to its fitted value (fixed value if not floated)."""
        return {
            parameter.name: (
                float(parameter.value)
                if parameter.fixed
                else float(result.values[parameter.name])
            )
            for parameter in self.parameters
        }

    def print_result(self, result, *, precision: int = 6) -> dict[str, float]:
        """Print validity, NLL and a value/error table, and return `result_values`."""
        if precision < 0:
            raise ValueError("precision must be non-negative")
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for parameter in self.parameters:
            value = values[parameter.name]
            error = 0.0 if parameter.fixed else float(result.errors[parameter.name])
            print(
                f"{parameter.name:24s} {value:16.{precision}g} "
                f"{error:16.{precision}g}"
            )
        return values

    def print_fit_fractions(
        self,
        result,
        *,
        acceptance_weighted: bool = False,
        include_interference: bool = False,
        precision: int = 3,
        with_errors: bool = True,
    ):
        """Print and return per-component fit fractions at `result`'s fitted values.

        ``with_errors=True`` -- the same flag ``DecayModel.print_fit_fractions``
        exposes -- also prints each fraction's (and, with
        ``include_interference=True``, each interference term's) delta-method
        standard error (see ``fit_fraction_errors()``) as an extra column.
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
        self,
        result,
        *,
        acceptance_weighted: bool = False,
    ) -> dict[str, float]:
        """Delta-method standard errors for print_fit_fractions()'s central values.

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
        self,
        result,
        *,
        include_fit_fractions: bool = True,
        acceptance_weighted_fractions: bool = False,
        include_correlation: bool = True,
    ) -> dict[str, object]:
        """Assemble a summary dict of fit results: validity, NLL, EDM, values, errors.

        Optionally includes fit fractions (`include_fit_fractions`) and the
        free-parameter correlation matrix (`include_correlation`, only if a
        covariance is available).
        """
        values = self.print_result(result)
        errors = {
            parameter.name: (
                0.0
                if parameter.fixed
                else float(result.errors[parameter.name])
            )
            for parameter in self.parameters
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
                result,
                acceptance_weighted=acceptance_weighted_fractions,
            )
        if include_correlation and getattr(result, "covariance", None) is not None:
            correlation = result.covariance.correlation()
            free = [parameter.name for parameter in self.parameters if not parameter.fixed]
            report["correlation"] = {
                first: {
                    second: float(correlation[first, second]) for second in free
                }
                for first in free
            }
        return report

    def _projection_components(
        self,
        values: Mapping[str, float],
        projection_sample: PhaseSpaceSample | None = None,
    ) -> list[tuple[str, PhaseSpaceSample, np.ndarray]]:
        sample = (
            self.model.normalization_sample
            if projection_sample is None
            else projection_sample
        )
        if self.extended:
            signal_scale = float(_resolve(self.signal_yield, values))
        elif self.background_categories:
            signal_scale = self.data.size * float(
                _resolve(self.signal_fraction, values)
            )
        else:
            signal_scale = float(self.data.size)
        signal_density = self._projection_signal_density(sample, values) if signal_scale else np.zeros(sample.size)
        components = [
            (
                "signal",
                sample,
                _scaled_projection_weights(sample, signal_density, signal_scale),
            )
        ]
        if not self.background_categories:
            return components
        if self.extended:
            bg_scales = [
                float(_resolve(category.yield_, values))
                for category in self.background_categories
            ]
        else:
            bg_total = self.data.size * (
                1.0 - float(_resolve(self.signal_fraction, values))
            )
            weights = np.asarray(
                self.base_objective.background_weights(values), dtype=float
            )
            bg_scales = [bg_total * float(weight) for weight in weights]
        for source, category, scale in zip(
            self.backgrounds,
            self.background_categories,
            bg_scales,
        ):
            if scale == 0:
                continue
            if not isinstance(source, BackgroundSpec):
                raise ValueError("plotting a precomputed background requires a BackgroundSpec with an evaluable shape")
            bg_sample = (
                source.normalization_sample or self.model.normalization_sample
            ) if projection_sample is None else projection_sample
            raw = jnp.asarray(source.shape(bg_sample.as_dict()))
            if self.veto is not None and source.apply_veto:
                raw = raw * jnp.asarray(self.veto(bg_sample.as_dict()))
            density = raw / category.normalization
            components.append(
                (
                    category.name,
                    bg_sample,
                    _scaled_projection_weights(bg_sample, density, scale),
                )
            )
        return components

    def plot_projection(
        self,
        result,
        variable: str = "s13",
        *,
        bins: int = 60,
        range: tuple[float, float] | None = None,
        show_components: bool = True,
        show_pulls: bool = False,
        log_scale: bool = False,
        projection_size: int = 1_000_000,
        projection_seed: int = 20260901,
        projection_sample: PhaseSpaceSample | None = None,
        folded: bool = False,
        partner_variable: str | None = None,
        fold_side: str = "low",
        ax=None,
    ):
        """Plot data and a smooth fitted projection.

        Fit/PDF normalization remains deterministic. A weighted phase-space MC
        sample is used only to render the one-dimensional model projection. The
        generated sample and coefficient-only prepared amplitudes are cached by
        the session, so plotting another invariant or changing bins does not
        repeat phase-space generation or fixed resonance dynamics.

        ``folded=True`` projects onto ``s_low = min(variable, partner_variable)``
        (``fold_side="low"``, the default) or ``s_high = max(...)``
        (``fold_side="high"``) event by event, for both data and each fitted
        component. ``partner_variable`` is the other exchange-symmetric
        invariant for a channel with two identical daughters, e.g.
        ``variable="s12"``, ``partner_variable="s13"``. This is the same
        ``min``/``max`` fold used by ``plot_dalitz``/``plot_square_dalitz`` and
        the folded efficiency/background models, applied here to a 1D
        projection; call it twice, with ``fold_side="low"`` and ``"high"``, to
        get the usual pair of folded spectra.

        ``show_pulls=True`` adds a ``(observed-expected)/sqrt(expected)`` panel
        below the histogram, sharing the x axis (same convention as
        :func:`~jaxpwa.plotting.plot_pulls`, and the same binning
        used here for the histogram itself -- not the independent binning
        ``goodness_of_fit_projection`` would choose for an actual GOF test).
        It builds its own two-row figure and therefore requires ``ax=None``;
        the return value is then ``(ax, ax_pulls)`` instead of a single ``ax``.
        """

        import matplotlib.pyplot as plt

        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        if show_pulls and ax is not None:
            raise ValueError(
                "show_pulls=True builds its own figure layout; pass ax=None"
            )
        fold_fn = np.minimum if fold_side == "low" else np.maximum

        def _folded_values(sample):
            values_ = _values(sample, variable)
            if not folded:
                return values_
            partner_values = _values(sample, partner_variable)
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        data_values = _folded_values(self.data)
        hist_range = range or (
            float(np.min(data_values)),
            float(np.max(data_values)),
        )
        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        ax_pulls = None
        if ax is None:
            if show_pulls:
                # Stack the pulls panel below the main one at the active
                # style's own base size (`plt.style.use(...)`, e.g. mplhep),
                # rather than a hardcoded absolute figsize -- only the extra
                # panel's proportional height is added on top of it.
                base_w, base_h = plt.rcParams["figure.figsize"]
                _, (ax, ax_pulls) = plt.subplots(
                    2, 1, figsize=(base_w, base_h * 1.3), sharex=True,
                    gridspec_kw={"height_ratios": (3, 1)},
                    constrained_layout=True,
                )
            else:
                _, ax = plt.subplots()
        unit = (
            r"GeV$^2$" if variable.startswith("s") and variable[1:].isdigit() else ""
        )
        _, observed, _, _ = plot_binned_data(
            data_values,
            bins=edges,
            ax=ax,
            label="data",
            unit=unit,
            log_scale=log_scale,
        )
        sample = (
            self._get_projection_sample(projection_size, projection_seed)
            if projection_sample is None
            else projection_sample
        )
        total = np.zeros(bins, dtype=float)
        for name, component_sample, weights in self._projection_components(
            values,
            sample,
        ):
            component_values = _folded_values(component_sample)
            counts, _ = np.histogram(
                component_values,
                bins=edges,
                weights=np.asarray(weights),
            )
            total += counts
            if show_components:
                ax.stairs(counts, edges, label=name)
        ax.stairs(total, edges, label="total fit", linewidth=2.0)
        label = (
            rf"$s_{{\mathrm{{{fold_side}}}}}$" if folded else rf"${variable}$"
        )
        axis_label = label + (" [GeV$^2$]" if unit else "")
        ax.legend()

        if not show_pulls:
            ax.set_xlabel(axis_label)
            return ax

        occupied = total > 0
        pulls = np.full(bins, np.nan)
        pulls[occupied] = (
            (observed[occupied] - total[occupied]) / np.sqrt(total[occupied])
        )
        _draw_pulls_1d(ax_pulls, edges, pulls)
        ax_pulls.set_xlabel(axis_label)
        return ax, ax_pulls

    def _total_density(
        self,
        sample: PhaseSpaceSample,
        values: Mapping[str, float],
    ) -> jnp.ndarray:
        """Fraction-weighted, unit-integral total fitted density f0(x).

        Mirrors the extended/signal_fraction/plain branch structure of
        ``_projection_components`` but returns *fractions* summing to one
        (rather than absolute event-count scales), so the result integrates
        to one over phase space -- what the point-to-point dissimilarity
        test's ``f0`` requires. Deliberately does not reuse
        ``_scaled_projection_weights``: that helper's normalization only
        cancels correctly when ``sample`` is itself a representative
        phase-space integration sample, which arbitrary points (data, or a
        resampled replica) are not.
        """

        if self.extended:
            signal_yield = float(_resolve(self.signal_yield, values))
            bg_yields = [
                float(_resolve(category.yield_, values))
                for category in self.background_categories
            ]
            total = signal_yield + sum(bg_yields)
            if total <= 0:
                raise ValueError("extended total yield must be positive")
            signal_fraction = signal_yield / total
            bg_fractions = [yield_ / total for yield_ in bg_yields]
        elif self.background_categories:
            signal_fraction = float(_resolve(self.signal_fraction, values))
            weights = np.asarray(
                self.base_objective.background_weights(values), dtype=float
            )
            bg_fractions = list((1.0 - signal_fraction) * weights)
        else:
            signal_fraction = 1.0
            bg_fractions = []

        density = signal_fraction * self._projection_signal_density(sample, values)
        for source, category, fraction in zip(
            self.backgrounds, self.background_categories, bg_fractions
        ):
            if fraction == 0:
                continue
            if not isinstance(source, BackgroundSpec):
                raise ValueError(
                    "goodness-of-fit density requires a BackgroundSpec with an "
                    "evaluable shape"
                )
            raw = jnp.asarray(source.shape(sample.as_dict()))
            if self.veto is not None and source.apply_veto:
                raw = raw * jnp.asarray(self.veto(sample.as_dict()))
            density = density + fraction * (raw / category.normalization)
        return density

    def _default_free_parameters(self) -> int:
        return sum(1 for parameter in self.parameters if not parameter.fixed)

    def goodness_of_fit_projection(
        self,
        result,
        variable: str = "s13",
        *,
        bins: int = 60,
        range: tuple[float, float] | None = None,
        folded: bool = False,
        partner_variable: str | None = None,
        fold_side: str = "low",
        projection_size: int = 100_000,
        projection_seed: int = 20260901,
        n_free_parameters: int | None = None,
    ) -> BinnedChi2Result:
        """Binned Pearson chi2 goodness-of-fit test on a 1D projection.

        Uses the same reweighted-MC-projection histogram as
        ``plot_projection`` for the expected counts, and the same
        ``folded``/``partner_variable``/``fold_side`` convention. See
        :class:`~jaxpwa.goodness_of_fit.BinnedChi2Result` for the
        degrees-of-freedom caveat that applies here: the free parameters came
        from an unbinned fit, not from minimizing this chi2.
        """

        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        fold_fn = np.minimum if fold_side == "low" else np.maximum

        def _folded_values(sample):
            values_ = _values(sample, variable)
            if not folded:
                return values_
            partner_values = _values(sample, partner_variable)
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        data_values = _folded_values(self.data)
        hist_range = range or (float(np.min(data_values)), float(np.max(data_values)))
        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        observed, _ = np.histogram(data_values, bins=edges)

        sample = self._get_projection_sample(projection_size, projection_seed)
        expected = np.zeros(bins, dtype=float)
        for _, component_sample, weights in self._projection_components(values, sample):
            component_values = _folded_values(component_sample)
            counts, _ = np.histogram(
                component_values, bins=edges, weights=np.asarray(weights)
            )
            expected += counts

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()
        return chi2_from_histograms(
            observed, expected, n_free_parameters=n_free_parameters, edges=(edges,)
        )

    def goodness_of_fit_chi2(
        self,
        result,
        x: str = "s13",
        y: str = "s23",
        *,
        bins: int = 25,
        range: tuple[tuple[float, float], tuple[float, float]] | None = None,
        folded: bool = False,
        square_dalitz: bool = False,
        mother_mass: float | None = None,
        masses: tuple[float, float, float] | None = None,
        pair: tuple[int, int] = (0, 1),
        projection_size: int = 200_000,
        projection_seed: int = 20260901,
        n_free_parameters: int | None = None,
    ) -> BinnedChi2Result:
        """Binned Pearson chi2 goodness-of-fit test on the Dalitz plane.

        Uses the same reweighted-MC-projection histogram trick as
        ``plot_projection``, in 2D, via ``np.histogram2d``.
        ``square_dalitz=True`` bins in Laura++ ``(m', theta')`` coordinates
        instead of raw ``(x, y)`` invariants (``mother_mass``/``masses`` are
        then required, matching ``plot_square_dalitz``); ``folded=True``
        folds onto the physically distinct half for two identical daughters,
        matching ``plot_dalitz(folded=True)``/``plot_square_dalitz(folded=True)``.
        """

        if square_dalitz and (mother_mass is None or masses is None):
            raise ValueError("square_dalitz=True requires mother_mass and masses")

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
        data_x, data_y = _coordinates(self.data)
        observed, x_edges, y_edges = np.histogram2d(
            data_x, data_y, bins=bins, range=hist_range
        )

        sample = self._get_projection_sample(projection_size, projection_seed)
        expected = np.zeros_like(observed)
        for _, component_sample, weights in self._projection_components(values, sample):
            component_x, component_y = _coordinates(component_sample)
            counts, _, _ = np.histogram2d(
                component_x,
                component_y,
                bins=[x_edges, y_edges],
                weights=np.asarray(weights),
            )
            expected += counts

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()
        return chi2_from_histograms(
            observed,
            expected,
            n_free_parameters=n_free_parameters,
            edges=(x_edges, y_edges),
        )

    def point_to_point_dissimilarity(
        self,
        result,
        *,
        x: str = "s13",
        y: str = "s23",
        sigma_bar: float = 0.01,
        mc_size: int | None = None,
        n_permutations: int = 200,
        seed: int = 20260901,
        max_total_events: int = 5_000,
        candidate_pool_size: int = 200_000,
    ) -> PointToPointResult:
        """Unbinned point-to-point dissimilarity goodness-of-fit test.

        Draws an unweighted Monte Carlo replica from the fitted total density
        (signal plus backgrounds, fraction-weighted) via
        :func:`~jaxpwa.sampling.weighted_resample`, matching the
        requirement (Williams, arXiv:1006.3019) that the reference sample be
        drawn from the fitted density ``f0``. See
        :class:`~jaxpwa.goodness_of_fit.PointToPointResult` and
        ``docs/goodness_of_fit.md`` for the ``sigma_bar``/``max_total_events``
        tuning notes; this test is O(n^2) in the pooled data+replica size, so
        keep ``mc_size`` modest.
        """

        values = self.result_values(result)
        n_data = self.data.size
        if mc_size is None:
            mc_size = min(10 * n_data, max(max_total_events - n_data, 1))

        candidate = self._get_projection_sample(candidate_pool_size, seed)
        candidate_density = self._total_density(candidate, values)
        target_weights = candidate.weights * candidate_density
        replica = weighted_resample(
            jax.random.PRNGKey(seed), candidate, target_weights, mc_size
        )

        data_density = self._total_density(self.data, values)
        replica_density = self._total_density(replica, values)
        phase_space_area = float(jnp.mean(self.model.normalization_sample.weights))

        data_xy = np.column_stack(
            [np.asarray(getattr(self.data, x)), np.asarray(getattr(self.data, y))]
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


__all__ = ["BackgroundSpec", "FitSession"]
