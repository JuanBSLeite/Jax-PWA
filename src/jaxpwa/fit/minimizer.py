"""Minuit driver using JAX values and automatic gradients."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .nesterov import minimize as nesterov_minimize
from .parameters import Parameter, ParameterKind

# ``FitSession.fit()`` constructs a new Minimizer for each call. The objective
# object itself is cached by the session, so use its identity to retain the JAX
# executable across those short-lived Minimizer instances. The compiled closure
# keeps only a weak reference to the objective, so this global lookup cannot keep
# a finished fit session alive.
_SHARED_BACKENDS: dict[tuple[int, tuple], tuple[weakref.ReferenceType, object]] = {}


@dataclass(frozen=True)
class GradientCheckResult:
    """Comparison between JAX and central finite-difference gradients."""

    names: tuple[str, ...]
    point: dict[str, float]
    jax_gradient: np.ndarray
    finite_difference_gradient: np.ndarray
    absolute_error: np.ndarray
    relative_error: np.ndarray

    @property
    def max_absolute_error(self) -> float:
        return float(np.max(self.absolute_error))

    @property
    def max_relative_error(self) -> float:
        return float(np.max(self.relative_error))


@dataclass(frozen=True)
class MultiStartResult:
    """Collection of independent minimizations and the best valid minimum."""

    best: object
    results: tuple[object, ...]
    starts: tuple[dict[str, float], ...]

    @property
    def valid_results(self) -> tuple[object, ...]:
        """Subset of ``results`` that Minuit marked valid with a finite fval."""

        return tuple(
            result
            for result in self.results
            if bool(result.valid) and np.isfinite(float(result.fval))
        )


class Minimizer:
    """Minimize a mapping-based JAX objective with iminuit.

    ``errordef=0.5`` is the Minuit convention for a negative log-likelihood.
    ``tolerance`` is passed directly to ``Minuit.tol``. The default ``1e-4``
    provides a stricter convergence target than iminuit's generic default while
    remaining numerically practical for the amplitude-fit likelihoods used here.

    The JAX value-and-gradient program is compiled lazily. It is cached both by
    the ``Minimizer`` instance and, while the objective remains alive, across
    different Minimizer instances that wrap the same objective and parameter
    layout. This means repeated ``FitSession.fit()`` or ``CPFitSession.fit()``
    calls reuse the already-compiled XLA executable instead of paying the JIT
    cost again.

    Value and gradient callbacks share a one-point host cache, so when Minuit
    asks for both at identical parameters the expensive device evaluation and
    device-to-host synchronization occur only once.

    ``fit`` exposes Minuit strategies 0, 1 and 2 directly. The historical
    project default remains strategy 2 with HESSE enabled; that path keeps the
    established two-MIGRAD refinement before HESSE. Strategies 0 and 1 use one
    MIGRAD pass, which is useful for faster preliminary fits of large models.

    ``hessian="jax"`` supplies automatic second derivatives to Minuit, including
    its internal HESSE calls during MIGRAD. The default ``"numerical"`` keeps
    Minuit's finite differences. JAX Hessians compile lazily. Floating-dynamics
    fits evaluate the Hessian one Hessian-vector product at a time. Second-order
    differentiability is required.
    """

    def __init__(
        self,
        objective: Callable,
        parameters: Sequence[Parameter],
        *,
        errordef: float = 0.5,
        tolerance: float = 1e-4,
        verbose: int = 0,
        hessian: str = "numerical",
    ):
        if errordef <= 0:
            raise ValueError("errordef must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")
        if hessian not in ("numerical", "jax"):
            raise ValueError("hessian must be 'numerical' or 'jax'")
        self.objective = objective
        self.parameters = tuple(parameters)
        self.errordef = float(errordef)
        self.tolerance = float(tolerance)
        self.verbose = int(verbose)
        self.hessian = hessian
        self._backend_cache = None

    def _log(self, message: str) -> None:
        if self.verbose >= 1:
            print(f"[Minimizer] {message}", flush=True)

    @staticmethod
    def _summary(result) -> str:
        return (
            f"valid={bool(result.valid)}  "
            f"NLL={float(result.fval):.6f}  "
            f"EDM={float(result.fmin.edm):.3e}  "
            f"nfcn={int(result.nfcn)}"
        )

    def _backend_signature(self) -> tuple:
        return tuple(
            (
                parameter.name,
                bool(parameter.fixed),
                float(parameter.value) if parameter.fixed else None,
                parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed,
            )
            for parameter in self.parameters
        )

    def _shared_backend(self):
        key = (
            id(self.objective),
            self._backend_signature(),
        )
        cached = _SHARED_BACKENDS.get(key)
        if cached is None:
            return key, None
        objective_ref, backend = cached
        if objective_ref() is self.objective:
            return key, backend
        _SHARED_BACKENDS.pop(key, None)
        return key, None

    def _backend(self):
        """Return persistent compiled JAX and Minuit callbacks."""

        if self._backend_cache is not None:
            return self._backend_cache

        free = tuple(parameter for parameter in self.parameters if not parameter.fixed)
        shared_key, shared = self._shared_backend()
        if shared is not None:
            # Only compiled callbacks are reusable. Defaults, limits and steps
            # belong to this Minimizer, not the instance that compiled them.
            _, names, fcn, grad, hessian = shared
            self._backend_cache = (free, names, fcn, grad, hessian)
            return self._backend_cache

        fixed = {
            parameter.name: parameter.value
            for parameter in self.parameters
            if parameter.fixed
        }
        names = tuple(parameter.name for parameter in free)
        if not free:
            raise ValueError("At least one free parameter is required")

        try:
            objective_ref = weakref.ref(
                self.objective,
                lambda _reference, cache_key=shared_key: _SHARED_BACKENDS.pop(
                    cache_key, None
                ),
            )
        except TypeError:
            objective_ref = None
            objective = self.objective

        if objective_ref is not None:
            def vector_objective(vector):
                objective = objective_ref()
                if objective is None:
                    raise RuntimeError(
                        "fit objective was released before JAX evaluation"
                    )
                mapping = dict(fixed)
                mapping.update({name: vector[i] for i, name in enumerate(names)})
                return objective(mapping)
        else:
            def vector_objective(vector):
                mapping = dict(fixed)
                mapping.update({name: vector[i] for i, name in enumerate(names)})
                return objective(mapping)

        value_and_grad = jax.jit(jax.value_and_grad(vector_objective))

        last_point: np.ndarray | None = None
        last_value: float | None = None
        last_gradient: np.ndarray | None = None

        def evaluate(values):
            nonlocal last_point, last_value, last_gradient
            point = np.asarray(values, dtype=float)
            if last_point is not None and np.array_equal(point, last_point):
                return last_value, last_gradient

            value_device, gradient_device = value_and_grad(jnp.asarray(point))
            value_host, gradient_host = jax.device_get((value_device, gradient_device))
            last_point = point.copy()
            last_value = float(value_host)
            last_gradient = np.asarray(gradient_host, dtype=float)
            return last_value, last_gradient

        def fcn(*values):
            value, _ = evaluate(values)
            return value

        def grad(*values):
            _, gradient = evaluate(values)
            return gradient

        has_floating_dynamics = any(
            parameter.kind is ParameterKind.DYNAMICS for parameter in free
        )
        gradient_function = jax.grad(vector_objective)
        if has_floating_dynamics:
            # Recompute the gradient's forward pass in the second-order pass
            # instead of storing its residuals for the whole normalization grid.
            gradient_function = jax.checkpoint(gradient_function)

        @jax.jit
        def hessian_vector_product(vector, tangent):
            # Differentiate the gradient in forward mode, not the original
            # custom_vjp objective. QMI's gradient has already expanded its
            # grouped reductions; this preserves them in second derivatives.
            # Reverse-over-reverse instead reintroduces highly contended FP64
            # scatter-adds on GPU when differentiating the saved knot gathers.
            return jax.jvp(
                gradient_function,
                (vector,),
                (tangent,),
            )[1]

        @jax.jit
        def hessian_program(vector):
            # Coefficient-only objectives are small enough to reuse one
            # linearization and evaluate every column inside one executable.
            _, pushforward = jax.linearize(gradient_function, vector)
            return jax.lax.map(
                pushforward,
                jnp.eye(len(names), dtype=vector.dtype),
            ).T

        hessian_point = None
        hessian_value = None
        @jax.jit
        def diagonal_program(vector):
            def diagonal_entry(index):
                tangent = jax.nn.one_hot(index, len(names), dtype=vector.dtype)
                product = jax.jvp(
                    gradient_function, (vector,), (tangent,),
                )[1]
                return product[index]

            # Keep the directional derivatives inside one execution and
            # retain only H_ii. In particular, do not build a full Hessian
            # or reuse a grid-wide linearization for floating dynamics.
            return jax.lax.map(diagonal_entry, jnp.arange(len(names)))

        diagonal_point = None
        diagonal_value = None

        def diagonal(*values):
            """Exact external-coordinate G2 without assembling a new full matrix."""
            nonlocal diagonal_point, diagonal_value
            point = np.asarray(values, dtype=float)
            if hessian_point is not None and np.array_equal(point, hessian_point):
                return np.diag(hessian_value)
            if diagonal_point is None or not np.array_equal(point, diagonal_point):
                diagonal_value = np.asarray(
                    jax.device_get(diagonal_program(jnp.asarray(point))), dtype=float
                )
                diagonal_point = point.copy()
            return diagonal_value

        def hessian(*values):
            nonlocal hessian_point, hessian_value
            point = np.asarray(values, dtype=float)
            if hessian_point is None or not np.array_equal(point, hessian_point):
                device_point = jnp.asarray(point)
                if has_floating_dynamics:
                    # Synchronize each product before launching the next, so
                    # temporary buffers cannot overlap across columns.
                    columns = []
                    for tangent in np.eye(len(names), dtype=point.dtype):
                        product = jax.device_get(
                            hessian_vector_product(
                                device_point,
                                jnp.asarray(tangent, dtype=device_point.dtype),
                            )
                        )
                        columns.append(np.asarray(product, dtype=float))
                    matrix = np.column_stack(columns)
                else:
                    matrix = np.asarray(
                        jax.device_get(hessian_program(device_point)), dtype=float
                    )
                # Minuit expects a symmetric matrix in external coordinates.
                # Its own transformations handle parameter bounds and errordef.
                hessian_value = 0.5 * (matrix + matrix.T)
                hessian_point = point.copy()
            return hessian_value

        # Keep the existing private backend tuple intact while sharing the
        # companion G2 callback and its cache with the Hessian executable.
        hessian.diagonal = diagonal
        backend = (free, names, fcn, grad, hessian)
        self._backend_cache = backend
        if objective_ref is not None:
            _SHARED_BACKENDS[shared_key] = (objective_ref, backend)
        return backend

    def _validate_start_values(
        self,
        start_values: Mapping[str, float] | None,
    ) -> dict[str, float]:
        values = {} if start_values is None else dict(start_values)
        known_names = {parameter.name for parameter in self.parameters}
        unknown = set(values) - known_names
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown starting parameters: {unknown_text}")
        for parameter in self.parameters:
            if parameter.name not in values or parameter.bounds is None:
                continue
            value = float(values[parameter.name])
            low, high = parameter.bounds
            if low is not None and value < low:
                raise ValueError(
                    f"starting value {value} is below the lower bound {low} for "
                    f"{parameter.name!r}"
                )
            if high is not None and value > high:
                raise ValueError(
                    f"starting value {value} is above the upper bound {high} for "
                    f"{parameter.name!r}"
                )
        return values

    @staticmethod
    def _validate_ncall(ncall: int | None) -> int | None:
        if ncall is None:
            return None
        if isinstance(ncall, bool) or not isinstance(ncall, int) or ncall <= 0:
            raise ValueError("ncall must be a positive integer or None")
        return ncall

    @staticmethod
    def _validate_strategy(strategy: int) -> int:
        if (
            isinstance(strategy, bool)
            or not isinstance(strategy, int)
            or strategy not in (0, 1, 2)
        ):
            raise ValueError("strategy must be one of 0, 1 or 2")
        return int(strategy)

    def check_gradient(
        self,
        values: Mapping[str, float] | None = None,
        *,
        step_scale: float = 1e-5,
        relative_floor: float = 1e-12,
        print_table: bool = True,
    ) -> GradientCheckResult:
        """Compare the JAX gradient against a central finite-difference gradient.

        Evaluated at ``values`` (or each free parameter's current value) and
        optionally prints a per-parameter comparison table.
        """

        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if relative_floor <= 0:
            raise ValueError("relative_floor must be positive")

        free, names, fcn, grad, _ = self._backend()
        supplied = self._validate_start_values(values)
        point = np.asarray(
            [
                float(supplied.get(parameter.name, parameter.value))
                for parameter in free
            ],
            dtype=float,
        )

        jax_gradient = np.asarray(grad(*point), dtype=float)
        finite_difference = np.empty_like(point)

        for i, value in enumerate(point):
            step = step_scale * max(abs(float(value)), 1.0)
            plus = point.copy()
            minus = point.copy()
            plus[i] += step
            minus[i] -= step
            finite_difference[i] = (fcn(*plus) - fcn(*minus)) / (2.0 * step)

        absolute_error = np.abs(jax_gradient - finite_difference)
        scale = np.maximum(
            np.maximum(np.abs(jax_gradient), np.abs(finite_difference)),
            relative_floor,
        )
        relative_error = absolute_error / scale

        result = GradientCheckResult(
            names=names,
            point={name: float(point[i]) for i, name in enumerate(names)},
            jax_gradient=jax_gradient,
            finite_difference_gradient=finite_difference,
            absolute_error=absolute_error,
            relative_error=relative_error,
        )

        if print_table:
            print(
                f"{'parameter':18s} {'JAX':>14s} {'finite diff':>14s} "
                f"{'abs err':>12s} {'rel err':>12s}"
            )
            for i, name in enumerate(names):
                print(
                    f"{name:18s} {jax_gradient[i]:14.6e} "
                    f"{finite_difference[i]:14.6e} {absolute_error[i]:12.3e} "
                    f"{relative_error[i]:12.3e}"
                )
            print(
                f"max abs error = {result.max_absolute_error:.3e}\n"
                f"max rel error = {result.max_relative_error:.3e}"
            )

        return result

    def jax_hessian(
        self, values: Mapping[str, float] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Exact objective Hessian from the memory-aware JAX backend.

        Returns ``(names, hessian)`` in free-parameter order. Floating-dynamics
        fits keep the same sequential HVP implementation used by
        ``hessian="jax"``, so callers such as the sWeight covariance correction
        do not need to materialize an event-by-parameter Jacobian.
        """
        free, names, _, _, hessian_callback = self._backend()
        supplied = self._validate_start_values(values)
        point = np.asarray(
            [
                float(supplied.get(parameter.name, parameter.value))
                for parameter in free
            ],
            dtype=float,
        )
        hessian = np.asarray(hessian_callback(*point), dtype=float)
        return names, 0.5 * (hessian + hessian.T)

    def jax_covariance(
        self, values: Mapping[str, float] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Exact-Hessian covariance from JAX autodiff, without running Minuit's HESSE.

        Returns ``(names, covariance)``: ``names`` in the same free-parameter
        order the rest of ``Minimizer`` uses internally, and ``covariance =
        2 * errordef * inverse(Hessian(objective))`` evaluated at ``values``
        (each free parameter's current value if omitted, or the fitted
        ``result.values`` for a postfit covariance) -- the identical
        errordef relation Minuit's own HESSE uses (see
        ``test_jax_hessian_correlated_covariance_with_bounds_and_fixed_parameter``,
        which checks this formula against Minuit's own covariance). Unlike
        ``hessian="jax"`` (which only feeds this same exact Hessian *into*
        Minuit's HESSE/MIGRAD), this computes the covariance directly and
        never calls Minuit for it. The Hessian must be invertible, or
        ``numpy.linalg.inv`` raises ``LinAlgError``. The returned dense array
        is accepted directly by ``_covariance_matrix``/``fit_fraction_errors``
        as long as ``parameter_names`` is passed as this same ``names``.
        """
        names, hessian = self.jax_hessian(values)
        covariance = 2.0 * self.errordef * np.linalg.inv(hessian)
        return names, 0.5 * (covariance + covariance.T)

    def _run(
        self,
        free,
        names,
        fcn,
        grad,
        hessian_callback,
        *,
        start_values: Mapping[str, float] | None,
        strategy: int,
        hesse: bool,
        simplex: bool,
        ncall: int | None = None,
    ):
        from iminuit import Minuit

        ncall = self._validate_ncall(ncall)
        strategy = self._validate_strategy(strategy)
        supplied = self._validate_start_values(start_values)
        start = tuple(
            float(supplied.get(parameter.name, parameter.value)) for parameter in free
        )
        derivatives = {}
        if self.hessian == "jax":
            derivatives = {
                "hessian": hessian_callback,
                "g2": hessian_callback.diagonal,
            }
        # iminuit 2.32 normally derives G2 from the Hessian, but its negative-
        # curvature recovery can still call the separate G2 callback. Without
        # it, non-convex starting points raise "NoneType is not callable".
        # G2 reuses a cached Hessian at the same point, otherwise computes
        # only its diagonal. Silence only the
        # misleading warning about supplying both, not numerical warnings.
        # iminuit issues that warning with stacklevel=2, so it is attributed
        # to *this* module (the Minuit(...) call site) rather than
        # iminuit.minuit - filter on that, not on iminuit's own module.
        from iminuit.warnings import IMinuitWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="hessian overrides g2, passing g2 has no effect",
                category=IMinuitWarning,
                module=__name__,
            )
            minuit = Minuit(fcn, *start, name=names, grad=grad, **derivatives)
        minuit.errordef = self.errordef
        minuit.tol = self.tolerance
        minuit.strategy = strategy
        minuit.print_level = max(0, self.verbose - 1)
        for parameter in free:
            if parameter.bounds is not None:
                minuit.limits[parameter.name] = parameter.bounds
            if parameter.step is not None:
                minuit.errors[parameter.name] = parameter.step

        best_fval = float("inf")
        best_values = {name: float(minuit.values[name]) for name in names}
        best_result = None

        def stage(label, method, **kwargs):
            nonlocal best_fval, best_values, best_result
            self._log(f"{label} started")
            started = perf_counter()
            nfcn, ngrad = minuit.nfcn, minuit.ngrad
            nhessian = minuit.nhessian
            method(ncall=ncall, **kwargs)
            nonlocal_best = float(minuit.fval)
            if np.isfinite(nonlocal_best) and nonlocal_best <= best_fval:
                best_fval = nonlocal_best
                best_values = {name: float(minuit.values[name]) for name in names}
                # Retain values, fval, validity and covariance together. Merely
                # assigning minuit.values leaves its previous FMin in place
                # until another minimization/HESSE recomputes it. The callbacks
                # are functions, so deepcopy shares their compiled JAX closures.
                best_result = deepcopy(minuit)
            self._log(
                f"{label} finished in {perf_counter() - started:.3f} s: "
                f"nfcn=+{minuit.nfcn - nfcn}, ngrad=+{minuit.ngrad - ngrad}, "
                f"nhessian=+{minuit.nhessian - nhessian}"
            )

        # Near the EDM boundary, numerical line-search noise can leave a
        # perfectly finite minimum marked invalid by a tiny margin. Give
        # MIGRAD one polishing pass before exposing that status to callers.
        # This must run *after* the best-fval restore below: restoring an
        # earlier stage's point can itself leave the fit invalid, and a
        # polish attempted only before the restore would be silently undone
        # by it, re-exposing the same invalid status the polish was meant to
        # fix.
        def polish_if_invalid():
            if not bool(minuit.valid):
                stage("MIGRAD polish", minuit.migrad, use_simplex=False)
                if hesse:
                    stage("HESSE polish", minuit.hesse)

        if simplex:
            stage("SIMPLEX", minuit.simplex)
        stage("MIGRAD 1", minuit.migrad, use_simplex=False)
        if strategy == 2:
            stage("MIGRAD 2", minuit.migrad, use_simplex=False)
        if hesse:
            stage("HESSE", minuit.hesse)
        if np.isfinite(best_fval) and float(minuit.fval) > best_fval:
            self._log("restoring the best accepted Minuit stage because a later "
                      "stage worsened the NLL")
            for name, value in best_values.items():
                minuit.values[name] = value
            if hesse:
                stage("HESSE (restored)", minuit.hesse)
        polish_if_invalid()
        if best_result is not None and (
            not np.isfinite(float(minuit.fval)) or float(minuit.fval) > best_fval
        ):
            self._log("returning the best Minuit stage because the final "
                      "continuation worsened the NLL or became non-finite")
            return best_result
        return minuit

    def fit(
        self,
        start_values: Mapping[str, float] | None = None,
        *,
        simplex: bool = False,
        ncall: int | None = None,
        strategy: int = 2,
        hesse: bool = True,
        method: str = "minuit",
        nesterov_max_iter: int = 1000,
        nesterov_gtol: float = 1e-4,
    ):
        """Fit with Minuit or a Nesterov prefit.

        ``ncall`` is passed to SIMPLEX (if requested), each MIGRAD call and
        HESSE (if requested). It is not a total budget for the complete fit;
        strategy 2 performs two MIGRAD calls. Minuit can exceed a stage's
        approximate limit while completing an iteration. ``method="nesterov"``
        returns the fast Nesterov endpoint; ``method="nesterov-minuit"`` uses
        it as the starting point for the ordinary strategy-1/2 Minuit fit.
        Non-finite or worsened Minuit continuations are rejected in favor of
        the Nesterov endpoint; an invalid continuation that does not worsen
        the NLL is returned with its invalid status.
        """
        ncall = self._validate_ncall(ncall)
        strategy = self._validate_strategy(strategy)
        if method not in ("minuit", "nesterov", "nesterov-minuit"):
            raise ValueError("method must be 'minuit', 'nesterov' or 'nesterov-minuit'")
        if method != "minuit" and strategy not in (1, 2):
            raise ValueError("Nesterov methods support strategy 1 or 2")
        if not isinstance(hesse, bool):
            raise ValueError("hesse must be a boolean")
        self._validate_start_values(start_values)
        prefit = None
        if method in ("nesterov", "nesterov-minuit"):
            prefit = nesterov_minimize(
                self.objective, self.parameters, start_values=start_values,
                max_iter=nesterov_max_iter, gtol=nesterov_gtol, verbose=self.verbose,
            )
            self._log(f"Nesterov prefit finished: status={prefit.status} "
                      f"NLL={prefit.fval:.9f} nfcn={prefit.nfcn}")
            if method == "nesterov":
                return prefit
            start_values = prefit.values
        free, names, fcn, grad, hessian_callback = self._backend()
        self._log(
            f"single fit with {len(free)} free parameters "
            f"(method={method}, simplex={simplex}, strategy={strategy}, hesse={hesse}, "
            f"hessian={self.hessian}, ncall={ncall}, tolerance={self.tolerance})"
        )
        result = self._run(
            free,
            names,
            fcn,
            grad,
            hessian_callback,
            start_values=start_values,
            strategy=strategy,
            hesse=hesse,
            simplex=simplex,
            ncall=ncall,
        )
        if (prefit is not None and
                (not np.isfinite(float(result.fval)) or
                 float(result.fval) > float(prefit.fval))):
            self._log("MIGRAD produced a non-finite or worsened continuation; "
                      "returning the Nesterov endpoint")
            return prefit
        if prefit is not None and not bool(result.valid):
            # An invalid MIGRAD (call limit, EDM above target, failed error
            # matrix) that still lowered the NLL is a better point than the
            # Nesterov endpoint, which carries no EDM or covariance check of
            # its own. Return it with its invalid status, as method="minuit"
            # does, instead of discarding the improvement.
            self._log("MIGRAD continuation is invalid but did not worsen the "
                      "Nesterov NLL; returning it with its invalid status")
        self._log(f"single fit finished: {self._summary(result)}")
        return result

    @staticmethod
    def _draw_parameter(parameter: Parameter, rng: np.random.Generator) -> float:
        if parameter.bounds is not None:
            low, high = parameter.bounds
            if low is not None and high is not None:
                return float(rng.uniform(low, high))

        scale = (
            10.0 * parameter.step
            if parameter.step is not None
            else max(abs(float(parameter.value)), 1.0) * 0.25
        )
        value = float(rng.normal(float(parameter.value), scale))
        if parameter.bounds is not None:
            low, high = parameter.bounds
            if low is not None:
                value = max(value, float(low))
            if high is not None:
                value = min(value, float(high))
        return value

    def random_start(self, *, seed: int | None = None) -> dict[str, float]:
        """Draw a random starting point for every free parameter.

        Uniform within bounds when both are set, otherwise Gaussian around the
        parameter's current value and clipped to any one-sided bound.
        """

        rng = np.random.default_rng(seed)
        return {
            parameter.name: self._draw_parameter(parameter, rng)
            for parameter in self.parameters
            if not parameter.fixed
        }

    def fit_multistart(
        self,
        n_starts: int = 20,
        *,
        seed: int | None = None,
        include_default: bool = False,
        simplex: bool = False,
        strategy: int = 1,
    ) -> MultiStartResult:
        """Run independent minimizations from random starts and refine the best.

        Each start is fit without HESSE; the valid minimum with the lowest
        ``fval`` is then re-run with the requested ``strategy`` and HESSE
        enabled. Raises if no start converges to a valid, finite minimum.
        """

        if n_starts < 1:
            raise ValueError("n_starts must be at least 1")
        strategy = self._validate_strategy(strategy)

        free, names, fcn, grad, hessian_callback = self._backend()
        rng = np.random.default_rng(seed)
        starts: list[dict[str, float]] = []
        if include_default:
            starts.append(
                {parameter.name: float(parameter.value) for parameter in free}
            )
        while len(starts) < n_starts:
            starts.append(
                {
                    parameter.name: self._draw_parameter(parameter, rng)
                    for parameter in free
                }
            )

        self._log(
            f"multistart with {len(free)} free parameters, {len(starts)} starts "
            f"(simplex={simplex}, strategy={strategy}, seed={seed})"
        )
        results = []
        for index, start in enumerate(starts, start=1):
            self._log(f"start {index}/{len(starts)} running")
            result = self._run(
                free,
                names,
                fcn,
                grad,
                hessian_callback,
                start_values=start,
                strategy=strategy,
                hesse=False,
                simplex=simplex,
            )
            results.append(result)
            self._log(f"start {index}/{len(starts)} finished: {self._summary(result)}")
        results = tuple(results)

        valid_indices = tuple(
            index
            for index, result in enumerate(results)
            if bool(result.valid) and np.isfinite(float(result.fval))
        )
        if not valid_indices:
            raise RuntimeError(
                "No valid finite minimum was found across the multistart scan"
            )

        best_index = min(valid_indices, key=lambda index: float(results[index].fval))
        preliminary = results[best_index]
        self._log(
            f"selected start {best_index + 1}/{len(starts)} for strategy-2/HESSE "
            f"refinement: {self._summary(preliminary)}"
        )
        best_values = {name: float(preliminary.values[name]) for name in names}
        best = self._run(
            free,
            names,
            fcn,
            grad,
            hessian_callback,
            start_values=best_values,
            strategy=2,
            hesse=True,
            simplex=False,
        )
        if not bool(best.valid) or not np.isfinite(float(best.fval)):
            raise RuntimeError("Best multistart solution is invalid after HESSE")
        self._log(f"final refined minimum: {self._summary(best)}")

        return MultiStartResult(best=best, results=results, starts=tuple(starts))
