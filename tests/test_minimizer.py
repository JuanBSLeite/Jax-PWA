import math

import numpy as np
import pytest

from jaxpwa import Minimizer, Parameter, enable_x64

enable_x64()


def test_minimizer_recovers_quadratic_minimum_from_displaced_start():
    parameters = (
        Parameter("x", -3.0, bounds=(-5.0, 5.0), step=0.1),
        Parameter("y", 4.0, bounds=(-5.0, 5.0), step=0.1),
    )

    def objective(values):
        return 0.5 * ((values["x"] - 1.25) / 0.3) ** 2 + 0.5 * (
            (values["y"] + 0.75) / 0.5
        ) ** 2

    result = Minimizer(objective, parameters).fit()
    assert result.valid
    assert math.isclose(float(result.values["x"]), 1.25, abs_tol=1e-8)
    assert math.isclose(float(result.values["y"]), -0.75, abs_tol=1e-8)


def test_minimizer_exposes_strategy_and_optional_hesse():
    parameter = Parameter("x", -2.0, bounds=(-5.0, 5.0), step=0.1)

    def objective(values):
        return (values["x"] - 1.5) ** 2

    result = Minimizer(objective, (parameter,)).fit(strategy=1, hesse=False)
    assert result.strategy == 1
    # A single strategy-1 MIGRAD without HESSE is intentionally a lightweight
    # fit path. Check that it reaches the expected minimum without requiring the
    # tighter numerical agreement of the default strategy-2 refinement.
    assert math.isclose(float(result.values["x"]), 1.5, abs_tol=2e-5)


def test_minimizer_can_run_nesterov_as_a_prefit_or_standalone():
    parameters = (
        Parameter("x", -2.0, bounds=(-5.0, 5.0), step=0.1),
        Parameter("y", 2.0, bounds=(-5.0, 5.0), step=0.1),
    )

    def objective(values):
        return (values["x"] - 1.5) ** 2 + (values["y"] + 0.75) ** 2

    standalone = Minimizer(objective, parameters).fit(
        method="nesterov", strategy=1, nesterov_max_iter=300,
    )
    assert standalone.optimizer == "nesterov"
    assert standalone.fval < 1e-6
    assert math.isclose(float(standalone.values["x"]), 1.5, abs_tol=6e-4)

    refined = Minimizer(objective, parameters).fit(
        method="nesterov-minuit", strategy=2, hesse=True,
        nesterov_max_iter=50,
    )
    assert float(refined.fval) < 1e-10
    assert math.isclose(float(refined.values["y"]), -0.75, abs_tol=1e-6)


def test_nesterov_minuit_keeps_an_invalid_but_improved_migrad_continuation():
    from jaxpwa.fit.nesterov import NesterovResult

    parameters = (
        Parameter("x", -1.5, bounds=(-5.0, 5.0), step=0.1),
        Parameter("y", 2.0, bounds=(-5.0, 5.0), step=0.1),
    )

    def rosenbrock(values):
        return (1.0 - values["x"]) ** 2 + 100.0 * (values["y"] - values["x"] ** 2) ** 2

    prefit = Minimizer(rosenbrock, parameters).fit(
        method="nesterov", strategy=1, nesterov_max_iter=3,
    )
    # A tiny call budget stops MIGRAD at its call limit (invalid) after it has
    # already lowered the NLL below the unconverged Nesterov endpoint.
    result = Minimizer(rosenbrock, parameters).fit(
        method="nesterov-minuit", strategy=1, hesse=False,
        nesterov_max_iter=3, ncall=15,
    )
    assert not isinstance(result, NesterovResult)
    assert not result.valid
    assert float(result.fval) < float(prefit.fval)


def test_nesterov_stops_when_stalled_at_a_log_singularity():
    import jax.numpy as jnp

    from jaxpwa.fit.nesterov import STALL_ITERATIONS, minimize

    # Unbounded below at x=0.2, like a signed-weight NLL whose density goes to
    # zero at a negative-weight event: the gradient diverges, backtracking
    # drives the step to floating-point resolution, and the NLL freezes.
    parameters = (Parameter("x", 1.0, step=0.1), Parameter("y", 0.3, step=0.1))

    def objective(values):
        return jnp.log((values["x"] - 0.2) ** 2 + 1e-30) + (values["y"] - 0.5) ** 2

    result = minimize(objective, parameters, max_iter=300)
    assert result.status == "stalled"
    assert not result.valid
    iterations = len(result.history) - 1
    assert iterations < 300
    frozen = [entry["nll"] for entry in result.history[-STALL_ITERATIONS - 1:]]
    assert len(set(frozen)) == 1


@pytest.mark.parametrize("strategy", [-1, 3, True, 1.5])
def test_minimizer_rejects_invalid_strategy(strategy):
    parameter = Parameter("x", 0.0)

    def objective(values):
        return values["x"] ** 2

    with pytest.raises(ValueError, match="strategy"):
        Minimizer(objective, (parameter,)).fit(strategy=strategy)


def test_minimizer_reuses_compiled_backend_and_shared_value_gradient_point():
    parameter = Parameter("x", 0.0, bounds=(-5.0, 5.0))

    def objective(values):
        return (values["x"] - 2.0) ** 2

    minimizer = Minimizer(objective, (parameter,))
    backend1 = minimizer._backend()
    backend2 = minimizer._backend()
    assert backend1 is backend2

    _, _, fcn, grad, _ = backend1
    assert math.isclose(fcn(1.5), 0.25, abs_tol=1e-12)
    assert np.allclose(grad(1.5), np.asarray([-1.0]), rtol=0.0, atol=1e-12)
    assert math.isclose(fcn(1.5), 0.25, abs_tol=1e-12)


def test_minimizer_reuses_backend_across_instances_for_same_objective():
    parameter = Parameter("x", 0.0, bounds=(-5.0, 5.0))

    def objective(values):
        return (values["x"] - 1.0) ** 2

    first = Minimizer(objective, (parameter,))
    second = Minimizer(objective, (parameter,), tolerance=1e-6, verbose=1)
    assert first._backend()[2] is second._backend()[2]
    assert first._backend()[3] is second._backend()[3]


def test_minimizer_does_not_share_backend_when_fixed_value_changes():
    def objective(values):
        return (values["x"] - values["offset"]) ** 2

    first = Minimizer(
        objective,
        (
            Parameter("x", 0.0, bounds=(-5.0, 5.0)),
            Parameter("offset", 1.0, fixed=True),
        ),
    )
    second = Minimizer(
        objective,
        (
            Parameter("x", 0.0, bounds=(-5.0, 5.0)),
            Parameter("offset", 2.0, fixed=True),
        ),
    )
    assert first._backend() is not second._backend()


def test_multistart_selects_global_minimum_of_multimodal_objective():
    parameters = (
        Parameter("x", -1.4, bounds=(-2.0, 2.0), step=0.05),
    )

    def objective(values):
        x = values["x"]
        return (x**2 - 1.0) ** 2 + 0.08 * (x - 1.0) ** 2

    scan = Minimizer(objective, parameters).fit_multistart(
        n_starts=12,
        seed=731,
        include_default=True,
    )
    assert scan.best.valid
    assert len(scan.results) == 12
    assert len(scan.valid_results) >= 1
    assert float(scan.best.fval) < 1e-10
    assert math.isclose(float(scan.best.values["x"]), 1.0, abs_tol=1e-5)
    assert float(scan.best.fval) <= min(
        float(result.fval) for result in scan.valid_results
    ) + 1e-8


def test_multistart_trial_does_not_depend_on_number_of_later_starts():
    """A Minuit trial must not inherit state from other multistart trials.

    For a fixed seed, the random-start sequence is prefix-stable. Therefore the
    first preliminary trial in scans with 1, 2 and 10 starts is identical. Its
    fitted result must also be identical. If it changes, some state is leaking
    between the Minuit driver and the objective/PDF.
    """

    parameters = (
        Parameter("x", 0.0, bounds=(-3.0, 3.0), step=0.02),
        Parameter("y", 0.0, bounds=(-3.0, 3.0), step=0.02),
        Parameter("z", 0.0, bounds=(-3.0, 3.0), step=0.02),
    )

    def objective(values):
        x, y, z = values["x"], values["y"], values["z"]
        return (
            0.5 * ((x - 0.7) / 0.4) ** 2
            + 0.5 * ((y + 1.1) / 0.6) ** 2
            + 0.5 * ((z - 0.2) / 0.3) ** 2
            + 0.07 * x * y
        )

    minimizer = Minimizer(objective, parameters)
    scans = tuple(
        minimizer.fit_multistart(
            n_starts=n_starts,
            seed=314159,
            include_default=False,
            simplex=False,
        )
        for n_starts in (1, 2, 10)
    )

    reference_start = scans[0].starts[0]
    reference_result = scans[0].results[0]
    for scan in scans[1:]:
        assert scan.starts[0] == reference_start
        result = scan.results[0]
        assert bool(result.valid) == bool(reference_result.valid)
        assert math.isclose(
            float(result.fval),
            float(reference_result.fval),
            abs_tol=1e-12,
        )
        for name in ("x", "y", "z"):
            assert math.isclose(
                float(result.values[name]),
                float(reference_result.values[name]),
                abs_tol=1e-12,
            )


def test_shared_callbacks_use_current_limits_defaults_and_steps(monkeypatch):
    from iminuit import Minuit

    def objective(values):
        return (values['x'] - 3.0) ** 2

    first = Minimizer(objective, (Parameter('x', 0.0, bounds=(-5, 5), step=0.1),))
    first._backend()
    second = Minimizer(objective, (Parameter('x', 0.5, bounds=(0, 1), step=0.01),))
    assert second._backend()[2] is first._backend()[2]
    observed = []
    original = Minuit.migrad

    def capture(self, *args, **kwargs):
        observed.append((self.values['x'], self.errors['x'], tuple(self.limits['x'])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Minuit, 'migrad', capture)
    result = second.fit(strategy=1)
    assert observed == [(0.5, 0.01, (0.0, 1.0))]
    assert result.valid
    assert 0.99 < result.values['x'] <= 1.0
    # An explicit start still overrides the new default, not its bounds.
    second.fit(start_values={'x': 0.25}, strategy=1)
    assert observed[1][0] == 0.25


@pytest.mark.parametrize('strategy', [1, 2])
@pytest.mark.parametrize('ncall', [None, 321])
def test_ncall_is_forwarded_to_every_requested_stage(monkeypatch, strategy, ncall):
    from iminuit import Minuit

    calls = []
    for stage in ('simplex', 'migrad', 'hesse'):
        original_stage = getattr(Minuit, stage)

        def record(self, *, _stage=stage, _original=original_stage, **kwargs):
            calls.append((_stage, kwargs))
            return _original(self, **kwargs)
        monkeypatch.setattr(Minuit, stage, record)

    minimizer = Minimizer(lambda v: v['x'] ** 2, (Parameter('x', 1.0),))
    minimizer.fit(ncall=ncall, strategy=strategy, simplex=True, hesse=True)
    assert calls == (
        [('simplex', {'ncall': ncall})]
        + [('migrad', {'ncall': ncall, 'use_simplex': False})] * strategy
        + [('hesse', {'ncall': ncall})]
    )


@pytest.mark.parametrize("errordef", [0.5, 1.0])
def test_jax_hessian_correlated_covariance_with_bounds_and_fixed_parameter(errordef):
    parameters = (
        Parameter("y", -0.2, bounds=(-4.0, 4.0), step=0.1),
        Parameter("offset", 1.2, fixed=True),
        Parameter("x", 0.5, bounds=(0.0, None), step=0.1),
    )

    def objective(values):
        x = values["x"] - values["offset"]
        y = values["y"] + 0.7
        return 0.5 * (4.0 * x*x + 2.0 * x*y + 3.0 * y*y)

    minimizer = Minimizer(
        objective, parameters, hessian="jax", errordef=errordef
    )
    result = minimizer.fit()
    assert result.valid
    assert result.fmin.has_accurate_covar
    assert result.nhessian > 0
    np.testing.assert_allclose(list(result.values), [-0.7, 1.2], atol=2e-6)
    # External coordinates in parameter order (y, x), including errordef scaling.
    expected = 2 * errordef * np.linalg.inv([[3.0, 1.0], [1.0, 4.0]])
    np.testing.assert_allclose(result.covariance, expected, rtol=2e-5)
    numerical = Minimizer(objective, parameters, errordef=errordef).fit()
    np.testing.assert_allclose(result.covariance, numerical.covariance, rtol=2e-5)
    assert result.nfcn < numerical.nfcn


def test_hessian_cache_tracks_point_and_shares_compilation_across_modes():
    def objective(v):
        return v["x"] ** 4 + v["offset"] * v["x"] ** 2

    params = (Parameter("x", 1.0), Parameter("offset", 3.0, fixed=True))
    first = Minimizer(objective, params)
    second = Minimizer(objective, params, hessian="jax")
    hessian = first._backend()[4]
    assert hessian is second._backend()[4]
    np.testing.assert_allclose(hessian(1.0), [[18.0]])
    np.testing.assert_allclose(hessian(2.0), [[54.0]])
    np.testing.assert_allclose(hessian(1.0), [[18.0]])


@pytest.mark.parametrize("floating_dynamics", [False, True])
def test_g2_computes_only_diagonal_and_reuses_curvature_caches(
    floating_dynamics, monkeypatch,
):
    import jax

    factory = Parameter.dynamics if floating_dynamics else Parameter.coefficient
    parameters = tuple(
        factory(f"shape.{name}", value, owner="shape")
        for name, value in (("z", 3.0), ("x", 1.0), ("y", 2.0))
    ) + (Parameter("offset", 4.0, fixed=True),)

    def objective(values):
        x, y, z = (values[f"shape.{name}"] for name in ("x", "y", "z"))
        return x**2 + x * y + y**3 + y * z + values["offset"] * z**4

    minimizer = Minimizer(objective, parameters, hessian="jax")
    hessian = minimizer._backend()[4]
    shared = Minimizer(objective, parameters)._backend()[4]
    assert hessian.diagonal is shared.diagonal

    transfers = []
    device_get = jax.device_get

    def record_transfer(value):
        transfers.append(value.shape)
        return device_get(value)

    monkeypatch.setattr(jax, "device_get", record_transfer)
    diagonal = hessian.diagonal(3.0, 1.0, 2.0)
    np.testing.assert_allclose(diagonal, [432.0, 2.0, 12.0], atol=1e-12)
    assert transfers == [(3,)]  # One execution; no full matrix/column transfers.
    assert hessian.diagonal(3.0, 1.0, 2.0) is diagonal
    assert transfers == [(3,)]

    np.testing.assert_allclose(hessian.diagonal(2.0, 1.0, 1.0), [192.0, 2.0, 6.0])
    assert transfers == [(3,), (3,)]
    # A diagonal-only cache must not masquerade as a full Hessian.
    matrix = hessian(2.0, 1.0, 1.0)
    np.testing.assert_allclose(
        matrix, [[192.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 6.0]],
    )
    transfers.clear()
    np.testing.assert_allclose(hessian.diagonal(2.0, 1.0, 1.0), np.diag(matrix))
    assert not transfers  # Reuse an already-computed full Hessian.
    np.testing.assert_allclose(hessian.diagonal(3.0, 1.0, 2.0), diagonal)
    assert transfers == [(3,)]  # Independent point tracking after a full Hessian.


def test_dynamic_hessian_dispatches_columns_as_separate_hvps(monkeypatch):
    """Floating dynamics must not put every Hessian column in one XLA program."""
    import jax

    parameters = (
        Parameter.dynamics("shape.x", 1.0, owner="shape"),
        Parameter.dynamics("shape.y", 2.0, owner="shape"),
    )

    def objective(values):
        x, y = values["shape.x"], values["shape.y"]
        return x**2 + x * y + y**3

    def reject_batched_columns(*args, **kwargs):
        raise AssertionError("floating-dynamics Hessian must not call jax.lax.map")

    monkeypatch.setattr(jax.lax, "map", reject_batched_columns)
    hessian = Minimizer(objective, parameters, hessian="jax")._backend()[4]
    np.testing.assert_allclose(hessian(1.0, 2.0), [[2.0, 1.0], [1.0, 12.0]])



@pytest.mark.parametrize("mode", [None, True, "automatic"])
def test_invalid_hessian_mode(mode):
    with pytest.raises(ValueError, match="hessian"):
        Minimizer(lambda v: v["x"] ** 2, (Parameter("x", 1.0),), hessian=mode)


def test_verbose_reports_optimizer_stages(capsys):
    Minimizer(lambda v: v["x"] ** 2, (Parameter("x", 1.0),), verbose=1).fit()
    output = capsys.readouterr().out
    for stage in ("MIGRAD 1", "MIGRAD 2", "HESSE"):
        assert f"{stage} started" in output
        assert f"{stage} finished in" in output


@pytest.mark.parametrize("interpolation", ["linear", "hermite"])
def test_jax_hessian_through_prepared_qmi_matches_gradient_differences(interpolation):
    import jax.numpy as jnp

    from jaxpwa import (
        QMI,
        DecayChannel,
        DecayModel,
        FitSession,
        NonResonant,
        RealImag,
        Resonance,
    )

    magnitudes = tuple(
        Parameter.dynamics(f"mag{i}", 1.0 + 0.1*i, owner="S") for i in range(4)
    )
    phases = tuple(
        Parameter.dynamics(f"phase{i}", 0.15*i, owner="S") for i in range(4)
    )
    model = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [
            NonResonant(RealImag(1.0, 0.0)),
            Resonance(
                "S", (0, 1), RealImag(1.0, 0.0), mass=1.0, width=0.1, spin=0,
                lineshape=QMI(
                    (0.28, 0.6, 1.0, 1.75), magnitudes, phases,
                    interpolation=interpolation,
                ),
            ),
        ],
        normalization_resolution=12,
    )
    session = FitSession(model, model.generate_phase_space(80, seed=131))
    free, _, _, grad, hessian = session.minimizer(hessian="jax")._backend()
    point = np.array([p.value for p in free])
    matrix = hessian(*point)
    # Probe a different point so G2 must use its own program through the
    # prepared QMI custom VJP rather than just read a cached full matrix.
    shifted = point + 0.01
    diagonal = hessian.diagonal(*shifted)
    np.testing.assert_allclose(diagonal, np.diag(hessian(*shifted)), rtol=2e-10)
    step = 1e-5
    finite_difference = np.column_stack([
        (grad(*(point + step*row)) - grad(*(point - step*row))) / (2*step)
        for row in np.eye(len(free))
    ])
    assert bool(jnp.all(jnp.isfinite(matrix)))
    np.testing.assert_allclose(matrix, finite_difference, rtol=2e-5, atol=2e-7)


def test_jax_hessian_recovers_from_negative_curvature_start():
    def objective(v):
        return v['x']**4 - v['x']**2 + v['y']**2

    result = Minimizer(
        objective, (Parameter('x', 0.1), Parameter('y', 1.0)), hessian='jax'
    ).fit()
    assert result.valid
    assert result.ng2 > 0
    np.testing.assert_allclose(abs(result.values['x']), np.sqrt(0.5), atol=1e-6)
    np.testing.assert_allclose(result.fval, -0.25, atol=1e-10)


@pytest.mark.parametrize('session_name', ['FitSession', 'CPFitSession'])
def test_session_hessian_option_reaches_single_and_multistart_fit(session_name):
    import jaxpwa as dpf

    # Isolate the composition layer with a well-identified correlated objective.
    session_type = getattr(dpf, session_name)

    class Session(session_type):
        @property
        def parameters(self):
            return (Parameter('x', 0.1), Parameter('y', 1.0))

        @property
        def objective(self):
            return lambda v: (v['x'] - 0.3)**2 + (v['y'] + v['x'])**2

    session = object.__new__(Session)
    result = session.fit(hessian='jax', strategy=1)
    assert result.valid
    assert result.nhessian > 0
    scan = session.fit_multistart(
        n_starts=1,
        include_default=True,
        hessian='jax',
    )
    assert scan.best.valid
    assert scan.best.nhessian > 0


def test_jax_hessian_returns_exact_objective_curvature():
    parameters = (
        Parameter("x", 0.4),
        Parameter("fixed", 2.0, fixed=True),
        Parameter("y", -0.3),
    )

    def objective(values):
        x = values["x"]
        y = values["y"]
        return 0.5 * (4.0 * x**2 + 2.0 * x * y + 3.0 * y**2)

    names, hessian = Minimizer(objective, parameters).jax_hessian(
        {"x": 0.2, "y": -0.1}
    )
    assert names == ("x", "y")
    np.testing.assert_allclose(hessian, [[4.0, 1.0], [1.0, 3.0]], rtol=1e-12)
