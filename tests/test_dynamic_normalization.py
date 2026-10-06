"""Floating-dynamics normalization must preserve global scales, interference and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    QMI,
    CPRealImag,
    DecayChannel,
    DecayModel,
    GounarisSakurai,
    Parameter,
    RealImag,
    RelativisticBreitWigner,
    Resonance,
    ZemachP,
)
from jaxpwa.amplitude.cache import _hermitian_matrix_sum
from jaxpwa.likelihood import CPJointNLL


def test_hermitian_matrix_sum_integrates_only_unique_component_pairs(monkeypatch):
    reductions = []
    einsum = jnp.einsum

    def record_reduction(expression, *arrays, **kwargs):
        assert expression in ("n,nj->j", "n,nd->d")
        reductions.append(arrays[1].shape[1])
        return einsum(expression, *arrays, **kwargs)

    monkeypatch.setattr(jnp, "einsum", record_reduction)
    values = jnp.arange(28).reshape((7, 4)) * (1.0 + 0.3j)
    matrix = _hermitian_matrix_sum(values, jnp.ones(7))
    assert sum(reductions) == 4 * (4 + 1) // 2
    np.testing.assert_allclose(matrix, np.asarray(values).conj().T @ values)


@pytest.mark.parametrize("scale", [0.0, 0.4, 1.0, -0.7])
def test_triangular_normalization_preserves_interference_factor_two(scale):
    # F=(1, t*(1+2i)), c=(1, 2-i): A=1+t*(4+3i), so
    # |A|^2=1+8t+25t^2. A missing/doubled interference factor changes 8t;
    # incorrectly mirroring the diagonal also changes the constant/quadratic.
    weights = jnp.asarray([1.0, 3.0, 0.0, 2.0])
    efficiency = jnp.asarray([0.5, 0.5, 1.0, 0.5])
    coefficients = jnp.asarray([1.0, 2.0 - 1.0j])
    measure = 0.75  # mean(weights * efficiency), not a weight-normalized mean.

    def matrix(t):
        values = jnp.broadcast_to(jnp.asarray([1.0 + 0j, t * (1 + 2j)]), (4, 2))
        return _hermitian_matrix_sum(values, weights * efficiency) / weights.size

    def normalization(t):
        return jnp.real(coefficients.conj() @ matrix(t) @ coefficients)

    actual_matrix = jax.jit(matrix)(scale)
    np.testing.assert_allclose(
        actual_matrix,
        measure * np.asarray([[1, scale*(1+2j)], [scale*(1-2j), 5*scale**2]]),
        rtol=1e-14, atol=1e-14,
    )
    diagonal = jnp.sum(jnp.abs(coefficients)**2 * jnp.real(jnp.diag(actual_matrix)))
    interference = 2 * jnp.real(
        coefficients[0].conj() * actual_matrix[0, 1] * coefficients[1]
    )
    value, gradient = jax.jit(jax.value_and_grad(normalization))(scale)
    curvature = jax.jit(jax.jacfwd(jax.grad(normalization)))(scale)
    np.testing.assert_allclose(diagonal, measure*(1+25*scale**2), rtol=1e-14)
    np.testing.assert_allclose(interference, measure*8*scale, rtol=1e-14)
    np.testing.assert_allclose(value, diagonal + interference, rtol=1e-14)
    np.testing.assert_allclose(gradient, measure*(8+50*scale), rtol=1e-14)
    np.testing.assert_allclose(curvature, measure*50, rtol=1e-14)


@pytest.mark.parametrize("size", [0, 1, 2, 5])
@pytest.mark.parametrize("signed_weights", [False, True])
def test_hermitian_matrix_sum_matches_independent_dense_product(size, signed_weights):
    rng = np.random.default_rng(507)
    values = rng.normal(size=(31, size)) + 1j * rng.normal(size=(31, size))
    weights = rng.uniform(0.3, 1.0, 31)
    weights[::7] = 0.0  # Veto/padded entries must not contribute.
    if signed_weights:
        weights[1::3] *= -1
    actual = np.asarray(jax.jit(_hermitian_matrix_sum)(values, weights))
    expected = values.conj().T @ (weights[:, None] * values)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(actual, actual.conj().T)


@pytest.mark.parametrize("size", [1, 3, 5])
@pytest.mark.parametrize("floating_weights", [False, True])
def test_hermitian_matrix_derivatives_match_direct_coherent_intensity(
    size, floating_weights,
):
    rng = np.random.default_rng(508)
    base = jnp.asarray(rng.normal(size=(31, size)) + 1j * rng.normal(size=(31, size)))
    coefficients = jnp.asarray(rng.normal(size=size) + 1j * rng.normal(size=size))
    coordinate = jnp.linspace(0.1, 1.0, 31)

    def inputs(point):
        scale = 1 + point[0] * coordinate[:, None]
        phase = point[1] * coordinate[:, None] * (1 + jnp.arange(size))
        values = base * scale * jnp.exp(1j * phase)
        weights = jnp.arange(1, 32)  # Also cover constant integer integration weights.
        if floating_weights:
            weights = weights * jnp.exp(point[2] * coordinate)
        return values, weights

    def matrix_objective(point):
        values, weights = inputs(point)
        matrix = _hermitian_matrix_sum(values, weights) / weights.size
        return jnp.real(coefficients.conj() @ matrix @ coefficients)

    def direct_objective(point):
        values, weights = inputs(point)
        return jnp.mean(weights * jnp.abs(values @ coefficients)**2)

    point = jnp.asarray([0.3, -0.5, 0.7])
    for transform in (jax.value_and_grad, lambda f: jax.jacfwd(jax.grad(f))):
        actual = jax.jit(transform(matrix_objective))(point)
        expected = jax.jit(transform(direct_objective))(point)
        for left, right in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)


def make_model(
    *,
    qmi=None,
    all_dynamic=False,
    normalize=True,
    charge=1,
    chunk_size="auto",
):
    mass = Parameter.dynamics("rho.mass", 0.775, owner="rho")
    width = Parameter.dynamics("rho.width", 0.149, owner="rho")
    shape = Parameter.dynamics("s.shape", 0.6, owner="s")
    coefficient = Parameter.coefficient("rho.x", 0.8, owner="rho")
    cp = CPRealImag(coefficient, 0.1, 0.07, -0.03)
    components = [
        Resonance(
            "rho",
            (2, 0),
            cp.for_charge(charge),
            spin=1,
            mass=mass,
            width=width,
            lineshape=GounarisSakurai(),
            angular=ZemachP(),
        ),
        Resonance(
            "s",
            (2, 0),
            RealImag(0.4, -0.2),
            spin=0,
            mass=1.0 if qmi else shape,
            width=0.4,
            lineshape=QMI(
                knots=(0.28, 0.5, 0.9, 1.7),
                real_parts=(1.0, shape, -0.3, 0.5),
                imaginary_parts=(0.2, -0.1, 0.4, 0.1),
                interpolation=qmi,
            )
            if qmi
            else RelativisticBreitWigner(),
            normalize_component=False,
        ),
    ]
    if not all_dynamic:
        components.insert(
            1,
            Resonance(
                "fixed",
                (2, 0),
                RealImag(0.3, 0.4),
                spin=2,
                mass=1.25,
                width=0.17,
                normalize_component=True,
            ),
        )
    return DecayModel(
        DecayChannel("D+", ("pi+", "pi+", "pi-")),
        components,
        normalization_method="square-dalitz",
        normalization_resolution=9,
        normalization_pair=(0, 1),
        normalization_chunk_size=chunk_size,
        normalize_components=normalize,
    )


def prepare_cache(
    *,
    qmi=None,
    all_dynamic=False,
    normalize=True,
    efficiency=True,
    charge=1,
    chunk_size="auto",
):
    """Return ``(model, cache, data, sample, acceptance)`` for a floating model."""
    model = make_model(
        qmi=qmi,
        all_dynamic=all_dynamic,
        normalize=normalize,
        charge=charge,
        chunk_size=chunk_size,
    )
    data = model.generate_phase_space(19, seed=381)
    sample = model.normalization_sample
    acceptance = (
        jnp.where(
            jnp.arange(sample.size) % 7 == 0, 0.0, jnp.linspace(0.3, 0.9, sample.size)
        )
        if efficiency
        else None
    )
    cache = model.prepare_cache(data, sample, efficiency_normalization=acceptance)
    return model, cache, data, sample, acceptance


def mapping(x):
    return dict(zip(("rho.mass", "rho.width", "s.shape", "rho.x"), x, strict=True))


def nll(cache, x):
    intensity, norm = cache.evaluate(mapping(x))
    return -jnp.log(intensity).sum() + intensity.size * jnp.log(norm)


def dense_nll(model, data, sample, acceptance, x):
    """Reference likelihood that never touches ``PreparedAmplitudeCache``.

    Component scales, intensity and normalization integral are recomputed from
    the model on the full data and normalization samples.
    """
    values = mapping(x)
    intensity = model.intensity(data.as_dict(), values)
    norm_intensity = model.intensity(sample.as_dict(), values)
    weights = sample.weights if acceptance is None else sample.weights * acceptance
    norm = jnp.mean(weights * norm_intensity)
    return -jnp.log(intensity).sum() + intensity.size * jnp.log(norm)


@pytest.mark.parametrize(
    "all_dynamic,normalize,efficiency",
    [(True, True, True)],
)
def test_floating_dynamics_values_gradients_and_hessian_match_dense_reference(
    all_dynamic, normalize, efficiency
):
    model, cache, data, sample, acceptance = prepare_cache(
        all_dynamic=all_dynamic, normalize=normalize, efficiency=efficiency
    )
    for point in ([0.775, 0.149, 0.6, 0.8], [0.79, 0.16, 0.62, 0.75]):
        x = jnp.asarray(point)
        expected_fn = lambda x: dense_nll(model, data, sample, acceptance, x)  # noqa: E731
        actual_fn = lambda x: nll(cache, x)  # noqa: E731
        for expected, actual in zip(
            jax.jit(jax.value_and_grad(expected_fn))(x),
            jax.jit(jax.value_and_grad(actual_fn))(x),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(jax.grad(actual_fn)))(x),
            jax.jit(jax.jacfwd(jax.grad(expected_fn)))(x),
            rtol=2e-10,
            atol=1e-9,
        )
        # The compatibility helpers assemble full component matrices; they
        # must agree with the block-wise hot path.
        dense_data, dense_norm = cache._evaluate_components(mapping(x))
        np.testing.assert_allclose(
            cache.amplitude(mapping(x)),
            dense_data @ cache.coefficient_vector(mapping(x)),
            rtol=2e-12,
            atol=1e-12,
        )
        _, dynamic_norm = cache._evaluate_dynamic_components(mapping(x))
        np.testing.assert_allclose(
            cache._matrix_from_dynamic(dynamic_norm),
            cache._matrix_with_dynamic_blocks(dense_norm),
            rtol=2e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("interpolation", ["hermite"])
def test_floating_qmi_second_derivatives_match_dense_reference(interpolation):
    model, cache, data, sample, acceptance = prepare_cache(qmi=interpolation)
    x = jnp.asarray([0.79, 0.16, 0.62, 0.75])
    expected_fn = lambda x: dense_nll(model, data, sample, acceptance, x)  # noqa: E731
    actual_fn = lambda x: nll(cache, x)  # noqa: E731
    np.testing.assert_allclose(
        jax.jit(jax.grad(actual_fn))(x),
        jax.jit(jax.grad(expected_fn))(x),
        rtol=2e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(jax.grad(actual_fn)))(x),
        jax.jit(jax.jacfwd(jax.grad(expected_fn)))(x),
        rtol=2e-9,
        atol=1e-8,
    )
    # The reusable fraction kernel receives its arrays as runtime inputs.
    values = mapping(x)
    kernel = cache._build_fraction_jacobian_kernel(tuple(values))
    actual = kernel(values, cache._fraction_jacobian_arrays())
    expected = jax.jacrev(lambda v: cache.fit_fractions(mapping(v)))(x)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-10)


def test_floating_cp_joint_normalization_gradient_matches_finite_differences():
    plus = prepare_cache(charge=1)[1]
    minus = prepare_cache(charge=-1)[1]
    joint = CPJointNLL(plus, minus)
    x = np.asarray([0.79, 0.16, 0.62, 0.75])
    fn = lambda v: joint(mapping(v))  # noqa: E731
    gradient = np.asarray(jax.jit(jax.grad(fn))(jnp.asarray(x)))
    step = 1e-6
    finite = np.asarray(
        [
            (float(fn(jnp.asarray(x + e))) - float(fn(jnp.asarray(x - e)))) / (2 * step)
            for e in step * np.eye(len(x))
        ]
    )
    np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-6)


# -- chunked normalization ----------------------------------------------------

CHUNK = 17  # far below the ~81-point test grid, so several blocks are scanned


def _assert_matches_dense(model, cache, data, sample, acceptance, *, hessian_rtol):
    for point in ([0.775, 0.149, 0.6, 0.8], [0.79, 0.16, 0.62, 0.75]):
        x = jnp.asarray(point)
        expected_fn = lambda x: dense_nll(model, data, sample, acceptance, x)  # noqa: E731
        actual_fn = lambda x: nll(cache, x)  # noqa: E731
        for expected, actual in zip(
            jax.jit(jax.value_and_grad(expected_fn))(x),
            jax.jit(jax.value_and_grad(actual_fn))(x),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(jax.grad(actual_fn)))(x),
            jax.jit(jax.jacfwd(jax.grad(expected_fn)))(x),
            rtol=hessian_rtol,
            atol=1e-8,
        )


@pytest.mark.parametrize(
    "all_dynamic,normalize,efficiency",
    [
        (False, True, True),
        (False, False, False),
        (True, True, True),
    ],
)
def test_chunked_values_gradients_and_hessian_match_dense_reference(
    all_dynamic, normalize, efficiency
):
    model, cache, data, sample, acceptance = prepare_cache(
        all_dynamic=all_dynamic,
        normalize=normalize,
        efficiency=efficiency,
        chunk_size=CHUNK,
    )
    assert cache.normalization_chunks is not None  # DecayModel threads the option
    assert cache.normalization_chunk_size == CHUNK
    _assert_matches_dense(
        model, cache, data, sample, acceptance, hessian_rtol=2e-10
    )


@pytest.mark.parametrize("interpolation", ["linear", "natural"])
def test_chunked_qmi_second_derivatives_match_dense_reference(interpolation):
    # QMI's prepared order/starts/ends are valid only for their exact block, so
    # each chunk must be prepared on its own rather than sliced afterwards.
    model, cache, data, sample, acceptance = prepare_cache(
        qmi=interpolation, chunk_size=CHUNK
    )
    assert cache.normalization_chunks is not None
    _assert_matches_dense(
        model, cache, data, sample, acceptance, hessian_rtol=2e-9
    )
    x = jnp.asarray([0.79, 0.16, 0.62, 0.75])
    values = mapping(x)
    kernel = cache._build_fraction_jacobian_kernel(tuple(values))
    actual = kernel(values, cache._fraction_jacobian_arrays())
    expected = jax.jacrev(lambda v: cache.fit_fractions(mapping(v)))(x)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-10)


def test_chunked_cp_joint_gradient_matches_finite_differences():
    plus = prepare_cache(charge=1, chunk_size=CHUNK)[1]
    minus = prepare_cache(charge=-1, chunk_size=CHUNK)[1]
    joint = CPJointNLL(plus, minus)
    x = np.asarray([0.79, 0.16, 0.62, 0.75])
    fn = lambda v: joint(mapping(v))  # noqa: E731
    gradient = np.asarray(jax.jit(jax.grad(fn))(jnp.asarray(x)))
    step = 1e-6
    finite = np.asarray(
        [
            (float(fn(jnp.asarray(x + e))) - float(fn(jnp.asarray(x - e)))) / (2 * step)
            for e in step * np.eye(len(x))
        ]
    )
    np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-6)


def test_chunk_size_is_balanced_to_minimize_padding():
    _, cache, _, sample, _ = prepare_cache(chunk_size=41)
    n = sample.size
    blocks = -(-n // 41)
    balanced = -(-n // blocks)
    assert cache.normalization_chunk_size == 41
    assert cache.effective_normalization_chunk_size == balanced
    assert cache.normalization_chunks[1].shape[:2] == (blocks, balanced)
    assert cache.normalization_padding_points == blocks * balanced - n
    assert cache.normalization_padding_points < blocks
    assert cache.normalization_padding_fraction == pytest.approx(
        cache.normalization_padding_points / n
    )


def test_explicit_chunk_size_must_be_positive_integer_or_auto():
    from jaxpwa.amplitude import PreparedAmplitudeCache

    for invalid in (0, -3, True, 1.5, "big"):
        with pytest.raises(ValueError, match="normalization_chunk_size"):
            PreparedAmplitudeCache.prepare(
                (), data={}, normalization_data={}, normalization_weights=jnp.ones(3),
                normalization_chunk_size=invalid,
            )


# -- memory-aware ("auto") chunk size -----------------------------------------


@pytest.fixture
def small_probe(monkeypatch):
    """Let the 81-point test grid take the auto path with tiny probes."""
    from jaxpwa.amplitude import memory

    monkeypatch.setattr(memory, "SINGLE_BLOCK_POINTS", 10)
    monkeypatch.setattr(memory, "PROBE_POINTS", 32)
    monkeypatch.setattr(memory, "MIN_CHUNK_POINTS", 8)
    return memory


def test_device_memory_budget_uses_free_memory(monkeypatch):
    from jaxpwa.amplitude import memory

    class Device:
        def __init__(self, stats):
            self._stats = stats

        def memory_stats(self):
            return self._stats

    monkeypatch.setattr(
        memory.jax, "devices",
        lambda: [Device({"bytes_limit": 1000, "bytes_in_use": 200})],
    )
    assert memory.device_memory_budget() == int(memory.MEMORY_FRACTION * 800)
    for stats in (None, {}, {"bytes_in_use": 5}):
        monkeypatch.setattr(memory.jax, "devices", lambda stats=stats: [Device(stats)])
        assert memory.device_memory_budget() is None


def test_chunk_points_from_budget_is_clipped():
    from jaxpwa.amplitude import memory

    assert memory.chunk_points_from_budget(10**7, 100.0, 1_000_000) == 100_000
    assert memory.chunk_points_from_budget(10**12, 100.0, 5_000) == 5_000
    assert memory.chunk_points_from_budget(0, 100.0, 5_000) == memory.MIN_CHUNK_POINTS
    assert memory.chunk_points_from_budget(1, 0.0, 5_000) == 5_000


def test_auto_chunk_size_follows_measured_bytes_per_point(small_probe, monkeypatch):
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: 500)
    monkeypatch.setattr(small_probe, "tree_nbytes", lambda tree: 0)
    # Pretend the probe needs 50 B/point -> a 500-byte budget allows 10 points.
    monkeypatch.setattr(
        small_probe, "compiled_temp_bytes",
        lambda function, *arguments: 50 * small_probe.PROBE_POINTS,
    )
    _, cache, _, _, _ = prepare_cache(chunk_size="auto")
    assert cache.normalization_chunks is not None
    assert cache.normalization_chunk_size == 10
    assert cache.effective_normalization_chunk_size <= 10


def test_auto_chunk_size_uses_one_block_when_memory_is_plentiful(
    small_probe, monkeypatch
):
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: 10**12)
    _, cache, _, _, _ = prepare_cache(chunk_size="auto")
    assert cache.normalization_chunks is None  # whole grid in one block


def test_auto_chunk_size_probes_real_program_and_stays_exact(small_probe, monkeypatch):
    # No faked measurement: compile the real Hessian-vector-product probe and
    # give it a budget (20 kB) far below what the 81-point grid needs (about
    # 1 kB per point), so several blocks are forced.
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: 20_000)
    model, cache, data, sample, acceptance = prepare_cache(chunk_size="auto")
    assert cache.normalization_chunks is not None
    assert small_probe.MIN_CHUNK_POINTS <= cache.normalization_chunk_size < sample.size
    _assert_matches_dense(
        model, cache, data, sample, acceptance, hessian_rtol=2e-10
    )


def test_auto_chunk_size_falls_back_without_memory_statistics(small_probe, monkeypatch):
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: None)
    calls = []
    monkeypatch.setattr(
        small_probe, "compiled_temp_bytes",
        lambda *args: calls.append(args) or 1,
    )
    _, cache, _, _, _ = prepare_cache(chunk_size="auto")
    assert not calls  # no probe is compiled without a budget
    assert cache.normalization_chunks is None  # 81 points < the 100000 default


def test_auto_chunk_size_probe_runs_once_per_model(small_probe, monkeypatch):
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: 500)
    monkeypatch.setattr(small_probe, "tree_nbytes", lambda tree: 0)
    calls = []

    def fake(function, *arguments):
        calls.append(1)
        return 50 * small_probe.PROBE_POINTS

    monkeypatch.setattr(small_probe, "compiled_temp_bytes", fake)
    model = make_model(chunk_size="auto")
    data = model.generate_phase_space(19, seed=381)
    sample = model.normalization_sample
    first = model.prepare_cache(data, sample)
    second = model.prepare_cache(data, sample)
    assert len(calls) == 1
    assert first.normalization_chunk_size == second.normalization_chunk_size


def test_auto_chunk_size_reserves_the_resident_prepared_blocks(small_probe, monkeypatch):
    monkeypatch.setattr(small_probe, "device_memory_budget", lambda: 10_000)
    monkeypatch.setattr(
        small_probe, "compiled_temp_bytes",
        lambda function, *arguments: 50 * small_probe.PROBE_POINTS,
    )
    # 10 000 B budget, 50 B/point temporaries: no residents -> 200 points (the
    # whole 81-point grid); residents of 9 900 B leave 100 B -> 2 points, which
    # the minimum raises to 8.
    monkeypatch.setattr(small_probe, "tree_nbytes", lambda tree: 0)
    assert prepare_cache(chunk_size="auto")[1].normalization_chunks is None
    n = prepare_cache(chunk_size=10**6)[3].size
    monkeypatch.setattr(
        small_probe, "tree_nbytes",
        lambda tree: int(9_900 * small_probe.PROBE_POINTS / n),
    )
    cache = prepare_cache(chunk_size="auto")[1]
    assert cache.normalization_chunk_size == small_probe.MIN_CHUNK_POINTS
