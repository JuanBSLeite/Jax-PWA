"""Laura++-style accept-reject pseudo-data generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa.amplitude.components import coefficient_value
from jaxpwa.fit import ParameterKind
from jaxpwa.kinematics import (
    PhaseSpaceMC,
    PhaseSpaceSample,
    dalitz_s13_limits,
)


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    result = jnp.ones((size,), dtype=jnp.float64)
    if efficiency is not None:
        result = result * jnp.asarray(efficiency(data))
    if veto is not None:
        result = result * jnp.asarray(veto(data), dtype=result.dtype)
    return result


def _pilot_size(size: int, pool_size: int | None) -> int:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if pool_size is not None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        return int(pool_size)
    return max(20_000, min(100_000, 2 * size))


def _derived_seed(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return (int(seed) + int(offset)) % (2**32)


def _merge_samples(samples: Sequence[PhaseSpaceSample]) -> PhaseSpaceSample:
    samples = tuple(sample for sample in samples if sample.size > 0)
    if not samples:
        raise ValueError("cannot merge an empty toy sample")

    have_momenta = [
        sample.p1 is not None and sample.p2 is not None and sample.p3 is not None
        for sample in samples
    ]
    if any(have_momenta) and not all(have_momenta):
        raise ValueError("all merged samples must consistently contain four-momenta")

    def concat(name: str):
        arrays = [getattr(sample, name) for sample in samples]
        if arrays[0] is None:
            return None
        return jnp.concatenate(arrays, axis=0)

    total = sum(sample.size for sample in samples)
    return PhaseSpaceSample(
        s12=concat("s12"),
        s13=concat("s13"),
        s23=concat("s23"),
        weights=jnp.ones((total,), dtype=samples[0].s12.dtype),
        p1=concat("p1"),
        p2=concat("p2"),
        p3=concat("p3"),
    )


def _empty_sample(
    model,
    seed: int | None = None,
    *,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    sample = model.generate_phase_space(
        1,
        seed=seed,
        include_momenta=include_momenta,
    )
    return PhaseSpaceSample(
        s12=sample.s12[:0],
        s13=sample.s13[:0],
        s23=sample.s23[:0],
        weights=jnp.ones((0,), dtype=sample.s12.dtype),
        p1=None if sample.p1 is None else sample.p1[:0],
        p2=None if sample.p2 is None else sample.p2[:0],
        p3=None if sample.p3 is None else sample.p3[:0],
    )


def _scores(pool: PhaseSpaceSample, density) -> jax.Array:
    values = jnp.asarray(pool.weights) * jnp.asarray(density)
    if values.shape != (pool.size,):
        raise ValueError(
            f"toy density must return one value per event, got {values.shape} "
            f"for {pool.size} events"
        )
    return values


def _generation_shape(shape, data):
    """Evaluate the density with respect to ordinary Dalitz area.

    Ordinary-Dalitz proposals and CP charge-split integrals use this measure.
    A Square-Dalitz density must therefore include its inverse Jacobian,
    exactly as for the fit. Raw
    ``generation_value`` is only valid for proposals uniform in Square Dalitz.
    """
    return shape(data)


def _square_background_pair(shape):
    """Recognize built-in SDP densities, including charge-scaled wrappers."""
    from jaxpwa.background.models import ChargeScaledBackground
    from jaxpwa.square_histograms import SquareDalitzHistogramBackground

    while isinstance(shape, ChargeScaledBackground):
        shape = shape.shape
    if isinstance(shape, SquareDalitzHistogramBackground) and shape.divide_jacobian:
        return shape.pair
    return None


def _background_sampling_value(shape, data):
    if _square_background_pair(shape) is not None:
        return shape.generation_value(data)
    return _generation_shape(shape, data)


def _square_proposal(model, size, seed, pair):
    from jaxpwa.kinematics import square_dalitz_to_invariants

    mp, tp = np.random.default_rng(seed).random((2, size))
    s12, s13, s23 = square_dalitz_to_invariants(
        mp,
        tp,
        mother_mass=model.channel.parent_mass,
        masses=model.channel.daughter_masses,
        pair=pair,
    )
    return PhaseSpaceSample(s12=s12, s13=s13, s23=s23, weights=jnp.ones_like(s12))


def _score_statistics(values: jax.Array, *, include_mean: bool = False):
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(values >= 0.0)
    maximum = jnp.max(values)
    if include_mean:
        valid_host, maximum_host, mean_host = jax.device_get(
            (valid, maximum, jnp.mean(values))
        )
        if not bool(valid_host):
            raise ValueError("toy generation weights must be finite and non-negative")
        return float(maximum_host), float(mean_host)

    valid_host, maximum_host = jax.device_get((valid, maximum))
    if not bool(valid_host):
        raise ValueError("toy generation weights must be finite and non-negative")
    return float(maximum_host)


def _stratified_grid_shape(n_pilot: int) -> tuple[int, int]:
    """Choose a square cell grid with enough pilot occupancy per cell."""

    side = int(np.sqrt(max(int(n_pilot), 1) / 32.0))
    side = min(32, max(2, side))
    return side, side


def _phase_space_cells(
    model,
    sample: PhaseSpaceSample,
    grid_shape: tuple[int, int],
) -> jax.Array:
    """Map physical invariant proposals back to their equal unit-square cells."""

    u_bins, v_bins = grid_shape
    mother_mass = float(model.channel.parent_mass)
    masses = tuple(float(value) for value in model.channel.daughter_masses)
    m1, m2, m3 = masses

    s12 = jnp.asarray(sample.s12)
    s13 = jnp.asarray(sample.s13)
    s_min = (m1 + m2) ** 2
    s_max = (mother_mass - m3) ** 2
    u = (s12 - s_min) / (s_max - s_min)
    low, high = dalitz_s13_limits(
        s12,
        mother_mass=mother_mass,
        masses=masses,
    )
    width = jnp.maximum(high - low, jnp.finfo(s12.dtype).tiny)
    v = (s13 - low) / width

    iu = jnp.clip(jnp.floor(u * u_bins).astype(jnp.int32), 0, u_bins - 1)
    iv = jnp.clip(jnp.floor(v * v_bins).astype(jnp.int32), 0, v_bins - 1)
    return iu * v_bins + iv


def _pilot_local_envelopes(
    model,
    pilot: PhaseSpaceSample,
    pilot_scores: jax.Array,
    *,
    grid_shape: tuple[int, int],
    envelope_safety: float,
) -> np.ndarray:
    """Build strictly positive local envelopes from a uniform pilot sample."""

    cells = np.asarray(
        jax.device_get(_phase_space_cells(model, pilot, grid_shape)), dtype=np.int32
    )
    scores = np.asarray(jax.device_get(pilot_scores), dtype=float)
    n_cells = int(grid_shape[0] * grid_shape[1])
    maxima = np.zeros((n_cells,), dtype=float)
    np.maximum.at(maxima, cells, scores)

    global_max = float(np.max(scores))
    if global_max <= 0.0:
        raise ValueError("toy density is zero over the pilot phase-space sample")

    # An unvisited cell must retain non-zero proposal support. With the
    # occupancy-aware grid this is uncommon; use the conservative global pilot
    # maximum there so no physical region is excluded.
    maxima[maxima <= 0.0] = global_max
    return float(envelope_safety) * maxima


def _update_local_envelopes(
    envelopes: np.ndarray,
    cells: jax.Array,
    scores: jax.Array,
    *,
    envelope_safety: float,
) -> np.ndarray:
    """Raise every exceeded local envelope before a full safe restart."""

    cells_host = np.asarray(jax.device_get(cells), dtype=np.int32)
    scores_host = np.asarray(jax.device_get(scores), dtype=float)
    maxima = np.zeros_like(envelopes)
    np.maximum.at(maxima, cells_host, scores_host)
    return np.maximum(envelopes, float(envelope_safety) * maxima)


def _frozen_model_intensity(model, values: Mapping[str, object]):
    """Build one JIT intensity with scales and coefficients frozen at toy truth.

    For coefficient-only models, reuse the same compact normalization template
    used by FitSession rather than integrating each component separately. This
    turns component-scale preparation into a one-time model cost shared by
    fits and repeated toy generation.
    """

    components = tuple(model.amplitude_model.components)
    has_floating_dynamics = any(
        parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed
        for parameter in model.parameters
    )

    scales = None
    if not has_floating_dynamics:
        template_key = bool(model.normalize_components)
        template = model._fixed_normalization_templates.get(template_key)
        if template is None:
            # One physical point is sufficient for the data side; the compact
            # preparation kernel computes all normalization scales together and
            # stores the reusable fixed-normalization template on the model.
            seed_sample = model.generate_phase_space(
                1,
                seed=781_237,
                include_momenta=False,
            )
            cache = model.prepare_cache(seed_sample)
            scales = jnp.asarray(cache.component_scales)
        else:
            scales = jnp.asarray(template[0])

    resolved = []
    for index, component in enumerate(components):
        scale = (
            jnp.asarray(scales[index])
            if scales is not None
            else jnp.asarray(model._component_scale(component, values))
        )
        coefficient = jnp.asarray(coefficient_value(component.coefficient, values))
        jax.block_until_ready(scale)
        jax.block_until_ready(coefficient)
        resolved.append((component, scale, coefficient))

    def intensity(data):
        total = None
        for component, scale, coefficient in resolved:
            dynamics = jnp.asarray(component.function(data, values))
            term = coefficient * scale * dynamics
            total = term if total is None else total + term
        total = jnp.asarray(total)
        return jnp.real(total * jnp.conj(total))

    return jax.jit(intensity)


def _attach_momenta(
    model, sample: PhaseSpaceSample, seed: int | None
) -> PhaseSpaceSample:
    if sample.p1 is not None:
        return sample
    generator = PhaseSpaceMC(
        model.channel.parent_mass,
        model.channel.daughter_masses,
    )
    return generator.attach_momenta(sample, seed=seed)


def _prepare_component_momenta(model, samples, seed, include_momenta):
    # Preserve accepted orientations; reconstruct only compact components.
    if not include_momenta:
        return [sample.without_momenta() for sample in samples]
    if any(sample.p1 is not None for sample in samples):
        return [
            _attach_momenta(model, sample, _derived_seed(seed, 920_000 + index))
            for index, sample in enumerate(samples)
        ]
    return samples


def _charge_probability(plus, minus):
    total = plus + minus
    if (
        not np.isfinite(plus)
        or not np.isfinite(minus)
        or plus < 0
        or minus < 0
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError(
            "CP component integrals must be finite, non-negative and have positive sum"
        )
    return plus / total


def _accept_reject_component(
    model,
    size: int,
    density_function,
    *,
    seed: int | None,
    pool_size: int | None,
    batch_size: int | None,
    envelope_safety: float,
    max_restarts: int,
    square_dalitz_pair: tuple[int, int] | None = None,
) -> PhaseSpaceSample:
    """Generate one unweighted component with monitored safe envelopes.

    Invariant-only densities use equal cells in the unit Dalitz proposal square
    with a separately monitored envelope per cell. The cell is drawn with
    probability proportional to its envelope, so accepting with score/envelope
    preserves the exact target density. Momentum-dependent custom densities
    retain the original monitored global-envelope fallback. With
    ``square_dalitz_pair`` set, proposals are uniform on the full Square-Dalitz
    domain and the callback supplies density per square area; a monitored
    global envelope avoids the inverse-Jacobian boundary singularity.
    """

    if size <= 0:
        raise ValueError("toy size must be positive")
    if envelope_safety <= 1.0:
        raise ValueError("envelope_safety must be greater than one")
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")

    compiled_density = jax.jit(density_function)
    n_pilot = _pilot_size(size, pool_size)

    # Built-in amplitudes need only Dalitz invariants.  Try the compact proposal
    # first and fall back to full four-vectors only for custom densities that
    # explicitly request momentum fields.
    compact_proposal = square_dalitz_pair is None
    square_momenta = False
    if square_dalitz_pair is not None:
        pilot = _square_proposal(
            model, n_pilot, _derived_seed(seed, 1), square_dalitz_pair
        )
    else:
        pilot = model.generate_phase_space(
            n_pilot,
            seed=_derived_seed(seed, 1),
            include_momenta=False,
        )
    try:
        pilot_scores = _scores(pilot, compiled_density(pilot.as_dict()))
    except KeyError:
        compact_proposal = False
        if square_dalitz_pair is not None:
            square_momenta = True
            pilot = _attach_momenta(model, pilot, _derived_seed(seed, 2))
        else:
            pilot = model.generate_phase_space(
                n_pilot,
                seed=_derived_seed(seed, 1),
                include_momenta=True,
            )
        pilot_scores = _scores(pilot, compiled_density(pilot.as_dict()))

    observed_max, mean_score = _score_statistics(
        pilot_scores,
        include_mean=True,
    )
    if observed_max <= 0.0:
        raise ValueError("toy density is zero over the pilot phase-space sample")

    grid_shape = None
    local_envelopes = None
    if compact_proposal:
        grid_shape = _stratified_grid_shape(n_pilot)
        local_envelopes = _pilot_local_envelopes(
            model,
            pilot,
            pilot_scores,
            grid_shape=grid_shape,
            envelope_safety=envelope_safety,
        )
        estimated_efficiency = mean_score / float(np.mean(local_envelopes))
    else:
        envelope = envelope_safety * observed_max
        estimated_efficiency = mean_score / envelope

    estimated_efficiency = min(max(estimated_efficiency, 1e-4), 1.0)
    if batch_size is None:
        if compact_proposal:
            # Reuse the pilot array shape for proposal-density evaluation.
            # JAX specializes on array shape, so the previous adaptive batch
            # (often 500k after a 100k pilot) compiled the full amplitude twice
            # on the first toy. Local envelopes make smaller repeated batches
            # efficient enough that avoiding this second compilation wins.
            batch_size = int(n_pilot)
        else:
            batch_size = int(
                min(
                    500_000,
                    max(4_096, np.ceil(1.25 * size / estimated_efficiency)),
                )
            )
    elif batch_size <= 0:
        raise ValueError("batch_size must be positive")
    else:
        batch_size = int(batch_size)

    accept_rng = np.random.default_rng(_derived_seed(seed, 913_579))
    accepted: list[PhaseSpaceSample] = []
    n_accepted = 0
    restarts = 0
    proposal_index = 0

    while n_accepted < size:
        proposal_index += 1
        if proposal_index > 10_000:
            raise RuntimeError("accept-reject toy generation did not converge")
        if compact_proposal:
            probabilities = local_envelopes / np.sum(local_envelopes)
            pool, proposal_cells = model.generate_stratified_phase_space(
                batch_size,
                integration_weights=False,
                cell_probabilities=jnp.asarray(probabilities),
                grid_shape=grid_shape,
                seed=_derived_seed(seed, 10_000 + proposal_index),
            )
        elif square_dalitz_pair is not None:
            pool = _square_proposal(
                model,
                batch_size,
                _derived_seed(seed, 10_000 + proposal_index),
                square_dalitz_pair,
            )
            if square_momenta:
                pool = _attach_momenta(
                    model, pool, _derived_seed(seed, 20_000 + proposal_index)
                )
            proposal_cells = None
        else:
            pool = model.generate_phase_space(
                batch_size,
                seed=_derived_seed(seed, 10_000 + proposal_index),
                include_momenta=True,
            )
            proposal_cells = None

        score = _scores(pool, compiled_density(pool.as_dict()))
        if compact_proposal:
            envelope_values = jnp.asarray(local_envelopes, dtype=score.dtype)[
                proposal_cells
            ]
            valid_device = jnp.all(jnp.isfinite(score)) & jnp.all(score >= 0.0)
            exceeded_device = jnp.any(score > envelope_values)
            valid, exceeded = jax.device_get((valid_device, exceeded_device))
            if not bool(valid):
                raise ValueError(
                    "toy generation weights must be finite and non-negative"
                )
            if bool(exceeded):
                restarts += 1
                if restarts > max_restarts:
                    raise RuntimeError(
                        "accept-reject local envelope was exceeded too many times; "
                        "increase pool_size or envelope_safety"
                    )
                local_envelopes = _update_local_envelopes(
                    local_envelopes,
                    proposal_cells,
                    score,
                    envelope_safety=envelope_safety,
                )
                accepted.clear()
                n_accepted = 0
                continue
            denominator = envelope_values
        else:
            batch_max = _score_statistics(score)
            if batch_max > envelope:
                restarts += 1
                if restarts > max_restarts:
                    raise RuntimeError(
                        "accept-reject envelope was exceeded too many times; "
                        "increase pool_size or envelope_safety"
                    )
                envelope = envelope_safety * batch_max
                accepted.clear()
                n_accepted = 0
                continue
            denominator = envelope

        accept_key = jax.random.key(int(accept_rng.integers(0, 2**32, dtype=np.uint32)))
        mask = jax.random.uniform(
            accept_key,
            (pool.size,),
            dtype=score.dtype,
        ) < (score / denominator)
        indices = np.flatnonzero(np.asarray(jax.device_get(mask), dtype=bool))
        if indices.size == 0:
            continue
        needed = size - n_accepted
        selected = indices[:needed]
        accepted.append(pool.take(jnp.asarray(selected, dtype=jnp.int32)))
        n_accepted += int(selected.size)

    toy = _merge_samples(accepted)
    if toy.size != size:
        raise RuntimeError("internal accept-reject toy generation count mismatch")
    return toy


@dataclass(frozen=True)
class ToyBackground:
    """One background component for high-level toy generation."""

    name: str
    shape: object
    fraction: float | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("toy background name must be non-empty")
        if not callable(self.shape):
            raise TypeError("toy background shape must be callable")
        if self.fraction is not None and not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("toy background fraction must lie in [0, 1]")


@dataclass(frozen=True)
class CPToyBackground:
    """One charge-aware background component for CP toy generation."""

    name: str
    plus_shape: object
    minus_shape: object | None = None
    fraction: float | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CP toy background name must be non-empty")
        if not callable(self.plus_shape):
            raise TypeError("CP toy plus_shape must be callable")
        if self.minus_shape is not None and not callable(self.minus_shape):
            raise TypeError("CP toy minus_shape must be callable")
        if self.fraction is not None and not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("CP toy background fraction must lie in [0, 1]")

    @property
    def resolved_minus_shape(self):
        """Effective B- shape: ``minus_shape`` if set, else ``plus_shape``."""

        return self.plus_shape if self.minus_shape is None else self.minus_shape


def _background_weights(backgrounds: Sequence[object]) -> np.ndarray:
    backgrounds = tuple(backgrounds)
    if not backgrounds:
        return np.empty((0,), dtype=float)
    if len(backgrounds) == 1:
        if backgrounds[0].fraction is not None:
            raise ValueError(
                "a single toy background does not need a relative fraction"
            )
        return np.ones((1,), dtype=float)

    explicit = []
    for background in backgrounds[:-1]:
        if background.fraction is None:
            raise ValueError(
                "all toy backgrounds except the last require a relative fraction"
            )
        explicit.append(float(background.fraction))
    if backgrounds[-1].fraction is not None:
        raise ValueError(
            "the last toy background is the remainder and must not define fraction"
        )
    remainder = 1.0 - sum(explicit)
    weights = np.asarray(explicit + [remainder], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError("toy background fractions sum to more than one")
    return weights


def generate_signal_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    seed: int | None = None,
    pool_size: int | None = None,
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    values = {} if parameters is None else parameters
    intensity = _frozen_model_intensity(model, values)

    def signal_density(data):
        return _acceptance(efficiency, veto, data) * intensity(data)

    toy = _accept_reject_component(
        model,
        size,
        signal_density,
        seed=seed,
        pool_size=pool_size,
        batch_size=batch_size,
        envelope_safety=envelope_safety,
        max_restarts=max_restarts,
    )
    if include_momenta:
        toy = _attach_momenta(model, toy, _derived_seed(seed, 900_001))
    return toy if include_momenta else toy.without_momenta()


def generate_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[ToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("backgrounds require signal_fraction < 1")

    rng = np.random.default_rng(seed)
    n_signal = int(rng.binomial(size, float(signal_fraction))) if backgrounds else size
    n_background = size - n_signal
    bg_weights = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_weights)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
    )
    values = {} if parameters is None else parameters
    samples: list[PhaseSpaceSample] = []

    if n_signal > 0:
        intensity = _frozen_model_intensity(model, values)

        def signal_density(data):
            return _acceptance(efficiency, veto, data) * intensity(data)

        samples.append(
            _accept_reject_component(
                model,
                n_signal,
                signal_density,
                seed=_derived_seed(seed, 1),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    for index, (background, count) in enumerate(
        zip(backgrounds, bg_counts, strict=True)
    ):
        if int(count) == 0:
            continue

        def background_density(data, background=background):
            result = jnp.asarray(_background_sampling_value(background.shape, data))
            if veto is not None and background.apply_veto:
                result = result * jnp.asarray(veto(data))
            return result

        samples.append(
            _accept_reject_component(
                model,
                int(count),
                background_density,
                square_dalitz_pair=_square_background_pair(background.shape),
                seed=_derived_seed(seed, 100 + index),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    samples = _prepare_component_momenta(model, samples, seed, include_momenta)
    toy = _merge_samples(samples)
    if shuffle and toy.size > 1:
        toy = toy.take(
            jax.random.permutation(
                jax.random.key(100 if seed is None else (int(seed) + 100) % (2**32)),
                toy.size,
            )
        )
        toy = PhaseSpaceSample(
            s12=toy.s12,
            s13=toy.s13,
            s23=toy.s23,
            weights=jnp.ones((toy.size,), dtype=toy.s12.dtype),
            p1=toy.p1,
            p2=toy.p2,
            p3=toy.p3,
        )
    if include_momenta:
        toy = _attach_momenta(model, toy, _derived_seed(seed, 900_002))
    return toy


def _integral(model, parameters_or_intensity, efficiency, veto) -> float:
    """Integrate accepted signal density for both toy-generation backends."""

    if callable(parameters_or_intensity):
        intensity = parameters_or_intensity
    else:
        parameters = parameters_or_intensity

        def intensity(data):
            return model.intensity(data, parameters)

    sample = model.normalization_sample
    data = sample.as_dict()
    value = jnp.mean(
        jnp.asarray(sample.weights)
        * _acceptance(efficiency, veto, data)
        * jnp.asarray(intensity(data))
    )
    return float(jax.device_get(value))


def generate_cp_toy(
    plus_model,
    minus_model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    plus_efficiency=None,
    minus_efficiency=None,
    plus_veto=None,
    minus_veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[CPToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one CP background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("CP backgrounds require signal_fraction < 1")

    values = {} if parameters is None else parameters
    rng = np.random.default_rng(seed)
    n_signal = int(rng.binomial(size, float(signal_fraction))) if backgrounds else size
    n_background = size - n_signal

    plus_intensity = _frozen_model_intensity(plus_model, values)
    minus_intensity = _frozen_model_intensity(minus_model, values)
    n_signal_plus = 0
    if n_signal:
        i_plus = _integral(plus_model, plus_intensity, plus_efficiency, plus_veto)
        i_minus = _integral(minus_model, minus_intensity, minus_efficiency, minus_veto)
        n_signal_plus = int(
            rng.binomial(n_signal, _charge_probability(i_plus, i_minus))
        )
    n_signal_minus = n_signal - n_signal_plus

    bg_mix = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_mix)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
    )
    plus_samples: list[PhaseSpaceSample] = []
    minus_samples: list[PhaseSpaceSample] = []

    if n_signal_plus > 0:

        def plus_signal_density(data):
            return _acceptance(plus_efficiency, plus_veto, data) * plus_intensity(data)

        plus_samples.append(
            _accept_reject_component(
                plus_model,
                n_signal_plus,
                plus_signal_density,
                seed=_derived_seed(seed, 10),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    if n_signal_minus > 0:

        def minus_signal_density(data):
            return _acceptance(minus_efficiency, minus_veto, data) * minus_intensity(
                data
            )

        minus_samples.append(
            _accept_reject_component(
                minus_model,
                n_signal_minus,
                minus_signal_density,
                seed=_derived_seed(seed, 20),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    for index, (background, count) in enumerate(
        zip(backgrounds, bg_counts, strict=True)
    ):
        if int(count) == 0:
            continue
        plus_norm_sample = plus_model.normalization_sample
        minus_norm_sample = minus_model.normalization_sample
        plus_norm_data = plus_norm_sample.as_dict()
        minus_norm_data = minus_norm_sample.as_dict()
        j_plus_values = jnp.asarray(
            _generation_shape(background.plus_shape, plus_norm_data)
        )
        j_minus_values = jnp.asarray(
            _generation_shape(background.resolved_minus_shape, minus_norm_data)
        )
        if background.apply_veto:
            if plus_veto is not None:
                j_plus_values = j_plus_values * jnp.asarray(plus_veto(plus_norm_data))
            if minus_veto is not None:
                j_minus_values = j_minus_values * jnp.asarray(
                    minus_veto(minus_norm_data)
                )
        j_plus = float(jnp.mean(jnp.asarray(plus_norm_sample.weights) * j_plus_values))
        j_minus = float(
            jnp.mean(jnp.asarray(minus_norm_sample.weights) * j_minus_values)
        )
        plus_probability = _charge_probability(j_plus, j_minus)
        count_plus = int(rng.binomial(int(count), plus_probability))
        count_minus = int(count) - count_plus

        if count_plus > 0:

            def plus_background_density(data, background=background):
                result = jnp.asarray(
                    _background_sampling_value(background.plus_shape, data)
                )
                if background.apply_veto and plus_veto is not None:
                    result = result * jnp.asarray(plus_veto(data))
                return result

            plus_samples.append(
                _accept_reject_component(
                    plus_model,
                    count_plus,
                    plus_background_density,
                    square_dalitz_pair=_square_background_pair(background.plus_shape),
                    seed=_derived_seed(seed, 100 + index),
                    pool_size=pool_size,
                    batch_size=batch_size,
                    envelope_safety=envelope_safety,
                    max_restarts=max_restarts,
                )
            )

        if count_minus > 0:

            def minus_background_density(data, background=background):
                result = jnp.asarray(
                    _background_sampling_value(background.resolved_minus_shape, data)
                )
                if background.apply_veto and minus_veto is not None:
                    result = result * jnp.asarray(minus_veto(data))
                return result

            minus_samples.append(
                _accept_reject_component(
                    minus_model,
                    count_minus,
                    minus_background_density,
                    square_dalitz_pair=_square_background_pair(
                        background.resolved_minus_shape
                    ),
                    seed=_derived_seed(seed, 200 + index),
                    pool_size=pool_size,
                    batch_size=batch_size,
                    envelope_safety=envelope_safety,
                    max_restarts=max_restarts,
                )
            )

    def finish(
        samples: list[PhaseSpaceSample], model, key_seed: int
    ) -> PhaseSpaceSample:
        if not samples:
            return _empty_sample(
                model,
                _derived_seed(seed, 500 + key_seed),
                include_momenta=include_momenta,
            )
        samples = _prepare_component_momenta(
            model, samples, _derived_seed(seed, key_seed), include_momenta
        )
        toy = _merge_samples(samples)
        if shuffle and toy.size > 1:
            toy = toy.take(
                jax.random.permutation(
                    jax.random.key(
                        (200 if seed is None else int(seed) + 200 + key_seed) % (2**32)
                    ),
                    toy.size,
                )
            )
            toy = PhaseSpaceSample(
                s12=toy.s12,
                s13=toy.s13,
                s23=toy.s23,
                weights=jnp.ones((toy.size,), dtype=toy.s12.dtype),
                p1=toy.p1,
                p2=toy.p2,
                p3=toy.p3,
            )
        if include_momenta:
            toy = _attach_momenta(
                model,
                toy,
                _derived_seed(seed, 910_000 + key_seed),
            )
        return toy

    plus_toy = finish(plus_samples, plus_model, 0)
    minus_toy = finish(minus_samples, minus_model, 1)
    if plus_toy.size + minus_toy.size != size:
        raise RuntimeError("internal CP toy generation count mismatch")
    return plus_toy, minus_toy


__all__ = [
    "CPToyBackground",
    "ToyBackground",
    "generate_cp_toy",
    "generate_signal_toy",
    "generate_toy",
]
