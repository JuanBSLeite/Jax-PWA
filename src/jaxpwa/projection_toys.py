"""Reusable generated toys for CP projection plots (never integration samples)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import numpy as np

from jaxpwa.kinematics import PhaseSpaceSample
from jaxpwa.likelihood.cp import _signal_yield_pair
from jaxpwa.toy_api import ToyBackground, generate_toy


@dataclass(frozen=True)
class CPProjectionToy:
    """Per-charge signal/background toys with constant projection weights.

    Created by ``CPFitSession.prepare_projection_toy``. Arrays reside on the
    host; no fit session, optimizer or prepared generator is kept alive. Reuse
    only with the originating session and fitted values. Delete this object
    after the last plot to release its samples.
    """

    plus_components: tuple
    minus_components: tuple
    parameters: tuple[tuple[str, float], ...]
    method: str
    _session_ref: object = field(repr=False, compare=False)

    def _check(self, session, values):
        if self._session_ref() is not session:
            raise ValueError("projection_toy belongs to a different fit session")
        if tuple(sorted(values.items())) != self.parameters:
            raise ValueError("fitted values changed; prepare a new projection_toy")


def _prepare_cp_projection_toy(session, result, size, seed, method, options):
    from jaxpwa.cp_workflow import CPBackgroundSpec, _resolve

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("projection_size must be a positive integer")
    if method not in ("inverse-transform", "accept-reject"):
        raise ValueError("method must be 'inverse-transform' or 'accept-reject'")
    allowed = {
        "pool_size",
        "batch_size",
        "envelope_safety",
        "max_restarts",
        "inverse_resolution",
        "inverse_quantile_resolution",
        "include_momenta",
    }
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"unsupported toy options: {sorted(unknown)}")
    if method == "inverse-transform" and any(
        options.get(key) is not None for key in ("pool_size", "batch_size")
    ):
        raise ValueError("pool_size/batch_size require method='accept-reject'")
    options = {"include_momenta": False, **options}
    values = session.result_values(result)
    categories = session.background_categories
    if session.extended:
        plus, minus, standalone = _signal_yield_pair(session.signal_yield, values)
        signal_scales = np.asarray([plus, minus], dtype=float)
        background_scales = np.asarray(
            [_resolve(c.yield_, values) for c in categories], dtype=float
        )
    else:
        total = session._total_data_events()
        fraction = (
            float(_resolve(session.signal_fraction, values)) if categories else 1.0
        )
        signal_scales = np.full(2, total * fraction)
        standalone = False
        background_scales = (
            total
            * (1.0 - fraction)
            * np.asarray(session.base_objective.background_weights(values), dtype=float)
            if categories
            else np.empty(0)
        )
    for scales in (signal_scales, background_scales):
        if np.any(~np.isfinite(scales) | (scales < 0)):
            raise ValueError("projection yields must be finite and non-negative")
    if not standalone and np.any(signal_scales):
        integrals = np.asarray(
            [
                session.plus_cache.evaluate(values)[1],
                session.minus_cache.evaluate(values)[1],
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(integrals) | (integrals < 0)) or integrals.sum() <= 0:
            raise ValueError(
                "signal projection requires positive finite joint integral"
            )
        signal_scales *= integrals / integrals.sum()

    rng = np.random.default_rng(seed)
    charges = []
    for charge_index, (model, efficiency, veto) in enumerate(
        (
            (session.plus_model, session.plus_efficiency, session.plus_veto),
            (session.minus_model, session.minus_efficiency, session.minus_veto),
        )
    ):
        specs = [("signal", signal_scales[charge_index], None)]
        for source, category, scale in zip(
            session.backgrounds, categories, background_scales, strict=True
        ):
            probability = float(
                category.plus_probability
                if charge_index == 0
                else category.minus_probability
            )
            if not np.isfinite(probability) or probability < 0:
                raise ValueError(
                    "background charge probabilities must be finite and non-negative"
                )
            if scale * probability == 0:
                continue
            if not isinstance(source, CPBackgroundSpec):
                raise ValueError(
                    "toy projections require CPBackgroundSpec with evaluable shapes"
                )
            shape = (
                source.plus_shape if charge_index == 0 else source.resolved_minus_shape
            )
            specs.append(
                (
                    category.name,
                    scale * probability,
                    ToyBackground(
                        category.name,
                        shape,
                        apply_veto=source.apply_veto,
                    ),
                )
            )
        specs = [spec for spec in specs if spec[1] > 0]
        if size < len(specs):
            raise ValueError(
                "projection_size must allow at least one toy per positive component"
            )
        components = []
        if specs:
            # Stratify by signal/background, preserving their exact fitted
            # yields even when integer toy counts must be rounded.
            scales = np.asarray([spec[1] for spec in specs])
            allocations = (size - len(specs)) * scales / scales.sum()
            counts = 1 + np.floor(allocations).astype(int)
            remaining = int(size - counts.sum())
            counts[np.argsort(-(allocations - np.floor(allocations)))[:remaining]] += 1
            for (name, scale, background), count in zip(specs, counts, strict=True):
                toy = generate_toy(
                    model,
                    int(count),
                    parameters=values,
                    efficiency=efficiency,
                    veto=veto,
                    signal_fraction=1.0 if background is None else 0.0,
                    backgrounds=() if background is None else (background,),
                    seed=int(rng.integers(0, 2**32)),
                    method=method,
                    **options,
                )
                # Retain compact host arrays across plots, not GPU buffers or
                # the CDF/envelope used to generate them.
                host = PhaseSpaceSample(
                    **{
                        key: None
                        if getattr(toy, key) is None
                        else np.asarray(getattr(toy, key))
                        for key in ("s12", "s13", "s23", "weights", "p1", "p2", "p3")
                    }
                )
                components.append((name, host, np.full(host.size, scale / host.size)))
                del toy
        charges.append(tuple(components))
    return CPProjectionToy(
        charges[0],
        charges[1],
        tuple(sorted(values.items())),
        method,
        weakref.ref(session),
    )
