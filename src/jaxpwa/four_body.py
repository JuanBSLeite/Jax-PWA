"""Four-body model composition over the existing amplitude and fit machinery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import permutations

import jax.numpy as jnp
import numpy as np

from .amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    PreparedAmplitudeCache,
)
from .amplitude.components import coefficient_value
from .decay import _collect_parameters
from .fit import ParameterKind
from .fit.parameters import _fixed_parameter_updates, _replace_fixed_parameters
from .integration import GridIntegrator
from .kinematics.nbody import NBodyPhaseSpaceMC, NBodySample
from .particle_properties import mass_gev, resolve_particle
from .pdf import SignalPDF


@dataclass(frozen=True)
class NBodyDecayChannel:
    """Mass/identity declaration for N labelled daughters (GeV, zero-based order).

    Identity labels are optional: absent labels mean distinguishable daughters.
    They must be supplied for automatic Bose symmetrization. This first model
    layer supports only scalar parents and scalar daughters.
    """

    parent_mass: float
    daughter_masses: tuple[float, ...]
    final_state_ids: tuple[object, ...] | None = None

    def __post_init__(self):
        NBodyPhaseSpaceMC(self.parent_mass, self.daughter_masses)
        if self.final_state_ids is not None:
            if len(self.final_state_ids) != self.nbody:
                raise ValueError("one identity label is required per daughter")
            for i, label in enumerate(self.final_state_ids):
                for j, other in enumerate(self.final_state_ids):
                    if (
                        label == other
                        and self.daughter_masses[i] != self.daughter_masses[j]
                    ):
                        raise ValueError("identical particles must have equal masses")

    @property
    def nbody(self):
        return len(self.daughter_masses)

    @classmethod
    def from_particles(cls, parent, final_state):
        particles = tuple(resolve_particle(name) for name in (parent, *final_state))
        if any(p.J != 0 for p in particles):
            raise ValueError("initial four-body support requires spinless externals")
        return cls(
            mass_gev(parent),
            tuple(mass_gev(p) for p in final_state),
            tuple(int(p.pdgid) for p in particles[1:]),
        )


@dataclass(frozen=True)
class _BoseSymmetrized:
    function: object
    exchanges: tuple[tuple[int, ...], ...]

    @property
    def _prefix(self):
        return f"_bose_{id(self)}:"

    def prepare_data(self, data):
        if any(k.startswith(self._prefix) for k in data):
            return data
        result = dict(data)
        for i, order in enumerate(self.exchanges):
            member = {"momenta": data["momenta"][:, order, :]}
            prepare = getattr(self.function, "prepare_data", None)
            compact = getattr(self.function, "compact_prepared_data", None)
            if prepare is not None:
                member = prepare(member)
            if compact is not None:
                member = compact(member)
            result.update({f"{self._prefix}{i}:{k}": v for k, v in member.items()})
        return result

    def __call__(self, data, parameters=None):
        # Components must derive their kinematics from momenta; never retain
        # stale s_ij/angle arrays after an exchange of labelled particles.
        prepared = self.prepare_data(data)
        total = 0
        for i in range(len(self.exchanges)):
            prefix = f"{self._prefix}{i}:"
            member = {
                k[len(prefix) :]: v for k, v in prepared.items() if k.startswith(prefix)
            }
            total = total + self.function(member, parameters)
        return total / np.sqrt(len(self.exchanges))

    def compact_prepared_data(self, data):
        return {
            k: v
            for k, v in self.prepare_data(data).items()
            if k.startswith(self._prefix)
        }


class FourBodyDecayModel:
    """Four-body coherent model with a fixed weighted MC normalization sample.

    Accepts existing AmplitudeComponents whose functions consume ``momenta``.
    FitSession(model, sample) and the low-level cache/PDF/Minimizer APIs are
    shared with three-body analyses. Dalitz grids, ROOT model serialization,
    CP/time-dependent convenience sessions and inverse-transform toys remain
    three-body-specific. See docs/four_body.md for the supported workflow.
    """

    def __init__(
        self,
        channel,
        components,
        *,
        normalization_sample,
        normalize_components=True,
        symmetrize=True,
        normalization_chunk_size="auto",
    ):
        if not isinstance(channel, NBodyDecayChannel) or channel.nbody != 4:
            raise ValueError(
                "FourBodyDecayModel requires a four-body NBodyDecayChannel"
            )
        self.channel = channel
        self.components = tuple(components)
        if not self.components or not all(
            isinstance(c, AmplitudeComponent) for c in self.components
        ):
            raise ValueError("supply at least one AmplitudeComponent")
        names = [c.name for c in self.components]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("component names must be nonempty and unique")
        self.normalize_components = bool(normalize_components)
        self.symmetrize = bool(symmetrize)
        self.normalization_chunk_size = normalization_chunk_size
        self._auto_chunk_memo = {}
        self._validate_sample(normalization_sample, integration=True)
        self.normalization_sample = normalization_sample
        self._fixed_normalization_templates = {}
        # Construct/validate on the host before any objective is JIT traced.
        _ = self.parameters, self.amplitude_model

    def _validate_sample(self, sample, *, integration=False):
        if not isinstance(sample, NBodySample) or sample.nbody != 4:
            raise TypeError("four-body samples must be NBodySample with four particles")
        sample.validate_physical(self.channel.parent_mass, self.channel.daughter_masses)
        if integration:
            sample.validate_integration()

    @cached_property
    def parameters(self):
        unique = {}
        for component in self.components:
            backend_names = {}
            for parameter in _collect_parameters(component.function):
                key = parameter.backend_name or parameter.name
                if key in backend_names and backend_names[key] != parameter.name:
                    raise ValueError(
                        "dynamics backend names must be unique per component"
                    )
                backend_names[key] = parameter.name
            for parameter in _collect_parameters(component):
                previous = unique.get(parameter.name)
                if previous is not None and previous != parameter:
                    raise ValueError(f"conflicting parameter {parameter.name!r}")
                if parameter.kind is ParameterKind.DYNAMICS:
                    if parameter.owner != component.name:
                        raise ValueError(
                            "dynamics Parameter.owner must match its component"
                        )
                elif (
                    parameter in _collect_parameters(component.function)
                    and not parameter.fixed
                ):
                    raise ValueError(
                        "floating function parameters must have kind DYNAMICS"
                    )
                unique[parameter.name] = parameter
        return tuple(unique.values())

    def with_fixed_parameters(
        self,
        *names: str,
        values: Mapping[str, float] | None = None,
    ) -> FourBodyDecayModel:
        """Return a new model with selected parameters fixed.

        Positional names retain their current values. Entries in ``values``
        are assigned the supplied value and fixed. The original model and its
        parameters remain unchanged.
        """
        updates = _fixed_parameter_updates(self.parameters, names, values)
        components = _replace_fixed_parameters(self.components, updates)
        return FourBodyDecayModel(
            self.channel,
            components,
            normalization_sample=self.normalization_sample,
            normalize_components=self.normalize_components,
            symmetrize=self.symmetrize,
            normalization_chunk_size=self.normalization_chunk_size,
        )

    @cached_property
    def amplitude_model(self):
        ids = self.channel.final_state_ids
        if ids is None or not self.symmetrize:
            return CoherentAmplitudeModel(self.components)
        exchanges = tuple(
            order
            for order in permutations(range(4))
            if all(ids[i] == ids[order[i]] for i in range(4))
        )
        if len(exchanges) == 1:
            return CoherentAmplitudeModel(self.components)
        return CoherentAmplitudeModel(
            tuple(
                AmplitudeComponent(
                    c.name,
                    _BoseSymmetrized(c.function, exchanges),
                    c.coefficient,
                    c.normalize_component,
                )
                for c in self.components
            )
        )

    def generate_phase_space(self, size, *, seed=None):
        return NBodyPhaseSpaceMC(
            self.channel.parent_mass, self.channel.daughter_masses
        ).generate(size, seed=seed)

    def _component_scale(self, component, values, normalization_sample=None):
        normalize = (
            self.normalize_components
            if component.normalize_component is None
            else component.normalize_component
        )
        if not normalize:
            return 1.0
        sample = (
            self.normalization_sample
            if normalization_sample is None
            else normalization_sample
        )
        f = component.function(sample.as_dict(), values)
        return 1 / jnp.sqrt(jnp.mean(sample.weights * jnp.abs(f) ** 2))

    def amplitude(self, data, values=None):
        """Pure-JAX amplitude; use prepare_cache for repeated fit evaluations."""
        return sum(
            coefficient_value(c.coefficient, values)
            * self._component_scale(c, values)
            * c.function(data, values)
            for c in self.amplitude_model.components
        )

    def intensity(self, data, values=None):
        return jnp.abs(self.amplitude(data, values)) ** 2

    def prepare_cache(
        self,
        data_sample,
        normalization_sample=None,
        *,
        efficiency_normalization=None,
        normalize_components=None,
    ):
        sample = (
            self.normalization_sample
            if normalization_sample is None
            else normalization_sample
        )
        self._validate_sample(data_sample)
        self._validate_sample(sample, integration=True)
        normalize = (
            self.normalize_components
            if normalize_components is None
            else bool(normalize_components)
        )
        reusable = (
            sample is self.normalization_sample
            and efficiency_normalization is None
            and not any(
                p.kind is ParameterKind.DYNAMICS and not p.fixed
                for p in self.parameters
            )
        )
        template = (
            self._fixed_normalization_templates.get(normalize) if reusable else None
        )
        if template is not None:
            scales, matrix = template
            return PreparedAmplitudeCache.prepare_from_fixed_normalization(
                self.amplitude_model.components,
                data={"momenta": data_sample.momenta},
                normalization_weights=sample.weights,
                parameters=self.parameters,
                normalization_matrix_fixed=matrix,
                component_scales=scales,
                normalize_components=normalize,
            )
        cache = PreparedAmplitudeCache.prepare(
            self.amplitude_model.components,
            data={"momenta": data_sample.momenta},
            normalization_data={"momenta": sample.momenta},
            normalization_weights=sample.weights,
            parameters=self.parameters,
            efficiency_normalization=efficiency_normalization,
            normalize_components=normalize,
            normalization_chunk_size=self.normalization_chunk_size,
            chunk_size_memo=self._auto_chunk_memo,
        )
        if reusable:
            self._fixed_normalization_templates[normalize] = (
                cache.component_scales,
                cache.normalization_matrix_fixed,
            )
        return cache

    def pdf(self, normalization_sample=None, *, efficiency=None):
        sample = (
            self.normalization_sample
            if normalization_sample is None
            else normalization_sample
        )
        self._validate_sample(sample, integration=True)
        kwargs = {} if efficiency is None else {"efficiency": efficiency}

        def intensity(data, values):
            amplitude = sum(
                coefficient_value(c.coefficient, values)
                * self._component_scale(c, values, sample)
                * c.function(data, values)
                for c in self.amplitude_model.components
            )
            return jnp.abs(amplitude) ** 2

        return SignalPDF(intensity, GridIntegrator(sample), **kwargs)

    def _fraction_cache(self, normalization_sample=None, efficiency=None):
        sample = (
            self.normalization_sample
            if normalization_sample is None
            else normalization_sample
        )
        acceptance = None if efficiency is None else efficiency(sample.as_dict())
        return self.prepare_cache(
            sample.take(jnp.array([0])), sample, efficiency_normalization=acceptance
        )

    def fit_fractions(
        self, fit_values=None, *, normalization_sample=None, efficiency=None
    ):
        cache = self._fraction_cache(normalization_sample, efficiency)
        return cache.fit_fractions({} if fit_values is None else fit_values)

    def interference_fractions(
        self, fit_values=None, *, normalization_sample=None, efficiency=None
    ):
        cache = self._fraction_cache(normalization_sample, efficiency)
        return cache.interference_fractions({} if fit_values is None else fit_values)
