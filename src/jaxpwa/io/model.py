"""Serialize a ``DecayModel`` definition (topology + parameters) to/from JSON.

This captures everything needed to rebuild an equivalent ``DecayModel``: the
decay channel, the component list (``Resonance``/``NonResonant``/
``DalitzAmplitude``, including their ``lineshape``/``angular``/``coefficient``
plugins and any embedded ``Parameter`` declarations), and the model-wide
normalization settings. Lazily-built JAX kernels, prepared caches and
compiled executables are never part of the specification -- a freshly
imported model rebuilds them the same way ``DecayModel.__init__`` always
does.

Only the built-in jaxpwa component/plugin classes registered in
``_REGISTRY`` can be exported or imported: a custom user plugin that is not a
plain ``@dataclass`` of numbers/strings/tuples/``Parameter``s (or, like
``SympyLineshape``, does not provide its own ``to_spec``/``from_spec`` pair)
cannot round-trip through this module. ``import_model`` only ever imports
names from this fixed registry, never an arbitrary module path named in the
file, so loading an untrusted model file cannot execute arbitrary code.

``DecayModel(normalization_method="toy-mc")`` is not supported: the external
``normalization_sample`` it depends on is not part of the specification.
Reconstruct that case directly with ``DecayModel(..., normalization_sample=...)``
after importing the component list, or use ``model_from_spec``/``import_model``
with one of the deterministic quadrature methods.

``export_model``/``model_to_spec`` also accept a ``FitSession`` directly (its
``.model`` attribute is exported), and ``export_cp_models``/``cp_models_to_spec``
accept a ``CPFitSession`` directly (its ``.plus_model``/``.minus_model`` are
exported together, sharing one parameter registry so a coefficient shared
between the two charges -- as ``CPRealImag.for_charge`` produces -- decodes back
to the same object). Neither captures the rest of the session: ``data``,
``efficiency``, ``veto``, ``backgrounds`` and ``constraints`` depend on
external samples/files already covered by ``docs/root_io.md`` and are not
part of the specification. Reload with
``FitSession(model=import_model(path), data=..., ...)`` or
``CPFitSession(plus_model=plus, minus_model=minus, plus_data=..., minus_data=...,
...)``.

``export_model``/``model_to_spec`` always serialize whatever a ``Parameter``'s
``.value`` currently is. Since ``Parameter``, ``Resonance`` and ``DecayModel``
are all frozen dataclasses, fitting never mutates them in place: ``Minimizer``/
``FitSession.fit`` return the best-fit values in a separate result object
(``FitSession.result_values(result)``), and ``model.parameters`` keeps holding
the initial/start values used to build the model. Exporting a model right
after a fit therefore still writes out those initial values unless you first
call ``model_with_fitted_values(model_or_session, session.result_values(result))``
(or ``cp_models_with_fitted_values`` for a CP pair) to build a new model whose
parameters carry the fitted numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

from ..coefficients import CPRealImag, RealImag
from ..decay import DalitzAmplitude, DecayChannel, DecayModel, NonResonant, Resonance
from ..dynamics import (
    LASS,
    QMI,
    BaBarFlatte,
    CovariantAngular,
    Flatte,
    GooFitLegacyAngular,
    GounarisSakurai,
    KMatrix,
    PipiKKRescattering,
    PolarFormFactorSymNR,
    Pole,
    QMIPixel,
    RelativisticBreitWigner,
    Rescattering2,
    RhoOmegaMixing,
    SigmaPole,
    SympyLineshape,
    ZemachP,
    ZemachPstar,
)
from ..fit.parameters import Parameter, ParameterKind

__all__ = [
    "cp_models_from_spec",
    "cp_models_to_spec",
    "cp_models_with_fitted_values",
    "export_cp_models",
    "export_model",
    "import_cp_models",
    "import_model",
    "model_from_spec",
    "model_to_spec",
    "model_with_fitted_values",
]

_SPEC_VERSION = 1

_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Resonance,
        NonResonant,
        DalitzAmplitude,
        RelativisticBreitWigner,
        Pole,
        SigmaPole,
        GounarisSakurai,
        RhoOmegaMixing,
        PipiKKRescattering,
        Flatte,
        BaBarFlatte,
        LASS,
        KMatrix,
        QMI,
        Rescattering2,
        SympyLineshape,
        CovariantAngular,
        ZemachP,
        ZemachPstar,
        GooFitLegacyAngular,
        QMIPixel,
        PolarFormFactorSymNR,
        RealImag,
        CPRealImag,
    )
}
# Former class names still accepted on load, so specs saved before a rename keep loading.
# Export always writes the current name (`_registered_name` checks `cls.__name__`).
_LEGACY_TYPE_NAMES: dict[str, type] = {"QMI2D": QMIPixel}

_DECAY_MODEL_SCALAR_KWARGS = (
    "normalize_components",
    "normalization_resolution",
    "normalization_method",
    "normalization_bin_width",
    "normalization_order_m13",
    "normalization_order_m23",
    "normalization_narrow_width",
    "normalization_narrow_window",
    "normalization_binning_factor",
)

# Optional keys: written by current versions, absent from specs saved while the
# option did not exist. Dropped memory-tuning options of older files are ignored.
_DECAY_MODEL_OPTIONAL_KWARGS = ("normalization_chunk_size",)


def _encode_parameter(value: Parameter) -> dict:
    result = asdict(value)
    result["kind"] = value.kind.value
    return {"parameter": result}


def _decode_parameter(entry: Mapping, registry: dict[str, Parameter]) -> Parameter:
    kwargs = dict(entry)
    kwargs["kind"] = ParameterKind(kwargs["kind"])
    if kwargs["bounds"] is not None:
        kwargs["bounds"] = tuple(kwargs["bounds"])
    value = Parameter(**kwargs)
    if value.name in registry:
        if registry[value.name] != value:
            raise ValueError(f"Conflicting definitions for parameter {value.name!r}")
        return registry[value.name]
    registry[value.name] = value
    return value


def _registered_name(cls: type) -> str:
    name = cls.__name__
    if _REGISTRY.get(name) is not cls:
        raise TypeError(
            f"cannot export {cls!r}: only the built-in jaxpwa "
            "component/plugin classes registered for model export are "
            "supported (custom plugins need their own to_spec/from_spec, "
            "like SympyLineshape)"
        )
    return name


def _encode(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, Parameter):
        return _encode_parameter(value)
    if hasattr(type(value), "to_spec"):
        return value.to_spec()
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {
            "type": _registered_name(cls),
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    raise TypeError(
        f"cannot export value of type {type(value)!r}; model export only "
        "supports plain numbers, strings, complex numbers, tuples/lists, "
        "Parameter instances and nested dataclasses built from those"
    )


def _decode(value: object, registry: dict[str, Parameter]) -> object:
    if isinstance(value, list):
        return tuple(_decode(item, registry) for item in value)
    if isinstance(value, dict):
        if set(value.keys()) == {"complex"}:
            real, imag = value["complex"]
            return complex(real, imag)
        if set(value.keys()) == {"parameter"}:
            return _decode_parameter(value["parameter"], registry)
        if "type" in value:
            cls = _REGISTRY.get(value["type"]) or _LEGACY_TYPE_NAMES.get(value["type"])
            if cls is None:
                raise ValueError(f"unknown component/plugin type {value['type']!r}")
            if "fields" not in value:
                if not hasattr(cls, "from_spec"):
                    raise ValueError(f"{value['type']!r} spec is missing 'fields'")
                return cls.from_spec(value, parameter_registry=registry)
            kwargs = {
                name: _decode(item, registry) for name, item in value["fields"].items()
            }
            return cls(**kwargs)
        return {key: _decode(item, registry) for key, item in value.items()}
    return value


def _resolve_decay_model(source: object) -> DecayModel:
    """Accept a ``DecayModel`` directly, or a ``FitSession``-like object.

    A ``FitSession`` (or anything exposing a ``.model`` attribute) is
    unwrapped to its ``DecayModel``. ``CPFitSession`` has two models and is
    intentionally rejected here; use ``cp_models_to_spec``/``export_cp_models``.
    """

    if isinstance(source, DecayModel):
        return source
    if hasattr(source, "plus_model") or hasattr(source, "minus_model"):
        raise TypeError(
            "export_model/model_to_spec cannot take a CPFitSession (it has "
            "two models); use export_cp_models/cp_models_to_spec instead"
        )
    model = getattr(source, "model", None)
    if isinstance(model, DecayModel):
        return model
    raise TypeError(
        f"cannot export {source!r}: expected a DecayModel, or an object with "
        "a '.model' attribute that is one (e.g. FitSession)"
    )


def model_to_spec(model: DecayModel) -> dict:
    """Return a JSON-compatible specification of ``model``'s topology and parameters.

    ``model`` may also be a ``FitSession`` (its ``.model`` is exported; the
    session's data/efficiency/veto/backgrounds/constraints are not part of
    the specification, see the module docstring).

    Raises ``ValueError`` if the model uses ``normalization_method="toy-mc"``,
    since its external ``normalization_sample`` cannot be recovered from the
    specification; see the module docstring.
    """

    model = _resolve_decay_model(model)
    if model.normalization_method == "toy-mc":
        raise ValueError(
            "model_to_spec/export_model cannot serialize "
            "normalization_method='toy-mc': the external normalization_sample "
            "is not part of the specification. Reconstruct that model with "
            "DecayModel(..., normalization_sample=...) directly instead."
        )
    channel = model.channel
    spec = {
        "type": "DecayModel",
        "version": _SPEC_VERSION,
        "channel": {
            "parent": channel.parent,
            "final_state": list(channel.final_state),
        },
        "components": [_encode(component) for component in model.components],
        "normalization_pair": list(model.normalization_pair),
    }
    for name in _DECAY_MODEL_SCALAR_KWARGS + _DECAY_MODEL_OPTIONAL_KWARGS:
        spec[name] = getattr(model, name)
    return spec


def model_from_spec(
    spec: Mapping, *, parameter_registry: dict[str, Parameter] | None = None
) -> DecayModel:
    """Reconstruct a ``DecayModel`` from a specification made by ``model_to_spec``.

    An optional ``{parameter.name: Parameter}`` registry preserves identity
    for parameters also shared with other models/components; conflicting
    definitions raise ``ValueError``. When given, ``parameter_registry`` is
    mutated in place with every parameter declared in ``spec`` -- pass the
    same dict to a later ``model_from_spec``/``cp_models_from_spec`` call to
    thread shared parameter identity across several models.
    """

    if spec.get("type") != "DecayModel" or spec.get("version") != _SPEC_VERSION:
        raise ValueError("unsupported DecayModel specification version/type")
    registry = {} if parameter_registry is None else parameter_registry
    channel = DecayChannel(
        parent=spec["channel"]["parent"],
        final_state=tuple(spec["channel"]["final_state"]),
    )
    components = [_decode(component, registry) for component in spec["components"]]
    # ``dynamics_microbatch_size`` (removed option) in old files is ignored.
    kwargs = {name: spec[name] for name in _DECAY_MODEL_SCALAR_KWARGS}
    kwargs.update(
        {name: spec[name] for name in _DECAY_MODEL_OPTIONAL_KWARGS if name in spec}
    )
    kwargs["normalization_pair"] = tuple(spec["normalization_pair"])
    return DecayModel(channel, components, **kwargs)


def export_model(model: DecayModel, path: str | Path) -> None:
    """Write ``model_to_spec(model)`` to ``path`` as JSON.

    ``model`` may also be a ``FitSession``; see ``model_to_spec``.
    """

    Path(path).write_text(json.dumps(model_to_spec(model), indent=2))


def import_model(
    path: str | Path, *, parameter_registry: dict[str, Parameter] | None = None
) -> DecayModel:
    """Read a ``DecayModel`` back from a file written by ``export_model``."""

    spec = json.loads(Path(path).read_text())
    return model_from_spec(spec, parameter_registry=parameter_registry)


def _resolve_cp_models(
    source: object, minus_model: object | None
) -> tuple[DecayModel, DecayModel]:
    if minus_model is not None:
        plus_model = source
    else:
        plus_model = getattr(source, "plus_model", None)
        minus_model = getattr(source, "minus_model", None)
    if not isinstance(plus_model, DecayModel) or not isinstance(
        minus_model, DecayModel
    ):
        raise TypeError(
            "cp_models_to_spec/export_cp_models expects two DecayModel "
            "arguments, or a single object with 'plus_model'/'minus_model' "
            "attributes (e.g. CPFitSession)"
        )
    return plus_model, minus_model


def cp_models_to_spec(source: object, minus_model: object | None = None) -> dict:
    """Return a joint specification for a B+/B- model pair.

    Call either as ``cp_models_to_spec(plus_model, minus_model)`` with two
    ``DecayModel`` instances, or ``cp_models_to_spec(session)`` with a
    ``CPFitSession`` (its ``.plus_model``/``.minus_model`` are used). The two
    models are encoded with one shared parameter registry, so a coefficient
    reused between charges via ``CPRealImag.for_charge`` round-trips back to
    the same object through ``cp_models_from_spec``.

    As with ``model_to_spec``, only the two amplitude models are captured --
    the session's data/efficiency/veto/backgrounds/constraints are not.
    """

    plus_model, minus_model = _resolve_cp_models(source, minus_model)
    return {
        "type": "CPModelPair",
        "version": _SPEC_VERSION,
        "plus_model": model_to_spec(plus_model),
        "minus_model": model_to_spec(minus_model),
    }


def cp_models_from_spec(
    spec: Mapping, *, parameter_registry: dict[str, Parameter] | None = None
) -> tuple[DecayModel, DecayModel]:
    """Reconstruct a ``(plus_model, minus_model)`` pair from ``cp_models_to_spec``."""

    if spec.get("type") != "CPModelPair" or spec.get("version") != _SPEC_VERSION:
        raise ValueError("unsupported CP model-pair specification version/type")
    registry = {} if parameter_registry is None else parameter_registry
    plus_model = model_from_spec(spec["plus_model"], parameter_registry=registry)
    minus_model = model_from_spec(spec["minus_model"], parameter_registry=registry)
    return plus_model, minus_model


def export_cp_models(
    source: object, path: str | Path, minus_model: object | None = None
) -> None:
    """Write ``cp_models_to_spec(source, minus_model)`` to ``path`` as JSON."""

    Path(path).write_text(json.dumps(cp_models_to_spec(source, minus_model), indent=2))


def import_cp_models(
    path: str | Path, *, parameter_registry: dict[str, Parameter] | None = None
) -> tuple[DecayModel, DecayModel]:
    """Read a ``(plus_model, minus_model)`` pair back from ``export_cp_models``."""

    spec = json.loads(Path(path).read_text())
    return cp_models_from_spec(spec, parameter_registry=parameter_registry)


def _apply_parameter_values(node: object, values: dict[str, float], fix: bool) -> None:
    if isinstance(node, dict):
        if set(node.keys()) == {"parameter"}:
            entry = node["parameter"]
            if entry["name"] in values:
                entry["value"] = float(values[entry["name"]])
                if fix:
                    entry["fixed"] = True
            return
        for item in node.values():
            _apply_parameter_values(item, values, fix)
    elif isinstance(node, list):
        for item in node:
            _apply_parameter_values(item, values, fix)


def model_with_fitted_values(
    source: object, values: Mapping[str, float], *, fix: bool = False
) -> DecayModel:
    """Return a new ``DecayModel`` with every ``Parameter.value`` taken from ``values``.

    ``source`` may be a ``DecayModel`` or a ``FitSession`` (its ``.model`` is
    used). ``values`` is typically ``session.result_values(result)`` (or
    ``dict(result.values)`` from a raw Minuit result): a ``{parameter.name:
    float}`` mapping. This is the supported way to "bake in" a fit result
    before exporting -- see the module docstring for why ``export_model``
    alone does not pick up fitted values automatically. Parameters whose name
    is not in ``values`` are left unchanged; a name in ``values`` that no
    parameter has is silently ignored.

    Pass ``fix=True`` to additionally mark every updated parameter as
    ``fixed=True`` in the returned model, e.g. to export a frozen snapshot of
    a best fit that can only be re-evaluated, not re-fitted.
    """

    model = _resolve_decay_model(source)
    spec = model_to_spec(model)
    _apply_parameter_values(spec, dict(values), fix)
    return model_from_spec(spec)


def cp_models_with_fitted_values(
    source: object,
    values: Mapping[str, float],
    minus_model: object | None = None,
    *,
    fix: bool = False,
) -> tuple[DecayModel, DecayModel]:
    """Like ``model_with_fitted_values``, for a B+/B- model pair.

    Call either as ``cp_models_with_fitted_values(plus_model, values,
    minus_model)`` or ``cp_models_with_fitted_values(session, values)`` with a
    ``CPFitSession`` (its ``.plus_model``/``.minus_model`` are used). A
    coefficient shared between the two charges (e.g. via
    ``CPRealImag.for_charge``) keeps that sharing in the returned models: it
    appears once per model in the specification, and the same ``values``
    entry updates both occurrences consistently.
    """

    plus_model, minus_model = _resolve_cp_models(source, minus_model)
    spec = cp_models_to_spec(plus_model, minus_model)
    _apply_parameter_values(spec, dict(values), fix)
    return cp_models_from_spec(spec)
