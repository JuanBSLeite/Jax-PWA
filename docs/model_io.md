# Model import/export

`export_model`/`import_model` (file-based) and `model_to_spec`/`model_from_spec` (in-memory
`dict`) serialize a `DecayModel`'s topology and parameters to JSON: the `DecayChannel`, every
component (`Resonance`/`NonResonant`/`DalitzAmplitude`, including their `lineshape`/`angular`/
`coefficient` plugins and any embedded `Parameter` declarations), and the model-wide
`normalization_*` settings. This lets a fitted or hand-built model be written to disk and
reconstructed later without re-running the Python script that built it.

```python
from jaxpwa import DecayChannel, DecayModel, Resonance, RealImag, export_model, import_model

model = DecayModel(
    DecayChannel("D+", ("pi-", "pi+", "pi+")),
    [Resonance("rho(770)0", pair=(0, 1), coefficient=RealImag(0.8, 0.2), spin=1)],
    normalization_method="square-dalitz",
)
export_model(model, "model.json")

restored = import_model("model.json")
```

`restored` is a fresh `DecayModel`: it re-runs the normal `DecayModel.__init__` path, so its
lazily-built JAX kernels, prepared caches and compiled executables are rebuilt exactly as they
would be for any newly constructed model. None of that transient state is ever part of the
specification.

## What can be serialized

A component field can hold a plain number, string, `bool`, `complex` number, a `Parameter`, or a
nested `dataclass` built from those (tuples included) -- which covers every built-in lineshape
(`RelativisticBreitWigner`, `Pole`, `SigmaPole`, `GounarisSakurai`, `RhoOmegaMixing`,
`PipiKKRescattering`, `Flatte`, `BaBarFlatte`, `LASS`, `KMatrix`, `QMI`, `Rescattering2`), every
angular model (`CovariantAngular`, `ZemachP`/`ZemachPstar`, `GooFitLegacyAngular`), the coefficient
classes (`RealImag`, `CPRealImag`), and the `QMI2D`/`PolarFormFactorSymNR` two-dimensional amplitudes. `SympyLineshape` is the
one exception with genuinely non-trivial internal state (a compiled kernel derived from a SymPy
expression tree); it already implements its own `to_spec()`/`from_spec()` pair (see
`dynamics/lineshape/sympy.py`), and `export_model`/`import_model` dispatch to it automatically
whenever they encounter one.

A custom plugin that is not a plain dataclass of the types above (or does not provide its own
`to_spec`/`from_spec`) cannot be exported; `export_model` raises `TypeError` naming the offending
value rather than silently dropping it.

`normalization_method="toy-mc"` is not supported: its external `normalization_sample` (arbitrary
JAX/numpy arrays from an external generator) is not part of the specification. `export_model`
raises `ValueError` for that case -- reconstruct that particular model directly with
`DecayModel(..., normalization_sample=...)` instead.

## Preserving shared parameters

Two components that share the same `Parameter` object (e.g. a common Blatt-Weisskopf radius) are
still two independent `Parameter` instances once decoded, but they compare equal (`Parameter` is a
value dataclass), so `DecayModel`'s conflicting-definition check still passes. To preserve object
*identity* with a `Parameter` also used elsewhere in your own code -- for example a shared
resonance mass reused between a `FitSession`'s model and a separately built one -- pass a
`{parameter.name: Parameter}` registry:

```python
restored = model_from_spec(spec, parameter_registry={shared_mass.name: shared_mass})
```

A conflicting definition for the same name (same name, different `value`/`bounds`/`fixed`/...)
raises `ValueError` instead of silently picking one.

## Exporting a model right after a fit

`Parameter`, `Resonance` and `DecayModel` are all frozen (immutable) dataclasses, and fitting
never mutates them in place: `FitSession.fit()`/`CPFitSession.fit()` report the best-fit values in
a separate result object, read back with `session.result_values(result)`. So `export_model`/
`model_to_spec` right after a fit still write out whatever `Parameter.value` the model was built
with -- the initial/start values, not the fitted ones -- unless you first bake the fit result into
a new model:

```python
result, updated_model = session.fit(update_model=True)
export_model(updated_model, "fitted_model.json")
```

`update_model=True` builds that new `DecayModel` for you (via `model_with_fitted_values`,
below) and changes `fit()`'s return value to the tuple `(result, updated_model)` instead of plain
`result`; it defaults to `False`; existing code that only reads `result` is unaffected.
`CPFitSession.fit(update_model=True)` is the same idea, returning
`(result, plus_model, minus_model)`.

You can also do this yourself for a model you did not just fit, e.g. one loaded with
`import_model` and combined with externally-known best-fit values:

```python
from jaxpwa import model_with_fitted_values

updated_model = model_with_fitted_values(model, session.result_values(result))
# or, equivalently, straight from the session:
updated_model = model_with_fitted_values(session, session.result_values(result))
```

`values` is a `{parameter.name: float}` mapping; a name in `values` that no parameter has is
silently ignored, and a parameter not named in `values` keeps its current `.value`. Pass
`fix=True` to also mark every updated parameter `fixed=True` in the returned model -- useful for
exporting a frozen snapshot of a best fit that can only be re-evaluated, not re-fitted.
`cp_models_with_fitted_values(source, values, minus_model=None, *, fix=False)` is the equivalent
for a B+/B- pair (or a `CPFitSession`), and keeps a coefficient shared between the two charges
(e.g. `CPRealImag.for_charge`) shared in the two returned models too.

## Exporting directly from a FitSession or CPFitSession

`export_model`/`model_to_spec` also accept a `FitSession` directly -- its `.model` is exported:

```python
session = FitSession(model, data)
export_model(session, "model.json")
```

Only the amplitude model is captured this way. `FitSession.data`, `.efficiency`, `.veto`,
`.backgrounds` and `.constraints` depend on external samples/files already covered by
`docs/root_io.md` and are not part of the specification. Reload with

```python
restored_session = FitSession(model=import_model("model.json"), data=data)
```

re-supplying `data` (and any efficiency/veto/backgrounds/constraints) yourself.

`CPFitSession` has two models (`.plus_model`/`.minus_model`), so it needs the dedicated
`export_cp_models`/`import_cp_models` (file) and `cp_models_to_spec`/`cp_models_from_spec`
(in-memory) pair instead of `export_model`/`import_model` -- calling the single-model functions on
a `CPFitSession` raises `TypeError` naming `export_cp_models` as the alternative:

```python
from jaxpwa import export_cp_models, import_cp_models

export_cp_models(session, "cp_models.json")  # or export_cp_models(plus_model, "cp_models.json", minus_model)

restored_plus, restored_minus = import_cp_models("cp_models.json")
restored_session = CPFitSession(restored_plus, restored_minus, plus_data, minus_data)
```

The two models are encoded and decoded through one shared parameter registry, so a coefficient
built with `CPRealImag.for_charge` -- shared between the B+ and B- models per
`docs/cp_coefficients.md` -- decodes back to the exact same `Parameter` object in both models, not
just an equal one. This matters for `CPFitSession`, whose joint normalization (see
`docs/cp_coefficients.md`, "CP fits share one normalization across charges") is built from the
same underlying `Parameter` list threaded through both charge models.

## Only jaxpwa classes are ever imported

`import_model`/`model_from_spec` resolve every serialized component/plugin type against a fixed,
built-in registry (`jaxpwa.io.model._REGISTRY`) -- the file never names a Python module
path to import. Loading an untrusted model file therefore cannot execute arbitrary code; at worst
an unrecognized `"type"` value raises `ValueError`.
