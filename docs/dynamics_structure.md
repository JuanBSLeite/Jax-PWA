# Dynamics module structure

One-dimensional resonance lineshapes are organized under `src/jaxpwa/dynamics/lineshape/`, with one physical model per file:

```text
dynamics/
  angular.py
  context.py
  resonance.py
  qmi_pixel.py
  lineshape/
    __init__.py
    common.py
    relativistic_breit_wigner.py
    gounaris_sakurai.py
    flatte.py
    pole.py
    lass.py
    kmatrix.py
    qmi.py
```

`common.py` contains only shared kinematic and Blatt-Weisskopf helpers. Public imports remain available from `jaxpwa` and `jaxpwa.dynamics`, so user code does not need to import implementation files directly.

Angular factors are interchangeable plugins passed through `Resonance(..., angular=...)`. The default remains `CovariantAngular()`. Two Laura++ Zemach conventions are also available:

```python
from jaxpwa import Zemach_P, Zemach_Pstar

rho = Resonance(
    "rho770",
    (0, 1),
    coefficient,
    mass=0.77526,
    width=0.1491,
    spin=1,
    angular=Zemach_P(),
)
```

`Zemach_P` uses the bachelor momentum `p` evaluated in the resonance rest frame, whereas `Zemach_Pstar` uses `p*` evaluated in the parent rest frame. For spin `L`, both multiply the Laura++ phase-convention Legendre polynomial by `(p q)^L` or `(p* q)^L`, respectively. The implementation currently follows the package-wide supported angular range `L=0..4`.

A legacy GooFit angular convention is also retained explicitly for compatibility with the historical `Ds -> pi pi pi` implementation:

```python
from jaxpwa import GooFitLegacyAngular

rho_legacy = Resonance(
    "rho770",
    (0, 1),
    coefficient,
    mass=0.77526,
    width=0.1491,
    spin=1,
    angular=GooFitLegacyAngular(),
    bachelor_momentum_frame="parent",
)
```

`GooFitLegacyAngular` is not the default and is intended only for reproducing or comparing against legacy GooFit models. It uses the resonance-rest-frame bachelor momentum `p` and implements the historical factors

- `L=0`: `1`,
- `L=1`: `4 p q cos(theta)`,
- `L=2`: `(16/3) (p q)^2 (3 cos(theta)^2 - 1)`.

Only `L=0..2` are supported by this legacy option. For a full reproduction of the historical GooFit resonance convention, use `bachelor_momentum_frame="parent"` as in the example so that the parent Blatt-Weisskopf factor uses the bachelor momentum in the parent rest frame. Without that setting, only the angular spin factor itself follows the GooFit legacy convention.

For the covariant and Zemach choices, the angular plugin does not implicitly alter the Blatt-Weisskopf convention. The parent-barrier bachelor-momentum frame remains controlled independently through `bachelor_momentum_frame`.

### Normalizing form factors at the pole

`Resonance(..., normalize_form_factors=True)` is the default: both the
resonance and parent barriers use `sqrt(P_L(q0*r)/P_L(q*r))`, equal to one
at the pole. Set `normalize_form_factors=False` for the raw Laura++ primed
convention, `1/sqrt(P_L(q*r))`, for both barriers:

```python
rho = Resonance(
    "rho0_770", (2, 0), coefficient,
    lineshape=GounarisSakurai(), angular=ZemachP(),
    mass=0.77526, width=0.1478, spin=1,
    resonance_radius=4.0, parent_radius=4.0,
    normalize_form_factors=False,
    normalize_component=False,
)
```

Here `P_1(x)=1+x²`, `P_2(x)=9+3x²+x⁴`, etc. The same option is available
on low-level `ResonanceAmplitude`; the helper
`blatt_weisskopf_from_momenta` accepts `normalize_at_pole=False`.
The running width always retains the ratio `B(q)/B(q0)` regardless of this
amplitude option. Spin-zero factors remain one. Pole momenta follow the existing
effective-pole-mass prescription for poles outside the kinematic interval.

This is independent of `normalize_component`, which rescales a whole component
to unit integral. Such rescaling cancels positive constant barrier scales for
fixed dynamics. With raw components, switching conventions requires transforming
coefficients; if pole masses or radii float, that transformation itself varies
with those parameters. Model JSON export/import preserves the option; older
specifications without it keep the default `True`.

`QMIPixel` remains at the `dynamics` level because it is a full two-dimensional Dalitz amplitude evaluated through `DalitzAmplitude`, rather than a one-dimensional `lineshape(mass, context)` plugin used by `Resonance`.

`lineshape/sympy.py` provides the optional `SympyLineshape` adapter. It uses the
same callable interface and parameter-resolution hooks; existing plugins and
resonance assembly are unchanged. SymPy is imported only when constructing or
loading symbolic definitions. See `docs/lineshapes.md`.
