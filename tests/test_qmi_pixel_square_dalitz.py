"""QMIPixel in Square-Dalitz coordinates; DP vs Square-DP likelihood (incl. CP)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import (
    QMI,
    CPFitSession,
    CPRealImag,
    DalitzAmplitude,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    QMIPixel,
    RealImag,
    RelativisticBreitWigner,
    Resonance,
    enable_x64,
    model_from_spec,
    model_to_spec,
)
from jaxpwa.kinematics import (
    SquareDalitzGrid,
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)

enable_x64()

CHANNEL = DecayChannel("D+", ("pi-", "pi+", "pi+"))
M = float(CHANNEL.parent_mass)
MASSES = tuple(float(m) for m in CHANNEL.daughter_masses)
PAIR = (1, 2)  # the identical pi+ pi+ pair


def _square_field(
    magnitudes,
    phases,
    *,
    interpolation="none",
    folded=False,
    mprime_edges=(0.0, 0.5, 1.0),
    thetaprime_edges=(0.0, 0.5, 1.0),
):
    return QMIPixel(
        s12_edges=mprime_edges,
        s13_edges=thetaprime_edges,
        magnitudes=magnitudes,
        phases=phases,
        interpolation=interpolation,
        folded=folded,
        coordinates="square-dalitz",
        mother_mass=M,
        masses=MASSES,
        pair=PAIR,
    )


def _events_at(mprime, thetaprime):
    s12, s13, s23 = square_dalitz_to_invariants(
        jnp.asarray(mprime),
        jnp.asarray(thetaprime),
        mother_mass=M,
        masses=MASSES,
        pair=PAIR,
    )
    return {"s12": s12, "s13": s13, "s23": s23}


def test_square_dalitz_pixel_looks_up_the_cell_of_each_event():
    field = _square_field(((1.0, 2.0), (3.0, 4.0)), ((0.0, 0.1), (0.2, 0.3)))
    mag, phase = field.interpolated_magnitude_phase(
        _events_at([0.2, 0.2, 0.7, 0.7], [0.2, 0.7, 0.2, 0.7])
    )
    np.testing.assert_allclose(mag, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(phase, [0.0, 0.1, 0.2, 0.3])


def test_square_dalitz_pixel_linear_reproduces_cell_centres():
    edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    magnitudes = tuple(tuple(1.0 + i + 0.5 * j for j in range(4)) for i in range(4))
    phases = tuple(tuple(0.1 * i - 0.2 * j for j in range(4)) for i in range(4))
    field = _square_field(
        magnitudes,
        phases,
        interpolation="linear",
        mprime_edges=edges,
        thetaprime_edges=edges,
    )
    centres = [0.125, 0.375, 0.625, 0.875]
    mp, tp = np.meshgrid(centres, centres, indexing="ij")
    mag, phase = field.interpolated_magnitude_phase(_events_at(mp.ravel(), tp.ravel()))
    np.testing.assert_allclose(mag, np.ravel(magnitudes), atol=1e-9)
    np.testing.assert_allclose(phase, np.ravel(phases), atol=1e-9)


@pytest.mark.parametrize("interpolation", ["none", "linear", "cubic"])
def test_folded_square_dalitz_pixel_is_symmetric_under_identical_pion_exchange(
    interpolation,
):
    edges_m = (0.0, 0.3, 0.6, 1.0)
    edges_t = (0.0, 0.2, 0.35, 0.5)
    magnitudes = tuple(tuple(1.0 + i + 0.3 * j * j for j in range(3)) for i in range(3))
    phases = tuple(tuple(0.2 * i - 0.1 * j for j in range(3)) for i in range(3))
    field = _square_field(
        magnitudes,
        phases,
        interpolation=interpolation,
        folded=True,
        mprime_edges=edges_m,
        thetaprime_edges=edges_t,
    )
    model = DecayModel(
        CHANNEL,
        [DalitzAmplitude("pixels", field, RealImag(1.0, 0.0))],
        normalization_resolution=20,
    )
    data = model.generate_phase_space(500, seed=3, include_momenta=False).as_dict()
    # Exchanging the two pi+ (daughters 1 and 2) swaps s12 and s13.
    swapped = {"s12": data["s13"], "s13": data["s12"], "s23": data["s23"]}
    np.testing.assert_allclose(field(data), field(swapped), atol=1e-12)


def test_unfolded_square_dalitz_pixel_distinguishes_the_two_halves():
    field = _square_field(((1.0, 2.0), (1.0, 2.0)), ((0.0, 0.0), (0.0, 0.0)))
    mag, _ = field.interpolated_magnitude_phase(_events_at([0.4, 0.4], [0.2, 0.8]))
    np.testing.assert_allclose(mag, [1.0, 2.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(coordinates="square-dalitz"), "requires mother_mass, masses and pair"),
        (
            dict(
                coordinates="square-dalitz", mother_mass=M, masses=MASSES, pair=(1, 1)
            ),
            "distinct",
        ),
        (dict(coordinates="polar"), "coordinates must be"),
        (dict(mother_mass=M, masses=MASSES, pair=PAIR), "only used with"),
    ],
)
def test_square_dalitz_pixel_rejects_inconsistent_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QMIPixel((0.0, 1.0), (0.0, 1.0), ((1.0,),), ((0.0,),), **kwargs)


def test_square_dalitz_pixel_rejects_out_of_range_and_unfoldable_edges():
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        _square_field(
            ((1.0,),), ((0.0,),), mprime_edges=(0.0, 1.2), thetaprime_edges=(0.0, 1.0)
        )
    with pytest.raises(ValueError, match="within \\[0, 0.5\\]"):
        _square_field(
            ((1.0,),),
            ((0.0,),),
            folded=True,
            mprime_edges=(0.0, 1.0),
            thetaprime_edges=(0.0, 1.0),
        )


def test_square_dalitz_pixel_model_round_trip():
    a00 = Parameter.dynamics("pixels.a00", 1.3, owner="pixels")
    field = _square_field(
        ((a00, 2.0), (3.0, 4.0)), ((0.0, 0.1), (0.2, 0.3)), interpolation="linear"
    )
    model = DecayModel(
        CHANNEL,
        [
            DalitzAmplitude("pixels", field, RealImag(1.0, 0.0)),
            NonResonant(RealImag(0.5, 0.2)),
        ],
        normalization_resolution=20,
    )
    restored = model_from_spec(model_to_spec(model))
    assert restored.components[0].dynamics == field
    data = model.generate_phase_space(50, seed=1, include_momenta=False).as_dict()
    np.testing.assert_allclose(
        restored.intensity(data), model.intensity(data), rtol=1e-12
    )


def _cp_channels():
    return (
        DecayChannel("B+", ("pi-", "pi+", "pi+")),
        DecayChannel("B-", ("pi+", "pi-", "pi-")),
    )


def test_cp_fit_session_floats_square_dalitz_pixels_of_both_charges():
    plus_channel, minus_channel = _cp_channels()
    mother = float(plus_channel.parent_mass)
    masses = tuple(float(m) for m in plus_channel.daughter_masses)
    edges = (0.0, 0.5, 1.0)
    half = (0.0, 0.25, 0.5)

    def model(channel, charge):
        label = "plus" if charge > 0 else "minus"
        cell = Parameter.dynamics(f"pixels.a00_{label}", 1.0, owner="pixels")
        phase = Parameter.dynamics(f"pixels.p01_{label}", 0.3, owner="pixels")
        field = QMIPixel(
            edges,
            half,
            ((cell, 1.0), (1.0, 1.0)),
            ((0.0, phase), (0.0, 0.0)),
            interpolation="linear",
            folded=True,
            coordinates="square-dalitz",
            mother_mass=mother,
            masses=masses,
            pair=(1, 2),
        )
        return DecayModel(
            channel,
            [
                DalitzAmplitude("pixels", field, RealImag(1.0, 0.0)),
                NonResonant(CPRealImag(0.6, -0.3, 0.1, 0.05).for_charge(charge)),
            ],
            normalize_components=False,
            normalization_method="square-dalitz",
            normalization_pair=(1, 2),
            normalization_resolution=40,
        )

    plus_model, minus_model = model(plus_channel, +1), model(minus_channel, -1)
    plus_data = plus_model.generate_phase_space(300, seed=11, include_momenta=False)
    minus_data = minus_model.generate_phase_space(300, seed=12, include_momenta=False)
    session = CPFitSession(plus_model, minus_model, plus_data, minus_data)
    values = {
        "pixels.a00_plus": 1.2,
        "pixels.p01_plus": 0.4,
        "pixels.a00_minus": 0.9,
        "pixels.p01_minus": 0.1,
    }
    nll = float(session.objective(values))
    gradient = jax.grad(lambda v: session.objective(v))(values)
    assert np.isfinite(nll)
    for name, value in gradient.items():
        assert np.isfinite(float(value)) and abs(float(value)) > 1e-8, name


def _cp_isobar_models(use_qmi):
    plus_channel, minus_channel = _cp_channels()
    rho = CPRealImag(
        Parameter.coefficient("rho.x", 1.0, owner="rho", fixed=True),
        Parameter.coefficient("rho.y", 0.0, owner="rho", fixed=True),
        Parameter.coefficient("rho.dx", 0.05, owner="rho"),
        0.0,
    )
    nr = CPRealImag(
        *(
            Parameter.coefficient(f"nr.{a}", v, owner="nr")
            for a, v in zip(
                ("x", "y", "dx", "dy"), (0.4, -0.2, 0.05, 0.02), strict=True
            )
        )
    )
    if use_qmi:
        knots = (0.28, 0.6, 1.0, 1.6, 3.0, 5.14)
        nodes = [
            CPRealImag(
                *(
                    Parameter.dynamics(f"S[{k}].{a}", v, owner="S")
                    for a, v in zip(
                        ("x", "y", "dx", "dy"),
                        (0.3 + 0.1 * k, -0.1 * k, 0.02, 0.01),
                        strict=True,
                    )
                )
            )
            for k in range(len(knots))
        ]

    def model(channel, charge):
        components = [
            Resonance(
                "rho",
                (0, 1),
                rho.for_charge(charge),
                lineshape=RelativisticBreitWigner(),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(nr.for_charge(charge), name="nr"),
        ]
        if use_qmi:
            charged = [node.for_charge(charge) for node in nodes]
            components.append(
                Resonance(
                    "S",
                    (0, 1),
                    RealImag(1.0, 0.0),
                    lineshape=QMI(
                        knots=knots,
                        real_parts=tuple(n.real_part for n in charged),
                        imaginary_parts=tuple(n.imag_part for n in charged),
                        interpolation="linear",
                    ),
                    mass=1.0,
                    width=0.1,
                    spin=0,
                    normalize_component=False,
                )
            )
        # Ordinary (non-square) Gauss-Legendre normalization: independent of the
        # Square-Dalitz quadrature used below.
        return DecayModel(
            channel,
            components,
            normalize_components=True,
            normalization_method="gauss-legendre",
            normalization_bin_width=0.005,
        )

    return model(plus_channel, +1), model(minus_channel, -1)


@pytest.mark.parametrize("use_qmi", [False, True], ids=["isobar", "qmi"])
def test_cp_joint_likelihood_is_the_same_in_dalitz_and_square_dalitz_coordinates(
    use_qmi,
):
    """Square-DP and DP log-likelihoods differ by a parameter-independent constant.

    In (m', theta') the per-event density is |A|^2 |J| / integral, with the integral
    computed by an independent Square-Dalitz quadrature; the DP one is |A|^2 / integral
    with the model's own DP normalization. B+ and B- share one normalization
    (CPJointNLL).
    """
    plus_model, minus_model = _cp_isobar_models(use_qmi)
    mother = float(plus_model.channel.parent_mass)
    masses = tuple(float(m) for m in plus_model.channel.daughter_masses)
    samples = {
        q: m.generate_phase_space(400, seed=20 + i, include_momenta=False).as_dict()
        for i, (q, m) in enumerate(((+1, plus_model), (-1, minus_model)))
    }
    models = {+1: plus_model, -1: minus_model}
    square_grid = SquareDalitzGrid(
        mother, masses, resolution=400, pair=(1, 2), quadrature="gauss-legendre"
    ).sample()
    square_data = square_grid.as_dict()

    def log_jacobian(data):
        mp, tp = invariants_to_square_dalitz(
            data["s12"],
            data["s13"],
            data["s23"],
            mother_mass=mother,
            masses=masses,
            pair=(1, 2),
        )
        return jnp.log(
            square_dalitz_jacobian(
                mp, tp, mother_mass=mother, masses=masses, pair=(1, 2)
            )
        )

    def log_likelihoods(values):
        dp_norm = sum(
            jnp.mean(
                models[q].normalization_sample.weights
                * models[q].intensity(models[q].normalization_sample.as_dict(), values)
            )
            for q in (+1, -1)
        )
        sq_norm = sum(
            jnp.mean(square_grid.weights * models[q].intensity(square_data, values))
            for q in (+1, -1)
        )
        n = sum(len(samples[q]["s12"]) for q in (+1, -1))
        log_density = sum(
            jnp.sum(jnp.log(models[q].intensity(samples[q], values))) for q in (+1, -1)
        )
        dp = log_density - n * jnp.log(dp_norm)
        sq = (
            log_density
            + sum(jnp.sum(log_jacobian(samples[q])) for q in (+1, -1))
            - n * jnp.log(sq_norm)
        )
        return float(dp), float(sq)

    free = [p for p in plus_model.parameters + minus_model.parameters if not p.fixed]
    nominal = {p.name: float(p.value) for p in free}
    rng = np.random.default_rng(5)
    differences = []
    for _ in range(4):
        point = {name: value + rng.normal(0.0, 0.15) for name, value in nominal.items()}
        dp, sq = log_likelihoods(point)
        differences.append(sq - dp)
    # Same constant at every parameter point (up to the two quadratures' accuracy, ~1e-5
    # relative here, far below the 0.5 that one standard deviation moves the
    # log-likelihood):
    # identical maxima, errors and pulls.
    assert np.ptp(differences) < 2e-2, differences
