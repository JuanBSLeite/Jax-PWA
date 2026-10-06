"""Quasi-model-independent S-wave parameterisation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp

from jaxpwa.fit.parameters import Parameter, ParameterKind

from ..context import ResonanceContext


@jax.custom_vjp
def _step_qmi_prepared(values, index, order, starts, ends):
    """Gather one value per bin; reverse AD uses grouped sums, not atomics."""
    return values[jnp.asarray(index, dtype=jnp.int32)]


def _step_qmi_prepared_fwd(values, index, order, starts, ends):
    return values[jnp.asarray(index, dtype=jnp.int32)], (order, starts, ends)


def _step_qmi_prepared_bwd(residual, cotangent):
    order, starts, ends = residual
    gradient = _grouped_interval_sums(
        cotangent[jnp.asarray(order, dtype=jnp.int32)], starts, ends
    )
    return gradient, None, None, None, None


_step_qmi_prepared.defvjp(_step_qmi_prepared_fwd, _step_qmi_prepared_bwd)


@jax.custom_vjp
def _linear_qmi_prepared(
    magnitudes,
    phases,
    index,
    fraction,
    order,
    starts,
    ends,
):
    """Evaluate prepared linear QMI interpolation with a reduction-based VJP.

    The ordinary reverse-mode derivative of indexed interpolation is expressed
    by XLA as large scatter-add operations.  On consumer GPUs, FP64 atomics can
    dominate the fit time by orders of magnitude.  Because linear QMI support
    is local, each event contributes only to the two knots bordering its fixed
    interval.  The custom VJP accumulates those contributions after grouping
    events by interval, avoiding the large reverse scatter while preserving the
    exact forward model.
    """

    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=magnitudes.dtype)
    magnitude = magnitudes[index32] + fraction * (
        magnitudes[index32 + 1] - magnitudes[index32]
    )
    phase = phases[index32] + fraction * (phases[index32 + 1] - phases[index32])
    return magnitude * jnp.exp(1j * phase)


def _linear_qmi_prepared_fwd(
    magnitudes,
    phases,
    index,
    fraction,
    order,
    starts,
    ends,
):
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=magnitudes.dtype)
    magnitude = magnitudes[index32] + fraction * (
        magnitudes[index32 + 1] - magnitudes[index32]
    )
    phase = phases[index32] + fraction * (phases[index32 + 1] - phases[index32])
    exp_phase = jnp.exp(1j * phase)
    value = magnitude * exp_phase
    residual = (value, exp_phase, fraction, order, starts, ends)
    return value, residual


def _grouped_interval_sums(values, starts, ends):
    """Sum a sorted value vector over the fixed QMI interpolation intervals."""

    prefix = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=values.dtype),
            jnp.cumsum(values),
        )
    )
    starts = jnp.asarray(starts, dtype=jnp.int32)
    ends = jnp.asarray(ends, dtype=jnp.int32)
    return prefix[ends] - prefix[starts]


def _linear_qmi_prepared_bwd(residual, cotangent):
    value, exp_phase, fraction, order, starts, ends = residual

    # JAX's real-parameter/complex-output VJP convention is
    # dL/dx = Re(g * dy/dx), where g is the incoming complex cotangent.
    d_magnitude = jnp.real(cotangent * exp_phase)
    d_phase = jnp.real(cotangent * (1j * value))

    order = jnp.asarray(order, dtype=jnp.int32)
    sorted_fraction = fraction[order]
    sorted_d_magnitude = d_magnitude[order]
    sorted_d_phase = d_phase[order]

    left_magnitude = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_magnitude,
        starts,
        ends,
    )
    right_magnitude = _grouped_interval_sums(
        sorted_fraction * sorted_d_magnitude,
        starts,
        ends,
    )
    left_phase = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_phase,
        starts,
        ends,
    )
    right_phase = _grouped_interval_sums(
        sorted_fraction * sorted_d_phase,
        starts,
        ends,
    )

    magnitude_gradient = jnp.concatenate(
        (
            left_magnitude[:1],
            left_magnitude[1:] + right_magnitude[:-1],
            right_magnitude[-1:],
        )
    )
    phase_gradient = jnp.concatenate(
        (
            left_phase[:1],
            left_phase[1:] + right_phase[:-1],
            right_phase[-1:],
        )
    )

    return (
        magnitude_gradient,
        phase_gradient,
        None,
        None,
        None,
        None,
        None,
    )


_linear_qmi_prepared.defvjp(
    _linear_qmi_prepared_fwd,
    _linear_qmi_prepared_bwd,
)


@jax.custom_vjp
def _linear_cartesian_qmi_prepared(
    real_parts,
    imaginary_parts,
    index,
    fraction,
    order,
    starts,
    ends,
):
    """Evaluate a prepared Cartesian QMI with a reduction-based VJP."""

    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=real_parts.dtype)
    real = real_parts[index32] + fraction * (
        real_parts[index32 + 1] - real_parts[index32]
    )
    imaginary = imaginary_parts[index32] + fraction * (
        imaginary_parts[index32 + 1] - imaginary_parts[index32]
    )
    return real + 1j * imaginary


def _linear_cartesian_qmi_prepared_fwd(
    real_parts,
    imaginary_parts,
    index,
    fraction,
    order,
    starts,
    ends,
):
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=real_parts.dtype)
    real = real_parts[index32] + fraction * (
        real_parts[index32 + 1] - real_parts[index32]
    )
    imaginary = imaginary_parts[index32] + fraction * (
        imaginary_parts[index32 + 1] - imaginary_parts[index32]
    )
    value = real + 1j * imaginary
    residual = (fraction, order, starts, ends)
    return value, residual


def _linear_cartesian_qmi_prepared_bwd(residual, cotangent):
    fraction, order, starts, ends = residual

    # JAX's real-parameter/complex-output VJP convention is
    # dL/dx = Re(g * dy/dx).
    d_real = jnp.real(cotangent)
    d_imaginary = jnp.real(1j * cotangent)

    order = jnp.asarray(order, dtype=jnp.int32)
    sorted_fraction = fraction[order]
    sorted_d_real = d_real[order]
    sorted_d_imaginary = d_imaginary[order]

    left_real = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_real,
        starts,
        ends,
    )
    right_real = _grouped_interval_sums(
        sorted_fraction * sorted_d_real,
        starts,
        ends,
    )
    left_imaginary = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_imaginary,
        starts,
        ends,
    )
    right_imaginary = _grouped_interval_sums(
        sorted_fraction * sorted_d_imaginary,
        starts,
        ends,
    )

    real_gradient = jnp.concatenate(
        (
            left_real[:1],
            left_real[1:] + right_real[:-1],
            right_real[-1:],
        )
    )
    imaginary_gradient = jnp.concatenate(
        (
            left_imaginary[:1],
            left_imaginary[1:] + right_imaginary[:-1],
            right_imaginary[-1:],
        )
    )

    return (
        real_gradient,
        imaginary_gradient,
        None,
        None,
        None,
        None,
        None,
    )


_linear_cartesian_qmi_prepared.defvjp(
    _linear_cartesian_qmi_prepared_fwd,
    _linear_cartesian_qmi_prepared_bwd,
)


def _assemble_interval_endpoint_sums(left, right):
    """Assemble per-interval left/right sums into per-knot gradients."""

    return jnp.concatenate(
        (
            left[:1],
            left[1:] + right[:-1],
            right[-1:],
        )
    )


def _cubic_blend(fraction):
    """Local cubic smoothstep weight on one interpolation interval."""

    return fraction * fraction * (3.0 - 2.0 * fraction)


def _cubic_qmi_prepared_impl(values, index, fraction):
    """Evaluate strictly local cubic interpolation between adjacent QMI knots."""

    values = jnp.asarray(values)
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=values.dtype)
    weight = _cubic_blend(fraction)
    return values[index32] + weight * (values[index32 + 1] - values[index32])


@jax.custom_vjp
def _cubic_qmi_prepared(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
):
    """Prepared local cubic QMI interpolation with reduction-based VJP.

    Each event depends only on the two knots bordering its fixed interval.
    The cubic smoothstep weight gives zero slope at both interval endpoints,
    so neighboring intervals meet continuously with a continuous first
    derivative, without any coupling to distant knots.
    """

    return _cubic_qmi_prepared_impl(values, index, fraction)


def _cubic_qmi_prepared_fwd(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
):
    value = _cubic_qmi_prepared_impl(values, index, fraction)
    residual = (fraction, order, starts, ends)
    return value, residual


def _cubic_qmi_prepared_bwd(residual, cotangent):
    fraction, order, starts, ends = residual

    fraction = jnp.asarray(fraction)
    order = jnp.asarray(order, dtype=jnp.int32)
    cotangent = jnp.asarray(cotangent, dtype=fraction.dtype)
    weight = _cubic_blend(fraction)

    sorted_weight = weight[order]
    sorted_cotangent = cotangent[order]

    left = _grouped_interval_sums(
        (1.0 - sorted_weight) * sorted_cotangent,
        starts,
        ends,
    )
    right = _grouped_interval_sums(
        sorted_weight * sorted_cotangent,
        starts,
        ends,
    )
    values_gradient = _assemble_interval_endpoint_sums(left, right)

    return (
        values_gradient,
        None,
        None,
        None,
        None,
        None,
    )


_cubic_qmi_prepared.defvjp(
    _cubic_qmi_prepared_fwd,
    _cubic_qmi_prepared_bwd,
)


def _natural_cubic(values, index, fraction, knots):
    """Global C2 spline in s, with zero endpoint second derivatives.

    Solve only the knot-sized system, never an event-by-knot basis matrix.
    All operations remain differentiable with respect to the knot values.
    """
    values = jnp.asarray(values)
    knots = jnp.asarray(knots, dtype=values.dtype)
    h = jnp.diff(knots)
    second = jnp.zeros_like(values)
    if values.size > 2:
        diagonal = 2 * (h[:-1] + h[1:])
        matrix = jnp.diag(diagonal)
        matrix = matrix + jnp.diag(h[1:-1], 1) + jnp.diag(h[1:-1], -1)
        rhs = 6 * jnp.diff(jnp.diff(values) / h)
        interior = jnp.linalg.solve(matrix, rhs)
        second = jnp.concatenate((second[:1], interior, second[-1:]))
    index = jnp.asarray(index, dtype=jnp.int32)
    b = jnp.asarray(fraction, dtype=values.dtype)
    a = 1 - b
    return (
        a * values[index]
        + b * values[index + 1]
        + h[index] ** 2
        / 6
        * ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1])
    )


def _hermite_slopes(values, knot_s):
    """Return local finite-difference slopes for cubic Hermite interpolation."""

    values = jnp.asarray(values)
    knot_s = jnp.asarray(knot_s, dtype=values.dtype)

    first = (values[1] - values[0]) / (knot_s[1] - knot_s[0])
    last = (values[-1] - values[-2]) / (knot_s[-1] - knot_s[-2])
    if values.shape[0] == 2:
        return jnp.stack((first, last))

    interior = (values[2:] - values[:-2]) / (knot_s[2:] - knot_s[:-2])
    return jnp.concatenate((first[None], interior, last[None]))


def _hermite_basis(fraction):
    """Cubic Hermite basis functions on t in [0, 1]."""

    t = fraction
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00, h10, h01, h11


def _hermite_qmi_prepared_impl(values, index, fraction, knot_s):
    """Evaluate local cubic Hermite interpolation on prepared QMI intervals."""

    values = jnp.asarray(values)
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=values.dtype)
    knot_s = jnp.asarray(knot_s, dtype=values.dtype)

    slopes = _hermite_slopes(values, knot_s)
    width = knot_s[index32 + 1] - knot_s[index32]
    h00, h10, h01, h11 = _hermite_basis(fraction)
    return (
        h00 * values[index32]
        + h10 * width * slopes[index32]
        + h01 * values[index32 + 1]
        + h11 * width * slopes[index32 + 1]
    )


@jax.custom_vjp
def _hermite_qmi_prepared(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
    knot_s,
):
    """Prepared cubic Hermite QMI interpolation with reduction-based VJP.

    The value in interval i depends only on y_i, y_(i+1) and their local
    finite-difference slopes.  Interior slopes use knots i-1 and i+1, so an
    interval depends on at most four neighboring knots and never on the full
    QMI grid.
    """

    return _hermite_qmi_prepared_impl(values, index, fraction, knot_s)


def _hermite_qmi_prepared_fwd(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
    knot_s,
):
    value = _hermite_qmi_prepared_impl(values, index, fraction, knot_s)
    residual = (index, fraction, order, starts, ends, knot_s)
    return value, residual


def _hermite_qmi_prepared_bwd(residual, cotangent):
    index, fraction, order, starts, ends, knot_s = residual

    index = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction)
    order = jnp.asarray(order, dtype=jnp.int32)
    knot_s = jnp.asarray(knot_s, dtype=fraction.dtype)
    cotangent = jnp.asarray(cotangent, dtype=fraction.dtype)

    interval_width = knot_s[1:] - knot_s[:-1]
    h00, h10, h01, h11 = _hermite_basis(fraction)
    sorted_h00 = h00[order]
    sorted_h10 = h10[order]
    sorted_h01 = h01[order]
    sorted_h11 = h11[order]
    sorted_cotangent = cotangent[order]

    sorted_width = interval_width[index][order]

    direct_left = _grouped_interval_sums(
        sorted_h00 * sorted_cotangent,
        starts,
        ends,
    )
    direct_right = _grouped_interval_sums(
        sorted_h01 * sorted_cotangent,
        starts,
        ends,
    )
    direct_gradient = _assemble_interval_endpoint_sums(
        direct_left,
        direct_right,
    )

    slope_left = _grouped_interval_sums(
        sorted_h10 * sorted_width * sorted_cotangent,
        starts,
        ends,
    )
    slope_right = _grouped_interval_sums(
        sorted_h11 * sorted_width * sorted_cotangent,
        starts,
        ends,
    )
    slope_gradient = _assemble_interval_endpoint_sums(
        slope_left,
        slope_right,
    )

    n_knots = knot_s.shape[0]
    first_scaled = slope_gradient[0] / (knot_s[1] - knot_s[0])
    last_scaled = slope_gradient[-1] / (knot_s[-1] - knot_s[-2])
    endpoint_gradient = jnp.pad(
        jnp.stack((-first_scaled, first_scaled)),
        (0, n_knots - 2),
    ) + jnp.pad(
        jnp.stack((-last_scaled, last_scaled)),
        (n_knots - 2, 0),
    )

    if n_knots > 2:
        interior_scaled = slope_gradient[1:-1] / (knot_s[2:] - knot_s[:-2])
        slope_value_gradient = (
            endpoint_gradient
            + jnp.pad(-interior_scaled, (0, 2))
            + jnp.pad(interior_scaled, (2, 0))
        )
    else:
        slope_value_gradient = endpoint_gradient

    values_gradient = direct_gradient + slope_value_gradient
    return (
        values_gradient,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_hermite_qmi_prepared.defvjp(
    _hermite_qmi_prepared_fwd,
    _hermite_qmi_prepared_bwd,
)


def _interval_index_and_fraction(x, xp, prepared_index=None):
    """Return the interpolation interval and local fraction.

    A prepared index lets large repeated QMI evaluations reuse the fixed knot
    lookup while recomputing only the inexpensive local fraction.
    """

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    x_clamped = jnp.clip(x, xp[0], xp[-1])
    if prepared_index is None:
        index = jnp.clip(
            jnp.searchsorted(xp, x_clamped, side="right") - 1,
            0,
            xp.shape[0] - 2,
        )
    else:
        index = jnp.asarray(prepared_index, dtype=jnp.int32)
        index = jnp.clip(index, 0, xp.shape[0] - 2)
    x0 = xp[index]
    x1 = xp[index + 1]
    fraction = (x_clamped - x0) / (x1 - x0)
    return index, fraction


def _linear_spline(x, xp, fp):
    """Piecewise-linear interpolation with endpoint clamping.

    This is implemented directly rather than through jax.numpy.interp.
    QMI evaluations happen on very large normalization grids and the extra
    interpolation primitive can make JAX tracing/compilation disproportionately
    slow. The explicit form uses the same search/index pattern as the cubic spline.
    """

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)

    index, fraction = _interval_index_and_fraction(x, xp)
    y0, y1 = fp[index], fp[index + 1]
    return y0 + fraction * (y1 - y0)


def _local_cubic_spline(x, xp, fp):
    """Piecewise local cubic interpolation with endpoint clamping.

    On each interval only the two adjacent knot values are used.  The
    interpolation weight is the cubic smoothstep 3t^2 - 2t^3 in s=m^2.
    """

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)

    index, fraction = _interval_index_and_fraction(x, xp)
    weight = _cubic_blend(fraction)
    y0, y1 = fp[index], fp[index + 1]
    return y0 + weight * (y1 - y0)


def _local_hermite_spline(x, xp, fp):
    """Piecewise cubic Hermite interpolation with local finite-difference slopes."""

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)

    index, fraction = _interval_index_and_fraction(x, xp)
    slopes = _hermite_slopes(fp, xp)
    width = xp[index + 1] - xp[index]
    h00, h10, h01, h11 = _hermite_basis(fraction)
    return (
        h00 * fp[index]
        + h10 * width * slopes[index]
        + h01 * fp[index + 1]
        + h11 * width * slopes[index + 1]
    )


@dataclass(frozen=True)
class QMI:
    """Scalar amplitude with interpolation or constant mass bins.

    With ``interpolation="none"``, ``knots`` are bin edges and each
    parameter array has ``len(knots)-1`` entries. Bins are left-closed,
    right-open, with the final edge in the last bin. Outside the supplied
    range the endpoint bin value is held constant, as for interpolated QMI.
    """

    knots: tuple[float, ...]
    magnitudes: tuple[object, ...] | None = None
    phases: tuple[object, ...] | None = None
    interpolation: str = "linear"
    real_parts: tuple[object, ...] | None = None
    imaginary_parts: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.knots) < 2:
            raise ValueError("QMI requires at least two knots")
        polar = self.magnitudes is not None or self.phases is not None
        cartesian = self.real_parts is not None or self.imaginary_parts is not None
        if polar == cartesian:
            raise ValueError(
                "QMI requires exactly one complete polar or Cartesian parameter set"
            )
        expected = len(self.knots) - (self.interpolation == "none")
        groups = (
            (("magnitudes", self.magnitudes), ("phases", self.phases))
            if polar
            else (
                ("real parts", self.real_parts),
                ("imaginary parts", self.imaginary_parts),
            )
        )
        for label, group in groups:
            if group is None or len(group) != expected:
                convention = (
                    "one entry per bin (len(knots)-1)"
                    if self.interpolation == "none"
                    else "the same length as knots"
                )
                raise ValueError(f"QMI {label} must have {convention}")
        values = (
            (self.magnitudes, self.phases)
            if polar
            else (self.real_parts, self.imaginary_parts)
        )
        for group in values:
            for value in group:
                if (
                    isinstance(value, Parameter)
                    and not value.fixed
                    and value.kind is not ParameterKind.DYNAMICS
                ):
                    raise ValueError(
                        f"QMI knot {value.name!r} must use Parameter.dynamics "
                        "with the component owner"
                    )
        knots = tuple(float(value) for value in self.knots)
        if not all(isfinite(value) for value in knots):
            raise ValueError("QMI knot masses must be finite")
        if any(
            right <= left for left, right in zip(knots[:-1], knots[1:], strict=True)
        ):
            raise ValueError("QMI knots must be strictly increasing")
        if knots[0] <= 0.0:
            raise ValueError("QMI knot masses must be positive")
        if self.interpolation not in {"none", "linear", "cubic", "hermite", "natural"}:
            raise ValueError(
                "QMI interpolation must be none, linear, cubic, hermite, or natural"
            )

    @property
    def size(self) -> int:
        """Number of mass knots (bin edges for interpolation="none")."""
        return len(self.knots)

    @property
    def parameterization(self) -> str:
        """Whether knots are "cartesian" (real/imag) or "polar" (magnitude/phase)."""
        return "cartesian" if self.real_parts is not None else "polar"

    def smoothness_constraint(self, strength=0.0, *, weights=None):
        """Build an optional complex-knot curvature penalty for any interpolation.

        Attach it explicitly with ``session.with_constraint(...)`` or
        ``ConstrainedNLL``. Merely creating it does not change this lineshape
        or its normalization. ``strength=0`` is an exact no-op; ``weights``
        optionally scales each interior knot's three-knot stencil.
        See :class:`jaxpwa.constraints.QMISmoothnessConstraint`
        for the nonuniform-s convention and the required fixed QMI scale.
        """
        from jaxpwa.constraints import QMISmoothnessConstraint

        return QMISmoothnessConstraint(self, strength=strength, weights=weights)

    def _interpolate(self, s, knot_s, values, prepared_index=None):
        index, fraction = _interval_index_and_fraction(
            s,
            knot_s,
            prepared_index,
        )
        if self.interpolation == "natural":
            return _natural_cubic(values, index, fraction, knot_s)
        if self.interpolation == "linear":
            weight = fraction
            return values[index] + weight * (values[index + 1] - values[index])
        if self.interpolation == "cubic":
            weight = _cubic_blend(fraction)
            return values[index] + weight * (values[index + 1] - values[index])

        slopes = _hermite_slopes(values, knot_s)
        width = knot_s[index + 1] - knot_s[index]
        h00, h10, h01, h11 = _hermite_basis(fraction)
        return (
            h00 * values[index]
            + h10 * width * slopes[index]
            + h01 * values[index + 1]
            + h11 * width * slopes[index + 1]
        )

    def _interpolated_pair(
        self,
        mass,
        first_values,
        second_values,
        prepared_index=None,
    ):
        prepared_fraction = None
        prepared_order = None
        prepared_starts = None
        prepared_ends = None
        if isinstance(prepared_index, tuple):
            if len(prepared_index) < 2:
                raise ValueError("prepared QMI data are incomplete")
            prepared_index, prepared_fraction, *extra = prepared_index
            if extra:
                prepared_order, prepared_starts, prepared_ends = extra

        if mass is None:
            if prepared_fraction is None:
                raise ValueError(
                    "prepared QMI evaluation requires interpolation fractions"
                )
            dtype_source = jnp.asarray(prepared_fraction)
            knot_s = jnp.asarray(self.knots, dtype=dtype_source.dtype) ** 2
            s = None
        else:
            mass = jnp.asarray(mass)
            knot_s = jnp.asarray(self.knots, dtype=mass.dtype) ** 2
            s = mass**2

        magnitudes = jnp.asarray(first_values, dtype=knot_s.dtype)
        phases = jnp.asarray(second_values, dtype=knot_s.dtype)

        if self.interpolation == "none":
            if prepared_index is None:
                index, _ = _interval_index_and_fraction(s, knot_s)
            else:
                index = jnp.asarray(prepared_index, dtype=jnp.int32)
            return magnitudes[index], phases[index]

        if self.interpolation == "natural":
            if prepared_fraction is None:
                index, fraction = _interval_index_and_fraction(
                    s, knot_s, prepared_index
                )
            else:
                index, fraction = prepared_index, prepared_fraction
            return (
                _natural_cubic(magnitudes, index, fraction, knot_s),
                _natural_cubic(phases, index, fraction, knot_s),
            )

        if self.interpolation == "linear":
            if prepared_fraction is None:
                index, fraction = _interval_index_and_fraction(
                    s, knot_s, prepared_index
                )
            else:
                index = jnp.asarray(prepared_index, dtype=jnp.int32)
                fraction = jnp.asarray(prepared_fraction, dtype=knot_s.dtype)
            magnitude = magnitudes[index] + fraction * (
                magnitudes[index + 1] - magnitudes[index]
            )
            phase = phases[index] + fraction * (phases[index + 1] - phases[index])
            return magnitude, phase

        if self.interpolation == "hermite" and prepared_fraction is not None:
            index = jnp.asarray(prepared_index, dtype=jnp.int32)
            fraction = jnp.asarray(prepared_fraction, dtype=knot_s.dtype)
            width = knot_s[index + 1] - knot_s[index]
            h00, h10, h01, h11 = _hermite_basis(fraction)

            def hermite(values):
                slopes = _hermite_slopes(values, knot_s)
                return (
                    h00 * values[index]
                    + h10 * width * slopes[index]
                    + h01 * values[index + 1]
                    + h11 * width * slopes[index + 1]
                )

            return hermite(magnitudes), hermite(phases)

        if prepared_fraction is not None:
            index = jnp.asarray(prepared_index, dtype=jnp.int32)
            fraction = jnp.asarray(prepared_fraction, dtype=knot_s.dtype)
            weight = _cubic_blend(fraction)
            magnitude = magnitudes[index] + weight * (
                magnitudes[index + 1] - magnitudes[index]
            )
            phase = phases[index] + weight * (phases[index + 1] - phases[index])
            return magnitude, phase

        return (
            self._interpolate(s, knot_s, magnitudes, prepared_index),
            self._interpolate(s, knot_s, phases, prepared_index),
        )

    def _interpolated_magnitude_phase(self, mass, prepared_index=None):
        if self.parameterization == "polar":
            return self._interpolated_pair(
                mass,
                self.magnitudes,
                self.phases,
                prepared_index,
            )
        real, imaginary = self._interpolated_pair(
            mass,
            self.real_parts,
            self.imaginary_parts,
            prepared_index,
        )
        value = real + 1j * imaginary
        return jnp.abs(value), jnp.angle(value)

    def interpolated_magnitude_phase(self, mass):
        """Return the interpolated (magnitude, phase) of the amplitude at each mass."""
        return self._interpolated_magnitude_phase(mass)

    def _interpolated_cartesian(self, mass, prepared_index=None):
        if self.parameterization == "cartesian":
            return self._interpolated_pair(
                mass,
                self.real_parts,
                self.imaginary_parts,
                prepared_index,
            )
        magnitude, phase = self._interpolated_pair(
            mass,
            self.magnitudes,
            self.phases,
            prepared_index,
        )
        return magnitude * jnp.cos(phase), magnitude * jnp.sin(phase)

    def interpolated_cartesian(self, mass):
        """Return the interpolated (real, imaginary) amplitude parts at each mass."""
        return self._interpolated_cartesian(mass)

    @property
    def prepared_mass_is_self_contained(self) -> bool:
        """Prepared QMI data no longer need the resonance-mass array."""

        return True

    def prepare_mass(self, mass, context: ResonanceContext):
        """Cache the fixed knot interval and interpolation fraction."""

        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        mass = jnp.asarray(mass)
        knot_s = jnp.asarray(self.knots, dtype=mass.dtype) ** 2
        index, fraction = _interval_index_and_fraction(mass**2, knot_s)
        dtype = jnp.int16 if self.size <= 32767 else jnp.int32
        compact_index = index.astype(dtype)

        # The fixed ordering and interval boundaries are used only by the
        # custom reverse-mode rule.  They cost one int32 per event but replace
        # the much more expensive FP64 scatter-add generated by generic AD.
        order = jnp.argsort(index).astype(jnp.int32)
        counts = jnp.bincount(index.astype(jnp.int32), length=self.size - 1)
        ends = jnp.cumsum(counts).astype(jnp.int32)
        starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))
        return compact_index, fraction, order, starts, ends

    def evaluate_prepared(self, mass, prepared_index, context: ResonanceContext):
        """Evaluate the complex amplitude from a cached knot interval/fraction."""
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")

        if (
            self.interpolation == "none"
            and isinstance(prepared_index, tuple)
            and len(prepared_index) >= 5
        ):
            index, fraction, order, starts, ends = prepared_index[:5]

            def step(values):
                return _step_qmi_prepared(
                    jnp.asarray(values, dtype=jnp.asarray(fraction).dtype),
                    index,
                    order,
                    starts,
                    ends,
                )

            if self.parameterization == "cartesian":
                return step(self.real_parts) + 1j * step(self.imaginary_parts)
            return step(self.magnitudes) * jnp.exp(1j * step(self.phases))

        if (
            self.interpolation == "hermite"
            and isinstance(prepared_index, tuple)
            and len(prepared_index) >= 5
        ):
            index, fraction, order, starts, ends = prepared_index[:5]
            dtype_source = jnp.asarray(fraction)
            knot_s = jnp.asarray(self.knots, dtype=dtype_source.dtype) ** 2

            def hermite(values):
                return _hermite_qmi_prepared(
                    jnp.asarray(values, dtype=dtype_source.dtype),
                    index,
                    fraction,
                    order,
                    starts,
                    ends,
                    knot_s,
                )

            if self.parameterization == "cartesian":
                real = hermite(self.real_parts)
                imaginary = hermite(self.imaginary_parts)
                return real + 1j * imaginary

            magnitude = hermite(self.magnitudes)
            phase = hermite(self.phases)
            return magnitude * jnp.exp(1j * phase)

        if (
            self.interpolation == "cubic"
            and isinstance(prepared_index, tuple)
            and len(prepared_index) >= 5
        ):
            index, fraction, order, starts, ends = prepared_index[:5]
            dtype_source = jnp.asarray(fraction)

            def cubic(values):
                return _cubic_qmi_prepared(
                    jnp.asarray(values, dtype=dtype_source.dtype),
                    index,
                    fraction,
                    order,
                    starts,
                    ends,
                )

            if self.parameterization == "cartesian":
                real = cubic(self.real_parts)
                imaginary = cubic(self.imaginary_parts)
                return real + 1j * imaginary

            magnitude = cubic(self.magnitudes)
            phase = cubic(self.phases)
            return magnitude * jnp.exp(1j * phase)

        if self.interpolation == "linear" and isinstance(prepared_index, tuple):
            if len(prepared_index) >= 5:
                index, fraction, order, starts, ends = prepared_index[:5]
                dtype_source = jnp.asarray(fraction)
                if self.parameterization == "cartesian":
                    real_parts = jnp.asarray(
                        self.real_parts,
                        dtype=dtype_source.dtype,
                    )
                    imaginary_parts = jnp.asarray(
                        self.imaginary_parts,
                        dtype=dtype_source.dtype,
                    )
                    return _linear_cartesian_qmi_prepared(
                        real_parts,
                        imaginary_parts,
                        index,
                        fraction,
                        order,
                        starts,
                        ends,
                    )
                magnitudes = jnp.asarray(self.magnitudes, dtype=dtype_source.dtype)
                phases = jnp.asarray(self.phases, dtype=dtype_source.dtype)
                return _linear_qmi_prepared(
                    magnitudes,
                    phases,
                    index,
                    fraction,
                    order,
                    starts,
                    ends,
                )

        if self.parameterization == "cartesian":
            real, imaginary = self._interpolated_cartesian(
                mass,
                prepared_index=prepared_index,
            )
            return real + 1j * imaginary
        magnitude, phase = self._interpolated_magnitude_phase(
            mass,
            prepared_index=prepared_index,
        )
        return magnitude * jnp.exp(1j * phase)

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        if self.parameterization == "cartesian":
            real, imaginary = self.interpolated_cartesian(mass)
            return real + 1j * imaginary
        magnitude, phase = self.interpolated_magnitude_phase(mass)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI"]
