"""Small plotting helpers for common analysis diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from jaxpwa.kinematics import fold_thetaprime, invariants_to_square_dalitz


def _values(sample, variable: str):
    if hasattr(sample, variable):
        return np.asarray(getattr(sample, variable))
    if isinstance(sample, dict) and variable in sample:
        return np.asarray(sample[variable])
    if hasattr(sample, "observable"):
        return np.asarray(sample.observable(variable))
    raise KeyError(f"sample does not contain {variable!r}")


def _bin_width_label(edges, unit: str) -> str:
    widths = np.diff(np.asarray(edges, dtype=float))
    if widths.size == 0:
        return "Candidates / bin"
    if not np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12):
        return "Candidates / bin"
    width = f"{float(widths[0]):.3g}"
    return f"Candidates / {width} {unit}" if unit else f"Candidates / {width}"


def _bin_area_label(x_edges, y_edges, unit: str) -> str:
    """Two-dimensional analogue of ``_bin_width_label`` for a colorbar label."""
    x_widths = np.diff(np.asarray(x_edges, dtype=float))
    y_widths = np.diff(np.asarray(y_edges, dtype=float))
    if x_widths.size == 0 or y_widths.size == 0:
        return "Candidates / bin"
    if not np.allclose(x_widths, x_widths[0], rtol=1e-10, atol=1e-12) or not np.allclose(
        y_widths, y_widths[0], rtol=1e-10, atol=1e-12
    ):
        return "Candidates / bin"
    area = f"{float(x_widths[0]) * float(y_widths[0]):.3g}"
    return f"Candidates / {area} {unit}" if unit else f"Candidates / {area}"


def binned_data(
    values,
    *,
    bins=60,
    range: tuple[float, float] | None = None,
    weights=None,
):
    """Return bin centers, counts, statistical uncertainties and bin edges.

    Unweighted data use the usual ``sqrt(N)`` Poisson approximation. Weighted
    data use ``sqrt(sum w^2)`` in each bin.
    """

    values = np.asarray(values)
    if weights is None:
        counts, edges = np.histogram(values, bins=bins, range=range)
        errors = np.sqrt(counts.astype(float))
    else:
        weights = np.asarray(weights)
        counts, edges = np.histogram(values, bins=bins, range=range, weights=weights)
        sumw2, _ = np.histogram(values, bins=edges, weights=weights**2)
        errors = np.sqrt(sumw2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, errors, edges


def plot_binned_data(
    values,
    *,
    bins=60,
    range: tuple[float, float] | None = None,
    weights=None,
    ax=None,
    label: str = "data",
    markersize: float = 4.5,
    unit: str | None = None,
    log_scale: bool = False,
):
    """Plot one-dimensional data as black circular points with error bars.

    When ``unit`` is supplied, the y-axis label includes the uniform bin width,
    for example ``Candidates / 0.25 GeV^2``. ``log_scale=True`` enables a
    logarithmic y axis.
    """

    if ax is None:
        _, ax = plt.subplots()
    centers, counts, errors, edges = binned_data(
        values, bins=bins, range=range, weights=weights
    )
    ax.errorbar(
        centers,
        counts,
        yerr=errors,
        fmt="o",
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=markersize,
        linestyle="none",
        label=label,
        zorder=10,
    )
    if unit is not None:
        ax.set_ylabel(_bin_width_label(edges, unit))
    if log_scale:
        ax.set_yscale("log")
    return ax, counts, errors, edges


def plot_dalitz(
    sample,
    *,
    x: str = "s13",
    y: str = "s23",
    weights=None,
    bins: int = 100,
    range: tuple[float, float] | None = None,
    margin: float = 0.0,
    ax=None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    unit: str | None = "GeV$^2$",
    colorbar: bool = True,
    log_scale: bool = False,
    folded: bool = False,
):
    """Plot a standard two-dimensional Dalitz histogram in one call.

    ``range``, when given, is a single ``(low, high)`` applied to both axes
    (matplotlib's ``hist2d`` otherwise autoscales each axis independently to
    its own data min/max, which is only appropriate for ``folded=False``: for
    ``folded=True`` a shared range is required, see below). ``margin`` -- a
    fraction of the auto-computed span, 0 by default -- pads that same
    exact-data-extent range on both ends instead of starting/ending exactly
    at the min/max value (matplotlib's ``hist2d`` sets the axis view limits
    to the tight data extent, unlike most other plots' default 5% autoscale
    margin); it has no effect when ``range`` is given explicitly. ``x_label``/
    ``y_label`` override the default axis text (which otherwise always
    overwrite whatever the caller may have set on a supplied ``ax``); either
    way, ``unit`` -- ``"GeV$^2$"`` by default, matching the invariant-mass-
    squared convention used everywhere else -- is still appended in brackets
    unless explicitly set to ``None``. The colorbar is labelled with the
    (uniform) bin area, e.g. ``Candidates / 0.05 GeV^4`` -- the
    two-dimensional analogue of ``plot_binned_data``'s
    ``Candidates / <width> <unit>`` y-axis label.

    Set ``log_scale=True`` to display the bin contents with logarithmic color
    normalization. Set ``folded=True`` when ``x`` and ``y`` are two
    exchange-symmetric invariants (two identical daughters sharing the third,
    bachelor particle) to plot only the physically distinct half
    ``x <= y``, folding each point via ``min``/``max`` first.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")

    x_values = _values(sample, x)
    y_values = _values(sample, y)
    hist_range = (range, range) if range is not None else None
    auto_folded_range = folded and hist_range is None
    if folded:
        if auto_folded_range:
            # Force identical bin edges on both axes -- the same
            # x_edges==y_edges convention `HistogramEfficiency`/
            # `HistogramBackground`/`QMIPixel` already require for a folded
            # domain. Without it, matplotlib's `hist2d` autoscales each axis
            # to its own data range; since folding systematically sends the
            # smaller of the pair to `x_values` and the larger to
            # `y_values`, those ranges differ, giving each axis a different
            # bin width and skewing the physical x==y fold boundary away
            # from a clean diagonal.
            low = float(min(np.min(x_values), np.min(y_values)))
            high = float(max(np.max(x_values), np.max(y_values)))
            hist_range = ((low, high), (low, high))
        x_values, y_values = (
            np.minimum(x_values, y_values),
            np.maximum(x_values, y_values),
        )

    if ax is None:
        _, ax = plt.subplots()
    counts, x_edges, y_edges, image = ax.hist2d(
        x_values,
        y_values,
        bins=bins,
        weights=weights,
        range=hist_range,
        norm=LogNorm() if log_scale else None,
    )
    if auto_folded_range:
        # `x_values` (=min) can never reach the shared upper bound used above
        # to align bin edges with `y_values` (=max) -- e.g. at the diagonal
        # corner x==y==high is a single kinematic point, not a range -- so
        # the *binning* needs that full shared span, but *displaying* it on
        # the x axis would just pad the plot with an empty margin. Crop the
        # visible limits to what each folded variable actually reaches,
        # without touching the underlying (correctly aligned) bin edges.
        pad = margin * (high - low)
        ax.set_xlim(float(np.min(x_values)) - pad, float(np.max(x_values)) + pad)
        ax.set_ylim(float(np.min(y_values)) - pad, float(np.max(y_values)) + pad)
    elif range is None and margin > 0:
        # `hist2d` sets the view limits to the exact data extent (unlike most
        # other matplotlib plots' default 5% autoscale margin); restore some
        # breathing room around it when requested.
        x_pad = margin * (float(np.max(x_values)) - float(np.min(x_values)))
        y_pad = margin * (float(np.max(y_values)) - float(np.min(y_values)))
        ax.set_xlim(float(np.min(x_values)) - x_pad, float(np.max(x_values)) + x_pad)
        ax.set_ylim(float(np.min(y_values)) - y_pad, float(np.max(y_values)) + y_pad)
    if x_label is not None:
        x_text = x_label
    elif folded:
        x_text = rf"min(${x}$, ${y}$)"
    else:
        x_text = rf"${x}$"
    if y_label is not None:
        y_text = y_label
    elif folded:
        y_text = rf"max(${x}$, ${y}$)"
    else:
        y_text = rf"${y}$"
    ax.set_xlabel(f"{x_text} [{unit}]" if unit else x_text)
    ax.set_ylabel(f"{y_text} [{unit}]" if unit else y_text)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(
            image, ax=ax, label=_bin_area_label(x_edges, y_edges, "GeV$^4$")
        )
    return ax


def plot_square_dalitz(
    sample,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
    weights=None,
    bins: int = 70,
    ax=None,
    title: str | None = None,
    colorbar: bool = True,
    log_scale: bool = False,
    folded: bool = False,
):
    """Plot a Square-Dalitz histogram from ordinary invariant coordinates.

    Set ``log_scale=True`` to display the bin contents with logarithmic color
    normalization. Set ``folded=True`` when ``pair`` is built from two
    identical daughters to fold ``theta'`` onto ``[0, 0.5]``
    (:func:`~jaxpwa.kinematics.fold_thetaprime`), plotting only the
    physically distinct half.
    """

    data = sample.as_dict() if hasattr(sample, "as_dict") else sample
    mp, tp = invariants_to_square_dalitz(
        data["s12"], data["s13"], data["s23"],
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )
    if folded:
        tp = fold_thetaprime(tp)
    if ax is None:
        _, ax = plt.subplots()
    hist = ax.hist2d(
        np.asarray(mp),
        np.asarray(tp),
        bins=bins,
        weights=weights,
        range=((0, 1), (0, 0.5) if folded else (0, 1)),
        norm=LogNorm() if log_scale else None,
    )
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$ (folded)" if folded else r"$\theta'$")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(hist[3], ax=ax)
    return ax


def _draw_pulls_1d(ax, edges, pulls):
    """Draw a 1D pull bar plot with a shaded +-2 band onto an existing axes.

    Shared by :func:`plot_pulls` and the ``show_pulls=True`` panel of
    ``FitSession``/``CPFitSession.plot_projection``, so both draw pulls
    identically.
    """

    edges = np.asarray(edges)
    pulls = np.asarray(pulls)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.axhspan(-2, 2, color="grey", alpha=0.2, zorder=0)
    ax.bar(centers, pulls, width=np.diff(edges), color="steelblue", zorder=1)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel(r"pull $(o-e)/\sqrt{e}$")
    return ax


def plot_pulls(result, *, ax=None):
    """Plot per-bin pulls from a ``BinnedChi2Result``.

    ``result`` is the return value of
    :func:`~jaxpwa.goodness_of_fit.chi2_from_histograms` or of a
    session's ``goodness_of_fit_projection``/``goodness_of_fit_chi2``.

    Dispatches on ``result.pulls.ndim``: a 1D result draws a bar/stem plot of
    ``(observed-expected)/sqrt(expected)`` against bin center with a shaded
    +-2 reference band; a 2D result draws a diverging ``pcolormesh`` centered
    at 0 over ``result.edges``. Bins dropped by
    :func:`~jaxpwa.goodness_of_fit.chi2_from_histograms` (zero
    expected count) show as ``nan`` and are left blank.
    """

    if ax is None:
        _, ax = plt.subplots()
    pulls = np.asarray(result.pulls)
    if pulls.ndim == 1:
        (edges,) = result.edges
        _draw_pulls_1d(ax, edges, pulls)
    elif pulls.ndim == 2:
        x_edges, y_edges = result.edges
        limit = float(np.nanmax(np.abs(pulls))) or 1.0
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            pulls.T,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
        )
        ax.figure.colorbar(mesh, ax=ax, label=r"pull $(o-e)/\sqrt{e}$")
    else:
        raise ValueError("result.pulls must be 1D or 2D")
    return ax


def plot_contour(result, x: str, y: str, *, cl=(0.68, 0.95), size: int = 100, ax=None):
    """Plot Minos profile-likelihood confidence-region contour(s) in (x, y).

    ``result`` is a fitted ``iminuit.Minuit`` instance -- the direct return
    value of ``Minimizer.fit()`` (and of ``FitSession``/``CPFitSession``/
    ``TimeDependentFitSession.fit()`` with the default ``update_model=False``).
    ``x``/``y`` are the two ``Parameter`` names to scan, e.g. ``"mix.x"``,
    ``"mix.y"``. Not specific to any session: any object with a fitted
    Minuit's ``.mncontour``/``.values`` interface works.

    Each level in ``cl`` is drawn as one closed curve from
    ``Minuit.mncontour(x, y, cl=level, size=size)``, which profiles out every
    *other* free parameter at each scan point (re-minimizing over them) --
    this is the Wilks'-theorem profile-likelihood region, not the cheaper
    Gaussian/covariance-ellipse approximation. Interpretation of ``cl``
    follows ``mncontour`` itself: ``0 < cl < 1`` is a probability (e.g. 0.68,
    0.95, matching the two levels a published (x, y) mixing-parameter
    confidence plot typically shows); ``cl >= 1`` is a number of standard
    deviations of a normal distribution. The fitted point is marked with a
    star. This is one full re-minimization per scan point per level (see
    ``Minuit.mncontour``'s own docs) -- reduce ``size`` for a faster, coarser
    contour, e.g. while iterating on a plot's appearance.

    1D projections of a 2D confidence region are larger than 1D Minos
    intervals at the same confidence level (also documented on
    ``mncontour``); this is expected, not a bug in either.
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    for level in cl:
        points = np.asarray(result.mncontour(x, y, cl=level, size=size))
        label = f"{level * 100:.1f}%" if level < 1 else rf"{level:g}$\sigma$"
        ax.plot(points[:, 0], points[:, 1], label=label)
    ax.plot(
        [float(result.values[x])], [float(result.values[y])],
        marker="*", color="black", markersize=12, linestyle="none",
        label="best fit",
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # Outside the axes: an inside "best" location tends to land on top of the
    # contours/best-fit marker it is supposed to label, especially for a
    # small number of roughly concentric confidence regions.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return ax


__all__ = [
    "binned_data",
    "plot_binned_data",
    "plot_contour",
    "plot_dalitz",
    "plot_pulls",
    "plot_square_dalitz",
]
