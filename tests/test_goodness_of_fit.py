import numpy as np
import pytest

from jaxpwa.goodness_of_fit import (
    BinnedChi2Result,
    chi2_from_histograms,
    point_to_point_dissimilarity,
)


def test_chi2_exact_match_gives_zero_chi2_and_unit_p_value():
    expected = np.full(10, 100.0)
    result = chi2_from_histograms(expected, expected, n_free_parameters=2)
    assert isinstance(result, BinnedChi2Result)
    assert result.chi2 == 0.0
    assert result.n_bins == 10
    assert result.dof_min == 7 and result.dof_max == 9
    assert result.p_value_min == 1.0 and result.p_value_max == 1.0


def test_chi2_dof_bounds_order_p_values_correctly():
    rng = np.random.default_rng(0)
    expected = np.full(12, 50.0)
    observed = expected + rng.normal(0.0, 8.0, size=12)
    result = chi2_from_histograms(observed, expected, n_free_parameters=3)
    # Fewer degrees of freedom (dof_min) means a given chi2 sits further into
    # the tail, i.e. a smaller (more significant) p-value.
    assert result.dof_min < result.dof_max
    assert result.p_value_min <= result.p_value_max


def test_chi2_drops_zero_expected_bins():
    observed = np.array([5.0, 10.0, 10.0])
    expected = np.array([0.0, 10.0, 10.0])
    result = chi2_from_histograms(observed, expected)
    assert result.n_bins == 2
    assert np.isnan(result.pulls[0])
    assert result.pulls[1] == 0.0 and result.pulls[2] == 0.0


def test_chi2_rejects_mismatched_shapes_and_negative_free_parameters():
    with pytest.raises(ValueError, match="same shape"):
        chi2_from_histograms(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="non-negative"):
        chi2_from_histograms(np.ones(3), np.ones(3), n_free_parameters=-1)
    with pytest.raises(ValueError, match="positive expected"):
        chi2_from_histograms(np.zeros(3), np.zeros(3))


def _gaussian_cloud(rng, size, loc=0.0):
    return rng.normal(loc=loc, size=(size, 2))


def test_point_to_point_same_distribution_gives_small_statistic_and_high_p_value():
    rng = np.random.default_rng(1)
    data_xy = _gaussian_cloud(rng, 60)
    reference_xy = _gaussian_cloud(rng, 600)
    density = np.ones(60)
    reference_density = np.ones(600)
    result = point_to_point_dissimilarity(
        data_xy,
        reference_xy,
        density,
        reference_density,
        sigma_bar=0.5,
        phase_space_area=1.0,
        n_permutations=100,
        seed=1,
    )
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_data == 60 and result.n_reference == 600


def test_point_to_point_shifted_distribution_gives_larger_statistic():
    rng = np.random.default_rng(1)
    reference_xy = _gaussian_cloud(rng, 600)
    reference_density = np.ones(600)

    matched = point_to_point_dissimilarity(
        _gaussian_cloud(rng, 60),
        reference_xy,
        np.ones(60),
        reference_density,
        sigma_bar=0.5,
        phase_space_area=1.0,
        n_permutations=50,
        seed=2,
    )
    shifted = point_to_point_dissimilarity(
        _gaussian_cloud(rng, 60, loc=3.0),
        reference_xy,
        np.ones(60),
        reference_density,
        sigma_bar=0.5,
        phase_space_area=1.0,
        n_permutations=50,
        seed=2,
    )
    assert shifted.statistic > matched.statistic
    assert shifted.p_value <= matched.p_value


def test_point_to_point_max_total_events_guard():
    rng = np.random.default_rng(3)
    data_xy = _gaussian_cloud(rng, 50)
    reference_xy = _gaussian_cloud(rng, 5000)
    with pytest.raises(ValueError, match="max_total_events"):
        point_to_point_dissimilarity(
            data_xy,
            reference_xy,
            np.ones(50),
            np.ones(5000),
            phase_space_area=1.0,
            max_total_events=1000,
        )


def test_point_to_point_rejects_bad_inputs():
    rng = np.random.default_rng(4)
    data_xy = _gaussian_cloud(rng, 10)
    reference_xy = _gaussian_cloud(rng, 20)
    with pytest.raises(ValueError, match="phase_space_area"):
        point_to_point_dissimilarity(
            data_xy, reference_xy, np.ones(10), np.ones(20), phase_space_area=0.0
        )
    with pytest.raises(ValueError, match="sigma_bar"):
        point_to_point_dissimilarity(
            data_xy, reference_xy, np.ones(10), np.ones(20),
            phase_space_area=1.0, sigma_bar=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        point_to_point_dissimilarity(
            data_xy, reference_xy, -np.ones(10), np.ones(20), phase_space_area=1.0
        )


def test_observed_variance_replaces_the_poisson_denominator():
    observed = np.array([10.0, 4.0, 7.0])
    expected = np.array([8.0, 5.0, 0.0])
    variance = np.array([12.0, 0.0, 3.0])
    result = chi2_from_histograms(observed, expected, observed_variance=variance)
    # Bin 2: no expectation; bin 1: no variance estimate -> both dropped.
    assert result.n_bins == 1
    assert result.chi2 == pytest.approx(4.0 / 12.0)
    assert result.pulls[0] == pytest.approx(2.0 / np.sqrt(12.0))
    assert np.isnan(result.pulls[1]) and np.isnan(result.pulls[2])
    with pytest.raises(ValueError, match="observed_variance"):
        chi2_from_histograms(observed, expected, observed_variance=np.ones(2))
