import numpy as np
import pytest
from scipy import stats

from exptools.sequential import (
    crossing_probabilities,
    equally_spaced,
    naive_false_positive_rate,
    obf_spending,
    operating_characteristics,
    spending_boundaries,
)


def test_spending_function_spends_alpha_by_the_end_and_little_early():
    assert obf_spending(1.0) == pytest.approx(0.05)
    t = np.linspace(0.01, 1, 50)
    assert np.all(np.diff(obf_spending(t)) > 0)
    assert obf_spending(0.2) < 1e-5


def test_spending_function_matches_gsdesign_sfldof_per_side():
    # gsDesign: f(t) = 2 - 2 * Phi(Phi^-1(1 - alpha/2) / sqrt(t)) at one-sided alpha = 0.025.
    t = np.array([0.1, 0.5, 0.9])
    per_side = 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - 0.025 / 2) / np.sqrt(t))
    assert obf_spending(t) == pytest.approx(2 * per_side)


def test_five_look_boundaries_match_published_values():
    published = [4.877, 3.357, 2.680, 2.290, 2.031]
    assert spending_boundaries(equally_spaced(5)) == pytest.approx(published, abs=6e-4)


@pytest.mark.parametrize("looks", [3, 14])
def test_boundaries_spend_exactly_alpha(looks):
    times = equally_spaced(looks)
    assert crossing_probabilities(spending_boundaries(times), times).reject == pytest.approx(0.05, abs=1e-6)


def test_boundaries_hold_alpha_in_a_brownian_motion_simulation():
    times = equally_spaced(14)
    bounds = spending_boundaries(times)
    rng = np.random.default_rng(0)
    steps = rng.normal(size=(200_000, 14)) * np.sqrt(np.diff(np.concatenate([[0.0], times])))
    z = np.cumsum(steps, axis=1) / np.sqrt(times)
    rate = np.mean((np.abs(z) >= bounds).any(axis=1))
    assert abs(rate - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 200_000)


@pytest.mark.parametrize(
    "looks, published",
    # Armitage, McPherson and Rowe (1969): repeated tests at nominal 5%, equally spaced.
    [(2, 0.083), (5, 0.142), (10, 0.193), (20, 0.248), (50, 0.320), (100, 0.374)],
)
def test_naive_peeking_matches_the_classic_table(looks, published):
    assert naive_false_positive_rate(looks) == pytest.approx(published, abs=1.5e-3)


def test_single_look_with_drift_is_ordinary_power():
    drift = 2.8
    expected = stats.norm.sf(1.959964 - drift) + stats.norm.cdf(-1.959964 - drift)
    assert crossing_probabilities(np.array([1.959964]), np.array([1.0]), drift).reject == pytest.approx(
        expected
    )


def test_sequential_design_trades_a_little_power_for_earlier_stopping():
    times = equally_spaced(14)
    bounds = spending_boundaries(times)
    drift = 3.0  # roughly 85% power for a single test at the end
    oc = operating_characteristics(bounds, times, drift)
    fixed = stats.norm.sf(1.959964 - drift)
    assert fixed - 0.03 < oc.power < fixed  # a small power cost
    assert oc.expected_information < 0.85  # stops early on average
    assert oc.stop.sum() == pytest.approx(1.0)


def test_rejects_looks_that_do_not_end_at_full_information():
    with pytest.raises(ValueError, match="ending at 1"):
        spending_boundaries(np.array([0.5, 0.9]))
