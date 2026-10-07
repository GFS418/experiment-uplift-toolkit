import numpy as np
import pytest
from scipy import stats

from exptools.power import (
    buyer_driven_mde,
    buyer_driven_sample_size,
    buyer_driven_variance,
    mde,
    sample_size,
)


def test_mde_known_value():
    # (1.96 + 0.8416) * sqrt(1/1000 + 1/1000)
    assert mde(1.0, 1.0, 1_000, 1_000) == pytest.approx(2.8016 * np.sqrt(2 / 1_000), rel=1e-4)


def test_sample_size_inverts_mde():
    n = sample_size(0.1, 2.0, 1.5)
    assert mde(2.0, 1.5, n, n) == pytest.approx(0.1)


def test_critical_value_override():
    assert mde(1.0, 1.0, 500, 500, critical=2.212) > mde(1.0, 1.0, 500, 500)


def test_buyer_driven_variance_is_binomial_for_a_zero_one_metric():
    p, lift = 0.01, 0.5
    assert buyer_driven_variance(p, p, lift) == pytest.approx(p * 1.5 * (1 - p * 1.5))


def test_buyer_driven_mde_solves_its_power_equation():
    # A spend-like metric: 1% buyers spending about $114 (second moment from the buyer spread).
    mean, second = 0.01 * 114, 0.01 * (114**2 + 103**2)
    lift = buyer_driven_mde(mean, second, 21_000, 21_000)
    se = np.sqrt(buyer_driven_variance(mean, second, lift) / 21_000 + (second - mean**2) / 21_000)
    assert stats.norm.sf(1.959964 - lift * mean / se) == pytest.approx(0.8, abs=1e-3)


def test_buyer_driven_mde_exceeds_the_equal_variance_answer():
    # Adding buyers adds variance, so the honest MDE is larger than the textbook one.
    mean, second = 1.14, 0.01 * (114**2 + 103**2)
    var = second - mean**2
    textbook_lift = mde(var, var, 21_000, 21_000) / mean
    assert buyer_driven_mde(mean, second, 21_000, 21_000) > textbook_lift


def test_buyer_driven_sample_size_round_trips():
    mean, second = 1.14, 0.01 * (114**2 + 103**2)
    n = buyer_driven_sample_size(mean, second, 0.25)
    assert buyer_driven_mde(mean, second, n, n) == pytest.approx(0.25, rel=1e-6)
