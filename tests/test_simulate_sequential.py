import numpy as np
import pytest
from scipy import stats

from exptools.sequential import equally_spaced, naive_false_positive_rate
from exptools.simulate_sequential import first_crossing, inject_buyers, simulate_z_paths, z_path


def test_z_path_is_welch_on_each_prefix():
    rng = np.random.default_rng(0)
    t, c = rng.exponential(size=1_000), rng.exponential(size=800)
    fractions = np.array([0.25, 0.5, 1.0])
    z = z_path(t, c, fractions)
    for f, value in zip(fractions, z, strict=True):
        ref = stats.ttest_ind(t[: round(f * 1_000)], c[: round(f * 800)], equal_var=False).statistic
        assert value == pytest.approx(ref)


def test_z_is_zero_before_any_variation():
    zeros = np.zeros(100)
    assert z_path(zeros, zeros, np.array([0.5, 1.0])).tolist() == [0.0, 0.0]


def test_simulated_peeking_on_normal_data_matches_the_recursion():
    pool = np.random.default_rng(1).normal(size=50_000)
    paths = simulate_z_paths(pool, 2_000, 2_000, equally_spaced(5), n_sims=4_000, seed=2)
    rate = np.mean(first_crossing(paths, np.full(5, 1.959964)) >= 0)
    expected = naive_false_positive_rate(5)
    assert abs(rate - expected) < 4 * np.sqrt(expected * (1 - expected) / 4_000)


def test_injected_buyers_lift_mean_spend_as_requested():
    rng = np.random.default_rng(3)
    pool = np.where(rng.random(20_000) < 0.01, rng.lognormal(4.5, 0.7, 20_000), 0.0)
    means = [inject_buyers(rng.choice(pool, 20_000), pool, 0.65, rng).mean() for _ in range(400)]
    assert np.mean(means) == pytest.approx(1.65 * pool.mean(), rel=0.02)


def test_first_crossing_finds_the_first_look_over_the_boundary():
    paths = np.array([[0.5, 2.5, 3.0], [0.1, 0.2, 0.3], [-4.0, 0.0, 0.0]])
    assert first_crossing(paths, np.array([3.0, 2.0, 2.0])).tolist() == [1, -1, 0]
