import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exptools.bootstrap import (
    Arm,
    acceleration,
    bca_interval,
    difference_influence,
    ratio_influence,
    replicate_moments,
)


def _arm(values) -> Arm:
    return Arm.from_frame(pd.DataFrame({"y": values}), ["y"])


def _zero_heavy(n: int, rng: np.random.Generator, buy_rate: float = 0.05) -> np.ndarray:
    # Mostly zeros with a lognormal tail, like spend.
    return np.where(rng.random(n) < buy_rate, np.round(rng.lognormal(4.5, 0.8, n), 2), 0.0)


def test_compressed_arm_matches_raw_moments():
    rng = np.random.default_rng(0)
    raw = _zero_heavy(2_000, rng)
    arm = _arm(raw)
    assert arm.n == raw.size
    assert len(arm.counts) < raw.size  # the zeros collapse into one row
    assert arm.mean("y") == pytest.approx(raw.mean())
    assert arm.var("y") == pytest.approx(raw.var(ddof=1))
    assert np.array_equal(np.sort(arm.expand("y")), np.sort(raw))


def test_replicates_have_bootstrap_mean_and_variance():
    # The bootstrap distribution of a mean is centered on the sample mean with
    # variance (plug-in variance) / n.
    rng = np.random.default_rng(1)
    arm = _arm(_zero_heavy(3_000, rng))
    reps = replicate_moments(arm, "y", arm.resample(20_000, rng))
    plug_in_sd = np.sqrt(arm.var("y") * (arm.n - 1) / arm.n / arm.n)
    assert reps.mean.mean() == pytest.approx(arm.mean("y"), abs=4 * plug_in_sd / np.sqrt(20_000))
    assert reps.mean.std() == pytest.approx(plug_in_sd, rel=0.03)
    assert reps.var.mean() == pytest.approx(arm.var("y"), rel=0.02)


def _jackknife_acceleration(samples, statistic):
    # Brute force, written to mirror scipy.stats.bootstrap's multi-sample BCa.
    nums, dens = [], []
    for j, sample in enumerate(samples):
        loo = []
        for i in range(sample.size):
            reduced = list(samples)
            reduced[j] = np.delete(sample, i)
            loo.append(statistic(*reduced))
        loo = np.array(loo)
        n = sample.size
        u = (n - 1) * (loo.mean() - loo)
        nums.append((u**3).sum() / n**3)
        dens.append((u**2).sum() / n**2)
    return sum(nums) / (6 * sum(dens) ** 1.5)


def test_difference_acceleration_equals_brute_force_jackknife():
    rng = np.random.default_rng(2)
    t, c = _zero_heavy(300, rng, 0.2), _zero_heavy(250, rng, 0.1)
    expected = _jackknife_acceleration([t, c], lambda a, b: a.mean() - b.mean())
    assert acceleration(difference_influence(_arm(t), _arm(c), "y")) == pytest.approx(expected, rel=1e-9)


def test_ratio_acceleration_equals_brute_force_jackknife():
    # The treatment and control terms nearly cancel here, so a first-order
    # (delta-method) approximation was off by 2x; the closed form is exact.
    rng = np.random.default_rng(3)
    t, c = _zero_heavy(400, rng, 0.3), _zero_heavy(400, rng, 0.25)
    expected = _jackknife_acceleration([t, c], lambda a, b: a.mean() / b.mean())
    assert acceleration(ratio_influence(_arm(t), _arm(c), "y")) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("kind", ["zero_heavy", "binary"])
def test_bca_matches_scipy_given_the_same_replicates(kind):
    rng = np.random.default_rng(4)
    if kind == "zero_heavy":
        t, c = _zero_heavy(400, rng, 0.15), _zero_heavy(380, rng, 0.1)
    else:  # many ties: exercises the half-weight tie rule in z0
        t, c = (rng.random(400) < 0.12).astype(float), (rng.random(380) < 0.08).astype(float)
    res = stats.bootstrap(
        (t, c),
        lambda a, b, axis: a.mean(axis=axis) - b.mean(axis=axis),
        method="BCa",
        n_resamples=5_000,
        vectorized=True,
        rng=rng,
    )
    accel = acceleration(difference_influence(_arm(t), _arm(c), "y"))
    low, high = bca_interval(t.mean() - c.mean(), res.bootstrap_distribution, accel)
    assert low == pytest.approx(res.confidence_interval.low, rel=1e-9)
    assert high == pytest.approx(res.confidence_interval.high, rel=1e-9)


def test_bca_refuses_an_estimate_outside_the_replicates():
    with pytest.raises(ValueError, match="BCa is undefined"):
        bca_interval(10.0, np.arange(5.0), 0.0)
