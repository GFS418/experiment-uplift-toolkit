"""Exact, fast bootstrap for differences in means.

A bootstrap resample of an arm affects a mean only through how many times each
distinct outcome row is drawn. Hillstrom's control arm has 21,306 customers
but only 98 distinct (visit, conversion, spend) rows, so one resample is a
single multinomial draw over 98 categories instead of 21,306 index draws.
This is the ordinary nonparametric bootstrap, not an approximation, and it is
what makes thousands of simulated experiments with B = 10,000 affordable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Arm:
    """One arm's customers, compressed to distinct outcome rows and their counts."""

    rows: np.ndarray  # (m, k) outcome rows; need not be distinct
    counts: np.ndarray  # (m,) customers with each row; zeros allowed
    metrics: tuple[str, ...]

    @classmethod
    def from_frame(cls, df: pd.DataFrame, metrics: Sequence[str]) -> Arm:
        sizes = df.groupby(list(metrics), sort=True).size()
        rows = np.array(sizes.index.tolist(), dtype=float).reshape(len(sizes), len(metrics))
        return cls(rows, sizes.to_numpy(dtype=np.int64), tuple(metrics))

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def column(self, metric: str) -> np.ndarray:
        return self.rows[:, self.metrics.index(metric)]

    def mean(self, metric: str) -> float:
        return float(self.counts @ self.column(metric) / self.n)

    def var(self, metric: str) -> float:
        """Sample variance (ddof = 1)."""
        centered = self.column(metric) - self.mean(metric)
        return float(self.counts @ centered**2 / (self.n - 1))

    def expand(self, metric: str) -> np.ndarray:
        """The metric as one value per customer, for functions that need raw arrays."""
        return np.repeat(self.column(metric), self.counts)

    def resample(self, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
        """How often each row is drawn in each bootstrap resample, shape (B, m)."""
        return rng.multinomial(self.n, self.counts / self.n, size=n_resamples)


@dataclass(frozen=True)
class Replicates:
    """Bootstrap distribution of an arm's mean and sample variance for one metric."""

    mean: np.ndarray  # (B,)
    var: np.ndarray  # (B,)


def replicate_moments(arm: Arm, metric: str, draws: np.ndarray) -> Replicates:
    x, n = arm.column(metric), arm.n
    m1 = draws @ x / n
    m2 = draws @ (x * x) / n
    return Replicates(mean=m1, var=np.maximum(m2 - m1**2, 0.0) * n / (n - 1))


def bootstrap_replicates(
    arms: dict[str, Arm], metrics: Sequence[str], n_resamples: int, rng: np.random.Generator
) -> dict[str, dict[str, Replicates]]:
    """Resample every arm independently, once, and summarize each metric.

    All comparisons reuse these draws. That matters for Dunnett: both
    comparisons with the control must see the same control resample, or the
    correlation between them, which Dunnett exists to exploit, is lost.
    """
    out = {}
    for name, arm in arms.items():
        draws = arm.resample(n_resamples, rng)
        out[name] = {metric: replicate_moments(arm, metric, draws) for metric in metrics}
    return out


def acceleration(influence: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """BCa acceleration from each sample's influence values and their counts.

    Same estimator as scipy.stats.bootstrap for multi-sample statistics:
    a = 1/6 * sum_j sum_i U_ji^3 / n_j^3 / (sum_j sum_i U_ji^2 / n_j^2)^(3/2),
    where U_ji are jackknife influence values. For means they have closed
    forms, so no leave-one-out loop over 21,000 customers is needed.
    """
    num = sum(counts @ u**3 / counts.sum() ** 3 for u, counts in influence)
    den = sum(counts @ u**2 / counts.sum() ** 2 for u, counts in influence)
    return float(num / (6 * den**1.5))


def jackknife_influence(
    arms: Sequence[Arm], metric: str, statistic: Callable[..., np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact jackknife influence values U = (n - 1) * (mean_i theta_(i) - theta_(i)).

    For a statistic of the arms' means, leaving one customer out changes only
    that arm's mean, to (n * mean - x) / (n - 1), so every leave-one-out value
    has a closed form: one evaluation per distinct row, not per customer.
    `statistic` takes one array of means per arm and must be vectorized.
    """
    means = [arm.mean(metric) for arm in arms]
    influence = []
    for j, arm in enumerate(arms):
        loo_mean = (arm.n * means[j] - arm.column(metric)) / (arm.n - 1)
        args = [np.full_like(loo_mean, mu) for mu in means]
        args[j] = loo_mean
        theta = statistic(*args)
        theta_bar = arm.counts @ theta / arm.n
        influence.append(((arm.n - 1) * (theta_bar - theta), arm.counts))
    return influence


def difference_influence(treat: Arm, control: Arm, metric: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Jackknife influence values of mean(treat) - mean(control)."""
    return jackknife_influence([treat, control], metric, np.subtract)


def ratio_influence(treat: Arm, control: Arm, metric: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Jackknife influence values of mean(treat) / mean(control), for relative lift."""
    return jackknife_influence([treat, control], metric, np.divide)


def bca_interval(
    estimate: float, replicates: np.ndarray, accel: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Efron's bias-corrected and accelerated percentile interval.

    z0 corrects for the bootstrap distribution being off-center relative to the
    estimate (ties counted half, as in SciPy, which matters for 0/1 metrics);
    the acceleration corrects for the standard error changing with the
    parameter, which is what skewed data like spend produce.
    """
    b = replicates.size
    below = (np.count_nonzero(replicates < estimate) + np.count_nonzero(replicates <= estimate)) / (2 * b)
    if not 0 < below < 1:
        raise ValueError("estimate lies outside the bootstrap distribution; BCa is undefined")
    z0 = stats.norm.ppf(below)
    tail = (1 - confidence) / 2
    z = stats.norm.ppf([tail, 1 - tail])
    levels = stats.norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
    low, high = np.quantile(replicates, levels)
    return float(low), float(high)
