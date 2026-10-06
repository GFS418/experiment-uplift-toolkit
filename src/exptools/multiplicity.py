"""Multiple-comparison procedures: Dunnett (two versions) and Benjamini-Hochberg."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from exptools.bootstrap import Arm, Replicates


@dataclass(frozen=True)
class Dunnett:
    """Many-to-one comparisons with a shared control, family-wise error controlled."""

    arms: tuple[str, ...]
    estimate: np.ndarray
    se: np.ndarray
    critical_value: float
    ci_low: np.ndarray
    ci_high: np.ndarray
    p_adjusted: np.ndarray

    def rejects(self, alpha: float = 0.05) -> np.ndarray:
        return self.p_adjusted < alpha


def dunnett_maxt(
    treatments: Mapping[str, Arm],
    control: Arm,
    metric: str,
    treatment_reps: Mapping[str, Replicates],
    control_reps: Replicates,
    confidence: float = 0.95,
) -> Dunnett:
    """Studentized bootstrap max-T version of Dunnett (single-step, Westfall-Young style).

    Dunnett's idea is to find the cutoff c for which all k studentized
    differences stay inside +/-c together 95% of the time. The parametric test
    gets c from a multivariate t, assuming normal data and equal variances;
    here c is the 95th percentile of max_j |T*_j| over bootstrap resamples,
    where T*_j = (d*_j - d_j) / se*_j. Each resample shares one control draw
    across comparisons, so the correlation Dunnett relies on is preserved.
    """
    names = tuple(treatments)
    estimate = np.array([treatments[a].mean(metric) - control.mean(metric) for a in names])
    se = np.sqrt(
        [treatments[a].var(metric) / treatments[a].n + control.var(metric) / control.n for a in names]
    )
    d_star = np.stack([treatment_reps[a].mean - control_reps.mean for a in names])
    se_star = np.sqrt(
        np.stack([treatment_reps[a].var / treatments[a].n + control_reps.var / control.n for a in names])
    )
    if (se_star == 0).any():
        raise ValueError("a bootstrap resample has zero variance; studentizing is undefined")
    t_max = np.abs((d_star - estimate[:, None]) / se_star).max(axis=0)
    c = float(np.quantile(t_max, confidence))
    exceed = (t_max[None, :] >= np.abs(estimate / se)[:, None]).sum(axis=1)
    p_adjusted = (1 + exceed) / (1 + t_max.size)
    return Dunnett(names, estimate, se, c, estimate - c * se, estimate + c * se, p_adjusted)


def dunnett_parametric(
    treatments: Mapping[str, Arm],
    control: Arm,
    metric: str,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> Dunnett:
    """Classical Dunnett via scipy.stats.dunnett: normal data, one pooled variance."""
    names = tuple(treatments)
    groups = [treatments[a] for a in names]
    result = stats.dunnett(*(g.expand(metric) for g in groups), control=control.expand(metric), rng=rng)
    ci = result.confidence_interval(confidence)
    estimate = np.array([g.mean(metric) - control.mean(metric) for g in groups])
    all_arms = [*groups, control]
    pooled_var = sum((g.n - 1) * g.var(metric) for g in all_arms) / sum(g.n - 1 for g in all_arms)
    se = np.sqrt([pooled_var * (1 / g.n + 1 / control.n) for g in groups])
    c = float(np.mean((ci.high - ci.low) / (2 * se)))
    return Dunnett(names, estimate, se, c, np.asarray(ci.low), np.asarray(ci.high), np.asarray(result.pvalue))


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Step-up BH: adjusted p-values and which hypotheses are rejected at FDR q.

    The i-th smallest p-value is compared with q * i / m; everything up to the
    largest i that passes is rejected. Adjusted p-values express the same rule:
    p_adj(i) = min over j >= i of p(j) * m / j, capped at 1.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= q
