"""Two-arm comparisons of means: Welch's t and the BCa bootstrap interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from exptools.bootstrap import Arm, Replicates, acceleration, bca_interval, difference_influence


@dataclass(frozen=True)
class Welch:
    estimate: float
    se: float
    df: float
    ci: tuple[float, float]
    p_value: float


def welch(treat: Arm, control: Arm, metric: str, confidence: float = 0.95) -> Welch:
    """Welch's unequal-variance t-test and interval for mean(treat) - mean(control)."""
    vt = treat.var(metric) / treat.n
    vc = control.var(metric) / control.n
    se = float(np.sqrt(vt + vc))
    df = (vt + vc) ** 2 / (vt**2 / (treat.n - 1) + vc**2 / (control.n - 1))
    estimate = treat.mean(metric) - control.mean(metric)
    q = stats.t.ppf(0.5 + confidence / 2, df)
    p_value = 2 * stats.t.sf(abs(estimate / se), df)
    return Welch(estimate, se, float(df), (estimate - q * se, estimate + q * se), float(p_value))


@dataclass(frozen=True)
class Comparison:
    metric: str
    treatment: str
    control: str
    estimate: float
    welch: Welch
    bca: tuple[float, float]


def compare(
    treat: Arm,
    control: Arm,
    metric: str,
    treat_reps: Replicates,
    control_reps: Replicates,
    *,
    names: tuple[str, str] = ("treatment", "control"),
    confidence: float = 0.95,
) -> Comparison:
    """Difference in means with a Welch interval and a BCa bootstrap interval."""
    estimate = treat.mean(metric) - control.mean(metric)
    accel = acceleration(difference_influence(treat, control, metric))
    bca = bca_interval(estimate, treat_reps.mean - control_reps.mean, accel, confidence)
    return Comparison(metric, *names, estimate, welch(treat, control, metric, confidence), bca)
