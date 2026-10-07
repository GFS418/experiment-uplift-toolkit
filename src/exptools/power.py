"""Minimum detectable effects and sample sizes for a difference in means.

Power is the chance of detecting an effect of a given size if it is real. For
a difference in means it depends on the effect, the two arms' variances, the
sample sizes, and the critical value. The textbook formula assumes the
treatment leaves the variance alone. For a rare-purchase metric that is wrong
in a predictable direction: an effect that works by adding buyers adds
variance too, so the textbook formula overstates power.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _z_total(alpha: float, power: float, critical: float | None) -> float:
    """Critical value plus the power quantile: how many standard errors the effect must span."""
    c = stats.norm.isf(alpha / 2) if critical is None else critical
    return float(c + stats.norm.ppf(power))


def mde(
    var_t: float,
    var_c: float,
    n_t: int,
    n_c: int,
    alpha: float = 0.05,
    power: float = 0.8,
    critical: float | None = None,
) -> float:
    """Smallest true difference a two-sided test detects with probability `power`.

    `critical` overrides the two-sided normal cutoff, e.g. 2.212 for the
    Dunnett family in this project.
    """
    return _z_total(alpha, power, critical) * float(np.sqrt(var_t / n_t + var_c / n_c))


def sample_size(
    effect: float,
    var_t: float,
    var_c: float,
    alpha: float = 0.05,
    power: float = 0.8,
    critical: float | None = None,
) -> float:
    """Customers per arm (equal arms) needed to detect `effect` with probability `power`."""
    return _z_total(alpha, power, critical) ** 2 * (var_t + var_c) / effect**2


def buyer_driven_variance(mean: float, second_moment: float, lift: float) -> float:
    """Treated arm's variance when a relative lift works entirely by adding buyers.

    Scaling the share of nonzero outcomes by (1 + lift), with the nonzero values'
    distribution unchanged, scales both E[Y] and E[Y^2] by (1 + lift). For a 0/1
    metric this is exactly the binomial variance at the new rate.
    """
    return (1 + lift) * second_moment - ((1 + lift) * mean) ** 2


def buyer_driven_mde(
    mean: float,
    second_moment: float,
    n_t: int,
    n_c: int,
    alpha: float = 0.05,
    power: float = 0.8,
    critical: float | None = None,
) -> float:
    """Smallest detectable relative lift when the effect works by adding buyers.

    `mean` and `second_moment` describe the control arm (E[Y] and E[Y^2]).
    Solves lift * mean = z * sqrt(Var_t(lift) / n_t + Var_c / n_c).
    """
    z = _z_total(alpha, power, critical)
    var_c = second_moment - mean**2
    lift_max = second_moment / mean**2 - 1  # beyond this every customer would be a buyer

    def gap(lift: float) -> float:
        var_t = buyer_driven_variance(mean, second_moment, lift)
        return lift * mean - z * np.sqrt(var_t / n_t + var_c / n_c)

    return float(optimize.brentq(gap, 1e-12, lift_max))


def buyer_driven_sample_size(
    mean: float,
    second_moment: float,
    lift: float,
    alpha: float = 0.05,
    power: float = 0.8,
    critical: float | None = None,
) -> float:
    """Customers per arm needed to detect a relative lift that works by adding buyers."""
    var_t = buyer_driven_variance(mean, second_moment, lift)
    return sample_size(lift * mean, var_t, second_moment - mean**2, alpha, power, critical)
