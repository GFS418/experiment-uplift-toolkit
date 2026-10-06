"""Covariate adjustment for experiments: CUPED and Lin's regression adjustment.

Both use pre-treatment covariates to remove outcome variance that has nothing
to do with the treatment. Randomization makes the covariates balanced in
expectation, so the adjustment cannot bias the estimate; it only shrinks the
noise. A covariate that the treatment itself can change breaks that guarantee,
which is why every caller must pass pre-treatment columns only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    """A difference between two arms with a standard error and a normal-theory interval."""

    estimate: float
    se: float

    def ci(self, confidence: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + confidence / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se

    @property
    def p_value(self) -> float:
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))


def difference_in_means(y: np.ndarray, arm: np.ndarray, treatment: str, control: str) -> Estimate:
    """Unadjusted difference with the unequal-variance (Welch / Neyman) standard error."""
    yt, yc = y[arm == treatment], y[arm == control]
    se = np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size)
    return Estimate(float(yt.mean() - yc.mean()), float(se))


def _demean_within_arms(v: np.ndarray, arm: np.ndarray) -> np.ndarray:
    out = v.astype(float)
    for a in np.unique(arm):
        mask = arm == a
        out[mask] -= out[mask].mean()
    return out


def cuped_theta(y: np.ndarray, x: np.ndarray, arm: np.ndarray) -> float:
    """Slope of y on x within arms, pooled across arms.

    Equal to the coefficient on x in an OLS of y on arm dummies and x, so CUPED
    with one covariate is ANCOVA. Demeaning within arms keeps the treatment
    effect itself out of the slope.
    """
    xd, yd = _demean_within_arms(x, arm), _demean_within_arms(y, arm)
    return float(xd @ yd / (xd @ xd))


def within_arm_correlation(y: np.ndarray, x: np.ndarray, arm: np.ndarray) -> float:
    """Correlation of y and x after removing arm means; CUPED removes about its square."""
    xd, yd = _demean_within_arms(x, arm), _demean_within_arms(y, arm)
    return float(xd @ yd / np.sqrt((xd @ xd) * (yd @ yd)))


def cuped(
    y: np.ndarray, x: np.ndarray, arm: np.ndarray, treatment: str, control: str, theta: float | None = None
) -> Estimate:
    """CUPED: compare y - theta * (x - mean(x)) across arms.

    The adjusted difference is the raw difference minus theta times the chance
    imbalance in x. The standard error treats theta as known; estimating it
    adds variance of order 1/n, negligible at this sample size (the simulation
    checks this).
    """
    theta = cuped_theta(y, x, arm) if theta is None else theta
    return difference_in_means(y - theta * (x - x.mean()), arm, treatment, control)


def ols_hc2(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients with the HC2 heteroskedasticity-robust covariance.

    HC2 rescales each squared residual by 1 / (1 - leverage). With arm dummies
    alone it reproduces the Welch standard error exactly, which is why Lin
    (2013) recommends it for experiments.
    """
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    leverage = np.sum((design @ xtx_inv) * design, axis=1)  # diagonal of the hat matrix
    meat = (design * (resid**2 / (1 - leverage))[:, None]).T @ design
    return beta, xtx_inv @ meat @ xtx_inv


def covariate_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Numeric columns as is; categoricals as 0/1 dummies with the first level dropped."""
    return pd.get_dummies(df[list(columns)], drop_first=True, dtype=float).to_numpy()


@dataclass(frozen=True)
class LinFit:
    """Lin's fully interacted regression: one adjusted effect per treatment arm."""

    arms: tuple[str, ...]  # treatment arms, in coefficient order
    coef: np.ndarray  # adjusted effect of each treatment arm vs control
    cov: np.ndarray  # HC2 covariance of those effects

    def vs_control(self, treatment: str) -> Estimate:
        i = self.arms.index(treatment)
        return Estimate(float(self.coef[i]), float(np.sqrt(self.cov[i, i])))

    def contrast(self, first: str, second: str) -> Estimate:
        """Effect of `first` minus effect of `second`, both vs the shared control."""
        i, j = self.arms.index(first), self.arms.index(second)
        var = self.cov[i, i] + self.cov[j, j] - 2 * self.cov[i, j]
        return Estimate(float(self.coef[i] - self.coef[j]), float(np.sqrt(var)))


def lin(y: np.ndarray, covariates: np.ndarray, arm: np.ndarray, control: str) -> LinFit:
    """Lin (2013) regression adjustment for any number of arms.

    Regress y on arm dummies, covariates centered at their overall mean, and
    every arm-by-covariate interaction. Each arm gets its own slopes, so the
    adjustment stays efficient when the effect varies with the covariates, and
    centering makes each arm coefficient the effect at the average customer.
    """
    treatments = tuple(a for a in np.unique(arm) if a != control)
    centered = covariates - covariates.mean(axis=0)
    dummies = np.column_stack([(arm == a).astype(float) for a in treatments])
    interactions = np.column_stack([dummies[:, [k]] * centered for k in range(len(treatments))])
    design = np.column_stack([np.ones(len(y)), dummies, centered, interactions])
    beta, cov = ols_hc2(design, y.astype(float))
    idx = slice(1, 1 + len(treatments))
    return LinFit(treatments, beta[idx], cov[idx, idx])


def variance_reduction(adjusted: Estimate, unadjusted: Estimate) -> float:
    """Share of the estimator's variance removed: 1 - Var(adjusted) / Var(unadjusted)."""
    return 1 - adjusted.se**2 / unadjusted.se**2
