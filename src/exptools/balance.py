"""Covariate balance checks for a randomized experiment.

In an RCT, imbalance in pre-treatment covariates can only come from chance or
from a broken randomization, so these checks audit the assignment mechanism.
The common "|SMD| > 0.1" rule comes from observational studies; with tens of
thousands of units per arm, chance alone produces SMDs with a standard
deviation near 0.01, so each SMD is reported next to its z-score on the
chance scale, plus one joint test across all covariates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _indicators(df: pd.DataFrame, covariates: Sequence[str], *, drop_first: bool) -> pd.DataFrame:
    """Numeric columns pass through; each categorical becomes 0/1 columns per level."""
    return pd.get_dummies(df[list(covariates)], drop_first=drop_first, dtype=float)


def standardized_differences(
    df: pd.DataFrame,
    assignment: str,
    control: str,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Standardized mean difference of every covariate, each arm vs control.

    smd = (mean_arm - mean_control) / sqrt((var_arm + var_control) / 2).
    Under randomization, smd is roughly normal with sd sqrt(1/n_arm + 1/n_control),
    so z = smd / that sd says whether an SMD is larger than chance allows.
    """
    x = _indicators(df, covariates, drop_first=False)
    is_control = (df[assignment] == control).to_numpy()
    control_x = x[is_control]
    rows = []
    for arm in sorted(set(df[assignment]) - {control}):
        arm_x = x[(df[assignment] == arm).to_numpy()]
        smd = (arm_x.mean() - control_x.mean()) / np.sqrt((arm_x.var() + control_x.var()) / 2)
        null_sd = np.sqrt(1 / len(arm_x) + 1 / len(control_x))
        rows.append(
            pd.DataFrame(
                {"arm": arm, "covariate": smd.index, "smd": smd.to_numpy(), "z": smd.to_numpy() / null_sd}
            )
        )
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True)
class JointBalanceResult:
    lr_stat: float
    df: int
    p_value: float
    n_params: int


def joint_balance_test(
    df: pd.DataFrame,
    assignment: str,
    covariates: Sequence[str],
) -> JointBalanceResult:
    """Likelihood-ratio test that the covariates do not predict assignment.

    Fits a multinomial logit of arm on all covariates and compares it with the
    intercept-only model. Under valid randomization the covariates carry no
    information about the arm, so the p-value is uniform on (0, 1).
    """
    x = sm.add_constant(_indicators(df, covariates, drop_first=True))
    arm_codes = pd.Categorical(df[assignment]).codes
    fit = sm.MNLogit(arm_codes, x).fit(disp=0, maxiter=200)
    if not fit.mle_retvals["converged"]:
        raise RuntimeError("multinomial logit did not converge")
    n_arms = len(np.unique(arm_codes))
    dof = (n_arms - 1) * (x.shape[1] - 1)
    return JointBalanceResult(
        lr_stat=float(fit.llr), df=dof, p_value=float(fit.llr_pvalue), n_params=x.shape[1] - 1
    )
