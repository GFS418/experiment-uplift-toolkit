import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from exptools.adjust import (
    covariate_matrix,
    cuped,
    cuped_theta,
    difference_in_means,
    lin,
    ols_hc2,
    variance_reduction,
    within_arm_correlation,
)
from exptools.data import require_pre_treatment


def _experiment(n: int, rng: np.random.Generator, rho: float = 0.6, effects=(0.0, 0.3, 0.1)):
    """Three arms; outcome correlated with a pre-treatment covariate at about rho."""
    arm = rng.choice(np.array(["c", "t1", "t2"]), size=n)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * noise
    for name, effect in zip(("c", "t1", "t2"), effects, strict=True):
        y[arm == name] += effect
    return y, x, arm


def test_hc2_matches_statsmodels():
    rng = np.random.default_rng(0)
    design = sm.add_constant(rng.normal(size=(500, 3)))
    y = design @ np.array([1.0, 0.5, -0.2, 0.0]) + rng.normal(size=500) * (1 + np.abs(design[:, 1]))
    beta, cov = ols_hc2(design, y)
    ref = sm.OLS(y, design).fit(cov_type="HC2")
    assert np.allclose(beta, ref.params)
    assert np.allclose(cov, ref.cov_params())


def test_hc2_with_arm_dummies_alone_reproduces_welch():
    # The reason HC2 is the right default for experiments.
    rng = np.random.default_rng(1)
    y, _, arm = _experiment(3_000, rng)
    fit = lin(y, np.empty((len(y), 0)), arm, "c")
    welch = difference_in_means(y, arm, "t1", "c")
    assert fit.vs_control("t1").estimate == pytest.approx(welch.estimate)
    assert fit.vs_control("t1").se == pytest.approx(welch.se, rel=1e-10)


def test_cuped_theta_is_the_ancova_slope():
    rng = np.random.default_rng(2)
    y, x, arm = _experiment(2_000, rng)
    dummies = pd.get_dummies(arm, drop_first=True, dtype=float).to_numpy()
    ancova = sm.OLS(y, sm.add_constant(np.column_stack([dummies, x]))).fit()
    assert cuped_theta(y, x, arm) == pytest.approx(ancova.params[-1])


def test_cuped_subtracts_theta_times_the_covariate_imbalance():
    rng = np.random.default_rng(3)
    y, x, arm = _experiment(2_000, rng)
    theta = cuped_theta(y, x, arm)
    raw = difference_in_means(y, arm, "t1", "c").estimate
    imbalance = x[arm == "t1"].mean() - x[arm == "c"].mean()
    assert cuped(y, x, arm, "t1", "c").estimate == pytest.approx(raw - theta * imbalance)


def test_variance_reduction_is_about_rho_squared():
    # With one covariate, CUPED removes the share of variance it explains.
    rng = np.random.default_rng(4)
    y, x, arm = _experiment(200_000, rng, rho=0.6)
    vr = variance_reduction(cuped(y, x, arm, "t1", "c"), difference_in_means(y, arm, "t1", "c"))
    assert vr == pytest.approx(0.36, abs=0.01)
    assert within_arm_correlation(y, x, arm) ** 2 == pytest.approx(vr, abs=0.01)


def test_lin_matches_statsmodels_interacted_regression():
    rng = np.random.default_rng(5)
    y, x, arm = _experiment(1_500, rng)
    z = rng.integers(0, 2, size=y.size).astype(float)
    covs = np.column_stack([x, z])
    fit = lin(y, covs, arm, "c")
    centered = covs - covs.mean(axis=0)
    d = np.column_stack([(arm == a).astype(float) for a in ("t1", "t2")])
    design = np.column_stack([np.ones(y.size), d, centered, d[:, [0]] * centered, d[:, [1]] * centered])
    ref = sm.OLS(y, design).fit(cov_type="HC2")
    assert fit.coef == pytest.approx(ref.params[1:3])
    assert fit.cov == pytest.approx(ref.cov_params()[1:3, 1:3])
    contrast = fit.contrast("t1", "t2")
    ref_cov = ref.cov_params()
    assert contrast.se == pytest.approx(np.sqrt(ref_cov[1, 1] + ref_cov[2, 2] - 2 * ref_cov[1, 2]))


def test_lin_beats_cuped_when_the_effect_varies_with_the_covariate():
    # The treated arm's slope differs from control's; Lin fits both, CUPED one.
    rng = np.random.default_rng(6)
    y, x, arm = _experiment(100_000, rng, rho=0.3)
    y[arm == "t1"] += 0.8 * x[arm == "t1"]
    raw = difference_in_means(y, arm, "t1", "c")
    vr_cuped = variance_reduction(cuped(y, x, arm, "t1", "c"), raw)
    vr_lin = variance_reduction(lin(y, x[:, None], arm, "c").vs_control("t1"), raw)
    assert vr_lin > vr_cuped + 0.05


def test_covariate_matrix_drops_one_level_per_categorical():
    df = pd.DataFrame({"n": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    assert covariate_matrix(df, ["n", "cat"]).shape == (3, 3)


def test_pre_treatment_guard_refuses_outcomes():
    require_pre_treatment(["history", "recency", "channel"])
    with pytest.raises(ValueError, match="visit"):
        require_pre_treatment(["history", "visit"])
