import numpy as np
import pandas as pd
import pytest

from exptools.uplift import (
    CONSTANT,
    GRID,
    CausalForest,
    DRLearner,
    TLearner,
    aipw_scores,
    arm_shares,
    budget_policy,
    calibration_test,
    cv_select,
    fold_ids,
    greedy_policy,
    policy_bootstrap,
    policy_value,
    qini_area,
    uplift_curve,
)

ARMS = ("c", "t1", "t2")
SMALL = {"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 50, "l2_regularization": 0.0}


def _experiment(n: int, rng: np.random.Generator, heterogeneous: bool = True):
    """Normal outcomes; t1's effect is 1 where x0 > 0 and 0 elsewhere (or 0.5 everywhere)."""
    X = rng.normal(size=(n, 3))
    arm = rng.choice(np.array(ARMS, dtype=object), size=n)
    tau1 = np.where(X[:, 0] > 0, 1.0, 0.0) if heterogeneous else np.full(n, 0.5)
    tau2 = np.full(n, 0.2)
    y = X[:, 1] + tau1 * (arm == "t1") + tau2 * (arm == "t2") + rng.normal(size=n)
    return X, y, arm, {"t1": tau1, "t2": tau2}


def test_grid_has_sixteen_settings():
    assert len(GRID) == 16
    assert len({tuple(sorted(g.items())) for g in GRID}) == 16


def test_arm_shares_are_exact_proportions():
    arm = np.array(["c"] * 3 + ["t1"] * 5 + ["t2"] * 2, dtype=object)
    assert arm_shares(arm) == {"c": 0.3, "t1": 0.5, "t2": 0.2}


def test_aipw_scores_are_unbiased_even_with_wrong_outcome_models():
    # Deliberately bad outcome predictions; known propensities keep the scores unbiased.
    rng = np.random.default_rng(0)
    X, y, arm, tau = _experiment(200_000, rng)
    wrong = {a: np.full(len(y), 5.0) for a in ARMS}
    scores = aipw_scores(y, arm, wrong, "t1", "c")
    right_half = X[:, 0] > 0
    for rows, truth in ((slice(None), tau["t1"].mean()), (right_half, 1.0), (~right_half, 0.0)):
        s = scores[rows]
        assert abs(s.mean() - truth) < 4 * s.std() / np.sqrt(len(s))


def test_cv_prefers_a_constant_when_there_is_nothing_to_learn():
    rng = np.random.default_rng(1)
    X, noise = rng.normal(size=(4_000, 3)), rng.normal(size=4_000)
    best, scores = cv_select(X, noise, fold_ids(4_000, 5, 0), [CONSTANT, SMALL])
    assert best == CONSTANT
    assert set(scores) == {
        CONSTANT,
        '{"l2_regularization": 0.0, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 50}',
    }


def test_dr_and_t_learners_recover_planted_heterogeneity():
    rng = np.random.default_rng(2)
    X, y, arm, tau = _experiment(30_000, rng)
    params = dict.fromkeys(ARMS, SMALL)
    dr = DRLearner(params, {"t1": SMALL, "t2": SMALL}, "c", fold_seed=3).fit(X, y, arm)
    t = TLearner(params, "c").fit(X, y, arm)
    for learner in (dr, t):
        assert np.corrcoef(learner.effect(X, "t1"), tau["t1"])[0, 1] > 0.8


def test_causal_forest_recovers_planted_heterogeneity():
    rng = np.random.default_rng(4)
    X, y, arm, tau = _experiment(20_000, rng)
    cf = CausalForest("c", ("t1", "t2")).fit(X, y, arm)
    assert np.corrcoef(cf.effect(X, "t1"), tau["t1"])[0, 1] > 0.8


def test_calibration_slope_is_one_for_true_effects_and_zero_for_noise():
    rng = np.random.default_rng(5)
    X, y, arm, tau = _experiment(100_000, rng)
    oracle = {"c": X[:, 1], "t1": X[:, 1] + tau["t1"], "t2": X[:, 1] + tau["t2"]}
    scores = aipw_scores(y, arm, oracle, "t1", "c")
    good = calibration_test(scores, tau["t1"])
    assert good.slope == pytest.approx(1.0, abs=0.06)
    assert good.p_value < 1e-6
    noise = calibration_test(scores, rng.normal(size=len(y)))
    assert abs(noise.slope) < 0.05
    assert not calibration_test(scores, np.full(len(y), 0.3)).applicable


def test_uplift_curve_ranks_true_effects_above_random():
    rng = np.random.default_rng(6)
    X, y, arm, tau = _experiment(60_000, rng)
    f = np.arange(1, 101) / 100
    perfect = uplift_curve(y, arm, "t1", "c", tau["t1"], f, np.random.default_rng(0))
    random = uplift_curve(y, arm, "t1", "c", rng.normal(size=len(y)), f, np.random.default_rng(0))
    # At f = 1 the curve is the plain difference in means, whatever the ranking.
    diff = y[arm == "t1"].mean() - y[arm == "c"].mean()
    assert perfect[-1] == pytest.approx(diff) and random[-1] == pytest.approx(diff)
    # Perfect ranking of a 0/1 effect on half the customers: the curve rises at slope 1 to
    # 0.5 then stays flat, so the area above the random line is exactly 0.125.
    assert qini_area(perfect, f) == pytest.approx(0.125, abs=0.01)
    assert abs(qini_area(random, f)) < 0.03


def test_policies():
    effects = {"t1": np.array([0.5, -0.1, 0.2, 0.0]), "t2": np.array([0.1, -0.2, 0.3, 0.0])}
    assert greedy_policy(effects, "c").tolist() == ["t1", "c", "t2", "c"]
    assert budget_policy(effects, "c", 0.5).tolist() == ["t1", "c", "t2", "c"]
    assert budget_policy(effects, "c", 0.25).tolist() == ["t1", "c", "c", "c"]


def test_policy_value_of_a_single_arm_policy_is_that_arms_mean():
    rng = np.random.default_rng(7)
    _, y, arm, _ = _experiment(9_000, rng)
    everyone_t1 = np.full(len(y), "t1", dtype=object)
    assert policy_value(y, arm, everyone_t1) == pytest.approx(y[arm == "t1"].mean())


def test_policy_value_is_unbiased_for_a_targeted_policy():
    # Truth: send t1 where x0 > 0 (effect 1), otherwise t2 (effect 0.2).
    rng = np.random.default_rng(8)
    estimates, truths = [], []
    for _ in range(300):
        X, y, arm, tau = _experiment(6_000, rng)
        policy = np.where(X[:, 0] > 0, "t1", "t2").astype(object)
        estimates.append(policy_value(y, arm, policy))
        truths.append(np.mean(X[:, 1] + np.where(X[:, 0] > 0, tau["t1"], tau["t2"])))
    gap = np.array(estimates) - np.array(truths)
    assert abs(gap.mean()) < 4 * gap.std() / np.sqrt(len(gap))


def test_policy_bootstrap_matches_the_analytic_spread_and_is_paired():
    rng = np.random.default_rng(9)
    spend = np.where(rng.random(30_000) < 0.01, rng.lognormal(4.5, 0.7, 30_000), 0.0)
    arm = rng.choice(np.array(ARMS, dtype=object), size=30_000)
    everyone = {a: np.full(len(arm), a, dtype=object) for a in ARMS}
    boot = policy_bootstrap(spend, arm, everyone, 4_000, np.random.default_rng(1))
    analytic_sd = spend[arm == "t1"].std() / np.sqrt(np.sum(arm == "t1"))
    assert boot["t1"].std() == pytest.approx(analytic_sd, rel=0.05)
    # Different arms' customers are resampled independently, so these are uncorrelated.
    assert abs(np.corrcoef(boot["t1"], boot["t2"])[0, 1]) < 0.06
    pd.testing.assert_index_equal(pd.Index(sorted(boot)), pd.Index(sorted(ARMS)))
