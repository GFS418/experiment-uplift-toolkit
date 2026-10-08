"""Heterogeneous treatment effects: three learners, and how to judge them on held-out data.

With three arms and one control, every learner predicts each e-mail's effect
for each customer. Propensities are known by design (each arm's share of the
sample), which is what makes the doubly robust learner's scores exactly
unbiased, however rough the outcome models are.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from exptools.adjust import ols_hc2
from exptools.bootstrap import Arm

FEATURES = ("recency", "history", "mens", "womens", "newbie", "zip_code", "channel")
CONSTANT = "constant"
BOOSTING = {"max_iter": 200, "early_stopping": False, "random_state": 2026}
GRID = tuple(
    {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": leaf, "l2_regularization": l2}
    for lr in (0.03, 0.1)
    for leaves in (7, 31)
    for leaf in (50, 200)
    for l2 in (0.0, 1.0)
)

Params = dict | str  # a GRID entry, or CONSTANT


def label(params: Params) -> str:
    return CONSTANT if params == CONSTANT else json.dumps(params, sort_keys=True)


class ConstantModel:
    """Predicts the training mean: the same effect for everyone."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConstantModel:
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value_)


def make_regressor(params: Params):
    return ConstantModel() if params == CONSTANT else HistGradientBoostingRegressor(**BOOSTING, **params)


def fold_ids(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % k


def cv_select(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, candidates: Sequence[Params]
) -> tuple[Params, dict[str, float]]:
    """Candidate with the lowest cross-validated mean squared error, and every candidate's score."""
    scores = {}
    for params in candidates:
        sse = 0.0
        for k in np.unique(folds):
            train, held_out = folds != k, folds == k
            pred = make_regressor(params).fit(X[train], y[train]).predict(X[held_out])
            sse += float(np.sum((y[held_out] - pred) ** 2))
        scores[label(params)] = sse / len(y)
    best = min(candidates, key=lambda c: scores[label(c)])
    return best, scores


def arm_shares(arm: np.ndarray) -> dict[str, float]:
    """Exact assignment probabilities within this sample: each arm's share."""
    names, counts = np.unique(arm, return_counts=True)
    return {str(a): c / len(arm) for a, c in zip(names, counts, strict=True)}


def aipw_scores(
    y: np.ndarray, arm: np.ndarray, m: Mapping[str, np.ndarray], treatment: str, control: str
) -> np.ndarray:
    """Doubly robust (AIPW) score for treatment vs control, one per customer.

    E[score | x] equals the true effect at x whenever the propensities are right
    and the outcome predictions m do not use the customer's own outcome.
    """
    e = arm_shares(arm)
    return (
        m[treatment]
        - m[control]
        + (arm == treatment) * (y - m[treatment]) / e[treatment]
        - (arm == control) * (y - m[control]) / e[control]
    )


@dataclass
class TLearner:
    """One outcome model per arm; an effect is a difference of two predictions."""

    outcome_params: Mapping[str, Params]
    control: str
    models_: dict = field(default_factory=dict, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, arm: np.ndarray) -> TLearner:
        self.models_ = {
            a: make_regressor(p).fit(X[arm == a], y[arm == a]) for a, p in self.outcome_params.items()
        }
        return self

    def outcomes(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {a: model.predict(X) for a, model in self.models_.items()}

    def effect(self, X: np.ndarray, treatment: str) -> np.ndarray:
        return self.models_[treatment].predict(X) - self.models_[self.control].predict(X)


@dataclass
class DRLearner:
    """Kennedy's DR-learner: regress cross-fitted AIPW scores on the features."""

    outcome_params: Mapping[str, Params]
    stage2_params: Mapping[str, Params]  # per treatment arm
    control: str
    fold_seed: int
    n_folds: int = 5
    scores_: dict = field(default_factory=dict, init=False)
    models_: dict = field(default_factory=dict, init=False)

    def cross_fitted_scores(self, X: np.ndarray, y: np.ndarray, arm: np.ndarray) -> dict[str, np.ndarray]:
        folds = fold_ids(len(y), self.n_folds, self.fold_seed)
        m = {a: np.empty(len(y)) for a in self.outcome_params}
        for k in range(self.n_folds):
            held_out = folds == k
            for a, params in self.outcome_params.items():
                fit_rows = ~held_out & (arm == a)
                m[a][held_out] = make_regressor(params).fit(X[fit_rows], y[fit_rows]).predict(X[held_out])
        return {t: aipw_scores(y, arm, m, t, self.control) for t in self.stage2_params}

    def fit(self, X: np.ndarray, y: np.ndarray, arm: np.ndarray) -> DRLearner:
        self.scores_ = self.cross_fitted_scores(X, y, arm)
        self.models_ = {t: make_regressor(p).fit(X, self.scores_[t]) for t, p in self.stage2_params.items()}
        return self

    def effect(self, X: np.ndarray, treatment: str) -> np.ndarray:
        return self.models_[treatment].predict(X)


@dataclass
class CausalForest:
    """EconML's CausalForestDML with propensities fixed at the arm shares (not tuned)."""

    control: str
    treatments: tuple[str, ...]
    seed: int = 2026
    model_: object = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, arm: np.ndarray) -> CausalForest:
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier

        self.model_ = CausalForestDML(
            model_y=HistGradientBoostingRegressor(**BOOSTING),
            model_t=DummyClassifier(strategy="prior"),  # known propensities: the arm shares
            discrete_treatment=True,
            categories=[self.control, *self.treatments],
            n_estimators=1_000,
            min_samples_leaf=50,
            honest=True,
            cv=5,
            random_state=self.seed,
        )
        self.model_.fit(y, arm, X=X)
        return self

    def effect(self, X: np.ndarray, treatment: str) -> np.ndarray:
        return self.model_.effect(X, T0=self.control, T1=treatment)


# ---- evaluation on held-out data ---------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Best-linear-predictor test (Chernozhukov, Demirer, Duflo and Fernandez-Val)."""

    ate: float
    slope: float  # 1 if predictions are calibrated, 0 if they carry no real heterogeneity
    slope_se: float
    p_value: float  # one-sided, slope > 0
    applicable: bool  # False when the predictions do not vary


def calibration_test(scores: np.ndarray, predicted: np.ndarray) -> Calibration:
    """Regress AIPW scores on an intercept and the centered predicted effect (HC2)."""
    centered = predicted - predicted.mean()
    if np.ptp(centered) == 0:
        return Calibration(float(scores.mean()), np.nan, np.nan, np.nan, applicable=False)
    design = np.column_stack([np.ones(len(scores)), centered])
    beta, cov = ols_hc2(design, scores)
    se = float(np.sqrt(cov[1, 1]))
    return Calibration(
        float(beta[0]), float(beta[1]), se, float(stats.norm.sf(beta[1] / se)), applicable=True
    )


def uplift_curve(
    y: np.ndarray,
    arm: np.ndarray,
    treatment: str,
    control: str,
    predicted: np.ndarray,
    fractions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gain from treating only the top share f by predicted effect, for each f.

    G(f) = f * (mean outcome of treated - mean outcome of controls, within the
    top share), in outcome units per customer of the whole population. Ties in
    the prediction are broken in a random order from `rng`.
    """
    keep = (arm == treatment) | (arm == control)
    y, treated, score = y[keep], arm[keep] == treatment, predicted[keep]
    order = np.lexsort((rng.permutation(len(score)), -score))
    y, treated = y[order], treated[order]
    cum_y_t, cum_y_c = np.cumsum(y * treated), np.cumsum(y * ~treated)
    cum_n_t, cum_n_c = np.cumsum(treated), np.cumsum(~treated)
    k = np.maximum(np.round(fractions * len(y)).astype(int), 1) - 1
    return fractions * (cum_y_t[k] / cum_n_t[k] - cum_y_c[k] / cum_n_c[k])


def qini_area(gains: np.ndarray, fractions: np.ndarray) -> float:
    """Average gap between the uplift curve and random targeting (the line f * G(1))."""
    if not np.isclose(fractions[-1], 1.0):
        raise ValueError("the last fraction must be 1")
    return float(np.mean(gains - fractions * gains[-1]))


def greedy_policy(effects: Mapping[str, np.ndarray], control: str) -> np.ndarray:
    """Each customer gets the e-mail with the largest predicted effect, or none if no effect is positive."""
    names = [control, *effects]
    stacked = np.column_stack([np.zeros(len(next(iter(effects.values())))), *effects.values()])
    return np.array(names, dtype=object)[np.argmax(stacked, axis=1)]


def budget_policy(effects: Mapping[str, np.ndarray], control: str, share: float) -> np.ndarray:
    """The top `share` of customers by best predicted effect get their best e-mail; the rest none."""
    names = list(effects)
    stacked = np.column_stack(list(effects.values()))
    best, best_effect = np.array(names, dtype=object)[stacked.argmax(axis=1)], stacked.max(axis=1)
    policy = np.full(len(best), control, dtype=object)
    top = np.argsort(-best_effect, kind="stable")[: round(share * len(best))]
    policy[top] = best[top]
    return policy


def policy_value(y: np.ndarray, arm: np.ndarray, policy: np.ndarray) -> float:
    """Inverse-propensity value with exact arm shares.

    For each arm, the outcomes of its customers whom the policy assigns to that
    arm, divided by the arm's size, summed over arms.
    """
    return float(sum(y[(arm == a) & (policy == a)].sum() / np.sum(arm == a) for a in np.unique(arm)))


def policy_bootstrap(
    y: np.ndarray,
    arm: np.ndarray,
    policies: Mapping[str, np.ndarray],
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Bootstrap distribution of each policy's value, resampling customers within arm.

    All policies share every resample, so differences between them are paired.
    Each customer's contribution is zero unless the policy matches their arm and
    they spent, so the exact compressed bootstrap of Phase 1 applies.
    """
    names = list(policies)
    out = {name: np.zeros(n_resamples) for name in names}
    for a in np.unique(arm):
        rows = arm == a
        contributions = pd.DataFrame({name: y[rows] * (policies[name][rows] == a) for name in names})
        compressed = Arm.from_frame(contributions, names)
        draws = compressed.resample(n_resamples, rng)
        for name in names:
            out[name] += draws @ compressed.column(name) / compressed.n
    return out
