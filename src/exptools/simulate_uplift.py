"""Plasmode validation for Phase 4: does the pipeline find planted heterogeneity,
and is its policy value honest?

Customers are resampled from the training half's control arm (features and
spend). Extra buyers are injected into the e-mail arms, either uniformly
(constant effects) or concentrated among past buyers of the e-mail's category
and in proportion to prior-year spend (heterogeneous effects). Each simulated
experiment is split, fit and evaluated as the real one will be, with the truth
known exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exptools.data import halves_by_arm
from exptools.uplift import (
    CONSTANT,
    GRID,
    DRLearner,
    Params,
    TLearner,
    aipw_scores,
    calibration_test,
    cv_select,
    fold_ids,
    greedy_policy,
    make_regressor,
    policy_bootstrap,
    policy_value,
)


@dataclass(frozen=True)
class Pool:
    X: np.ndarray  # model features, one row per customer
    spend: np.ndarray
    history: np.ndarray
    category: Mapping[str, np.ndarray]  # treatment arm -> 1 if the customer bought that e-mail's category

    @property
    def size(self) -> int:
        return len(self.spend)


@dataclass(frozen=True)
class Scenario:
    name: str
    lifts: Mapping[str, float]  # treatment arm -> relative lift in mean spend
    heterogeneous: bool


@dataclass(frozen=True)
class Config:
    """The frozen spend configuration: outcome models per arm, second stage per treatment."""

    outcome_params: Mapping[str, Params]
    stage2_params: Mapping[str, Params]


def conversion_probability(pool: Pool, treatment: str, lift: float, heterogeneous: bool) -> np.ndarray:
    """Chance each pool customer is turned into a buyer by the e-mail.

    Non-buyers convert at an average rate lift * p / (1 - p), p being the buyer
    share, which raises mean spend by `lift`. Under heterogeneity a customer's
    chance is proportional to (1 + 3 * bought the category) * history: past
    buyers of a category are four times as responsive to its e-mail.
    """
    buyer = pool.spend > 0
    p = buyer.mean()
    weight = (1 + 3 * pool.category[treatment]) * pool.history if heterogeneous else np.ones(pool.size)
    q = np.where(buyer, 0.0, lift * p / (1 - p) * weight / weight[~buyer].mean())
    if (q > 1).any():
        raise ValueError("lift too large: a conversion probability exceeds 1")
    return q


def expected_spend(pool: Pool, assignment: np.ndarray, q: Mapping[str, np.ndarray]) -> np.ndarray:
    """Each pool customer's expected spend under the arm they are assigned."""
    mean_buyer_spend = pool.spend[pool.spend > 0].mean()
    out = pool.spend.astype(float)
    for t, qt in q.items():
        chosen = assignment == t
        out[chosen] += qt[chosen] * mean_buyer_spend  # q is zero for existing buyers
    return out


@dataclass(frozen=True)
class Experiment:
    """One simulated experiment, already split into training and test halves."""

    X: np.ndarray
    y: np.ndarray
    arm: np.ndarray
    train: np.ndarray  # boolean mask
    q: Mapping[str, np.ndarray]  # each treatment's conversion probability for every pool customer
    treatments: tuple[str, ...]


def draw_experiment(
    pool: Pool, sizes: Mapping[str, int], control: str, scenario: Scenario, rng: np.random.Generator
) -> Experiment:
    arms = tuple(sizes)
    treatments = tuple(a for a in arms if a != control)
    arm = np.concatenate([np.full(n, a, dtype=object) for a, n in sizes.items()])
    rows = rng.integers(0, pool.size, size=len(arm))
    X, y = pool.X[rows], pool.spend[rows].astype(float)
    q = {t: conversion_probability(pool, t, scenario.lifts[t], scenario.heterogeneous) for t in treatments}
    buyer_spend = pool.spend[pool.spend > 0]
    for t in treatments:
        convert = (arm == t) & (rng.random(len(arm)) < q[t][rows])
        y[convert] = rng.choice(buyer_spend, size=int(convert.sum()))
    train = halves_by_arm(arm, rng, arms) == "train"
    return Experiment(X, y, arm, train, q, treatments)


def simulate_once(
    pool: Pool,
    sizes: Mapping[str, int],
    control: str,
    scenario: Scenario,
    config: Config,
    seed: np.random.SeedSequence,
    n_resamples: int = 1_000,
) -> dict:
    rng = np.random.default_rng(seed)
    arms = tuple(sizes)
    e = draw_experiment(pool, sizes, control, scenario, rng)
    X, y, arm, train, q, treatments = e.X, e.y, e.arm, e.train, e.q, e.treatments
    test = ~train
    dr = DRLearner(config.outcome_params, config.stage2_params, control, int(rng.integers(2**31)))
    dr.fit(X[train], y[train], arm[train])
    outcome_models = TLearner(config.outcome_params, control).fit(X[train], y[train], arm[train])
    m_test = outcome_models.outcomes(X[test])

    record: dict = {}
    effects = {}
    for t in treatments:
        effects[t] = dr.effect(X[test], t)
        cal = calibration_test(aipw_scores(y[test], arm[test], m_test, t, control), effects[t])
        record[f"{t}|applicable"] = cal.applicable
        record[f"{t}|slope"] = cal.slope
        record[f"{t}|p"] = cal.p_value

    policy = greedy_policy(effects, control)
    boot = policy_bootstrap(y[test], arm[test], {"dr": policy}, n_resamples, rng)["dr"]
    record["value"] = policy_value(y[test], arm[test], policy)
    record["low"], record["high"] = np.percentile(boot, [2.5, 97.5])

    # Truth: the fitted policy applied to every customer of the population.
    pool_policy = greedy_policy({t: dr.effect(pool.X, t) for t in treatments}, control)
    record["truth"] = float(expected_spend(pool, pool_policy, q).mean())
    record["best_single_truth"] = max(
        float(expected_spend(pool, np.full(pool.size, t, dtype=object), q).mean()) for t in treatments
    )
    for a in arms:
        record[f"share|{a}"] = float(np.mean(policy == a))
    return record


def _one_thread_per_worker() -> None:
    # Eight workers each starting a full set of boosting threads would oversubscribe the CPU.
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


def simulate_retuned_once(
    pool: Pool,
    sizes: Mapping[str, int],
    control: str,
    scenario: Scenario,
    config: Config,
    seed: np.random.SeedSequence,
    candidates: Sequence[Params] = (CONSTANT, *GRID),
) -> dict:
    """The full procedure inside each experiment: the DR second stage is re-selected by
    cross-validation (constant vs the tree settings) instead of frozen, and the T-learner's
    predictions get the same calibration test."""
    rng = np.random.default_rng(seed)
    e = draw_experiment(pool, sizes, control, scenario, rng)
    X, y, arm, train, treatments = e.X, e.y, e.arm, e.train, e.treatments
    test = ~train
    fold_seed = int(rng.integers(2**31))
    dr = DRLearner(config.outcome_params, dict.fromkeys(treatments, CONSTANT), control, fold_seed)
    scores = dr.cross_fitted_scores(X[train], y[train], arm[train])
    folds = fold_ids(int(train.sum()), dr.n_folds, fold_seed)  # the same folds as the cross-fitting
    t_learner = TLearner(config.outcome_params, control).fit(X[train], y[train], arm[train])
    m_test = t_learner.outcomes(X[test])

    record: dict = {}
    for t in treatments:
        chosen, _ = cv_select(X[train], scores[t], folds, candidates)
        dr_effect = make_regressor(chosen).fit(X[train], scores[t]).predict(X[test])
        test_scores = aipw_scores(y[test], arm[test], m_test, t, control)
        for name, predicted in (("dr", dr_effect), ("t", t_learner.effect(X[test], t))):
            cal = calibration_test(test_scores, predicted)
            record[f"{t}|{name}_applicable"] = cal.applicable
            record[f"{t}|{name}_slope"] = cal.slope
            record[f"{t}|{name}_p"] = cal.p_value
        record[f"{t}|dr_constant"] = chosen == CONSTANT
    return record


def run(
    pool: Pool,
    sizes: Mapping[str, int],
    control: str,
    scenario: Scenario,
    config: Config,
    n_sims: int,
    seed: int,
    workers: int = 8,
    simulate: Callable[..., dict] = simulate_once,
) -> pd.DataFrame:
    seeds = np.random.SeedSequence(seed).spawn(n_sims)
    with ProcessPoolExecutor(max_workers=workers, initializer=_one_thread_per_worker) as ex:
        args = (
            [pool] * n_sims,
            [sizes] * n_sims,
            [control] * n_sims,
            [scenario] * n_sims,
            [config] * n_sims,
            seeds,
        )
        records = list(ex.map(simulate, *args, chunksize=5))
    return pd.DataFrame.from_records(records).assign(scenario=scenario.name)
