"""Plasmode simulations for covariate adjustment.

As in `simulate`, real control customers are resampled into pseudo-arms, but
each customer keeps their pre-treatment covariates, so CUPED and Lin's
regression adjustment can be checked against a known truth. The known-effect
scenario makes the effect grow with prior-year spend, the case where Lin's
separate slopes per arm should beat CUPED's single slope, and also raises
visits, so a post-treatment covariate visibly absorbs part of the effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exptools.adjust import Estimate, cuped, difference_in_means, lin

METRICS = ("visit", "conversion", "spend")


@dataclass(frozen=True)
class Pool:
    covariates: np.ndarray  # (N, p) regression-adjustment covariates
    history: np.ndarray  # (N,) the CUPED covariate
    outcomes: Mapping[str, np.ndarray]  # metric -> (N,)

    @property
    def size(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class Effect:
    """A known effect for one pseudo-arm: extra buyers (more likely with higher
    prior-year spend) and extra visitors who do not buy."""

    arm: str
    spend_lift: float  # relative lift in mean spend
    visit_rate: float  # chance a remaining non-visitor starts visiting


def buyer_probability(pool: Pool, spend_lift: float) -> np.ndarray:
    """Chance each pool customer is converted into a buyer, proportional to history.

    Non-buyers convert at an average rate q0 = lift * p / (1 - p), where p is
    the buyer share, which raises mean spend by `spend_lift`; existing buyers
    are left alone.
    """
    buyer = pool.outcomes["spend"] > 0
    p = buyer.mean()
    q0 = spend_lift * p / (1 - p)
    q = np.where(buyer, 0.0, q0 * pool.history / pool.history[~buyer].mean())
    if (q > 1).any():
        raise ValueError("lift too large: a conversion probability exceeds 1")
    return q


def true_effects(pool: Pool, effect: Effect) -> dict[str, float]:
    """Exact effect of `effect` on each metric's mean in the pool's population."""
    q = buyer_probability(pool, effect.spend_lift)
    spend = pool.outcomes["spend"]
    non_visitor = 1 - pool.outcomes["visit"]
    return {
        "spend": float(q.mean() * spend[spend > 0].mean()),
        "conversion": float(q.mean()),
        "visit": float((non_visitor * (q + (1 - q) * effect.visit_rate)).mean()),
    }


def simulate_once(
    pool: Pool,
    sizes: Mapping[str, int],
    control: str,
    effect: Effect | None,
    seed: np.random.SeedSequence,
) -> dict:
    rng = np.random.default_rng(seed)
    arm = np.concatenate([np.full(n, a) for a, n in sizes.items()])
    rows = rng.integers(0, pool.size, size=len(arm))
    outcomes = {m: pool.outcomes[m][rows].astype(float) for m in METRICS}
    history, covariates = pool.history[rows], pool.covariates[rows]
    treatments = [a for a in sizes if a != control]
    truth = {(m, a): 0.0 for m in METRICS for a in treatments}

    if effect is not None:
        lifted = arm == effect.arm
        q = buyer_probability(pool, effect.spend_lift)[rows]
        new_buyer = lifted & (rng.random(len(arm)) < q)
        buyer_spend = pool.outcomes["spend"][pool.outcomes["spend"] > 0]
        outcomes["spend"][new_buyer] = rng.choice(buyer_spend, size=int(new_buyer.sum()))
        outcomes["conversion"][new_buyer] = 1
        outcomes["visit"][new_buyer] = 1
        new_visitor = lifted & (outcomes["visit"] == 0) & (rng.random(len(arm)) < effect.visit_rate)
        outcomes["visit"][new_visitor] = 1
        for m, value in true_effects(pool, effect).items():
            truth[(m, effect.arm)] = value

    record: dict = {}

    def keep(m: str, a: str, method: str, est: Estimate) -> None:
        err = est.estimate - truth[(m, a)]
        record[f"{m}|{a}|{method}|err"] = err
        record[f"{m}|{a}|{method}|se"] = est.se
        record[f"{m}|{a}|{method}|covered"] = abs(err) <= 1.959964 * est.se

    for m in METRICS:
        y = outcomes[m]
        fit = lin(y, covariates, arm, control)
        for a in treatments:
            keep(m, a, "raw", difference_in_means(y, arm, a, control))
            keep(m, a, "cuped", cuped(y, history, arm, a, control))
            keep(m, a, "lin", fit.vs_control(a))
    # Counterexample: CUPED on visit, which the treatment changes. Shown, never used.
    for a in treatments:
        keep("spend", a, "cuped_on_visit", cuped(outcomes["spend"], outcomes["visit"], arm, a, control))
    return record


def run(
    pool: Pool,
    sizes: Mapping[str, int],
    control: str,
    effect: Effect | None,
    n_sims: int,
    seed: int,
    workers: int = 8,
) -> pd.DataFrame:
    seeds = np.random.SeedSequence(seed).spawn(n_sims)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        args = ([pool] * n_sims, [sizes] * n_sims, [control] * n_sims, [effect] * n_sims, seeds)
        records = list(ex.map(simulate_once, *args, chunksize=25))
    return pd.DataFrame.from_records(records)
