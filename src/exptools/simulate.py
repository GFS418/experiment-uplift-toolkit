"""Plasmode simulations: validate methods on data shaped like the real outcomes.

Each simulated experiment resamples customers from a pool (the real control
arm) into pseudo-arms of the real arm sizes, so the truth is known: without an
injected effect every pseudo-arm has the pool's mean and every true difference
is zero. An injected effect turns non-buyers into buyers at a known rate, so
the true difference is known as well.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exptools.bootstrap import Arm, bootstrap_replicates
from exptools.inference import compare
from exptools.multiplicity import benjamini_hochberg, dunnett_maxt, dunnett_parametric


@dataclass(frozen=True)
class Scenario:
    name: str
    lift: float  # relative lift in mean spend injected into `lifted_arm`; 0 = every null true
    lifted_arm: str | None = None


@dataclass(frozen=True)
class Design:
    pool: Arm
    sizes: Mapping[str, int]  # arm name -> number of customers, control included
    control: str
    alpha: float = 0.05
    n_resamples: int = 10_000

    @property
    def treatments(self) -> tuple[str, ...]:
        return tuple(a for a in self.sizes if a != self.control)


def plasmode_arm(pool: Arm, n: int, rng: np.random.Generator) -> Arm:
    """n customers drawn with replacement from the pool."""
    return Arm(pool.rows, rng.multinomial(n, pool.counts / pool.n), pool.metrics)


def inject_buyers(arm: Arm, pool: Arm, lift: float, rng: np.random.Generator) -> Arm:
    """Turn non-buyers into buyers so that mean spend rises by `lift` in expectation.

    With buyer share p and mean buyer spend mu+, converting a fraction
    q = lift * p / (1 - p) of non-buyers adds q * (1 - p) * mu+ = lift * mean
    to mean spend. New buyers take outcome rows (visit, conversion, spend)
    drawn from the pool's buyers, so the three metrics stay consistent.
    """
    buyer = pool.column("spend") > 0
    p = pool.counts[buyer].sum() / pool.n
    q = lift * p / (1 - p)
    counts = arm.counts.copy()
    converted = rng.binomial(counts[~buyer], q)
    counts[~buyer] -= converted
    counts[buyer] += rng.multinomial(converted.sum(), pool.counts[buyer] / pool.counts[buyer].sum())
    return Arm(arm.rows, counts, arm.metrics)


def _record_interval(record: dict, key: str, truth: float, low: float, high: float) -> None:
    record[f"{key}_miss_low"] = truth < low  # interval sits entirely above the truth
    record[f"{key}_miss_high"] = truth > high  # interval sits entirely below the truth


def simulate_once(design: Design, scenario: Scenario, seed: np.random.SeedSequence) -> dict:
    """One simulated experiment, analyzed exactly as the real one will be."""
    rng = np.random.default_rng(seed)
    pool, control = design.pool, design.control
    arms = {name: plasmode_arm(pool, n, rng) for name, n in design.sizes.items()}
    truth = dict.fromkeys(design.treatments, 0.0)
    if scenario.lift:
        arms[scenario.lifted_arm] = inject_buyers(arms[scenario.lifted_arm], pool, scenario.lift, rng)
        truth[scenario.lifted_arm] = scenario.lift * pool.mean("spend")

    reps = bootstrap_replicates(arms, pool.metrics, design.n_resamples, rng)
    record: dict = {}
    treatments = {a: arms[a] for a in design.treatments}

    # Primary family: spend, each treatment vs control.
    for a in design.treatments:
        c = compare(arms[a], arms[control], "spend", reps[a]["spend"], reps[control]["spend"])
        _record_interval(record, f"spend|{a}|bca", truth[a], *c.bca)
        _record_interval(record, f"spend|{a}|welch", truth[a], *c.welch.ci)
    for label, result in (
        (
            "maxt",
            dunnett_maxt(
                treatments,
                arms[control],
                "spend",
                {a: reps[a]["spend"] for a in treatments},
                reps[control]["spend"],
            ),
        ),
        ("parametric", dunnett_parametric(treatments, arms[control], "spend", rng)),
    ):
        record[f"dunnett|{label}|critical"] = result.critical_value
        for i, a in enumerate(result.arms):
            record[f"dunnett|{label}|{a}|reject"] = bool(result.p_adjusted[i] < design.alpha)
            _record_interval(record, f"dunnett|{label}|{a}", truth[a], result.ci_low[i], result.ci_high[i])

    # Exploratory family: only meaningful when every null is true.
    if not scenario.lift:
        first, second = design.treatments
        hypotheses = [
            *((m, a, control) for m in ("visit", "conversion") for a in design.treatments),
            *((m, first, second) for m in ("visit", "conversion", "spend")),
        ]
        p_values = []
        for metric, t, ctl in hypotheses:
            c = compare(arms[t], arms[ctl], metric, reps[t][metric], reps[ctl][metric])
            p_values.append(c.welch.p_value)
            record[f"{metric}|{t}|{ctl}|welch_reject"] = c.welch.p_value < design.alpha
            _record_interval(record, f"{metric}|{t}|{ctl}|bca", 0.0, *c.bca)
        record["bh|any_reject"] = bool(benjamini_hochberg(p_values, design.alpha)[1].any())
    return record


def run_scenario(
    design: Design, scenario: Scenario, n_sims: int, seed: int, workers: int = 8
) -> pd.DataFrame:
    seeds = np.random.SeedSequence(seed).spawn(n_sims)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(simulate_once, [design] * n_sims, [scenario] * n_sims, seeds, chunksize=25))
    return pd.DataFrame.from_records(records).assign(scenario=scenario.name)
