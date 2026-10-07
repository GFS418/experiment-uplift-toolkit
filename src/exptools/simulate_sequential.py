"""Peeking simulations on data shaped like the real spend.

Customers arrive in random order (the accrual model in the Phase 3 design
note): each pseudo-arm is a sequence of real control customers drawn with
replacement, optionally with extra buyers injected as in Phase 1a, and Welch's
z is computed on the cumulative data at every look.
"""

from __future__ import annotations

import numpy as np


def inject_buyers(
    spend: np.ndarray, pool_spend: np.ndarray, lift: float, rng: np.random.Generator
) -> np.ndarray:
    """Turn non-buyers into buyers so mean spend rises by `lift` in expectation.

    Same mechanism as `simulate.inject_buyers`, on one value per customer:
    with buyer share p, a fraction lift * p / (1 - p) of non-buyers convert,
    and new buyers' spend is drawn from the pool's buyers.
    """
    buyers = pool_spend[pool_spend > 0]
    p = buyers.size / pool_spend.size
    out = spend.copy()
    convert = (out == 0) & (rng.random(out.size) < lift * p / (1 - p))
    out[convert] = rng.choice(buyers, size=int(convert.sum()))
    return out


def _prefix_mean_and_variance_of_mean(x: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.round(fractions * x.size).astype(int)
    s1 = np.concatenate([[0.0], np.cumsum(x)])[n]
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])[n]
    mean = s1 / n
    var = np.maximum(s2 - n * mean**2, 0.0) / (n - 1)
    return mean, var / n


def z_path(treat: np.ndarray, control: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Welch's z on the first share f of each arm's customers, for every look fraction f.

    Zero when neither arm has any variation yet (for spend: no buyers so far).
    """
    mt, vt = _prefix_mean_and_variance_of_mean(treat, fractions)
    mc, vc = _prefix_mean_and_variance_of_mean(control, fractions)
    se = np.sqrt(vt + vc)
    return np.divide(mt - mc, se, out=np.zeros_like(se), where=se > 0)


def simulate_z_paths(
    pool_spend: np.ndarray,
    n_t: int,
    n_c: int,
    fractions: np.ndarray,
    n_sims: int,
    seed: int,
    lift: float = 0.0,
) -> np.ndarray:
    """z at every look for `n_sims` simulated experiments, shape (n_sims, len(fractions))."""
    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims, len(fractions)))
    for i in range(n_sims):
        treat = rng.choice(pool_spend, size=n_t)
        control = rng.choice(pool_spend, size=n_c)
        if lift:
            treat = inject_buyers(treat, pool_spend, lift, rng)
        paths[i] = z_path(treat, control, fractions)
    return paths


def first_crossing(paths: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Index of the first look where |z| reaches its boundary, or -1 where none does."""
    crossed = np.abs(paths) >= bounds
    return np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1)
