import numpy as np
import pytest

from exptools.simulate_adjustment import Effect, Pool, buyer_probability, simulate_once, true_effects


def _pool(rng: np.random.Generator, n: int = 20_000) -> Pool:
    history = rng.lognormal(5.0, 0.8, n)
    buy = rng.random(n) < 0.01 * history / history.mean()
    visit = buy | (rng.random(n) < 0.1)
    spend = np.where(buy, np.round(rng.lognormal(4.5, 0.7, n), 2), 0.0)
    covariates = np.column_stack([history, rng.integers(1, 13, n)])
    outcomes = {"visit": visit.astype(float), "conversion": buy.astype(float), "spend": spend}
    return Pool(covariates, history, outcomes)


def test_true_effects_match_the_injection_on_average():
    # The unadjusted difference is unbiased by construction, so its average
    # error over many simulated experiments is zero exactly when the stated
    # truth is right.
    rng = np.random.default_rng(0)
    pool = _pool(rng)
    sizes = {"c": 6_000, "t1": 6_000, "t2": 6_000}
    effect = Effect("t1", spend_lift=1.0, visit_rate=0.08)
    seeds = np.random.SeedSequence(1).spawn(300)
    records = [simulate_once(pool, sizes, "c", effect, s) for s in seeds]
    for metric in ("visit", "conversion", "spend"):
        errors = np.array([r[f"{metric}|t1|raw|err"] for r in records])
        assert abs(errors.mean()) < 4 * errors.std() / np.sqrt(len(errors)), metric


def test_spend_truth_is_the_requested_lift():
    rng = np.random.default_rng(2)
    pool = _pool(rng)
    truth = true_effects(pool, Effect("t1", spend_lift=1.0, visit_rate=0.0))
    assert truth["spend"] == pytest.approx(pool.outcomes["spend"].mean())


def test_conversion_chance_grows_with_history_and_is_bounded():
    rng = np.random.default_rng(3)
    pool = _pool(rng)
    q = buyer_probability(pool, 1.0)
    non_buyer = pool.outcomes["spend"] == 0
    assert np.corrcoef(q[non_buyer], pool.history[non_buyer])[0, 1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="exceeds 1"):
        buyer_probability(pool, 1_000.0)
