import numpy as np
import pytest

from exptools.data import halves_by_arm
from exptools.simulate_uplift import (
    Config,
    Pool,
    Scenario,
    conversion_probability,
    expected_spend,
    simulate_once,
)
from exptools.uplift import CONSTANT


def _pool(rng: np.random.Generator, n: int = 10_000) -> Pool:
    history = rng.lognormal(5.0, 0.8, n)
    spend = np.where(rng.random(n) < 0.006, np.round(rng.lognormal(4.5, 0.7, n), 2), 0.0)
    mens, womens = rng.integers(0, 2, n), rng.integers(0, 2, n)
    X = np.column_stack([history, mens, womens])
    return Pool(X, spend, history, {"m": mens, "w": womens})


def test_average_lift_is_as_requested_in_both_scenarios():
    rng = np.random.default_rng(0)
    pool = _pool(rng)
    for heterogeneous in (False, True):
        q = conversion_probability(pool, "m", 1.18, heterogeneous)
        lifted = expected_spend(pool, np.full(pool.size, "m", dtype=object), {"m": q})
        assert lifted.mean() == pytest.approx(2.18 * pool.spend.mean())


def test_heterogeneous_conversion_favours_past_buyers_of_the_category():
    rng = np.random.default_rng(1)
    pool = _pool(rng)
    q = conversion_probability(pool, "m", 1.18, True)
    non_buyer = pool.spend == 0
    ratio = (q / pool.history)[non_buyer & (pool.category["m"] == 1)].mean() / (q / pool.history)[
        non_buyer & (pool.category["m"] == 0)
    ].mean()
    assert ratio == pytest.approx(4.0)


def test_stratified_halves_split_every_arm_in_two():
    arm = np.array(["c"] * 11 + ["m"] * 10, dtype=object)
    half = halves_by_arm(arm, np.random.default_rng(0), ("c", "m"))
    assert np.sum((arm == "c") & (half == "train")) == 5
    assert np.sum((arm == "m") & (half == "train")) == 5


def test_simulated_policy_value_is_unbiased_against_its_truth():
    # Constant models make every simulated policy a single-arm policy, so this
    # isolates the data generation, the split, and the value estimator.
    rng = np.random.default_rng(2)
    pool = _pool(rng)
    config = Config(dict.fromkeys(("c", "m", "w"), CONSTANT), {"m": CONSTANT, "w": CONSTANT})
    scenario = Scenario("constant", {"m": 1.18, "w": 0.65}, heterogeneous=False)
    sizes = {"c": 6_000, "m": 6_000, "w": 6_000}
    records = [
        simulate_once(pool, sizes, "c", scenario, config, s, n_resamples=200)
        for s in np.random.SeedSequence(3).spawn(200)
    ]
    gaps = np.array([r["value"] - r["truth"] for r in records])
    assert abs(gaps.mean()) < 4 * gaps.std() / np.sqrt(len(gaps))
    assert not records[0]["m|applicable"]  # a constant second stage leaves nothing to calibrate
    covered = np.mean([r["low"] <= r["truth"] <= r["high"] for r in records])
    assert covered > 0.88


def test_retuned_simulation_records_both_learners_tests():
    from exptools.simulate_uplift import simulate_retuned_once

    rng = np.random.default_rng(4)
    pool = _pool(rng)
    config = Config(dict.fromkeys(("c", "m", "w"), CONSTANT), {"m": CONSTANT, "w": CONSTANT})
    scenario = Scenario("heterogeneous", {"m": 1.18, "w": 0.65}, heterogeneous=True)
    record = simulate_retuned_once(
        pool,
        {"c": 3_000, "m": 3_000, "w": 3_000},
        "c",
        scenario,
        config,
        np.random.SeedSequence(5),
        candidates=(CONSTANT,),
    )
    # Only a constant was offered, so the DR test cannot apply; the T-learner's always can.
    assert record["m|dr_constant"] and not record["m|dr_applicable"]
    assert set(record) >= {"m|t_p", "m|t_slope", "w|t_p", "w|dr_p"}
