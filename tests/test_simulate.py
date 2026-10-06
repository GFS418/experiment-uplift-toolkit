import numpy as np
import pandas as pd
import pytest

from exptools.bootstrap import Arm
from exptools.simulate import inject_buyers, plasmode_arm


def _pool(rng, n=20_000):
    buy = rng.random(n) < 0.01
    spend = np.where(buy, np.round(rng.lognormal(4.5, 0.7, n), 2), 0.0)
    visit = np.where(buy, 1, (rng.random(n) < 0.1).astype(int))
    df = pd.DataFrame({"visit": visit, "conversion": buy.astype(int), "spend": spend})
    return Arm.from_frame(df, ["visit", "conversion", "spend"])


def test_plasmode_arm_draws_the_requested_size_from_the_pool():
    rng = np.random.default_rng(0)
    pool = _pool(rng)
    arm = plasmode_arm(pool, 21_387, rng)
    assert arm.n == 21_387
    assert np.array_equal(arm.rows, pool.rows)


def test_injection_lifts_mean_spend_by_the_requested_share():
    rng = np.random.default_rng(1)
    pool = _pool(rng)
    lifted = [
        inject_buyers(plasmode_arm(pool, 20_000, rng), pool, 0.5, rng).mean("spend") for _ in range(400)
    ]
    assert np.mean(lifted) == pytest.approx(1.5 * pool.mean("spend"), rel=0.02)


def test_injected_buyers_keep_metrics_consistent():
    rng = np.random.default_rng(2)
    pool = _pool(rng)
    arm = inject_buyers(plasmode_arm(pool, 20_000, rng), pool, 1.0, rng)
    visit, conversion, spend = (arm.column(m) for m in ("visit", "conversion", "spend"))
    occupied = arm.counts > 0
    assert np.all((spend[occupied] > 0) == (conversion[occupied] == 1))
    assert np.all(visit[occupied] >= conversion[occupied])
