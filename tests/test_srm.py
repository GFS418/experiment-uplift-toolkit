import numpy as np
import pytest
from scipy import stats

from exptools.srm import SRM_ALPHA, srm_test


def test_exact_allocation_has_no_mismatch():
    result = srm_test({"a": 100, "b": 100, "c": 100}, {"a": 1, "b": 1, "c": 1})
    assert result.chi2 == 0
    assert result.p_value == pytest.approx(1.0)
    assert not result.mismatch


def test_known_value():
    # (60 - 50)^2 / 50 + (40 - 50)^2 / 50 = 4 on 1 df
    result = srm_test({"a": 60, "b": 40}, {"a": 0.5, "b": 0.5})
    assert result.chi2 == pytest.approx(4.0)
    assert result.df == 1
    assert result.p_value == pytest.approx(stats.chi2.sf(4.0, 1))


def test_allocation_is_normalized():
    counts = {"a": 70, "b": 30}
    assert srm_test(counts, {"a": 7, "b": 3}).p_value == pytest.approx(
        srm_test(counts, {"a": 0.7, "b": 0.3}).p_value
    )


def test_rejects_mismatched_arm_names():
    with pytest.raises(ValueError, match="arms differ"):
        srm_test({"a": 1, "b": 1}, {"a": 1, "c": 1})


@pytest.mark.parametrize("shares", [(1 / 3, 1 / 3, 1 / 3), (0.8, 0.1, 0.1)])
def test_false_alarm_rate_matches_alpha_by_simulation(shares):
    # Under a correct randomization the test should fire at its nominal rate.
    # alpha = 0.05 here (not SRM_ALPHA) so 4,000 draws pin the rate tightly.
    rng = np.random.default_rng(2026)
    arms, n_sims, alpha = ("a", "b", "c"), 4_000, 0.05
    allocation = dict(zip(arms, shares, strict=True))
    draws = rng.multinomial(64_000, shares, size=n_sims)
    rate = np.mean([srm_test(dict(zip(arms, d, strict=True)), allocation).p_value < alpha for d in draws])
    mc_se = np.sqrt(alpha * (1 - alpha) / n_sims)
    assert abs(rate - alpha) < 4 * mc_se


def test_detects_a_real_mismatch_by_simulation():
    # A 35/32.5/32.5 split where 1/3 each was designed is a broken randomization.
    rng = np.random.default_rng(7)
    draws = rng.multinomial(64_000, (0.35, 0.325, 0.325), size=500)
    allocation = {"a": 1, "b": 1, "c": 1}
    detected = [srm_test(dict(zip("abc", d, strict=True)), allocation).mismatch for d in draws]
    assert np.mean(detected) > 0.99
    assert SRM_ALPHA == 0.001
