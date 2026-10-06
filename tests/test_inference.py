import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exptools.bootstrap import Arm, bootstrap_replicates
from exptools.inference import compare, welch


def _arm(values) -> Arm:
    return Arm.from_frame(pd.DataFrame({"y": values}), ["y"])


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    t = np.where(rng.random(500) < 0.1, rng.gamma(2.0, 40.0, 500), 0.0)
    c = np.where(rng.random(450) < 0.07, rng.gamma(2.0, 40.0, 450), 0.0)
    ours = welch(_arm(t), _arm(c), "y")
    ref = stats.ttest_ind(t, c, equal_var=False)
    ref_ci = ref.confidence_interval(0.95)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert ours.df == pytest.approx(ref.df, rel=1e-9)
    assert ours.ci[0] == pytest.approx(ref_ci.low, rel=1e-9)
    assert ours.ci[1] == pytest.approx(ref_ci.high, rel=1e-9)


def test_compare_reports_both_intervals_around_the_estimate():
    rng = np.random.default_rng(1)
    arms = {
        "t": _arm(np.where(rng.random(3_000) < 0.1, 50.0, 0.0)),
        "c": _arm(np.where(rng.random(3_000) < 0.05, 50.0, 0.0)),
    }
    reps = bootstrap_replicates(arms, ["y"], 4_000, rng)
    result = compare(arms["t"], arms["c"], "y", reps["t"]["y"], reps["c"]["y"], names=("t", "c"))
    assert (result.treatment, result.control) == ("t", "c")
    assert result.bca[0] < result.estimate < result.bca[1]
    assert result.welch.ci[0] < result.estimate < result.welch.ci[1]
    # Same data, two methods: the intervals should roughly agree in width.
    assert np.diff(result.bca)[0] == pytest.approx(np.diff(result.welch.ci)[0], rel=0.1)
