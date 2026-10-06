import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from exptools.bootstrap import Arm, bootstrap_replicates
from exptools.multiplicity import benjamini_hochberg, dunnett_maxt, dunnett_parametric


def _arm(values) -> Arm:
    return Arm.from_frame(pd.DataFrame({"y": values}), ["y"])


def test_bh_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.concatenate([rng.uniform(size=5), rng.uniform(0, 0.02, size=2)])
        adjusted, rejected = benjamini_hochberg(p, 0.05)
        ref_reject, ref_adjusted, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
        assert np.allclose(adjusted, ref_adjusted)
        assert np.array_equal(rejected, ref_reject)


def test_bh_known_example():
    # Sorted thresholds for m = 4 at q = 0.05: 0.0125, 0.025, 0.0375, 0.05.
    # 0.03 fails its own threshold (0.025) but is rescued by 0.035 <= 0.0375.
    _, rejected = benjamini_hochberg([0.01, 0.03, 0.035, 0.2], 0.05)
    assert rejected.tolist() == [True, True, True, False]


def _normal_arms(rng, n=20_000, shifts=(0.0, 0.0)):
    # Rounded to 2 decimals so the compressed representation stays small.
    control = _arm(np.round(rng.normal(size=n), 2))
    treatments = {f"t{i}": _arm(np.round(rng.normal(s, 1.0, size=n), 2)) for i, s in enumerate(shifts)}
    return treatments, control


def test_maxt_critical_value_matches_dunnett_theory():
    # Normal data, equal variances and sizes: the two statistics correlate at
    # 0.5 and the two-sided 95% Dunnett critical value is 2.212.
    rng = np.random.default_rng(1)
    treatments, control = _normal_arms(rng)
    reps = bootstrap_replicates({**treatments, "c": control}, ["y"], 20_000, rng)
    result = dunnett_maxt(treatments, control, "y", {a: reps[a]["y"] for a in treatments}, reps["c"]["y"])
    assert result.critical_value == pytest.approx(2.212, abs=0.04)
    assert np.all(result.ci_low < result.estimate) and np.all(result.estimate < result.ci_high)


def test_maxt_and_parametric_agree_on_normal_data():
    rng = np.random.default_rng(2)
    treatments, control = _normal_arms(rng, shifts=(0.03, 0.0))
    reps = bootstrap_replicates({**treatments, "c": control}, ["y"], 20_000, rng)
    maxt = dunnett_maxt(treatments, control, "y", {a: reps[a]["y"] for a in treatments}, reps["c"]["y"])
    parametric = dunnett_parametric(treatments, control, "y", rng)
    assert maxt.p_adjusted == pytest.approx(parametric.p_adjusted, abs=0.01)
    assert maxt.ci_low == pytest.approx(parametric.ci_low, abs=0.002)


def test_parametric_wrapper_reproduces_scipy():
    rng = np.random.default_rng(3)
    treatments, control = _normal_arms(rng, n=500, shifts=(0.2, -0.1))
    ours = dunnett_parametric(treatments, control, "y", np.random.default_rng(9))
    ref = stats.dunnett(
        *(t.expand("y") for t in treatments.values()),
        control=control.expand("y"),
        rng=np.random.default_rng(9),
    )
    assert np.allclose(ours.p_adjusted, ref.pvalue)
    assert np.allclose(ours.estimate / ours.se, ref.statistic)  # same pooled standard error
