import numpy as np
import pandas as pd
import pytest

from exptools.balance import joint_balance_test, standardized_differences


def _synthetic(n: int, rng: np.random.Generator, assignment=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "flag": rng.integers(0, 2, size=n),
            "region": rng.choice(["north", "south", "west"], size=n),
        }
    )
    df["arm"] = assignment if assignment is not None else rng.choice(["c", "t1", "t2"], size=n)
    return df


def test_smd_known_value():
    df = pd.DataFrame({"arm": ["c"] * 4 + ["t"] * 4, "x": [0, 0, 2, 2, 1, 1, 3, 3]})
    # means 1 and 2; both sample variances 4/3; smd = 1 / sqrt(4/3)
    out = standardized_differences(df, "arm", "c", ["x"])
    assert out.loc[0, "smd"] == pytest.approx(1 / np.sqrt(4 / 3))
    assert out.loc[0, "z"] == pytest.approx(out.loc[0, "smd"] / np.sqrt(1 / 4 + 1 / 4))


def test_smd_expands_every_categorical_level():
    df = _synthetic(600, np.random.default_rng(0))
    out = standardized_differences(df, "arm", "c", ["x", "flag", "region"])
    assert set(out["arm"]) == {"t1", "t2"}
    assert set(out["covariate"]) == {"x", "flag", "region_north", "region_south", "region_west"}


def test_smd_z_is_standard_normal_under_randomization():
    # Across many randomizations, z should have sd near 1 (it is the chance scale).
    rng = np.random.default_rng(11)
    base = _synthetic(4_000, rng)
    zs = [
        standardized_differences(
            base.assign(arm=rng.choice(["c", "t1", "t2"], size=len(base))), "arm", "c", ["x"]
        )["z"].iloc[0]
        for _ in range(400)
    ]
    assert np.std(zs) == pytest.approx(1.0, abs=0.12)


def test_joint_test_false_positive_rate_matches_alpha_by_simulation():
    rng = np.random.default_rng(2026)
    n_sims, alpha = 400, 0.05
    p_values = [
        joint_balance_test(_synthetic(1_500, rng), "arm", ["x", "flag", "region"]).p_value
        for _ in range(n_sims)
    ]
    rate = np.mean(np.array(p_values) < alpha)
    assert abs(rate - alpha) < 3 * np.sqrt(alpha * (1 - alpha) / n_sims)


def test_joint_test_detects_covariate_dependent_assignment():
    rng = np.random.default_rng(3)
    df = _synthetic(3_000, rng)
    # Broken randomization: high-x units are pushed into t1.
    df.loc[df["x"] > 1, "arm"] = "t1"
    result = joint_balance_test(df, "arm", ["x", "flag", "region"])
    assert result.p_value < 1e-6
    assert result.df == 2 * 4  # 2 non-reference arms x (x, flag, 2 region dummies)
