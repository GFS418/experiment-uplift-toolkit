import numpy as np
import pandas as pd
import pytest

from exptools import criteo


def _treatment(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < 0.85).astype(np.int8)


def test_split_is_stratified_and_deterministic():
    treatment = _treatment(10_001, np.random.default_rng(0))
    half = criteo.split_halves(treatment)
    for flag in (0, 1):
        rows = treatment == flag
        assert (half[rows] == "train").sum() == rows.sum() // 2
    assert (criteo.split_halves(treatment) == half).all()


def test_nested_subsamples_keep_the_treatment_share_and_nest():
    arm = criteo.arm_labels(_treatment(50_000, np.random.default_rng(1)))
    order = criteo.nested_subsample_order(arm, seed=3)
    small, large = (criteo.nested_subsample(arm, n, order) for n in (2_000, 20_000))
    assert len(small) == 2_000 and len(large) == 20_000
    assert set(small) <= set(large)
    share = np.mean(arm == criteo.TREATED)
    assert np.mean(arm[large] == criteo.TREATED) == pytest.approx(share, abs=1e-3)


def test_only_the_anonymized_features_may_enter_a_model():
    criteo.require_features([f"f{i}" for i in range(12)])
    for leaked in ("exposure", "visit", "treatment"):
        with pytest.raises(ValueError, match=leaked):
            criteo.require_features(["f0", leaked])


def test_validation_names_every_problem():
    df = pd.DataFrame({c: [0, 1] for c in criteo.COLUMNS}).astype("float32")
    df.loc[0, "visit"] = 2
    with pytest.raises(ValueError, match="13,979,592 rows") as err:
        criteo.validate(df)
    assert "visit is not 0/1" in str(err.value)


def test_arm_labels_are_strings_for_the_phase4_learners():
    assert criteo.arm_labels(np.array([1, 0, 1])).tolist() == ["treated", "control", "treated"]
