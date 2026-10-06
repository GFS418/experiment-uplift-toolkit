import shutil

import pandas as pd
import pytest

from exptools import data

needs_raw = pytest.mark.skipif(
    not data.DEFAULT_PATH.exists(), reason="raw data not downloaded (scripts/download_data.py)"
)


@needs_raw
def test_blind_load_excludes_outcomes():
    df = data.load_hillstrom()
    assert len(df) == data.N_ROWS
    assert list(df.columns) == [*data.PRE_TREATMENT, data.ASSIGNMENT]
    assert not set(data.OUTCOMES) & set(df.columns)


@needs_raw
def test_checksum_catches_a_modified_file(tmp_path):
    copy = tmp_path / "hillstrom.csv"
    shutil.copy(data.DEFAULT_PATH, copy)
    with open(copy, "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.verify(copy)


def test_outcome_validation_on_synthetic_rows():
    # Exercised on synthetic rows so the real outcomes stay unread until the
    # analysis plan is committed.
    good = pd.DataFrame({"visit": [0, 1], "conversion": [0, 1], "spend": [0.0, 29.99]})
    data._validate_outcomes(good)
    with pytest.raises(ValueError, match="negative spend"):
        data._validate_outcomes(good.assign(spend=[0.0, -1.0]))
    with pytest.raises(ValueError, match="visit is not 0/1"):
        data._validate_outcomes(good.assign(visit=[0, 2]))


@needs_raw
def test_arms_filter_keeps_only_the_requested_arms():
    # Phase 1a's guarantee: method validation reads the control arm's rows only.
    df = data.load_hillstrom(arms=[data.CONTROL])
    assert set(df[data.ASSIGNMENT]) == {data.CONTROL}
    assert len(df) == 21_306


def test_arms_filter_rejects_unknown_arms():
    with pytest.raises(ValueError, match="unknown arms"):
        data.load_hillstrom(arms=["Spam E-Mail"])
