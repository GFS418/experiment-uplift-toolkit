"""Hillstrom MineThatData e-mail experiment: download, verify, and load.

The loader is blind by default. It reads the assignment column and the
pre-treatment covariates only; the outcome columns (visit, conversion, spend)
need an explicit ``load_outcomes=True``, which the analysis plan forbids until
the plan itself is committed (see ``prereg/analysis_plan.md``).

Source: Kevin Hillstrom, "MineThatData E-Mail Analytics And Data Mining
Challenge" (March 2008). The file is served over plain HTTP only, so integrity
rests on the pinned SHA-256 below. No license is stated, so the repository
ships this downloader rather than the data.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
N_ROWS = 64_000

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "raw" / "hillstrom.csv"

ASSIGNMENT = "segment"
CONTROL = "No E-Mail"
ARMS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")

# Measured over the year before the e-mail was sent, so unaffected by treatment.
PRE_TREATMENT = (
    "recency",  # months since last purchase
    "history_segment",  # binned version of `history`
    "history",  # dollars spent in the past year
    "mens",  # 1 = bought men's merchandise in the past year
    "womens",  # 1 = bought women's merchandise in the past year
    "zip_code",  # Urban / Suburban / Rural
    "newbie",  # 1 = new customer in the past twelve months
    "channel",  # purchase channel(s) in the past year
)
# Measured in the two weeks after the e-mail.
OUTCOMES = ("visit", "conversion", "spend")
FILE_COLUMNS = (*PRE_TREATMENT, ASSIGNMENT, *OUTCOMES)

CATEGORIES = {
    "zip_code": ("Urban", "Surburban", "Rural"),  # sic: the source file's spelling
    "channel": ("Phone", "Web", "Multichannel"),
}


def require_pre_treatment(columns: Sequence[str]) -> None:
    """Refuse any column that is not measured before the e-mail went out.

    Adjusting for a variable the treatment can change (a visit, say) removes
    part of the treatment effect along with the noise, so every adjustment
    and model input passes through this check.
    """
    if leaked := [c for c in columns if c not in PRE_TREATMENT]:
        raise ValueError(f"not pre-treatment, refusing to adjust for: {leaked}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path = DEFAULT_PATH, *, overwrite: bool = False) -> Path:
    """Fetch the raw CSV and verify it against the pinned checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        verify(path)
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(HILLSTROM_URL, tmp)
    tmp.replace(path)
    verify(path)
    return path


def verify(path: Path = DEFAULT_PATH) -> None:
    """Check the checksum and the header without reading any data values."""
    actual = sha256(path)
    if actual != HILLSTROM_SHA256:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {HILLSTROM_SHA256}, got {actual}")
    with open(path) as f:
        header = tuple(f.readline().strip().split(","))
    if header != FILE_COLUMNS:
        raise ValueError(f"unexpected header: {header}")


def load_hillstrom(
    path: Path = DEFAULT_PATH,
    *,
    load_outcomes: bool = False,
    arms: Sequence[str] = ARMS,
) -> pd.DataFrame:
    """Load the experiment, outcome columns excluded unless explicitly requested.

    ``arms`` keeps only those arms' rows, dropped before any outcome check runs,
    so method validation can read the control arm's outcomes and nothing else.
    """
    if unknown := set(arms) - set(ARMS):
        raise ValueError(f"unknown arms: {unknown}")
    verify(path)
    columns = [*PRE_TREATMENT, ASSIGNMENT, *(OUTCOMES if load_outcomes else ())]
    df = pd.read_csv(path, usecols=columns)[columns]
    _validate_pre_treatment(df)
    df = df[df[ASSIGNMENT].isin(arms)].reset_index(drop=True)
    if load_outcomes:
        _validate_outcomes(df)
    return df


def _validate_pre_treatment(df: pd.DataFrame) -> None:
    problems = []
    if len(df) != N_ROWS:
        problems.append(f"expected {N_ROWS} rows, got {len(df)}")
    if df.isna().any().any():
        problems.append(f"nulls in {df.columns[df.isna().any()].tolist()}")
    if not df[ASSIGNMENT].isin(ARMS).all():
        problems.append(f"unknown arms: {set(df[ASSIGNMENT]) - set(ARMS)}")
    for col in ("mens", "womens", "newbie"):
        if not df[col].isin((0, 1)).all():
            problems.append(f"{col} is not 0/1")
    if not df["recency"].between(1, 12).all():
        problems.append("recency outside 1-12 months")
    if (df["history"] < 0).any():
        problems.append("negative history")
    for col, levels in CATEGORIES.items():
        if not df[col].isin(levels).all():
            problems.append(f"unknown {col} levels: {set(df[col]) - set(levels)}")
    if problems:
        raise ValueError("pre-treatment checks failed: " + "; ".join(problems))


def _validate_outcomes(df: pd.DataFrame) -> None:
    problems = []
    for col in ("visit", "conversion"):
        if not df[col].isin((0, 1)).all():
            problems.append(f"{col} is not 0/1")
    if df[list(OUTCOMES)].isna().any().any():
        problems.append("nulls in outcomes")
    if (df["spend"] < 0).any():
        problems.append("negative spend")
    if problems:
        raise ValueError("outcome checks failed: " + "; ".join(problems))
