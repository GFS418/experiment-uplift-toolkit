"""Criteo Uplift Prediction Dataset v2.1: download, verify, convert, split.

Source: Criteo's own Hugging Face organization (criteo/criteo-uplift); the
download URL on Criteo's dataset page redirects to a storage host that no
longer exists. License CC BY-NC-SA 4.0; cite Diemert, Betlei, Renaudin and
Amini (2018), "A Large Scale Benchmark for Uplift Modeling" (AdKDD). The
repository ships this downloader, never the data (Phase 4c design note).
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from exptools.data import REPO_ROOT, halves_by_arm

URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"  # published by the host
N_BYTES = 311_422_618
N_ROWS = 13_979_592  # v2.1; the leaky first release had 25,309,483
RAW = REPO_ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PARQUET = REPO_ROOT / "data" / "processed" / "criteo-uplift-v2.1.parquet"

FEATURES = tuple(f"f{i}" for i in range(12))
OUTCOMES = ("visit", "conversion")
COLUMNS = (*FEATURES, "treatment", "conversion", "visit", "exposure")
BINARY = ("treatment", "conversion", "visit", "exposure")
CONTROL, TREATED = "control", "treated"
CARD = {"visit": 0.046992, "conversion": 0.00292, "treatment": 0.85}  # the dataset card's key figures
SPLIT_SEED = 2026  # note, section 4
SUBSAMPLE_SEED = 202651  # note, section 4


def require_features(columns: Sequence[str]) -> None:
    """Only the 12 anonymized features may enter a model.

    `exposure` (was an ad actually shown?) is decided after treatment, so it
    is refused here along with the outcomes and the treatment itself.
    """
    if leaked := [c for c in columns if c not in FEATURES]:
        raise ValueError(f"not a pre-treatment feature, refusing to use: {leaked}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path = RAW) -> Path:
    """Fetch the compressed CSV and verify its size and the host's published SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    if path.stat().st_size != N_BYTES or sha256(path) != SHA256:
        raise ValueError(f"{path.name} does not match the published size and SHA-256")
    return path


def validate(df: pd.DataFrame) -> None:
    """Schema and domain checks over every row; none of them computes an outcome statistic."""
    problems = []
    if len(df) != N_ROWS:
        problems.append(f"expected {N_ROWS:,} rows (v2.1), got {len(df):,}")
    if tuple(df.columns) != COLUMNS:
        problems.append(f"unexpected columns {tuple(df.columns)}")
    if df.isna().any().any():
        problems.append("missing values")
    for col in BINARY:
        if col in df and not df[col].isin((0, 1)).all():
            problems.append(f"{col} is not 0/1")
    if problems:
        raise ValueError("Criteo checks failed: " + "; ".join(problems))


def convert(raw: Path = RAW, parquet: Path = PARQUET) -> Path:
    """One-time CSV-to-Parquet conversion: float32 features, int8 flags."""
    if parquet.exists():
        return parquet
    dtypes = {**dict.fromkeys(FEATURES, "float32"), **dict.fromkeys(BINARY, "int8")}
    df = pd.read_csv(raw, dtype=dtypes)
    validate(df)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)
    return parquet


def load(columns: Sequence[str] | None = None, parquet: Path = PARQUET) -> pd.DataFrame:
    return pd.read_parquet(parquet, columns=list(columns) if columns is not None else None)


def arm_labels(treatment: np.ndarray) -> np.ndarray:
    """String arm labels, so the Phase 4 learners and scores apply unchanged."""
    return np.where(np.asarray(treatment) == 1, TREATED, CONTROL).astype(object)


def split_halves(treatment: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """'train' or 'test' per row, stratified by treatment (control rows shuffled first)."""
    return halves_by_arm(arm_labels(treatment), np.random.default_rng(seed), (CONTROL, TREATED))


def nested_subsample_order(arm: np.ndarray, seed: int = SUBSAMPLE_SEED) -> dict[str, np.ndarray]:
    """One seeded shuffle of each arm's rows; every training size takes a prefix of these."""
    rng = np.random.default_rng(seed)
    return {a: rng.permutation(np.flatnonzero(arm == a)) for a in (CONTROL, TREATED)}


def nested_subsample(arm: np.ndarray, n: int, order: dict[str, np.ndarray]) -> np.ndarray:
    """Row positions of a size-n sample that keeps the treatment share; larger n contain smaller."""
    n_treated = round(n * len(order[TREATED]) / len(arm))
    rows = np.concatenate([order[TREATED][:n_treated], order[CONTROL][: n - n_treated]])
    return np.sort(rows)
