"""Fingerprints of fitted models' predictions.

Phases 4a and 4c-a freeze each final model by fitting it twice and recording a
fingerprint of its predictions; Phases 4b and 4c-b refit and must reproduce the
same fingerprint before they may score a test half. One definition, so the
freezing and the checking can never drift apart.
"""

from __future__ import annotations

import hashlib

import numpy as np


def fingerprint(values: np.ndarray) -> str:
    """First 16 hex digits of the SHA-256 of the predictions rounded to 10 decimals.

    Rounding absorbs most last-bit floating-point noise, but a value sitting on a
    rounding boundary can still flip, so the check relies on refits being bit-for-bit
    deterministic (verified by fitting every frozen model twice). The dtype is left
    as given: the committed fingerprints were computed that way.
    """
    return hashlib.sha256(np.round(values, 10).tobytes()).hexdigest()[:16]
