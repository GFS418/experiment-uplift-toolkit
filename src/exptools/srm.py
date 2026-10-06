"""Sample-ratio-mismatch (SRM) check.

If the arm sizes differ from the designed allocation by more than chance
allows, the randomization or the logging is broken (for example, one arm loses
users to a bug). Effect estimates from such an experiment are not trustworthy,
so the check runs before any outcome is looked at.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Alarm threshold. Strict on purpose: the test is run on every experiment, so a
# loose threshold raises false alarms, and a true SRM is usually so large that
# its p-value is far below this.
SRM_ALPHA = 0.001


@dataclass(frozen=True)
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    df: int
    p_value: float
    alpha: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.alpha


def srm_test(
    counts: Mapping[str, int],
    allocation: Mapping[str, float],
    *,
    alpha: float = SRM_ALPHA,
) -> SRMResult:
    """Chi-square goodness-of-fit test of observed arm sizes against the design.

    ``allocation`` holds the designed share of units per arm; it is normalized,
    so ``{"a": 1, "b": 1}`` and ``{"a": 0.5, "b": 0.5}`` mean the same thing.
    """
    if set(counts) != set(allocation):
        raise ValueError(f"arms differ: {sorted(counts)} vs {sorted(allocation)}")
    arms = sorted(counts)
    observed = np.array([counts[a] for a in arms], dtype=float)
    shares = np.array([allocation[a] for a in arms], dtype=float)
    if (shares <= 0).any():
        raise ValueError("every arm needs a positive allocation")
    expected = observed.sum() * shares / shares.sum()
    chi2, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        observed={a: int(o) for a, o in zip(arms, observed, strict=True)},
        expected={a: float(e) for a, e in zip(arms, expected, strict=True)},
        chi2=float(chi2),
        df=len(arms) - 1,
        p_value=float(p_value),
        alpha=alpha,
    )
