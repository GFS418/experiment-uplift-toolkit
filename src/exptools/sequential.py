"""Group-sequential testing: alpha spending, boundaries, and crossing probabilities.

Looking at an experiment repeatedly and stopping at the first |z| > 1.96 tests
the same hypothesis many times, so the false-positive rate climbs with every
look. A group-sequential design fixes the looks in advance and spends the 5%
error budget across them: alpha(t) is how much may have been spent by
information fraction t. The O'Brien-Fleming-type function spends almost
nothing early, so early boundaries are very high and the last one is close to
the fixed-sample 1.96.

Probabilities are computed by numerical integration rather than simulation.
With S(t) = Z(t) * sqrt(t), the statistic has independent normal increments
(a Brownian motion observed at the looks), so the density of the paths that
have not yet stopped can be carried from one look to the next (Armitage,
McPherson and Rowe, 1969).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

GRID_POINTS = 1201  # odd, for Simpson's rule


def obf_spending(t: np.ndarray | float, alpha: float = 0.05) -> np.ndarray:
    """Two-sided alpha spent by information fraction t, O'Brien-Fleming type.

    Each side spends gsDesign's sfLDOF at one-sided level alpha / 2:
    f(t) = 2 - 2 * Phi(Phi^-1(1 - alpha/4) / sqrt(t)), which for alpha = 0.05
    puts 2.2414 inside Phi. This is the convention of standard group-sequential
    software. The older two-sided form 2 - 2 * Phi(1.96 / sqrt(t)) also spends
    0.05 in total but spends more early (first of five boundaries 4.38 vs 4.88).
    """
    per_side = 2 * stats.norm.sf(stats.norm.isf(alpha / 4) / np.sqrt(np.asarray(t, dtype=float)))
    return 2 * per_side


def _simpson_weights(n: int, h: float) -> np.ndarray:
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * h / 3


class _Recursion:
    """Sub-density of S = Z * sqrt(t) over paths that have not crossed a boundary yet."""

    def __init__(self, drift: float, grid_points: int = GRID_POINTS):
        self.drift = drift  # expected z at full information (t = 1)
        self.grid_points = grid_points
        self.t = 0.0
        self.grid: np.ndarray | None = None
        self.mass: np.ndarray | None = None  # density times quadrature weights

    def exit_probabilities(self, c: float, t: float) -> tuple[float, float]:
        """P(first crossing at a look at fraction t with boundary +/-c), upper and lower."""
        b, dt = c * np.sqrt(t), t - self.t
        if self.grid is None:  # first look: S(t) ~ N(drift * t, t)
            sd, centers, mass = np.sqrt(t), np.array([0.0]), np.array([1.0])
        else:
            sd, centers, mass = np.sqrt(dt), self.grid, self.mass
        mean = centers + self.drift * dt
        return float(mass @ stats.norm.sf((b - mean) / sd)), float(mass @ stats.norm.cdf((-b - mean) / sd))

    def advance(self, c: float, t: float) -> None:
        """Move to the look at fraction t, keeping only paths inside +/-c."""
        b, dt = c * np.sqrt(t), t - self.t
        grid = np.linspace(-b, b, self.grid_points)
        if self.grid is None:
            density = stats.norm.pdf(grid, loc=self.drift * t, scale=np.sqrt(t))
        else:
            sd = np.sqrt(dt)
            kernel = stats.norm.pdf((grid[:, None] - self.grid[None, :] - self.drift * dt) / sd) / sd
            density = kernel @ self.mass
        self.grid, self.t = grid, t
        self.mass = density * _simpson_weights(self.grid_points, grid[1] - grid[0])


@dataclass(frozen=True)
class Crossing:
    """Probability of stopping at each look, split by which boundary was crossed."""

    upper: np.ndarray
    lower: np.ndarray

    @property
    def reject(self) -> float:
        return float(self.upper.sum() + self.lower.sum())


def crossing_probabilities(
    bounds: np.ndarray, times: np.ndarray, drift: float = 0.0, grid_points: int = GRID_POINTS
) -> Crossing:
    """First-crossing probabilities of +/-bounds[k] at information fractions times[k].

    `drift` is the expected z-statistic at full information: 0 under the null,
    effect / standard error at the final sample size under an alternative.
    """
    rec = _Recursion(drift, grid_points)
    upper, lower = np.zeros(len(times)), np.zeros(len(times))
    for k, (c, t) in enumerate(zip(bounds, times, strict=True)):
        upper[k], lower[k] = rec.exit_probabilities(c, t)
        rec.advance(c, t)
    return Crossing(upper, lower)


def spending_boundaries(
    times: np.ndarray,
    alpha: float = 0.05,
    spending: Callable[..., np.ndarray] = obf_spending,
    grid_points: int = GRID_POINTS,
) -> np.ndarray:
    """Symmetric two-sided boundaries that spend alpha(t_k) - alpha(t_(k-1)) at look k."""
    times = np.asarray(times, dtype=float)
    if not np.isclose(times[-1], 1.0) or np.any(np.diff(times) <= 0):
        raise ValueError("looks must be increasing information fractions ending at 1")
    budget = np.diff(np.concatenate([[0.0], spending(times, alpha)]))
    rec = _Recursion(0.0, grid_points)
    bounds = np.empty(len(times))
    for k, t in enumerate(times):

        def log_gap(c: float, t: float = t, target: float = budget[k]) -> float:
            # Log scale: early O'Brien-Fleming budgets are as small as 1e-13 or less.
            return np.log(max(sum(rec.exit_probabilities(c, t)), 1e-300)) - np.log(target)

        high = 10.0
        while log_gap(high) > 0:
            high *= 2
        bounds[k] = optimize.brentq(log_gap, 1e-6, high, xtol=1e-10)
        rec.advance(bounds[k], t)
    return bounds


def equally_spaced(k: int) -> np.ndarray:
    return np.arange(1, k + 1) / k


def naive_false_positive_rate(looks: int, z: float = 1.959964, grid_points: int = GRID_POINTS) -> float:
    """Chance of ever seeing |z| > `z` across equally spaced looks when nothing is going on."""
    return crossing_probabilities(np.full(looks, z), equally_spaced(looks), 0.0, grid_points).reject


@dataclass(frozen=True)
class OperatingCharacteristics:
    power: float  # P(reject with the correct sign)
    reject: float  # P(reject at all)
    expected_information: float  # average share of the maximum sample used before stopping
    stop: np.ndarray  # P(stopping at each look); the last look always stops


def operating_characteristics(
    bounds: np.ndarray, times: np.ndarray, drift: float
) -> OperatingCharacteristics:
    cp = crossing_probabilities(bounds, times, drift)
    stop = cp.upper + cp.lower
    stop[-1] = 1 - stop[:-1].sum()
    power = cp.upper.sum() if drift >= 0 else cp.lower.sum()
    return OperatingCharacteristics(float(power), cp.reject, float(stop @ np.asarray(times)), stop)
