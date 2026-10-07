"""Phase 3: power, peeking, and a group-sequential fix (prereg/phase3_design_note.md).

Simulation only; nothing here changes the confirmatory analysis. Every simulated
experiment draws its customers from the real control arm's spend. Section
numbers below refer to the design note.

Writes reports/phase3_simulations.parquet and reports/phase3_power_and_peeking.md.
`--from-cache` rebuilds the report from the parquet without re-simulating.
"""

import sys
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import stats

from exptools.data import CONTROL, REPO_ROOT, load_hillstrom
from exptools.power import (
    buyer_driven_mde,
    buyer_driven_sample_size,
    buyer_driven_variance,
    mde,
    power,
    sample_size,
)
from exptools.sequential import (
    crossing_probabilities,
    equally_spaced,
    naive_false_positive_rate,
    operating_characteristics,
    spending_boundaries,
)
from exptools.simulate_sequential import first_crossing, simulate_z_paths, z_path

N_T, N_C = 21_307, 21_306  # section 3: the men's comparison
DAYS = 14  # section 5: one look per day
SCHEDULES = (1, 2, 3, 5, 7, 10, 14, 20, 28, 50, 100)  # section 4
N_AA, SEED_AA = 20_000, 202631  # sections 4 and 5
N_EFFECT = 10_000  # section 5
EFFECTS = {"+65% (women's observed lift)": (0.65, 202632), "+118% (men's observed lift)": (1.18, 202633)}
N_CALIBRATION = 10_000  # section 6
CALIBRATION_SEEDS = {"equal variances": 202634, "buyer-driven": 202635}
LIFTS = (0.10, 0.25, 0.50)  # section 6
# Post-run diagnostic, not in the design note: added after the first run to explain section 2.
DIAGNOSTIC_SCHEDULES, N_DIAGNOSTIC, SEED_DIAGNOSTIC = (100, 50, 20, 14, 10, 3, 1), 20_000, 202636
Z = stats.norm.isf(0.025)
PUBLISHED_FIVE_LOOKS = (4.877, 3.357, 2.680, 2.290, 2.031)
PHASE1A_LIFT, PHASE1A_POWER, DUNNETT_C = 0.5, 0.653, 2.212  # reports/phase1a_method_validation.md
CACHE = REPO_ROOT / "reports" / "phase3_simulations.parquet"
REPORT = REPO_ROOT / "reports" / "phase3_power_and_peeking.md"

# Written after reading the first run's output (it was empty for that run). The
# simulations are seeded and deterministic, so regenerating reproduces these numbers.
INTERPRETATION = [
    "## 4. Reading the results",
    "",
    "1. **Daily peeking turns a 5% false-positive rate into 21%.** Stopping at the first daily",
    "   |z| > 1.96 over 14 days declares an effect in 21.2% of experiments that have none, four",
    "   times the promised rate. More frequent looks make it worse: 31.9% at 100 looks.",
    "2. **Rare purchases soften very frequent peeking, not daily peeking.** At 100 looks,",
    "   Hillstrom-shaped spend inflates less than normal data (31.9% vs 37.4%) because the earliest",
    "   looks barely count: at 213 customers per arm (about one buyer each), a single look fires",
    "   0.33% of the time instead of 5%. By the daily schedule's first look (1,522 per arm, about 9",
    "   buyers) it is back to 4.4%, so daily peeking on spend (21.2%) is almost as bad as on normal",
    "   data (22.0%).",
    "3. **The O'Brien-Fleming design restores 5%.** On the same 20,000 experiments it declares an",
    "   effect 4.9% of the time (+/-0.3%). It does so by demanding overwhelming evidence early:",
    "   |z| of at least 8.31 on day 1, 3.01 on day 7, and 2.10 on day 14.",
    "4. **It costs about one point of power and buys days.** For a women's-sized effect (+65%),",
    "   power drops from 91.4% to 90.5%, and the decision arrives on day 9.5 on average instead of",
    "   day 14. For a men's-sized effect (+118%), power stays at 100%, and the test stops on day 6.2",
    "   on average, by day 7 in 82% of experiments: a 56% shorter experiment, with the error rate",
    "   still at 5%.",
    "5. **Theory and simulation agree.** The recursion predicts the expected stopping day within",
    "   0.1 day. Simulated power runs about 0.8 points above the normal-theory prediction at +65%",
    "   (roughly 3 Monte Carlo standard errors), plausibly because spend's right skew helps a",
    "   positive effect clear the bar. The theory errs on the cautious side.",
    "6. **The textbook power formula overstates power for spend.** At its own MDE (+48%) it",
    "   promises 80%; the simulation delivers 71.4%, matching the buyer-driven formula's 71.1%. At",
    "   the buyer-driven MDE (+54%), simulated power is 80.7%. The same formula also predicts Phase",
    "   1a's 65% power at a +50% lift almost exactly (65.2% predicted, 65.3% measured), where the",
    "   textbook formula said 75.7%. The reason: an effect that adds buyers adds variance.",
    "7. **What this experiment could detect.** With about 21,300 customers per arm, a spend lift",
    "   needed to be about +54% for 80% power, and conversion about +39%. A +25% spend lift would",
    "   need about 89,000 customers per arm, and a +10% lift about 519,000. Only visits were",
    "   sensitive to modest effects (+8%). The real e-mails' spend effects (+65% and +118%) were",
    "   large enough to detect; smaller ones would likely have been missed.",
    "",
]


def look_grid() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """All schedules' look fractions in one sorted array, and each schedule's positions in it."""
    union = sorted({Fraction(k, n) for n in SCHEDULES for k in range(1, n + 1)})
    position = {f: i for i, f in enumerate(union)}
    index = {n: np.array([position[Fraction(k, n)] for k in range(1, n + 1)]) for n in SCHEDULES}
    return np.array([float(f) for f in union]), index


def moments(x: np.ndarray) -> tuple[float, float]:
    """Mean and E[Y^2] of the control arm: the population every simulation draws from."""
    return float(x.mean()), float((x * x).mean())


def calibration_lifts(spend: np.ndarray) -> dict[str, float]:
    mean, second = moments(spend)
    var = second - mean**2
    return {
        "equal variances": mde(var, var, N_T, N_C) / mean,
        "buyer-driven": buyer_driven_mde(mean, second, N_T, N_C),
    }


def drift(spend: np.ndarray, lift: float) -> float:
    """Expected z at the final look for a buyer-driven lift (exact for the injection model)."""
    mean, second = moments(spend)
    var_t = buyer_driven_variance(mean, second, lift)
    return lift * mean / np.sqrt(var_t / N_T + (second - mean**2) / N_C)


def simulate(spend: np.ndarray) -> pd.DataFrame:
    fractions, index = look_grid()
    daily = equally_spaced(DAYS)
    bounds = spending_boundaries(daily)
    frames = []

    paths = simulate_z_paths(spend, N_T, N_C, fractions, N_AA, SEED_AA)
    aa = pd.DataFrame({f"naive_{n}": (np.abs(paths[:, index[n]]) > Z).any(axis=1) for n in SCHEDULES})
    aa["gsd_stop"] = first_crossing(paths[:, index[DAYS]], bounds)
    frames.append(aa.assign(scenario="A/A"))

    for name, (lift, seed) in EFFECTS.items():
        p = simulate_z_paths(spend, N_T, N_C, daily, N_EFFECT, seed, lift=lift)
        stop = first_crossing(p, bounds)
        z_at_stop = p[np.arange(len(p)), np.maximum(stop, 0)]
        frames.append(
            pd.DataFrame(
                {"gsd_stop": stop, "gsd_positive": (stop >= 0) & (z_at_stop > 0), "z_final": p[:, -1]}
            ).assign(scenario=name)
        )

    for name, lift in calibration_lifts(spend).items():
        z = simulate_z_paths(
            spend, N_T, N_C, np.array([1.0]), N_CALIBRATION, CALIBRATION_SEEDS[name], lift=lift
        )
        frames.append(pd.DataFrame({"z_final": z[:, 0]}).assign(scenario=f"MDE, {name}"))
    return pd.concat(frames, ignore_index=True)


def rate(flags: pd.Series) -> str:
    p = float(flags.astype(float).mean())
    return f"{p:.3f} +/-{1.96 * np.sqrt(p * (1 - p) / flags.size):.3f}"


def power_section(df: pd.DataFrame, sims: pd.DataFrame) -> list[str]:
    spend = df["spend"].to_numpy(dtype=float)
    lines = [
        "## 1. Power: what this experiment can detect (design note, section 6)",
        "",
        f"Two-sided alpha 0.05, 80% power, {N_T:,} vs {N_C:,} customers, control arm's observed mean and",
        "variance. Equal variances is the textbook formula. Buyer-driven lets the treated arm's variance",
        "grow with the lift, as it must when the lift comes from extra buyers; for the 0/1 metrics it is",
        "exactly the binomial variance at the treated rate. For the Dunnett family (cutoff 2.212 instead",
        "of 1.96), multiply every MDE by about 1.09 and every sample size by about 1.19.",
        "",
        "| Metric | Control mean | Control SD | MDE, equal variances | MDE, buyer-driven |",
        "|---|---:|---:|---:|---:|",
    ]
    sizes = []
    for metric in ("spend", "conversion", "visit"):
        mean, second = moments(df[metric].to_numpy(dtype=float))
        var = second - mean**2
        eq = mde(var, var, N_T, N_C)
        bd = buyer_driven_mde(mean, second, N_T, N_C)
        fmt = (lambda v: f"${v:.3f}") if metric == "spend" else (lambda v: f"{100 * v:.2f} pp")
        mean_fmt = f"${mean:.3f}" if metric == "spend" else f"{mean:.2%}"
        sd_fmt = f"${np.sqrt(var):.2f}" if metric == "spend" else f"{np.sqrt(var):.3f}"
        lines.append(
            f"| {metric} | {mean_fmt} | {sd_fmt} | {fmt(eq)} ({eq / mean:+.0%}) "
            f"| {fmt(bd * mean)} ({bd:+.0%}) |"
        )
        sizes.append(
            (
                metric,
                [
                    (sample_size(lift * mean, var, var), buyer_driven_sample_size(mean, second, lift))
                    for lift in LIFTS
                ],
            )
        )
    lines += [
        "",
        "Customers needed per arm (equal variances / buyer-driven):",
        "",
        "| Metric | " + " | ".join(f"{lift:+.0%} lift" for lift in LIFTS) + " |",
        "|---|" + "---:|" * len(LIFTS),
    ]
    for metric, pairs in sizes:
        cells = [f"{np.ceil(a):,.0f} / {np.ceil(b):,.0f}" for a, b in pairs]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    mean, second = moments(spend)
    var = second - mean**2
    lines += [
        "",
        "### Which formula is right? Simulated power at each formula's spend MDE",
        "",
        f"{N_CALIBRATION:,} simulated experiments at each lift, extra buyers injected as in Phase 1a.",
        "",
        "| Lift tested | Equal-variance formula predicts | Buyer-driven formula predicts | Simulated power |",
        "|---|---:|---:|---:|",
    ]
    for name, lift in calibration_lifts(spend).items():
        s = sims[sims["scenario"] == f"MDE, {name}"]
        eq = power(lift * mean, var, var, N_T, N_C)
        bd = power(lift * mean, buyer_driven_variance(mean, second, lift), var, N_T, N_C)
        lines.append(f"| {lift:+.1%} ({name} MDE) | {eq:.3f} | {bd:.3f} | {rate(s['z_final'] > Z)} |")
    eq_1a = power(PHASE1A_LIFT * mean, var, var, N_T, N_C, critical=DUNNETT_C)
    bd_1a = power(
        PHASE1A_LIFT * mean,
        buyer_driven_variance(mean, second, PHASE1A_LIFT),
        var,
        N_T,
        N_C,
        critical=DUNNETT_C,
    )
    lines += [
        "",
        "Phase 1a check (+50% lift, Dunnett cutoff 2.212): the equal-variance formula predicts "
        f"{eq_1a:.3f}, the",
        f"buyer-driven formula {bd_1a:.3f}; Phase 1a's simulation measured {PHASE1A_POWER:.3f}.",
        "",
    ]
    return lines


def peeking_section(sims: pd.DataFrame) -> list[str]:
    aa = sims[sims["scenario"] == "A/A"]
    lines = [
        "## 2. Peeking without a correction (design note, section 4)",
        "",
        f"{N_AA:,} A/A experiments (no effect), each checked under every schedule: stop and declare",
        "significance at the first look with |z| > 1.96. Hillstrom-shaped spend vs normally",
        "distributed data (exact, from the recursion).",
        "",
        "| Looks | Time between looks (14-day test) | False positives, Hillstrom spend "
        "| Normal data (exact) |",
        "|---:|---|---:|---:|",
    ]
    for n in SCHEDULES:
        hours = 14 * 24 / n
        gap = (
            "once, at the end"
            if n == 1
            else (f"{hours / 24:.1f} days" if hours >= 24 else f"{hours:.1f} hours")
        )
        bold = "**" if n == DAYS else ""
        lines.append(
            f"| {bold}{n}{bold} | {gap} | {bold}{rate(aa[f'naive_{n}'])}{bold} "
            f"| {naive_false_positive_rate(n):.3f} |"
        )
    lines.append("")
    return lines


def sequential_section(spend: np.ndarray, sims: pd.DataFrame) -> list[str]:
    daily = equally_spaced(DAYS)
    bounds = spending_boundaries(daily)
    crossing = crossing_probabilities(bounds, daily)
    spent = np.cumsum(crossing.upper + crossing.lower)
    five = spending_boundaries(equally_spaced(5))
    lines = [
        "## 3. The fix: group-sequential design with O'Brien-Fleming-type spending (section 5)",
        "",
        "Boundaries for 14 daily looks, two-sided alpha 0.05, gsDesign's `sfLDOF` convention.",
        "Check: the same code gives five-look boundaries "
        + ", ".join(f"{b:.3f}" for b in five)
        + " against the published "
        + ", ".join(f"{b:.3f}" for b in PUBLISHED_FIVE_LOOKS)
        + ".",
        "",
        "| Day | Information | Stop if \\|z\\| >= | Nominal two-sided p at the boundary "
        "| Alpha spent so far |",
        "|---:|---:|---:|---:|---:|",
    ]
    for day, (t, c, a) in enumerate(zip(daily, bounds, spent, strict=True), start=1):
        lines.append(f"| {day} | {t:.3f} | {c:.3f} | {2 * stats.norm.sf(c):.2g} | {a:.2g} |")

    aa = sims[sims["scenario"] == "A/A"]
    lines += [
        "",
        f"**Validation on the same {N_AA:,} A/A experiments:** false positives "
        f"{rate(aa['gsd_stop'] >= 0)} with the group-sequential boundaries, against "
        f"{rate(aa[f'naive_{DAYS}'])} when peeking daily at 1.96.",
        "",
        "### What it costs and what it buys",
        "",
        f"{N_EFFECT:,} experiments per effect, extra buyers injected into one arm. A single test at day"
        " 14 uses",
        "all customers; the group-sequential design stops at the first day its boundary is crossed.",
        "Theory columns come from the recursion with drift; the simulations use real spend.",
        "",
        "| True effect | Single test: power (sim / theory) | Sequential: power (sim / theory) "
        "| Expected days to decision (sim / theory) | Stopped by day 7 | Median stopping day |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, (lift, _) in EFFECTS.items():
        s = sims[sims["scenario"] == name]
        theta = drift(spend, lift)
        oc = operating_characteristics(bounds, daily, theta)
        day = np.where(s["gsd_stop"] >= 0, s["gsd_stop"] + 1, DAYS)
        lines.append(
            f"| {name} | {(s['z_final'] > Z).mean():.3f} / {stats.norm.sf(Z - theta):.3f} "
            f"| {s['gsd_positive'].astype(float).mean():.3f} / {oc.power:.3f} "
            f"| {day.mean():.1f} / {DAYS * oc.expected_information:.1f} "
            f"| {(day <= 7).mean():.0%} | {np.median(day):.0f} |"
        )
    lines.append("")
    return lines


def diagnostic_section(spend: np.ndarray) -> list[str]:
    rng = np.random.default_rng(SEED_DIAGNOSTIC)
    share = (spend > 0).mean()
    lines = [
        "### Post-run diagnostic: why rare purchases dampen very frequent peeking",
        "",
        "Added after the first run to explain the table above; not part of the design note. The",
        f"false-positive rate of one look at each schedule's first-look sample size ({N_DIAGNOSTIC:,} A/A",
        f"draws each, seed {SEED_DIAGNOSTIC}). With k buyers in one arm and none in the other, z is about",
        "sqrt(k), so with only a few buyers a crossing needs a lopsided split such as 4 against 0.",
        "",
        "| Schedule | First look: customers per arm | Expected buyers per arm "
        "| One look's false-positive rate |",
        "|---:|---:|---:|---:|",
    ]
    for looks in DIAGNOSTIC_SCHEDULES:
        n = round(N_T / looks)
        z = np.array(
            [
                z_path(rng.choice(spend, n), rng.choice(spend, n), np.array([1.0]))[0]
                for _ in range(N_DIAGNOSTIC)
            ]
        )
        label = f"{looks} look{'s' if looks > 1 else ''}"
        lines.append(f"| {label} | {n:,} | {n * share:.1f} | {np.mean(np.abs(z) > Z):.4f} |")
    lines.append("")
    return lines


def main(from_cache: bool) -> None:
    df = load_hillstrom(load_outcomes=True, arms=[CONTROL])
    spend = df["spend"].to_numpy(dtype=float)
    if from_cache:
        sims = pd.read_parquet(CACHE)
    else:
        sims = simulate(spend)
        sims.to_parquet(CACHE)
    lines = [
        "# Phase 3: power, peeking, and a group-sequential fix",
        "",
        "Generated by `scripts/sequential_peeking.py` from the design note committed before this run",
        "(`prereg/phase3_design_note.md`, commit `d828858`). Simulation only: every experiment draws its",
        "customers from the real control arm, and nothing here changes the confirmatory analysis.",
        "Customers are assumed to arrive in random order over 14 days (Hillstrom has no timestamps).",
        "",
        *power_section(df, sims),
        *peeking_section(sims),
        *diagnostic_section(spend),
        *sequential_section(spend, sims),
        *INTERPRETATION,
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
