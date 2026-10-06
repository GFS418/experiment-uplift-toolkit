"""Phase 1a: validate the pre-registered methods on the control arm, before any comparison.

Reads outcomes for the control arm only. Runs two plasmode scenarios built from
real control customers (pre-registration, section 5):

- complete null: three pseudo-arms from control at the real arm sizes, so every
  true difference is zero;
- known effect: the same, but the pseudo-men's arm gains buyers at a rate that
  lifts mean spend by 50%, while the pseudo-women's arm stays null.

Writes reports/phase1a_simulations.parquet (one row per simulated experiment)
and reports/phase1a_method_validation.md. `--from-cache` rebuilds the report
from the parquet without re-simulating.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from exptools.bootstrap import Arm
from exptools.data import ARMS, ASSIGNMENT, CONTROL, OUTCOMES, REPO_ROOT, load_hillstrom
from exptools.simulate import Design, Scenario, run_scenario

N_SIMS = 2_000
MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
LIFT = 0.5
SCENARIOS = {
    "complete null": (Scenario("complete null", 0.0), 202601),
    "known effect": (Scenario("known effect", LIFT, MENS), 202602),
}
# Arm sizes are assignment counts, known since Phase 0 (no outcomes involved).
SIZES = {"No E-Mail": 21_306, MENS: 21_307, WOMENS: 21_387}
CACHE = REPO_ROOT / "reports" / "phase1a_simulations.parquet"
REPORT = REPO_ROOT / "reports" / "phase1a_method_validation.md"
DUNNETT_C = 2.212  # two-sided 95%, two comparisons sharing a control (Phase 0)

# Written after reading the seeded run (seeds 202601 and 202602). The simulation
# is deterministic, so regenerating the report reproduces the numbers cited here.
INTERPRETATION = [
    "## Reading the results",
    "",
    "1. **Every method holds its error rate on data shaped like real spend.** Despite a",
    "   skewness of 26, intervals cover 0.946 to 0.956 and both Dunnett versions keep the",
    "   family-wise error at 0.048 under the complete null. With about 21,000 customers per",
    "   arm, the central limit theorem tames the tail. The bootstrap was validated, not",
    "   needed: Welch would also have been fine. This is a result that did not matter, and it is",
    "   reported as such.",
    "2. **The complete null cannot expose skewness problems in a difference.** When both arms",
    "   come from the same distribution at nearly equal sizes, their skewness cancels in",
    "   the difference, so misses split evenly. The known-effect scenario is the real test;",
    "   there, too, misses stay within Monte Carlo error of 0.025 on each side.",
    "3. **SciPy's parametric Dunnett is lopsided when variances differ.** It pools one",
    "   variance across arms. When the men's arm gains buyers, its variance grows by about",
    "   1.5x, which inflates the pooled variance. The women's comparison becomes too",
    "   strict (false-positive rate 0.015 against an expected 0.027; women's interval",
    "   covers 0.986), and the men's becomes slightly too loose (covers 0.970 against",
    "   0.973), which is where its extra power comes from (0.671 against 0.653). A",
    "   back-of-envelope calculation predicts 0.017 and 0.967. The bootstrap max-T stays",
    "   near 0.977 on both arms. This is the pre-registered reason for making max-T primary,",
    "   now backed by evidence.",
    "4. **The primary metric is underpowered for moderate effects.** A true +50% lift in",
    "   spend is detected only about 65% of the time. If the real effects turn out smaller,",
    '   a non-significant result means "not detectable with 21,000 customers per arm,"',
    '   not "no effect." This is written down before any treatment-arm outcome is read.',
    "",
]


def rate(flags: pd.Series) -> tuple[float, float]:
    p = float(flags.mean())
    return p, float(np.sqrt(p * (1 - p) / flags.size))


def coverage_row(sims: pd.DataFrame, key: str, label: str) -> tuple[str, bool]:
    low, high = sims[f"{key}_miss_low"], sims[f"{key}_miss_high"]
    cov, se = rate(~(low | high))
    under = cov + 1.96 * se < 0.95
    row = (
        f"| {label} | {cov:.3f} | +/-{1.96 * se:.3f} | {low.mean():.3f} | {high.mean():.3f} "
        f"| {'**under-covers**' if under else 'ok'} |"
    )
    return row, under


def main(from_cache: bool) -> None:
    control = load_hillstrom(load_outcomes=True, arms=[CONTROL])
    assert set(control[ASSIGNMENT]) == {CONTROL}, "Phase 1a may read the control arm only"
    pool = Arm.from_frame(control, OUTCOMES)
    design = Design(pool=pool, sizes=SIZES, control=CONTROL)
    assert tuple(SIZES) == ARMS

    if from_cache:
        sims = pd.read_parquet(CACHE)
    else:
        sims = pd.concat(
            [run_scenario(design, sc, N_SIMS, seed) for sc, seed in SCENARIOS.values()], ignore_index=True
        )
        sims.to_parquet(CACHE)

    spend = control["spend"].to_numpy()
    sd, skew = spend.std(ddof=1), stats.skew(spend)
    mde = (DUNNETT_C + stats.norm.ppf(0.8)) * sd * np.sqrt(1 / SIZES[MENS] + 1 / SIZES[CONTROL])
    lines = [
        "# Phase 1a: method validation on the control arm",
        "",
        "Generated by `scripts/validate_methods.py`. Only control-arm outcomes were read;",
        "no treatment arm's outcomes and no comparison between real arms were computed.",
        "",
        "## The data the methods must handle (control arm)",
        "",
        "| | |",
        "|---|---:|",
        f"| Customers | {len(control):,} |",
        f"| Visit rate | {control['visit'].mean():.2%} |",
        f"| Conversion rate (= share with spend > 0) | {control['conversion'].mean():.2%} |",
        f"| Buyers | {int((spend > 0).sum())} |",
        f"| Mean spend per customer | ${spend.mean():.3f} |",
        f"| Standard deviation of spend | ${sd:.2f} |",
        f"| Skewness of spend | {skew:.1f} |",
        f"| Implied skewness of an arm mean (skewness / sqrt(n)) | {skew / np.sqrt(len(spend)):.2f} |",
        f"| Distinct (visit, conversion, spend) rows | {len(pool.counts)} |",
        "",
        "Consistency of the outcome definitions in the control arm: "
        f"{int(((control.conversion == 1) & (control.visit == 0)).sum())} converters without a visit, "
        f"{int(((control.spend > 0) != (control.conversion == 1)).sum())} rows where "
        "spend > 0 and conversion disagree.",
        "",
        "**Minimum detectable effect on spend, now in dollars.** At the Dunnett critical value",
        f"(2.212) and 80% power, with the control arm's standard deviation for both arms: "
        f"**${mde:.3f} per customer (${1000 * mde:,.0f} per 1,000), "
        f"{mde / spend.mean():.0%} of control's mean**.",
        "An e-mail that adds buyers also adds variance, so the true MDE is somewhat larger.",
        "",
        "## Simulation design",
        "",
        f"{N_SIMS:,} simulated experiments per scenario, each analyzed with exactly the",
        "pre-registered procedure: B = 10,000 bootstrap resamples (no reduction was needed),",
        "BCa and Welch intervals, the bootstrap max-T Dunnett, and SciPy's parametric Dunnett.",
        "",
        "- **Complete null:** three pseudo-arms resampled from real control customers at the",
        "  real arm sizes. Every true difference is zero.",
        "- **Known effect:** the pseudo-men's arm converts non-buyers into buyers (outcome rows",
        f"  drawn from real control buyers) at the rate that lifts mean spend by {LIFT:.0%}:",
        f"  a true effect of ${LIFT * spend.mean():.3f} per customer. The pseudo-women's arm stays null.",
        "  The 50% was chosen because it sits near the MDE above, from control data only.",
        "",
        "Coverage should be 0.95, with misses split evenly: 0.025 with the interval entirely",
        "above the truth (miss low) and 0.025 entirely below it (miss high). Monte Carlo",
        "margin = 1.96 x the simulation's standard error. Pre-registered rule: a method",
        "under-covers when coverage + margin < 0.95.",
        "",
    ]

    under = {}
    for name in SCENARIOS:
        s = sims[sims["scenario"] == name]
        lines += [
            f"## Scenario: {name}",
            "",
            "### Spend: marginal 95% intervals",
            "",
            "| Comparison and method | Coverage | MC margin | Miss low | Miss high | Verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for arm in (MENS, WOMENS):
            for method in ("bca", "welch"):
                row, flag = coverage_row(s, f"spend|{arm}|{method}", f"{arm} vs control, {method.upper()}")
                lines.append(row)
                under[(name, arm, method)] = flag
        lines += [
            "",
            "### Spend: Dunnett (simultaneous 95% intervals, family-wise error)",
            "",
            "| Implementation | Mean critical value | Family-wise error | Rejects men's | Rejects women's "
            "| Simultaneous coverage | Men's interval covers | Women's interval covers |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        true_nulls = [WOMENS] if name == "known effect" else [MENS, WOMENS]
        for label in ("maxt", "parametric"):
            rejects = {a: s[f"dunnett|{label}|{a}|reject"] for a in (MENS, WOMENS)}
            fwer, fwer_se = rate(pd.concat([rejects[a] for a in true_nulls], axis=1).any(axis=1))
            missed = pd.concat(
                [
                    s[f"dunnett|{label}|{a}_miss_low"] | s[f"dunnett|{label}|{a}_miss_high"]
                    for a in (MENS, WOMENS)
                ],
                axis=1,
                keys=(MENS, WOMENS),
            )
            sim_cov, sim_se = rate(~missed.any(axis=1))
            lines.append(
                f"| {'Bootstrap max-T' if label == 'maxt' else 'SciPy parametric'} "
                f"| {s[f'dunnett|{label}|critical'].mean():.3f} | {fwer:.3f} +/-{1.96 * fwer_se:.3f} "
                f"| {rejects[MENS].mean():.3f} | {rejects[WOMENS].mean():.3f} "
                f"| {sim_cov:.3f} +/-{1.96 * sim_se:.3f} "
                f"| {1 - missed[MENS].mean():.3f} | {1 - missed[WOMENS].mean():.3f} |"
            )
        lines += [
            "",
            f"Family-wise error counts false rejections of true nulls only ({', '.join(true_nulls)}).",
            "Each simultaneous interval alone should cover about 0.973 (two-sided, cutoff 2.212).",
        ]
        if name == "known effect":
            lines.append("The 'rejects men's' column is power at a true +50% lift.")
        lines.append("")

        if name == "complete null":
            hyps = [c[: -len("|welch_reject")] for c in s.columns if c.endswith("|welch_reject")]
            lines += [
                "### Exploratory family (7 hypotheses, every null true)",
                "",
                "| Hypothesis | Welch false-positive rate (alpha 0.05) | BCa coverage |",
                "|---|---:|---:|",
            ]
            for h in hyps:
                fp, fp_se = rate(s[f"{h}|welch_reject"])
                cov, _ = rate(~(s[f"{h}|bca_miss_low"] | s[f"{h}|bca_miss_high"]))
                metric, t, ctl = h.split("|")
                lines.append(f"| {metric}: {t} vs {ctl} | {fp:.3f} +/-{1.96 * fp_se:.3f} | {cov:.3f} |")
            bh, bh_se = rate(s["bh|any_reject"])
            lines += [
                "",
                f"Benjamini-Hochberg at q = 0.05 rejected at least one of the 7 true nulls in "
                f"**{bh:.3f} +/-{1.96 * bh_se:.3f}** of experiments. With every null true, that is its",
                "false discovery rate, which BH promises to hold at or below 0.05.",
                "",
            ]

    bca_fails = [k for k, v in under.items() if k[2] == "bca" and v]
    switch = [k for k in bca_fails if not under[(k[0], k[1], "welch")]]
    lines += [
        "## Pre-registered decision: headline interval for spend",
        "",
        "Rule (section 5): if BCa under-covers and Welch does not, Welch becomes the headline",
        "interval; both are shown either way.",
        "",
        f"- BCa under-covers in: {', '.join(f'{a} ({s})' for s, a, _ in bca_fails) or 'none'}.",
        f"- Of those, Welch is adequate in: {', '.join(f'{a} ({s})' for s, a, _ in switch) or 'none'}.",
        "",
        f"**Decision: {'Welch' if switch else 'BCa'} is the headline interval for spend.**",
        "",
        *INTERPRETATION,
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
