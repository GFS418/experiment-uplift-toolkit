"""Phase 2: variance reduction with pre-treatment covariates (plan, section 7).

1. Validates CUPED and Lin's regression adjustment on 4,000 plasmode
   experiments built from real control customers with their covariates, where
   the truth is known.
2. Applies both to the real experiment and measures the variance each removes,
   next to the unadjusted estimates, which stay the confirmatory ones (Phase 1b).
3. Shows the failure the leakage rule exists to prevent: CUPED on `visit`.

Writes reports/phase2_simulations.parquet and reports/phase2_variance_reduction.md.
`--from-cache` rebuilds the report from the parquet without re-simulating.
"""

import sys

import numpy as np
import pandas as pd

from exptools.adjust import (
    Estimate,
    covariate_matrix,
    cuped,
    cuped_theta,
    difference_in_means,
    lin,
    variance_reduction,
    within_arm_correlation,
)
from exptools.data import ASSIGNMENT, CONTROL, OUTCOMES, REPO_ROOT, load_hillstrom, require_pre_treatment
from exptools.simulate_adjustment import Effect, Pool, run, true_effects

MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
CUPED_COVARIATE = "history"  # plan, section 7
LIN_COVARIATES = ("recency", "history", "mens", "womens", "newbie", "zip_code", "channel")  # plan, section 7
require_pre_treatment([CUPED_COVARIATE, *LIN_COVARIATES])  # the leakage rule, enforced before anything runs

SIZES = {CONTROL: 21_306, MENS: 21_307, WOMENS: 21_387}
N_SIMS = 2_000
KNOWN_EFFECT = Effect(MENS, spend_lift=1.0, visit_rate=0.08)
SCENARIOS = {"complete null": (None, 202621), "known effect": (KNOWN_EFFECT, 202622)}
COMPARISONS = ((MENS, CONTROL), (WOMENS, CONTROL), (MENS, WOMENS))
CACHE = REPO_ROOT / "reports" / "phase2_simulations.parquet"
REPORT = REPO_ROOT / "reports" / "phase2_variance_reduction.md"
SHORT = {CONTROL: "no e-mail", MENS: "men's e-mail", WOMENS: "women's e-mail"}
METHODS = {
    "raw": "Unadjusted",
    "cuped": "CUPED (history)",
    "lin": "Lin (7 covariates)",
    "cuped_on_visit": "CUPED on visit (invalid)",
}

# Written after reading the first run's output (it was empty for that run). The
# simulations are seeded and the analysis is deterministic, so regenerating the
# report reproduces every number cited here.
INTERPRETATION = [
    "## 4. Reading the results",
    "",
    "1. **Both adjustments are safe.** Against a known truth, CUPED and Lin are unbiased (at most",
    "   0.3% of the true effect, inside Monte Carlo error) and their intervals cover 0.941 to 0.960",
    "   across every metric, arm, and scenario. One borderline number: under the null, reported",
    "   spend standard errors run about 4% small (calibration 0.96, about 2 simulation standard",
    "   errors from 1.00). It is identical for the unadjusted estimator (the estimates correlate",
    "   0.9992 across simulations), so it comes from spend's heavy tail in that set of draws, not",
    "   from the adjustment. The known-effect scenario shows 1.00 and coverage stays within error",
    "   of 0.95.",
    "2. **They remove almost no variance from the money metrics.** Prior-year spend correlates",
    "   0.022 with two-week spend within arms, so CUPED can remove about 0.05%; it removes 0.04%.",
    "   Lin, with all seven covariates, removes at most 0.56% for spend and 0.49% for conversion.",
    "   The point estimates move by 0.03 standard errors or less, so they are unchanged.",
    "3. **Why:** CUPED works when the covariate is the same metric over a comparable earlier",
    "   window, from users whose behavior repeats, such as last month's sessions predicting this",
    "   month's. Here the covariate is a year of spend and the outcome is two weeks of spend that",
    "   is 99% zeros. Whether a customer happens to buy in a given fortnight is close to",
    "   unpredictable from their annual total. The theory (variance removed is about rho^2)",
    "   predicted this before any estimator ran.",
    "4. **Visits are the exception, slightly.** Lin removes 2.6% to 3.2% of the variance for visits,",
    "   because recency, channel, and new-customer status predict visiting better than prior spend",
    "   alone (CUPED: 0.4%). The visit estimates shift by up to 0.22 standard errors, in line with",
    "   the roughly 0.17 that removing 3% of the variance should cause: the adjustment is cancelling",
    "   chance imbalance, which is its job.",
    "5. **Effects that vary with prior spend did not change the picture.** The known-effect",
    "   scenario made the chance of buying proportional to prior-year spend, the case where Lin's",
    "   separate slopes per arm should beat CUPED's single slope. For spend, neither removed",
    "   anything: the noise is in who buys and how much, which prior-year spend barely predicts.",
    "   A synthetic unit test confirms that Lin does win when the slopes differ materially.",
    "6. **Uninformative covariates cost a sliver.** A few variance reductions are slightly negative",
    "   (Lin on conversion, women's vs no e-mail: -0.12%). Estimating 27 extra slopes that predict",
    "   nothing adds a little noise. At 64,000 customers it is negligible, but it is a reason not to",
    "   adjust for covariates by habit.",
    "7. **The counterexample is the real lesson.** CUPED on `visit` erases 71% (men's) and 76%",
    "   (women's) of the real spend effect, while reporting its intervals as 3% more precise.",
    "   Against a known truth it is biased by -77% and covers 4% of the time. An adjustment that",
    "   makes an estimate look more precise is not evidence that it is more correct.",
    "8. **What this means downstream.** The power calculations in Phase 3 should use the",
    "   unadjusted variance; adjustment will not rescue the minimum detectable effect here.",
    "",
]


def pool_from(df: pd.DataFrame) -> Pool:
    return Pool(
        covariates=covariate_matrix(df, LIN_COVARIATES),
        history=df[CUPED_COVARIATE].to_numpy(dtype=float),
        outcomes={m: df[m].to_numpy(dtype=float) for m in OUTCOMES},
    )


def effect_fmt(metric: str, x: float, decimals: int = 3) -> str:
    if metric == "spend":
        return f"{'-' if x < 0 else '+'}${abs(x):.{decimals}f}"
    return f"{100 * x:+.{decimals}f} pp"


def ci_fmt(metric: str, est: Estimate) -> str:
    low, high = est.ci()
    plain = (
        (lambda v: f"{'-' if v < 0 else ''}${abs(v):.3f}")
        if metric == "spend"
        else (lambda v: f"{100 * v:.2f}")
    )
    return f"{effect_fmt(metric, est.estimate)} [{plain(low)}, {plain(high)}]"


def sim_summary(s: pd.DataFrame, metric: str, arm: str, method: str, truth: float) -> dict:
    err = s[f"{metric}|{arm}|{method}|err"]
    covered = s[f"{metric}|{arm}|{method}|covered"].astype(float)
    n = len(s)
    return {
        "bias": err.mean(),
        "bias_mc": 1.96 * err.std() / np.sqrt(n),
        "rel_bias": err.mean() / truth if truth else np.nan,
        "coverage": covered.mean(),
        "coverage_mc": 1.96 * np.sqrt(covered.mean() * (1 - covered.mean()) / n),
        "calibration": s[f"{metric}|{arm}|{method}|se"].mean() / err.std(),
        "vr": 1 - err.var() / s[f"{metric}|{arm}|raw|err"].var(),
    }


def simulation_section(sims: pd.DataFrame, pool: Pool) -> list[str]:
    truths = true_effects(pool, KNOWN_EFFECT)
    lines = [
        "## 1. Validation against a known truth",
        "",
        f"{N_SIMS:,} simulated experiments per scenario, built by resampling real control customers",
        "(covariates and outcomes together) into three pseudo-arms of the real sizes. Each is analyzed",
        "with exactly the estimators used on the real data below.",
        "",
        "- **Complete null:** no effect anywhere.",
        f"- **Known effect** in the pseudo-men's arm: +{KNOWN_EFFECT.spend_lift:.0%} mean spend through new",
        "  buyers whose chance of buying is proportional to prior-year spend (so the effect varies with",
        f"  the CUPED covariate), plus {KNOWN_EFFECT.visit_rate:.0%} of remaining non-visitors starting"
        " to visit.",
        f"  True effects: spend {effect_fmt('spend', truths['spend'])}, conversion "
        f"{effect_fmt('conversion', truths['conversion'])}, visit {effect_fmt('visit', truths['visit'])}.",
        "",
        "Columns: bias with its Monte Carlo margin (1.96 standard errors); coverage of the 95% interval;",
        "SE calibration (average reported standard error / actual spread of the estimates, ideally 1.00);",
        "variance removed (actual, from the spread across simulations, relative to unadjusted).",
        "",
    ]
    for name, (effect, _) in SCENARIOS.items():
        s = sims[sims["scenario"] == name]
        lines += [
            f"### Scenario: {name} (men's pseudo-arm vs control)",
            "",
            "| Metric | Method | Bias | Bias / true effect | Coverage | SE calibration | Variance removed |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for metric in ("spend", "conversion", "visit"):
            methods = ["raw", "cuped", "lin"] + (["cuped_on_visit"] if metric == "spend" else [])
            truth = truths[metric] if effect else 0.0
            for method in methods:
                r = sim_summary(s, metric, MENS, method, truth)
                rel = "n/a" if np.isnan(r["rel_bias"]) else f"{r['rel_bias']:+.1%}"
                lines.append(
                    f"| {metric} | {METHODS[method]} | {effect_fmt(metric, r['bias'], 4)} "
                    f"+/-{effect_fmt(metric, r['bias_mc'], 4).lstrip('+')} | {rel} "
                    f"| {r['coverage']:.3f} +/-{r['coverage_mc']:.3f} | {r['calibration']:.3f} "
                    f"| {r['vr']:+.1%} |"
                )
        women = [
            sim_summary(s, m, WOMENS, method, 0.0)
            for m in ("spend", "conversion", "visit")
            for method in ("raw", "cuped", "lin")
        ]
        worst_bias = max(abs(r["bias"]) / (r["bias_mc"] / 1.96) for r in women)
        lines += [
            "",
            f"Women's pseudo-arm (null in both scenarios), all metrics and valid methods: coverage "
            f"{min(r['coverage'] for r in women):.3f} to {max(r['coverage'] for r in women):.3f}, "
            f"largest bias {worst_bias:.1f} Monte Carlo standard errors from zero.",
            "",
        ]
    return lines


def real_section(df: pd.DataFrame) -> list[str]:
    arm = df[ASSIGNMENT].to_numpy()
    x = df[CUPED_COVARIATE].to_numpy(dtype=float)
    covariates = covariate_matrix(df, LIN_COVARIATES)
    lines = [
        "## 2. The real experiment: what the adjustment buys",
        "",
        "Same point estimates as Phase 1b in the unadjusted column; intervals here are normal-theory for",
        "all three methods so their widths compare directly (the confirmatory BCa intervals are in",
        "Phase 1b). Shift = adjusted minus unadjusted estimate, in unadjusted standard errors.",
        "",
    ]
    gains = []
    for metric in ("spend", "conversion", "visit"):
        y = df[metric].to_numpy(dtype=float)
        fit = lin(y, covariates, arm, CONTROL)
        rho = within_arm_correlation(y, x, arm)
        lines += [
            f"### {metric.capitalize()}"
            + (" per customer (primary metric)" if metric == "spend" else " rate (percentage points)"),
            "",
            f"Within-arm correlation with prior-year spend: {rho:.3f}, so CUPED can remove about "
            f"rho^2 = {rho**2:.2%} of the variance. Theta = {cuped_theta(y, x, arm):.5f}.",
            "",
            "| Comparison | Unadjusted [95% CI] | CUPED [95% CI] | Lin [95% CI] | Variance removed: CUPED "
            "| Variance removed: Lin | Shift: CUPED / Lin |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for t, c in COMPARISONS:
            raw = difference_in_means(y, arm, t, c)
            cu = cuped(y, x, arm, t, c)
            ln = fit.vs_control(t) if c == CONTROL else fit.contrast(t, c)
            vr_cu, vr_ln = variance_reduction(cu, raw), variance_reduction(ln, raw)
            if c == CONTROL:
                gains.append((metric, t, vr_ln))
            lines.append(
                f"| {SHORT[t]} vs {SHORT[c]} | {ci_fmt(metric, raw)} | {ci_fmt(metric, cu)} "
                f"| {ci_fmt(metric, ln)} | {vr_cu:.2%} | {vr_ln:.2%} "
                f"| {(cu.estimate - raw.estimate) / raw.se:+.2f} "
                f"/ {(ln.estimate - raw.estimate) / raw.se:+.2f} |"
            )
        lines.append("")
    lines += [
        "### In sample-size terms",
        "",
        "Removing a share V of the variance is worth 1 / (1 - V) times the customers, and shrinks the",
        "minimum detectable effect by a factor sqrt(1 - V).",
        "",
        "| Metric | Comparison | Variance removed (Lin) | Equivalent extra customers | MDE shrinks by |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, t, vr in gains:
        lines.append(
            f"| {metric} | {SHORT[t]} vs no e-mail | {vr:.2%} | {1 / (1 - vr) - 1:+.1%} "
            f"| {1 - np.sqrt(1 - vr):.2%} |"
        )
    spend, visit = df["spend"].to_numpy(dtype=float), df["visit"].to_numpy(dtype=float)
    lines += [
        "",
        "## 3. Counterexample: adjusting for a post-treatment variable",
        "",
        "`visit` is measured after the e-mail and the e-mail changes it. Using it as the CUPED covariate",
        "removes the part of the spend effect that runs through extra visits. This analysis is invalid",
        "and shown only to demonstrate the leakage rule (plan, section 7).",
        "",
        '| E-mail vs no e-mail | Unadjusted spend effect | "CUPED on visit" | Effect removed '
        "| Variance removed |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in (MENS, WOMENS):
        raw, leak = difference_in_means(spend, arm, t, CONTROL), cuped(spend, visit, arm, t, CONTROL)
        lines.append(
            f"| {SHORT[t]} | {effect_fmt('spend', raw.estimate)} | {effect_fmt('spend', leak.estimate)} "
            f"| {1 - leak.estimate / raw.estimate:.0%} | {variance_reduction(leak, raw):.0%} |"
        )
    lines.append("")
    return lines


def main(from_cache: bool) -> None:
    df = load_hillstrom(load_outcomes=True)
    pool = pool_from(df[df[ASSIGNMENT] == CONTROL])
    if from_cache:
        sims = pd.read_parquet(CACHE)
    else:
        frames = [
            run(pool, SIZES, CONTROL, effect, N_SIMS, seed).assign(scenario=name)
            for name, (effect, seed) in SCENARIOS.items()
        ]
        sims = pd.concat(frames, ignore_index=True)
        sims.to_parquet(CACHE)
    lines = [
        "# Phase 2: variance reduction with pre-treatment covariates",
        "",
        "Generated by `scripts/adjust_variance.py` (plan, section 7). The unadjusted Phase 1b estimates",
        "remain the confirmatory results; this phase measures how much noise pre-treatment covariates",
        "remove, and checks first that removing it does not bias anything.",
        "",
        *simulation_section(sims, pool),
        *real_section(df),
        *INTERPRETATION,
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
