"""Phase 1b: the pre-registered analysis of the Hillstrom experiment.

The first code in the repository that reads treatment-arm outcomes. Every
choice is fixed by the analysis plan (prereg/analysis_plan.md, commit b120b77)
and the Phase 1a validation (commit d10be07), which kept BCa as the headline
interval. Writes reports/phase1b_analysis.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exptools.bootstrap import (
    Arm,
    Replicates,
    acceleration,
    bca_interval,
    bootstrap_replicates,
    ratio_influence,
)
from exptools.data import ARMS, ASSIGNMENT, CONTROL, OUTCOMES, REPO_ROOT, load_hillstrom
from exptools.inference import Comparison, compare
from exptools.multiplicity import Dunnett, benjamini_hochberg, dunnett_maxt, dunnett_parametric

SEED = 2026  # plan, section 5
N_RESAMPLES = 10_000  # plan, section 5
ALPHA = 0.05  # family-wise for the primary family, false discovery rate for the exploratory one
WINSOR_QUANTILE = 0.999  # plan, section 3: sensitivity analysis only
MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
TREATMENTS = (MENS, WOMENS)
EXPLORATORY = (  # plan, section 4
    ("visit", MENS, CONTROL),
    ("visit", WOMENS, CONTROL),
    ("visit", MENS, WOMENS),
    ("conversion", MENS, CONTROL),
    ("conversion", WOMENS, CONTROL),
    ("conversion", MENS, WOMENS),
    ("spend", MENS, WOMENS),
)
REPORT = REPO_ROOT / "reports" / "phase1b_analysis.md"
SHORT = {CONTROL: "no e-mail", MENS: "men's e-mail", WOMENS: "women's e-mail"}

# Written after reading the first run's output (it was empty for that run). The
# analysis is deterministic (seed 2026), so regenerating reproduces these numbers.
INTERPRETATION = [
    "## 7. Reading the results",
    "",
    "1. **Both e-mails increase spend, and the men's e-mail roughly doubles it.** The men's",
    "   e-mail adds $0.77 per customer (+118%) and the women's $0.42 (+65%). Both pass the",
    "   pre-registered Dunnett correction (adjusted p < 0.0001 and 0.0023). That is about $770",
    "   and $424 of extra revenue per 1,000 e-mails sent.",
    "2. **The uncertainty is large relative to the effects.** The simultaneous intervals run",
    "   from $0.45 to $1.09 per customer (men's) and $0.13 to $0.72 (women's). Relative lifts",
    "   are less precise still (women's: +22% to +125%), because the control mean they divide",
    '   by is itself small and noisy. Quote the range, not just "+65%".',
    "3. **The parametric cross-check behaved as Phase 1a predicted.** Spend standard deviations",
    "   differ by arm ($11.59, $17.75, $15.12). SciPy's single pooled variance overstates the",
    "   women's comparison's standard error by 12%, so its p-value is about 3x larger (0.0068",
    "   vs 0.0023) and its interval wider. For the men's comparison the pooled value happens",
    "   to be right (within 0.3%), and the two versions agree to within a cent. The decision",
    "   is the same either way.",
    "4. **The effects are not driven by a few large orders.** Capping the top 0.1% of spend at",
    "   $243.66 (64 customers, 53 of them e-mailed) shrinks the effects by 14% (men's) and 10%",
    "   (women's), and both stay significant. Some shrinkage is expected even without outliers:",
    "   the e-mailed arms have more large orders because they have more buyers.",
    "5. **More buyers, not bigger baskets (descriptive).** Conversion rose 119% (men's) and 54%",
    "   (women's), while mean spend per buyer stayed between $114 and $122 in every arm. Buyers",
    "   in different arms are different people, so this is suggestive, not a causal",
    "   decomposition. Visits moved the most: +7.7 and +4.5 percentage points.",
    "6. **Men's beats women's on spend only as a weak lead.** All seven exploratory hypotheses",
    "   pass Benjamini-Hochberg, but the head-to-head spend comparison is fragile: +$0.35 per",
    "   customer, BCa interval $0.035 to $0.663, p = 0.031. It passes only because the other six",
    "   are overwhelmingly significant, which gives the last-ranked test BH's full 0.05",
    "   threshold; Bonferroni would put it at 0.21. The firmer evidence that the men's e-mail is",
    "   the better single e-mail comes from visits (+3.1 points) and conversion (+0.37 points).",
    "   Who should get which e-mail is Phase 4's question.",
    "7. **The power caveat did not bite this time.** Conversion was flagged as rare and",
    "   low-powered, and spend needed about a +50% lift for reliable detection. The real effects",
    "   (+54% to +119%) were large enough. The caveat still stands for smaller e-mail effects.",
    "8. **Revenue says send; profit is unmeasured.** A revenue break-even of $0.42 to $0.77 per",
    "   e-mail is far above the cost of sending one. But margins are unknown, and so are the",
    "   costs this data cannot see: unsubscribes and list fatigue, the guardrails the plan",
    "   notes are missing.",
    "9. **Honesty about blinding.** The ranking (men's > women's > no e-mail) is this dataset's",
    "   widely reported headline, which is why the plan says the analysis could not be truly",
    "   blind. What the pre-registration protected were the choices: metric, correction,",
    "   interval method, and sensitivity analysis were all fixed before these numbers existed in",
    "   this repository.",
    "",
]


@dataclass(frozen=True)
class Row:
    comparison: Comparison
    lift: float  # mean(treatment) / mean(control) - 1
    lift_ci: tuple[float, float]


@dataclass(frozen=True)
class Results:
    primary: dict[str, Row]
    maxt: Dunnett
    parametric: Dunnett
    exploratory: list[Row]
    bh_adjusted: np.ndarray
    bh_reject: np.ndarray


def build_arms(df: pd.DataFrame, spend_cap: float | None = None) -> dict[str, Arm]:
    if spend_cap is not None:
        df = df.assign(spend=df["spend"].clip(upper=spend_cap))
    return {name: Arm.from_frame(df[df[ASSIGNMENT] == name], OUTCOMES) for name in ARMS}


def relative_lift(
    treat: Arm, control: Arm, metric: str, treat_reps: Replicates, control_reps: Replicates
) -> tuple[float, tuple[float, float]]:
    """mean(treat) / mean(control) - 1, with a BCa interval on the ratio."""
    ratio = treat.mean(metric) / control.mean(metric)
    accel = acceleration(ratio_influence(treat, control, metric))
    low, high = bca_interval(ratio, treat_reps.mean / control_reps.mean, accel)
    return ratio - 1, (low - 1, high - 1)


def analyze(arms: dict[str, Arm], rng: np.random.Generator, *, exploratory: bool = True) -> Results:
    reps = bootstrap_replicates(arms, OUTCOMES, N_RESAMPLES, rng)

    def row(metric: str, t: str, c: str) -> Row:
        tr, cr = reps[t][metric], reps[c][metric]
        cmp = compare(arms[t], arms[c], metric, tr, cr, names=(t, c))
        return Row(cmp, *relative_lift(arms[t], arms[c], metric, tr, cr))

    treatments = {t: arms[t] for t in TREATMENTS}
    spend_reps = {t: reps[t]["spend"] for t in TREATMENTS}
    maxt = dunnett_maxt(treatments, arms[CONTROL], "spend", spend_reps, reps[CONTROL]["spend"])
    parametric = dunnett_parametric(treatments, arms[CONTROL], "spend", rng)
    primary = {t: row("spend", t, CONTROL) for t in TREATMENTS}
    rows = [row(*h) for h in EXPLORATORY] if exploratory else []
    adjusted, reject = (
        benjamini_hochberg([r.comparison.welch.p_value for r in rows], ALPHA) if rows else ([], [])
    )
    return Results(primary, maxt, parametric, rows, np.asarray(adjusted), np.asarray(reject))


# ---- formatting -------------------------------------------------------------


def money(x: float) -> str:
    return f"-${-x:,.3f}" if x < 0 else f"${x:,.3f}"


def money0(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def points(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def lift(x: float) -> str:
    return f"{x:+.0%}"


def interval(ci: tuple[float, float], fmt: Callable[[float], str]) -> str:
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def p_analytic(p: float) -> str:
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def p_bootstrap(p: float) -> str:
    # Resolution is 1 / (B + 1); the floor means no resample was as extreme as the data.
    return "< 0.0001" if p <= 1 / (N_RESAMPLES + 1) else f"{p:.4f}"


def per_thousand(metric: str, x: float) -> str:
    if metric == "spend":
        return money0(1000 * x)
    noun = "visitors" if metric == "visit" else "buyers"
    return f"{1000 * x:+.1f} {noun}"


def unit(metric: str) -> Callable[[float], str]:
    return money if metric == "spend" else points


# ---- report -----------------------------------------------------------------


def render(df: pd.DataFrame, arms: dict[str, Arm], res: Results, cap: float, winsor: Results) -> list[str]:
    lines = [
        "# Phase 1b: pre-registered analysis",
        "",
        "Generated by `scripts/analyze_experiment.py`, the first code in this repository to read",
        "treatment-arm outcomes. Plan: commit `b120b77`. Method validation: commit `d10be07` (BCa kept",
        f"as the headline interval). Bootstrap: B = {N_RESAMPLES:,} resamples within arms, seed {SEED}.",
        "",
        "## 1. Outcome data checks, all arms",
        "",
        "| Arm | Customers | Converters without a visit | Rows where spend > 0 and conversion disagree |",
        "|---|---:|---:|---:|",
    ]
    for name in ARMS:
        d = df[df[ASSIGNMENT] == name]
        lines.append(
            f"| {name} | {len(d):,} | {int(((d.conversion == 1) & (d.visit == 0)).sum())} "
            f"| {int(((d.spend > 0) != (d.conversion == 1)).sum())} |"
        )
    lines += [
        "",
        "## 2. Arm summaries (descriptive)",
        "",
        "| Arm | Visit rate | Conversion rate | Buyers | Mean spend per customer | SD of spend "
        "| Mean spend per buyer* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        a = arms[name]
        spend, counts = a.column("spend"), a.counts
        buyers = int(counts[spend > 0].sum())
        per_buyer = counts[spend > 0] @ spend[spend > 0] / buyers
        lines.append(
            f"| {name} | {a.mean('visit'):.2%} | {a.mean('conversion'):.2%} | {buyers} "
            f"| {money(a.mean('spend'))} | ${np.sqrt(a.var('spend')):.2f} | ${per_buyer:.2f} |"
        )
    lines += [
        "",
        "*Descriptive only. It conditions on a post-treatment outcome (buying), so differences in it",
        "are not causal effects (plan, section 3).",
        "",
        "## 3. Primary result: spend per customer (confirmatory)",
        "",
        "Family: each e-mail vs no e-mail, family-wise error 0.05, Dunnett via bootstrap max-T.",
        "The marginal BCa interval describes one comparison on its own; the simultaneous interval",
        "is the one consistent with the family-wise decision, so it is wider by design.",
        "",
        "| E-mail vs no e-mail | Effect per customer | Per 1,000 customers | Relative lift "
        "| 95% BCa (marginal) | 95% simultaneous (max-T) | Dunnett-adjusted p |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for i, t in enumerate(res.maxt.arms):
        r = res.primary[t]
        lines.append(
            f"| {SHORT[t]} | {money(r.comparison.estimate)} | {per_thousand('spend', r.comparison.estimate)} "
            f"| {lift(r.lift)} {interval(r.lift_ci, lift)} | {interval(r.comparison.bca, money)} "
            f"| {interval((res.maxt.ci_low[i], res.maxt.ci_high[i]), money)} "
            f"| {p_bootstrap(res.maxt.p_adjusted[i])} |"
        )
    lines += [
        "",
        "Cross-checks (same comparisons):",
        "",
        "| E-mail vs no e-mail | Welch 95% (marginal) | Welch p (unadjusted) | Parametric Dunnett 95% "
        "| Parametric Dunnett p |",
        "|---|---|---:|---|---:|",
    ]
    for i, t in enumerate(res.parametric.arms):
        w = res.primary[t].comparison.welch
        lines.append(
            f"| {SHORT[t]} | {interval(w.ci, money)} | {p_analytic(w.p_value)} "
            f"| {interval((res.parametric.ci_low[i], res.parametric.ci_high[i]), money)} "
            f"| {p_analytic(res.parametric.p_adjusted[i])} |"
        )
    lines += [
        "",
        f"Critical values: max-T {res.maxt.critical_value:.3f}, "
        f"parametric {res.parametric.critical_value:.3f}.",
        f"A bootstrap p-value of < 0.0001 means none of the {N_RESAMPLES:,} resamples was as extreme as",
        "the data; it is the method's resolution limit.",
        "",
        "**Pre-registered decision** (adjusted p < 0.05 and a positive estimate):",
        "",
    ]
    for i, t in enumerate(res.maxt.arms):
        declared = res.maxt.p_adjusted[i] < ALPHA and res.maxt.estimate[i] > 0
        verdict = "increases" if declared else "is not shown to increase"
        lines.append(f"- The {SHORT[t]} **{verdict}** spend per customer.")
    lines += [
        "",
        "## 4. Exploratory family (Benjamini-Hochberg, q = 0.05)",
        "",
        "Leads for future tests, not findings. Visit and conversion effects are in percentage points.",
        "",
        "| Hypothesis | Effect | Per 1,000 customers | Relative lift | 95% BCa | Welch p "
        "| BH-adjusted p | Lead at q = 0.05 |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for r, adj, rej in zip(res.exploratory, res.bh_adjusted, res.bh_reject, strict=True):
        c = r.comparison
        fmt = unit(c.metric)
        lines.append(
            f"| {c.metric}: {SHORT[c.treatment]} vs {SHORT[c.control]} | {fmt(c.estimate)} "
            f"| {per_thousand(c.metric, c.estimate)} | {lift(r.lift)} {interval(r.lift_ci, lift)} "
            f"| {interval(c.bca, fmt)} | {p_analytic(c.welch.p_value)} | {p_analytic(adj)} "
            f"| {'yes' if rej else 'no'} |"
        )
    capped = {name: int((df.loc[df[ASSIGNMENT] == name, "spend"] > cap).sum()) for name in ARMS}
    lines += [
        "",
        "## 5. Sensitivity: spend winsorized at the pooled 99.9th percentile (not confirmatory)",
        "",
        f"Cap: ${cap:.2f}, computed over all {len(df):,} customers pooled so it cannot depend on",
        "assignment. Customers above the cap: " + ", ".join(f"{SHORT[n]} {capped[n]}" for n in ARMS) + ".",
        "",
        "| E-mail vs no e-mail | Effect per customer | 95% BCa (marginal) | 95% simultaneous (max-T) "
        "| Dunnett-adjusted p | Change from raw estimate |",
        "|---|---:|---|---|---:|---:|",
    ]
    for i, t in enumerate(winsor.maxt.arms):
        r = winsor.primary[t]
        raw = res.primary[t].comparison.estimate
        lines.append(
            f"| {SHORT[t]} | {money(r.comparison.estimate)} | {interval(r.comparison.bca, money)} "
            f"| {interval((winsor.maxt.ci_low[i], winsor.maxt.ci_high[i]), money)} "
            f"| {p_bootstrap(winsor.maxt.p_adjusted[i])} | {(r.comparison.estimate - raw) / raw:+.1%} |"
        )
    lines += [
        "",
        "## 6. Practical significance",
        "",
        "Incremental revenue per 1,000 customers e-mailed, with the marginal BCa interval. The effect",
        "per customer is also the revenue break-even: the most one e-mail could cost before losing",
        "money. It is an upper bound, because no margin data exists and profit is revenue times margin.",
        "",
        "| E-mail | Incremental revenue per 1,000 e-mails | 95% BCa | Revenue break-even per e-mail |",
        "|---|---:|---|---:|",
    ]
    for t in TREATMENTS:
        c = res.primary[t].comparison
        lines.append(
            f"| {SHORT[t]} | {money0(1000 * c.estimate)} "
            f"| {interval((1000 * c.bca[0], 1000 * c.bca[1]), money0)} | {money(c.estimate)} |"
        )
    lines += ["", *INTERPRETATION]
    return lines


def main() -> None:
    df = load_hillstrom(load_outcomes=True)
    arms = build_arms(df)
    results = analyze(arms, np.random.default_rng(SEED))
    cap = float(np.quantile(df["spend"], WINSOR_QUANTILE))
    winsor = analyze(build_arms(df, spend_cap=cap), np.random.default_rng(SEED), exploratory=False)
    lines = render(df, arms, results, cap, winsor)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
