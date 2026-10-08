"""Phase 4b: score the frozen models on the test half, once, and run the pre-specified moderators.

The only code in the repository that reads the test half (Phase 4 design note,
sections 5, 6 and 8). Before anything is scored, every frozen model is refit
on the training half and must reproduce its Phase 4a fingerprint exactly;
otherwise the script stops. Writes reports/phase4b_test_evaluation.md and
reports/phase4b_results.json; `--from-cache` rebuilds the report from the JSON.
"""

import json
import sys

import numpy as np
import pandas as pd

from exptools.adjust import covariate_matrix, difference_in_means
from exptools.data import (
    ARMS,
    ASSIGNMENT,
    CONTROL,
    REPO_ROOT,
    load_half,
    load_hillstrom,
    require_pre_treatment,
)
from exptools.multiplicity import benjamini_hochberg
from exptools.uplift import (
    FEATURES,
    CausalForest,
    DRLearner,
    TLearner,
    aipw_scores,
    budget_policy,
    calibration_test,
    greedy_policy,
    moderator_tests,
    policy_bootstrap,
    policy_value,
    qini_area,
    uplift_curve,
)

MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
TREATMENTS = (MENS, WOMENS)
OUTCOME_NAMES = ("spend", "visit")
LEARNERS = ("DR-learner", "T-learner", "causal forest")
FOLD_SEED = 202640
FRACTIONS = np.arange(1, 101) / 100  # note, section 5: 1% steps
TIE_SEED, BOOT_SEED = 202643, 202644  # note, section 5
N_UPLIFT_BOOT, N_POLICY_BOOT = 2_000, 10_000
BUDGET = 10_000
BUDGET_SHARE = BUDGET / 64_000
SHORT = {MENS: "men's e-mail", WOMENS: "women's e-mail", CONTROL: "no e-mail"}
CONFIG = REPO_ROOT / "reports" / "phase4a_frozen_config.json"
RESULTS = REPO_ROOT / "reports" / "phase4b_results.json"
REPORT = REPO_ROOT / "reports" / "phase4b_test_evaluation.md"

require_pre_treatment(FEATURES)

# Written after reading the first run's output; empty until then.
INTERPRETATION: list[str] = []


def fingerprint(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.round(values, 10).tobytes()).hexdigest()[:16]


def refit_and_verify(train: pd.DataFrame, frozen: dict) -> dict:
    """Refit every frozen model on the training half and check its Phase 4a fingerprint."""
    X, arm = covariate_matrix(train, FEATURES), train[ASSIGNMENT].to_numpy(dtype=object)
    models = {}
    for outcome in OUTCOME_NAMES:
        cfg, y = frozen[outcome], train[outcome].to_numpy(dtype=float)
        models[outcome] = {
            "T-learner": TLearner(cfg["outcome_params"], CONTROL).fit(X, y, arm),
            "DR-learner": DRLearner(cfg["outcome_params"], cfg["stage2_params"], CONTROL, FOLD_SEED).fit(
                X, y, arm
            ),
            "causal forest": CausalForest(CONTROL, TREATMENTS).fit(X, y, arm),
        }
        for learner in LEARNERS:
            for t in TREATMENTS:
                got = fingerprint(models[outcome][learner].effect(X, t))
                if got != cfg["fingerprints"][f"{learner}|{t}"]:
                    raise RuntimeError(
                        f"{learner} ({outcome}, {t}) does not reproduce its frozen fingerprint"
                    )
    return models


def bootstrap_area(y, arm, t, pred, rng) -> np.ndarray:
    """Qini-type area under resampling of test customers within arm, model held fixed."""
    keep = (arm == t) | (arm == CONTROL)
    y, arm, pred = y[keep], arm[keep], pred[keep]
    groups = [np.flatnonzero(arm == a) for a in (t, CONTROL)]
    areas = np.empty(N_UPLIFT_BOOT)
    for b in range(N_UPLIFT_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groups])
        areas[b] = qini_area(uplift_curve(y[idx], arm[idx], t, CONTROL, pred[idx], FRACTIONS, rng), FRACTIONS)
    return areas


def evaluate(test: pd.DataFrame, models: dict) -> dict:
    X, arm = covariate_matrix(test, FEATURES), test[ASSIGNMENT].to_numpy(dtype=object)
    out: dict = {"n_test": len(test), "heterogeneity": {}, "policies": {}, "budget": {}}
    effects = {}
    area_rng = np.random.default_rng(BOOT_SEED)
    for outcome in OUTCOME_NAMES:
        y = test[outcome].to_numpy(dtype=float)
        m_test = models[outcome]["T-learner"].outcomes(X)  # outcome models from the whole training half
        for learner in LEARNERS:
            for t in TREATMENTS:
                pred = models[outcome][learner].effect(X, t)
                effects[(outcome, learner, t)] = pred
                cal = calibration_test(aipw_scores(y, arm, m_test, t, CONTROL), pred)
                curve = uplift_curve(y, arm, t, CONTROL, pred, FRACTIONS, np.random.default_rng(TIE_SEED))
                areas = bootstrap_area(y, arm, t, pred, area_rng)
                out["heterogeneity"][f"{outcome}|{learner}|{t}"] = {
                    "ate": cal.ate,
                    "slope": cal.slope,
                    "slope_se": cal.slope_se,
                    "p_value": cal.p_value,
                    "applicable": cal.applicable,
                    "prediction_sd": float(pred.std()),
                    "area": qini_area(curve, FRACTIONS),
                    "area_low": float(np.percentile(areas, 2.5)),
                    "area_high": float(np.percentile(areas, 97.5)),
                    "curve_deciles": curve[9::10].tolist(),
                }

    # Policies on spend (note, section 5).
    y = test["spend"].to_numpy(dtype=float)
    policies = {
        **{
            learner: greedy_policy({t: effects[("spend", learner, t)] for t in TREATMENTS}, CONTROL)
            for learner in LEARNERS
        },
        **{f"everyone: {SHORT[a]}": np.full(len(y), a, dtype=object) for a in ARMS},
        **{
            f"budget ({learner})": budget_policy(
                {t: effects[("spend", learner, t)] for t in TREATMENTS},
                CONTROL,
                BUDGET_SHARE,
                np.random.default_rng(TIE_SEED),
            )
            for learner in LEARNERS
        },
    }
    boot = policy_bootstrap(y, arm, policies, N_POLICY_BOOT, np.random.default_rng(BOOT_SEED))
    blankets = [f"everyone: {SHORT[a]}" for a in ARMS]
    values = {name: policy_value(y, arm, p) for name, p in policies.items()}
    values["random assignment"] = float(np.mean([values[b] for b in blankets]))
    boot["random assignment"] = np.mean([boot[b] for b in blankets], axis=0)
    best_single = f"everyone: {SHORT[MENS]}"
    for name, value in values.items():
        diff = boot[name] - boot[best_single]
        out["policies"][name] = {
            "value": value,
            "low": float(np.percentile(boot[name], 2.5)),
            "high": float(np.percentile(boot[name], 97.5)),
            "vs_mens": value - values[best_single],
            "vs_mens_low": float(np.percentile(diff, 2.5)),
            "vs_mens_high": float(np.percentile(diff, 97.5)),
            "shares": {a: float(np.mean(policies[name] == a)) for a in ARMS} if name in policies else None,
        }

    # The 10,000-e-mail question: incremental revenue of 10,000 e-mails, model-targeted vs random men's.
    # DR-learner primary; T-learner and causal forest exploratory (plan, section 12, 2026-10-08).
    none = f"everyone: {SHORT[CONTROL]}"
    random_mens = BUDGET_SHARE * (boot[best_single] - boot[none]) * 64_000
    point_random = BUDGET_SHARE * (values[best_single] - values[none]) * 64_000
    for learner in LEARNERS:
        name = f"budget ({learner})"
        targeted = (boot[name] - boot[none]) * 64_000
        point_targeted = (values[name] - values[none]) * 64_000
        out["budget"][learner] = {
            "targeted": point_targeted,
            "targeted_low": float(np.percentile(targeted, 2.5)),
            "targeted_high": float(np.percentile(targeted, 97.5)),
            "random_mens": point_random,
            "random_mens_low": float(np.percentile(random_mens, 2.5)),
            "random_mens_high": float(np.percentile(random_mens, 97.5)),
            "gain": point_targeted - point_random,
            "gain_low": float(np.percentile(targeted - random_mens, 2.5)),
            "gain_high": float(np.percentile(targeted - random_mens, 97.5)),
        }
    return out


def budget_profile(models: dict) -> dict:
    """Who each frozen learner would e-mail with a 10,000 budget, scored over all 64,000 customers."""
    df = load_hillstrom()  # features only
    X = covariate_matrix(df, FEATURES)

    def describe(rows: np.ndarray) -> dict:
        d = df[rows]
        return {
            "mean history": float(d["history"].mean()),
            "mean recency (months)": float(d["recency"].mean()),
            "bought men's": float(d["mens"].mean()),
            "bought women's": float(d["womens"].mean()),
            "newbie": float(d["newbie"].mean()),
            **{f"channel {c}": float((d["channel"] == c).mean()) for c in ("Phone", "Web", "Multichannel")},
        }

    profile = {"all 64,000": describe(np.ones(len(df), dtype=bool))}
    emails = {}
    for learner in LEARNERS:
        model = models["spend"][learner]
        policy = budget_policy(
            {t: model.effect(X, t) for t in TREATMENTS},
            CONTROL,
            BUDGET_SHARE,
            np.random.default_rng(TIE_SEED),
        )
        chosen = policy != CONTROL
        profile[f"{learner}'s 10,000"] = describe(chosen)
        emails[learner] = {SHORT[t]: float(np.mean(policy[chosen] == t)) for t in TREATMENTS}
    return {"profile": profile, "emails": emails}


def moderator_section_data() -> dict:
    """Pre-specified moderator tests on all 64,000 customers (note, section 6)."""
    df = load_hillstrom(load_outcomes=True)
    arm = df[ASSIGNMENT].to_numpy(dtype=object)
    only_mens = (df["mens"] == 1) & (df["womens"] == 0)
    only_womens = (df["mens"] == 0) & (df["womens"] == 1)
    moderators = {
        "purchase category": np.column_stack([only_mens, only_womens]),  # reference: bought both
        "recency": df["recency"].to_numpy(),
        "newbie": df["newbie"].to_numpy(),
        "channel": np.column_stack(
            [df["channel"] == "Phone", df["channel"] == "Web"]
        ),  # reference: multichannel
    }
    categories = {"men's only": only_mens, "women's only": only_womens, "both": ~only_mens & ~only_womens}
    out = {}
    for outcome in OUTCOME_NAMES:
        y = df[outcome].to_numpy(dtype=float)
        tests = []
        for name, z in moderators.items():
            for t, res in moderator_tests(y, arm, CONTROL, TREATMENTS, z).items():
                tests.append(
                    {"moderator": name, "e-mail": t, "chi2": res.chi2, "df": res.df, "p": res.p_value}
                )
        adjusted, reject = benjamini_hochberg([r["p"] for r in tests], 0.05)
        for r, a, rej in zip(tests, adjusted, reject, strict=True):
            r["bh_p"], r["lead"] = float(a), bool(rej)
        subgroups = []
        for cat, rows in categories.items():
            rows = rows.to_numpy()
            for t in TREATMENTS:
                est = difference_in_means(y[rows], arm[rows], t, CONTROL)
                low, high = est.ci()
                subgroups.append(
                    {
                        "category": cat,
                        "e-mail": t,
                        "customers": int(rows.sum()),
                        "effect": est.estimate,
                        "low": low,
                        "high": high,
                    }
                )
        out[outcome] = {"tests": tests, "subgroups": subgroups}
    return out


def money(x: float, decimals: int = 3) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.{decimals}f}"


def span(d: dict, key: str) -> str:
    return f"{money(d[key], 0)} [{money(d[key + '_low'], 0)}, {money(d[key + '_high'], 0)}]"


def unit(outcome: str, x: float) -> str:
    return money(x, 4) if outcome == "spend" else f"{100 * x:+.2f} pp"


def render(results: dict) -> list[str]:
    het, pol, bud = results["heterogeneity"], results["policies"], results["budget"]
    lines = [
        "# Phase 4b: the frozen uplift models on the test half",
        "",
        "Generated by `scripts/uplift_phase4b.py`, the only code that reads the test half, under the Phase 4",
        "design note. Every frozen model reproduced its Phase 4a fingerprint before scoring. Test half:",
        f"{results['n_test']:,} customers.",
        "",
        "## 1. Is there heterogeneity the models can see? (calibration test, note section 5)",
        "",
        "Slope 1 means the predicted effects are calibrated; slope 0 means they capture no real",
        "heterogeneity. One-sided p for slope > 0. Primary: DR-learner on spend, Holm across the two",
        "e-mails.",
        "",
        "| Outcome | Learner | Effect of | Average effect | Slope (SE) | One-sided p | SD of predictions |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, r in het.items():
        outcome, learner, t = key.split("|")
        slope = f"{r['slope']:.2f} ({r['slope_se']:.2f})" if r["applicable"] else "n/a (constant)"
        p = f"{r['p_value']:.3f}" if r["applicable"] else "n/a"
        lines.append(
            f"| {outcome} | {learner} | {SHORT[t]} | {unit(outcome, r['ate'])} | {slope} | {p} "
            f"| {unit(outcome, r['prediction_sd'])} |"
        )
    primary = [het[f"spend|DR-learner|{t}"] for t in TREATMENTS]
    ps = [r["p_value"] if r["applicable"] else 1.0 for r in primary]
    holm = sorted(ps)
    rejected = holm[0] <= 0.025
    lines += [
        "",
        "**Primary test (DR-learner, spend, Holm):** "
        + ("heterogeneity detected." if rejected else "no heterogeneity detected.")
        + " p-values: "
        + ", ".join(
            f"{SHORT[t]} {('n/a' if not r['applicable'] else f'{r["p_value"]:.3f}')}"
            for t, r in zip(TREATMENTS, primary, strict=True)
        )
        + ".",
        "",
        "## 2. Uplift curves: does ranking by predicted effect beat random targeting?",
        "",
        "Qini-type area: the average gap between the uplift curve and the random-targeting line, per",
        "customer, with a 95% bootstrap interval (2,000 resamples of test customers, models fixed).",
        "",
        "| Outcome | Learner | Effect of | Area | 95% interval |",
        "|---|---|---|---:|---|",
    ]
    for key, r in het.items():
        outcome, learner, t = key.split("|")
        lines.append(
            f"| {outcome} | {learner} | {SHORT[t]} | {unit(outcome, r['area'])} "
            f"| [{unit(outcome, r['area_low'])}, {unit(outcome, r['area_high'])}] |"
        )
    lines += [
        "",
        "## 3. Policy value on the test half (spend)",
        "",
        "Inverse-propensity values with the arm shares; 95% paired bootstrap intervals (10,000 resamples).",
        "",
        "| Policy | Spend per customer | 95% interval | vs everyone getting the men's e-mail, per 1,000 "
        "| 95% interval |",
        "|---|---:|---|---:|---|",
    ]
    for name, r in pol.items():
        lines.append(
            f"| {name} | {money(r['value'])} | [{money(r['low'])}, {money(r['high'])}] "
            f"| {money(1000 * r['vs_mens'], 0)} "
            f"| [{money(1000 * r['vs_mens_low'], 0)}, {money(1000 * r['vs_mens_high'], 0)}] |"
        )
    lines += ["", "Who each learned policy e-mails (share of test customers):", ""]
    lines += ["| Policy | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for name, r in pol.items():
        if r["shares"] and name in LEARNERS:
            lines.append(f"| {name} | " + " | ".join(f"{r['shares'][a]:.0%}" for a in ARMS) + " |")
    lines += [
        "",
        "## 4. The 10,000-e-mail question",
        "",
        "Incremental revenue from 10,000 e-mails, estimated on the test half and scaled to 64,000",
        "customers, with 95% paired bootstrap intervals. The DR-learner is primary; the T-learner and",
        "the causal forest are exploratory (plan, section 12, 2026-10-08).",
        "",
        "| Targeted by | Revenue from 10,000 targeted e-mails | Men's e-mail to a random 10,000 "
        "| Value of targeting |",
        "|---|---:|---:|---:|",
        *(
            f"| {learner} | {span(bud[learner], 'targeted')} | {span(bud[learner], 'random_mens')} "
            f"| {span(bud[learner], 'gain')} |"
            for learner in LEARNERS
        ),
        "",
        "Who each learner would e-mail (all 64,000 customers scored with the frozen models):",
        "",
        "| | " + " | ".join(results["profile"]["profile"]) + " |",
        "|---|" + "---:|" * len(results["profile"]["profile"]),
        *(
            f"| {row} | "
            + " | ".join(
                (f"${v:,.0f}" if row == "mean history" else f"{v:.1f}" if "recency" in row else f"{v:.0%}")
                for v in (group[row] for group in results["profile"]["profile"].values())
            )
            + " |"
            for row in next(iter(results["profile"]["profile"].values()))
        ),
        "",
        "E-mail the chosen 10,000 would get: "
        + "; ".join(
            f"{learner}: " + ", ".join(f"{k} {v:.0%}" for k, v in shares.items())
            for learner, shares in results["profile"]["emails"].items()
        )
        + ".",
        "",
        "## 5. Pre-specified moderators (all 64,000 customers, note section 6)",
        "",
        "Wald tests of whether each e-mail's effect varies with the moderator; Benjamini-Hochberg within",
        "each outcome's family of 8. Exploratory: leads, not findings.",
        "",
    ]
    for outcome, block in results["moderators"].items():
        lines += [
            f"### {outcome.capitalize()}",
            "",
            "| Moderator | E-mail | Chi-square (df) | p | BH-adjusted p | Lead |",
            "|---|---|---:|---:|---:|---|",
        ]
        for r in block["tests"]:
            lines.append(
                f"| {r['moderator']} | {SHORT[r['e-mail']]} | {r['chi2']:.2f} ({r['df']}) | {r['p']:.4f} "
                f"| {r['bh_p']:.4f} | {'yes' if r['lead'] else 'no'} |"
            )
        lines += [
            "",
            "| Purchase category | E-mail | Customers | Effect | 95% CI |",
            "|---|---|---:|---:|---|",
        ]
        for r in block["subgroups"]:
            lines.append(
                f"| {r['category']} | {SHORT[r['e-mail']]} | {r['customers']:,} "
                f"| {unit(outcome, r['effect'])} | [{unit(outcome, r['low'])}, {unit(outcome, r['high'])}] |"
            )
        lines.append("")
    return [*lines, *INTERPRETATION]


def main(from_cache: bool) -> None:
    if from_cache:
        results = json.loads(RESULTS.read_text())
    else:
        frozen = json.loads(CONFIG.read_text())
        models = refit_and_verify(load_half("train"), frozen)
        results = evaluate(load_half("test"), models)  # the test half, opened once
        results["profile"] = budget_profile(models)
        results["moderators"] = moderator_section_data()
        RESULTS.write_text(json.dumps(results, indent=2))
    lines = render(results)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
