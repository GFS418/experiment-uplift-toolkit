"""Phase 4a: tune and freeze the uplift models on the training half, then validate them.

Reads the training half only; the test half stays sealed until this phase is
committed (Phase 4 design note, sections 2 to 4 and 8). Writes
reports/phase4a_frozen_config.json, reports/phase4a_simulations.parquet and
reports/phase4a_tuning_and_validation.md. `--from-cache` rebuilds the report
from the JSON and the parquets without refitting or re-simulating; `--extra` runs only the
added check (plan, section 12, 2026-10-08) on the frozen configuration.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from exptools.adjust import covariate_matrix
from exptools.data import ARMS, ASSIGNMENT, CONTROL, REPO_ROOT, load_half, require_pre_treatment
from exptools.simulate_uplift import (
    Config,
    Pool,
    Scenario,
    conversion_probability,
    run,
    simulate_retuned_once,
)
from exptools.uplift import CONSTANT, FEATURES, GRID, CausalForest, DRLearner, TLearner, cv_select, fold_ids

MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
TREATMENTS = (MENS, WOMENS)
OUTCOME_NAMES = ("spend", "visit")  # note, section 2
FOLD_SEED = 202640  # note, section 3
SIZES = {CONTROL: 21_306, MENS: 21_307, WOMENS: 21_387}  # note, section 4: the real arm sizes
LIFTS = {MENS: 1.18, WOMENS: 0.65}  # note, section 4: the observed lifts
SCENARIOS = {"constant effects": (False, 202641), "heterogeneous effects": (True, 202642)}
N_SIMS = 200
# Added after the first run, before the test half was opened (plan, section 12, 2026-10-08).
EXTRA_SCENARIOS = {"constant effects": (False, 202645), "heterogeneous effects": (True, 202646)}
SHORT = {MENS: "men's e-mail", WOMENS: "women's e-mail", CONTROL: "no e-mail"}
CONFIG = REPO_ROOT / "reports" / "phase4a_frozen_config.json"
CACHE = REPO_ROOT / "reports" / "phase4a_simulations.parquet"
EXTRA_CACHE = REPO_ROOT / "reports" / "phase4a_extra_simulations.parquet"
REPORT = REPO_ROOT / "reports" / "phase4a_tuning_and_validation.md"

require_pre_treatment(FEATURES)  # the leakage guard, before any model is fit

# Written after reading the first run's output (it was empty for that run). The
# tuning, models and simulations are seeded and deterministic.
INTERPRETATION = [
    "## 3. Reading the results",
    "",
    "1. **Cross-validation found no heterogeneity worth modeling.** For both outcomes and both e-mails,",
    "   predicting the same effect for everyone beat the best of 16 tree models on the DR loss, by 0.07%",
    "   to 0.23%. The frozen DR-learner therefore predicts one number per e-mail: +$0.85 (men's) and",
    "   +$0.38 (women's) on spend, +7.4 and +4.9 points on visits, close to Phase 1b's full-sample",
    "   effects.",
    "2. **The outcome models are as cautious as the grid allows.** Every arm chose the slower learning",
    "   rate and 7-leaf trees, and all but one chose at least 200 customers per leaf: consistent with",
    "   Phase 2, where the features barely predicted spend.",
    "3. **What this fixes for Phase 4b, as pre-registered.** The primary calibration test has nothing",
    "   to test (the predictions are constant), the DR-learner's policy is \"everyone gets the men's",
    '   e-mail", and its 10,000-e-mail answer is a random 10,000. The T-learner and the causal forest',
    "   do vary their predictions (in-sample SD $1.44 and $0.89 for the men's e-mail on spend); 4b's",
    "   held-out calibration tests will show whether that variation is signal or noise.",
    "4. **The policy value estimator is honest.** Against the exact truth its bias is -$0.013 +/-",
    "   $0.025 and +$0.002 +/- $0.024 per customer, and its 95% intervals cover the truth 93.0% and",
    "   92.5% of the time: within Monte Carlo error of 95% for 200 experiments (about +/-3 points),",
    "   though possibly slightly low, as percentile intervals for skewed spend tend to be.",
    "5. **Even the choice between the two e-mails is uncertain on half the data.** In 10 of 200",
    "   experiments per scenario, the training half ranked the women's e-mail (+65%) above the men's",
    "   (+118%). That costs $0.33 per customer when it happens and $0.017 on average. The two scenarios",
    "   show the same figure because their average lifts are equal by construction and a constant",
    "   model only ever sends one e-mail.",
    "6. **The added check: the procedure could not have seen heterogeneity this large.** In the",
    "   heterogeneous scenario, past buyers of a category are four times as responsive to its",
    "   e-mail: the men's e-mail's planted effect averages $0.74 per customer with an SD of $1.00",
    "   (10th to 90th percentile: $0.05 to $1.95). Re-running the full DR procedure inside each",
    "   simulated experiment detected that only 3.0% of the time (Holm), no better than its",
    "   false-alarm rate, and cross-validation chose a tree model in 19% to 26% of experiments,",
    "   about as often as when effects were constant.",
    "7. **The T-learner's calibration test is valid, with some power.** Under constant effects it",
    "   fires 3.5% of the time (Holm; 7.0% and 5.0% per e-mail, within Monte Carlo error of 5%).",
    "   With the planted heterogeneity it fires 22% of the time (30% for the men's e-mail alone),",
    "   and its mean slope of 0.29 says most of the spread in its predictions is noise.",
    '8. **How to read Phase 4b.** A null result on the test half will mean "not detectable at this',
    '   sample size", not "everyone responds the same". Two-week spend is 99% zeros, so each',
    "   customer's response is far too noisy to learn from 32,000 training customers. The",
    "   T-learner's and causal forest's tests are the only ones with any chance, and on this",
    "   evidence even they would miss heterogeneity of the planted size most of the time.",
    "",
]


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.round(values, 10).tobytes()).hexdigest()[:16]


def features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return covariate_matrix(df, FEATURES), df[ASSIGNMENT].to_numpy(dtype=object)


def tune(train: pd.DataFrame) -> dict:
    """Cross-validated choices inside the training half (note, section 3)."""
    X, arm = features(train)
    folds = fold_ids(len(train), 5, FOLD_SEED)
    frozen = {}
    for outcome in OUTCOME_NAMES:
        y = train[outcome].to_numpy(dtype=float)
        outcome_params, outcome_mse = {}, {}
        for a in ARMS:
            rows = arm == a
            outcome_params[a], outcome_mse[a] = cv_select(X[rows], y[rows], folds[rows], GRID)
        dr = DRLearner(outcome_params, dict.fromkeys(TREATMENTS, CONSTANT), CONTROL, FOLD_SEED)
        scores = dr.cross_fitted_scores(X, y, arm)
        stage2_params, stage2_loss = {}, {}
        for t in TREATMENTS:
            stage2_params[t], stage2_loss[t] = cv_select(X, scores[t], folds, [CONSTANT, *GRID])
        frozen[outcome] = {
            "outcome_params": outcome_params,
            "outcome_cv_mse": outcome_mse,
            "stage2_params": stage2_params,
            "stage2_cv_loss": stage2_loss,
        }
    return frozen


def freeze(train: pd.DataFrame, frozen: dict) -> None:
    """Fit every final model twice; record fingerprints only if the two fits agree exactly."""
    X, arm = features(train)
    for outcome in OUTCOME_NAMES:
        cfg, y = frozen[outcome], train[outcome].to_numpy(dtype=float)
        makers = {
            "T-learner": lambda cfg=cfg: TLearner(cfg["outcome_params"], CONTROL),
            "DR-learner": lambda cfg=cfg: DRLearner(
                cfg["outcome_params"], cfg["stage2_params"], CONTROL, FOLD_SEED
            ),
            "causal forest": lambda: CausalForest(CONTROL, TREATMENTS),
        }
        prints, spreads = {}, {}
        for name, make in makers.items():
            first, second = make().fit(X, y, arm), make().fit(X, y, arm)
            for t in TREATMENTS:
                a, b = first.effect(X, t), second.effect(X, t)
                if fingerprint(a) != fingerprint(b):
                    raise RuntimeError(f"{name} ({outcome}, {t}) is not reproducible")
                prints[f"{name}|{t}"] = fingerprint(a)
                spreads[f"{name}|{t}"] = {
                    "mean": float(a.mean()),
                    "sd": float(a.std()),
                    "p5": float(np.percentile(a, 5)),
                    "p95": float(np.percentile(a, 95)),
                }
        cfg["fingerprints"], cfg["training_predictions"] = prints, spreads


def control_pool(train: pd.DataFrame) -> Pool:
    """The training half's control customers: the population of every simulated experiment."""
    control = train[train[ASSIGNMENT] == CONTROL].reset_index(drop=True)
    X = covariate_matrix(control, FEATURES)
    if X.shape[1] != features(train)[0].shape[1]:
        raise RuntimeError("control customers do not cover every category level")
    return Pool(
        X=X,
        spend=control["spend"].to_numpy(dtype=float),
        history=control["history"].to_numpy(dtype=float),
        category={MENS: control["mens"].to_numpy(), WOMENS: control["womens"].to_numpy()},
    )


def validate(train: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Plasmode validation from the training half's control customers (note, section 4)."""
    pool = control_pool(train)
    config = Config(frozen["spend"]["outcome_params"], frozen["spend"]["stage2_params"])
    frames = [
        run(pool, SIZES, CONTROL, Scenario(name, LIFTS, heterogeneous), config, N_SIMS, seed)
        for name, (heterogeneous, seed) in SCENARIOS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def extra_check(train: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """The added check: the DR second stage re-selected inside every simulated experiment."""
    pool = control_pool(train)
    config = Config(frozen["spend"]["outcome_params"], frozen["spend"]["stage2_params"])
    frames = [
        run(
            pool,
            SIZES,
            CONTROL,
            Scenario(name, LIFTS, heterogeneous),
            config,
            N_SIMS,
            seed,
            simulate=simulate_retuned_once,
        )
        for name, (heterogeneous, seed) in EXTRA_SCENARIOS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def describe(params) -> str:
    if params == CONSTANT:
        return "constant (same effect for everyone)"
    return (
        f"learning rate {params['learning_rate']}, {params['max_leaf_nodes']} leaves, "
        f"min leaf {params['min_samples_leaf']}, L2 {params['l2_regularization']}"
    )


def tuning_section(frozen: dict, train: pd.DataFrame) -> list[str]:
    counts = train[ASSIGNMENT].value_counts()
    lines = [
        "## 1. Tuning on the training half (note, section 3)",
        "",
        f"Training half: {len(train):,} customers ("
        + ", ".join(f"{SHORT.get(a, a)} {counts[a]:,}" for a in ARMS)
        + "). Five-fold cross-validation, folds seeded 202640.",
        "",
    ]
    for outcome in OUTCOME_NAMES:
        cfg = frozen[outcome]
        lines += [
            f"### {outcome.capitalize()}",
            "",
            "| Outcome model for | Selected | CV mean squared error | Worst setting's |",
            "|---|---|---:|---:|",
        ]
        for a in ARMS:
            mse = cfg["outcome_cv_mse"][a]
            best = min(mse.values())
            lines.append(
                f"| {a} | {describe(cfg['outcome_params'][a])} | {best:.6g} | {max(mse.values()):.6g} |"
            )
        lines += [
            "",
            "Second stage of the DR-learner, selected by the DR loss (mean squared error against the",
            "cross-fitted AIPW scores). The constant model predicts the same effect for everyone.",
            "",
            "| Effect of | Selected | Constant's loss | Best tree model's loss | Tree model vs constant |",
            "|---|---|---:|---:|---:|",
        ]
        for t in TREATMENTS:
            loss = cfg["stage2_cv_loss"][t]
            constant = loss[CONSTANT]
            best_tree = min(v for k, v in loss.items() if k != CONSTANT)
            lines.append(
                f"| {SHORT[t]} | {describe(cfg['stage2_params'][t])} | {constant:.6g} | {best_tree:.6g} "
                f"| {best_tree / constant - 1:+.3%} |"
            )
        lines += [
            "",
            "Frozen models' predicted effects on the training half (in-sample, descriptive only):",
            "",
            "| Learner | Effect of | Mean | SD across customers | 5th to 95th percentile | Fingerprint |",
            "|---|---|---:|---:|---|---|",
        ]
        unit = (
            (lambda v: f"{'-' if v < 0 else ''}${abs(v):.3f}")
            if outcome == "spend"
            else (lambda v: f"{100 * v:.2f} pp")
        )
        for key, s in cfg["training_predictions"].items():
            name, t = key.split("|")
            lines.append(
                f"| {name} | {SHORT[t]} | {unit(s['mean'])} | {unit(s['sd'])} "
                f"| {unit(s['p5'])} to {unit(s['p95'])} | `{cfg['fingerprints'][key]}` |"
            )
        lines.append("")
    return lines


def validation_section(sims: pd.DataFrame) -> list[str]:
    lines = [
        "## 2. Validation by simulation (note, section 4)",
        "",
        f"{N_SIMS} simulated experiments per scenario from the training half's control customers, with the",
        "real arm sizes, the same split procedure, and the frozen spend configuration (DR-learner).",
        "Calibration rejections count a non-applicable test (constant predictions) as no rejection.",
        "",
        "| Scenario | Calibration test applicable | Rejects: men's | Rejects: women's | Either, Holm "
        "| Policy value bias | 95% interval covers truth | True gain over best single e-mail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SCENARIOS:
        s = sims[sims["scenario"] == name]
        p = s[[f"{t}|p" for t in TREATMENTS]].fillna(1.0).to_numpy()
        reject = p < 0.05
        holm_any = p.min(axis=1) <= 0.025
        applicable = s[[f"{t}|applicable" for t in TREATMENTS]].astype(bool).to_numpy().mean()
        gap = s["value"] - s["truth"]
        covered = ((s["low"] <= s["truth"]) & (s["truth"] <= s["high"])).mean()
        gain = s["truth"] - s["best_single_truth"]
        lines.append(
            f"| {name} | {applicable:.0%} | {reject[:, 0].mean():.3f} | {reject[:, 1].mean():.3f} "
            f"| {holm_any.mean():.3f} | ${gap.mean():+.4f} +/-${1.96 * gap.std() / np.sqrt(len(gap)):.4f} "
            f"| {covered:.3f} | ${gain.mean():+.4f} per customer |"
        )
    lines += ["", "Average share of test customers the fitted policy sends to each arm:", ""]
    lines += ["| Scenario | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for name in SCENARIOS:
        s = sims[sims["scenario"] == name]
        lines.append(f"| {name} | " + " | ".join(f"{s[f'share|{a}'].mean():.0%}" for a in ARMS) + " |")
    lines.append("")
    return lines


def extra_section(extra: pd.DataFrame, pool: Pool) -> list[str]:
    lines = [
        "## 2b. Added check: the second stage re-selected in every simulated experiment",
        "",
        "Added after this phase's first run, at the user's request, before the test half was opened",
        "(plan, section 12). In each of 200 experiments per scenario (seeds 202645 and 202646), the",
        "DR-learner's second stage is chosen by the same cross-validation as on the real data (constant",
        "vs the 16 tree settings), and the T-learner's predictions get the same calibration test. A test",
        "that cannot apply (constant predictions) counts as no rejection. One-sided, alpha 0.05.",
        "",
        "| Scenario | DR picks a tree model: men's / women's | DR rejects: men's / women's | DR either, Holm "
        "| T-learner rejects: men's / women's | T-learner either, Holm "
        "| T-learner mean slope: men's / women's |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in EXTRA_SCENARIOS:
        s = extra[extra["scenario"] == name]
        tree = [1 - s[f"{t}|dr_constant"].astype(float).mean() for t in TREATMENTS]
        cells = []
        for learner in ("dr", "t"):
            p = s[[f"{t}|{learner}_p" for t in TREATMENTS]].astype(float).fillna(1.0).to_numpy()
            cells.append((p < 0.05).mean(axis=0))
            cells.append((p.min(axis=1) <= 0.025).mean())
        slopes = [s[f"{t}|t_slope"].astype(float).mean() for t in TREATMENTS]
        lines.append(
            f"| {name} | {tree[0]:.0%} / {tree[1]:.0%} | {cells[0][0]:.3f} / {cells[0][1]:.3f} "
            f"| {cells[1]:.3f} "
            f"| {cells[2][0]:.3f} / {cells[2][1]:.3f} | {cells[3]:.3f} | {slopes[0]:.2f} / {slopes[1]:.2f} |"
        )
    mean_buyer_spend = pool.spend[pool.spend > 0].mean()
    lines += ["", "Size of the planted heterogeneity (expected effect per customer of the population):", ""]
    for t in TREATMENTS:
        tau = conversion_probability(pool, t, LIFTS[t], True) * mean_buyer_spend
        lines.append(
            f"- {SHORT[t]}: mean ${tau.mean():.3f}, SD ${tau.std():.3f}, 10th to 90th percentile "
            f"${np.percentile(tau, 10):.3f} to ${np.percentile(tau, 90):.3f}"
        )
    margin = 1.96 * np.sqrt(0.05 * 0.95 / N_SIMS)
    lines += [
        "",
        f"Under constant effects every rejection is a false alarm; with {N_SIMS} experiments, a correctly",
        f"sized test lands within about +/-{margin:.3f} of 0.05 (single tests) or of 0.05 at most (Holm).",
        "",
    ]
    return lines


def main(from_cache: bool, extra_only: bool) -> None:
    train = load_half("train")  # the test half stays sealed
    if from_cache or extra_only:
        frozen = json.loads(CONFIG.read_text())
        sims = pd.read_parquet(CACHE)
    else:
        frozen = tune(train)
        freeze(train, frozen)
        CONFIG.write_text(json.dumps(frozen, indent=2, sort_keys=True))
        sims = validate(train, frozen)
        sims.to_parquet(CACHE)
    if from_cache:
        extra = pd.read_parquet(EXTRA_CACHE) if EXTRA_CACHE.exists() else None
    else:
        extra = extra_check(train, frozen)
        extra.to_parquet(EXTRA_CACHE)
    lines = [
        "# Phase 4a: tuning, freezing, and validating the uplift models",
        "",
        "Generated by `scripts/uplift_phase4a.py` under the Phase 4 design note. Training half only: the",
        "test half has not been read. The frozen configuration, with every cross-validation score and a",
        "fingerprint of each final model, is in `reports/phase4a_frozen_config.json`; Phase 4b must",
        "reproduce those fingerprints before it may score the test half.",
        "",
        *tuning_section(frozen, train),
        *validation_section(sims),
        *(extra_section(extra, control_pool(train)) if extra is not None else []),
        *INTERPRETATION,
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv, extra_only="--extra" in sys.argv)
