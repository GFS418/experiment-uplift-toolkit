"""Phase 4a: tune and freeze the uplift models on the training half, then validate them.

Reads the training half only; the test half stays sealed until this phase is
committed (Phase 4 design note, sections 2 to 4 and 8). Writes
reports/phase4a_frozen_config.json, reports/phase4a_simulations.parquet and
reports/phase4a_tuning_and_validation.md. `--from-cache` rebuilds the report
from the JSON and the parquet without refitting or re-simulating.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from exptools.adjust import covariate_matrix
from exptools.data import ARMS, ASSIGNMENT, CONTROL, REPO_ROOT, load_half, require_pre_treatment
from exptools.simulate_uplift import Config, Pool, Scenario, run
from exptools.uplift import CONSTANT, FEATURES, GRID, CausalForest, DRLearner, TLearner, cv_select, fold_ids

MENS, WOMENS = "Mens E-Mail", "Womens E-Mail"
TREATMENTS = (MENS, WOMENS)
OUTCOME_NAMES = ("spend", "visit")  # note, section 2
FOLD_SEED = 202640  # note, section 3
SIZES = {CONTROL: 21_306, MENS: 21_307, WOMENS: 21_387}  # note, section 4: the real arm sizes
LIFTS = {MENS: 1.18, WOMENS: 0.65}  # note, section 4: the observed lifts
SCENARIOS = {"constant effects": (False, 202641), "heterogeneous effects": (True, 202642)}
N_SIMS = 200
SHORT = {MENS: "men's e-mail", WOMENS: "women's e-mail", CONTROL: "no e-mail"}
CONFIG = REPO_ROOT / "reports" / "phase4a_frozen_config.json"
CACHE = REPO_ROOT / "reports" / "phase4a_simulations.parquet"
REPORT = REPO_ROOT / "reports" / "phase4a_tuning_and_validation.md"

require_pre_treatment(FEATURES)  # the leakage guard, before any model is fit

# Written after reading the first run's output; empty until then.
INTERPRETATION: list[str] = []


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


def validate(train: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Plasmode validation from the training half's control customers (note, section 4)."""
    control = train[train[ASSIGNMENT] == CONTROL].reset_index(drop=True)
    X = covariate_matrix(control, FEATURES)
    if X.shape[1] != features(train)[0].shape[1]:
        raise RuntimeError("control customers do not cover every category level")
    pool = Pool(
        X=X,
        spend=control["spend"].to_numpy(dtype=float),
        history=control["history"].to_numpy(dtype=float),
        category={MENS: control["mens"].to_numpy(), WOMENS: control["womens"].to_numpy()},
    )
    config = Config(frozen["spend"]["outcome_params"], frozen["spend"]["stage2_params"])
    frames = [
        run(pool, SIZES, CONTROL, Scenario(name, LIFTS, heterogeneous), config, N_SIMS, seed)
        for name, (heterogeneous, seed) in SCENARIOS.items()
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
        unit = (lambda v: f"${v:.3f}") if outcome == "spend" else (lambda v: f"{100 * v:.2f} pp")
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


def main(from_cache: bool) -> None:
    train = load_half("train")  # the test half stays sealed
    if from_cache:
        frozen = json.loads(CONFIG.read_text())
        sims = pd.read_parquet(CACHE)
    else:
        frozen = tune(train)
        freeze(train, frozen)
        CONFIG.write_text(json.dumps(frozen, indent=2, sort_keys=True))
        sims = validate(train, frozen)
        sims.to_parquet(CACHE)
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
        *INTERPRETATION,
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
