"""Phase 4c-a: Criteo checks, training-half effects, tuning and freezing at every size, A/A control.

Under prereg/phase4c_criteo_design_note.md (commit 92118f1). The test half stays sealed: every
outcome statistic here comes from the training half, and the checks over all rows (schema, 0/1
values, balance of the features) never touch an outcome. Writes reports/phase4c_checks.json,
reports/phase4c_frozen_config.json, reports/phase4c_aa_control.parquet and
reports/phase4c_a_checks_tuning.md; `--from-cache` rebuilds the report from those files.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from exptools import criteo
from exptools.adjust import difference_in_means
from exptools.criteo import CONTROL, FEATURES, OUTCOMES, TREATED
from exptools.data import REPO_ROOT, halves_by_arm
from exptools.freeze import fingerprint
from exptools.uplift import (
    CONSTANT,
    GRID,
    CausalForest,
    DRLearner,
    TLearner,
    aipw_scores,
    calibration_test,
    cv_select,
    fold_ids,
)

SIZES = (32_000, 320_000, 3_200_000, "full")  # note, section 4
FOREST_MAX = 320_000  # note, section 5
FOLD_SEED = 202652  # note, section 5
AA_SEED = 202655  # note, section 6
AA_PLAN = (
    (32_000, 20),
    (320_000, 20),
    (3_200_000, 5),
)  # note, section 6 (largest size capped, plan section 12)
CHECKS = REPO_ROOT / "reports" / "phase4c_checks.json"
CONFIG = REPO_ROOT / "reports" / "phase4c_frozen_config.json"
AA = REPO_ROOT / "reports" / "phase4c_aa_control.parquet"
REPORT = REPO_ROOT / "reports" / "phase4c_a_checks_tuning.md"

criteo.require_features(FEATURES)  # the leakage guard: exposure never enters a model

# Written after reading the first run's output (it was empty for that run). The report is
# rebuilt exactly from the cached JSON and parquet with --from-cache.
INTERPRETATION = [
    "## 4. Reading the results",
    "",
    "1. **The data are what the card says, except for balance.** Checksum, row count, columns, the",
    "   0.85 treatment share and the training-half outcome rates all match. No control customer was",
    "   exposed, and 3.6% of treated customers actually saw an ad.",
    "2. **Treatment is not independent of the features.** The largest standardized difference is",
    "   0.049 (z = 65), and the joint test gives LR = 5,345 on 12 df. A post-hoc diagnostic from",
    "   features and treatment only (plan, section 12) puts held-out treatment probability between",
    "   0.80 and 0.90 for 98.8% of customers (AUC 0.51), with about 1% at a treated share of 0.91 to",
    "   0.94. That is plausibly one advertiser's test run at a higher ratio, since the dataset pools",
    "   several tests.",
    "3. **The imbalance moves the average effect.** On the training half, the raw difference in",
    "   visit rates is +1.026 points (SE 0.021), but the full-size DR-learner, which adjusts for the",
    "   features through its outcome models, averages +0.719: 30% smaller, about 15 standard",
    "   errors apart. For conversion the gap is 14% (+0.116 vs +0.100 points). Customers more likely",
    "   to be treated also visit more for other reasons, so the raw comparison is partly",
    "   confounded. As the note pre-commits, the primary analysis keeps the constant propensity; a",
    "   sensitivity analysis with estimated propensities is the natural check for Phase 4c-b.",
    "4. **With more data, cross-validation starts preferring heterogeneity.** For visit, a tree",
    "   second stage beats the constant at 320,000, 3.2 million and the full size, by 0.13% to",
    "   0.24% of the DR loss; for conversion it does at 3.2 million and full size, by 0.08% and",
    "   0.15%. The 32,000-size choice for conversion (1.1%) is likely noise: in Phase 4a,",
    "   cross-validation picked a tree about 20% of the time with no heterogeneity at all. The",
    "   outcome models also grow richer with more data (31 leaves and a faster learning rate for",
    "   treated visits at the largest sizes).",
    "5. **The frozen models predict real spread, to be tested.** At full size the DR-learner's",
    "   predicted visit effects have an SD of 2.17 points around a 0.72-point mean (5th to 95th",
    "   percentile: +0.04 to +4.08). In-sample spread proves nothing; the sealed test half decides.",
    "6. **The A/A control is clean.** Across 160 applicable calibration tests where every true",
    "   effect is zero, 7 rejected (4.4%), with mean slopes near zero. It is a coarse check, but it",
    "   shows no sign that the pipeline manufactures heterogeneity, including at 2.97 million.",
    "",
]

START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - START:7.0f}s] {message}", flush=True)


def checks(df: pd.DataFrame, train: np.ndarray) -> dict:
    """Section 3: integrity, balance and training-half effects."""
    treated = df["treatment"].to_numpy() == 1
    exposed = df["exposure"].to_numpy() == 1
    X = df[list(FEATURES)].to_numpy(dtype=np.float64)
    n_t, n_c = treated.sum(), (~treated).sum()
    smd = (X[treated].mean(0) - X[~treated].mean(0)) / np.sqrt(
        (X[treated].var(0, ddof=1) + X[~treated].var(0, ddof=1)) / 2
    )
    logit = sm.Logit(treated.astype(float), sm.add_constant(X)).fit(disp=0, maxiter=100)
    out = {
        "rows": int(len(df)),
        "sha256": criteo.SHA256,
        "treated_share": float(treated.mean()),
        "exposed_controls": int((exposed & ~treated).sum()),
        "exposed_share_of_treated_train": float(exposed[treated & train].mean()),
        "train_rates": {o: float(df[o].to_numpy()[train].mean()) for o in OUTCOMES},
        "balance": {
            "smd": dict(zip(FEATURES, smd.tolist(), strict=True)),
            "z": dict(zip(FEATURES, (smd / np.sqrt(1 / n_t + 1 / n_c)).tolist(), strict=True)),
            "lr": float(logit.llr),
            "df": int(logit.df_model),
            "p": float(logit.llr_pvalue),
            "converged": bool(logit.mle_retvals["converged"]),
        },
        "train_effects": {},
    }
    arm = criteo.arm_labels(df["treatment"].to_numpy()[train])
    for o in OUTCOMES:
        y = df[o].to_numpy(dtype=float)[train]
        est = difference_in_means(y, arm, TREATED, CONTROL)
        low, high = est.ci()
        base = y[arm == CONTROL].mean()
        out["train_effects"][o] = {
            "estimate": est.estimate,
            "se": est.se,
            "low": low,
            "high": high,
            "control_rate": float(base),
            "relative": est.estimate / base,
        }
    return out


def tune_and_freeze(X_train: np.ndarray, arm_train: np.ndarray, y_train: dict) -> dict:
    """Sections 4 and 5: the Phase 4 procedure at every nested training size."""
    order = criteo.nested_subsample_order(arm_train)
    frozen = {}
    for size in SIZES:
        rows = (
            np.arange(len(arm_train)) if size == "full" else criteo.nested_subsample(arm_train, size, order)
        )
        X, arm = X_train[rows], arm_train[rows]
        folds = fold_ids(len(rows), 5, FOLD_SEED)
        entry = {"n": int(len(rows)), "treated_share": float(np.mean(arm == TREATED))}
        for outcome in OUTCOMES:
            t0 = time.perf_counter()
            y = y_train[outcome][rows]
            outcome_params, outcome_mse = {}, {}
            for a in (CONTROL, TREATED):
                sel = arm == a
                outcome_params[a], outcome_mse[a] = cv_select(X[sel], y[sel], folds[sel], GRID)
            dr = DRLearner(outcome_params, {TREATED: CONSTANT}, CONTROL, FOLD_SEED)
            scores = dr.cross_fitted_scores(X, y, arm)[TREATED]
            stage2, stage2_loss = cv_select(X, scores, folds, [CONSTANT, *GRID])
            cfg = {
                "outcome_params": outcome_params,
                "outcome_cv_mse": outcome_mse,
                "stage2_params": stage2,
                "stage2_cv_loss": stage2_loss,
            }
            log(
                f"size {len(rows):>9,} {outcome}: tuned "
                f"(second stage: {'constant' if stage2 == CONSTANT else 'tree'})"
            )
            freeze(cfg, X, y, arm)
            cfg["seconds"] = round(time.perf_counter() - t0)
            log(f"size {len(rows):>9,} {outcome}: frozen and fingerprinted in {cfg['seconds']}s")
            entry[outcome] = cfg
        frozen[str(size)] = entry
    return frozen


def freeze(cfg: dict, X: np.ndarray, y: np.ndarray, arm: np.ndarray) -> None:
    """Fit each final model twice; keep fingerprints only if the two fits agree exactly."""
    makers = {
        "T-learner": lambda: TLearner(cfg["outcome_params"], CONTROL),
        "DR-learner": lambda: DRLearner(
            cfg["outcome_params"], {TREATED: cfg["stage2_params"]}, CONTROL, FOLD_SEED
        ),
    }
    if len(y) <= FOREST_MAX:
        makers["causal forest"] = lambda: CausalForest(CONTROL, (TREATED,))
    prints, spreads = {}, {}
    for name, make in makers.items():
        a = make().fit(X, y, arm).effect(X, TREATED)
        b = make().fit(X, y, arm).effect(X, TREATED)
        if fingerprint(a) != fingerprint(b):
            raise RuntimeError(f"{name} is not reproducible")
        prints[name] = fingerprint(a)
        spreads[name] = {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }
    cfg["fingerprints"], cfg["training_predictions"] = prints, spreads


def aa_control(X_train: np.ndarray, arm_train: np.ndarray, y_train: dict, frozen: dict) -> pd.DataFrame:
    """Section 6: pseudo-treatment among customers who were all treated; every true effect is zero."""
    pool = np.flatnonzero(arm_train == TREATED)
    share = float(np.mean(arm_train == TREATED))
    rng = np.random.default_rng(AA_SEED)
    records = []
    for size, reps in AA_PLAN:
        for rep in range(reps):
            pseudo = np.full(len(pool), CONTROL, dtype=object)
            pseudo[rng.permutation(len(pool))[: round(share * len(pool))]] = TREATED
            half = halves_by_arm(pseudo, rng, (CONTROL, TREATED))
            train_pos, test_pos = np.flatnonzero(half == "train"), np.flatnonzero(half == "test")
            n = min(size, len(train_pos))
            order = criteo.nested_subsample_order(pseudo[train_pos], seed=int(rng.integers(2**31)))
            sample = train_pos[criteo.nested_subsample(pseudo[train_pos], n, order)]
            X_s, arm_s = X_train[pool[sample]], pseudo[sample]
            X_t, arm_t = X_train[pool[test_pos]], pseudo[test_pos]
            for outcome in OUTCOMES:
                cfg = frozen[str(size)][outcome]
                y_s, y_t = y_train[outcome][pool[sample]], y_train[outcome][pool[test_pos]]
                t_learner = TLearner(cfg["outcome_params"], CONTROL).fit(X_s, y_s, arm_s)
                dr_seed = int(rng.integers(2**31))
                dr = DRLearner(cfg["outcome_params"], {TREATED: cfg["stage2_params"]}, CONTROL, dr_seed)
                dr.fit(X_s, y_s, arm_s)
                scores = aipw_scores(y_t, arm_t, t_learner.outcomes(X_t), TREATED, CONTROL)
                for name, model in (("T-learner", t_learner), ("DR-learner", dr)):
                    cal = calibration_test(scores, model.effect(X_t, TREATED))
                    records.append(
                        {
                            "planned_size": size,
                            "n": n,
                            "rep": rep,
                            "outcome": outcome,
                            "learner": name,
                            "applicable": cal.applicable,
                            "slope": cal.slope,
                            "p": cal.p_value,
                        }
                    )
            log(f"A/A size {n:>9,} repetition {rep + 1}/{reps} done")
    return pd.DataFrame.from_records(records)


# ---- report -------------------------------------------------------------------------------------


def describe(params) -> str:
    if params == CONSTANT:
        return "constant"
    return (
        f"lr {params['learning_rate']}, {params['max_leaf_nodes']} leaves, "
        f"leaf {params['min_samples_leaf']}, L2 {params['l2_regularization']}"
    )


def unit(outcome: str, x: float) -> str:
    return f"{100 * x:+.3f} pp" if outcome == "visit" else f"{100 * x:+.4f} pp"


def render(chk: dict, frozen: dict, aa: pd.DataFrame) -> list[str]:
    bal, eff = chk["balance"], chk["train_effects"]
    worst = max(bal["z"], key=lambda f: abs(bal["z"][f]))
    lines = [
        "# Phase 4c-a: Criteo checks, tuning at every size, and an A/A control",
        "",
        "Generated by `scripts/criteo_phase4c_a.py` under the Phase 4c design note (commit `92118f1`).",
        "The test half has not been read: outcome statistics come from the training half only.",
        "",
        "## 1. Data checks (note, section 3)",
        "",
        f"- **Integrity:** SHA-256 matches the host's published value; {chk['rows']:,} rows (v2.1); the 16",
        "  expected columns; no missing values; every flag is 0 or 1.",
        f"- **Treatment share:** {chk['treated_share']:.4f} (dataset card: 0.85).",
        f"- **Exposure:** {chk['exposed_controls']:,} control customers were exposed (should be 0);",
        f"  {chk['exposed_share_of_treated_train']:.1%} of treated customers in the training half saw an ad.",
        f"- **Outcome rates, training half:** visit {chk['train_rates']['visit']:.4%} (card, all rows:",
        f"  {criteo.CARD['visit']:.4%}); conversion {chk['train_rates']['conversion']:.4%} "
        f"(card: {criteo.CARD['conversion']:.4%}).",
        "",
        "**Balance of the 12 features, treated vs control (all rows; features only):**",
        "",
        "| Feature | SMD | z (chance scale) |",
        "|---|---:|---:|",
        *(f"| {f} | {bal['smd'][f]:+.4f} | {bal['z'][f]:+.2f} |" for f in FEATURES),
        "",
        f"Largest |SMD| {max(abs(v) for v in bal['smd'].values()):.4f} ({worst}). "
        "Joint likelihood-ratio test",
        f"(logistic regression of treatment on the features): LR = {bal['lr']:,.1f} on {bal['df']} df,",
        f"p = {bal['p']:.3g}.",
        "",
        "**Average effects of treatment, training half:**",
        "",
        "| Outcome | Control rate | Effect | 95% CI | Relative lift |",
        "|---|---:|---:|---|---:|",
        *(
            f"| {o} | {eff[o]['control_rate']:.4%} | {unit(o, eff[o]['estimate'])} "
            f"| [{unit(o, eff[o]['low'])}, {unit(o, eff[o]['high'])}] | {eff[o]['relative']:+.1%} |"
            for o in OUTCOMES
        ),
        "",
        "## 2. Tuning and freezing at every training size (note, sections 4 and 5)",
        "",
    ]
    for outcome in OUTCOMES:
        lines += [
            f"### {outcome.capitalize()}",
            "",
            "| Training size | Outcome model, control | Outcome model, treated | DR second stage "
            "| Best tree's DR loss vs constant | Minutes |",
            "|---:|---|---|---|---:|---:|",
        ]
        for size in SIZES:
            entry = frozen[str(size)]
            cfg = entry[outcome]
            loss = cfg["stage2_cv_loss"]
            best_tree = min(v for k, v in loss.items() if k != CONSTANT)
            lines.append(
                f"| {entry['n']:,} | {describe(cfg['outcome_params'][CONTROL])} "
                f"| {describe(cfg['outcome_params'][TREATED])} | {describe(cfg['stage2_params'])} "
                f"| {best_tree / loss[CONSTANT] - 1:+.4%} | {cfg['seconds'] / 60:.1f} |"
            )
        lines += [
            "",
            "Frozen models' predicted effects on their own training sample (in-sample, descriptive only):",
            "",
            "| Training size | Learner | Mean | SD across customers | 5th to 95th percentile | Fingerprint |",
            "|---:|---|---:|---:|---|---|",
        ]
        for size in SIZES:
            entry = frozen[str(size)]
            for name, s in entry[outcome]["training_predictions"].items():
                lines.append(
                    f"| {entry['n']:,} | {name} | {unit(outcome, s['mean'])} | {100 * s['sd']:.3f} pp "
                    f"| {unit(outcome, s['p5'])} to {unit(outcome, s['p95'])} "
                    f"| `{entry[outcome]['fingerprints'][name]}` |"
                )
        lines.append("")
    lines += [
        "## 3. A/A negative control (note, section 6)",
        "",
        "Pseudo-treatment at the real share among training-half customers who were all treated, so every",
        "true effect is zero. Calibration rejections (one-sided, 0.05) should be rare; a test that cannot",
        "apply (constant predictions) counts as no rejection. The largest planned size runs at the whole",
        "pseudo-training half (plan, section 12).",
        "",
        "| Size | Repetitions | Outcome | Learner | Test applicable | Rejections | Mean slope |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for (size, outcome, learner), g in aa.groupby(["n", "outcome", "learner"], sort=True):
        rejections = int((g["p"].astype(float).fillna(1.0) < 0.05).sum())
        lines.append(
            f"| {size:,} | {len(g)} | {outcome} | {learner} | {g['applicable'].astype(bool).mean():.0%} "
            f"| {rejections} of {len(g)} | {g['slope'].astype(float).mean():.2f} |"
        )
    return [*lines, "", *INTERPRETATION]


def main(from_cache: bool) -> None:
    if from_cache:
        chk, frozen, aa = json.loads(CHECKS.read_text()), json.loads(CONFIG.read_text()), pd.read_parquet(AA)
    else:
        criteo.download()
        criteo.convert()
        df = criteo.load()
        half = criteo.split_halves(df["treatment"].to_numpy())
        train = half == "train"
        log(f"loaded {len(df):,} rows; training half {train.sum():,}")
        chk = checks(df, train)
        CHECKS.write_text(json.dumps(chk, indent=2))
        log("checks done")
        X_train = df.loc[train, list(FEATURES)].to_numpy(dtype=np.float32)
        arm_train = criteo.arm_labels(df.loc[train, "treatment"].to_numpy())
        y_train = {o: df.loc[train, o].to_numpy(dtype=float) for o in OUTCOMES}
        del df  # the test half's outcomes are never held from here on
        frozen = tune_and_freeze(X_train, arm_train, y_train)
        CONFIG.write_text(json.dumps(frozen, indent=2, sort_keys=True))
        aa = aa_control(X_train, arm_train, y_train, frozen)
        aa.to_parquet(AA)
        log("A/A control done")
    lines = render(chk, frozen, aa)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
