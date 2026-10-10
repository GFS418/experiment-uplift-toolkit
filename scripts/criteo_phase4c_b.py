"""Phase 4c-b: score the frozen Criteo models on the test half, once.

Under prereg/phase4c_criteo_design_note.md, section 7, plus the sensitivity analysis logged in the
plan's section 12 (2026-10-09). Before anything is scored, every frozen model is refit on its
training sample and must reproduce its Phase 4c-a fingerprint; otherwise the script stops.
Writes reports/phase4c_b_results.json and reports/phase4c_b_test_evaluation.md;
`--from-cache` rebuilds the report from the JSON.
"""

import json
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from exptools import criteo
from exptools.criteo import CONTROL, FEATURES, OUTCOMES, TREATED
from exptools.freeze import fingerprint
from exptools.uplift import CausalForest, DRLearner, TLearner, calibration_test

SIZES = ("32000", "320000", "3200000", "full")
FOREST_MAX = 320_000
FOLD_SEED = 202652
TIE_SEED, BOOT_SEED, PROPENSITY_SEED = 202653, 202654, 202656
N_AREA_BOOT, N_POLICY_BOOT = 2_000, 10_000
BUDGETS = (0.10, 0.25, 0.50)
CONFIG = criteo.REPO_ROOT / "reports" / "phase4c_frozen_config.json"
RESULTS = criteo.REPO_ROOT / "reports" / "phase4c_b_results.json"
REPORT = criteo.REPO_ROOT / "reports" / "phase4c_b_test_evaluation.md"

criteo.require_features(FEATURES)

# Written after reading the first run's output (it was empty for that run). The report is
# rebuilt exactly from reports/phase4c_b_results.json with --from-cache; the test half is not reread.
INTERPRETATION = [
    "## 6. Reading the results",
    "",
    "1. **At full scale, the heterogeneity is real and the predictions are calibrated.** The",
    "   pre-registered test passes for both outcomes: the full-size DR-learner's slope is 1.09",
    "   (SE 0.02) for visits and 0.85 (SE 0.07) for conversions. A slope near 1 means customers",
    "   predicted to respond more really do, by about the predicted amount.",
    "2. **The learning curve answers Phase 4's open question.** At Hillstrom's training size",
    "   (32,000), cross-validation keeps the DR-learner constant for visits, while the T-learner's",
    "   predictions are mostly noise (slope 0.25). The DR-learner models visit heterogeneity from",
    "   320,000 on (slope 0.87) and is calibrated from 3.2 million (1.07). The T-learner improves",
    "   steadily (0.25, 0.55, 0.81, 0.87) but stays over-dispersed. Ranking quality (uplift area)",
    "   plateaus at about 0.37 points per customer from 3.2 million on.",
    "3. **Not every step is monotone.** For conversion, cross-validation chose a constant at",
    "   320,000 (no targeting value there) but a tree at 32,000 that turned out useful (slope 0.27).",
    "   For a 0.29% outcome, selection between nearly tied models is noisy at moderate sizes.",
    "4. **The causal forest learns fastest.** At 320,000 it is calibrated (0.97 for both outcomes)",
    "   and matches the full-size DR-learner's ranking quality (visit area 0.352 vs 0.373 points).",
    "5. **Targeting pays, but the imbalance inflates it.** Treating the top quarter by the full-size",
    "   DR-learner's predictions yields 7.3 more visits per 1,000 customers than treating a random",
    "   quarter with the constant 0.85 probability, but 4.9 [4.6, 5.3] with estimated probabilities.",
    "   For conversions it is 0.80 vs 0.66 per 1,000. The calibration slopes barely move under the",
    "   sensitivity analysis (1.09 to 1.14 for visit), so the heterogeneity finding is robust, but",
    "   the size of the targeting gain is not: quote the range.",
    "6. **The imbalance finding replicates on held-out data.** On the test half the raw visit",
    "   difference is +1.04 points, but both doubly robust versions give about +0.77 (constant or",
    "   estimated probability). The raw comparison overstates the average effect by about a third.",
    "7. **What this means for Hillstrom.** Phase 4's null was a sample-size result, not proof of",
    "   uniform effects. On Criteo, 32,000 customers was also too few for the DR-learner, and",
    "   reliable, calibrated targeting needed hundreds of thousands to millions.",
    "",
]

START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - START:7.0f}s] {message}", flush=True)


def scores(y, treated, m_t, m_c, e):
    """AIPW scores; e is a scalar (the constant share) or a per-customer array."""
    return m_t - m_c + treated * (y - m_t) / e - (~treated) * (y - m_c) / (1 - e)


def ranks_to_bins(pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Percentile bin 0..99 by predicted effect, highest first; ties in a seeded random order."""
    order = np.lexsort((rng.permutation(len(pred)), -pred))
    bins = np.empty(len(pred), dtype=np.int64)
    bins[order] = np.arange(len(pred)) * 100 // len(pred)
    return bins


def curve_from_cells(cells_t: np.ndarray, cells_c: np.ndarray) -> np.ndarray:
    """Uplift curve at f = 1%..100% from per-bin (count, outcome sum) cells of each arm."""
    n_t, s_t = np.cumsum(cells_t[..., 0], -1), np.cumsum(cells_t[..., 1], -1)
    n_c, s_c = np.cumsum(cells_c[..., 0], -1), np.cumsum(cells_c[..., 1], -1)
    f = np.arange(1, 101) / 100
    return f * (s_t / n_t - s_c / n_c)


def area(gains: np.ndarray) -> np.ndarray:
    f = np.arange(1, 101) / 100
    return np.mean(gains - f * gains[..., -1:], axis=-1)


def uplift_area(y, treated, pred, rng_tie, rng_boot):
    """Qini-type area with an exact bootstrap: customers resampled within arm as multinomial
    counts over (percentile bin, 0/1 outcome) cells, so 7 million rows are never re-sorted."""
    bins = ranks_to_bins(pred, rng_tie)
    out = {}
    cells = {}
    for arm, mask in ((TREATED, treated), (CONTROL, ~treated)):
        cell = bins[mask] * 2 + y[mask].astype(np.int64)
        counts = np.bincount(cell, minlength=200)
        cells[arm] = counts

    def to_cells(counts):  # (..., 200) -> (..., 100, 2) as (customers, outcome sum)
        c = counts.reshape(*counts.shape[:-1], 100, 2)
        return np.stack([c.sum(-1), c[..., 1]], axis=-1)

    point = area(curve_from_cells(to_cells(cells[TREATED]), to_cells(cells[CONTROL])))
    draws = {arm: rng_boot.multinomial(c.sum(), c / c.sum(), size=N_AREA_BOOT) for arm, c in cells.items()}
    boot = area(curve_from_cells(to_cells(draws[TREATED]), to_cells(draws[CONTROL])))
    out["area"], out["low"], out["high"] = float(point), *np.percentile(boot, [2.5, 97.5]).tolist()
    return out


def targeting_constant(y, treated, pred, rng_tie, rng_boot):
    """Treat the top b by predicted effect vs a random b; IPW with the exact shares; exact
    paired bootstrap over (in policy, outcome) cells within arm."""
    bins = ranks_to_bins(pred, rng_tie)
    out = {}
    for b in BUDGETS:
        chosen = bins < round(100 * b)
        cells = {}
        for arm, mask in ((TREATED, treated), (CONTROL, ~treated)):
            cells[arm] = np.bincount(chosen[mask] * 2 + y[mask].astype(np.int64), minlength=4)

        def gain(ct, cc, b=b):
            # value of the policy: treated customers it treats + control customers it leaves alone
            n_t, n_c = ct.sum(-1), cc.sum(-1)
            v_policy = ct[..., 3] / n_t + cc[..., 1] / n_c
            v_random = b * (ct[..., 1] + ct[..., 3]) / n_t + (1 - b) * (cc[..., 1] + cc[..., 3]) / n_c
            return v_policy - v_random

        point = gain(cells[TREATED], cells[CONTROL])
        d_t = rng_boot.multinomial(
            cells[TREATED].sum(), cells[TREATED] / cells[TREATED].sum(), size=N_POLICY_BOOT
        )
        d_c = rng_boot.multinomial(
            cells[CONTROL].sum(), cells[CONTROL] / cells[CONTROL].sum(), size=N_POLICY_BOOT
        )
        boot = gain(d_t, d_c)
        out[f"{b:.2f}"] = {
            "gain": float(point),
            "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5)),
        }
    return out


def targeting_estimated(y, treated, pred, e, rng_tie):
    """Sensitivity: same policies, IPW with estimated probabilities; normal-approximation intervals."""
    bins = ranks_to_bins(pred, rng_tie)
    out = {}
    w_t, w_c = treated * y / e, (~treated) * y / (1 - e)
    for b in BUDGETS:
        chosen = bins < round(100 * b)
        contrib = np.where(chosen, w_t, w_c) - (b * w_t + (1 - b) * w_c)
        g, se = contrib.mean(), contrib.std(ddof=1) / np.sqrt(len(contrib))
        out[f"{b:.2f}"] = {"gain": float(g), "low": float(g - 1.96 * se), "high": float(g + 1.96 * se)}
    return out


def calibration(psi, pred):
    cal = calibration_test(psi, pred)
    return {
        "ate": cal.ate,
        "slope": cal.slope,
        "slope_se": cal.slope_se,
        "p": cal.p_value,
        "applicable": cal.applicable,
    }


def main_run() -> dict:
    frozen = json.loads(CONFIG.read_text())
    df = criteo.load()
    half = criteo.split_halves(df["treatment"].to_numpy())
    train, test = half == "train", half == "test"
    X_tr = df.loc[train, list(FEATURES)].to_numpy(np.float32)
    arm_tr = criteo.arm_labels(df.loc[train, "treatment"].to_numpy())
    y_tr = {o: df.loc[train, o].to_numpy(float) for o in OUTCOMES}
    order = criteo.nested_subsample_order(arm_tr)

    # 1. Refit every frozen model and check its fingerprint before the test half is touched.
    models = {}
    for size in SIZES:
        rows = np.arange(len(arm_tr)) if size == "full" else criteo.nested_subsample(arm_tr, int(size), order)
        X, arm = X_tr[rows], arm_tr[rows]
        for outcome in OUTCOMES:
            cfg, y = frozen[size][outcome], y_tr[outcome][rows]
            fitted = {
                "T-learner": TLearner(cfg["outcome_params"], CONTROL).fit(X, y, arm),
                "DR-learner": DRLearner(
                    cfg["outcome_params"], {TREATED: cfg["stage2_params"]}, CONTROL, FOLD_SEED
                ).fit(X, y, arm),
            }
            if len(rows) <= FOREST_MAX:
                fitted["causal forest"] = CausalForest(CONTROL, (TREATED,)).fit(X, y, arm)
            for name, model in fitted.items():
                if fingerprint(model.effect(X, TREATED)) != cfg["fingerprints"][name]:
                    raise RuntimeError(
                        f"{name} at {size} ({outcome}) does not reproduce its frozen fingerprint"
                    )
            models[(size, outcome)] = fitted
            log(f"refit and verified: size {size} {outcome}")

    # Sensitivity: treatment probability from the training half's features.
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        max_iter=200,
        early_stopping=False,
        random_state=PROPENSITY_SEED,
    )
    clf.fit(X_tr, (arm_tr == TREATED).astype(int))
    del X_tr, y_tr

    # 2. The test half, opened once.
    X_te = df.loc[test, list(FEATURES)].to_numpy(np.float32)
    treated = df.loc[test, "treatment"].to_numpy() == 1
    y_te = {o: df.loc[test, o].to_numpy(float) for o in OUTCOMES}
    del df
    e_const = treated.mean()
    e_hat = np.clip(clf.predict_proba(X_te)[:, 1], 0.01, 0.99)
    log(f"test half opened: {len(treated):,} customers")

    results = {
        "n_test": int(len(treated)),
        "treated_share_test": float(e_const),
        "propensity": {
            "p1": float(np.percentile(e_hat, 1)),
            "p50": float(np.median(e_hat)),
            "p99": float(np.percentile(e_hat, 99)),
        },
        "evaluations": {},
    }
    for (size, outcome), fitted in models.items():
        y = y_te[outcome]
        m = fitted["T-learner"].outcomes(X_te)
        psi_c = scores(y, treated, m[TREATED], m[CONTROL], e_const)
        psi_e = scores(y, treated, m[TREATED], m[CONTROL], e_hat)
        for name, model in fitted.items():
            pred = model.effect(X_te, TREATED)
            key = f"{size}|{outcome}|{name}"
            results["evaluations"][key] = {
                "n_train": frozen[size]["n"],
                "prediction_sd": float(pred.std()),
                "calibration": calibration(psi_c, pred),
                "calibration_estimated_e": calibration(psi_e, pred),
                "uplift": uplift_area(
                    y, treated, pred, np.random.default_rng(TIE_SEED), np.random.default_rng(BOOT_SEED)
                ),
                "targeting": targeting_constant(
                    y, treated, pred, np.random.default_rng(TIE_SEED), np.random.default_rng(BOOT_SEED)
                ),
                "targeting_estimated_e": targeting_estimated(
                    y, treated, pred, e_hat, np.random.default_rng(TIE_SEED)
                ),
            }
        results.setdefault("ate", {})[f"{size}|{outcome}"] = {
            "raw": float(y[treated].mean() - y[~treated].mean()),
            "aipw_constant": float(psi_c.mean()),
            "aipw_estimated": float(psi_e.mean()),
        }
        log(f"evaluated: size {size} {outcome}")

    # Primary family: DR-learner at full size, visit and conversion, Holm.
    ps = []
    for outcome in OUTCOMES:
        c = results["evaluations"][f"full|{outcome}|DR-learner"]["calibration"]
        ps.append(c["p"] if c["applicable"] else 1.0)
    order_p = np.argsort(ps)
    holm = np.minimum(1, np.maximum.accumulate(np.array(ps)[order_p] * np.array([2, 1])))
    adj = np.empty(2)
    adj[order_p] = holm
    results["primary"] = {
        o: {"p": ps[i], "holm_p": float(adj[i]), "reject": bool(adj[i] < 0.05)}
        for i, o in enumerate(OUTCOMES)
    }
    return results


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def pp(x, d=3):
    return f"{100 * x:+.{d}f} pp"


def render(r: dict) -> list[str]:
    ev = r["evaluations"]
    label = {"32000": "32,000", "320000": "320,000", "3200000": "3,200,000", "full": "6,989,795"}
    lines = [
        "# Phase 4c-b: the frozen Criteo models on the test half",
        "",
        "Generated by `scripts/criteo_phase4c_b.py`, the only code that reads the Criteo test"
        " half, under the",
        "Phase 4c design note plus the logged sensitivity analysis. Every frozen model reproduced its 4c-a",
        f"fingerprint before scoring. Test half: {r['n_test']:,} customers,"
        f" {r['treated_share_test']:.4f} treated.",
        "",
        "## 1. Primary test: are the full-size DR-learner's predictions calibrated?",
        "",
        "| Outcome | Slope (SE) | One-sided p | Holm-adjusted p | Heterogeneity detected |",
        "|---|---:|---:|---:|---|",
    ]
    for o in OUTCOMES:
        c = ev[f"full|{o}|DR-learner"]["calibration"]
        pr = r["primary"][o]
        slope = f"{c['slope']:.2f} ({c['slope_se']:.2f})" if c["applicable"] else "n/a (constant)"
        lines.append(
            f"| {o} | {slope} | {fmt_p(c['p'])} | {fmt_p(pr['holm_p'])} | {'yes' if pr['reject'] else 'no'} |"
        )
    lines += [
        "",
        "## 2. Learning curve: calibration on the test half at every training size",
        "",
        "Slope 1 = calibrated predictions; 0 = no real heterogeneity. One-sided p for slope > 0. The last",
        "column repeats the test with estimated treatment probabilities (sensitivity analysis).",
        "",
        "| Outcome | Training size | Learner | SD of predictions | Slope (SE) | p | Slope, estimated"
        " probabilities (p) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for o in OUTCOMES:
        for s in SIZES:
            for name in ("DR-learner", "T-learner", "causal forest"):
                k = f"{s}|{o}|{name}"
                if k not in ev:
                    continue
                c, ce = ev[k]["calibration"], ev[k]["calibration_estimated_e"]
                if c["applicable"]:
                    a = f"{c['slope']:.2f} ({c['slope_se']:.2f})"
                    b = f"{ce['slope']:.2f} ({fmt_p(ce['p'])})"
                else:
                    a, b = "n/a (constant)", "n/a"
                lines.append(
                    f"| {o} | {label[s]} | {name} | {pp(ev[k]['prediction_sd'])} | {a} |"
                    f" {fmt_p(c['p'])} | {b} |"
                )
    lines += [
        "",
        "## 3. Uplift curves: does ranking by predicted effect beat random targeting?",
        "",
        "Qini-type area per customer (percentage points), 95% exact bootstrap interval (2,000 resamples).",
        "",
        "| Outcome | Training size | Learner | Area | 95% interval |",
        "|---|---:|---|---:|---|",
    ]
    for o in OUTCOMES:
        for s in SIZES:
            for name in ("DR-learner", "T-learner", "causal forest"):
                k = f"{s}|{o}|{name}"
                if k in ev:
                    u = ev[k]["uplift"]
                    lines.append(
                        f"| {o} | {label[s]} | {name} | {pp(u['area'], 4)} | [{pp(u['low'], 4)},"
                        f" {pp(u['high'], 4)}] |"
                    )
    lines += [
        "",
        "## 4. Targeting value: treat the top share by predicted effect vs the same share at random",
        "",
        "Extra visits or conversions per 1,000 customers. Constant probability: exact paired bootstrap.",
        "Estimated probabilities (sensitivity): normal approximation.",
        "",
        "| Outcome | Training size | Learner | Budget | Gain per 1,000 [95%] | With estimated"
        " probabilities [95%] |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for o in OUTCOMES:
        for s in SIZES:
            for name in ("DR-learner", "T-learner", "causal forest"):
                k = f"{s}|{o}|{name}"
                if k not in ev:
                    continue
                for b in BUDGETS:
                    t, te = ev[k]["targeting"][f"{b:.2f}"], ev[k]["targeting_estimated_e"][f"{b:.2f}"]
                    lines.append(
                        f"| {o} | {label[s]} | {name} | {b:.0%} | {1000 * t['gain']:+.2f}"
                        f" [{1000 * t['low']:+.2f}, {1000 * t['high']:+.2f}] "
                        f"| {1000 * te['gain']:+.2f} [{1000 * te['low']:+.2f}, {1000 * te['high']:+.2f}] |"
                    )
    lines += [
        "",
        "## 5. Average effect on the test half: raw vs adjusted",
        "",
        "Estimated treatment probability on the test half: 1st percentile"
        f" {r['propensity']['p1']:.3f}, median "
        f"{r['propensity']['p50']:.3f}, 99th percentile {r['propensity']['p99']:.3f}.",
        "",
        "| Outcome | Raw difference | AIPW, constant probability (full-size outcome models) | AIPW,"
        " estimated probabilities |",
        "|---|---:|---:|---:|",
    ]
    for o in OUTCOMES:
        a = r["ate"][f"full|{o}"]
        lines.append(
            f"| {o} | {pp(a['raw'], 4)} | {pp(a['aipw_constant'], 4)} | {pp(a['aipw_estimated'], 4)} |"
        )
    return [*lines, "", *INTERPRETATION]


def main(from_cache: bool) -> None:
    if from_cache:
        results = json.loads(RESULTS.read_text())
    else:
        results = main_run()
        RESULTS.write_text(json.dumps(results, indent=2))
    lines = render(results)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
