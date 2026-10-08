# experiment-uplift-toolkit

A/B testing done carefully, on a real randomized experiment: 64,000 customers
randomized to a men's e-mail, a women's e-mail, or no e-mail (Hillstrom, 2008).
The deliverable is a small tested package, `exptools`, plus a decision memo.
Every method is validated by simulation before it is trusted on real data.

**Status: Phase 3 complete.** The analysis plan was pre-registered in
[`prereg/analysis_plan.md`](prereg/analysis_plan.md) before any outcome was
read, and the randomization checks pass
([`reports/phase0_randomization_checks.md`](reports/phase0_randomization_checks.md)).
Every inference method was validated on 4,000 simulated experiments built from
real control-arm customers before any comparison between real arms
([`reports/phase1a_method_validation.md`](reports/phase1a_method_validation.md)).
The pre-registered analysis then found that both e-mails increase spend per
customer: the men's e-mail by $0.77 (+118%) and the women's by $0.42 (+65%), both
significant after the Dunnett correction
([`reports/phase1b_analysis.md`](reports/phase1b_analysis.md)).
Adjusting for pre-treatment covariates then removed almost none of the noise in these estimates:
CUPED removes 0.04% of the spend variance and full regression adjustment at most
0.56%, because two-week spend is 99% zeros and barely correlated with prior-year
spend. Adjusting for a post-treatment variable instead would have erased 71% of
the men's e-mail effect
([`reports/phase2_variance_reduction.md`](reports/phase2_variance_reduction.md)).
Simulation then showed that checking results daily and stopping at the first
significant look turns a 5% false-positive rate into 21%. A group-sequential
design with O'Brien-Fleming-type alpha spending restores 4.9%, costs about one
point of power, and stops a men's-sized effect on day 6 on average instead of
day 14 ([`reports/phase3_power_and_peeking.md`](reports/phase3_power_and_peeking.md)).

## Setup

```bash
uv sync
uv run python scripts/download_data.py      # fetches the CSV, verifies its SHA-256
uv run python scripts/check_randomization.py
uv run python scripts/validate_methods.py   # about 70 s on 8 cores
uv run python scripts/analyze_experiment.py
uv run python scripts/adjust_variance.py    # about 2 min on 8 cores
uv run python scripts/sequential_peeking.py # about 30 s
uv run python scripts/uplift_phase4a.py     # tuning, freezing, validation: about 55 min
uv run pytest
```

## Layout

| Path | What it holds |
|---|---|
| `src/exptools/data.py` | Download, checksum, and a loader that hides outcome columns by default |
| `src/exptools/srm.py` | Sample-ratio-mismatch test |
| `src/exptools/balance.py` | Standardized differences and a joint balance test |
| `src/exptools/bootstrap.py` | Exact compressed bootstrap and BCa intervals |
| `src/exptools/inference.py` | Welch and BCa comparisons of two arms |
| `src/exptools/multiplicity.py` | Dunnett (bootstrap max-T and parametric) and Benjamini-Hochberg |
| `src/exptools/simulate.py` | Plasmode simulations that check methods against a known truth |
| `src/exptools/adjust.py` | CUPED, Lin's regression adjustment, HC2 standard errors |
| `src/exptools/simulate_adjustment.py` | Plasmode simulations for the covariate adjustments |
| `src/exptools/power.py` | Minimum detectable effects and sample sizes, including buyer-driven variance |
| `src/exptools/sequential.py` | Alpha spending, group-sequential boundaries, exact crossing probabilities |
| `src/exptools/simulate_sequential.py` | Peeking simulations on spend-shaped data |
| `src/exptools/uplift.py` | T-, DR-learner and causal forest; calibration test, uplift curves, policy values, moderator tests |
| `src/exptools/simulate_uplift.py` | Plasmode validation of the uplift pipeline against a known truth |
| `prereg/` | The pre-registered analysis plan, its deviations log, and the Phase 3 and 4 design notes |
| `reports/` | Generated results |
| `tests/` | Known-answer tests plus simulation checks of each method's error rate |

## Phases

0. Repo, data, pre-registration, randomization checks (done)
1. Classical analysis: method validation on the control arm (1a), then Dunnett-corrected effects on spend with bootstrap intervals (1b) (done)
2. Variance reduction: CUPED and regression adjustment (done)
3. Power, A/A peeking simulation, sequential testing (done)
4. Heterogeneous effects, uplift models, targeting policy, Criteo scale-up
5. Package polish, CI, decision memo
