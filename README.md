# experiment-uplift-toolkit

A/B testing done carefully, on a real randomized experiment: 64,000 customers
randomized to a men's e-mail, a women's e-mail, or no e-mail (Hillstrom, 2008).
The deliverable is a small tested package, `exptools`, plus a decision memo.
Every method is validated by simulation before it is trusted on real data.

**Status: Phase 1 complete.** The analysis plan was pre-registered in
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

## Setup

```bash
uv sync
uv run python scripts/download_data.py      # fetches the CSV, verifies its SHA-256
uv run python scripts/check_randomization.py
uv run python scripts/validate_methods.py   # about 70 s on 8 cores
uv run python scripts/analyze_experiment.py
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
| `prereg/` | The pre-registered analysis plan and its deviations log |
| `reports/` | Generated results |
| `tests/` | Known-answer tests plus simulation checks of each method's error rate |

## Phases

0. Repo, data, pre-registration, randomization checks (done)
1. Classical analysis: method validation on the control arm (1a), then Dunnett-corrected effects on spend with bootstrap intervals (1b) (done)
2. Variance reduction: CUPED and regression adjustment
3. Power, A/A peeking simulation, sequential testing
4. Heterogeneous effects, uplift models, targeting policy, Criteo scale-up
5. Package polish, CI, decision memo
