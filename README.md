# experiment-uplift-toolkit

A/B testing done carefully, on a real randomized experiment: 64,000 customers
randomized to a men's e-mail, a women's e-mail, or no e-mail (Hillstrom, 2008).
The deliverable is a small tested package, `exptools`, plus a decision memo.
Every method is validated by simulation before it is trusted on real data.

**Status: Phase 0.** The analysis plan is pre-registered in
[`prereg/analysis_plan.md`](prereg/analysis_plan.md) before any outcome is
read. The randomization checks pass
([`reports/phase0_randomization_checks.md`](reports/phase0_randomization_checks.md)).

## Setup

```bash
uv sync
uv run python scripts/download_data.py      # fetches the CSV, verifies its SHA-256
uv run python scripts/check_randomization.py
uv run pytest
```

## Layout

| Path | What it holds |
|---|---|
| `src/exptools/data.py` | Download, checksum, and a loader that hides outcome columns by default |
| `src/exptools/srm.py` | Sample-ratio-mismatch test |
| `src/exptools/balance.py` | Standardized differences and a joint balance test |
| `prereg/` | The pre-registered analysis plan and its deviations log |
| `reports/` | Generated results |
| `tests/` | Known-answer tests plus simulation checks of each method's error rate |

## Phases

0. Repo, data, pre-registration, randomization checks (done)
1. Classical analysis: Dunnett-corrected effects on spend, bootstrap intervals
2. Variance reduction: CUPED and regression adjustment
3. Power, A/A peeking simulation, sequential testing
4. Heterogeneous effects, uplift models, targeting policy, Criteo scale-up
5. Package polish, CI, decision memo
