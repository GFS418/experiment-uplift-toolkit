# Pre-registered analysis plan: Hillstrom e-mail experiment

**Status: binding as of commit `b120b77` (2026-10-05).** The memo cites that
hash; every later edit is listed in the deviations log (section 12). No outcome column (`visit`,
`conversion`, `spend`) is loaded before then: the loader in `exptools.data`
excludes outcomes unless called with `load_outcomes=True`, and nothing in the
repository makes that call before this commit.

## 0. What this pre-registration can and cannot claim

Hillstrom is a famous public dataset (published in 2008 as a data-mining
challenge), and its headline results circulate widely in blog posts, course
material, and uplift-modeling library tutorials. **The analysis cannot be truly
blind**: anyone working with this data may have seen published numbers.

What pre-registration still buys: metrics, comparisons, corrections, interval
methods, covariates, and data splits are fixed in writing before any outcome
statistic is computed in this repository, so none of them can be tuned to the
results. Treat this as a demonstrated habit, not as evidence of independence
from prior knowledge.

## 1. The experiment

| | |
|---|---|
| Source | Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (March 2008) |
| File | `Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`, SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |
| Population | 64,000 customers who last purchased within the prior twelve months |
| Design | Individual randomization, 1/3 each: men's e-mail, women's e-mail, no e-mail (control) |
| Outcome window | The two weeks after the campaign |
| Pre-treatment covariates | recency, history_segment, history, mens, womens, zip_code, newbie, channel (all measured over the prior year) |

**Estimand.** The intent-to-treat average effect of *being sent* an e-mail,
versus no e-mail, on two-week outcomes. Delivery and opens are not observed, so
this is the effect of the send decision, which is also the decision the business
controls. The customer is both the unit of randomization and the unit of
analysis, so standard errors need no clustering.

**Analysis population.** All 64,000 randomized customers, in the arm they were
assigned to. No exclusions and no outlier removal.

## 2. Gate: pre-outcome checks

These use only assignment and pre-treatment columns. If either alarm fires,
effect estimates are not reported as causal until the cause is found.

- **Sample-ratio mismatch:** chi-square goodness-of-fit of arm sizes against
  1/3 each. Alarm at p < 0.001.
- **Covariate balance:** standardized mean differences with chance-scale
  z-scores, plus one joint test (multinomial logit of arm on all covariates,
  likelihood-ratio test). Alarm at p < 0.001 on the joint test.

Results: `reports/phase0_randomization_checks.md` (SRM p = 0.904; joint balance
p = 0.628; no alarms).

## 3. Metrics

| Role | Metric | Definition | Notes |
|---|---|---|---|
| **Primary** | Spend per customer | Mean two-week `spend` over all customers in the arm, zeros included | The business decision metric. Reported in dollars and as incremental dollars per 1,000 customers e-mailed. Mostly zeros with a heavy right tail. |
| Secondary | Conversion rate | Mean of `conversion` | Closest to how money is spent, but a **rare outcome with low power** (see section 6): a null result is weak evidence of no effect. |
| Secondary | Visit rate | Mean of `visit` | Well-powered leading indicator; a proxy, since visits do not pay. |

**Guardrails: none available.** A real e-mail test would track unsubscribes and
spam complaints; Hillstrom records neither. This is stated as a limitation
rather than replaced with an invented guardrail.

**Excluded by design:** metrics conditioned on a post-treatment event, such as
conversion among visitors. The e-mail changes who visits, so such ratios compare
different kinds of customers across arms (post-treatment selection bias).

**Spend is not trimmed or winsorized in the primary analysis.** Sensitivity
analysis (reported, not confirmatory): spend winsorized at its 99.9th percentile
computed over all arms pooled, so the cap does not depend on assignment.

## 4. Hypotheses and multiplicity

All tests two-sided.

### Primary (confirmatory) family: family-wise error rate 0.05, Dunnett

| | Hypothesis |
|---|---|
| H1 | Mean spend, men's e-mail = mean spend, control |
| H2 | Mean spend, women's e-mail = mean spend, control |

Correction: **Dunnett's many-to-one procedure**, single-step, giving adjusted
p-values and simultaneous 95% confidence intervals. Dunnett accounts for the
positive correlation between H1 and H2 created by their shared control arm.

**Implementation.** Primary: a studentized bootstrap max-T version
of Dunnett. Resample customers within each arm (B = 10,000), compute
T*_j = (d*_j - d_j) / se*_j for both comparisons, and use the 95th percentile of
max_j |T*_j| as the simultaneous critical value. The adjusted p-value for H_j is
the share of replicates with max|T*| >= |t_j|. Cross-check: `scipy.stats.dunnett`,
the parametric version, which assumes normal data with equal variances in every
arm. Spend violates the first assumption and probably the second. Both are
reported.

**Decision rule (pre-committed).** An e-mail version is declared to increase
spend if its Dunnett-adjusted p < 0.05 and its estimate is positive. The memo
leads with this family.

### Secondary (exploratory) family: false discovery rate 0.05, Benjamini-Hochberg

| Metric | Comparisons |
|---|---|
| Visit rate | men's vs control, women's vs control, men's vs women's |
| Conversion rate | men's vs control, women's vs control, men's vs women's |
| Spend per customer | men's vs women's |

Seven hypotheses. P-values from Welch's t-test (for 0/1 metrics at this sample
size, equivalent to the two-proportion z-test). Every exploratory result is
labeled as such in tables, figures, and the memo, and is treated as a lead for
a future test, not a finding.

**Men's vs women's on spend is exploratory.** Dunnett covers
comparisons with the control only, so the head-to-head comparison of the two
e-mails moves to this family. The question of which e-mail to send to whom is
answered properly in section 9 on held-out data.

## 5. Estimation and intervals

- **Point estimates:** difference in arm means, reported in absolute units,
  as relative lift, and as dollars per 1,000 customers.
- **Marginal 95% intervals, every comparison:** BCa bootstrap, B = 10,000,
  resampling customers within arms, seed 2026. Cross-check: Welch interval.
- **Simultaneous 95% intervals, primary family:** from the Dunnett procedure in
  section 4.
- **Validation before use.** Before any treatment arm's outcomes are read,
  plasmode simulations built from the control arm alone check the methods on
  data shaped like the real spend:
  - A/A: two pseudo-arms resampled from control at the real arm sizes (true
    difference zero), giving null coverage of the BCa and Welch intervals and
    the family-wise error rate of both Dunnett implementations.
  - Known effect: a pseudo-treatment arm with extra conversions injected at a
    known rate, spend drawn from control's nonzero spend, so the true
    difference is known.
  - 2,000 simulated experiments per scenario, with Monte Carlo standard errors
    (about 0.5 percentage points at 95% coverage). B may drop to 2,000 inside
    the simulation for compute; that will be stated.
- **Under-coverage is reported, whatever it shows.** "Under-coverage" means the
  Monte Carlo 95% interval for coverage lies entirely below 0.95.
  **Fallback:** if BCa under-covers and Welch does not, Welch
  becomes the headline interval and both are still shown. The switch is
  decided by the simulation, before the real comparisons are computed.

## 6. Power and minimum detectable effects (computed before outcomes)

About 21,300 customers per arm, alpha 0.05 two-sided, 80% power.

| Metric | Minimum detectable effect |
|---|---|
| Any metric, in standard-deviation units | 0.027 SD unadjusted; 0.030 SD at the Dunnett critical value of 2.21 |
| 0/1 metric, assumed baseline 0.5% | 0.19 percentage points (38% relative) |
| 0/1 metric, assumed baseline 1% | 0.27 percentage points (27% relative) |
| 0/1 metric, assumed baseline 10% | 0.81 percentage points (8% relative) |

Baseline rates are assumptions, since outcomes are unread; the Benjamini-Hochberg
thresholds make the exploratory MDEs larger still. The table quantifies why
conversion has low power: unless an e-mail lifts conversion by roughly 30% or
more, a non-significant conversion result says little. The spend MDE in dollars
depends on the spend standard deviation, which is unknown before outcomes; the
Phase 3 calculator uses the observed variance. Post-hoc "observed power" is not
reported, because it is a function of the p-value and adds no information.

## 7. Variance reduction (Phase 2)

The confirmatory test stays the unadjusted difference in means. Adjusted
estimators are pre-specified secondary analyses that quantify variance reduction.

- **CUPED:** one covariate, `history` (prior-year spend), theta estimated pooled
  across arms.
- **Regression adjustment:** Lin (2013) fully interacted OLS, with outcome on arm
  indicators, centered covariates (recency, history, mens, womens, newbie,
  zip_code, channel), and arm-by-covariate interactions, using HC2 robust
  standard errors. `history_segment` is excluded because it is a binned copy of
  `history`.
- **Reported:** variance reduction, 1 - Var(adjusted) / Var(unadjusted), per
  metric and comparison, with point estimates shown next to the unadjusted ones.
- **Leakage rule:** only the eight pre-treatment columns may enter any
  adjustment or model.

## 8. Peeking and sequential testing (Phase 3)

Hillstrom has no timestamps, so interim looks at it are impossible and the
confirmatory analysis is fixed-horizon. Phase 3 is simulation work (A/A peeking,
then a group-sequential or always-valid fix), specified in its own short design
note before it runs. Nothing in Phase 3 changes the confirmatory analysis.

## 9. Heterogeneous effects and targeting (Phase 4)

- **Hold-out:** one split, 50% train and 50% test, stratified by arm, seed
  2026, drawn before any model is fit. Model selection uses cross-validation
  inside the training half; each final model is scored on the test half once.
- **Pre-specified moderators** (exploratory, Benjamini-Hochberg across them):
  prior purchase category (men's only, women's only, both) crossed with e-mail
  version (the "send what they already buy" hypothesis), recency, newbie,
  channel.
- **Uplift models:** meta-learners (T-, X-, DR-learner) or a causal forest,
  using pre-treatment features only. Evaluated with Qini and uplift curves on
  the test half, with bootstrap intervals.
- **Targeting policy:** each test customer gets the arm with the highest
  predicted spend uplift. Policy value is estimated on the test half by
  inverse-propensity weighting with the known assignment probabilities of 1/3.
  Benchmarks: everyone gets the men's e-mail, everyone gets the women's e-mail,
  and a random assignment. The challenge's own question is answered too: with a
  budget of 10,000 e-mails, which customers?
- **Criteo:** scale-up of the uplift-modeling step only, specified in its own
  design note.

## 10. Practical significance

Effects are translated into incremental revenue per 1,000 customers e-mailed.
No cost or margin data exists, so results are revenue, not profit. The effect
per customer is reported as a revenue break-even: the most an e-mail could cost
before it loses money, an upper bound once margins are considered.

## 11. Reproducibility

Python 3.12, dependencies locked in `uv.lock`, seeds fixed, and every method
in `exptools` tested, including simulation checks of its error rates.

## 12. Deviations log

Any change after the plan is committed is recorded here with its date, what
changed, and why. Analyses not described above are labeled post hoc wherever
they appear.

| Date | Change | Reason |
|---|---|---|
| 2026-10-05 | The three items marked [TO CONFIRM] (max-T Dunnett as primary, men's vs women's spend in the exploratory family, the Welch fallback rule) were committed unchanged in `b120b77` and are adopted exactly as written. Markers removed; status line updated. | Clarification only, no change in content. Made before any treatment-arm outcome was read. |
| 2026-10-05 | Section 5 left the injected effect's size open ("a known rate"). Phase 1a set it to a +50% lift in mean spend, delivered as extra buyers. | Chosen from control-arm data alone, as the size closest to the minimum detectable effect. Made before any treatment-arm outcome was read. |
| 2026-10-05 | Added validation checks not listed in section 5: Welch false-positive rates and BCa coverage for the seven exploratory hypotheses, Benjamini-Hochberg's error rate under the complete null, and the coverage of each Dunnett interval on its own. | Extra checks only; no analysis changed. Results in `reports/phase1a_method_validation.md`. |
| 2026-10-05 | Section 5 reports relative lift as a point estimate without naming an interval. Phase 1b gives it a 95% BCa interval on the ratio of arm means (exact jackknife acceleration, same B = 10,000 resamples and seed). | Specification of an unspecified detail, written before the Phase 1b script first read treatment-arm outcomes. |
| 2026-10-05 | Section 7 details left open, fixed before the Phase 2 script first ran: CUPED's theta is the within-arm pooled slope of the outcome on `history` (identical to the ANCOVA coefficient), its standard error is Welch's on the adjusted outcomes with theta treated as known, and adjusted estimators get normal-theory 95% intervals. Lin's covariates enter as numeric columns plus 0/1 dummies for `zip_code` and `channel` (first level dropped), centered at the full-sample mean. | Specification of unspecified details; the validation below checks each one. |
| 2026-10-05 | Phase 2 validation design: plasmode simulations from control customers with their covariates (2,000 per scenario): a complete null, and a known effect in the pseudo-men's arm (+100% mean spend through new buyers whose chance grows with prior-year spend, plus 8% of remaining non-visitors starting to visit). | Implements the brief's "CUPED unbiased" check; the history-dependent effect is where CUPED and Lin should differ. |
| 2026-10-05 | Added a counterexample: CUPED with the post-treatment `visit` as covariate, in the simulation and on the real data. It is labeled invalid and never used for any estimate. | Demonstrates what the leakage rule in section 7 prevents. |
| 2026-10-07 | Phase 3 report: added a post-run diagnostic (one look's false-positive rate at each schedule's first-look sample size, 20,000 draws each, seed 202636), not in the Phase 3 design note. | Explains why rare purchases dampen peeking inflation at very frequent looks. Labeled post hoc in the report. |
