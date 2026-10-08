# Phase 4 design note: heterogeneous effects, uplift models, and targeting

**Status: written 2026-10-07, before any Phase 4 model was fit; binding once committed.** It
makes section 9 of the analysis plan specific. Where this note is more specific than the plan,
the note governs; any later change is logged in section 12 of the plan. The Phase 1b estimates
remain the confirmatory results.

## 1. Questions

1. Do the e-mails' effects on spend vary across customers in a way a model can detect, and does
   that hold up on held-out data?
2. Does sending each customer the e-mail a model predicts is best beat sending everyone the men's
   e-mail, the best single e-mail in Phase 1b?
3. The challenge's own question: with a budget of 10,000 e-mails, who should get them, and how
   much is targeting worth over picking 10,000 at random?
4. Which pre-specified customer groups respond differently?

**An honest prior.** Phase 2 found the covariates explain under 1% of spend's variance. That
limits how precisely any model can tell responders apart, so "no detectable heterogeneity in
spend" is a live outcome, and it will be reported as such. Visits, which the covariates predict
better, are analyzed alongside so that a weak spend result can be read correctly.

## 2. Data, split, and sealing

- **Split:** one generator seeded 2026 shuffles each arm's rows (in file order, arms taken in the
  order No E-Mail, Mens E-Mail, Womens E-Mail); the first half of each arm, rounded down, is the
  training half and the rest the test half. That gives 31,999 training and 32,001 test customers.
- **Sealing:** the test half stays unread until the frozen models and their validation are
  committed (section 8). Phase 4a code loads the training half only, through a loader that
  returns test rows only when asked by name, as the Phase 0 loader did for outcomes.
- **Assignment probabilities:** the stratified split fixes each arm's count within each half, so
  the exact probability of being in an arm, within a half, is that arm's share of the half
  (within 0.3% of 1/3). Those shares are used wherever the plan says "known propensities".
- **Features:** recency, history, mens, womens, newbie, zip_code and channel (0/1 dummies).
  `history_segment` is excluded as a binned copy of `history`. The leakage guard
  (`require_pre_treatment`) runs before any model is fit.
- **Outcomes:** spend (primary; the targeting policy optimizes it) and visit (alongside).

## 3. Models (training half only)

Each model predicts two effects per customer: men's e-mail vs no e-mail and women's e-mail vs no
e-mail.

- **T-learner (baseline):** one gradient-boosted regression per arm (scikit-learn
  `HistGradientBoostingRegressor`, squared error). Effect = difference of the two predictions.
- **DR-learner (primary; own implementation):** 5-fold cross-fitting inside the training half
  (fold assignment seeded 202640). For each fold, the per-arm outcome models are fit on the other
  folds, and each customer gets a doubly robust score for each contrast a vs control, with e the
  arm shares from section 2:
  psi_a = m_a(x) - m_c(x) + 1[A = a] (Y - m_a(x)) / e_a - 1[A = control] (Y - m_c(x)) / e_c.
  A second-stage gradient-boosted regression of psi_a on x is the effect. These are the AIPW
  scores of the political-corruption DML project. Because the propensities are known by design,
  the scores are unbiased for the conditional effect however wrong the outcome models are.
- **Causal forest (cross-check):** EconML 0.17 `CausalForestDML`, three discrete treatments,
  propensities fixed at the arm shares by a constant classifier, gradient-boosted outcome model,
  1,000 trees, honest splitting, `min_samples_leaf` 50, `random_state` 2026. Not tuned.
- **Hyperparameters** for the T-learner's outcome models, the DR-learner's outcome models, and the
  DR-learner's second stage: 5-fold cross-validation inside the training half over the grid
  learning rate {0.03, 0.1} x max leaf nodes {7, 31} x min samples per leaf {50, 200} x L2
  penalty {0, 1}, with 200 boosting iterations, no early stopping, `random_state` 2026. Outcome
  models are selected by mean squared error; the second stage by mean squared error against psi
  (the DR loss).
- **A constant is a candidate for the second stage.** If cross-validation prefers predicting the
  same effect for everyone, the DR-learner does so, its policy reduces to the best single e-mail,
  and that is reported, not overridden.
- **The primary model is fixed now:** the DR-learner's predictions define the targeting policy.
  The T-learner and the causal forest are evaluated alongside. No model is chosen by its
  test-half performance. The X-learner is not fit; the DR-learner covers the same ground with
  better guarantees when propensities are known.

## 4. Validation before the test half is opened (Phase 4a)

Plasmode simulations built from the training half's control customers (covariates and spend),
using the frozen hyperparameters, with the real arm sizes and the same split procedure:

- **Constant effects:** the pseudo-men's arm +118% and the pseudo-women's arm +65% (the observed
  lifts), every non-buyer equally likely to convert. The calibration test (section 5) should
  reject about 5% of the time.
- **Heterogeneous effects:** the same average lifts, but a non-buyer's chance of converting is
  proportional to (1 + 3 x bought men's merchandise) x history for the men's e-mail, and
  (1 + 3 x bought women's merchandise) x history for the women's e-mail. Previous buyers of a
  category are four times as responsive to that category's e-mail. This measures the test's power
  and the policy's true gain over the best single e-mail.
- 200 simulated experiments per scenario (seeds 202641 and 202642), DR-learner only, with these
  outputs:
  - the calibration test's false-positive rate and power
  - the bias of the inverse-propensity policy value against its exact truth
  - the coverage of its 95% interval (1,000 bootstrap resamples inside each simulation)
- If a method misbehaves, it is fixed and re-validated before the test half is opened, and the
  change is logged.

## 5. Test-half evaluation (Phase 4b)

Each frozen model is applied to the test half once. All metrics below are computed for each
learner, each outcome, and each e-mail vs no e-mail, unless a bullet says otherwise.

- **Calibration, the heterogeneity test:** the best-linear-predictor test of Chernozhukov,
  Demirer, Duflo and Fernández-Val. On the test half, AIPW scores (outcome models trained on the
  whole training half) are regressed on an intercept and the centered predicted effect, with HC2
  standard errors. The intercept estimates the average effect. The slope is 1 if the predictions
  are calibrated and 0 if they capture no real heterogeneity. The test is slope > 0, one-sided at
  0.05, since a model whose predictions run against the truth has failed just as surely.
  **Primary:** the DR-learner on spend, Holm-corrected across the two e-mails. Everything else is
  exploratory.
- **Uplift curves:** test customers in the e-mail's arm and the control arm are ranked by
  predicted effect, ties broken in a seeded random order (seed 202643). For each top share f, in
  1% steps:
  gain G(f) = f x (mean outcome of the e-mailed - mean outcome of the controls, within that top
  share).
  Random targeting gives f x G(1). Summary: the average gap between the curve and the random line
  (a Qini-type area, in outcome units per customer), with a 95% bootstrap interval. The interval
  uses 2,000 resamples of test customers within arm, with the model held fixed (seed 202644).
- **Policy value (spend, DR-learner primary):**
  - Policy: each test customer gets the e-mail with the larger predicted spend effect, or no
    e-mail if both predictions are at or below zero.
  - Value: inverse-propensity weighting with the arm shares: the sum, over arms, of the outcomes
    of that arm's customers whom the policy assigns to that arm, divided by the arm's size.
  - Benchmarks: everyone gets the men's e-mail; everyone gets the women's e-mail; no one is
    e-mailed; random assignment.
  - Headline: DR policy minus the all-men's policy, with a 95% paired bootstrap interval (10,000
    resamples, seed 202644), in dollars per 1,000 customers.
- **The 10,000-e-mail question (DR-learner):**
  - Policy: the 15.625% of test customers (10,000 of 64,000) with the largest predicted
    best-e-mail effect get that e-mail; the rest get none.
  - Value: its incremental revenue over e-mailing no one, scaled to 10,000 e-mails, against
    sending the men's e-mail to a random 10,000, with a 95% paired bootstrap interval.
  - Who they are: described by scoring all 64,000 customers with the frozen model.

## 6. Pre-specified moderators (Phase 4b, all 64,000 customers)

- For each e-mail vs no e-mail, a test of whether its effect varies with each of four moderators:
  - prior purchase category: men's only, women's only, or both (2 degrees of freedom)
  - recency (linear)
  - newbie
  - channel (2 degrees of freedom)
- Wald tests from OLS with arm-by-moderator interactions and HC2 standard errors.
- Two families of 8 tests, one for spend and one for visit. Benjamini-Hochberg runs within each
  family, and all results are exploratory. Effects within each purchase category are reported
  with intervals.
- No model selection is involved, so these use all customers. They run only after the models are
  frozen, so they cannot feed back into them.

## 7. Defaults chosen here, not dictated by the plan

- DR-learner as primary, with the T-learner and the causal forest alongside; the X-learner not
  fit.
- The hyperparameter grid, and a constant as a second-stage candidate.
- The calibration test, which the plan does not name.
- The validation scenarios.
- The arm shares as exact propensities.
- The budget scaled to the test half.

## 8. Commit points

1. This note, before any model is fit.
2. Phase 4a: hyperparameter selection, the frozen configuration, and the validation results,
   committed before the test half is opened.
3. Phase 4b: the test-half evaluation and the moderators.

The Criteo scale-up gets its own note after Phase 4b.
