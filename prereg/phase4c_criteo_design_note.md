# Phase 4c design note: the uplift procedure at scale, on Criteo

**Status: written 2026-10-08, before the Criteo data was downloaded; binding once committed.**
Section 9 of the analysis plan requires this note: the Criteo work is a scale-up of the
uplift-modeling step only. Any later change is logged in section 12 of the plan.

## 1. The question

On Hillstrom, the Phase 4 procedure found no spend heterogeneity worth acting on. The power
check showed it would have missed even a planted fourfold difference in responsiveness most of
the time. Phase 4c asks the natural next question:

**Does the same procedure detect heterogeneous effects, and does targeting beat treating at
random, when the training set is up to about 200 times larger?**

The answer is a learning curve. The same procedure is run at four training sizes, from
Hillstrom's size to the full Criteo training half, and every model is scored on one fixed
test half.

**What this can and cannot claim.** Criteo is a public benchmark with published results, and
for privacy its data were "sub-sampled non-uniformly so that the original incrementality level
cannot be deduced". The effects here are therefore not Criteo's real ad incrementality, and the
anonymized features cannot be interpreted. The results describe how the methods behave at
scale, not anything about Criteo's advertising.

## 2. Data

- **Source:** `criteo-research-uplift-v2.1.csv.gz` from Criteo's own Hugging Face organization
  (huggingface.co/datasets/criteo/criteo-uplift): 311,422,618 bytes, SHA-256
  `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`, as published by the host.
  The download URL on Criteo's dataset page now redirects to a storage host that no longer
  exists.
- **License:** Creative Commons BY-NC-SA 4.0. Non-commercial use, with attribution to Diemert,
  Betlei, Renaudin and Amini (2018), "A Large Scale Benchmark for Uplift Modeling" (AdKDD). The
  repository ships a downloader, never the data.
- **Version:** v2.1, the "unbiased" release, with 13,979,592 rows. The original 25-million-row
  release had a leak: incrementality varied by advertiser and the features could identify
  advertisers. A row count other than 13,979,592 stops the pipeline.
- **Columns:**
  - 12 anonymized float features, f0 to f11.
  - `treatment` (1 = eligible for targeting, 0 = held out), with about 85% treated.
  - Outcomes `visit` (4.70% on average) and `conversion` (0.29%).
  - `exposure`, whether an ad was actually shown. **Exposure is post-treatment** (only treated
    users can be exposed) and is never used as a feature. The leakage guard enforces this.

## 3. Checks before any model (Phase 4c-a)

- **Integrity:** checksum, row count, the 16 columns, and the rates against the dataset card's key
  figures. Any deviation is reported.
- **Balance:** standardized mean differences of the 12 features between treatment and control,
  plus a joint likelihood-ratio test (logistic regression of treatment on the features). With 14
  million rows the joint test can flag differences far too small to matter, so it is read
  together with the SMDs. Imbalance would point to the non-uniform subsampling and is reported as
  a caveat, not fixed.
- **Average effects on the training half only:** treatment minus control for visit and
  conversion, with intervals. These sanity checks need no test-half data.

## 4. Split and training sizes

- **Split:** stratified by treatment, one generator seeded 2026: each group's rows are shuffled
  and the first half (rounded down) is the training half. That gives about 6.99 million customers
  in each half. The test half stays sealed until the models are frozen (section 8).
- **Training sizes:** n = 32,000 (Hillstrom's training-half size), 320,000, 3,200,000, and the
  full training half. The sizes are nested: one seeded permutation of the training half
  (seed 202651), stratified by treatment, and each size takes its first n customers.
- **Propensities:** the treatment share within each training sample, and within the test half,
  exactly as in Phase 4.

## 5. Models (identical procedure at every size)

- **Learners:**
  - T-learner and DR-learner, as in Phase 4: AIPW scores with the exact treatment share,
    5-fold cross-fitting (folds seeded 202652).
  - EconML's causal forest, at 32,000 and 320,000 only, as a cross-check. Beyond that its
    1,000 trees outgrow memory, and it adds nothing the other two learners do not.
- **Tuning:** the Phase 4 grid at every size, selected by 5-fold cross-validation inside the
  training sample. The grid is learning rate {0.03, 0.1} x max leaf nodes {7, 31} x min samples
  per leaf {50, 200} x L2 {0, 1}, with 200 iterations. Outcome models are selected by mean squared
  error, and the DR second stage by the DR loss, with a constant as a candidate.
- **Timing:** a fit takes 5 to 7 seconds at 5.6 million rows, measured on synthetic data of the
  same shape before writing this note. The full procedure at every size is about two hours of
  compute.
- **Freezing:** every final model is fit twice and fingerprinted, as in Phase 4a. Phase 4c-b
  refits and must reproduce each fingerprint before it scores the test half.

## 6. A/A negative control (Phase 4c-a, test half still sealed)

A real-data check that the pipeline does not manufacture heterogeneity:

- Take the training half's treated customers, so that everyone truly received treatment.
- Assign a random pseudo-treatment at the real 85% share, and split them 50/50 into
  pseudo-training and pseudo-test halves.
- Run the T- and DR-learner with each size's frozen hyperparameters (seed 202655).
- Since every true pseudo-effect is zero, the calibration test should fire about 5% of the
  time. Run 20 repetitions at 32,000 and at 320,000 and 5 at 3,200,000.
- This is a coarse check that can catch gross miscalibration, not a precise size estimate.
  Phase 4a's simulations already measured the test's size on Hillstrom-shaped data.

## 7. Test-half evaluation (Phase 4c-b)

For each learner, training size and outcome (visit and conversion):

- **Calibration test,** as in Phase 4: AIPW scores on the test half, from outcome models
  trained on the training sample, regressed on the centered predicted effect with HC2 standard
  errors. One-sided test of slope > 0.
- **Uplift curve:** the gain from treating the top share f by predicted effect, in 1% steps,
  with ties broken in a seeded order (seed 202653).
  - Summary: the Qini-type area, as in Phase 4.
  - Interval: 95% bootstrap with 2,000 resamples. Each resample redraws customers within arm
    exactly, as multinomial counts over the (percentile bin, outcome) cells; outcomes are 0/1,
    so this is the exact bootstrap without re-sorting 7 million rows (seed 202654).
- **Targeting value:** treat the top 10%, 25% or 50% of customers by predicted effect, against
  treating the same share at random. Values come from inverse-propensity weighting with the
  exact shares, with 95% paired bootstrap intervals (10,000 resamples).

**Primary hypotheses:** at the full training size, the DR-learner's predicted effects on visit
and on conversion are calibrated with slope > 0, one-sided at 0.05, Holm-corrected across the
two outcomes. If cross-validation picks a constant second stage at full size, the test does
not apply, and that is reported as the answer. Everything else is exploratory, including the
learning curve across sizes, the T-learner, the causal forest, uplift areas and targeting
values.

## 8. Commit points

1. This note, before the data is downloaded.
2. **Phase 4c-a:** download and checks, training-half average effects, tuning and freezing at
   every size, and the A/A control. All of it is committed before the test half is opened.
3. **Phase 4c-b:** the test-half evaluation.

## 9. Not included, and why

- **A planted-heterogeneity power study at full scale.** Each replicate is a full pipeline run,
  so meaningful power estimates would take many hours. The learning curve on real data, plus
  the A/A control, answers the question asked here. It can be added on request.
- **Anything using `exposure`,** such as an instrumental-variable estimate of the effect of
  actually seeing an ad. It is a different question.
- **Descriptions of who gets targeted.** The features are anonymized random projections.
