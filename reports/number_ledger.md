# Number ledger

Every headline figure, quoted exactly as it appears in its report and section.
`tests/test_number_ledger.py` fails if any quoted value is missing from the cited
section, so a number cited by ledger ID is a number someone can check.

| ID | Figure | Value as written in the source | Source (reports/) | Section |
|---|---|---|---|---|
| H0-1 | Sample-ratio-mismatch test, arm sizes vs 1/3 each | `p = 0.904` | phase0_randomization_checks.md | Sample-ratio mismatch |
| H0-2 | Joint covariate balance test | `p = 0.628` | phase0_randomization_checks.md | Joint balance test |
| H1a-1 | Control-arm buyers, of 21,306 customers | `Buyers \| 122` | phase1a_method_validation.md | The data the methods must handle |
| H1a-2 | Skewness of spend, control arm | `Skewness of spend \| 26.4` | phase1a_method_validation.md | The data the methods must handle |
| H1a-3 | Spend MDE at 80% power, Dunnett cutoff | `$0.343 per customer ($343 per 1,000), 53% of control's mean` | phase1a_method_validation.md | The data the methods must handle |
| H1a-4 | Family-wise error of the max-T Dunnett, no effect | `Bootstrap max-T \| 2.206 \| 0.048 +/-0.009` | phase1a_method_validation.md | Scenario: complete null |
| H1a-5 | Benjamini-Hochberg error rate, 7 true nulls | `0.049 +/-0.009` | phase1a_method_validation.md | Scenario: complete null |
| H1a-6 | Power at a true +50% spend lift (max-T) | `Bootstrap max-T \| 2.214 \| 0.022 +/-0.006 \| 0.653` | phase1a_method_validation.md | Scenario: known effect |
| H1b-1 | Men's e-mail on spend: effect, per 1,000, lift, marginal CI, simultaneous CI, adjusted p | `men's e-mail \| $0.770 \| $770 \| +118% [+63%, +191%] \| [$0.495, $1.061] \| [$0.446, $1.094] \| < 0.0001` | phase1b_analysis.md | 3. Primary result |
| H1b-2 | Women's e-mail on spend, same columns | `women's e-mail \| $0.424 \| $424 \| +65% [+22%, +125%] \| [$0.176, $0.698] \| [$0.134, $0.715] \| 0.0023` | phase1b_analysis.md | 3. Primary result |
| H1b-3 | Parametric Dunnett cross-check, women's: simultaneous CI and p | `[$0.103, $0.746] \| 0.0068` | phase1b_analysis.md | 3. Primary result |
| H1b-4 | Men's vs women's e-mail on spend (exploratory) | `$0.345 \| $345 \| +32% [+2%, +70%] \| [$0.035, $0.663] \| 0.0305` | phase1b_analysis.md | 4. Exploratory family |
| H1b-5 | Winsorized spend: change in the men's effect | `-14.4%` | phase1b_analysis.md | 5. Sensitivity |
| H1b-6 | Winsorized spend: change in the women's effect | `-9.8%` | phase1b_analysis.md | 5. Sensitivity |
| H2-1 | Within-arm correlation of spend with prior-year spend | `Within-arm correlation with prior-year spend: 0.022` | phase2_variance_reduction.md | 2. The real experiment |
| H2-2 | Spend variance removed, men's vs control: CUPED, Lin | `0.04% \| 0.56%` | phase2_variance_reduction.md | 2. The real experiment |
| H2-3 | CUPED on post-treatment visit: men's spend effect and share removed | `+$0.770 \| +$0.223 \| 71%` | phase2_variance_reduction.md | 3. Counterexample |
| H2-4 | Same for the women's e-mail | `+$0.424 \| +$0.102 \| 76%` | phase2_variance_reduction.md | 3. Counterexample |
| H2-5 | CUPED on visit, simulated bias against the known truth | `-77.3%` | phase2_variance_reduction.md | 1. Validation against a known truth |
| H3-1 | False positives, daily peeking for 14 days | `**0.212 +/-0.006**` | phase3_power_and_peeking.md | 2. Peeking without a correction |
| H3-2 | False positives, 100 looks | `0.319 +/-0.006` | phase3_power_and_peeking.md | 2. Peeking without a correction |
| H3-3 | False positives, O'Brien-Fleming design | `false positives 0.049 +/-0.003 with the group-sequential boundaries` | phase3_power_and_peeking.md | 3. The fix |
| H3-4 | +65% effect: power single vs sequential, expected days, stopped by day 7 | `0.914 / 0.907 \| 0.905 / 0.896 \| 9.5 / 9.6 \| 26%` | phase3_power_and_peeking.md | 3. The fix |
| H3-5 | +118% effect: expected days, stopped by day 7 | `6.2 / 6.2 \| 82%` | phase3_power_and_peeking.md | 3. The fix |
| H3-6 | Spend MDE: equal-variance vs buyer-driven formula | `$0.315 (+48%) \| $0.354 (+54%)` | phase3_power_and_peeking.md | 1. Power |
| H3-7 | At the equal-variance MDE: promised, buyer-driven, simulated power | `0.800 \| 0.711 \| 0.714 +/-0.009` | phase3_power_and_peeking.md | 1. Power |
| H3-8 | Buyer-driven prediction vs Phase 1a's simulated power at +50% | `buyer-driven formula 0.652; Phase 1a's simulation measured 0.653` | phase3_power_and_peeking.md | 1. Power |
| H3-9 | Customers per arm for +10% and +25% spend lifts (equal-variance / buyer-driven) | `494,656 / 519,303 \| 79,145 / 88,999` | phase3_power_and_peeking.md | 1. Power |
| H4a-1 | DR second stage, spend, men's: constant vs best tree loss | `constant (same effect for everyone) \| 1519.95 \| 1523.28 \| +0.219%` | phase4a_tuning_and_validation.md | 1. Tuning on the training half |
| H4a-2 | Planted 4x heterogeneity: tree picked, DR rejections, DR Holm, T rejections, T Holm | `heterogeneous effects \| 19% / 26% \| 0.025 / 0.020 \| 0.030 \| 0.300 / 0.065 \| 0.220` | phase4a_tuning_and_validation.md | 2b. Added check |
| H4a-3 | Constant effects: the same columns (false alarms) | `constant effects \| 19% / 22% \| 0.000 / 0.005 \| 0.000 \| 0.070 / 0.050 \| 0.035` | phase4a_tuning_and_validation.md | 2b. Added check |
| H4a-4 | Size of the planted heterogeneity, men's e-mail | `mean $0.738, SD $1.001, 10th to 90th percentile $0.052 to $1.953` | phase4a_tuning_and_validation.md | 2b. Added check |
| H4b-1 | Causal forest's targeted policy vs everyone getting the men's e-mail | `causal forest \| $1.443 \| [$1.119, $1.790] \| $75 \| [-$221, $384]` | phase4b_test_evaluation.md | 3. Policy value |
| H4b-2 | T-learner's targeted policy, same comparison | `T-learner \| $1.297 \| [$1.002, $1.614] \| -$71 \| [-$368, $230]` | phase4b_test_evaluation.md | 3. Policy value |
| H4b-3 | 10,000 e-mails: men's to a random 10,000, and the causal forest's targeting gain | `$6,879 [$3,137, $10,678] \| $119 [-$11,391, $12,480]` | phase4b_test_evaluation.md | 4. The 10,000-e-mail question |
| H4b-4 | Women's e-mail visit lift, women's-only buyers | `women's only \| women's e-mail \| 28,734 \| +7.40 pp` | phase4b_test_evaluation.md | 5. Pre-specified moderators |
| H4b-5 | Women's e-mail visit lift, men's-only buyers | `men's only \| women's e-mail \| 28,818 \| +1.11 pp` | phase4b_test_evaluation.md | 5. Pre-specified moderators |
| H4b-6 | Moderator test, purchase category x women's e-mail, visits | `97.34 (2) \| 7.3e-22` | phase4b_test_evaluation.md | 5. Pre-specified moderators |
| H4c-1 | Criteo rows, v2.1 | `13,979,592 rows (v2.1)` | phase4c_a_checks_tuning.md | 1. Data checks |
| H4c-2 | Largest feature imbalance, treated vs control | `Largest \|SMD\| 0.0488 (f3)` | phase4c_a_checks_tuning.md | 1. Data checks |
| H4c-3 | Joint balance test | `LR = 5,345.2 on 12 df` | phase4c_a_checks_tuning.md | 1. Data checks |
| H4c-4 | Raw training-half visit effect: estimate, CI, relative lift | `+1.026 pp \| [+0.985 pp, +1.067 pp] \| +26.8%` | phase4c_a_checks_tuning.md | 1. Data checks |
| H4c-5 | Full-size DR-learner's mean predicted visit effect | `6,989,795 \| DR-learner \| +0.719 pp` | phase4c_a_checks_tuning.md | 2. Tuning and freezing |
| H4c-6 | A/A negative control: rejections among 160 applicable tests | `7 rejected (4.4%)` | phase4c_a_checks_tuning.md | 4. Reading the results |
| H4c-7 | Primary test, visit: full-size DR calibration slope (SE) | `visit \| 1.09 (0.02)` | phase4c_b_test_evaluation.md | 1. Primary test |
| H4c-8 | Primary test, conversion | `conversion \| 0.85 (0.07)` | phase4c_b_test_evaluation.md | 1. Primary test |
| H4c-9 | Learning curve, visit: T-learner slope at 32,000 | `32,000 \| T-learner \| +4.861 pp \| 0.25 (0.01)` | phase4c_b_test_evaluation.md | 2. Learning curve |
| H4c-10 | Learning curve, visit: DR-learner slope at 320,000 | `320,000 \| DR-learner \| +2.260 pp \| 0.87 (0.02)` | phase4c_b_test_evaluation.md | 2. Learning curve |
| H4c-11 | Learning curve, visit: DR-learner slope at 3,200,000 | `3,200,000 \| DR-learner \| +2.184 pp \| 1.07 (0.02)` | phase4c_b_test_evaluation.md | 2. Learning curve |
| H4c-12 | Learning curve, visit: causal forest slope at 320,000 | `320,000 \| causal forest \| +1.947 pp \| 0.97 (0.02)` | phase4c_b_test_evaluation.md | 2. Learning curve |
| H4c-13 | Uplift area, full-size DR-learner, visit | `6,989,795 \| DR-learner \| +0.3727 pp` | phase4c_b_test_evaluation.md | 3. Uplift curves |
| H4c-14 | Top 25% vs random 25%, visits per 1,000: constant vs estimated probabilities | `25% \| +7.31 [+7.02, +7.60] \| +4.95 [+4.63, +5.27]` | phase4c_b_test_evaluation.md | 4. Targeting value |
| H4c-15 | Same for conversions | `25% \| +0.80 [+0.73, +0.87] \| +0.66 [+0.58, +0.73]` | phase4c_b_test_evaluation.md | 4. Targeting value |
| H4c-16 | Test-half visit effect: raw, AIPW constant, AIPW estimated | `+1.0425 pp \| +0.7628 pp \| +0.7740 pp` | phase4c_b_test_evaluation.md | 5. Average effect |
