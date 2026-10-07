# Phase 3 design note: power, peeking, and a group-sequential fix

**Status: written 2026-10-07, before any Phase 3 simulation ran on data shaped like Hillstrom's;
binding once committed.** Required by section 8 of the analysis plan. Phase 3 is simulation
only. Nothing in it changes the confirmatory analysis (Phase 1b).

## 1. Questions

1. How large an effect can this experiment detect, and how many customers would smaller
   effects need? (A power and minimum-detectable-effect calculator driven by the observed
   variance.)
2. How much does looking at the results repeatedly, and stopping at the first significant
   look, inflate the false-positive rate on data shaped like Hillstrom's spend?
3. Does a group-sequential design with O'Brien-Fleming-type alpha spending restore the 5%
   error rate on that data, what does it cost in power, and how much earlier does it stop?

## 2. Accrual model (an assumption)

Hillstrom has no timestamps: every customer was e-mailed at once, and each outcome is a
two-week total. Peeking is therefore simulated under the standard web-experiment model:
customers arrive in random order over 14 days in equal daily batches, each contributes their
full two-week outcome, and the analyst looks once a day.

**Limitation.** In a real e-mail test, everyone is enrolled on day one and outcomes accumulate
during the window, so peeking means looking at partial outcomes. That process has a different
correlation between looks and needs purchase timing that this data does not contain.

## 3. Data and test statistic

- Plasmode data from the control arm's spend, the primary metric: two pseudo-arms of 21,307 and
  21,306 customers (the sizes of the men's comparison), drawn with replacement from real
  control customers. Under the null both come from control; known effects inject extra buyers
  into one arm exactly as in Phase 1a.
- At every look: Welch's z on the cumulative data, two-sided. When the standard error is zero
  (no buyer yet in either arm), z = 0.
- One comparison (an e-mail vs no e-mail). Group-sequential designs for the three-arm Dunnett
  family are out of scope.

## 4. Peeking without a correction

- Rule: stop and declare significance at the first look with |z| > 1.96.
- Schedules of K equally spaced looks, K in {1, 2, 3, 5, 7, 10, 14, 20, 28, 50, 100}; K = 14 is
  one look per day.
- 20,000 A/A simulations (seed 202631). Every simulation is evaluated under every schedule.
  Monte Carlo standard errors are at most 0.35 percentage points.
- Reference: the exact rates for normally distributed data, from the same numerical recursion
  used for the boundaries below.

## 5. The fix: group-sequential design with alpha spending

- Lan-DeMets alpha spending with the O'Brien-Fleming-type function, in the convention of
  gsDesign's `sfLDOF`: each side spends f(t) = 2 - 2 * Phi(Phi^-1(1 - 0.025/2) / sqrt(t)) at
  one-sided level 0.025, where t is the information fraction. Two-sided alpha 0.05, symmetric
  efficacy boundaries, 14 equally spaced looks, no futility boundary.
  The older two-sided form 2 - 2 * Phi(1.96 / sqrt(t)) also spends 0.05 in total, but spends
  more early; the gsDesign convention was chosen so the boundaries can be reproduced in standard
  software.
- Boundaries are computed by numerical integration (the Armitage-McPherson-Rowe recursion) and
  checked two ways: against the published five-look boundaries (4.877, 3.357, 2.680, 2.290,
  2.031), and against a Monte Carlo simulation of Brownian motion.
- Validation: the A/A false-positive rate must lie within Monte Carlo error of 5% on the same
  20,000 Hillstrom-shaped simulations as section 4.
- Cost and benefit: 10,000 simulations at each of two known effects, +65% and +118% mean spend
  (the observed women's and men's lifts; seeds 202632 and 202633). Report power against a
  single test at the same maximum sample size, and the expected share of customers (equivalently
  days) used before stopping. Theoretical values from the recursion with drift sit alongside.

## 6. Power calculator

- Minimum detectable effect at the real sample size, and customers per arm needed for +10%,
  +25%, and +50% lifts, for spend, conversion, and visit, from the control arm's observed mean
  and variance. Two-sided alpha 0.05, 80% power.
- Two variance assumptions:
  - **Equal variances** (the textbook formula): the treated arm has control's variance.
  - **Buyer-driven:** the effect works by adding buyers, so with lift L the treated arm's
    variance is (1 + L) * E[Y^2] - ((1 + L) * mean)^2. For 0/1 metrics this is exactly the
    binomial variance at the new rate.
- Validation: 10,000 plasmode simulations at each variant's spend MDE (seeds 202634 and
  202635). The variant whose simulated power is close to 80% is the one to trust. Phase 1a
  saw 65% power at a +50% lift where the equal-variance formula promised more; this checks
  whether the buyer-driven variance explains the gap.

## 7. Defaults chosen here, not dictated by the plan

14 daily looks; no futility boundary; one two-arm comparison; the calculator as a tested
function with a report table rather than a Streamlit app.

## 8. Deviations

Any change after this note is committed is logged in section 12 of the analysis plan.
