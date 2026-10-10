"""Experiment-analysis toolkit: each method is validated by simulation before use.

Modules:
- data, criteo: download, checksum, and sealed loading of the Hillstrom and Criteo data
- srm, balance: randomization checks
- bootstrap, inference, multiplicity: exact bootstrap and BCa intervals, Welch,
  Dunnett (bootstrap max-T and parametric), Benjamini-Hochberg
- adjust: CUPED and Lin's regression adjustment with HC2 standard errors
- power, sequential: minimum detectable effects, alpha spending, group-sequential boundaries
- uplift: T-learner, DR-learner and causal forest; calibration test, uplift curves,
  policy values, moderator tests
- simulate, simulate_adjustment, simulate_sequential, simulate_uplift: plasmode
  simulations that check each method against a known truth
- freeze: fingerprints that prove a refit is the model that was frozen
"""
