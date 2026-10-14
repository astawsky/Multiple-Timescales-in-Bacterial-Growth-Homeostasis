# Fixes after the adversarial review (work in progress)

**Status: partial. Not all 16 experiments are refitted, and the HTML pages and notebook one level up are NOT yet updated to these results.**

## Round 1: `fit3.py` (pipeline fixes)
Rebuilt from the unfiltered `physical_units.csv`:
- per-trap sister/neighbour flag
- the repository's |z|>3 per-lineage filter applied to model output too
- tau rounded to the imaging interval
- real lineage lengths
- no hard-coded variance ratios
- partial first cycles (start_time = 0) dropped
- 5-start Nelder-Mead
- model ladder M0-M4 and a lineage-split cross-validation (`python fit3.py fit|ladder <datasets>`)

Results:
- `fit3_results.json` holds 7 of 16 experiments (SM, the three MM runs and the three Tanouchi runs). The lambda_LB entry is invalid (imaging-interval bug, since fixed).
- `ladder_results.json` covers SM and Tan27 completely and Wang 0525 up to M3.
- Note: fit3 used an instrumental-variable estimate of k. That correction was later shown to be wrong (see below).

## Round 2: the four remaining fixes, each tested on SM, Tanouchi and Wang 0525

**1. Measurement noise (`meas_noise.py` → `meas_noise.json`).**
Measured from the raw images instead of fitted: the fitted-length error is 0.5-2% (6% in Wang 0702).
Consequences:
- Measurement noise cannot explain the mother/daughter length mismatch.
- OLS k is consistent, so the IV correction is reverted.

**2. Optimizer (`fit4.py 2`).**
CMA-ES on a seed-averaged loss. Restart losses on SM span 0.64-1.34 (Nelder-Mead: 0.71-10.7).
- sigma_Delta is stable at 0.25.
- nu_Delta is not identified (4 to 185).
- In Wang, two runs reach equal loss with different parameters.

**3. Inherited timing state (`fit4.py 3`).**
Sister tau correlation at birth improves from 0.33 to 0.55 (data 0.70) and the loss drops from 0.88 to 0.51.
The held-out checks fail:
- the decay is too fast (cycle 2: 0.21 vs 0.54)
- neighbours are over-predicted (0.40 vs 0.15)

A single AR(1) timing state is not the right mechanism.

**4. Pooling within a lab (`fit4.py 4`).**
- Separate Tanouchi fits disagree between runs (rho_slow 0.54 vs 0.80).
- Pooled fits agree (0.57 and 0.54), with held-out accuracy unchanged.
- Tanouchi lineages show almost no lineage-level structure.

Wang check: `fit4.py wang`. All runs are logged in `fit4.log`; results are in `fit4_results.json`.

## Not done
- Full 16-experiment refit with fixes 1, 2 and 4.
- Bootstrap error bars and the transfer test on the fixed pipeline.
- A different mechanism for sister timing.
- Updating the write-ups and notebook.
