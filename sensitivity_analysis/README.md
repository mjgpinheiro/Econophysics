# Sensitivity Analysis & Optimal-Control Benchmark

Reproduction code for **Sections 6.4 (Monte Carlo Sensitivity Analysis)
and 6.5 (Optimal-Control Benchmark)** of:

> Pinheiro, M.R.; Pinheiro, M.J. *Information Loss in Scalar Monetary
> Aggregation: A Tensorial Langevin Framework for Financial Shock
> Propagation and Policy Targeting.* Submitted to *Entropy* (MDPI), 2026.

The core model here is a vectorised reimplementation of
`Money_Tensor_crisis_model_v2.ipynb` (same repository), verified to
reproduce every number of the baseline calibration exactly:
alpha(A) = -0.030, lambda_max(P) = 18.30, kappa(P) = 7.6, recovery
times 40.7 / 69.9 / 13.7 / 45.1 / 12.8 quarters, sector peak drops
18.9 / 5.8 / 3.9 %, aggregate drop 8.6 %, and the 21-quarter
structural lag (17.0 vs 38.0 quarters).

## Requirements

Python >= 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`.

## Files and run order

| Script | Produces | Manuscript object |
|---|---|---|
| `model.py` | (library, imported by all others) | — |
| `run_mc.py` | `mc_results.csv`, `sweeps.npz` + console summary | Table 2 (all medians, percentiles, ordinal percentages, censoring) |
| `make_fig.py` | `fig8_robustness.png/pdf` | Figure 8 |
| `run_lqr.py` | `lqr_results.npz` + console summary | Fig. 9A curve (budget-matched LQR, rho* = 128, tau_r = 21.3 q) |
| `lqr_pareto.py` | `lqr_pareto.npz` + console summary | Fig. 9B/C data; the 1.18x and 2.34x frontier ratios |
| `make_fig9.py` | `fig9_lqr.png/pdf` | Figure 9 |

```bash
python run_mc.py       # ~2-4 min: 1000 draws x 5 policies + sweeps
python make_fig.py
python run_lqr.py      # LQR frontier + budget-matched point
python lqr_pareto.py   # Pareto plane + equal-effort frontier points
python make_fig9.py
```

The Monte Carlo random seed is fixed (`20260713` in `run_mc.py`), so
all outputs — including every percentage quoted in the manuscript —
are exactly reproducible. Pre-computed outputs (`mc_results.csv`,
`*.npz`, figures) are included for convenience.

## Monte Carlo design (Section 6.4)

N = 1000 draws with independent multiplicative perturbations:
each non-zero coupling entry x U(0.5, 1.5); each friction rate
x U(0.7, 1.3); each shock amplitude x U(0.7, 1.3); each shock decay
x U(0.8, 1.2). All draws stable and positivity-preserving. Five
policy scenarios per draw (no policy; uniform/targeted, baseline
envelope; uniform/targeted, symmetric exit rule) on a 120-quarter
horizon; recovery times right-censored at 120 q.

## LQR benchmark (Section 6.5)

Exact linear-quadratic regulator for the cost functional
C = int [ x'Px + rho ||u||^2 ] dt (Q = P, R = rho I, B = I), solved
via the algebraic Riccati equation and deployed at t0 = 8 quarters.
`run_lqr.py` finds the gross-budget-matched rho*; `lqr_pareto.py`
traces the Pareto frontier in the (int ||u||^2 dt, int V dt) plane
and evaluates it at the control effort of each heuristic.

## Note on the coupling table

The eleven non-zero coupling entries used throughout (in `model.py`,
`K_ENTRIES`) are those of `Money_Tensor_crisis_model_v2.ipynb` and of
the corrected Table 1 of the revised manuscript. The slowest mode of
the system (alpha(A) = -0.030) is the Services-Government channel,
whose +0.02 self-coupling lowers its effective friction from 0.05 to
0.03 per quarter.
