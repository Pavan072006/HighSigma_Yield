| spec (target σ) | \|Vos\| > | brute force MC, N=1e+07: P [95% CI] (fails) | ML-guided IS: P [95% CI] | σ-equiv. (IS) | FORM Φ(−β) | Gaussian fit | IS SPICE sims | brute-force sims for ±10% |
|---|---|---|---|---|---|---|---|---|
| 3σ (Φ(−3) = 1.35e-03) | 18.1 mV | 1.56e-03 [1.53e-03, 1.58e-03] (15578) | **1.61e-03** [1.50e-03, 1.71e-03] | 2.95 | 1.54e-03 | 1.54e-03 | 5006 (+900 shared) | 2.39e+05 |
| 4σ (Φ(−4) = 3.17e-05) | 22.6 mV | 3.76e-05 [3.40e-05, 4.16e-05] (376) | **3.62e-05** [3.47e-05, 3.77e-05] | 3.97 | 3.76e-05 | 3.78e-05 | 5006 (+900 shared) | 1.06e+07 |
| 5σ (Φ(−5) = 2.87e-07) | 27.0 mV | 3.00e-07 [1.02e-07, 8.82e-07] (3) | **4.12e-07** [3.70e-07, 4.55e-07] | 4.93 | 3.99e-07 | 4.02e-07 | 5006 (+900 shared) | 9.32e+08 |
| 6σ (Φ(−6) = 9.87e-10) | 31.5 mV | < 3.84e-07 (0 fails) | **1.49e-09** [1.40e-09, 1.58e-09] | 5.93 | 1.47e-09 | 1.49e-09 | 5006 (+900 shared) | 2.58e+11 |

Shared setup: 600 training + 300 validation sims; surrogate held-out RMSE 0.48 µV. IS sims for ±10% at one sigma level (from observed CI widths): 3σ ≈ 1395, 4σ ≈ 2003, 5σ ≈ 2393, 6σ ≈ 2941. Brute force: 1e+07 sims in 77 min on 12 cores.
