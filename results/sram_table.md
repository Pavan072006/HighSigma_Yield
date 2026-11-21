6T SRAM read SNM, GF180 3.3 V, 12 mismatch variables. Pilot fit: 359.0 ± 9.81 mV; spec_k = mean - k·std.

| spec | SNM < mV | Gaussian fit | brute-force MC (N=1e+06) [95% CI] | ML-guided IS [95% CI] | real sigma | IS / Gaussian |
|---|---|---|---|---|---|---|
| 3σ | 329.6 | 1.35e-03 | 4.14e-03 [4.0e-03, 4.3e-03] (4137) | 4.08e-03 [3.9e-03, 4.3e-03] | 2.65 | 3x |
| 4σ | 319.8 | 3.17e-05 | 3.70e-04 [3.3e-04, 4.1e-04] (370) | 3.34e-04 [3.1e-04, 3.5e-04] | 3.40 | 11x |
| 5σ | 310.0 | 2.87e-07 | 1.90e-05 [1.2e-05, 3.0e-05] (19) | 1.63e-05 [1.5e-05, 1.8e-05] | 4.15 | 57x |
| 6σ | 300.1 | 9.87e-10 | < 3.8e-06 (0 fails) | 4.83e-07 [4.6e-07, 5.1e-07] | 4.90 | 490x |

Lobe correlation -0.52; IS sims 20924, surrogate RMSE [0.08, 0.09] mV.
