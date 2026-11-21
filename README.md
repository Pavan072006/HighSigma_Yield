# High-sigma Monte Carlo for OTA input offset (GF180MCU)

**What:** estimate the probability that a 5-transistor OTA's input offset exceeds a spec
at the **4, 5 and 6 sigma** level (failure probabilities of 1e-5 down to 1e-9). The method
is ML-guided importance sampling, and every number is checked against brute-force Monte
Carlo where that's affordable.

**Why:** blocks that are repeated many times or shipped in huge volumes (SRAM cells,
comparators in an ADC, offset-trimmed amplifiers) must fail extremely rarely. Plain Monte
Carlo needs ~400/P simulations for ±10% accuracy. At 6 sigma (P≈1e-9) that's ~4e11 SPICE runs,
about 25 CPU-years. Commercial tools (Solido High-Sigma Monte Carlo, Cadence Spectre's
scaled-sigma sampling / worst-case distance) exist to close that gap. This project rebuilds the
textbook version from scratch with open-source tools: **ngspice + GF180 PDK + scikit-learn**.

**Result in one line:** at the 6 sigma spec, P(fail) = **1.49e-9 (95% CI 1.40–1.58e-9)**
from **~5,900 SPICE sims**. Brute force would need **~2.6e11** sims for ±10%. Where brute force can
still see failures (3σ, 4σ and 5σ, using 1e7 samples in 77 min), the IS estimate falls inside its
confidence interval.

**Second case, where the Gaussian shortcut fails:** a 6T SRAM cell's read noise margin. Specs set
from a Gaussian fit as "6σ" are really **4.9σ**, with **490× more failures** than the fit predicts. The same
IS method measures it with ~5k sims per spec, matching 1e6 brute-force samples at 3σ, 4σ and 5σ
(see [below](#second-metric-sram-read-margin-non-gaussian)).

Part of a portfolio. The design under test is the gm/Id hand design from
[LLM_Analog_Sizing](https://github.com/Pavan072006/LLM_Analog_Sizing) (project 1: LLM-agent OTA sizing + layout).

---

## Method

```
x ~ N(0, I_12)           12 normalised mismatch variables: dVth and dbeta of M1..M6
   │  Pelgrom: dVth_i = σVth_i·x_i,  dβ_i = σβ_i·x_{6+i}   (σ from the PDK's own coefficients)
   ▼
ngspice: 250 OTA copies per netlist, unity-gain follower, one .op  →  Vos = Vout − VCM
   │  12 cores in parallel, ~2 ms per sample
   ▼
1. explore   600 sims from a widened Gaussian N(0, 3²I), to reach the tails
2. learn     Gaussian-process regression Vos ≈ g(x)  (linear + ARD-RBF kernel)
             classifier = |g(x)| > spec; checked on 300 held-out sims
3. MPFP      most probable failure point per spec and side: min |x|² s.t. g(x)=±spec
             (HL-RF / FORM iteration on the surrogate, then 3 SPICE sims to correct it)
4. IS        q(x) = 0.1·N(0,I) + 0.9·Σ_side w_side·N(x*_side, I)
             P = mean(fail(x)·p(x)/q(x)), pass/fail always from SPICE, 5 × 1000 sims per spec
```

* **Mismatch model.** The GF180 ngspice models include their own Pelgrom mismatch (`fets_mm`
  section, `sw_stat_mismatch`). Its coefficients: A_VT = 7.15 / 6.66 mV·µm and A_β = 0.70 / 0.28 %·µm
  for nfet/pfet_03v3. We use the same formula and coefficients but draw the random numbers
  ourselves, because importance sampling has to *choose* the samples. Check (step 1): offset std is
  4.45 mV (ours) vs 4.67 mV (PDK `agauss`). The 5% gap is a known quirk of the PDK wrapper: for
  multi-finger devices it uses per-finger area, which adds √2 too much mismatch on the 2-finger
  M3/M4. Accounting for that predicts 4.61 mV.
* **Specs.** Fail if |Vos| > spec, with spec_k = mean + k·std of a quick 3000-sample MC
  (18.1 / 22.6 / 27.0 / 31.5 mV for k = 3…6). Because of the +4.8 mV systematic offset, the
  negative side is 2 sigma further away. The method still finds and samples both sides.
* **Why the surrogate is safe.** It only decides *where* to sample. Every IS sample's pass/fail
  comes from SPICE, and the p/q weights keep the estimate unbiased for any proposal. A poor surrogate
  widens the CI but can't shift the answer. The 10% nominal component bounds the weights (≤10).

## Results

| spec (target σ) | \|Vos\| > | brute force MC, N=1e+07: P [95% CI] (fails) | ML-guided IS: P [95% CI] | σ-equiv. (IS) | FORM Φ(−β) | Gaussian fit | IS SPICE sims | brute-force sims for ±10% |
|---|---|---|---|---|---|---|---|---|
| 3σ (Φ(−3) = 1.35e-03) | 18.1 mV | 1.56e-03 [1.53e-03, 1.58e-03] (15578) | **1.61e-03** [1.50e-03, 1.71e-03] | 2.95 | 1.54e-03 | 1.54e-03 | 5006 (+900 shared) | 2.39e+05 |
| 4σ (Φ(−4) = 3.17e-05) | 22.6 mV | 3.76e-05 [3.40e-05, 4.16e-05] (376) | **3.62e-05** [3.47e-05, 3.77e-05] | 3.97 | 3.76e-05 | 3.78e-05 | 5006 (+900 shared) | 1.06e+07 |
| 5σ (Φ(−5) = 2.87e-07) | 27.0 mV | 3.00e-07 [1.02e-07, 8.82e-07] (3) | **4.12e-07** [3.70e-07, 4.55e-07] | 4.93 | 3.99e-07 | 4.02e-07 | 5006 (+900 shared) | 9.32e+08 |
| 6σ (Φ(−6) = 9.87e-10) | 31.5 mV | < 3.84e-07 (0 fails) | **1.49e-09** [1.40e-09, 1.58e-09] | 5.93 | 1.47e-09 | 1.49e-09 | 5006 (+900 shared) | 2.58e+11 |

(Auto-generated version with the latest numbers: [`results/results_table.md`](results/results_table.md).)

* **Sims needed for ±10% (95%) with IS:** ≈1.4k (3σ), 2.0k (4σ), 2.4k (5σ), 2.9k (6σ). The cost
  barely grows with sigma, while brute force grows as 1/P. At 6σ that's ~10^8× fewer sims.
* **Repeatability:** 5 independent IS runs per spec. At 6σ they give 1.38–1.64e-9, consistent with
  the per-run ±17% CI.
* **Surrogate:** held-out RMSE 0.5 µV over a ±30 mV range; the classifier is 100% accurate on held-out
  sims at every spec. The 6σ MPFP direction is (Vth_M1, Vth_M2, Vth_M3, Vth_M4) ≈ (−0.69, +0.69,
  −0.16, +0.16), with everything else < 0.06. Offset is the input pair plus the load scaled by
  gm3/gm1, as hand analysis predicts. The tail transistors don't matter.
* **Honest reading:** this offset is almost linear in the mismatch variables, so the tail
  *is* Gaussian. FORM Φ(−β) and a Gaussian fit of the 1e7 MC agree with IS within a few percent.
  The demo shows that the machinery works and is validated. It does *not* show a case where
  Gaussian extrapolation fails (see limitations).

| | |
|---|---|
| ![histogram](results/fig1_offset_hist.png) | ![boundary](results/fig2_boundary.png) |
| ![convergence](results/fig3_convergence.png) | ![p vs spec](results/fig4_pfail_vs_spec.png) |

1. Offset histogram (1e7 MC, log scale) with Gaussian fit and spec lines.
2. Surrogate fail boundary in the two most important variables (Vth of M1, M2). The circles are
   sigma contours, the stars are MPFPs, and the dots are training sims and the 6σ IS cloud.
3. Convergence: at 3σ brute force needs ~1e5–1e7 sims to settle, while IS settles in ~1e3. On the right,
   IS converges about equally fast at every sigma level.
4. P(fail) vs spec: brute force, IS, FORM and Gaussian fit.

## How to run

Everything runs inside the IIC-OSIC-TOOLS container (ngspice-47, GF180 PDK, Python 3.12 with
numpy/scipy/matplotlib/scikit-learn):

```bash
docker start iic-osic-tools_shell_uid_1000
docker exec iic-osic-tools_shell_uid_1000 bash -lc 'cd /foss/designs/highsigma_mc && ./run_all.sh'
```

or step by step (inside the container, in `/foss/designs/highsigma_mc`):

| step | command | time (12 cores) | output |
|---|---|---|---|
| 0 | `python3 -m hsmc.ota_mc` | 1 s | sigmas, nominal offset, batch sanity check |
| 1 | `python3 scripts/01_check_mismatch.py` | 5 s | `results/01_mismatch_check.json` |
| 2 | `python3 scripts/02_brute_mc.py 1000000` (results use 1e7: ~77 min) | ~10 min | `results/02_brute_mc.{json,npz}` |
| 3 | `python3 scripts/03_highsigma_is.py` | ~2 min | `results/03_highsigma_is.json`, IS samples, GP |
| 4 | `python3 scripts/04_plots.py` | 10 s | `results/fig*.png` |
| 5 | `python3 scripts/05_table.py` | 1 s | `results/results_table.md` |
| 6 | `python3 scripts/06_p1_designs.py` | ~3 min | `results/p1_designs.{md,json}` |
| 7 | `python3 scripts/07_sram_mc.py 1000000` | ~38 min | SRAM brute-force MC, `results/07_sram_mc.json` |
| 8 | `python3 scripts/08_sram_is.py` | ~5 min | SRAM IS, `results/08_sram_is.json` |
| 9 | `python3 scripts/09_sram_report.py` | 5 s | `results/fig5_sram_snm.png`, `results/sram_table.md` |

Code: [`hsmc/ota_mc.py`](hsmc/ota_mc.py) (mismatch model, batched netlist, parallel runner, specs)
[`hsmc/sram_snm.py`](hsmc/sram_snm.py) (6T SRAM read SNM) and [`scripts/`](scripts). Concepts and interview Q&A: [`NOTES.md`](NOTES.md).

## Applied to the LLM agent's designs

`python3 scripts/06_p1_designs.py` runs the offset analysis on the 8 final designs from
[LLM_Analog_Sizing](https://github.com/Pavan072006/LLM_Analog_Sizing) (4000 MC sims each), with an offset spec of |Vos| ≤ 10 mV, then the
full ML-guided importance sampling on the worst one. Table: [results/p1_designs.md](results/p1_designs.md).

* The optimiser never saw an offset spec. Five designs still reach > 5σ offset yield, S5 only 2.9σ, and
  **S1 and S7 fail on 10–13 % of chips** (1.1–1.3σ).
* The cause is visible in the mismatch numbers: minimising power pushed the PMOS load very small
  (S7: σVth of M3 = 14.5 mV vs 0.9 mV for the input pair), so the load dominates the offset.
* These OTAs also have a 2–6 mV systematic offset (mean), so a Gaussian estimate has to include the mean.
  With it, Gaussian, MC and importance sampling agree; on S7 IS matches the Gaussian within 2 % at 3–4σ
  and gives 1.5× higher P(fail) at 6σ (2.8e-8 vs 1.9e-8).
* Next step: add offset as a constraint in the sizing loop, using IS for the yield check.

## Second metric: SRAM read margin (non-Gaussian)

The offset above is nearly Gaussian, so a simple fit would have worked. The **read static noise margin
(SNM)** of a 6T SRAM cell is the classic case where it does not. In a read, both bit lines and the word
line are high. SNM is the side of the largest square that fits inside each lobe of the "butterfly" curve
(the two half-cells' transfer curves). The cell's SNM is the **smaller** of the two lobes.

* **Cell:** GF180 3.3 V devices, L = 0.28 µm, pull-down 0.8 µm, pass-gate 0.4 µm, pull-up 0.3 µm.
  12 mismatch variables (Vth and β of 6 transistors), σVth 15–21 mV. Nominal SNM 369 mV.
* **Simulation:** one DC sweep per half-cell, 200 cells per ngspice run. SNM is computed with the rotated-axis
  method (checked against a brute-force largest-square search). ~2.3 ms per cell on 12 cores.
* **Specs, the usual way:** fit a Gaussian to a 10k-sample pilot MC (359.0 ± 9.8 mV) and set
  spec_k = mean − k·std. Fail if SNM < spec.
* **IS:** one GP surrogate per lobe, an MPFP per lobe, and a mixture proposal. Pass/fail always comes from SPICE.

![SRAM SNM](results/fig5_sram_snm.png)

| spec | SNM < | Gaussian fit says | brute-force MC, N=1e6 [95% CI] (fails) | ML-guided IS [95% CI] | real sigma |
|---|---|---|---|---|---|
| "3σ" | 329.6 mV | 1.35e-03 | 4.14e-03 [4.0e-03, 4.3e-03] (4137) | 4.08e-03 [3.9e-03, 4.3e-03] | 2.65 |
| "4σ" | 319.8 mV | 3.17e-05 | 3.70e-04 [3.3e-04, 4.1e-04] (370) | 3.34e-04 [3.1e-04, 3.5e-04] | 3.40 |
| "5σ" | 310.0 mV | 2.87e-07 | 1.90e-05 [1.2e-05, 3.0e-05] (19) | 1.63e-05 [1.5e-05, 1.8e-05] | 4.15 |
| "6σ" | 300.1 mV | 9.87e-10 | < 3.8e-06 (0 fails) | 4.83e-07 [4.6e-07, 5.1e-07] | 4.90 |

* **The Gaussian fit is wrong by 3× at 3σ and 490× at 6σ.** A designer who trusted it would ship a
  cell rated "6σ" that is really 4.9σ. In a 1 Mb array, that means ~0.5 failing cells per chip instead of ~0.001.
* **Why:** each lobe alone is close to Gaussian (mean 368.5 mV, std 13.6 mV, and FORM per lobe agrees
  with IS within 6 %). Taking the minimum of two anti-correlated lobes (correlation −0.52) pulls the
  mean down and squeezes the spread to 9.8 mV, and either lobe can fail. A Gaussian fitted to the
  minimum therefore has a tail that is far too thin.
* **What dominates:** at the 6σ MPFP, the pull-down and pass-gate Vth (PD 0.78, PG 0.38, opposite-side
  PD 0.28). The pull-up hardly matters, as expected for read stability.
* **Cost:** IS used 20,924 sims for all four specs. Brute force needed 1e6 sims (38 min) to see 19 failures at 5σ.
  For ±10 % at 6σ it would need ~8e8 sims, which is out of reach.
* **Honest note:** the per-lobe surrogates use the known structure of SNM (min of two lobes). A single
  surrogate on SNM itself would have to learn the kink where the lobes swap.

## Limitations (honest)

* **Offset is near-linear.** For the OTA offset, Gaussian extrapolation would have worked too. The SRAM
  read margin is the non-Gaussian case. Write margin and comparator decisions are not covered yet.
* **Pre-layout, local mismatch only.** No layout effects (gradients, LOD/WPE), no global process
  variation (you'd repeat the analysis per global corner), DC only, typical corner, 27 °C.
* **Statistics assumed Gaussian out to 6σ.** The Pelgrom model is characterised by the foundry at
  a few sigma; its tails at 6σ are an extrapolation, as in every high-sigma flow.
* **Direct validation stops at 5σ.** 1e7 brute-force samples give 15,578 / 376 / 3 failures at
  3σ / 4σ / 5σ, and zero at 6σ. The 5σ check rests on only 3 failures (a wide CI). 6σ is checked only
  indirectly (FORM, Gaussian fit, repeatability across seeds).
* **Low dimension (12).** Mean-shift IS is easy here. Hundreds of variables (a whole SRAM column)
  would need variable screening, or scaled-sigma sampling / subset simulation.
* One design, one spec type. Not a product.
