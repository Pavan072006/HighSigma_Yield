# NOTES: concepts behind high-sigma Monte Carlo (read before the interview)

## 0. The 30-second pitch (our numbers)

"I estimated how often a 5T OTA's input offset exceeds a spec at 3 to 6 sigma in GF180, using real
ngspice sims with the PDK's Pelgrom mismatch coefficients. Brute force with 10 million samples
sees only 3 failures at 5σ and none at 6σ. I trained a Gaussian-process surrogate on 600 sims and used it to find the most
probable failure point. Then I importance-sampled around that point with real SPICE. At the 6σ
spec I get P = 1.49e-9 ± 6% from about 6,000 sims. Brute force would need about 3e11. At 3σ, 4σ
and 5σ the estimate sits inside the brute-force confidence interval. The ML only chooses where to sample:
pass/fail always comes from SPICE, so a bad model can't bias the answer."

## 1. Sigma levels vs failure probability

"k sigma" is shorthand for a failure probability: the area of a standard Gaussian
beyond k standard deviations (one-sided tail, `P = Phi(-k)`).

| sigma | P(fail)   | 1 fail in ... | brute-force sims for +-10% (95%) |
|------:|----------:|--------------:|---------------------------------:|
| 3     | 1.35e-3   | 741           | ~2.8e5                           |
| 4     | 3.17e-5   | 31,600        | ~1.2e7                           |
| 5     | 2.87e-7   | 3.5 million   | ~1.3e9                           |
| 6     | 9.87e-10  | 1 billion     | ~3.9e11                          |

Rule of thumb: to estimate P with relative error e at 95% confidence, plain MC needs
`N ≈ (1.96/e)^2 / P` samples, i.e. about **100/P samples for ±20%, 400/P for ±10%**.

**Why care about 6 sigma?** A block that's repeated many times on a chip, or that ships in
huge volumes, needs a tiny per-instance failure rate. An SRAM with 10^8 bitcells needs each
cell to be around 6 sigma. An ADC with 1000 comparators, or a part sold 100 million times, also
needs 4-6 sigma on the critical blocks. Our 5T OTA offset is a stand-in for such a block.

## 2. Mismatch and Pelgrom's law

Two transistors drawn identically are never quite the same, because of random dopant
fluctuation, line-edge roughness and oxide variation. We model each device with:

* `dVth ~ N(0, sigma_Vth)`: threshold voltage shift (BSIM4 instance param `delvto`)
* `dbeta/beta ~ N(0, sigma_beta)`: current-factor (mobility) shift (`mulu0 = 1 - dbeta`)

**Pelgrom's law**: `sigma(dVth_pair) = A_VT / sqrt(W*L)`. Bigger area means more averaging,
so less mismatch. To halve the offset you need **4x the area**. That's the basic trade-off
between area, offset and speed (bigger devices bring more capacitance).

GF180 ships the coefficients inside its model file (`sm141064.spice`, section `fets_mm`):
`A_VT = 7.15 mV·um` (nfet_03v3) and `6.66 mV·um` (pfet_03v3). One device gets `A_VT/sqrt(2)`,
because the coefficient is defined for the *difference* of a pair. Our input pair
(W=7.7 um, L=0.5 um) therefore has sigma_Vth = 3.06 mV per device.

**Why not just turn on the PDK's switch (`sw_stat_mismatch=1`)?** It works (we checked: offset
std 4.67 mV versus 4.45 mV for ours). But it draws the random numbers itself with `agauss()`,
and importance sampling needs to *choose* where to sample. So we copy the PDK's formula and
supply the numbers ourselves. The 5% gap comes from the PDK computing sigma from the
*per-finger* area of multi-finger devices (M3/M4 have 2 fingers), which overstates their
mismatch by sqrt(2). With that effect included we predict 4.61 mV, which agrees.

**Offset of the 5T OTA (from the sensitivities we measured):**
`Vos ≈ (dVth1 - dVth2) + (gm3/gm1)(dVth3 - dVth4) + small beta terms`.
The input pair dominates (3.1 mV/sigma each). The PMOS load adds 0.7 mV/sigma
(gm3/gm1 = (gm/Id)3/(gm/Id)1 = 8/12). The tail (M5/M6) has almost no effect, because it moves
both sides together. So 12 random variables, but only about 4 matter. Also a systematic
+4.8 mV offset (the follower's Vds mismatch between M3 and M4), which makes failures one-sided.

## 3. Brute-force Monte Carlo and its confidence interval

Draw N random chips, count failures k, then `P_hat = k/N`. Each sample is a coin flip,
so `std(P_hat) = sqrt(P(1-P)/N)` and the relative error ≈ `1/sqrt(k)`. **The precision
depends on how many failures you saw, not on N itself.** With 0 to 10 failures use the
Wilson interval (the normal approximation breaks down there). With 0 failures in N samples,
the 95% upper bound is about 3/N (the "rule of three").

## 4. Importance sampling (IS)

Plain MC wastes almost every sample on passing chips. IS draws from a different
distribution q(x) that lands near the failure region, then corrects for that bias with weights:

    P = E_p[ fail(x) ] = E_q[ fail(x) * p(x)/q(x) ]
    P_hat = (1/N) * sum fail(x_i) * w_i,    w_i = p(x_i)/q(x_i),   x_i ~ q

* This is **unbiased for any q** that covers the failure region. q only changes the variance.
* **Mean shift**: q = N(x*, I), with x* the **Most Probable Failure Point (MPFP)**: the
  failing point closest to the origin in normalised space (highest probability density).
  About half the samples fail, and the weights `exp(-x·x* + |x*|²/2)` are moderate.
* `beta = |x*|` is the "reliability index". For a linear failure boundary, `P = Phi(-beta)` exactly
  (FORM, first-order reliability method). IS then corrects whatever FORM gets wrong because of curvature.
* **Mixture proposal**: `q = 0.1 N(0,I) + 0.9 * sum_sides w_s N(x*_s, I)`. One component per
  failure region (|Vos| > spec has a + side and a - side). The 10% "defensive" nominal
  component caps the weight at 10, so a sample landing somewhere unexpected can't blow up the variance.
* CI: the weighted samples are i.i.d., so `se = std(f*w)/sqrt(N)` and the 95% CI is ±1.96 se.
  We also repeat the whole thing with 5 seeds and check that the runs scatter about as much as the CI says.

## 5. Why the ML surrogate/classifier helps

Finding x* needs an optimisation (minimise |x|² subject to Vos(x) = spec). Doing that directly
on SPICE costs hundreds of sims per spec, and needs gradients through a simulator. Instead:

1. Simulate a few hundred points from a **widened** Gaussian (sigma×3), which reaches
   out to the 4-6 sigma region.
2. Fit a **Gaussian-process regression** `Vos_hat = g(x)`. The kernel is linear + RBF with one
   length scale per input (ARD) + noise. Thresholding `|g(x)| > spec` gives the pass/fail
   **classifier**, and one regression model serves every spec level.
3. Run the MPFP optimisation on g (microseconds per call), then correct x* with 3 real SPICE
   sims along its direction.

**Key safety property:** the final pass/fail of every IS sample comes from **SPICE, not the
model**. A wrong surrogate only puts the shift in a worse place, which makes the estimate
noisier (a wider CI) but never biased. Methods that *replace* SPICE with the classifier for
most samples (classifier screening, Solido-style) save even more sims, but need guard bands
or active learning to stay accurate near the boundary.

Training on the regression value (Vos) rather than just pass/fail labels uses more of the
information: a sample at 10 mV says more than "pass".

## 6. How the estimate is validated

1. **Mismatch model** vs the PDK's own statistical model (std within 5%, with the gap explained).
2. **At 3, 4 and 5 sigma**, where 1e7 brute-force samples still see failures (15,578 / 376 / 3):
   the IS estimate must fall inside the brute-force CI. It does.
3. **Repeatability**: 5 independent IS runs per spec. Their spread must match the stated CI.
4. **Sanity vs theory**: compare with FORM `Phi(-beta)` and with a Gaussian fit of the offset
   distribution. Offset is nearly linear in the mismatch here, so these should be close. If they
   are, that's the honest conclusion, not a failure of the method.

## 7. Likely interview questions (with short answers)

**Q: Why not just run more Monte Carlo?** 6 sigma means about 1e-9. You'd need ~4e11 sims for ±10%.
At 2 ms each that's 25 CPU-years. IS gets there with a few thousand.

**Q: Isn't the estimate biased if your ML model is wrong?** No. The model only chooses *where* to
sample. Pass/fail always comes from SPICE, and the weights p/q make the estimator unbiased for any
proposal that covers the failure region. A bad model costs variance, not bias. The defensive
mixture component guarantees coverage.

**Q: What happens with multiple failure regions?** A single mean shift only covers one. We find the
MPFP for each side and use a mixture. Generally you'd cluster the classifier's failing samples
and put one component per cluster.

**Q: Curse of dimensionality?** Plain IS weights degrade in high dimensions if the proposal
variance is wrong. With unit-variance mean shift, the weight depends only on the projection onto
x*, so it works in hundreds of dims if the boundary is near-linear. In practice you also screen out
insensitive variables first (our tail transistors). For truly high-dimensional problems (SRAM
arrays), people use scaled-sigma sampling or subset simulation.

**Q: Gaussian extrapolation gives the same answer, so why bother?** Here the offset is almost
linear in Vth, so it's nearly Gaussian and extrapolation works. Many real metrics are not:
SRAM read/write margins, comparator metastability, anything with a saturating or min/max
behaviour. There the tail is heavier or lighter than Gaussian, and extrapolating from 3 sigma can be
wrong by orders of magnitude. IS with real SPICE samples measures the actual tail.

**Q: How do you choose N_IS?** Run until the relative CI half-width is below a target (e.g. 10%).
Here we measured ~1.4k (3σ) to ~2.9k (6σ) IS sims for ±10%. The cost grows only slowly with
sigma, while brute force grows as 1/P. That's the main appeal.

**Q: What does Solido / Cadence do?** Solido High-Sigma Monte Carlo generates millions-billions of
candidate samples, uses an ML model to *order* them by how likely they are to fail, and simulates
the most-likely-failing ones first until the tail is resolved. Cadence Spectre's "scaled sigma
sampling" and "worst-case distance" are related. Our method is the textbook form: surrogate, then
MPFP, then mean-shift IS.

**Q: Global vs local variation?** We modelled only local mismatch (within die). Global
process variation (corners or a global MC) shifts both devices together and barely moves the
offset. You'd run the high-sigma analysis at each global corner.

**Q: Limitations of your setup?** Pre-layout only. Pelgrom only (no gradient effects or LOD/WPE
from layout). DC offset only. One design. The model's mismatch statistics are assumed Gaussian
out to 6 sigma, and real foundry data is rarely characterised that far.
