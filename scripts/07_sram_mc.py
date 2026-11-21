#!/usr/bin/env python3
"""Step 7 - SRAM read margin: brute-force Monte Carlo baseline"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hsmc.sram_snm import DIM, pilot_specs, simulate  # noqa: E402

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), c + h


N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
CHUNK = 100_000
pil = pilot_specs()
specs = {int(k): v for k, v in pil["specs_mV"].items()}
rng = np.random.default_rng(2027)
lobes = np.empty((0, 2), dtype=np.float32)
t0 = time.time()
while len(lobes) < N:
    n = min(CHUNK, N - len(lobes))
    lobes = np.vstack([lobes, (simulate(rng.standard_normal((n, DIM))) * 1e3).astype(np.float32)])
    snm = lobes.min(1)
    print(f"{len(lobes):>9d} samples {time.time()-t0:6.0f} s  fails@3={int((snm < specs[3]).sum())} "
          f"@4={int((snm < specs[4]).sum())}", flush=True)

snm = lobes.min(1)
ok = np.isfinite(snm)
res = {"N": int(ok.sum()), "n_nan": int((~ok).sum()), "seconds": time.time() - t0,
       "pilot": pil, "lobe_corr": float(np.corrcoef(lobes[ok].T)[0, 1]),
       "lobe_mean_mV": lobes[ok].mean(0).tolist(), "lobe_std_mV": lobes[ok].std(0).tolist(), "specs": {}}
for k, s in specs.items():
    nf = int((snm[ok] < s).sum())
    lo, hi = wilson(nf, res["N"])
    res["specs"][str(k)] = {"spec_mV": s, "n_fail": nf, "p_fail": nf / res["N"], "ci95": [lo, hi],
                            "p_gauss_fit": float(norm.sf(k))}
    print(f"spec {k}: SNM < {s:.1f} mV  {nf} fails -> P={nf/res['N']:.3e} [{lo:.2e}, {hi:.2e}]  "
          f"Gaussian fit says {norm.sf(k):.2e}")
np.savez_compressed(ROOT / "results" / "07_sram_mc.npz", lobes=lobes)
(ROOT / "results" / "07_sram_mc.json").write_text(json.dumps(res, indent=1))
