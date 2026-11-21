#!/usr/bin/env python3
"""Step 2 - brute-force Monte Carlo baseline"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hsmc.ota_mc import DIM, SPECS_MV, fails, simulate  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), c + h


N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
CHUNK = 200_000
rng = np.random.default_rng(2026)
vos = np.empty(0, dtype=np.float32)
t0 = time.time()
while len(vos) < N:
    n = min(CHUNK, N - len(vos))
    v = simulate(rng.standard_normal((n, DIM)), batch=250)
    vos = np.concatenate([vos, v.astype(np.float32)])
    print(f"{len(vos):>10d} samples  {time.time()-t0:7.0f} s  "
          f"fails@3sig={int(fails(vos, SPECS_MV[3]).sum())}", flush=True)

mu, sd = float(np.nanmean(vos)), float(np.nanstd(vos))
res = {"N": len(vos), "seconds": time.time() - t0, "n_nan": int(np.isnan(vos).sum()),
       "mean_mV": mu * 1e3, "std_mV": sd * 1e3, "specs": {}}
for k, s in SPECS_MV.items():
    f = fails(vos, s)
    nf = int(f.sum())
    lo, hi = wilson(nf, len(vos))
    p_gauss = norm.sf((s * 1e-3 - mu) / sd) + norm.cdf((-s * 1e-3 - mu) / sd)
    res["specs"][str(k)] = {"spec_mV": s, "n_fail": nf, "p_fail": nf / len(vos),
                            "ci95": [lo, hi], "p_gauss_fit": float(p_gauss),
                            "target_phi_minus_k": float(norm.sf(k))}
    print(f"spec {k} sigma (|Vos|>{s} mV): {nf} fails / {len(vos)} -> P={nf/len(vos):.3e} "
          f"[{lo:.2e}, {hi:.2e}]   Gaussian-fit P={p_gauss:.3e}")

f3 = fails(vos, SPECS_MV[3])
idx = np.unique(np.logspace(2, np.log10(len(vos)), 200).astype(int))
res["convergence_3sig"] = {"n": idx.tolist(), "p": (np.cumsum(f3)[idx - 1] / idx).tolist()}

out = ROOT / "results"
np.savez_compressed(out / "02_brute_mc.npz", vos=vos)
(out / "02_brute_mc.json").write_text(json.dumps(res, indent=1))
