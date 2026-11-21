#!/usr/bin/env python3
"""Step 8 - SRAM read margin: ML-guided importance sampling, one surrogate per lobe"""
import json
import pathlib
import sys
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hsmc.sram_snm import DIM, VAR_NAMES, pilot_specs, simulate  # noqa: E402

warnings.filterwarnings("ignore", module="sklearn")

N_TRAIN, TRAIN_SCALE = 600, 3.0
N_VAL = 300
N_IS, N_REP = 1000, 5
pil = pilot_specs()
SPECS = {int(k): v for k, v in pil["specs_mV"].items()}
rng = np.random.default_rng(8)
sims = {"train": 0, "validate": 0, "mpfp": 0, "is": 0}
t0 = time.time()

# surrogate: one GP per lobe
Xtr = rng.standard_normal((N_TRAIN, DIM)) * TRAIN_SCALE
Ytr = simulate(Xtr, batch=50) * 1e3
sims["train"] = N_TRAIN
ok = np.isfinite(Ytr).all(1)
gps = []
for j in range(2):
    kernel = (DotProduct(sigma_0=1.0) + ConstantKernel(1.0) * RBF(length_scale=np.full(DIM, 5.0),
              length_scale_bounds=(0.1, 1e4)) + WhiteKernel(1e-4, (1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=j)
    gps.append(gp.fit(Xtr[ok], Ytr[ok, j]))
print(f"A. {N_TRAIN} training sims, {np.sum(~ok)} failed, {time.time()-t0:.0f}s")

Xv = np.vstack([rng.standard_normal((N_VAL // 2, DIM)), rng.standard_normal((N_VAL // 2, DIM)) * TRAIN_SCALE])
Yv = simulate(Xv, batch=50) * 1e3
sims["validate"] = N_VAL
Pv = np.stack([gp.predict(Xv) for gp in gps], 1)
rmse = np.sqrt(np.nanmean((Pv - Yv) ** 2, 0)).tolist()
cls_acc = {k: float(np.nanmean((Pv.min(1) < s) == (Yv.min(1) < s))) for k, s in SPECS.items()}
print(f"B. surrogate RMSE per lobe {np.round(rmse, 2)} mV; classifier accuracy {cls_acc}")


def mpfp(gp, target, iters=100):
    g = lambda x: gp.predict(np.atleast_2d(x))[0]
    x = np.zeros(DIM)
    E = np.eye(DIM) * 1e-2
    for _ in range(iters):
        gr = (gp.predict(x + E) - gp.predict(x - E)) / 2e-2
        x_new = (gr @ x - (g(x) - target)) / (gr @ gr) * gr
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    return x if abs(g(x) - target) < 1e-3 * abs(target) else None


def spice_correct(x, j, target):
    a = np.array([0.95, 1.0, 1.05])
    v = simulate(x[None, :] * a[:, None])[:, j] * 1e3
    sims["mpfp"] += 3
    return x * np.polyval(np.polyfit(v, a, 1), target), float(v[1])


def log_q(X, centers, wts):
    return logsumexp(np.vstack([np.log(w) - 0.5 * np.sum((X - c) ** 2, 1) for c, w in zip(centers, wts)]), 0)


results = {}
for k, s in SPECS.items():
    sides = []
    for j in range(2):
        x = mpfp(gps[j], s)
        if x is None:
            print(f"C. spec {k} lobe {j}: MPFP search failed")
            continue
        xc, v = spice_correct(x, j, s)
        sides.append({"lobe": j, "x": xc, "beta": float(np.linalg.norm(xc)), "spice_at_surrogate_mpfp": v})
        print(f"C. spec {k} lobe {j}: beta {np.linalg.norm(x):.3f} -> {np.linalg.norm(xc):.3f} after SPICE check")
    pb = np.array([norm.sf(d["beta"]) for d in sides])
    centers = [np.zeros(DIM)] + [d["x"] for d in sides]
    wts = np.concatenate([[0.1], 0.9 * pb / pb.sum()])
    ps = []
    for rep in range(N_REP):
        r = np.random.default_rng(1000 * k + rep)
        X = r.standard_normal((N_IS, DIM)) + np.array(centers)[r.choice(len(wts), size=N_IS, p=wts)]
        L = simulate(X, batch=100) * 1e3
        sims["is"] += N_IS
        f = (np.nan_to_num(L.min(1), nan=np.inf) < s).astype(float)
        w = np.exp(-0.5 * np.sum(X ** 2, 1) - log_q(X, centers, wts))
        ps.append(float(np.mean(f * w)))
    ps = np.array(ps)
    p, se = float(ps.mean()), float(ps.std(ddof=1) / np.sqrt(N_REP))
    results[str(k)] = {"spec_mV": s, "p_pooled": p, "ci95_pooled": [p - 1.96 * se, p + 1.96 * se],
                       "reps": ps.tolist(), "sigma_equivalent": float(norm.isf(p)) if p > 0 else None,
                       "beta": [d["beta"] for d in sides], "form_p": float(pb.sum()),
                       "gauss_fit_p": float(norm.sf(k)),
                       "mpfp": [{"lobe": d["lobe"], "beta": d["beta"],
                                 "direction": dict(zip(VAR_NAMES, (d["x"] / d["beta"]).round(3).tolist()))}
                                for d in sides]}
    print(f"D. spec {k} (SNM < {s:.1f} mV): IS P = {p:.3e} [{p-1.96*se:.2e}, {p+1.96*se:.2e}] "
          f"= {results[str(k)]['sigma_equivalent']:.2f} sigma;  Gaussian fit says {norm.sf(k):.2e}")

out = {"sims": sims, "seconds": time.time() - t0, "surrogate_rmse_mV": rmse, "classifier_accuracy": cls_acc,
       "N_TRAIN": N_TRAIN, "N_IS": N_IS, "N_REP": N_REP, "pilot": pil, "specs": results}
(ROOT / "results" / "08_sram_is.json").write_text(json.dumps(out, indent=1))
print("sims used:", sims, f"total {sum(sims.values())}, {time.time()-t0:.0f}s")
