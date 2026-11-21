#!/usr/bin/env python3
"""Step 3 - ML-guided importance sampling for 4-6 sigma failure probabilities"""
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
from hsmc.ota_mc import DIM, SPECS_MV, VAR_NAMES, fails, simulate  # noqa: E402

warnings.filterwarnings("ignore", module="sklearn")

N_TRAIN, TRAIN_SCALE = 600, 3.0
N_VAL = 300
N_IS, N_REP = 1000, 5
rng = np.random.default_rng(7)
sims = {"train": 0, "validate": 0, "mpfp": 0, "is": 0}
t0 = time.time()

Xtr = rng.standard_normal((N_TRAIN, DIM)) * TRAIN_SCALE
ytr = simulate(Xtr, batch=50)
sims["train"] = N_TRAIN
ok = ~np.isnan(ytr)
print(f"A. {N_TRAIN} training sims, {np.sum(~ok)} non-converged, {time.time()-t0:.0f}s")

kernel = (DotProduct(sigma_0=1.0) + ConstantKernel(1.0) * RBF(length_scale=np.full(DIM, 5.0),
          length_scale_bounds=(0.1, 1e4)) + WhiteKernel(1e-4, (1e-8, 1e-1)))
gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
gp.fit(Xtr[ok], ytr[ok] * 1e3)
g = lambda x: gp.predict(np.atleast_2d(x))[0]
print("   kernel:", gp.kernel_)

Xv = np.vstack([rng.standard_normal((N_VAL // 2, DIM)),
                rng.standard_normal((N_VAL // 2, DIM)) * TRAIN_SCALE])
yv = simulate(Xv, batch=50) * 1e3
sims["validate"] = N_VAL
pv = gp.predict(Xv)
rmse = float(np.sqrt(np.nanmean((pv - yv) ** 2)))
cls_acc = {k: float(np.nanmean((np.abs(pv) > s) == (np.abs(yv) > s))) for k, s in SPECS_MV.items()}
print(f"B. surrogate RMSE on {N_VAL} held-out sims: {rmse*1e3:.2f} uV "
      f"(range {np.nanmin(yv):.1f}..{np.nanmax(yv):.1f} mV); classifier accuracy {cls_acc}")


def grad_g(x, h=1e-2):
    E = np.eye(DIM) * h
    return (gp.predict(x + E) - gp.predict(x - E)) / (2 * h)


def mpfp(target, iters=100):
    x = np.zeros(DIM)
    for _ in range(iters):
        gr = grad_g(x)
        x_new = (gr @ x - (g(x) - target)) / (gr @ gr) * gr
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    if abs(g(x) - target) > 1e-3 * abs(target):
        return None
    return x


def spice_correct(x, target):
    a = np.array([0.95, 1.0, 1.05])
    v = simulate(x[None, :] * a[:, None]) * 1e3
    sims["mpfp"] += 3
    c = np.polyfit(v, a, 1)
    return x * np.polyval(c, target), v[1]


mp = {}
for k, s in SPECS_MV.items():
    mp[k] = []
    for sign in (+1, -1):
        x = mpfp(sign * s)
        if x is None:
            print(f"C. spec {k}s side {sign:+d}: MPFP search FAILED - side left out of the proposal")
            continue
        xc, v_at_x = spice_correct(x, sign * s)
        beta = float(np.linalg.norm(xc))
        mp[k].append({"side": sign, "x": xc, "beta": beta, "beta_surrogate": float(np.linalg.norm(x)),
                      "spice_Vos_at_surrogate_mpfp_mV": float(v_at_x)})
        print(f"C. spec {k}s side {sign:+d}: beta_surr={np.linalg.norm(x):.3f} -> "
              f"beta_spice={beta:.3f} (SPICE Vos at surrogate MPFP {v_at_x:.2f} mV, target {sign*s})")


def log_q(X, centers, wts):
    comps = [np.log(w) - 0.5 * np.sum((X - c) ** 2, axis=1) for c, w in zip(centers, wts)]
    return logsumexp(np.vstack(comps), axis=0)


results = {}
for k, s in SPECS_MV.items():
    sides = mp[k]
    pb = np.array([norm.sf(d["beta"]) for d in sides])
    centers = [np.zeros(DIM)] + [d["x"] for d in sides]
    wts = np.concatenate([[0.1], 0.9 * pb / pb.sum()])
    reps = []
    for rep in range(N_REP):
        r = np.random.default_rng(1000 * k + rep)
        comp = r.choice(len(wts), size=N_IS, p=wts)
        X = r.standard_normal((N_IS, DIM)) + np.array(centers)[comp]
        v = simulate(X, batch=100)
        sims["is"] += N_IS
        f = fails(v, s)
        w = np.exp(-0.5 * np.sum(X ** 2, axis=1) - log_q(X, centers, wts))
        fw = f * w
        n = np.arange(1, N_IS + 1)
        run = np.cumsum(fw) / n
        run_se = np.sqrt(np.maximum(np.cumsum(fw ** 2) / n - run ** 2, 0) / n)
        p = float(run[-1]); se = float(run_se[-1])
        reps.append({"p": p, "ci95": [p - 1.96 * se, p + 1.96 * se], "rel_ci": 1.96 * se / p if p else None,
                     "n_fail": int(f.sum()), "run_p": run.tolist(), "run_se": run_se.tolist()})
        if rep == 0:
            np.savez_compressed(ROOT / "results" / f"03_is_samples_{k}sig.npz", X=X, vos=v, w=w, f=f)
    ps = np.array([q["p"] for q in reps])
    # pooled estimate over all repetitions (N_REP*N_IS sims)
    p_pool = float(ps.mean()); se_pool = float(ps.std(ddof=1) / np.sqrt(N_REP))
    sig_eq = float(norm.isf(p_pool)) if p_pool > 0 else None
    results[str(k)] = {
        "spec_mV": s, "beta": [d["beta"] for d in sides], "mixture_weights": wts.tolist(),
        "mpfp": [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in d.items()} for d in sides],
        "reps": reps, "p_pooled": p_pool, "ci95_pooled": [p_pool - 1.96 * se_pool, p_pool + 1.96 * se_pool],
        "sigma_equivalent": sig_eq, "first_order_p": float(pb.sum()),
        "sims_per_rep": N_IS,
        "is_sims_for_10pct": float(np.mean([N_IS * (q["rel_ci"] / 0.1) ** 2 for q in reps if q["rel_ci"]])),
        "brute_force_sims_for_10pct": float((1.96 / 0.1) ** 2 * (1 - p_pool) / p_pool) if p_pool else None,
    }
    print(f"D. spec {k}s (|Vos|>{s} mV): IS P = {p_pool:.3e}  95% CI [{results[str(k)]['ci95_pooled'][0]:.2e}, "
          f"{results[str(k)]['ci95_pooled'][1]:.2e}]  (= {sig_eq:.2f} sigma)  reps: "
          + ", ".join(f"{q['p']:.2e}" for q in reps))

d6 = mp[6][0]["x"] / np.linalg.norm(mp[6][0]["x"])
out = {"sims": sims, "seconds": time.time() - t0, "surrogate_rmse_mV": rmse,
       "classifier_accuracy": cls_acc, "kernel": str(gp.kernel_),
       "mpfp_direction_6sig": dict(zip(VAR_NAMES, d6.round(4).tolist())),
       "N_TRAIN": N_TRAIN, "TRAIN_SCALE": TRAIN_SCALE, "N_IS": N_IS, "N_REP": N_REP, "specs": results}
(ROOT / "results" / "03_highsigma_is.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(ROOT / "results" / "03_train.npz", Xtr=Xtr, ytr=ytr, Xv=Xv, yv=yv)
import pickle  # noqa: E402
pickle.dump(gp, open(ROOT / "results" / "03_gp.pkl", "wb"))
print("sims used:", sims, f"total {sum(sims.values())}, {time.time()-t0:.0f}s")
