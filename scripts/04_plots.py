#!/usr/bin/env python3
"""Step 4 - plots (saved to results/)"""
import json
import pathlib
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hsmc.ota_mc import DIM, SPECS_MV, VAR_NAMES  # noqa: E402

R = ROOT / "results"
C_MC, C_IS, C_REF, C_FAIL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "legend.frameon": False, "lines.linewidth": 2})

bf = json.loads((R / "02_brute_mc.json").read_text())
vos = np.load(R / "02_brute_mc.npz")["vos"].astype(float) * 1e3
isr = json.loads((R / "03_highsigma_is.json").read_text())
mu, sd = bf["mean_mV"], bf["std_mV"]

fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(-25, 35, 241)
h, e = np.histogram(vos[~np.isnan(vos)], bins=bins)
w = e[1] - e[0]
ax.bar(e[:-1], h, width=w, align="edge", color=C_MC, alpha=0.75, label=f"brute-force MC (N={bf['N']:.0e})")
xx = np.linspace(-25, 35, 600)
ax.plot(xx, bf["N"] * w * norm.pdf(xx, mu, sd), color=C_IS, label=f"Gaussian fit  mu={mu:.2f} mV, sd={sd:.2f} mV")
ax.set_yscale("log"); ax.set_ylim(0.5, h.max() * 3)
for k, s in SPECS_MV.items():
    ax.axvline(s, color=INK2, lw=1, ls="--")
    ax.text(s + 0.3, h.max() * 1.2, f"{k}σ spec = {s} mV", rotation=90, ha="left", va="top", fontsize=8, color=INK2)
ax.set_xlabel("input offset Vos = Vout − VCM  [mV]"); ax.set_ylabel("count (log)")
ax.set_title("5T OTA input offset under local mismatch (GF180, Pelgrom)", color=INK, fontsize=11)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(R / "fig1_offset_hist.png", dpi=150); plt.close(fig)

gp = pickle.load(open(R / "03_gp.pkl", "rb"))
d6 = np.array([isr["mpfp_direction_6sig"][n] for n in VAR_NAMES])
i, j = np.argsort(-np.abs(d6))[:2]
tr = np.load(R / "03_train.npz")
fig, ax = plt.subplots(figsize=(6.4, 6))
g = np.linspace(-9, 9, 181)
A, B = np.meshgrid(g, g)
X = np.zeros((A.size, DIM)); X[:, i] = A.ravel(); X[:, j] = B.ravel()
Z = gp.predict(X).reshape(A.shape)
for r in (3, 4, 5, 6):
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(r * np.cos(t), r * np.sin(t), color=GRID, lw=1.2, zorder=1)
    ax.text(r * 0.707, r * 0.707, f"{r}σ", color=INK2, fontsize=8, ha="left", va="bottom")
cs = ax.contour(A, B, Z, levels=sorted([-SPECS_MV[6], SPECS_MV[3], SPECS_MV[6]]),
                colors=[INK2, C_IS, C_FAIL], linewidths=2, zorder=3)
ax.clabel(cs, fmt=lambda v: f"Vos={v:.1f} mV", fontsize=8)
ax.contourf(A, B, np.abs(Z) > SPECS_MV[6], levels=[0.5, 1.5], colors=[C_FAIL], alpha=0.08, zorder=0)
Xtr, ytr = tr["Xtr"], tr["ytr"] * 1e3
ax.scatter(Xtr[:, i], Xtr[:, j], s=9, color=C_MC, alpha=0.5, lw=0, label="training sims (widened σ×3)", zorder=2)
iss = np.load(R / "03_is_samples_6sig.npz")
ax.scatter(iss["X"][:, i], iss["X"][:, j], s=9, c=np.where(iss["f"] > 0, C_FAIL, C_REF), alpha=0.5, lw=0, zorder=2)
ax.scatter([], [], s=9, color=C_REF, label="IS sims @6σ spec: pass")
ax.scatter([], [], s=9, color=C_FAIL, label="IS sims @6σ spec: fail")
for k in SPECS_MV:
    for m in isr["specs"][str(k)]["mpfp"]:
        x = np.array(m["x"])
        ax.plot(x[i], x[j], marker="*", ms=13, color=INK, mec="white", mew=1, zorder=4, ls="none")
ax.plot([], [], marker="*", ms=11, color=INK, ls="none", label="MPFPs, 3-6σ specs (12-D points, projected)")
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal")
ax.set_xlabel(f"x[{VAR_NAMES[i]}]  (normalised, σ units)"); ax.set_ylabel(f"x[{VAR_NAMES[j]}]  (normalised, σ units)")
ax.set_title("GP surrogate fail boundary, slice with other 10 variables = 0", color=INK, fontsize=11)
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout(); fig.savefig(R / "fig2_boundary.png", dpi=150); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
cv = bf["convergence_3sig"]
n = np.array(cv["n"]); p = np.array(cv["p"])
se = np.sqrt(np.maximum(p * (1 - p), 1e-30) / n)
a1.plot(n, p, color=C_MC, label="brute-force MC")
a1.fill_between(n, np.maximum(p - 1.96 * se, 1e-6), p + 1.96 * se, color=C_MC, alpha=0.15, lw=0)
off = isr["sims"]["train"] + isr["sims"]["validate"] + isr["sims"]["mpfp"]
r0 = isr["specs"]["3"]["reps"][0]
ni = np.arange(1, len(r0["run_p"]) + 1); pi_ = np.array(r0["run_p"]); si = np.array(r0["run_se"])
a1.plot(ni + off, pi_, color=C_IS, label=f"ML-guided IS (incl. {off} setup sims)")
a1.fill_between(ni + off, np.maximum(pi_ - 1.96 * si, 1e-6), pi_ + 1.96 * si, color=C_IS, alpha=0.15, lw=0)
a1.axhline(bf["specs"]["3"]["p_fail"], color=INK2, lw=1, ls="--")
a1.text(n[-1], bf["specs"]["3"]["p_fail"] * 1.06, f"MC final {bf['specs']['3']['p_fail']:.3e}",
        ha="right", va="bottom", fontsize=8, color=INK2)
a1.set_xscale("log"); a1.set_ylim(0, 3 * bf["specs"]["3"]["p_fail"])
a1.set_xlabel("# SPICE simulations"); a1.set_ylabel("P(fail) estimate")
a1.set_title(f"3σ spec (|Vos|>{SPECS_MV[3]} mV): MC vs IS, 95% CI band", color=INK, fontsize=10)
a1.legend(fontsize=8, loc="upper right")
cols = {3: C_MC, 4: C_REF, 5: C_IS, 6: "#6250d6"}
for k in SPECS_MV:
    rr = isr["specs"][str(k)]["reps"][0]
    pk = np.array(rr["run_p"]); sk = np.array(rr["run_se"]); nk = np.arange(1, len(pk) + 1)
    a2.plot(nk, pk, color=cols[k], label=f"{k}σ spec ({SPECS_MV[k]} mV)")
    a2.fill_between(nk, np.maximum(pk - 1.96 * sk, pk * 0.05), pk + 1.96 * sk, color=cols[k], alpha=0.15, lw=0)
a2.set_yscale("log"); a2.set_xlabel("# IS SPICE simulations (one repetition)")
a2.set_ylabel("P(fail) estimate (log)")
a2.set_title("IS converges at the same rate for every sigma level", color=INK, fontsize=10)
a2.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(R / "fig3_convergence.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.6))
ss = np.linspace(15, 33, 200)
ax.plot(ss, norm.sf((ss - mu) / sd) + norm.cdf((-ss - mu) / sd), color=C_REF, lw=1.5,
        label="Gaussian fit to MC (extrapolated)")
ks = list(SPECS_MV)
sv = np.array([SPECS_MV[k] for k in ks])
form = [isr["specs"][str(k)]["first_order_p"] for k in ks]
ax.plot(sv, form, "s", color=INK2, ms=6, mfc="none", label="FORM  Φ(−β) at SPICE-corrected MPFP")
pis = np.array([isr["specs"][str(k)]["p_pooled"] for k in ks])
ci = np.array([isr["specs"][str(k)]["ci95_pooled"] for k in ks])
ax.errorbar(sv + 0.15, pis, yerr=[pis - np.maximum(ci[:, 0], pis * 1e-3), ci[:, 1] - pis], fmt="o",
            color=C_IS, ms=7, capsize=4, lw=1.5, label=f"ML-guided IS (95% CI)")
for k in ks:
    b = bf["specs"][str(k)]
    if b["n_fail"] > 0:
        ax.errorbar(SPECS_MV[k] - 0.15, b["p_fail"], yerr=[[b["p_fail"] - b["ci95"][0]], [b["ci95"][1] - b["p_fail"]]],
                    fmt="D", color=C_MC, ms=6, capsize=4, lw=1.5)
    else:
        ax.plot(SPECS_MV[k] - 0.15, b["ci95"][1], marker="v", color=C_MC, ms=7, ls="none")
ax.plot([], [], "D", color=C_MC, label=f"brute-force MC N={bf['N']:.0e} (95% CI; ▼ = upper bound, 0 fails)")
for k in ks:
    ax.text(SPECS_MV[k], 2e-11, f"{k}σ", ha="center", color=INK2, fontsize=9)
ax.set_yscale("log"); ax.set_ylim(1e-11, 1e-2)
ax.set_xlabel("offset spec  |Vos| > spec  [mV]"); ax.set_ylabel("P(fail)")
ax.set_title("Failure probability vs spec", color=INK, fontsize=11)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(R / "fig4_pfail_vs_spec.png", dpi=150); plt.close(fig)
print("saved fig1..fig4 to", R)
