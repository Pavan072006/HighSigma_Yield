#!/usr/bin/env python3
"""Step 9 - SRAM read margin: figure and table"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

R = pathlib.Path(__file__).resolve().parent.parent / "results"
mc = json.loads((R / "07_sram_mc.json").read_text())
isr = json.loads((R / "08_sram_is.json").read_text())
lobes = np.load(R / "07_sram_mc.npz")["lobes"]
snm = lobes.min(1)
snm = snm[np.isfinite(snm)]
pil = mc["pilot"]
mu, sd = pil["mean_mV"], pil["std_mV"]
ks = sorted(int(k) for k in isr["specs"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
h, e = np.histogram(snm, bins=150)
c = 0.5 * (e[1:] + e[:-1])
ax[0].semilogy(c, h / h.sum() / (e[1] - e[0]), ".", ms=3, color="#1f4fbf", label=f"brute-force MC (N={len(snm):.0e})")
xx = np.linspace(e[0], e[-1], 400)
ax[0].semilogy(xx, norm.pdf(xx, mu, sd), "-", color="#d9822b", label="Gaussian fit to 10k pilot")
for k in ks:
    ax[0].axvline(isr["specs"][str(k)]["spec_mV"], color="0.6", lw=0.7, ls=":")
ax[0].set_ylim(1e-7, None)
ax[0].set_xlabel("read SNM = min(lobe L, lobe R)  [mV]")
ax[0].set_ylabel("density [1/mV]")
ax[0].legend(fontsize=8)
ax[0].set_title("Left tail is heavier than the Gaussian fit")

pg = [norm.sf(k) for k in ks]
pi = [isr["specs"][str(k)]["p_pooled"] for k in ks]
ci = np.array([isr["specs"][str(k)]["ci95_pooled"] for k in ks])
pf = [isr["specs"][str(k)]["form_p"] for k in ks]
ax[1].semilogy(ks, pg, "s--", color="#d9822b", label="Gaussian fit: Phi(-k)")
ax[1].errorbar(ks, pi, yerr=[np.array(pi) - ci[:, 0], ci[:, 1] - np.array(pi)], fmt="o-", color="#1f4fbf",
               capsize=3, label="ML-guided IS (5k sims per spec)")
ax[1].semilogy(ks, pf, "x:", color="0.4", label="FORM, sum over both lobes")
kb = [k for k in ks if mc["specs"][str(k)]["n_fail"] > 0]
pb = [mc["specs"][str(k)]["p_fail"] for k in kb]
cb = np.array([mc["specs"][str(k)]["ci95"] for k in kb])
ax[1].errorbar(kb, pb, yerr=[np.array(pb) - cb[:, 0], cb[:, 1] - np.array(pb)], fmt="^", color="#2a9d4b",
               capsize=3, label="brute-force MC")
ax[1].set_xticks(ks, [f"{k}σ spec\n{isr['specs'][str(k)]['spec_mV']:.0f} mV" for k in ks], fontsize=8)
ax[1].set_ylabel("P(read failure)")
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=8)
rat = np.array(pi) / np.array(pg)
ax[1].set_title(f"Specs set from a Gaussian fit fail {rat.min():.0f}-{rat.max():.0f}x more often")
fig.tight_layout()
fig.savefig(R / "fig5_sram_snm.png", dpi=130)

lines = [f"6T SRAM read SNM, GF180 3.3 V, 12 mismatch variables. Pilot fit: {mu:.1f} ± {sd:.2f} mV; "
         f"spec_k = mean - k·std.", "",
         "| spec | SNM < mV | Gaussian fit | brute-force MC (N=%.0e) [95%% CI] | ML-guided IS [95%% CI] | real sigma | IS / Gaussian |"
         % mc["N"], "|---|---|---|---|---|---|---|"]
for k in ks:
    s, b = isr["specs"][str(k)], mc["specs"][str(k)]
    bt = f"{b['p_fail']:.2e} [{b['ci95'][0]:.1e}, {b['ci95'][1]:.1e}] ({b['n_fail']})" if b["n_fail"] else \
        f"< {b['ci95'][1]:.1e} (0 fails)"
    lines.append(f"| {k}σ | {s['spec_mV']:.1f} | {norm.sf(k):.2e} | {bt} | {s['p_pooled']:.2e} "
                 f"[{s['ci95_pooled'][0]:.1e}, {s['ci95_pooled'][1]:.1e}] | {s['sigma_equivalent']:.2f} | "
                 f"{s['p_pooled'] / norm.sf(k):.0f}x |")
lines += ["", f"Lobe correlation {mc['lobe_corr']:+.2f}; IS sims {sum(isr['sims'].values())}, "
          f"surrogate RMSE {np.round(isr['surrogate_rmse_mV'], 2).tolist()} mV."]
(R / "sram_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
