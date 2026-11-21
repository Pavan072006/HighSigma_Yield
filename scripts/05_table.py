#!/usr/bin/env python3
import json
import pathlib

from scipy.stats import norm

R = pathlib.Path(__file__).resolve().parent.parent / "results"
bf = json.loads((R / "02_brute_mc.json").read_text())
isr = json.loads((R / "03_highsigma_is.json").read_text())
setup = isr["sims"]["train"] + isr["sims"]["validate"]
nbf = bf["N"]


def e(x):
    return f"{x:.2e}"


rows = ["| spec (target σ) | \\|Vos\\| > | brute force MC, N=%.0e: P [95%% CI] (fails) | ML-guided IS: P [95%% CI] | σ-equiv. (IS) | FORM Φ(−β) | Gaussian fit | IS SPICE sims | brute-force sims for ±10%% |" % nbf,
        "|---|---|---|---|---|---|---|---|---|"]
for k, s in isr["specs"].items():
    b = bf["specs"][k]
    if b["n_fail"]:
        bft = f"{e(b['p_fail'])} [{e(b['ci95'][0])}, {e(b['ci95'][1])}] ({b['n_fail']})"
    else:
        bft = f"< {e(b['ci95'][1])} (0 fails)"
    n_is = s["sims_per_rep"] * isr["N_REP"] + sum(1 for _ in s["mpfp"]) * 3
    rows.append(f"| {k}σ (Φ(−{k}) = {e(norm.sf(int(k)))}) | {s['spec_mV']} mV | {bft} | "
                f"**{e(s['p_pooled'])}** [{e(s['ci95_pooled'][0])}, {e(s['ci95_pooled'][1])}] | "
                f"{s['sigma_equivalent']:.2f} | {e(s['first_order_p'])} | {e(b['p_gauss_fit'])} | "
                f"{n_is} (+{setup} shared) | {e(s['brute_force_sims_for_10pct'])} |")
txt = "\n".join(rows)
txt += (f"\n\nShared setup: {isr['sims']['train']} training + {isr['sims']['validate']} validation sims; "
        f"surrogate held-out RMSE {isr['surrogate_rmse_mV']*1e3:.2f} µV. "
        f"IS sims for ±10% at one sigma level (from observed CI widths): "
        + ", ".join(f"{k}σ ≈ {v['is_sims_for_10pct']:.0f}" for k, v in isr["specs"].items())
        + f". Brute force: {nbf:.0e} sims in {bf['seconds']/60:.0f} min on 12 cores.")
print(txt)
(R / "results_table.md").write_text(txt + "\n")
