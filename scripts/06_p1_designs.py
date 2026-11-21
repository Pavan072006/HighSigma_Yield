#!/usr/bin/env python3
"""Step 6 - offset yield of the final designs from LLM_Analog_Sizing"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFINE = ROOT.parent / "analog_agent" / "results" / "refine_groq-openai-gpt-oss-120b-effort-low.json"
OUT = ROOT / "results" / "p1"
OUT.mkdir(parents=True, exist_ok=True)
N_MC = 4000
SPEC_MV = 10.0


def gauss(spec, mu, sd):
    return float(norm.sf((spec - mu) / sd) + norm.cdf((-spec - mu) / sd))


designs = {r["spec"]: r["trace"][-1]["sizing"] for r in json.loads(REFINE.read_text())}
designs["hand"] = {"W1": 7.718e-6, "L1": 0.5e-6, "W3": 23.879e-6, "L3": 1.0e-6,
                   "W5": 11.165e-6, "L5": 1.0e-6, "Itail": 41.888e-6}

# quick MC per design (fresh process so the module picks up the sizing)
MC = """
import json, sys, numpy as np
sys.path.insert(0, %r)
from hsmc.ota_mc import DIM, SIG_VTH, simulate
v = simulate(np.random.default_rng(1).standard_normal((%d, DIM)))
v = v[np.isfinite(v)] * 1e3
print(json.dumps(dict(mu=float(v.mean()), sd=float(v.std()), n=int(v.size), svt1=float(SIG_VTH[0] * 1e3),
                      svt3=float(SIG_VTH[2] * 1e3), p_mc=float(np.mean(np.abs(v) > %r)))))
"""
rows = {}
for name, sz in designs.items():
    f = OUT / f"sizing_{name}.json"
    f.write_text(json.dumps(sz))
    env = dict(os.environ, HSMC_SIZING=str(f))
    r = subprocess.run([sys.executable, "-c", MC % (str(ROOT), N_MC, SPEC_MV)], env=env,
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    d["p_gauss"] = gauss(SPEC_MV, d["mu"], d["sd"])
    d["sigma_level"] = float(norm.isf(d["p_gauss"]))
    rows[name] = d
    print(f"{name:5} mean {d['mu']:.2f} mV, sigma {d['sd']:.2f} mV -> yield sigma {d['sigma_level']:.2f}", flush=True)

# full ML-guided importance sampling on the worst design
worst = max((k for k in rows if k != "hand"), key=lambda k: rows[k]["sd"])
sd = rows[worst]["sd"]
specs = {k: round(k * sd, 3) for k in (3, 4, 5, 6)}
wdir = OUT / f"is_{worst}"
env = dict(os.environ, HSMC_SIZING=str(OUT / f"sizing_{worst}.json"), HSMC_SPECS=json.dumps(specs),
           HSMC_OUT=str(wdir))
subprocess.run([sys.executable, str(ROOT / "scripts" / "03_highsigma_is.py")], env=env, check=True)
isr = json.loads((wdir / "03_highsigma_is.json").read_text())["specs"]

lines = [f"Offset spec |Vos| <= {SPEC_MV:.0f} mV. sigma(Vos) from {N_MC} Monte Carlo sims per design.", "",
         "| design | mean Vos mV | sigma(Vos) mV | sigma_Vth M1 / M3 mV | P(fail) Gaussian | P(fail) MC | yield in sigma |",
         "|---|---|---|---|---|---|---|"]
for k, d in rows.items():
    lines.append(f"| {k} | {d['mu']:.2f} | {d['sd']:.2f} | {d['svt1']:.2f} / {d['svt3']:.2f} | "
                 f"{d['p_gauss']:.2e} | {d['p_mc']:.2e} | {d['sigma_level']:.2f} |")
lines += ["", f"ML-guided importance sampling on the worst design ({worst}), specs at k * sigma(Vos):", "",
          "| k | spec mV | IS P(fail) [95% CI] | Gaussian (with mean) |", "|---|---|---|---|"]
for k, s in specs.items():
    r = isr[str(k)]
    lines.append(f"| {k} | {s:.1f} | {r['p_pooled']:.2e} [{r['ci95_pooled'][0]:.2e}, {r['ci95_pooled'][1]:.2e}] | "
                 f"{gauss(s, rows[worst]['mu'], sd):.2e} |")
(ROOT / "results" / "p1_designs.md").write_text("\n".join(lines) + "\n")
keep = ("spec_mV", "p_pooled", "ci95_pooled", "sigma_equivalent")
isr = {k: {f: v[f] for f in keep} for k, v in isr.items()}
(ROOT / "results" / "p1_designs.json").write_text(json.dumps(dict(rows=rows, worst=worst, is_results=isr), indent=1))
print("\n".join(lines))
