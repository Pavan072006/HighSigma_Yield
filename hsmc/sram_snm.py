"""6T SRAM read static noise margin under local mismatch (GF180 3.3 V devices)"""
import json
import math
import os
import uuid
import subprocess
from multiprocessing import Pool

import numpy as np

from hsmc.ota_mc import PAR_L, PAR_W, PDK_MM, ROOT, WORK, WRAPPERS, _dev_line, _model_includes

VDD = 3.3
STEP = 0.02
SIZING = {"W_pu": 0.3e-6, "W_pd": 0.8e-6, "W_pg": 0.4e-6, "L": 0.28e-6}

# half cell: name, d, g, s, b, type, W key
HALF = [
    ("PU", "y", "in", "vdd", "vdd", "pfet_03v3", "W_pu"),
    ("PD", "y", "in", "0", "0", "nfet_03v3", "W_pd"),
    ("PG", "vdd", "vdd", "y", "0", "nfet_03v3", "W_pg"),
]
DEVICES = [(s, *d) for s in ("L", "R") for d in HALF]
NDEV = len(DEVICES)
DIM = 2 * NDEV
VAR_NAMES = [f"Vth_{d[1]}{d[0]}" for d in DEVICES] + [f"beta_{d[1]}{d[0]}" for d in DEVICES]


def sigmas(sz=SIZING):
    svt, sk = [], []
    for _, _, _, _, _, _, typ, wk in DEVICES:
        a = math.sqrt((sz[wk] - PAR_W) * (sz["L"] - PAR_L)) * 1e6
        svt.append(0.7071 * PDK_MM[typ]["par_vth"] / a)
        sk.append(0.7071 * PDK_MM[typ]["par_k"] / a)
    return np.array(svt), np.array(sk)


SIG_VTH, SIG_BETA = sigmas()


def netlist_batch(X, out):
    lines = ["* 6T SRAM read SNM batch", _model_includes(), WRAPPERS,
             f"Vdd vdd 0 {VDD}", "Vin in 0 0"]
    vecs = []
    for i, x in enumerate(X):
        dvt, dk = SIG_VTH * x[:NDEV], SIG_BETA * x[NDEV:]
        for j, (side, nm, d, g, s, b, typ, wk) in enumerate(DEVICES):
            loc = lambda n: f"{n}_{i}{side}" if n == "y" else n
            lines.append(_dev_line(f"{nm}{side}_{i}", loc(d), loc(g), loc(s), loc(b), typ,
                                   SIZING[wk], SIZING["L"], dvt[j], dk[j]))
        vecs += [f"v(y_{i}L)", f"v(y_{i}R)"]
    lines += [".control", f"dc Vin 0 {VDD} {STEP}", "set wr_singlescale", "set wr_vecnames",
              "option numdgt=7", f"wrdata {out} " + " ".join(vecs), ".endc", ".end", ""]
    return "\n".join(lines)


def snm_lobes(x, fl, fr):
    s = math.sqrt(2)
    u1, v1 = (x - fl) / s, (x + fl) / s
    u2, v2 = (fr - x) / s, (fr + x) / s
    o = np.argsort(u2)
    u = np.linspace(max(u1.min(), u2.min()), min(u1.max(), u2.max()), 1500)
    d = np.interp(u, u1, v1) - np.interp(u, u2[o], v2[o])
    return d.max() / s, (-d).max() / s


def sim_batch(X):
    X = np.atleast_2d(X)
    tag = uuid.uuid4().hex[:10]
    cir, out = WORK / f"s_{tag}.cir", WORK / f"s_{tag}.txt"
    cir.write_text(netlist_batch(X, out.name))
    spi = WORK / ".spiceinit"
    if not spi.exists():
        spi.write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
    try:
        subprocess.run(["ngspice", "-b", cir.name], cwd=WORK, capture_output=True, timeout=900)
        D = np.loadtxt(out, skiprows=1)
    except Exception:
        D = None
    cir.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    if D is None or D.ndim != 2 or D.shape[1] != 1 + 2 * len(X):
        if len(X) == 1:
            return np.full((1, 2), np.nan)
        h = len(X) // 2
        return np.vstack([sim_batch(X[:h]), sim_batch(X[h:])])
    x = D[:, 0]
    return np.array([snm_lobes(x, D[:, 1 + 2 * i], D[:, 2 + 2 * i]) for i in range(len(X))])


def _job(X):
    return sim_batch(X)


def simulate(X, batch=200, procs=None):
    X = np.atleast_2d(X)
    chunks = [X[i:i + batch] for i in range(0, len(X), batch)]
    if len(chunks) == 1:
        return sim_batch(chunks[0])
    with Pool(min(procs or os.cpu_count(), len(chunks))) as p:
        return np.vstack(p.map(_job, chunks, chunksize=1))


# specs the usual way: Gaussian fit to a 10k pilot MC, spec_k = mean - k * std
def pilot_specs(n=10000):
    f = ROOT / "results" / "07_sram_pilot.json"
    if f.exists():
        return json.loads(f.read_text())
    s = simulate(np.random.default_rng(0).standard_normal((n, DIM))).min(1) * 1e3
    s = s[np.isfinite(s)]
    mu, sd = float(s.mean()), float(s.std())
    out = {"n": int(s.size), "mean_mV": mu, "std_mV": sd,
           "specs_mV": {str(k): mu - k * sd for k in (3, 4, 5, 6)}}
    f.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import time
    print("sigma_Vth (mV):", np.round(SIG_VTH * 1e3, 2))
    t = time.time()
    print("nominal lobes (mV):", np.round(sim_batch(np.zeros((1, DIM)))[0] * 1e3, 1), f"({time.time()-t:.1f}s)")
    rng = np.random.default_rng(0)
    for n in (200, 2400):
        t = time.time()
        L = simulate(rng.standard_normal((n, DIM)))
        s = L.min(1) * 1e3
        print(f"{n} samples: {time.time()-t:.1f}s, nan={np.isnan(s).sum()}, SNM mean {np.nanmean(s):.1f} "
              f"std {np.nanstd(s):.1f} mV, min {np.nanmin(s):.1f}")
