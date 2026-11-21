import math
import os
import pathlib
import re
import subprocess
import uuid
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
WORK.mkdir(parents=True, exist_ok=True)

SIZING = {
    "W1": 7.718e-6, "L1": 0.5e-6,
    "W3": 23.879e-6, "L3": 1.0e-6,
    "W5": 11.165e-6, "L5": 1.0e-6,
    "Itail": 41.888e-6,
}
VDD, VCM = 3.3, 1.65
MAX_FINGER_W = 20e-6
DIFF_EXT = 0.48e-6

# PDK mismatch coefficients (sm141064.spice, .lib fets_mm)
PDK_MM = {"nfet_03v3": {"par_vth": 7.148e-3, "par_k": 7.008e-3},
          "pfet_03v3": {"par_vth": 6.660e-3, "par_k": 2.833e-3}}
PAR_L, PAR_W = 0.15e-6, -0.1e-6

# device table: name, d, g, s, b, type, W key, L key
DEVICES = [
    ("M1", "x", "vp", "tail", "0", "nfet_03v3", "W1", "L1"),
    ("M2", "out", "out", "tail", "0", "nfet_03v3", "W1", "L1"),
    ("M3", "x", "x", "vdd", "vdd", "pfet_03v3", "W3", "L3"),
    ("M4", "out", "x", "vdd", "vdd", "pfet_03v3", "W3", "L3"),
    ("M5", "tail", "nb", "0", "0", "nfet_03v3", "W5", "L5"),
    ("M6", "nb", "nb", "0", "0", "nfet_03v3", "W5", "L5"),
]
NDEV = len(DEVICES)
DIM = 2 * NDEV
VAR_NAMES = [f"Vth_{d[0]}" for d in DEVICES] + [f"beta_{d[0]}" for d in DEVICES]

SPECS_MV = {3: 18.1, 4: 22.6, 5: 27.0, 6: 31.5}


def fails(vos, spec_mv):
    v = np.asarray(vos)
    return (~(np.abs(v) <= spec_mv * 1e-3)).astype(float)


def sigmas(sz=SIZING):
    svt, sk = [], []
    for _, _, _, _, _, typ, wk, lk in DEVICES:
        W, L = sz[wk], sz[lk]
        nf = max(1, math.ceil(W / MAX_FINGER_W))
        weff = nf * (W / nf - PAR_W)
        leff = L - PAR_L
        a = math.sqrt(weff * leff) * 1e6
        svt.append(0.7071 * PDK_MM[typ]["par_vth"] / a)
        sk.append(0.7071 * PDK_MM[typ]["par_k"] / a)
    return np.array(svt), np.array(sk)


SIG_VTH, SIG_BETA = sigmas()


def _model_includes():
    root = pathlib.Path(os.environ.get("PDK_ROOT", "/foss/pdks"))
    d = next(root / c / "libs.tech" / "ngspice" for c in ("gf180mcuD", "gf180mcuC")
             if (root / c / "libs.tech" / "ngspice").is_dir())
    return f'.include "{d}/design.ngspice"\n.lib "{d}/sm141064.ngspice" typical'


WRAPPERS = """
.subckt nmm d g s b w=1u l=1u m=1 ad=0 as=0 pd=0 ps=0 dvt=0 dk=0
m0 d g s b nfet_03v3 w=w l=l m=m ad=ad as=as pd=pd ps=ps delvto=dvt mulu0='1-dk'
.ends nmm
.subckt pmm d g s b w=1u l=1u m=1 ad=0 as=0 pd=0 ps=0 dvt=0 dk=0
m0 d g s b pfet_03v3 w=w l=l m=m ad=ad as=as pd=pd ps=ps delvto=dvt mulu0='1-dk'
.ends pmm
"""


def _dev_line(name, d, g, s, b, typ, W, L, dvt, dk, pdk_stat=False):
    nf = max(1, math.ceil(W / MAX_FINGER_W))
    wf = W / nf
    ad = wf * DIFF_EXT
    pd = 2 * (wf + DIFF_EXT)
    geo = (f"w={wf*1e6:.4f}u l={L*1e6:.4f}u m={nf} ad={ad*1e12:.4f}p as={ad*1e12:.4f}p "
           f"pd={pd*1e6:.4f}u ps={pd*1e6:.4f}u")
    if pdk_stat:
        return f"X{name} {d} {g} {s} {b} {typ} {geo}"
    sub = "nmm" if typ.startswith("n") else "pmm"
    return f"X{name} {d} {g} {s} {b} {sub} {geo} dvt={dvt:.6e} dk={dk:.6e}"


def netlist_batch(X, sz=SIZING, pdk_stat=False):
    lines = ["* 5T OTA offset batch", _model_includes()]
    if pdk_stat:
        lines.append(".param sw_stat_mismatch=1")
    lines += [WRAPPERS, f"Vdd vdd 0 {VDD}", f"Vp vp 0 {VCM}"]
    for i, x in enumerate(X):
        dvt = SIG_VTH * x[:NDEV]
        dk = SIG_BETA * x[NDEV:]
        loc = lambda n: n if n in ("0", "vdd", "vp") else f"{n}_{i}"
        lines.append(f"I{i} vdd nb_{i} {sz['Itail']}")
        for j, (nm, d, g, s, b, typ, wk, lk) in enumerate(DEVICES):
            lines.append(_dev_line(f"{nm}_{i}", loc(d), loc(g), loc(s), loc(b), typ,
                                   sz[wk], sz[lk], dvt[j], dk[j], pdk_stat))
    lines.append(".control\nop")
    lines += [f"print v(out_{i})" for i in range(len(X))]
    lines.append(".endc\n.end\n")
    return "\n".join(lines)


def _run(net, timeout=600):
    tag = uuid.uuid4().hex[:10]
    cir = WORK / f"b_{tag}.cir"
    cir.write_text(net)
    spi = WORK / ".spiceinit"
    if not spi.exists():
        spi.write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
    try:
        r = subprocess.run(["ngspice", "-b", cir.name], cwd=WORK, capture_output=True,
                           text=True, timeout=timeout)
        log = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        log = ""
    cir.unlink(missing_ok=True)
    return log


def _parse(log, n):
    vals = dict((int(m.group(1)), float(m.group(2))) for m in
                re.finditer(r"^v\(out_(\d+)\)\s*=\s*([-+0-9.eE]+)", log, re.MULTILINE))
    return np.array([vals.get(i, np.nan) for i in range(n)])


def sim_batch(X, pdk_stat=False):
    X = np.atleast_2d(X)
    v = _parse(_run(netlist_batch(X, pdk_stat=pdk_stat)), len(X)) - VCM
    if np.isnan(v).all() and len(X) > 1:
        h = len(X) // 2
        return np.concatenate([sim_batch(X[:h], pdk_stat), sim_batch(X[h:], pdk_stat)])
    return v


def _job(args):
    X, pdk_stat = args
    return sim_batch(X, pdk_stat)


def simulate(X, batch=250, procs=None, pdk_stat=False):
    X = np.atleast_2d(X)
    chunks = [X[i:i + batch] for i in range(0, len(X), batch)]
    procs = procs or os.cpu_count()
    if len(chunks) == 1:
        return sim_batch(chunks[0], pdk_stat)
    with Pool(min(procs, len(chunks))) as p:
        out = p.map(_job, [(c, pdk_stat) for c in chunks], chunksize=1)
    return np.concatenate(out)


if __name__ == "__main__":
    import time
    print("sigma_Vth (mV):  ", np.round(SIG_VTH * 1e3, 3))
    print("sigma_beta (%):  ", np.round(SIG_BETA * 100, 3))
    t = time.time()
    print("nominal Vos (mV):", sim_batch(np.zeros((1, DIM)))[0] * 1e3, f"({time.time()-t:.1f}s)")
    rng = np.random.default_rng(0)
    for n in (10, 250):
        t = time.time()
        v = sim_batch(rng.standard_normal((n, DIM)))
        print(f"batch {n}: {time.time()-t:.1f}s, nan={np.isnan(v).sum()}, std={np.nanstd(v)*1e3:.2f} mV")
