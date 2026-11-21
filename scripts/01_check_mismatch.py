#!/usr/bin/env python3
"""Step 1 - check the mismatch model"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hsmc.ota_mc import DIM, SIG_BETA, SIG_VTH, VAR_NAMES, sim_batch, simulate  # noqa: E402

N = 3000
rng = np.random.default_rng(1)
nom = sim_batch(np.zeros((1, DIM)))[0]
ours = simulate(rng.standard_normal((N, DIM)))
pdk = simulate(np.zeros((N, DIM)), batch=500, pdk_stat=True)

X = np.vstack([np.zeros(DIM), np.eye(DIM)])
sens = (sim_batch(X)[1:] - nom)

res = {
    "nominal_Vos_mV": nom * 1e3,
    "ours": {"mean_mV": float(np.nanmean(ours) * 1e3), "std_mV": float(np.nanstd(ours) * 1e3),
             "n_nan": int(np.isnan(ours).sum())},
    "pdk_agauss": {"mean_mV": float(np.nanmean(pdk) * 1e3), "std_mV": float(np.nanstd(pdk) * 1e3),
                   "n_unique": int(len(np.unique(np.round(pdk, 9))))},
    "sigma_vth_mV": dict(zip(VAR_NAMES[:6], np.round(SIG_VTH * 1e3, 4).tolist())),
    "sigma_beta_pct": dict(zip(VAR_NAMES[6:], np.round(SIG_BETA * 100, 4).tolist())),
    "sens_mV_per_sigma": dict(zip(VAR_NAMES, np.round(sens * 1e3, 4).tolist())),
    "linear_predicted_std_mV": float(np.sqrt((sens ** 2).sum()) * 1e3),
}
print(json.dumps(res, indent=1))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "01_mismatch_check.json").write_text(json.dumps(res, indent=1))
np.savez(ROOT / "results" / "01_mismatch_check.npz", ours=ours, pdk=pdk)
