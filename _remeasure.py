#!/usr/bin/env python3
"""Re-evaluate saved checkpoints with BOTH estimators (raw np.std and the robust
IQR/1.349 core sigma) + median bias, per energy bin. No training. Imports the code of
the CURRENT working directory, so run it from the matching code dir:
    cd transformer    && python _remeasure.py runs/sw_d192 runs/abl runs/dual_v1
    cd transformer_m4 && python ../transformer/_remeasure.py runs/m4
    cd transformer_m2 && python ../transformer/_remeasure.py runs/m2
Writes <run>/robust.json. SUB env var caps the (random) subsample size used.
"""
import sys, os, json
sys.path.insert(0, ".")
import numpy as np, torch
from utils.config import load_config
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer

def robust(r):
    q75, q25 = np.percentile(r, [75, 25]); return (q75 - q25) / 1.349

bins = np.array([0,10,20,50,100,150,200,300,500,1000,2000,4000.])
runs = sys.argv[1:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
SUB = int(os.environ.get("SUB", "150000"))
print("device", dev, "subsample", SUB)

cache = {}   # cache_dir -> (meta, batches, e_true)
for run in runs:
    cfg, _ = load_config(run + "/config.json", ["device=" + dev, "train.num_workers=0"])
    cd = cfg.paths.cache_dir
    if cd not in cache:
        meta = load_meta(cd); ds = EcalTokens(cd, "test", meta)
        loader = make_loader(ds, cfg, shuffle=True, seed=0)   # random subsample (NOT length-sorted)
        bs = []; n = 0
        for b in loader:
            bs.append({k: v.to(dev) for k, v in b.items()}); n += b["energy"].shape[0]
            if n >= SUB: break
        et = np.concatenate([b["energy"].cpu().numpy() for b in bs])
        cache[cd] = (meta, bs, et)
    meta, bs, et = cache[cd]
    cu = cfg.data.get("concept_use", None); nc = len(cu) if cu else meta["n_concepts"]
    m = EcalTransformer(cfg, nc).to(dev); m.eval()
    m.load_state_dict(torch.load(run + "/best.pt", map_location=dev)["model"])
    ep = []
    with torch.no_grad():
        for b in bs:
            ep.append(m.predict_energy_gev(m(b)["energy"].float()).cpu().numpy())
    ep = np.concatenate(ep); rel = (ep - et) / np.clip(et, 1e-3, None)
    finite = np.isfinite(rel)
    o = {"center": [], "res_raw": [], "res_robust": [], "bias_med": [], "frac_out": [], "count": []}
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (et >= lo) & (et < hi) & finite
        if msk.sum() < 50: continue
        rm = rel[msk]
        o["center"].append(0.5 * (lo + hi)); o["res_raw"].append(float(np.std(rm)))
        o["res_robust"].append(float(robust(rm))); o["bias_med"].append(float(np.median(rm)))
        o["frac_out"].append(float(np.mean(np.abs(rm) > 0.20)))   # tail: |dE/E| > 20%
        o["count"].append(int(msk.sum()))
    json.dump(o, open(run + "/robust.json", "w"), indent=2)
    le = lambda key: np.mean([v for c, v in zip(o["center"], o[key]) if c <= 2000])
    print("%-12s  robust<=2TeV=%.2f%%  raw<=2TeV=%.2f%%  tail<=2TeV=%.1f%%  (N=%d)" % (
        run.split("/")[-1], 100 * le("res_robust"), 100 * le("res_raw"), 100 * le("frac_out"), len(et)))
