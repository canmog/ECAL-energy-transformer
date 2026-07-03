"""Diagnostic: how does relaxed selection (contained-only, my cache_full) differ
from the previous strict selection (contained + single-shower + positive-ref,
cache_sel) on the 50-file set? Reads only scalar branches => CPU-only, fast.

    python _diag_sel.py
"""
import glob
import numpy as np
import uproot

PAT = "/aifs/user/data/lishanglin/datasets/full/*-full.root"
BR = ["mcEne", "kx_N_Shwr", "kx_EleEne", "kx_ShwrStat"]
STOP = 2_000_000   # sample size (events with mcEne read); files are iid


def stat0(a):
    a = np.asarray(a)
    if a.dtype == object:
        return np.array([np.asarray(x).ravel()[0] for x in a], dtype=np.int64)
    return (a[:, 0] if a.ndim > 1 else a).astype(np.int64)


paths = sorted(glob.glob(PAT))
print(f"{len(paths)} files; sampling up to {STOP} truth events")
acc = dict(truth=0, cont=0, cont_ss=0, cont_pr=0, cont_both=0)
loge_cont, loge_both = [], []
seen = 0
for ch in uproot.iterate([f"{p}:tree3dfit" for p in paths], BR,
                         step_size=100000, library="np"):
    mc = np.asarray(ch["mcEne"]).astype(np.float64)
    ns = np.asarray(ch["kx_N_Shwr"]).ravel().astype(np.int64)
    ee = np.asarray(ch["kx_EleEne"]).ravel().astype(np.float64)
    st = stat0(ch["kx_ShwrStat"])
    t = mc > 0
    c = t & ((st & 7) == 7)
    ss = ns == 1
    pr = ee > 0
    acc["truth"] += int(t.sum())
    acc["cont"] += int(c.sum())
    acc["cont_ss"] += int((c & ss).sum())
    acc["cont_pr"] += int((c & pr).sum())
    acc["cont_both"] += int((c & ss & pr).sum())
    loge_cont.append(np.log(np.clip(mc[c], 1e-3, None)))
    loge_both.append(np.log(np.clip(mc[c & ss & pr], 1e-3, None)))
    seen += int(t.sum())
    if seen >= STOP:
        break

lc = np.concatenate(loge_cont)
lb = np.concatenate(loge_both)
T, C = acc["truth"], acc["cont"]
print(f"\n--- among {T} truth events (mcEne>0) ---")
print(f"contained (Stat&7==7)        : {C} ({100*C/T:.1f}% of truth)  <- MY run (cache_full) keeps these")
print(f"\n--- WITHIN the {C} contained events ---")
print(f"single-shower (N_Shwr==1)    : {acc['cont_ss']} ({100*acc['cont_ss']/C:.1f}%)  => multi-shower = {100*(1-acc['cont_ss']/C):.1f}%")
print(f"positive-ref (EleEne>0)      : {acc['cont_pr']} ({100*acc['cont_pr']/C:.1f}%)  => bad-ref      = {100*(1-acc['cont_pr']/C):.1f}%")
print(f"BOTH (prev strict cuts)      : {acc['cont_both']} ({100*acc['cont_both']/C:.1f}% of contained)  <- PREV run (cache_sel) kept these")
print(f"\n=> relaxing single-shower+positive-ref let in {100*(1-acc['cont_both']/C):.1f}% extra (harder) events vs the previous selection")
print(f"\n--- energy spectrum (log mcEne) ---")
print(f"contained-only  (mine) : mean={lc.mean():.3f} std={lc.std():.3f}  geo-mean={np.exp(lc.mean()):.0f} GeV")
print(f"contained+strict(prev) : mean={lb.mean():.3f} std={lb.std():.3f}  geo-mean={np.exp(lb.mean()):.0f} GeV")
