"""Final four-method comparison figures (f15 sigma, f16 bias) — Gaussian core, fp32.

Curves (same 507k cache_full10 test events unless noted):
  * 3D fit, uncorrected  : kx_EleEne from runs/methodfig2/anchors_test.npz
                           (split replication verified vs the stored EneL2Cor column)
  * pure transformer     : m3_d256 (no physics scaffolding), runs/methodfig2/m3_d256.npz
  * best DNN             : eneRec sel_sup anchored MLP — ITS OWN clean single-file sample,
                           runs/methodfig2/enerec_sel_sup.npz (dashed reference)
  * champion             : v3_d192_ep100 best_robust, per-bin gauss from
                           runs/rescore_ep100/rescore_gauss.json

    python make_final_methods_fig.py     # -> transformerReport/figs/f15,f16 + summary md
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from utils.stats import gauss_core, robust_sigma

BASE = "/aifs/user/data/lishanglin/chenhao"
MF2 = f"{BASE}/ecalTransformer/runs/methodfig2"
OUT = f"{BASE}/transformerReport/figs"
BINS = [0, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 3000, 4000]
E_CUT = 2000.0


def curve_from_resid(e_true, r, nmin=100):
    c, sig, mu = [], [], []
    fin = np.isfinite(r)
    e_true, r = e_true[fin], r[fin]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < nmin:
            continue
        g = gauss_core(r[m])
        c.append(0.5 * (lo + hi)); sig.append(g["sigma"] * 100); mu.append(g["mu"] * 100)
    m2 = e_true <= E_CUT
    g2 = gauss_core(r[m2])
    return (np.array(c), np.array(sig), np.array(mu),
            g2["sigma"] * 100, float(np.mean(np.abs(r) > 0.2)) * 100)


curves = {}  # label -> dict(c, sig, mu, le2, out, color, marker, ls)

z = np.load(f"{MF2}/anchors_test.npz")
et, ele = z["energy"].astype(np.float64), z["ele"].astype(np.float64)
c, s, m, le2, ofr = curve_from_resid(et, (ele - et) / np.clip(et, 1e-3, None))
curves["3D fit, uncorrected (kx_EleEne)"] = dict(
    c=c, sig=s, mu=m, le2=le2, out=ofr, color="#7f7f7f", marker="o", ls="-")

z = np.load(f"{MF2}/m3_d256.npz")
et, ep = z["e_true"].astype(np.float64), z["e_pred"].astype(np.float64)
c, s, m, le2, ofr = curve_from_resid(et, (ep - et) / np.clip(et, 1e-3, None))
curves["pure transformer (m3, no physics)"] = dict(
    c=c, sig=s, mu=m, le2=le2, out=ofr, color="#e6550d", marker="s", ls="-")

z = np.load(f"{MF2}/enerec_sel_sup.npz")
et, ep = z["e_true"].astype(np.float64), z["e_pred"].astype(np.float64)
c, s, m, le2, ofr = curve_from_resid(et, (ep - et) / np.clip(et, 1e-3, None), nmin=60)
curves["best DNN (eneRec, clean single-file)"] = dict(
    c=c, sig=s, mu=m, le2=le2, out=ofr, color="#d62728", marker="^", ls="--")

d = json.load(open(f"{BASE}/ecalTransformer/runs/rescore_ep100/rescore_gauss.json"))
b = d["results"]["v3_ep100"]["bins"]
curves["champion (v3 ep100, physics-grounded)"] = dict(
    c=np.array([x["center"] for x in b]),
    sig=np.array([x["gauss"] for x in b]) * 100,
    mu=np.array([x["gauss_mu"] for x in b]) * 100,
    le2=d["results"]["v3_ep100"]["gauss_le_cut"] * 100,
    out=d["results"]["v3_ep100"]["outlier"] * 100,
    color="#2ca02c", marker="D", ls="-")

os.makedirs(OUT, exist_ok=True)
allsig = np.concatenate([v["sig"] for v in curves.values()])
split = 3.0
need_top = allsig.max() > split

# ---- f15: sigma/E, segmented (page-16 house style) if any curve exceeds the zoom
if need_top:
    fig, (axT, axB) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.8),
                                   gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.0})
    axes = (axT, axB)
else:
    fig, axB = plt.subplots(figsize=(7.0, 4.4))
    axes = (axB,)
for lab, v in curves.items():
    for ax in axes:
        ax.plot(v["c"], v["sig"], marker=v["marker"], color=v["color"], ls=v["ls"],
                ms=5, lw=1.8, label=lab)
if need_top:
    axT.set_ylim(split, allsig.max() * 1.1)
    axT.legend(fontsize=8, loc="upper left")
    axT.set_title(r"Gaussian-core $\sigma/E$ (fp32) --- final method comparison")
else:
    axB.legend(fontsize=8, loc="upper left")
    axB.set_title(r"Gaussian-core $\sigma/E$ (fp32) --- final method comparison")
axB.set_ylim(0.85, split)
for ax in axes:
    ax.set_xscale("log"); ax.grid(True, alpha=.3, which="both")
    ax.axvline(2000, color="gray", ls=":", lw=1, alpha=.6)
axB.set_xlabel(r"$E_{\rm true}$ [GeV]")
fig.supylabel(r"Gaussian-core $\sigma/E$ [%]", fontsize=11)
fig.savefig(f"{OUT}/f15_final_methods.pdf", bbox_inches="tight"); plt.close()

# ---- f16: fitted core mean (single axis; the uncorrected fit's TeV dive IS the story)
plt.figure(figsize=(7.0, 4.4)); ax = plt.gca()
ax.axhline(0, color="k", lw=.8)
for lab, v in curves.items():
    ax.plot(v["c"], v["mu"], marker=v["marker"], color=v["color"], ls=v["ls"],
            ms=5, lw=1.8, label=lab)
ax.set_xscale("log"); ax.set_xlabel(r"$E_{\rm true}$ [GeV]")
ax.set_ylabel(r"Gaussian-fit mean $\mu$ [%]")
ax.set_title("Core calibration (fp32) --- final method comparison")
ax.axvline(2000, color="gray", ls=":", lw=1, alpha=.6)
ax.grid(True, alpha=.3, which="both"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/f16_final_bias.pdf"); plt.close()

lines = ["| method | gauss <=2 TeV | outlier |,", "|---|---|---|"]
for lab, v in curves.items():
    lines.append(f"| {lab} | {v['le2']:.2f}% | {v['out']:.2f}% |")
    print(f"{lab:45s} gauss<=2TeV {v['le2']:.2f}%  outlier {v['out']:.2f}%  "
          f"per-bin: " + " ".join(f"{c:.0f}:{s:.2f}" for c, s in zip(v["c"], v["sig"])))
with open(f"{MF2}/final_summary.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {OUT}/f15_final_methods.pdf, f16_final_bias.pdf")
