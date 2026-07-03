"""Figures for the 2026-07-02 Gaussian-core addendum (report + slides).

    python make_gauss_figs.py --summary     # f9-f12 + f14 from rescore_gauss.json (fast)
    python make_gauss_figs.py --fits        # f13 fit-overlay grid (model inference; CPU ok)

Outputs -> transformerReport/figs/ (the slides' graphicspath also includes that dir):
    f9_gauss_methods.pdf    Gaussian-core sigma/E vs E: v2cham / sw_d192 / m1_d320 / 3D fit
    f10_gauss_vs_robust.pdf estimator effect (gauss vs robust vs raw), v2cham and 3D fit
    f11_gauss_bias.pdf      fitted Gaussian mean vs E (calibration)
    f12_rescore_bars.pdf    gauss & robust core <=2 TeV, every rescored checkpoint
    f13_gauss_fits.pdf      residual histograms + iterated Gaussian fits, 3 methods x 4 bins
    f14_outlier_rates.png   copied from the outlier study (rate vs variable deciles)

--fits draws the two transformer rows from a per-bin SUBSAMPLE (default 8000 ev/bin,
CPU-friendly); every quoted number still comes from the full-sample rescore JSON or
the per-panel fit itself. The 3D-fit row uses ALL events (no model needed).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from utils.config import NS
from utils.stats import gauss_core, robust_sigma
from data.dataset import EcalTokens, load_meta, collate
from models.model import EcalTransformer

BASE = "/aifs/user/data/lishanglin/chenhao"
CACHE = f"{BASE}/transformer/cache_full10"
# fp32 re-scoring = the numbers of record; rescore_d320 overrides the v3_d320 row
# that the main fp32 pass caught mid-training (epoch 10).
RESCORE = f"{BASE}/transformer_v3/runs/rescore_fp32/rescore_gauss.json"
RESCORE_EXTRA = f"{BASE}/transformer_v3/runs/rescore_d320/rescore_gauss.json"
OUT = f"{BASE}/transformerReport/figs"
DISPLAY_BINS = [(50, 100), (300, 500), (1000, 2000), (3000, 4000)]

STYLE = {  # label -> (rescore key, color, marker, linestyle)
    "v2cham (champion)": ("v2cham_best", "#2ca02c", "D", "-"),
    "sw_d192 (v1 best)": ("sw_d192", "#1f77b4", "s", "--"),
    "m1_d320":           ("m1_d320", "#9467bd", "v", "-."),
    "3D fit (kx_EneL2Cor)": ("3dfit_EneL2Cor", "#7f7f7f", "o", "-"),
}


def bins_of(res, lab):
    b = res[lab]["bins"]
    c = np.array([x["center"] for x in b])
    return c, {k: np.array([x[k] for x in b]) for k in
               ("gauss", "gauss_mu", "robust", "raw")}


def fig_summary(res):
    os.makedirs(OUT, exist_ok=True)
    # ---- f9: Gaussian-core methods comparison. SEGMENTED y-axis (house style of
    # fig_v2_methods_robust): top = the 3D-fit TeV blow-up, bottom = a 0.9-2.0% zoom
    # where the three transformer curves actually separate.
    fig, (axT, axB) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.8),
                                   gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.0})
    for lab, (key, col, mk, ls) in STYLE.items():
        c, v = bins_of(res, key)
        for ax in (axT, axB):
            ax.plot(c, v["gauss"] * 100, marker=mk, color=col, ls=ls, ms=5, lw=1.8,
                    label=lab)
    axT.set_ylim(2.0, 7.7); axT.set_yticks([3, 5, 7])
    axB.set_ylim(0.9, 2.0); axB.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    axT.set_title(r"Gaussian-core $\sigma/E$ (iterative $\pm2\sigma$ fit, fp32) --- "
                  r"methods compared")
    axT.legend(fontsize=8.5, loc="upper left")
    for ax in (axT, axB):
        ax.set_xscale("log"); ax.grid(True, alpha=.3, which="both")
        ax.axvline(2000, color="gray", ls=":", lw=1, alpha=.6)
    axB.text(2060, 0.95, "2 TeV", color="gray", fontsize=7)
    axB.set_xlabel(r"$E_{\rm true}$ [GeV]")
    fig.supylabel(r"Gaussian-core $\sigma/E$ [%]", fontsize=11)
    fig.savefig(f"{OUT}/f9_gauss_methods.pdf", bbox_inches="tight"); plt.close()

    # ---- f10: estimator effect, champion vs 3D fit
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, lab in zip(axes, ["v2cham (champion)", "3D fit (kx_EneL2Cor)"]):
        key = STYLE[lab][0]
        c, v = bins_of(res, key)
        ax.plot(c, v["gauss"] * 100, "o-", color="#2ca02c", ms=4, label=r"Gaussian core ($\pm2\sigma$ fit)")
        ax.plot(c, v["robust"] * 100, "d-", color="#1f77b4", ms=4, alpha=.8, label="robust core (IQR/1.349)")
        ax.plot(c, v["raw"] * 100, "s--", color="#d62728", ms=4, alpha=.6, label="raw std")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\rm true}$ [GeV]"); ax.set_ylabel(r"$\sigma/E$ [%]")
        ax.set_title(lab); ax.grid(True, alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("The estimator matters: same events, three width definitions (fp32)", y=1.0)
    plt.tight_layout(); plt.savefig(f"{OUT}/f10_gauss_vs_robust.pdf"); plt.close()

    # ---- f11: fitted Gaussian mean (calibration) vs E. Segmented like f9: the 3D
    # fit's TeV drift (+6%) otherwise crushes the transformer curves (within +/-2%).
    fig, (axT, axB) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.8),
                                   gridspec_kw={"height_ratios": [1, 1.7], "hspace": 0.0})
    for lab, (key, col, mk, ls) in STYLE.items():
        c, v = bins_of(res, key)
        for ax in (axT, axB):
            ax.plot(c, v["gauss_mu"] * 100, marker=mk, color=col, ls=ls, ms=5, lw=1.8,
                    label=lab)
    axB.axhline(0, color="k", lw=.8)
    axT.set_ylim(1.8, 6.6); axT.set_yticks([2, 4, 6])
    axB.set_ylim(-2.5, 1.8); axB.set_yticks([-2, -1, 0, 1])
    axT.set_title("Core calibration: fitted Gaussian mean vs energy (fp32)")
    axT.legend(fontsize=8.5, loc="upper left")
    for ax in (axT, axB):
        ax.set_xscale("log"); ax.grid(True, alpha=.3, which="both")
        ax.axvline(2000, color="gray", ls=":", lw=1, alpha=.6)
    axB.set_xlabel(r"$E_{\rm true}$ [GeV]")
    fig.supylabel(r"Gaussian-fit mean $\mu$ [%]", fontsize=11)
    fig.savefig(f"{OUT}/f11_gauss_bias.pdf", bbox_inches="tight"); plt.close()

    # ---- f12: all rescored checkpoints, gauss + robust core <=2 TeV
    rows = [(lab, s) for lab, s in res.items() if "error" not in s]
    rows.sort(key=lambda t: t[1]["gauss_le_cut"])
    labs = [t[0] for t in rows]
    gv = np.array([t[1]["gauss_le_cut"] for t in rows]) * 100
    rv = np.array([t[1]["robust_le_cut"] for t in rows]) * 100
    y = np.arange(len(rows))
    plt.figure(figsize=(7.4, 0.34 * len(rows) + 1.6)); ax = plt.gca()
    ax.barh(y, gv, height=0.62, color="#2ca02c", alpha=.85, label="Gaussian core")
    ax.plot(rv, y, "d", color="#1f77b4", ms=6, label="robust core (IQR/1.349)")
    for i, g in enumerate(gv):
        ax.text(g + 0.015, i, f"{g:.2f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rv.max(), gv.max()) * 1.14)
    ax.set_xlabel(r"core $\sigma/E$ ($E\leq 2$ TeV) [%]")
    ax.set_title("Gaussian-core re-scoring, fp32 (same 507k test events)", fontsize=11)
    ax.grid(True, axis="x", alpha=.3); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(f"{OUT}/f12_rescore_bars.pdf"); plt.close()

    # ---- f14: copy the outlier-rate panel from the outlier study (if present)
    src = f"{BASE}/transformer_v3/runs/outliers/outlier_rates.png"
    if os.path.exists(src):
        shutil.copyfile(src, f"{OUT}/f14_outlier_rates.png")
    print("wrote f9-f12 (+f14 if available) ->", OUT)


@torch.no_grad()
def model_residuals(run_dir, ckpt, ds, idx, device="cpu", chunk=64):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = NS(json.load(f))
    meta = load_meta(CACHE)
    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, ckpt),
                                     map_location=device)["model"])
    model.eval()
    ep, et = [], []
    order = np.argsort([ds.off[i + 1] - ds.off[i] for i in idx])  # near-uniform padding
    idx = np.asarray(idx)[order]
    for s in range(0, len(idx), chunk):
        batch = collate([ds[int(i)] for i in idx[s:s + chunk]])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        ep.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
        et.append(batch["energy"].cpu().numpy())
    e_pred, e_true = np.concatenate(ep), np.concatenate(et)
    return (e_pred - e_true) / np.clip(e_true, 1e-3, None), e_true


def fig_fits(res, per_bin=8000):
    os.makedirs(OUT, exist_ok=True)
    ROWS = [("v2cham (champion)", f"{BASE}/transformer_v2cham/runs/v2cham_d192",
             "best.pt", f"subsample {per_bin}/bin"),
            ("sw_d192 (v1 best)", f"{BASE}/transformer/runs/sw_d192",
             "best.pt", f"subsample {per_bin}/bin"),
            ("3D fit (kx_EneL2Cor)", None, None, "all events")]
    cache_npz = f"{BASE}/transformer_v3/runs/fitfigs_resid.npz"
    rows = []  # (row label, residuals, e_true, note)
    if os.path.exists(cache_npz):
        print("using cached residuals:", cache_npz)
        z = np.load(cache_npz)
        rows = [(lab, z[f"r_{i}"], z[f"e_{i}"], note)
                for i, (lab, _, _, note) in enumerate(ROWS)]
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
        rng = np.random.default_rng(0)
        meta = load_meta(CACHE)
        ds = EcalTokens(CACHE, "test", meta)
        e_all = ds.energy
        idx = np.concatenate([
            rng.choice(np.nonzero((e_all >= lo) & (e_all < hi))[0],
                       size=min(per_bin, int(((e_all >= lo) & (e_all < hi)).sum())),
                       replace=False) for lo, hi in DISPLAY_BINS])
        for lab, run_dir, ck, note in ROWS[:2]:
            print("inference:", lab, f"({len(idx)} events, cpu)")
            r, et = model_residuals(run_dir, ck, ds, idx)
            rows.append((lab, r, et, note))
        z = np.load(f"{BASE}/transformer_m2/cache_m2/test.npz")
        et3, anc = z["energy"].astype(np.float64), z["anchor"].astype(np.float64)
        rows.append((ROWS[2][0], (anc - et3) / np.clip(et3, 1e-3, None), et3, ROWS[2][3]))
        np.savez(cache_npz,
                 **{f"r_{i}": r for i, (_, r, _, _) in enumerate(rows)},
                 **{f"e_{i}": e for i, (_, _, e, _) in enumerate(rows)})

    # Pre-fit every panel, then share ONE x-range per energy column (5 sigma_G of the
    # widest method, i.e. the 3D fit) so widths are visually comparable across rows.
    stats = {}
    for i, (lab, r, et, note) in enumerate(rows):
        for j, (lo, hi) in enumerate(DISPLAY_BINS):
            rb = r[(et >= lo) & (et < hi)]
            rb = rb[np.isfinite(rb)]
            stats[i, j] = (rb, gauss_core(rb), robust_sigma(rb))
    col_half = [float(np.clip(max(5 * stats[i, j][1]["sigma"] for i in range(len(rows))),
                              0.05, 0.5)) for j in range(len(DISPLAY_BINS))]

    fig, axes = plt.subplots(len(rows), len(DISPLAY_BINS),
                             figsize=(4.0 * len(DISPLAY_BINS), 2.9 * len(rows)))
    for i, (lab, r, et, note) in enumerate(rows):
        for j, (lo, hi) in enumerate(DISPLAY_BINS):
            ax = axes[i, j]
            rb, g, rob = stats[i, j]
            sig, mu = g["sigma"], g["mu"]
            half = col_half[j]
            # adaptive bin count: ~4 bins per sigma_G so a narrow core inside a wide
            # shared window is still resolved into a Gaussian shape
            nb = int(np.clip(round(2 * half / (sig / 4)), 60, 240))
            ax.hist(rb[(rb > -half) & (rb < half)], bins=nb, range=(-half, half),
                    color="#9ecae1", edgecolor="none")
            # Gaussian normalised to the CORE events (counts inside mu+/-2sigma / 0.9545)
            n2 = np.sum((rb > mu - 2 * sig) & (rb < mu + 2 * sig))
            amp = (n2 / 0.9545) * (2 * half / nb) / (np.sqrt(2 * np.pi) * sig)
            xs = np.linspace(-half, half, 600)
            ax.plot(xs, amp * np.exp(-0.5 * ((xs - mu) / sig) ** 2), "r-", lw=1.4)
            ax.axvspan(mu - 2 * sig, mu + 2 * sig, color="red", alpha=.06)
            ax.set_xlim(-half, half)
            out_frac = float(np.mean(np.abs(rb) > 0.20))
            ax.text(0.03, 0.95,
                    f"$\\sigma_G$={sig*100:.2f}%\n$\\mu$={mu*100:+.2f}%\n"
                    f"rob={rob*100:.2f}%\nout={out_frac*100:.1f}%",
                    transform=ax.transAxes, va="top", fontsize=7.5)
            clip_frac = float(np.mean((rb < -half) | (rb > half)))
            if clip_frac > 0.002:
                ax.text(0.97, 0.95, f"{clip_frac*100:.1f}%\noutside",
                        transform=ax.transAxes, va="top", ha="right", fontsize=7,
                        color="#b30000")
            if i == 0:
                ax.set_title(f"{lo}--{hi} GeV", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lab}\n({note})", fontsize=8.5)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$(E_{\rm pred}-E_{\rm true})/E_{\rm true}$", fontsize=8)
    fig.suptitle("Iterated Gaussian-core fits (red; shaded = final $\\pm2\\sigma$ window); "
                 "one shared x-range per energy column, so widths compare across methods.",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(f"{OUT}/f13_gauss_fits.pdf", bbox_inches="tight"); plt.close()
    print("wrote f13_gauss_fits.pdf ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--per-bin", type=int, default=8000)
    args = ap.parse_args()
    with open(RESCORE) as f:
        res = json.load(f)["results"]
    if os.path.exists(RESCORE_EXTRA):        # final v3_d320 (+extra v3 checkpoints)
        with open(RESCORE_EXTRA) as f:
            extra = json.load(f)["results"]
        res.update({k: v for k, v in extra.items() if k != "3dfit_EneL2Cor"})
    if args.summary:
        fig_summary(res)
    if args.fits:
        fig_fits(res, args.per_bin)
    if not (args.summary or args.fits):
        print("nothing to do: pass --summary and/or --fits")


if __name__ == "__main__":
    main()
