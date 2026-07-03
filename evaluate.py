"""Evaluate a trained model on the test split.

    python evaluate.py --config config/base.yaml            # uses <out_dir>/best.pt

v3: the PRIMARY resolution estimator is the GAUSSIAN CORE sigma (iterative binned
fit in mu +/- 2 sigma, utils/stats.gauss_core) with the fitted mean as the bias --
the same estimator train.py selects checkpoints on. The IQR/1.349 robust core is
kept as the cross-check; raw std / outlier fraction stay as the tail diagnostics.

Writes metrics.json + resolution_vs_E.png / bias_vs_E.png / scatter.png /
residual_hist.png to the run's out_dir.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from utils.config import load_config
from utils.stats import gauss_core, robust_sigma
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer


def amp_dtype_of(cfg):
    """Honor train.amp_dtype: bf16 -> autocast bf16; fp32/none -> no autocast.
    (Mirrors train.py so eval runs at the SAME precision the model trained in.)"""
    name = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "none"):
        return None
    raise ValueError(f"train.amp_dtype={name!r} unsupported (use bf16 | fp32 | none)")


@torch.no_grad()
def run_model(model, loader, device, amp_dtype):
    ep, et, cp, ct, rerr = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        ep.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
        et.append(batch["energy"].cpu().numpy())
        cp.append(out["concepts"].float().cpu().numpy())
        ct.append(batch["concepts"].cpu().numpy())
        v = batch["valid"]
        e = ((out["recon"].float() - batch["recon"]) ** 2 * v).sum(1) / v.sum(1).clamp_min(1)
        rerr.append(e.cpu().numpy())
    return (np.concatenate(ep), np.concatenate(et), np.concatenate(cp),
            np.concatenate(ct), np.concatenate(rerr))


def binned(e_true, r, bins):
    """Per-bin metrics. v3 PRIMARY = Gaussian core (iterative mu+/-2sigma fit) + fitted
    mean; robust core (IQR/1.349) is the cross-check; raw std + mean bias diagnostics."""
    out = {k: [] for k in ("center", "res_gauss", "bias_gauss", "gauss_ok",
                           "res", "bias", "res_raw", "bias_raw", "count")}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < 20:
            continue
        rb = r[m]
        g = gauss_core(rb)
        out["center"].append(0.5 * (lo + hi))
        out["res_gauss"].append(float(g["sigma"]))   # Gaussian core (v3 PRIMARY)
        out["bias_gauss"].append(float(g["mu"]))     # fitted Gaussian mean
        out["gauss_ok"].append(int(g["ok"]))
        out["res"].append(robust_sigma(rb))          # robust core (cross-check)
        out["bias"].append(float(np.median(rb)))     # median bias (robust)
        out["res_raw"].append(float(np.std(rb)))     # raw std (diagnostic)
        out["bias_raw"].append(float(np.mean(rb)))   # mean bias (diagnostic)
        out["count"].append(int(m.sum()))
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    device = cfg.device
    meta = load_meta(cfg.paths.cache_dir)
    ckpt_path = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    test_ds = EcalTokens(cfg.paths.cache_dir, "test", meta, concept_use=concept_use)
    loader = make_loader(test_ds, cfg, shuffle=False)
    e_pred, e_true, c_pred, c_true, rerr = run_model(model, loader, device, amp_dtype_of(cfg))
    r = (e_pred - e_true) / np.clip(e_true, 1e-3, None)

    bins = np.array(cfg.eval.energy_bins, dtype=float)
    b = binned(e_true, r, bins)

    # per-concept resolution (de-standardise to raw 3D-fit units; subset-aware)
    cstd = np.asarray(test_ds.cstd); cmean = np.asarray(test_ds.cmean)
    cp_raw = c_pred * cstd + cmean
    ct_raw = c_true * cstd + cmean
    concept_metrics = {}
    for j, name in enumerate(test_ds.concept_names):
        var = np.var(ct_raw[:, j]) + 1e-9
        r2 = 1.0 - np.mean((cp_raw[:, j] - ct_raw[:, j]) ** 2) / var
        concept_metrics[name] = {"rmse": float(np.sqrt(np.mean((cp_raw[:, j]-ct_raw[:, j])**2))),
                                 "r2": float(r2)}

    # Reliable-range summary: the ECAL can't measure electrons past ~2 TeV (rear
    # leakage), so report sigma/E restricted to <= e_cut alongside the full range.
    roll = cfg.loss.get("energy_rolloff", None)
    e_cut = float(roll.get("e_cut", 2000.0)) if roll is not None else 2000.0
    m2 = e_true <= e_cut
    le2 = b["center"] <= e_cut
    g_all = gauss_core(r)
    g_cut = gauss_core(r[m2]) if m2.any() else None
    metrics = {
        "n_test": int(len(e_true)),
        "sigma_estimator": "gauss_core_iter2sigma",
        # PRIMARY metrics (v3): Gaussian core sigma + fitted-mean bias -- the SAME
        # estimator train.py selects checkpoints on, so eval and selection agree.
        "overall_res_gauss": float(g_all["sigma"]), "overall_bias_gauss": float(g_all["mu"]),
        "gauss_ok": int(g_all["ok"]),
        "bias_aware_metric_gauss": float(np.hypot(g_all["sigma"], g_all["mu"])),
        "binwise_mean_res_gauss": float(np.mean(b["res_gauss"])) if len(b["res_gauss"]) else None,
        "overall_res_gauss_le_cut": float(g_cut["sigma"]) if g_cut else None,
        "binwise_mean_res_gauss_le_cut": (float(np.mean(b["res_gauss"][le2]))
                                          if le2.any() else None),
        # CROSS-CHECK: robust core (IQR/1.349) + median bias (the v2 primary).
        "overall_res": robust_sigma(r), "overall_bias": float(np.median(r)),
        "bias_aware_metric": float(np.sqrt(robust_sigma(r) ** 2 + np.median(r) ** 2)),
        "binwise_mean_res": float(np.mean(b["res"])) if len(b["res"]) else None,
        "overall_res_le_cut": robust_sigma(r[m2]) if m2.any() else None,
        "binwise_mean_res_le_cut": float(np.mean(b["res"][le2])) if le2.any() else None,
        # DIAGNOSTICS: raw std + mean bias (tail-sensitive legacy definition) and the
        # outlier fraction that quantifies the non-Gaussian tail separately.
        "overall_res_raw_std": float(np.std(r)), "overall_bias_mean": float(np.mean(r)),
        "binwise_mean_res_raw_std": float(np.mean(b["res_raw"])) if len(b["res_raw"]) else None,
        "overall_res_le_cut_raw_std": float(np.std(r[m2])) if m2.any() else None,
        "outlier_frac": float(np.mean(np.abs(r) > 0.20)),
        "e_cut_gev": e_cut,
        "recon_logmse": float(np.mean(rerr)),
        "bins": {k: v.tolist() for k, v in b.items()},
        "concepts": concept_metrics,
    }
    with open(os.path.join(cfg.paths.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in
          ["overall_res_gauss", "overall_bias_gauss", "overall_res_gauss_le_cut",
           "binwise_mean_res_gauss", "overall_res", "overall_res_le_cut",
           "overall_res_raw_std", "outlier_frac", "recon_logmse"]}, indent=2))
    print("(overall_res_gauss = Gaussian core, iterative +/-2sigma fit; "
          "overall_res = robust core IQR/1.349; overall_res_raw_std = legacy std)")
    print("concept R^2:", {k: round(v["r2"], 3) for k, v in concept_metrics.items()})

    od = cfg.paths.out_dir
    plt.figure()
    plt.plot(b["center"], b["res_gauss"] * 100, "o-", label="Gaussian core (+/-2$\\sigma$ fit)")
    plt.plot(b["center"], b["res"] * 100, "d-", alpha=.6, label="robust core (IQR/1.349)")
    plt.plot(b["center"], b["res_raw"] * 100, "s--", alpha=.45, label="raw std")
    plt.legend(); plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("sigma/E [%]")
    plt.title("Energy resolution"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "resolution_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.axhline(0, color="k", lw=.8)
    plt.plot(b["center"], b["bias_gauss"] * 100, "o-", label="Gaussian-fit mean")
    plt.plot(b["center"], b["bias"] * 100, "s-", alpha=.6, label="median bias")
    plt.plot(b["center"], b["bias_raw"] * 100, "^--", alpha=.45, label="mean bias")
    plt.legend(); plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("bias [%]")
    plt.title("Energy bias"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "bias_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.scatter(e_true, e_pred, s=3, alpha=.2)
    lim = [max(e_true.min(), 1e-1), e_true.max()]
    plt.plot(lim, lim, "r--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("E_true [GeV]"); plt.ylabel("E_pred [GeV]"); plt.title("DNN energy")
    plt.savefig(os.path.join(od, "scatter.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.hist(r, bins=120, range=(-0.3, 0.3), density=False)
    # overlay the fitted Gaussian core so the fit quality is visible
    xs = np.linspace(-0.3, 0.3, 400)
    n_in = np.sum((r > -0.3) & (r < 0.3))
    scale = n_in * (0.6 / 120)
    plt.plot(xs, scale / (np.sqrt(2*np.pi) * g_all["sigma"])
             * np.exp(-0.5 * ((xs - g_all["mu"]) / g_all["sigma"]) ** 2),
             "r-", lw=1.2, label="Gaussian core fit")
    plt.legend()
    plt.xlabel("(E_pred - E_true)/E_true"); plt.title(
        f"gauss core={g_all['sigma']*100:.2f}% (mu={g_all['mu']*100:+.2f}%)  "
        f"robust={robust_sigma(r)*100:.2f}%  raw-std={np.std(r)*100:.2f}%")
    plt.savefig(os.path.join(od, "residual_hist.png"), dpi=130, bbox_inches="tight"); plt.close()
    print("wrote metrics.json + 4 plots ->", od)


if __name__ == "__main__":
    main()
