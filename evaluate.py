"""Evaluate a trained model on the test split.

    python evaluate.py --config config/base.yaml            # uses <out_dir>/best.pt

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
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer


@torch.no_grad()
def run_model(model, loader, device):
    ep, et, cp, ct, rerr = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
    centers, res, bias, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < 20:
            continue
        centers.append(0.5 * (lo + hi))
        res.append(float(np.std(r[m])))
        bias.append(float(np.mean(r[m])))
        counts.append(int(m.sum()))
    return map(np.array, (centers, res, bias, counts))


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
    e_pred, e_true, c_pred, c_true, rerr = run_model(model, loader, device)
    r = (e_pred - e_true) / np.clip(e_true, 1e-3, None)

    bins = np.array(cfg.eval.energy_bins, dtype=float)
    centers, res, bias, counts = binned(e_true, r, bins)

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
    le2 = centers <= e_cut
    metrics = {
        "n_test": int(len(e_true)),
        "overall_res": float(np.std(r)), "overall_bias": float(np.mean(r)),
        "bias_aware_metric": float(np.sqrt(np.std(r) ** 2 + np.mean(r) ** 2)),
        "binwise_mean_res": float(np.mean(res)) if len(res) else None,
        "e_cut_gev": e_cut,
        "overall_res_le_cut": float(np.std(r[m2])) if m2.any() else None,
        "binwise_mean_res_le_cut": float(np.mean(res[le2])) if le2.any() else None,
        "recon_logmse": float(np.mean(rerr)),
        "bins": {"center": centers.tolist(), "res": res.tolist(),
                 "bias": bias.tolist(), "count": counts.tolist()},
        "concepts": concept_metrics,
    }
    with open(os.path.join(cfg.paths.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in
          ["overall_res", "overall_bias", "binwise_mean_res",
           "overall_res_le_cut", "binwise_mean_res_le_cut", "recon_logmse"]}, indent=2))
    print("concept R^2:", {k: round(v["r2"], 3) for k, v in concept_metrics.items()})

    od = cfg.paths.out_dir
    plt.figure(); plt.plot(centers, np.array(res) * 100, "o-")
    plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("sigma/E [%]")
    plt.title("Energy resolution"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "resolution_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.axhline(0, color="k", lw=.8); plt.plot(centers, np.array(bias) * 100, "s-")
    plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("bias [%]")
    plt.title("Energy bias"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "bias_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.scatter(e_true, e_pred, s=3, alpha=.2)
    lim = [max(e_true.min(), 1e-1), e_true.max()]
    plt.plot(lim, lim, "r--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("E_true [GeV]"); plt.ylabel("E_pred [GeV]"); plt.title("DNN energy")
    plt.savefig(os.path.join(od, "scatter.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.hist(r, bins=120, range=(-0.3, 0.3))
    plt.xlabel("(E_pred - E_true)/E_true"); plt.title(
        f"res={np.std(r)*100:.2f}%  bias={np.mean(r)*100:+.2f}%")
    plt.savefig(os.path.join(od, "residual_hist.png"), dpi=130, bbox_inches="tight"); plt.close()
    print("wrote metrics.json + 4 plots ->", od)


if __name__ == "__main__":
    main()
