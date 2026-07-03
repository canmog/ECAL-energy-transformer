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


def amp_dtype_of(cfg):
    """Honor train.amp_dtype: bf16 -> autocast bf16; fp32/none -> no autocast.
    (Mirrors train.py so eval runs at the SAME precision the model trained in.)"""
    name = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "none"):
        return None
    raise ValueError(f"train.amp_dtype={name!r} unsupported (use bf16 | fp32 | none)")


def robust_sigma(x):
    """Robust Gaussian-core width sigma = IQR/1.349 (tail-insensitive). This is the
    calorimetry-convention resolution estimator and the SAME one train.py selects on."""
    q75, q25 = np.percentile(x, [75, 25])
    return float((q75 - q25) / 1.349)


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
    """Per-bin metrics. PRIMARY = robust core sigma (IQR/1.349) + median bias (the
    estimator train.py selects on); raw std + mean bias are kept as tail diagnostics."""
    centers, res, bias, res_raw, bias_raw, counts = [], [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < 20:
            continue
        rb = r[m]
        centers.append(0.5 * (lo + hi))
        res.append(robust_sigma(rb))            # robust core (PRIMARY)
        bias.append(float(np.median(rb)))       # median bias (robust)
        res_raw.append(float(np.std(rb)))       # raw std (diagnostic)
        bias_raw.append(float(np.mean(rb)))     # mean bias (diagnostic)
        counts.append(int(m.sum()))
    return map(np.array, (centers, res, bias, res_raw, bias_raw, counts))


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
    centers, res, bias, res_raw, bias_raw, counts = binned(e_true, r, bins)

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
        "sigma_estimator": "robust_core_iqr/1.349",
        # PRIMARY metrics: robust core sigma (IQR/1.349) + median bias -- the SAME
        # estimator train.py selects checkpoints on, so eval and selection now agree.
        "overall_res": robust_sigma(r), "overall_bias": float(np.median(r)),
        "bias_aware_metric": float(np.sqrt(robust_sigma(r) ** 2 + np.median(r) ** 2)),
        "binwise_mean_res": float(np.mean(res)) if len(res) else None,
        "overall_res_le_cut": robust_sigma(r[m2]) if m2.any() else None,
        "binwise_mean_res_le_cut": float(np.mean(res[le2])) if le2.any() else None,
        # DIAGNOSTICS: raw std + mean bias (tail-sensitive legacy definition) and the
        # outlier fraction that quantifies the non-Gaussian tail separately.
        "overall_res_raw_std": float(np.std(r)), "overall_bias_mean": float(np.mean(r)),
        "binwise_mean_res_raw_std": float(np.mean(res_raw)) if len(res_raw) else None,
        "overall_res_le_cut_raw_std": float(np.std(r[m2])) if m2.any() else None,
        "outlier_frac": float(np.mean(np.abs(r) > 0.20)),
        "e_cut_gev": e_cut,
        "recon_logmse": float(np.mean(rerr)),
        "bins": {"center": centers.tolist(), "res": res.tolist(), "bias": bias.tolist(),
                 "res_raw": res_raw.tolist(), "bias_raw": bias_raw.tolist(),
                 "count": counts.tolist()},
        "concepts": concept_metrics,
    }
    with open(os.path.join(cfg.paths.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in
          ["overall_res", "overall_bias", "binwise_mean_res",
           "overall_res_le_cut", "binwise_mean_res_le_cut",
           "overall_res_raw_std", "recon_logmse"]}, indent=2))
    print("(overall_res = robust core IQR/1.349; overall_res_raw_std = legacy std)")
    print("concept R^2:", {k: round(v["r2"], 3) for k, v in concept_metrics.items()})

    od = cfg.paths.out_dir
    plt.figure()
    plt.plot(centers, np.array(res) * 100, "o-", label="robust core (IQR/1.349)")
    plt.plot(centers, np.array(res_raw) * 100, "s--", alpha=.45, label="raw std")
    plt.legend(); plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("sigma/E [%]")
    plt.title("Energy resolution"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "resolution_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.axhline(0, color="k", lw=.8)
    plt.plot(centers, np.array(bias) * 100, "s-", label="median bias")
    plt.plot(centers, np.array(bias_raw) * 100, "^--", alpha=.45, label="mean bias")
    plt.legend(); plt.xscale("log"); plt.xlabel("E_true [GeV]"); plt.ylabel("bias [%]")
    plt.title("Energy bias"); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(od, "bias_vs_E.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.scatter(e_true, e_pred, s=3, alpha=.2)
    lim = [max(e_true.min(), 1e-1), e_true.max()]
    plt.plot(lim, lim, "r--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("E_true [GeV]"); plt.ylabel("E_pred [GeV]"); plt.title("DNN energy")
    plt.savefig(os.path.join(od, "scatter.png"), dpi=130, bbox_inches="tight"); plt.close()

    plt.figure(); plt.hist(r, bins=120, range=(-0.3, 0.3))
    plt.xlabel("(E_pred - E_true)/E_true"); plt.title(
        f"core={robust_sigma(r)*100:.2f}% (IQR/1.349)  med-bias={np.median(r)*100:+.2f}%"
        f"  raw-std={np.std(r)*100:.2f}%")
    plt.savefig(os.path.join(od, "residual_hist.png"), dpi=130, bbox_inches="tight"); plt.close()
    print("wrote metrics.json + 4 plots ->", od)


if __name__ == "__main__":
    main()
