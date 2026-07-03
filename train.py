"""Train the physics-grounded ECAL Transformer.

    python train.py --config config/base.yaml
    python train.py --set train.compile=false model.d_model=256

Multi-task: energy (global) + recon (micro kx_expehit) + concept (soft bottleneck),
auto-balanced by Kendall-Gal uncertainty weighting. Model selection uses a
BIAS-AWARE energy metric sqrt(res^2 + bias^2) so a collapsed constant predictor
cannot win on spread.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from utils.config import load_config
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer
from losses.objectives import energy_loss, concept_loss, recon_loss
from losses.uncertainty import UncertaintyWeighter, FixedWeighter, NormalizedWeighter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def build_loaders(cfg, meta):
    cd = cfg.paths.cache_dir
    cu = cfg.data.get("concept_use", None)
    tr = EcalTokens(cd, "train", meta, train=True, augment=cfg.train.augment.to_dict(),
                    concept_use=cu)
    va = EcalTokens(cd, "val", meta, train=False, concept_use=cu)
    return (make_loader(tr, cfg, shuffle=True, seed=cfg.seed),
            make_loader(va, cfg, shuffle=False))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def energy_weight(energy_gev, cfg):
    """Per-sample energy-region weight (None = uniform).

    loss.energy_rolloff: flat (=1) through the region of interest, then a power-law
    decay above e_cut (the ECAL's ~2 TeV reliability limit) so the leakage-dominated
    high-E tail stops driving the loss and the selection metric WITHOUT a hard cut.
    w(E) = min(1, (e_cut / E)^index). index ~ 2.7 mirrors the cosmic-ray spectrum."""
    roll = cfg.loss.get("energy_rolloff", None)
    if roll is None or not roll.get("enabled", False):
        return None
    e_cut = float(roll.get("e_cut", 2000.0))
    index = float(roll.get("index", 2.7))
    return torch.clamp((e_cut / energy_gev.clamp_min(1e-3)) ** index, max=1.0)


def compute_losses(out, batch, cfg):
    w = energy_weight(batch["energy"], cfg)            # per-sample weight on energy heads
    losses = {}
    if "energy" in out:
        losses["energy"] = energy_loss(out["energy"], batch["log_e"], cfg.loss.energy.delta, w=w)
    if "e_phys" in out:                                # dual head: ground the physics energy
        losses["e_phys"] = energy_loss(out["e_phys"], batch["log_e"], cfg.loss.energy.delta, w=w)
    if "recon" in out:
        losses["recon"] = recon_loss(out["recon"], batch["recon"], batch["valid"],
                                     cfg.loss.recon.delta)
    losses["concept"] = concept_loss(out["concepts"], batch["concepts"], cfg.loss.concept.delta)
    return losses


def amp_dtype_of(cfg):
    """Honor train.amp_dtype: bf16 -> autocast bf16; fp32/none -> no autocast."""
    name = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "none"):
        return None
    raise ValueError(f"train.amp_dtype={name!r} unsupported (use bf16 | fp32 | none; "
                     "fp16 would need a GradScaler and is deliberately not offered)")


@torch.no_grad()
def validate(model, raw, loader, cfg, device, amp_dtype):
    model.eval()
    rs, ets, agg = [], [], {}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        e_pred = raw.predict_energy_gev(out["energy"].float())
        r = ((e_pred - batch["energy"]) / batch["energy"].clamp_min(1e-3)).cpu().numpy()
        rs.append(r)
        ets.append(batch["energy"].cpu().numpy())
        for k, v in compute_losses(out, batch, cfg).items():
            agg.setdefault(k, []).append(float(v))
    r = np.concatenate(rs); et = np.concatenate(ets)
    # Model selection on the ROBUST core sigma (IQR/1.349) + MEDIAN bias, restricted to the
    # reliable range (E <= e_cut). Robust = tail-insensitive => STABLE selection (raw std
    # bounces with a handful of outliers and conflates core width with tail weight). The
    # raw std and the outlier fraction are logged SEPARATELY as the tail diagnostics; they
    # do NOT drive selection -- tails are a data-quality issue, not the energy metric's job.
    roll = cfg.loss.get("energy_rolloff", None)
    e_cut = float(roll.get("e_cut", 2000.0)) if roll is not None else 2000.0
    sel = (et <= e_cut) & np.isfinite(r)
    rr = r[sel] if int(sel.sum()) >= 50 else r[np.isfinite(r)]
    q75, q25 = np.percentile(rr, [75, 25])
    res = float((q75 - q25) / 1.349)                # robust core sigma  (the resolution)
    bias = float(np.median(rr))                     # median bias        (robust)
    metric = float(np.sqrt(res ** 2 + bias ** 2))   # bias-aware robust  -> drives selection
    outlier = float(np.mean(np.abs(rr) > 0.20))     # tail: fraction |dE/E| > 20%
    res_raw = float(np.std(rr))                     # raw std (tail-sensitive), logged only
    means = {k: float(np.mean(v)) for k, v in agg.items()}
    return {"val_bias": bias, "val_res": res, "val_res_raw": res_raw,
            "val_outlier": outlier, "val_metric": metric,
            **{f"val_{k}": v for k, v in means.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    cfg, raw_cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    meta = load_meta(cfg.paths.cache_dir)
    train_loader, val_loader = build_loaders(cfg, meta)

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    raw = EcalTransformer(cfg, n_concepts).to(device)
    raw.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])

    task_names = []
    if cfg.heads.energy.enabled:
        task_names.append("energy")
        if cfg.model.get("dual_energy_head", False):
            task_names.append("e_phys")            # ground the physics energy baseline
    if cfg.heads.recon.enabled:
        task_names.append("recon")
    task_names.append("concept")

    # weighting: normalized (EMA-norm + fixed manual weights) | uncertainty (Kendall-Gal)
    # | fixed (plain sum). Back-compat default keyed off the old uncertainty_weighting bool.
    mode = cfg.loss.get("weighting",
                        "uncertainty" if cfg.loss.get("uncertainty_weighting", True) else "fixed")
    if mode == "normalized":
        wd = cfg.loss.get("weights", None)
        wd = wd.to_dict() if hasattr(wd, "to_dict") else (wd or {})
        weighter = NormalizedWeighter(task_names, weights=wd).to(device)
    elif mode == "uncertainty":
        weighter = UncertaintyWeighter(task_names).to(device)
    else:
        weighter = FixedWeighter(task_names).to(device)

    model = torch.compile(raw, dynamic=True) if cfg.train.compile else raw
    amp_dtype = amp_dtype_of(cfg)

    if str(cfg.train.optimizer).lower() != "adamw":
        raise ValueError(f"train.optimizer={cfg.train.optimizer!r}: only adamw is implemented")
    # Mild weight decay on the Kendall-Gal log-variances (was 0.0 to avoid biasing
    # weights toward 1). With the loss normalisation in UncertaintyWeighter the weights
    # no longer blow up, and this gentle pull toward s->0 self-limits any residual
    # drift instead of diverging on large datasets. Set loss.logvar_weight_decay=0 to
    # restore the old un-decayed behaviour.
    groups = [{"params": list(raw.parameters())}]
    wparams = list(weighter.parameters())
    if wparams:
        groups.append({"params": wparams,
                       "weight_decay": cfg.loss.get("logvar_weight_decay", 0.01)})
    all_params = list(raw.parameters()) + wparams
    opt = torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, cfg.train.warmup_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, cfg.train.epochs - cfg.train.warmup_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [cfg.train.warmup_epochs])

    csv_path = os.path.join(cfg.paths.out_dir, "metrics.csv")
    with open(os.path.join(cfg.paths.out_dir, "config.json"), "w") as f:
        json.dump(raw_cfg, f, indent=2)

    best, bad, header = float("inf"), 0, None
    for epoch in range(cfg.train.epochs):
        model.train()
        tot, task_sums, n_batches = 0.0, {t: 0.0 for t in task_names}, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch)
                losses = compute_losses(out, batch, cfg)
                total, _ = weighter(losses)
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
            opt.step()
            tot += float(total.detach())
            for t, l in losses.items():
                task_sums[t] += float(l.detach())
            n_batches += 1
        sched.step()

        val = validate(model, raw, val_loader, cfg, device, amp_dtype)
        # NOTE: train_total is the optimization objective and includes the +0.5*s
        # regularizer (can be negative); the train_<task> columns are the raw,
        # comparable per-task losses averaged over the epoch.
        wlogs = {f"w_{t}": w for t, w in weighter.task_weights().items()}
        wlogs.update({f"train_{t}": task_sums[t] / max(n_batches, 1) for t in task_names})
        row = {"epoch": epoch, "train_total": tot / max(n_batches, 1),
               "lr": opt.param_groups[0]["lr"], **val, **wlogs}
        if header is None:
            header = list(row.keys())
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, header).writeheader()
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, header).writerow(row)
        print(f"[{epoch:3d}] train={row['train_total']:.4f} "
              f"core_res={val['val_res']*100:.2f}% bias={val['val_bias']*100:+.2f}% "
              f"tail={val['val_outlier']*100:.1f}% metric={val['val_metric']*100:.2f}% "
              f"(raw={val['val_res_raw']*100:.2f}%) concept={val['val_concept']:.3f}")

        ckpt = {"model": raw.state_dict(), "weighter": weighter.state_dict(),
                "meta": meta, "config": raw_cfg, "epoch": epoch, "val": val}
        torch.save(ckpt, os.path.join(cfg.paths.out_dir, "last.pt"))
        if val["val_metric"] < best:
            best, bad = val["val_metric"], 0
            torch.save(ckpt, os.path.join(cfg.paths.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.train.early_stop_patience:
                print(f"early stop at epoch {epoch} (best metric {best*100:.2f}%)")
                break
    print(f"done. best bias-aware metric = {best*100:.2f}%  ->  {cfg.paths.out_dir}/best.pt")


if __name__ == "__main__":
    main()
