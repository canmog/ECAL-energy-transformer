"""Train the physics-grounded ECAL incidence-direction transformer.

The primary task is MC-truth (dx/dz,dy/dz). Energy, expected-cell
reconstruction, and the 11 concept targets retain the ecalTransformer settings
as auxiliary tasks. Checkpoints are selected on validation 68% angular
containment, not on an energy-resolution estimator.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from losses.objectives import (angle_loss, concept_loss, direction_chord_loss,
                               energy_loss, recon_loss)
from losses.uncertainty import FixedWeighter, NormalizedWeighter, UncertaintyWeighter
from models.model import EcalTransformer
from utils.angle import angular_error_np, containment_summary
from utils.config import load_config
from utils.stats import gauss_core

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def build_loaders(cfg, meta):
    cache = cfg.paths.cache_dir
    concept_use = cfg.data.get("concept_use", None)
    train = EcalTokens(
        cache, "train", meta, train=True,
        augment=cfg.train.augment.to_dict(), concept_use=concept_use,
        e_max=cfg.data.get("train_e_max", None),
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        threshold_jitter_mev=cfg.train.get("threshold_jitter_mev", None),
        max_events=cfg.data.get("train_max_events", None), subset_seed=cfg.seed)
    val = EcalTokens(
        cache, "val", meta, train=False, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        max_events=cfg.data.get("val_max_events", None), subset_seed=cfg.seed + 1)
    return (make_loader(train, cfg, shuffle=True, seed=cfg.seed),
            make_loader(val, cfg, shuffle=False))


def to_device(batch, device):
    # Identity fields are kept on CPU; the model and losses do not consume them.
    return {k: (v if k in ("run", "event") else v.to(device, non_blocking=True))
            for k, v in batch.items()}


def energy_weight(energy_gev, cfg):
    roll = cfg.loss.get("energy_rolloff", None)
    if roll is None or not roll.get("enabled", False):
        return None
    e_cut = float(roll.get("e_cut", 2000.0))
    index = float(roll.get("index", 2.7))
    return torch.clamp((e_cut / energy_gev.clamp_min(1e-3)) ** index, max=1.0)


def angle_energy_weight(energy_gev, cfg):
    """Optional train-only emphasis for the low-energy direction deficit.

    The weight uses MC energy only while computing the supervised angle loss; it
    is never an inference input.  Batch-mean normalization keeps the overall angle
    loss scale stable so this changes event priority rather than the task weight.
    """
    spec = cfg.loss.get("angle_energy_weight", None)
    if spec is None or not spec.get("enabled", False):
        return None
    pivot = float(spec.get("pivot_gev", 300.0))
    power = float(spec.get("power", 0.5))
    w_min = float(spec.get("w_min", 0.5))
    w_max = float(spec.get("w_max", 3.0))
    if pivot <= 0 or power < 0 or w_min <= 0 or w_max < w_min:
        raise ValueError("invalid loss.angle_energy_weight configuration")
    weight = (pivot / energy_gev.clamp_min(1e-3)).pow(power).clamp(w_min, w_max)
    if spec.get("normalize_batch", True):
        weight = weight / weight.mean().clamp_min(1e-6)
    return weight


def aux_weight(batch, cfg):
    """Fit-quality weight for noisy 3D-fit concept/recon supervision only."""
    fqw = cfg.loss.get("fit_quality_weight", None)
    if fqw is None or not fqw.get("enabled", False):
        return None
    scale = float(fqw.get("scale", 0.0))
    if scale <= 0:
        return None
    beta = float(fqw.get("beta", 1.0))
    w_min = float(fqw.get("w_min", 0.2))
    return ((scale / batch["fit_resid"].clamp_min(1e-9)) ** beta).clamp(w_min, 1.0)


def compute_losses(out, batch, cfg, apply_aux_weight=True):
    ew = energy_weight(batch["energy"], cfg)
    aw = aux_weight(batch, cfg) if apply_aux_weight else None
    fqw = cfg.loss.get("fit_quality_weight", None)
    apply_to = (fqw.get("apply_to", ["concept", "recon"])
                if fqw is not None and aw is not None else [])
    losses = {}
    if "energy" in out:
        losses["energy"] = energy_loss(
            out["energy"], batch["log_e"], cfg.loss.energy.delta, w=ew)
    if "e_phys" in out:
        losses["e_phys"] = energy_loss(
            out["e_phys"], batch["log_e"], cfg.loss.energy.delta, w=ew)
    if "angle" in out:
        # MC truth: deliberately never weighted by 3D-fit quality.
        angle_mode = str(cfg.loss.angle.get("mode", "huber")).lower()
        angle_w = angle_energy_weight(batch["energy"], cfg)
        if "angle_residual_std" in out:
            physical_residual = batch["angle"] - out["angle_baseline"]
            target_residual_std = (
                (physical_residual - out["angle_residual_mean"])
                / out["angle_residual_scale"])
            huber = angle_loss(
                out["angle_residual_std"], target_residual_std,
                cfg.loss.angle.delta, w=angle_w)
        else:
            huber = angle_loss(
                out["angle"], batch["angle_std"], cfg.loss.angle.delta,
                w=angle_w)
        if angle_mode == "huber":
            losses["angle"] = huber
        elif angle_mode == "chord":
            losses["angle"] = direction_chord_loss(
                out["angle_slopes"], batch["angle"],
                cfg.loss.angle.get("epsilon", 1e-3), w=angle_w)
        elif angle_mode == "hybrid":
            chord = direction_chord_loss(
                out["angle_slopes"], batch["angle"],
                cfg.loss.angle.get("epsilon", 1e-3), w=angle_w)
            losses["angle"] = huber + float(
                cfg.loss.angle.get("chord_weight", 1.0)) * chord
        else:
            raise ValueError(
                f"unknown loss.angle.mode={angle_mode!r}; use huber, chord, or hybrid")
    if "recon" in out:
        losses["recon"] = recon_loss(
            out["recon"], batch["recon"], batch["valid"],
            cfg.loss.recon.delta, w=aw if "recon" in apply_to else None)
    losses["concept"] = concept_loss(
        out["concepts"], batch["concepts"], cfg.loss.concept.delta,
        w=aw if "concept" in apply_to else None)
    return losses


def amp_dtype_of(cfg):
    name = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "none"):
        return None
    raise ValueError(f"train.amp_dtype={name!r} unsupported; use bf16, fp32, or none")


@torch.no_grad()
def validate(model, raw, loader, cfg, device, amp_dtype):
    model.eval()
    pred_all, true_all, fit_all, baseline_all = [], [], [], []
    energy_residual_all, energy_true_all = [], []
    joint_energy_angle = bool(cfg.train.get("joint_energy_angle", False))
    loss_sums, n_events = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        pred_all.append(raw.predict_angle_slopes(out["angle"].float()).cpu().numpy())
        true_all.append(batch["angle"].cpu().numpy())
        fit_all.append(batch["fit_angle"].cpu().numpy())
        if "angle_baseline" in out:
            baseline_all.append(out["angle_baseline"].float().cpu().numpy())
        if joint_energy_angle:
            energy_pred = raw.predict_energy_gev(out["energy"].float())
            energy_residual_all.append((
                (energy_pred - batch["energy"])
                / batch["energy"].clamp_min(1e-3)).cpu().numpy())
            energy_true_all.append(batch["energy"].cpu().numpy())
        nb = int(batch["angle"].shape[0])
        n_events += nb
        for name, value in compute_losses(
                out, batch, cfg, apply_aux_weight=False).items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * nb

    pred = np.concatenate(pred_all)
    truth = np.concatenate(true_all)
    fit = np.concatenate(fit_all)
    err = angular_error_np(pred, truth)
    fit_err = angular_error_np(fit, truth)
    summary = containment_summary(err)
    fit_summary = containment_summary(fit_err)
    residual = pred - truth
    finite = np.isfinite(residual).all(axis=1) & np.isfinite(err)
    if not finite.all():
        raise RuntimeError(f"non-finite validation direction predictions: {(~finite).sum()}")
    means = {f"val_{name}": value / max(n_events, 1)
             for name, value in loss_sums.items()}
    result = {
        "val_metric": summary["p68"],
        "val_angle_median": summary["median"],
        "val_angle_p68": summary["p68"],
        "val_angle_p90": summary["p90"],
        "val_angle_p95": summary["p95"],
        "val_angle_mean": summary["mean"],
        "val_angle_rms": summary["rms"],
        "val_angle_outlier20mrad": float(np.mean(err > 0.020)),
        "val_kx_bias": float(np.mean(residual[:, 0])),
        "val_ky_bias": float(np.mean(residual[:, 1])),
        "val_kx_rmse": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "val_ky_rmse": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "val_fit_angle_p68": fit_summary["p68"],
        **means,
    }
    if baseline_all:
        baseline_summary = containment_summary(angular_error_np(
            np.concatenate(baseline_all), truth))
        result.update({
            "val_centroid_angle_median": baseline_summary["median"],
            "val_centroid_angle_p68": baseline_summary["p68"],
            "val_centroid_angle_p90": baseline_summary["p90"],
        })
    if joint_energy_angle:
        energy_residual = np.concatenate(energy_residual_all)
        energy_true = np.concatenate(energy_true_all)
        e_cut = float(cfg.train.get("energy_select_e_cut_gev", 2000.0))
        select = ((energy_true <= e_cut) & np.isfinite(energy_residual))
        selected = (energy_residual[select] if int(select.sum()) >= 50
                    else energy_residual[np.isfinite(energy_residual)])
        q25, q75 = np.percentile(selected, [25, 75])
        robust = float((q75 - q25) / 1.349)
        median_bias = float(np.median(selected))
        raw_std = float(np.std(selected))
        mean_bias = float(np.mean(selected))
        outlier = float(np.mean(np.abs(selected) > 0.20))
        gaussian = gauss_core(selected)
        gauss_res = float(gaussian["sigma"])
        gauss_bias = float(gaussian["mu"])
        energy_metric = float(np.hypot(gauss_res, gauss_bias))
        result.update({
            "val_energy_metric": energy_metric,
            "val_energy_res_gauss": gauss_res,
            "val_energy_bias_gauss": gauss_bias,
            "val_energy_gauss_ok": int(gaussian["ok"]),
            "val_energy_res_robust": robust,
            "val_energy_bias_median": median_bias,
            "val_energy_res_raw": raw_std,
            "val_energy_bias_mean": mean_bias,
            "val_energy_outlier20pct": outlier,
            "val_energy_select_count": int(len(selected)),
        })
    return result


def build_weighter(cfg, task_names, device):
    mode = cfg.loss.get(
        "weighting", "uncertainty" if cfg.loss.get("uncertainty_weighting", True)
        else "fixed")
    if mode == "normalized":
        weights = cfg.loss.get("weights", None)
        weights = weights.to_dict() if hasattr(weights, "to_dict") else (weights or {})
        return NormalizedWeighter(task_names, weights=weights).to(device)
    if mode == "uncertainty":
        return UncertaintyWeighter(task_names).to(device)
    if mode == "fixed":
        return FixedWeighter(task_names).to(device)
    raise ValueError(f"unknown loss.weighting={mode!r}")


@torch.no_grad()
def estimate_residual_norm(raw, loader, cfg, device):
    """Estimate robust train-only normalization for MC-minus-centroid residuals.

    The loader applies the same runtime threshold censoring, optional 20--30 MeV
    threshold jitter, and physical reflections as training.  Median/MAD keeps the
    scale stable in the presence of the centroid baseline's long shower tails.
    """
    if not raw.residual_normalization:
        return
    maximum = int(cfg.heads.angle.get("residual_norm_max_events", 200000))
    residuals, seen = [], 0
    for batch in loader:
        batch = to_device(batch, device)
        baseline = raw.centroid_slope(batch)
        residual = (batch["angle"] - baseline).float().cpu().numpy()
        take = min(len(residual), maximum - seen)
        residuals.append(residual[:take])
        seen += take
        if seen >= maximum:
            break
    if not residuals:
        raise RuntimeError("cannot estimate residual normalization from an empty loader")
    residual = np.concatenate(residuals, axis=0)
    center = np.median(residual, axis=0)
    scale = 1.4826 * np.median(np.abs(residual - center), axis=0)
    scale = np.maximum(scale, 1e-4)
    raw.set_angle_residual_norm(center, scale)
    print("[angle_residual_norm] "
          f"events={len(residual)} center={center.tolist()} scale={scale.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    cfg, raw_cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    meta = load_meta(cfg.paths.cache_dir)
    train_loader, val_loader = build_loaders(cfg, meta)
    fqw = cfg.loss.get("fit_quality_weight", None)
    if fqw is not None and fqw.get("enabled", False):
        scale = float(np.median(train_loader.dataset.fit_resid))
        setattr(fqw, "scale", scale)
        print(f"[fit_quality_weight] train median={scale:.5g} "
              f"apply_to={fqw.get('apply_to')}")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    raw = EcalTransformer(cfg, n_concepts).to(device)
    raw.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    raw.set_angle_norm(meta["angle_mean"], meta["angle_std"])
    estimate_residual_norm(raw, train_loader, cfg, device)

    task_names = []
    if cfg.heads.energy.enabled:
        task_names.append("energy")
        if cfg.model.get("dual_energy_head", False):
            task_names.append("e_phys")
    if cfg.heads.angle.enabled:
        task_names.append("angle")
    if cfg.heads.recon.enabled:
        task_names.append("recon")
    task_names.append("concept")
    weighter = build_weighter(cfg, task_names, device)

    compiled = False
    if cfg.train.compile:
        try:
            model = torch.compile(raw, dynamic=True)
            compiled = True
        except Exception as exc:
            print(f"[warn] torch.compile setup failed ({type(exc).__name__}: {exc}); eager")
            model = raw
    else:
        model = raw
    amp_dtype = amp_dtype_of(cfg)

    if str(cfg.train.optimizer).lower() != "adamw":
        raise ValueError("only AdamW is implemented")
    if cfg.train.get("wd_exclude_1d", False):
        decay = [p for p in raw.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay = [p for p in raw.parameters() if p.requires_grad and p.ndim < 2]
        groups = [{"params": decay, "weight_decay": cfg.train.weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
    else:
        groups = [{"params": list(raw.parameters())}]
    weighter_params = list(weighter.parameters())
    if weighter_params:
        groups.append({"params": weighter_params,
                       "weight_decay": cfg.loss.get("logvar_weight_decay", 0.01)})
    all_params = list(raw.parameters()) + weighter_params
    optimizer = torch.optim.AdamW(
        groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, 0.01, 1.0, cfg.train.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, cfg.train.epochs - cfg.train.warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm, cosine], [cfg.train.warmup_epochs])

    with open(os.path.join(cfg.paths.out_dir, "config.json"), "w") as handle:
        json.dump(raw_cfg, handle, indent=2)
    csv_path = os.path.join(cfg.paths.out_dir, "metrics.csv")
    best = {"p68": float("inf"), "median": float("inf"), "mean": float("inf")}
    best_file = {"p68": "best.pt", "median": "best_median.pt", "mean": "best_mean.pt"}
    best_key = {"p68": "val_angle_p68", "median": "val_angle_median",
                "mean": "val_angle_mean"}
    joint_energy_angle = bool(cfg.train.get("joint_energy_angle", False))
    if joint_energy_angle:
        best["energy"] = float("inf")
        best_file["energy"] = "best_energy.pt"
        best_key["energy"] = "val_energy_metric"
    bad_epochs = 0
    header = None

    for epoch in range(cfg.train.epochs):
        model.train()
        total_sum = 0.0
        task_sums = {name: 0.0 for name in task_names}
        train_events = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(batch)
                    losses = compute_losses(out, batch, cfg)
                    total, _ = weighter(losses)
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
                optimizer.step()
            except Exception as exc:
                if not compiled:
                    raise
                print(f"[warn] compiled step failed ({type(exc).__name__}: {exc}); "
                      "switching to eager and skipping this batch")
                model, compiled = raw, False
                continue
            nb = int(batch["angle"].shape[0])
            train_events += nb
            total_sum += float(total.detach()) * nb
            for name, loss in losses.items():
                task_sums[name] += float(loss.detach()) * nb
        scheduler.step()

        val = validate(model, raw, val_loader, cfg, device, amp_dtype)
        row = {
            "epoch": epoch,
            "train_total": total_sum / max(train_events, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **val,
            **{f"w_{name}": value for name, value in weighter.task_weights().items()},
            **{f"train_{name}": task_sums[name] / max(train_events, 1)
               for name in task_names},
        }
        if header is None:
            header = list(row)
            with open(csv_path, "w", newline="") as handle:
                csv.DictWriter(handle, header).writeheader()
        with open(csv_path, "a", newline="") as handle:
            csv.DictWriter(handle, header).writerow(row)

        centroid_text = (f" centroid-p68={1e3*val['val_centroid_angle_p68']:.3f}"
                         if "val_centroid_angle_p68" in val else "")
        energy_text = (
            f" energy-gauss={100*val['val_energy_res_gauss']:.3f}%"
            f"/{100*val['val_energy_bias_gauss']:+.3f}%"
            f" energy-tail={100*val['val_energy_outlier20pct']:.2f}%"
            if joint_energy_angle else "")
        print(f"[{epoch:3d}] train={row['train_total']:.4f} "
              f"angle p68={1e3*val['val_angle_p68']:.3f} mrad "
              f"median={1e3*val['val_angle_median']:.3f} "
              f"p90={1e3*val['val_angle_p90']:.3f} "
              f"fit-p68={1e3*val['val_fit_angle_p68']:.3f} "
              f"out20={100*val['val_angle_outlier20mrad']:.2f}%"
              f"{centroid_text}{energy_text}")

        checkpoint = {"model": raw.state_dict(), "weighter": weighter.state_dict(),
                      "meta": meta, "config": raw_cfg, "epoch": epoch, "val": val}
        torch.save(checkpoint, os.path.join(cfg.paths.out_dir, "last.pt"))
        improved_angle = val[best_key["p68"]] < best["p68"]
        improved_energy = (joint_energy_angle
                           and val[best_key["energy"]] < best["energy"])
        for name in best:
            value = val[best_key[name]]
            if value < best[name]:
                best[name] = value
                torch.save(checkpoint, os.path.join(cfg.paths.out_dir, best_file[name]))
                if name == "p68":
                    torch.save(checkpoint, os.path.join(
                        cfg.paths.out_dir, "best_angle.pt"))
        improved = improved_angle or improved_energy
        if improved:
            bad_epochs = 0
        else:
            bad_epochs += 1
            min_epochs = int(cfg.train.get("early_stop_min_epochs", 0))
            if (bad_epochs >= cfg.train.early_stop_patience
                    and epoch + 1 >= min_epochs):
                reason = ("neither angle nor energy improved"
                          if joint_energy_angle else "angle did not improve")
                print(f"early stop at epoch {epoch}; {reason}; "
                      f"best p68={1e3*best['p68']:.3f} mrad")
                break

    energy_done = (f" energy metric={100*best['energy']:.3f}%"
                   if joint_energy_angle else "")
    print(f"done: best p68={1e3*best['p68']:.3f} mrad "
          f"median={1e3*best['median']:.3f} mean={1e3*best['mean']:.3f}"
          f"{energy_done} -> {cfg.paths.out_dir}")


if __name__ == "__main__":
    main()
