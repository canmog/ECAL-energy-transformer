"""Train energy, incidence direction, or both with explicit data contracts."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from data.schema import require_meta
from losses.objectives import (angle_loss, concept_loss, direction_chord_loss,
                               energy_loss, recon_loss)
from losses.uncertainty import FixedWeighter, NormalizedWeighter, UncertaintyWeighter
from models.model import EcalTransformer
from utils.angle import angular_error_np, containment_summary
from utils.config import load_config
from utils.stats import gauss_core
from utils.tasks import (cache_fields_for_tasks, validate_task_heads,
                         validate_training_contract)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def build_loaders(cfg, meta):
    cache = cfg.paths.cache_dir
    concept_use = cfg.data.get("concept_use", None)
    mode, tasks, _ = validate_training_contract(cfg)
    fqw = cfg.loss.get("fit_quality_weight", None)
    fit_quality = bool(fqw is not None and fqw.get("enabled", False)
                       and set(_fit_quality_apply_to(cfg)) & set(tasks))
    angle_weight = _angle_energy_weight_enabled(cfg) and "angle" in tasks
    required = cache_fields_for_tasks(
        tasks, fit_quality=fit_quality, angle_energy_weight=angle_weight)
    train = EcalTokens(
        cache, "train", meta, train=True,
        augment=cfg.train.augment.to_dict(), concept_use=concept_use,
        e_max=cfg.data.get("train_e_max", None),
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        threshold_jitter_mev=cfg.train.get("threshold_jitter_mev", None),
        max_events=cfg.data.get("train_max_events", None), subset_seed=cfg.seed,
        task_mode=mode, required_fields=required,
        compute_fit_resid=fit_quality, operation=f"{mode} training")
    val = EcalTokens(
        cache, "val", meta, train=False, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        max_events=cfg.data.get("val_max_events", None), subset_seed=cfg.seed + 1,
        task_mode=mode, required_fields=required,
        compute_fit_resid=fit_quality, operation=f"{mode} validation")
    return (make_loader(train, cfg, shuffle=True, seed=cfg.seed),
            make_loader(val, cfg, shuffle=False))


def to_device(batch, device):
    # Identity fields are kept on CPU; the model and losses do not consume them.
    return {k: (v if k in ("run", "event") else v.to(device, non_blocking=True))
            for k, v in batch.items()}


def _fit_quality_apply_to(cfg):
    spec = cfg.loss.get("fit_quality_weight", None)
    if spec is None:
        return ()
    apply_to = spec.get("apply_to", ["concept", "recon"])
    if isinstance(apply_to, str):
        raise ValueError(
            "loss.fit_quality_weight.apply_to must be a YAML list, for example "
            "[concept, recon], not a string")
    unknown = sorted(set(apply_to) - {"concept", "recon"})
    if unknown:
        raise ValueError(
            f"loss.fit_quality_weight.apply_to contains unsupported tasks {unknown}")
    return tuple(apply_to)


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


def _angle_energy_weight_enabled(cfg):
    spec = cfg.loss.get("angle_energy_weight", None)
    return bool(spec is not None and spec.get("enabled", False))


def aux_weight(batch, cfg):
    """Fit-quality weight for noisy 3D-fit concept/recon supervision only."""
    fqw = cfg.loss.get("fit_quality_weight", None)
    if fqw is None or not fqw.get("enabled", False):
        return None
    scale = float(fqw.get("scale", 0.0))
    if scale <= 0:
        return None
    if "fit_resid" not in batch:
        raise KeyError(
            "fit-quality weighting requires batch field 'fit_resid'; rebuild or "
            "select a cache containing tok_expe")
    beta = float(fqw.get("beta", 1.0))
    w_min = float(fqw.get("w_min", 0.2))
    return ((scale / batch["fit_resid"].clamp_min(1e-9)) ** beta).clamp(w_min, 1.0)


def compute_losses(out, batch, cfg, apply_aux_weight=True, tasks=None):
    tasks = tuple(tasks or validate_task_heads(cfg))
    ew = energy_weight(batch["energy"], cfg) if "energy" in tasks else None
    configured_apply_to = _fit_quality_apply_to(cfg)
    needs_aux_weight = bool(set(configured_apply_to) & set(tasks))
    aw = (aux_weight(batch, cfg)
          if apply_aux_weight and needs_aux_weight else None)
    apply_to = configured_apply_to if aw is not None else ()
    losses = {}
    if "energy" in tasks:
        losses["energy"] = energy_loss(
            out["energy"], batch["log_e"], cfg.loss.energy.delta, w=ew)
    if "energy" in tasks and "e_phys" in out:
        losses["e_phys"] = energy_loss(
            out["e_phys"], batch["log_e"], cfg.loss.energy.delta, w=ew)
    if "angle" in tasks:
        # MC truth: deliberately never weighted by 3D-fit quality.
        angle_mode = str(cfg.loss.angle.get("mode", "huber")).lower()
        angle_w = (angle_energy_weight(batch["energy"], cfg)
                   if _angle_energy_weight_enabled(cfg) else None)
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
    if "recon" in tasks:
        losses["recon"] = recon_loss(
            out["recon"], batch["recon"], batch["valid"],
            cfg.loss.recon.delta, w=aw if "recon" in apply_to else None)
    if "concept" in tasks:
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
    mode, tasks, _ = validate_training_contract(cfg)
    pred_all, true_all, fit_all, baseline_all = [], [], [], []
    energy_residual_all, energy_true_all = [], []
    loss_sums, n_events = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        if "angle" in tasks:
            pred_all.append(raw.predict_angle_slopes(out["angle"].float()).cpu().numpy())
            true_all.append(batch["angle"].cpu().numpy())
            if "fit_angle" in batch:
                fit_all.append(batch["fit_angle"].cpu().numpy())
            if "angle_baseline" in out:
                baseline_all.append(out["angle_baseline"].float().cpu().numpy())
        if "energy" in tasks:
            energy_pred = raw.predict_energy_gev(out["energy"].float())
            energy_residual_all.append((
                (energy_pred - batch["energy"])
                / batch["energy"].clamp_min(1e-3)).cpu().numpy())
            energy_true_all.append(batch["energy"].cpu().numpy())
        nb = int(batch["feats"].shape[0])
        n_events += nb
        for name, value in compute_losses(
                out, batch, cfg, apply_aux_weight=False, tasks=tasks).items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * nb

    means = {f"val_{name}": value / max(n_events, 1)
             for name, value in loss_sums.items()}
    result = dict(means)

    if "angle" in tasks:
        pred = np.concatenate(pred_all)
        truth = np.concatenate(true_all)
        err = angular_error_np(pred, truth)
        summary = containment_summary(err)
        residual = pred - truth
        finite = np.isfinite(residual).all(axis=1) & np.isfinite(err)
        if not finite.all():
            raise RuntimeError(
                f"non-finite validation direction predictions: {(~finite).sum()}")
        result.update({
            "val_angle_metric": summary["p68"],
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
        })
        if fit_all:
            fit_summary = containment_summary(angular_error_np(
                np.concatenate(fit_all), truth))
            result["val_fit_angle_p68"] = fit_summary["p68"]
        if baseline_all:
            baseline_summary = containment_summary(angular_error_np(
                np.concatenate(baseline_all), truth))
            result.update({
                "val_centroid_angle_median": baseline_summary["median"],
                "val_centroid_angle_p68": baseline_summary["p68"],
                "val_centroid_angle_p90": baseline_summary["p90"],
            })

    if "energy" in tasks:
        energy_residual = np.concatenate(energy_residual_all)
        energy_true = np.concatenate(energy_true_all)
        roll = cfg.loss.get("energy_rolloff", None)
        default_cut = float(roll.get("e_cut", 2000.0)) if roll is not None else 2000.0
        e_cut = float(cfg.train.get("energy_select_e_cut_gev", default_cut))
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
        select_cfg = cfg.train.get("select_metric", None)
        core_name = select_cfg.get("core", "gauss") if select_cfg else "gauss"
        bias_name = select_cfg.get("bias", "gauss") if select_cfg else "gauss"
        tail_name = select_cfg.get("tail_term", "none") if select_cfg else "none"
        tail_weight = float(select_cfg.get("tail_weight", 0.0)) if select_cfg else 0.0
        core = {"gauss": gauss_res, "robust": robust, "raw_std": raw_std}[core_name]
        bias_value = {"gauss": gauss_bias, "median": median_bias,
                      "mean": mean_bias, "none": 0.0}[bias_name]
        tail_value = {"none": 0.0, "outlier": outlier, "raw_std": raw_std,
                      "excess": raw_std - robust}[tail_name]
        energy_metric = float(np.hypot(core, bias_value) + tail_weight * tail_value)
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
        if mode == "energy":
            # Preserve the established energy-only CSV/API field names.
            result.update({
                "val_bias": median_bias,
                "val_res": robust,
                "val_res_raw": raw_std,
                "val_res_gauss": gauss_res,
                "val_bias_gauss": gauss_bias,
                "val_gauss_ok": int(gaussian["ok"]),
                "val_outlier": outlier,
            })

    _, _, selection = validate_training_contract(cfg)
    if selection == "angle" and "val_angle_metric" in result:
        result["val_metric"] = result["val_angle_metric"]
    elif selection == "energy" and "val_energy_metric" in result:
        result["val_metric"] = result["val_energy_metric"]
    else:
        raise ValueError(
            f"task.selection={selection!r} is not available for active tasks {tasks}")
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


def checkpoint_specs(tasks):
    """All independently useful validation selections plus the configured primary."""
    specs = {"primary": ("val_metric", "best.pt")}
    if "energy" in tasks:
        specs.update({
            "energy": ("val_energy_metric", "best_energy.pt"),
            "energy_gauss": ("val_energy_res_gauss", "best_gauss.pt"),
            "energy_robust": ("val_energy_res_robust", "best_robust.pt"),
            "energy_raw": ("val_energy_res_raw", "best_raw.pt"),
        })
    if "angle" in tasks:
        specs.update({
            "angle": ("val_angle_p68", "best_angle.pt"),
            "angle_median": ("val_angle_median", "best_angle_median.pt"),
            "angle_mean": ("val_angle_mean", "best_angle_mean.pt"),
        })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    cfg, raw_cfg = load_config(args.config, args.overrides)
    mode, tasks, _ = validate_training_contract(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    meta = load_meta(cfg.paths.cache_dir)
    train_loader, val_loader = build_loaders(cfg, meta)
    fqw = cfg.loss.get("fit_quality_weight", None)
    fit_quality_active = bool(
        fqw is not None and fqw.get("enabled", False)
        and set(_fit_quality_apply_to(cfg)) & set(tasks))
    if fit_quality_active:
        scale = float(np.median(train_loader.dataset.fit_resid))
        setattr(fqw, "scale", scale)
        print(f"[fit_quality_weight] train median={scale:.5g} "
              f"apply_to={fqw.get('apply_to')}")

    concept_use = cfg.data.get("concept_use", None)
    require_meta(meta, ("n_concepts",), cache_dir=cfg.paths.cache_dir,
                 operation=f"{mode} model construction")
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    raw = EcalTransformer(cfg, n_concepts).to(device)
    if "energy" in tasks:
        require_meta(meta, ("log_energy_mean", "log_energy_std"),
                     cache_dir=cfg.paths.cache_dir, operation=f"{mode} training")
        raw.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    elif "log_energy_mean" in meta and "log_energy_std" in meta:
        raw.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    if "angle" in tasks:
        require_meta(meta, ("angle_mean", "angle_std"),
                     cache_dir=cfg.paths.cache_dir, operation=f"{mode} training")
        raw.set_angle_norm(meta["angle_mean"], meta["angle_std"])
        estimate_residual_norm(raw, train_loader, cfg, device)

    task_names = list(tasks)
    if "energy" in tasks and cfg.model.get("dual_energy_head", False):
        task_names.insert(task_names.index("energy") + 1, "e_phys")
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

    raw_cfg.setdefault("task", {})
    raw_cfg["task"].setdefault("mode", mode)
    raw_cfg["task"]["resolved_active_tasks"] = list(tasks)
    if os.path.exists("GIT_VERSION"):
        with open("GIT_VERSION") as handle:
            raw_cfg["git_version"] = handle.read().strip()
    with open(os.path.join(cfg.paths.out_dir, "config.json"), "w") as handle:
        json.dump(raw_cfg, handle, indent=2)
    csv_path = os.path.join(cfg.paths.out_dir, "metrics.csv")
    specs = checkpoint_specs(tasks)
    best = {name: float("inf") for name in specs}
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
                    losses = compute_losses(out, batch, cfg, tasks=tasks)
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
            nb = int(batch["feats"].shape[0])
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

        log_parts = [f"[{epoch:3d}] mode={mode} train={row['train_total']:.4f}"]
        if "angle" in tasks:
            log_parts.append(
                f"angle-p68={1e3*val['val_angle_p68']:.3f}mrad "
                f"median={1e3*val['val_angle_median']:.3f} "
                f"p90={1e3*val['val_angle_p90']:.3f} "
                f"out20={100*val['val_angle_outlier20mrad']:.2f}%")
            if "val_fit_angle_p68" in val:
                log_parts.append(f"fit-p68={1e3*val['val_fit_angle_p68']:.3f}mrad")
            if "val_centroid_angle_p68" in val:
                log_parts.append(
                    f"centroid-p68={1e3*val['val_centroid_angle_p68']:.3f}mrad")
        if "energy" in tasks:
            log_parts.append(
                f"energy-gauss={100*val['val_energy_res_gauss']:.3f}%"
                f"/{100*val['val_energy_bias_gauss']:+.3f}% "
                f"energy-tail={100*val['val_energy_outlier20pct']:.2f}%")
        print(" ".join(log_parts))

        checkpoint = {"model": raw.state_dict(), "weighter": weighter.state_dict(),
                      "meta": meta, "config": raw_cfg, "task_mode": mode,
                      "active_tasks": list(tasks), "epoch": epoch, "val": val}
        torch.save(checkpoint, os.path.join(cfg.paths.out_dir, "last.pt"))
        improved_primary = val[specs["primary"][0]] < best["primary"]
        for name, (metric_key, filename) in specs.items():
            value = val[metric_key]
            if value < best[name]:
                best[name] = value
                torch.save(checkpoint, os.path.join(cfg.paths.out_dir, filename))
        if improved_primary:
            bad_epochs = 0
        else:
            bad_epochs += 1
            min_epochs = int(cfg.train.get("early_stop_min_epochs", 0))
            if (bad_epochs >= cfg.train.early_stop_patience
                    and epoch + 1 >= min_epochs):
                print(f"early stop at epoch {epoch}; configured primary metric "
                      f"did not improve; best={best['primary']:.6g}")
                break

    print(f"done: mode={mode} best={best} -> {cfg.paths.out_dir}")


if __name__ == "__main__":
    main()
