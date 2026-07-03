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
from utils.stats import gauss_core
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


def aux_weight(batch, cfg):
    """v2m1 per-event fit-quality weight for the PHYSICS (concept/recon) losses.
    w = clip((m/resid)^beta, w_min, 1), m = train-median residual (set in main()).
    Energy loss is NEVER weighted (mcEne is truth). None when disabled / no fit_resid."""
    fqw = cfg.loss.get("fit_quality_weight", None)
    if fqw is None or not fqw.get("enabled", False) or "fit_resid" not in batch:
        return None
    scale = float(fqw.get("scale", 0.0))
    if scale <= 0.0:
        return None
    beta = float(fqw.get("beta", 1.0)); w_min = float(fqw.get("w_min", 0.2))
    resid = batch["fit_resid"].clamp_min(1e-9)
    return ((scale / resid) ** beta).clamp(w_min, 1.0)


def compute_losses(out, batch, cfg, apply_aux_weight=True):
    w = energy_weight(batch["energy"], cfg)            # per-sample weight on energy heads
    w_a = aux_weight(batch, cfg) if apply_aux_weight else None   # fit-quality weight (aux)
    fqw = cfg.loss.get("fit_quality_weight", None)
    apply_to = (fqw.get("apply_to", ["concept", "recon"]) if (fqw and w_a is not None) else [])
    wc = w_a if "concept" in apply_to else None
    wr = w_a if "recon" in apply_to else None
    losses = {}
    if "energy" in out:
        losses["energy"] = energy_loss(out["energy"], batch["log_e"], cfg.loss.energy.delta, w=w)
    if "e_phys" in out:                                # dual head: ground the physics energy
        losses["e_phys"] = energy_loss(out["e_phys"], batch["log_e"], cfg.loss.energy.delta, w=w)
    if "recon" in out:
        losses["recon"] = recon_loss(out["recon"], batch["recon"], batch["valid"],
                                     cfg.loss.recon.delta, w=wr)
    losses["concept"] = concept_loss(out["concepts"], batch["concepts"], cfg.loss.concept.delta, w=wc)
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
    rs, ets, agg, nev = [], [], {}, []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        e_pred = raw.predict_energy_gev(out["energy"].float())
        r = ((e_pred - batch["energy"]) / batch["energy"].clamp_min(1e-3)).cpu().numpy()
        rs.append(r)
        ets.append(batch["energy"].cpu().numpy())
        # EVENT-weight the per-task loss logs: the token-budget sampler yields very
        # different event-counts per batch, so a plain mean over batches is size-biased.
        # Store each batch's SUM (= mean * n_events); divide by total events below.
        nb = int(batch["energy"].shape[0])
        nev.append(nb)
        # log UNWEIGHTED aux losses so val_concept/val_recon stay comparable to baseline;
        # the fit-quality weight only shapes the TRAINING objective.
        for k, v in compute_losses(out, batch, cfg, apply_aux_weight=False).items():
            agg.setdefault(k, []).append(float(v) * nb)
    r = np.concatenate(rs); et = np.concatenate(ets)
    # v3: model selection on the GAUSSIAN CORE sigma (iterative binned fit in mu+/-2sigma,
    # utils/stats.gauss_core) + the fitted Gaussian mean as the bias, restricted to the
    # reliable range (E <= e_cut). This is the calorimetry-standard estimator; IQR/1.349
    # stays as the seed/cross-check column, and raw std / outlier fraction stay as the
    # tail diagnostics. gauss_core falls back to the robust core if the fit fails (early
    # epochs), so selection degrades gracefully instead of crashing.
    roll = cfg.loss.get("energy_rolloff", None)
    e_cut = float(roll.get("e_cut", 2000.0)) if roll is not None else 2000.0
    sel = (et <= e_cut) & np.isfinite(r)
    rr = r[sel] if int(sel.sum()) >= 50 else r[np.isfinite(r)]
    q75, q25 = np.percentile(rr, [75, 25])
    res = float((q75 - q25) / 1.349)                # robust core sigma  (cross-check)
    bias = float(np.median(rr))                     # median bias        (robust)
    bias_mean = float(np.mean(rr))                  # mean bias          (raw)
    outlier = float(np.mean(np.abs(rr) > 0.20))     # tail: fraction |dE/E| > 20%
    res_raw = float(np.std(rr))                     # raw std (tail-sensitive)
    g = gauss_core(rr)
    res_gauss = float(g["sigma"])                   # Gaussian core sigma (v3 PRIMARY)
    bias_gauss = float(g["mu"])                     # fitted Gaussian mean
    # v2b (a): CONFIGURABLE composite selection metric. v3 defaults core=gauss/bias=gauss;
    # set core=robust bias=median to reproduce the v2cham selection exactly.
    sm = cfg.train.get("select_metric", None)
    core_k = sm.get("core", "gauss") if sm else "gauss"
    bias_k = sm.get("bias", "gauss") if sm else "gauss"
    tail_k = sm.get("tail_term", "none") if sm else "none"
    tail_w = float(sm.get("tail_weight", 0.0)) if sm else 0.0
    core = {"gauss": res_gauss, "robust": res, "raw_std": res_raw}[core_k]
    bval = {"gauss": bias_gauss, "median": bias, "mean": bias_mean, "none": 0.0}[bias_k]
    tval = {"none": 0.0, "outlier": outlier, "raw_std": res_raw, "excess": res_raw - res}[tail_k]
    metric = float(np.sqrt(core ** 2 + bval ** 2) + tail_w * tval)   # -> drives selection
    n_total = max(sum(nev), 1)
    means = {k: float(np.sum(v)) / n_total for k, v in agg.items()}
    return {"val_bias": bias, "val_res": res, "val_res_raw": res_raw,
            "val_res_gauss": res_gauss, "val_bias_gauss": bias_gauss,
            "val_gauss_ok": int(g["ok"]),
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

    # v2m1 fit-quality weight scale = TRAIN-median residual (no leakage); set on cfg for
    # compute_losses (champion = wdx + v2m1 + v2b multi-best/select).
    fqw = cfg.loss.get("fit_quality_weight", None)
    if fqw is not None and fqw.get("enabled", False):
        m = float(np.median(train_loader.dataset.fit_resid))
        setattr(fqw, "scale", m)
        print(f"[fit_quality_weight] enabled: apply_to={fqw.get('apply_to')}, "
              f"beta={fqw.get('beta', 1.0)}, w_min={fqw.get('w_min', 0.2)}, "
              f"train-median resid scale={m:.4g}")

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

    # Keep compile=true but make it NON-FATAL. torch.compile is lazy (it traces on
    # the FIRST forward), and inductor has a documented dynamic-shape crash on this
    # full scaffolded model (sympy `assert p>=0`; it killed the m1/m4/m2 runs and can
    # also fire on a mid-training recompile). Guard the construct here AND the runtime
    # step below; eager is numerically identical, so we just fall back to it.
    compiled = False
    if cfg.train.compile:
        try:
            model = torch.compile(raw, dynamic=True)
            compiled = True
        except Exception as e:
            print(f"[warn] torch.compile() failed ({type(e).__name__}: {e}); using eager")
            model = raw
    else:
        model = raw
    amp_dtype = amp_dtype_of(cfg)

    if str(cfg.train.optimizer).lower() != "adamw":
        raise ValueError(f"train.optimizer={cfg.train.optimizer!r}: only adamw is implemented")
    # Mild weight decay on the Kendall-Gal log-variances (was 0.0 to avoid biasing
    # weights toward 1). With the loss normalisation in UncertaintyWeighter the weights
    # no longer blow up, and this gentle pull toward s->0 self-limits any residual
    # drift instead of diverging on large datasets. Set loss.logvar_weight_decay=0 to
    # restore the old un-decayed behaviour.
    # Weight-decay groups. By DEFAULT every model param is decayed at train.weight_decay
    # -- this exactly reproduces sw_d192 (the v2 baseline contract). Set
    # train.wd_exclude_1d=true for the standard transformer recipe: exclude 1-D params
    # (LayerNorm scales, biases, the AttnPool query) from WD. That is a deliberate A/B
    # CHANGE to the trained objective, not part of the reproduction, hence opt-in.
    if cfg.train.get("wd_exclude_1d", False):
        decay = [p for p in raw.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay = [p for p in raw.parameters() if p.requires_grad and p.ndim < 2]
        groups = [{"params": decay, "weight_decay": cfg.train.weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
    else:
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
    # code provenance: GIT_VERSION is written by the repo's post-commit hook on the
    # dev machine and reaches the server via Mutagen (the server has no .git).
    if os.path.exists("GIT_VERSION"):
        with open("GIT_VERSION") as f:
            raw_cfg["git_version"] = f.read().strip()
    with open(os.path.join(cfg.paths.out_dir, "config.json"), "w") as f:
        json.dump(raw_cfg, f, indent=2)

    # v2b (b): track MULTIPLE "best" checkpoints in one run so the selection-metric effect
    # can be compared post-hoc without retraining. best.pt = configured composite metric
    # (also drives early stop; v3 default = Gaussian core + fitted-mean bias);
    # best_gauss.pt = min Gaussian core alone; best_robust.pt = min robust core;
    # best_raw.pt = min raw std (the sw_d192-style selection).
    bests = {"metric": float("inf"), "gauss": float("inf"),
             "robust": float("inf"), "raw": float("inf")}
    bfile = {"metric": "best.pt", "gauss": "best_gauss.pt",
             "robust": "best_robust.pt", "raw": "best_raw.pt"}
    bcrit = {"metric": "val_metric", "gauss": "val_res_gauss",
             "robust": "val_res", "raw": "val_res_raw"}
    bad, header = 0, None
    for epoch in range(cfg.train.epochs):
        model.train()
        tot, task_sums, n_batches, n_events = 0.0, {t: 0.0 for t in task_names}, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            try:
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(batch)
                    losses = compute_losses(out, batch, cfg)
                    total, _ = weighter(losses)
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, cfg.train.grad_clip)
                opt.step()
            except Exception as e:
                # inductor can raise at the first forward or on a mid-training
                # recompile. Fall back to eager ONCE (skipping this one batch) and
                # keep training; if we are already eager this is a real error.
                if not compiled:
                    raise
                print(f"[warn] compiled step failed ({type(e).__name__}: {e}); "
                      "switching to eager for the rest of training")
                model, compiled = raw, False
                continue
            # EVENT-weight epoch averages: batches vary a lot in event-count under the
            # token-budget sampler, so accumulate each batch's SUM (= mean * n_events)
            # and divide by total events below -> per-event means, not size-biased.
            nb = int(batch["energy"].shape[0])
            tot += float(total.detach()) * nb
            for t, l in losses.items():
                task_sums[t] += float(l.detach()) * nb
            n_batches += 1
            n_events += nb
        sched.step()

        val = validate(model, raw, val_loader, cfg, device, amp_dtype)
        # NOTE: train_total is the optimization objective and includes the +0.5*s
        # regularizer (can be negative); the train_<task> columns are the raw,
        # comparable per-task losses, EVENT-weighted (per-event mean) over the epoch.
        wlogs = {f"w_{t}": w for t, w in weighter.task_weights().items()}
        wlogs.update({f"train_{t}": task_sums[t] / max(n_events, 1) for t in task_names})
        row = {"epoch": epoch, "train_total": tot / max(n_events, 1),
               "lr": opt.param_groups[0]["lr"], **val, **wlogs}
        if header is None:
            header = list(row.keys())
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, header).writeheader()
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, header).writerow(row)
        print(f"[{epoch:3d}] train={row['train_total']:.4f} "
              f"gauss={val['val_res_gauss']*100:.2f}%/{val['val_bias_gauss']*100:+.2f}% "
              f"robust={val['val_res']*100:.2f}% tail={val['val_outlier']*100:.1f}% "
              f"metric={val['val_metric']*100:.2f}% "
              f"(raw={val['val_res_raw']*100:.2f}%) concept={val['val_concept']:.3f}")

        ckpt = {"model": raw.state_dict(), "weighter": weighter.state_dict(),
                "meta": meta, "config": raw_cfg, "epoch": epoch, "val": val}
        torch.save(ckpt, os.path.join(cfg.paths.out_dir, "last.pt"))
        improved = val["val_metric"] < bests["metric"]      # composite drives early stop
        for key in ("metric", "gauss", "robust", "raw"):
            if val[bcrit[key]] < bests[key]:
                bests[key] = val[bcrit[key]]
                torch.save(ckpt, os.path.join(cfg.paths.out_dir, bfile[key]))
        if improved:
            bad = 0
        else:
            bad += 1
            if bad >= cfg.train.early_stop_patience:
                print(f"early stop at epoch {epoch} (best composite {bests['metric']*100:.2f}%)")
                break
    print(f"done. best composite={bests['metric']*100:.2f}%  gauss={bests['gauss']*100:.2f}%  "
          f"robust={bests['robust']*100:.2f}%  raw={bests['raw']*100:.2f}%  ->  "
          f"{cfg.paths.out_dir}/{{best,best_gauss,best_robust,best_raw}}.pt")


if __name__ == "__main__":
    main()
