"""Evaluate ECAL incidence direction against MC truth and the 3D-fit baseline.

Inference is deliberately fp32. Primary resolution is the 68% containment of the
3D opening angle, reported in degrees by the production config; angular-error
distributions are radial/non-negative,
so the energy project's Gaussian residual-core estimator is not applicable.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from data.schema import require_meta
from models.model import EcalTransformer
from utils.angle import (angular_error_np, containment_summary,
                         slopes_to_unit_np)
from utils.config import load_config
from utils.stats import gauss_core
from utils.tasks import cache_fields_for_tasks, validate_training_contract


def robust_sigma(values):
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float((q75 - q25) / 1.349)


def load_optional_split_array(cache_dir, split, name):
    """Load a small optional event-level field from either cache layout."""
    path = os.path.join(cache_dir, split, f"{name}.npy")
    if os.path.isfile(path):
        return np.asarray(np.load(path, mmap_mode="r"))
    packed = os.path.join(cache_dir, f"{split}.npz")
    with np.load(packed) as arrays:
        return np.asarray(arrays[name]) if name in arrays.files else None


def align_cache_field_to_output(cache_dir, split, values, output_run, output_event):
    """Align a cache-order event field to token-budget loader output order.

    Even with shuffle disabled, TokenBudgetBatchSampler sorts events by token
    length to control padding.  Physics arrays collected from model batches are
    therefore not in cache row order.  Stable (run,event) identity prevents a
    silent subgroup-mask permutation.
    """
    cache_run = load_optional_split_array(cache_dir, split, "run")
    cache_event = load_optional_split_array(cache_dir, split, "event")
    if cache_run is None or cache_event is None:
        raise ValueError("cache lacks run/event identity needed for field alignment")
    cache_key = (cache_run.astype(np.uint64) << np.uint64(32)) | cache_event.astype(np.uint64)
    output_key = (output_run.astype(np.uint64) << np.uint64(32)) | output_event.astype(np.uint64)
    if len(np.unique(cache_key)) != len(cache_key):
        raise ValueError("cache run/event identities are not unique")
    order = np.argsort(cache_key)
    sorted_key = cache_key[order]
    position = np.searchsorted(sorted_key, output_key)
    if np.any(position == len(sorted_key)) or not np.array_equal(
            sorted_key[position], output_key):
        raise ValueError("model output contains run/event identity absent from cache")
    return np.asarray(values)[order[position]]


def training_output_status(checkpoint):
    """Mark which inference heads actually received a training objective."""
    raw = checkpoint.get("config", {})
    heads = raw.get("heads", {})
    loss = raw.get("loss", {})
    weighting = loss.get(
        "weighting", "uncertainty" if loss.get("uncertainty_weighting", True)
        else "fixed")
    weights = loss.get("weights", {}) if weighting == "normalized" else {}
    status = {}
    for name in ("angle", "energy", "recon", "concept"):
        enabled = True if name == "concept" else heads.get(name, {}).get("enabled", False)
        weight = float(weights.get(name, 1.0))
        status[name] = {
            "head_enabled": bool(enabled),
            "trained_with_nonzero_loss": bool(enabled and weight != 0.0),
            "configured_weight": weight if weighting == "normalized" else None,
        }
    return {"seed": raw.get("seed"), "loss_weighting": weighting, "outputs": status}


@torch.no_grad()
def run_model(model, loader, device, tasks):
    fields = {name: [] for name in
              ("pred", "truth", "fit", "energy", "run", "event",
               "concept_pred", "concept_true", "recon_mse", "energy_pred",
               "centroid")}
    model.eval()
    for batch in loader:
        run = batch["run"].numpy()
        event = batch["event"].numpy()
        gpu = {k: (v if k in ("run", "event") else v.to(device))
               for k, v in batch.items()}
        out = model(gpu)  # fp32 on purpose
        fields["pred"].append(model.predict_angle_slopes(out["angle"].float()).cpu().numpy())
        fields["truth"].append(gpu["angle"].cpu().numpy())
        fields["fit"].append(gpu["fit_angle"].cpu().numpy())
        if "angle_baseline" in out:
            fields["centroid"].append(out["angle_baseline"].float().cpu().numpy())
        fields["energy"].append(gpu["energy"].cpu().numpy())
        fields["run"].append(run)
        fields["event"].append(event)
        if "concept" in tasks:
            fields["concept_pred"].append(out["concepts"].float().cpu().numpy())
            fields["concept_true"].append(gpu["concepts"].cpu().numpy())
        if "recon" in tasks:
            valid = gpu["valid"]
            mse = ((out["recon"].float() - gpu["recon"]) ** 2 * valid).sum(1)
            mse = mse / valid.sum(1).clamp_min(1)
            fields["recon_mse"].append(mse.cpu().numpy())
        if "energy" in tasks:
            fields["energy_pred"].append(
                model.predict_energy_gev(out["energy"].float()).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in fields.items() if parts}


def binned(values, model_err, fit_err, bins, angle_scale=1e3, angle_unit="mrad"):
    result = {name: [] for name in
              ("center", "low", "high", "count",
               f"model_median_{angle_unit}", f"model_p68_{angle_unit}",
               f"model_p90_{angle_unit}", f"fit_median_{angle_unit}",
               f"fit_p68_{angle_unit}", f"fit_p90_{angle_unit}")}
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (values >= low) & (values < high)
        if int(mask.sum()) < 20:
            continue
        model = containment_summary(model_err[mask])
        fit = containment_summary(fit_err[mask])
        result["center"].append(0.5 * (low + high))
        result["low"].append(low)
        result["high"].append(high)
        result["count"].append(int(mask.sum()))
        for prefix, summary in (("model", model), ("fit", fit)):
            for metric in ("median", "p68", "p90"):
                result[f"{prefix}_{metric}_{angle_unit}"].append(
                    angle_scale * summary[metric])
    return result


def component_metrics(pred, truth):
    residual = pred - truth
    out = {}
    for index, name in enumerate(("kx", "ky")):
        r = residual[:, index]
        out[name] = {
            "bias": float(np.mean(r)),
            "median_bias": float(np.median(r)),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "std": float(np.std(r)),
            "robust_sigma": robust_sigma(r),
        }
    return out


def energy_summary(residual):
    """Energy-resolution estimators matching the promoted energy report."""
    residual = np.asarray(residual, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    gaussian = gauss_core(residual)
    return {
        "count": int(len(residual)),
        "gaussian_core_percent": 100.0 * float(gaussian["sigma"]),
        "gaussian_bias_percent": 100.0 * float(gaussian["mu"]),
        "gaussian_fit_ok": bool(gaussian["ok"]),
        "robust_core_percent": 100.0 * robust_sigma(residual),
        "median_bias_percent": 100.0 * float(np.median(residual)),
        "raw_std_percent": 100.0 * float(np.std(residual)),
        "mean_bias_percent": 100.0 * float(np.mean(residual)),
        "outlier_fraction_gt20pct": float(np.mean(np.abs(residual) > 0.20)),
    }


def energy_binned(energy, residual, bins):
    out = {name: [] for name in (
        "center", "low", "high", "count", "gaussian_core_percent",
        "gaussian_bias_percent", "robust_core_percent", "raw_std_percent",
        "outlier_fraction_gt20pct")}
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (energy >= low) & (energy < high) & np.isfinite(residual)
        if int(mask.sum()) < 20:
            continue
        summary = energy_summary(residual[mask])
        out["center"].append(float(0.5 * (low + high)))
        out["low"].append(float(low))
        out["high"].append(float(high))
        for name in out:
            if name not in ("center", "low", "high"):
                out[name].append(summary[name])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    mode, tasks, _ = validate_training_contract(cfg)
    if "angle" not in tasks:
        raise ValueError(
            f"evaluate_angle.py requires an active angle task; active tasks are {tasks}")
    device = cfg.device
    cache_meta = load_meta(cfg.paths.cache_dir)
    checkpoint_path = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Inference normalization belongs to the trained model, not the population
    # on which it is being tested.  This matters for an uncut cache whose energy,
    # angle, and concept distributions differ substantially from the training
    # selection.  The legacy same-cache evaluation is unchanged because the two
    # metadata dictionaries are then identical.
    meta = checkpoint.get("meta", cache_meta)
    require_meta(meta, ("n_concepts", "geometry_data_type", "angle_mean", "angle_std"),
                 cache_dir=checkpoint_path, operation="angle checkpoint loading")
    require_meta(cache_meta, ("geometry_data_type",),
                 cache_dir=cfg.paths.cache_dir, operation="angle evaluation")
    if "concept" in tasks:
        require_meta(meta, ("concept_names", "concept_mean", "concept_std"),
                     cache_dir=checkpoint_path, operation="concept evaluation")
        require_meta(cache_meta, ("concept_names",), cache_dir=cfg.paths.cache_dir,
                     operation="concept evaluation")
        if list(meta["concept_names"]) != list(cache_meta["concept_names"]):
            raise ValueError("checkpoint and evaluation cache concept schemas differ")
    if meta["geometry_data_type"] != cache_meta["geometry_data_type"]:
        raise ValueError("checkpoint and evaluation cache geometries differ")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    if "energy" in tasks:
        require_meta(meta, ("log_energy_mean", "log_energy_std"),
                     cache_dir=checkpoint_path, operation="energy evaluation")
        model.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    model.set_angle_norm(meta["angle_mean"], meta["angle_std"])
    model.load_state_dict(checkpoint["model"], strict=True)

    # Direction scoring always needs MC direction and energy (for the standard
    # resolution-vs-energy report). The configured production comparison also
    # requires the 3D-fit reference. Aux targets are required only when their
    # losses were active, and are never fabricated.
    require_fit = bool(cfg.eval.get("require_fit_angle", True))
    if not require_fit:
        raise ValueError(
            "evaluate_angle.py is an MC comparison and requires "
            "eval.require_fit_angle=true; use export_predictions.py for "
            "target-free inference")
    required = set(cache_fields_for_tasks(tasks))
    required.update(("angle", "energy"))
    dataset = EcalTokens(
        cfg.paths.cache_dir, "test", meta, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        max_events=cfg.data.get("eval_max_events", None), subset_seed=cfg.seed + 2,
        task_mode=mode, required_fields=required, require_fit_angle=True,
        require_identity=True, compute_fit_resid=False,
        operation="angle MC evaluation")
    arrays = run_model(
        model, make_loader(dataset, cfg, shuffle=False), device, tasks)
    pred, truth, fit = arrays["pred"], arrays["truth"], arrays["fit"]
    if not np.isfinite(pred).all():
        raise RuntimeError(f"non-finite predictions: {(~np.isfinite(pred)).sum()}")
    model_err = angular_error_np(pred, truth)
    fit_err = angular_error_np(fit, truth)
    model_summary = containment_summary(model_err)
    fit_summary = containment_summary(fit_err)

    unit_truth = slopes_to_unit_np(truth)
    incidence_deg = np.degrees(np.arccos(np.clip(-unit_truth[:, 2], -1.0, 1.0)))
    angle_unit = str(cfg.eval.get("angle_unit", "degree")).lower()
    if angle_unit in ("degree", "degrees", "deg"):
        angle_unit, angle_scale = "deg", 180.0 / np.pi
    elif angle_unit == "mrad":
        angle_scale = 1e3
    else:
        raise ValueError("eval.angle_unit must be degree/deg or mrad")
    energy_bins = np.asarray(cfg.eval.energy_bins, dtype=float)
    incidence_bins = np.asarray(cfg.eval.incidence_bins_deg, dtype=float)
    by_energy = binned(
        arrays["energy"], model_err, fit_err, energy_bins, angle_scale, angle_unit)
    by_incidence = binned(
        incidence_deg, model_err, fit_err, incidence_bins, angle_scale, angle_unit)

    concepts = {}
    if "concept" in tasks:
        cmean = np.asarray(dataset.cmean)
        cstd = np.asarray(dataset.cstd)
        cp = arrays["concept_pred"] * cstd + cmean
        ct = arrays["concept_true"] * cstd + cmean
        for index, name in enumerate(dataset.concept_names):
            valid_concept = np.isfinite(cp[:, index]) & np.isfinite(ct[:, index])
            pred_col, true_col = cp[valid_concept, index], ct[valid_concept, index]
            mse = np.mean((pred_col - true_col) ** 2)
            variance = np.var(true_col) + 1e-12
            concepts[name] = {
                "count": int(valid_concept.sum()),
                "rmse": float(np.sqrt(mse)),
                "r2": float(1.0 - mse / variance),
            }

    energy_residual = None
    energy_select = None
    energy_metrics = None
    if "energy" in tasks:
        energy_residual = ((arrays["energy_pred"] - arrays["energy"]) /
                           np.clip(arrays["energy"], 1e-3, None))
        energy_e_cut = float(cfg.eval.get("energy_select_e_cut_gev", 2000.0))
        energy_select = ((arrays["energy"] <= energy_e_cut)
                         & np.isfinite(energy_residual))
        energy_metrics = energy_summary(energy_residual[energy_select])
        energy_metrics["selection_max_gev"] = energy_e_cut
        energy_metrics["all_energy"] = energy_summary(energy_residual)
        energy_metrics["by_energy_gev"] = energy_binned(
            arrays["energy"], energy_residual, energy_bins)

    # The memmap all-event cache stores the two fields that defined the old
    # training selection.  Report both deployment-population performance and
    # the familiar contained/single-shower domain in the same output.
    fit_status = load_optional_split_array(
        cfg.paths.cache_dir, "test", "fit_status")
    n_shower = load_optional_split_array(
        cfg.paths.cache_dir, "test", "n_shower")
    group_metrics = {}
    if fit_status is not None and n_shower is not None:
        fit_status = align_cache_field_to_output(
            cfg.paths.cache_dir, "test", fit_status, arrays["run"], arrays["event"])
        n_shower = align_cache_field_to_output(
            cfg.paths.cache_dir, "test", n_shower, arrays["run"], arrays["event"])
        if len(fit_status) != len(model_err) or len(n_shower) != len(model_err):
            raise ValueError("quality-field alignment did not match evaluation events")
        contained = (fit_status.astype(np.int64) & 7) == 7
        single = n_shower.astype(np.int64) == 1
        groups = {
            "all_uncut": np.ones(len(model_err), dtype=bool),
            "old_selection_contained_single": contained & single,
            "outside_old_selection": ~(contained & single),
            "contained_any_nshower": contained,
            "single_shower_any_status": single,
        }
        centroid_err = (angular_error_np(arrays["centroid"], truth)
                        if "centroid" in arrays else None)
        for group_name, mask in groups.items():
            model_group = containment_summary(model_err[mask])
            finite_fit = mask & np.isfinite(fit_err)
            fit_group = (containment_summary(fit_err[finite_fit])
                         if int(finite_fit.sum()) else None)
            entry = {
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "model": {
                    (key if key == "count" else key + "_" + angle_unit):
                    (value if key == "count" else angle_scale * value)
                    for key, value in model_group.items()
                },
                "fit3d_finite_fraction": float(finite_fit.sum() / max(mask.sum(), 1)),
            }
            if energy_residual is not None:
                entry["energy_reconstruction"] = energy_summary(
                    energy_residual[mask & energy_select])
            if "recon_mse" in arrays:
                entry["recon_logmse_mean"] = float(
                    np.mean(arrays["recon_mse"][mask]))
            if fit_group is not None:
                entry["fit3d"] = {
                    (key if key == "count" else key + "_" + angle_unit):
                    (value if key == "count" else angle_scale * value)
                    for key, value in fit_group.items()
                }
            if centroid_err is not None:
                centroid_group = containment_summary(centroid_err[mask])
                entry["centroid_baseline"] = {
                    (key if key == "count" else key + "_" + angle_unit):
                    (value if key == "count" else angle_scale * value)
                    for key, value in centroid_group.items()
                }
            group_metrics[group_name] = entry

    metrics = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_training": training_output_status(checkpoint),
        "evaluation_cache": cfg.paths.cache_dir,
        "evaluation_cache_selection": cache_meta.get("selection"),
        "normalization_source": "checkpoint_training_meta",
        "n_test": int(len(model_err)),
        "direction_representation": "slopes_dx_dz_dy_dz",
        "inference_precision": "fp32",
        "angle_unit": angle_unit,
        "model": {(key if key == "count" else key + "_" + angle_unit):
                  (value if key == "count" else angle_scale * value)
                  for key, value in model_summary.items()},
        "fit3d": {(key if key == "count" else key + "_" + angle_unit):
                  (value if key == "count" else angle_scale * value)
                  for key, value in fit_summary.items()},
        "improvement_p68_percent": float(
            100.0 * (fit_summary["p68"] - model_summary["p68"]) / fit_summary["p68"]),
        "outlier_fraction_gt1p1459deg": float(np.mean(model_err > 0.020)),
        "fit3d_outlier_fraction_gt1p1459deg": float(np.mean(fit_err > 0.020)),
        "components": component_metrics(pred, truth),
        "fit3d_components": component_metrics(fit, truth),
        "by_energy_gev": by_energy,
        "by_incidence_deg": by_incidence,
    }
    if energy_metrics is not None:
        metrics.update({
            "aux_energy_residual_std": float(np.std(energy_residual)),
            "aux_energy_bias": float(np.mean(energy_residual)),
            "energy_reconstruction": energy_metrics,
        })
    if "recon_mse" in arrays:
        metrics["aux_recon_logmse"] = float(np.mean(arrays["recon_mse"]))
    if concepts:
        metrics["concepts"] = concepts
    if group_metrics:
        metrics["event_groups"] = group_metrics
    if "centroid" in arrays:
        centroid_summary = containment_summary(angular_error_np(
            arrays["centroid"], truth))
        metrics["centroid_baseline"] = {
            (key if key == "count" else key + "_" + angle_unit):
            (value if key == "count" else angle_scale * value)
            for key, value in centroid_summary.items()
        }
    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    headline = {
        "n_test": metrics["n_test"],
        f"model_median_{angle_unit}": metrics["model"][f"median_{angle_unit}"],
        f"model_p68_{angle_unit}": metrics["model"][f"p68_{angle_unit}"],
        f"model_p90_{angle_unit}": metrics["model"][f"p90_{angle_unit}"],
        f"fit3d_p68_{angle_unit}": metrics["fit3d"][f"p68_{angle_unit}"],
        "improvement_p68_percent": metrics["improvement_p68_percent"],
        "outlier_fraction_gt1p1459deg": metrics["outlier_fraction_gt1p1459deg"],
    }
    if energy_metrics is not None:
        headline.update({
            "energy_gaussian_core_percent": energy_metrics[
                "gaussian_core_percent"],
            "energy_gaussian_bias_percent": energy_metrics[
                "gaussian_bias_percent"],
        })
    print(json.dumps(headline, indent=2))

    out_dir = cfg.paths.out_dir
    plt.figure()
    plt.plot(by_energy["center"], by_energy[f"model_p68_{angle_unit}"], "o-", label="Transformer")
    plt.plot(by_energy["center"], by_energy[f"fit_p68_{angle_unit}"], "s--", label="3D fit")
    plt.xscale("log"); plt.xlabel("MC energy [GeV]"); plt.ylabel(f"68% angle [{angle_unit}]")
    plt.grid(True, alpha=.3); plt.legend(); plt.title("Direction resolution vs energy")
    plt.savefig(os.path.join(out_dir, "angle_resolution_vs_energy.png"),
                dpi=140, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(by_incidence["center"], by_incidence[f"model_p68_{angle_unit}"], "o-", label="Transformer")
    plt.plot(by_incidence["center"], by_incidence[f"fit_p68_{angle_unit}"], "s--", label="3D fit")
    plt.xlabel("Incidence angle from -z [deg]"); plt.ylabel(f"68% angle [{angle_unit}]")
    plt.grid(True, alpha=.3); plt.legend(); plt.title("Direction resolution vs incidence")
    plt.savefig(os.path.join(out_dir, "angle_resolution_vs_incidence.png"),
                dpi=140, bbox_inches="tight"); plt.close()

    finite_fit_err = fit_err[np.isfinite(fit_err)]
    upper = max(np.quantile(model_err, .995),
                np.quantile(finite_fit_err, .995)) * angle_scale
    plt.figure()
    plt.hist(model_err * angle_scale, bins=120, range=(0, upper), histtype="step", label="Transformer")
    plt.hist(finite_fit_err * angle_scale, bins=120, range=(0, upper),
             histtype="step", label="3D fit")
    plt.xlabel(f"Opening-angle error [{angle_unit}]"); plt.ylabel("Events"); plt.legend()
    plt.title("MC direction residual")
    plt.savefig(os.path.join(out_dir, "angular_error_hist.png"),
                dpi=140, bbox_inches="tight"); plt.close()

    residual = pred - truth
    plt.figure()
    plt.hist(residual[:, 0], bins=120, range=(-.05, .05), histtype="step", label="kx")
    plt.hist(residual[:, 1], bins=120, range=(-.05, .05), histtype="step", label="ky")
    plt.xlabel("Predicted - true slope"); plt.ylabel("Events"); plt.legend()
    plt.title("Slope-component residuals")
    plt.savefig(os.path.join(out_dir, "slope_residuals.png"),
                dpi=140, bbox_inches="tight"); plt.close()
    print("wrote metrics.json + 4 plots ->", out_dir)


if __name__ == "__main__":
    main()
