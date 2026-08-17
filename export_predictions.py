"""Export model predictions from MC or target-free ISS token caches.

By default the cache contract is detector inputs plus stable ``run,event``
identity only. MC truth and the 3D-fit reference are opt-in export columns.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from data.schema import require_meta
from models.model import EcalTransformer
from utils.angle import slope_to_theta_phi_np, slopes_to_unit_np
from utils.config import load_config
from utils.tasks import validate_training_contract


def _append_direction(columns, prefix, slopes):
    theta, phi = slope_to_theta_phi_np(slopes)
    columns[f"{prefix}_kx"] = slopes[:, 0].astype(np.float32)
    columns[f"{prefix}_ky"] = slopes[:, 1].astype(np.float32)
    columns[f"{prefix}_dir"] = slopes_to_unit_np(slopes).astype(np.float32)
    columns[f"{prefix}_theta"] = theta.astype(np.float32)
    columns[f"{prefix}_phi"] = phi.astype(np.float32)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/angle.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-angle-truth", action="store_true")
    parser.add_argument("--include-energy-truth", action="store_true")
    parser.add_argument("--include-fit-angle", action="store_true")
    args = parser.parse_args(argv)
    cfg, _ = load_config(args.config, args.overrides)
    mode, tasks, _ = validate_training_contract(cfg)
    if not set(tasks) & {"energy", "angle"}:
        raise ValueError(f"no prediction task is active: {tasks}")

    cache_meta = load_meta(cfg.paths.cache_dir)
    checkpoint_path = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")
    checkpoint = torch.load(
        checkpoint_path, map_location=cfg.device, weights_only=False)
    training_meta = checkpoint.get("meta")
    if training_meta is None:
        raise ValueError(
            f"prediction checkpoint {checkpoint_path!r} has no training metadata; "
            "normalization cannot be reconstructed safely")
    require_meta(training_meta, ("n_concepts", "geometry_data_type"),
                 cache_dir=checkpoint_path, operation="prediction checkpoint loading")
    require_meta(cache_meta, ("geometry_data_type",),
                 cache_dir=cfg.paths.cache_dir, operation="prediction cache loading")
    if training_meta["geometry_data_type"] != cache_meta["geometry_data_type"]:
        raise ValueError(
            "checkpoint and prediction cache geometries differ: "
            f"{training_meta['geometry_data_type']!r} != "
            f"{cache_meta['geometry_data_type']!r}")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else training_meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(cfg.device)
    if "energy" in tasks:
        require_meta(training_meta, ("log_energy_mean", "log_energy_std"),
                     cache_dir=checkpoint_path, operation="energy prediction")
        model.set_energy_norm(
            training_meta["log_energy_mean"], training_meta["log_energy_std"])
    if "angle" in tasks:
        require_meta(training_meta, ("angle_mean", "angle_std"),
                     cache_dir=checkpoint_path, operation="angle prediction")
        model.set_angle_norm(training_meta["angle_mean"], training_meta["angle_std"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    required = set()
    if args.include_angle_truth:
        required.add("angle")
    if args.include_energy_truth:
        required.add("energy")
    # Dataset normalization for optional truth uses training statistics. Cache
    # metadata supplies and verifies the detector geometry only.
    dataset_meta = dict(training_meta)
    dataset_meta["geometry_data_type"] = cache_meta["geometry_data_type"]
    dataset = EcalTokens(
        cfg.paths.cache_dir, args.split, dataset_meta, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        max_events=cfg.data.get("predict_max_events", None), subset_seed=cfg.seed + 3,
        task_mode=mode, include_targets=False, required_fields=required,
        require_fit_angle=args.include_fit_angle, require_identity=True,
        compute_fit_resid=False, operation="target-free prediction export")
    loader = make_loader(dataset, cfg, shuffle=False)

    collected = {"run": [], "event": []}
    if "angle" in tasks:
        collected["pred_angle"] = []
    if "energy" in tasks:
        collected["pred_energy"] = []
    for name, enabled in (
            ("truth_angle", args.include_angle_truth),
            ("truth_energy", args.include_energy_truth),
            ("fit_angle", args.include_fit_angle)):
        if enabled:
            collected[name] = []

    for batch in loader:
        collected["run"].append(batch["run"].numpy())
        collected["event"].append(batch["event"].numpy())
        if args.include_angle_truth:
            collected["truth_angle"].append(batch["angle"].numpy())
        if args.include_energy_truth:
            collected["truth_energy"].append(batch["energy"].numpy())
        if args.include_fit_angle:
            collected["fit_angle"].append(batch["fit_angle"].numpy())
        gpu = {key: (value if key in ("run", "event") else value.to(cfg.device))
               for key, value in batch.items()}
        output = model(gpu)  # fp32 by design for exported physics values
        if "angle" in tasks:
            collected["pred_angle"].append(
                model.predict_angle_slopes(output["angle"].float()).cpu().numpy())
        if "energy" in tasks:
            collected["pred_energy"].append(
                model.predict_energy_gev(output["energy"].float()).cpu().numpy())

    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    columns = {
        "run": arrays["run"].astype(np.uint32),
        "event": arrays["event"].astype(np.uint32),
    }
    if "pred_angle" in arrays:
        _append_direction(columns, "pred", arrays["pred_angle"])
    if "pred_energy" in arrays:
        columns["pred_energy_gev"] = arrays["pred_energy"].astype(np.float32)
    if "truth_angle" in arrays:
        _append_direction(columns, "truth", arrays["truth_angle"])
    if "truth_energy" in arrays:
        columns["truth_energy_gev"] = arrays["truth_energy"].astype(np.float32)
    if "fit_angle" in arrays:
        columns["fit_kx"] = arrays["fit_angle"][:, 0].astype(np.float32)
        columns["fit_ky"] = arrays["fit_angle"][:, 1].astype(np.float32)

    output_path = args.output or os.path.join(
        cfg.paths.out_dir, f"predictions_{args.split}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **columns)
    print(f"wrote {output_path} events={len(columns['run'])} "
          f"fields={sorted(columns)} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
