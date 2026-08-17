"""Export fp32 incidence-direction predictions with stable run/event identity."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer
from utils.angle import slope_to_theta_phi_np, slopes_to_unit_np
from utils.config import load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    meta = load_meta(cfg.paths.cache_dir)
    checkpoint_path = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")
    output = args.output or os.path.join(cfg.paths.out_dir, f"predictions_{args.split}.npz")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(cfg.device)
    model.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    model.set_angle_norm(meta["angle_mean"], meta["angle_std"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device)["model"])
    model.eval()

    dataset = EcalTokens(
        cfg.paths.cache_dir, args.split, meta, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None))
    loader = make_loader(dataset, cfg, shuffle=False)
    pred, truth, fit, energy, run, event = [], [], [], [], [], []
    for batch in loader:
        run.append(batch["run"].numpy())
        event.append(batch["event"].numpy())
        energy.append(batch["energy"].numpy())
        truth.append(batch["angle"].numpy())
        fit.append(batch["fit_angle"].numpy())
        gpu = {k: (v if k in ("run", "event") else v.to(cfg.device))
               for k, v in batch.items()}
        out = model(gpu)  # fp32
        pred.append(model.predict_angle_slopes(out["angle"].float()).cpu().numpy())

    pred = np.concatenate(pred)
    truth = np.concatenate(truth)
    fit = np.concatenate(fit)
    pred_theta, pred_phi = slope_to_theta_phi_np(pred)
    truth_theta, truth_phi = slope_to_theta_phi_np(truth)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        run=np.concatenate(run).astype(np.uint32),
        event=np.concatenate(event).astype(np.uint32),
        energy_gev=np.concatenate(energy).astype(np.float32),
        pred_kx=pred[:, 0].astype(np.float32),
        pred_ky=pred[:, 1].astype(np.float32),
        pred_dir=slopes_to_unit_np(pred).astype(np.float32),
        pred_theta=pred_theta.astype(np.float32),
        pred_phi=pred_phi.astype(np.float32),
        truth_kx=truth[:, 0].astype(np.float32),
        truth_ky=truth[:, 1].astype(np.float32),
        truth_dir=slopes_to_unit_np(truth).astype(np.float32),
        truth_theta=truth_theta.astype(np.float32),
        truth_phi=truth_phi.astype(np.float32),
        fit_kx=fit[:, 0].astype(np.float32),
        fit_ky=fit[:, 1].astype(np.float32),
    )
    print(f"wrote {output} events={len(pred)} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
