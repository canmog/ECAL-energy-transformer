"""Physics and symmetry probes for the trained direction model.

1. Linear concept decodability in h_phys versus h_free.
2. Causal h_phys/h_free ablations measured with angular containment.
3. Exact X/Y mirror equivariance of the direction output.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta, make_loader
from data.geometry import (COMPONENT_OF_VIEW, N_CELL, build_geometry_table,
                           t_norm_table)
from models.model import EcalTransformer
from utils.angle import angular_error_np, containment_summary
from utils.config import load_config


def ridge_r2(x, y, lam=1.0):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    half = len(x) // 2
    x_train = np.concatenate([x[:half], np.ones((half, 1))], axis=1)
    x_test = np.concatenate([x[half:], np.ones((len(x) - half, 1))], axis=1)
    matrix = x_train.T @ x_train + lam * np.eye(x_train.shape[1])
    weights = np.linalg.solve(matrix, x_train.T @ y[:half])
    prediction = x_test @ weights
    residual = ((prediction - y[half:]) ** 2).sum(axis=0)
    total = ((y[half:] - y[half:].mean(axis=0)) ** 2).sum(axis=0) + 1e-9
    return 1.0 - residual / total


def mirror_batch(batch, view_index, tnorm):
    mirrored = {key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch.items()}
    valid = mirrored["valid"]
    is_view = (mirrored["feats"][:, :, 4 + view_index] == 1) & valid
    layer = mirrored["pos_id"] // N_CELL
    cell = mirrored["pos_id"] % N_CELL
    cell = torch.where(is_view, (N_CELL - 1) - cell, cell)
    mirrored["pos_id"] = layer * N_CELL + cell
    mirrored["feats"][:, :, 1] = torch.where(
        is_view, tnorm[mirrored["pos_id"]], mirrored["feats"][:, :, 1])
    return mirrored


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    meta = load_meta(cfg.paths.cache_dir)
    checkpoint = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")
    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(cfg.device)
    model.set_energy_norm(meta["log_energy_mean"], meta["log_energy_std"])
    model.set_angle_norm(meta["angle_mean"], meta["angle_std"])
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device)["model"])
    model.eval()

    dataset = EcalTokens(
        cfg.paths.cache_dir, args.split, meta, concept_use=concept_use,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None))
    loader = make_loader(dataset, cfg, shuffle=False)
    tnorm = torch.tensor(t_norm_table(build_geometry_table(meta["geometry_data_type"])),
                         device=cfg.device)
    phys, free, concepts = [], [], []
    base_all, truth_all, no_phys_all, no_free_all = [], [], [], []
    x_unmirrored, y_unmirrored = [], []

    for batch in loader:
        gpu = {key: (value if key in ("run", "event") else value.to(cfg.device))
               for key, value in batch.items()}
        base = model(gpu)
        no_phys = model(gpu, phys_scale=0.0)
        no_free = model(gpu, free_scale=0.0)
        valid = gpu["valid"]
        denominator = valid.sum(1, keepdim=True).clamp_min(1)
        pool = lambda h: ((h * valid.unsqueeze(-1)).sum(1) / denominator).float().cpu().numpy()
        phys.append(pool(base["h_phys"]))
        free.append(pool(base["h_free"]))
        concepts.append(gpu["concepts"].cpu().numpy())

        base_slopes = model.predict_angle_slopes(base["angle"].float()).cpu().numpy()
        base_all.append(base_slopes)
        truth_all.append(gpu["angle"].cpu().numpy())
        no_phys_all.append(model.predict_angle_slopes(no_phys["angle"].float()).cpu().numpy())
        no_free_all.append(model.predict_angle_slopes(no_free["angle"].float()).cpu().numpy())

        for view, destination in ((0, y_unmirrored), (1, x_unmirrored)):
            mirrored = mirror_batch(gpu, view, tnorm)
            prediction = model.predict_angle_slopes(model(mirrored)["angle"].float()).cpu().numpy()
            component = COMPONENT_OF_VIEW[view]
            prediction[:, component] *= -1.0  # transform back to original frame
            destination.append(prediction)

    phys = np.concatenate(phys); free = np.concatenate(free)
    concepts = np.concatenate(concepts)
    base = np.concatenate(base_all); truth = np.concatenate(truth_all)
    no_phys = np.concatenate(no_phys_all); no_free = np.concatenate(no_free_all)
    x_back = np.concatenate(x_unmirrored); y_back = np.concatenate(y_unmirrored)

    r2_phys = ridge_r2(phys, concepts)
    r2_free = ridge_r2(free, concepts)
    print("\n=== LINEAR CONCEPT PROBE (R2) ===")
    print(f"  {'concept':<14} {'h_phys':>9} {'h_free':>9}")
    for index, name in enumerate(dataset.concept_names):
        print(f"  {name:<14} {r2_phys[index]:>9.3f} {r2_free[index]:>9.3f}")

    print("\n=== SUBSPACE ABLATION (p68 vs MC truth) ===")
    for name, prediction in (("normal", base), ("kill h_phys", no_phys),
                             ("kill h_free", no_free)):
        summary = containment_summary(angular_error_np(prediction, truth))
        print(f"  {name:<12}: p68={1e3*summary['p68']:.3f} mrad "
              f"median={1e3*summary['median']:.3f} mrad")

    print("\n=== MIRROR EQUIVARIANCE (prediction transformed back) ===")
    for name, prediction in (("X mirror", x_back), ("Y mirror", y_back)):
        summary = containment_summary(angular_error_np(prediction, base))
        print(f"  {name:<8}: p68={1e3*summary['p68']:.4f} mrad "
              f"p90={1e3*summary['p90']:.4f} mrad")


if __name__ == "__main__":
    main()
