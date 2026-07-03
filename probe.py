"""Did the network learn PHYSICS, or a spurious correlation? Three falsifiable tests.

    python probe.py --config config/base.yaml

1. LINEAR PROBE   — fit a *linear* map from the frozen physics sub-space h_phys to
   each 3D-fit concept; report R^2. Contrast with the free sub-space h_free. High
   R^2 on h_phys and low on h_free = the physics is concentrated and disentangled.
   BOTH sub-spaces are masked-MEAN pooled: pooling h_phys with the trained concept
   attention pool would tilt the comparison in h_phys's favour by construction.
2. SUBSPACE ABLATION (causal) — zero h_phys vs h_free at the tap and measure the
   shift in the energy output. Larger shift from killing h_phys = the energy head
   genuinely routes through the physics representation.
3. MIRROR STRESS TEST — flip every X-view token to its mirror cell and look up the
   flipped cell's TRUE coordinate from the geometry table (exact array mirror; a
   naive -t_norm would be off by ~2 mm of alignment offsets). Energy must be
   invariant; reports RMS dE/E.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from utils.config import load_config
from data.dataset import EcalTokens, load_meta, make_loader
from data.geometry import N_CELL, build_geometry_table, t_norm_table
from models.model import EcalTransformer


def amp_dtype_of(cfg):
    """Honor train.amp_dtype: bf16 -> autocast bf16; fp32/none -> no autocast.
    (Mirrors train.py so the probe runs at the SAME precision the model trained in.)"""
    name = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "none"):
        return None
    raise ValueError(f"train.amp_dtype={name!r} unsupported (use bf16 | fp32 | none)")


def ridge_r2(X, Y, lam=1.0):
    """Closed-form ridge on a 50/50 split; return per-target R^2 on the held-out half.

    Shuffle before splitting: the token-budget loader yields events sorted by
    length (~energy), so a positional split would put small showers in one half."""
    rng = np.random.default_rng(0)
    p = rng.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    n = X.shape[0]; h = n // 2
    Xtr, Ytr, Xte, Yte = X[:h], Y[:h], X[h:], Y[h:]
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], 1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], 1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((pred - Yte) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    device = cfg.device
    meta = load_meta(cfg.paths.cache_dir)
    import os
    ckpt = args.ckpt or os.path.join(cfg.paths.out_dir, "best.pt")

    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    amp_dtype = amp_dtype_of(cfg)
    probe_ds = EcalTokens(cfg.paths.cache_dir, args.split, meta, concept_use=concept_use)
    loader = make_loader(probe_ds, cfg, shuffle=False)
    tnorm_lut = torch.tensor(t_norm_table(build_geometry_table(meta["geometry_data_type"])),
                             device=device)

    phys, free, ctrue = [], [], []
    e_base, e_no_phys, e_no_free, e_mirror = [], [], [], []
    e_true_l, e_physonly, e_freeonly = [], [], []
    has_dual = False
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid = batch["valid"]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
            o_np = model(batch, phys_scale=0.0)
            o_nf = model(batch, free_scale=0.0)
        denom = valid.sum(1, keepdim=True).clamp_min(1)
        mean_pool = lambda h: ((h * valid.unsqueeze(-1)).sum(1) / denom).float().cpu().numpy()
        phys.append(mean_pool(out["h_phys"]))
        free.append(mean_pool(out["h_free"]))
        ctrue.append(batch["concepts"].cpu().numpy())
        e_base.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
        e_no_phys.append(model.predict_energy_gev(o_np["energy"].float()).cpu().numpy())
        e_no_free.append(model.predict_energy_gev(o_nf["energy"].float()).cpu().numpy())
        e_true_l.append(batch["energy"].cpu().numpy())
        if "e_phys" in out:                    # dual head: physics-only / free-only energy
            has_dual = True
            e_physonly.append(model.predict_energy_gev(out["e_phys"].float()).cpu().numpy())
            e_freeonly.append(model.predict_energy_gev(out["e_free"].float()).cpu().numpy())

        # mirror: remap X-view cells to 71-cell, then take the flipped cell's TRUE
        # t from the geometry table (exact array mirror, alignment offsets included)
        mb = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        is_x = (mb["feats"][:, :, 4] == 1) & valid
        layer = mb["pos_id"] // N_CELL; cell = mb["pos_id"] % N_CELL
        cell = torch.where(is_x, (N_CELL - 1) - cell, cell)
        mb["pos_id"] = layer * N_CELL + cell
        mb["feats"][:, :, 1] = torch.where(is_x, tnorm_lut[mb["pos_id"]], mb["feats"][:, :, 1])
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            om = model(mb)
        e_mirror.append(model.predict_energy_gev(om["energy"].float()).cpu().numpy())

    phys = np.concatenate(phys); free = np.concatenate(free); ctrue = np.concatenate(ctrue)
    e_base = np.concatenate(e_base); e_true = np.concatenate(e_true_l)
    rel = lambda a: np.sqrt(np.mean(((np.concatenate(a) - e_base) / np.clip(e_base, 1e-3, None)) ** 2))
    sigE = lambda e: float(np.std((e - e_true) / np.clip(e_true, 1e-3, None)))

    print("\n=== 1. LINEAR PROBE  (R^2: h_phys vs h_free) ===")
    r2p = ridge_r2(phys, ctrue); r2f = ridge_r2(free, ctrue)
    print(f"  {'concept':<12} {'h_phys':>8} {'h_free':>8}")
    for j, name in enumerate(probe_ds.concept_names):
        print(f"  {name:<12} {r2p[j]:>8.3f} {r2f[j]:>8.3f}")
    print(f"  {'MEAN':<12} {r2p.mean():>8.3f} {r2f.mean():>8.3f}   "
          "(want h_phys >> h_free: physics is concentrated & linearly decodable)")

    print("\n=== 2. SUBSPACE ABLATION  (RMS dE/E vs normal energy) ===")
    print(f"  kill h_phys: {rel(e_no_phys)*100:6.2f}%     kill h_free: {rel(e_no_free)*100:6.2f}%")
    print("  (larger shift from killing h_phys => energy routes through the physics sub-space)")

    if has_dual:
        ep = np.concatenate(e_physonly); ef = np.concatenate(e_freeonly)
        add = float(np.sqrt(np.mean(((e_base - ep) / np.clip(e_base, 1e-3, None)) ** 2)))
        print("\n=== 2b. DUAL-HEAD DECOMPOSITION  (sigma/E vs TRUE energy) ===")
        print(f"  full (e_phys+e_free): {sigE(e_base)*100:6.2f}%   physics-only e_phys: "
              f"{sigE(ep)*100:6.2f}%   free-only e_free: {sigE(ef)*100:6.2f}%")
        print(f"  free-head contribution (RMS dE/E vs physics-only): {add*100:6.2f}%")
        print("  (physics-only should already be a sensible energy; free head a modest correction)")

    print("\n=== 3. MIRROR STRESS TEST  (energy must be x-reflection invariant) ===")
    print(f"  RMS dE/E under x-mirror: {rel(e_mirror)*100:6.3f}%   (small = symmetry learned)")


if __name__ == "__main__":
    main()
