"""Validate the augmented full ROOT schema, direction convention, shapes, and units."""
import argparse
import glob
import sys

import numpy as np
import uproot

sys.path.insert(0, ".")
from data.geometry import N_CELL, N_LAYER
from utils.angle import angular_error_np
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    d = cfg.data
    paths = sorted(glob.glob(cfg.paths.root_file))
    if not paths:
        raise FileNotFoundError(f"no ROOT files match {cfg.paths.root_file!r}")

    needed = [d.ehit_branch, d.expehit_branch, d.target_branch,
              d.stat_branch, d.nshwr_branch, d.ref_branch,
              d.run_branch, d.event_branch,
              *list(d.angle_branches), *list(d.fit_angle_branches),
              "mcTheta", "mcPhi", "mcDirX", "mcDirY", "mcDirZ"]
    needed += [c.branch for c in d.concepts if c.get("branch")]
    needed = sorted(set(needed))

    total = 0
    for path in paths:
        tree = uproot.open(f"{path}:{cfg.paths.tree}")
        missing = sorted(set(needed) - set(tree.keys()))
        if missing:
            raise RuntimeError(f"{path}: missing branches {missing}")
        total += tree.num_entries
    print(f"schema OK in {len(paths)} files; total entries={total}")

    path = paths[0]
    tree = uproot.open(f"{path}:{cfg.paths.tree}")
    n = min(5000, tree.num_entries)
    read = [d.ehit_branch, d.expehit_branch, d.target_branch,
            *list(d.angle_branches), *list(d.fit_angle_branches),
            d.stat_branch, d.nshwr_branch,
            "mcTheta", "mcPhi", "mcDirX", "mcDirY", "mcDirZ"]
    arrays = tree.arrays(read, entry_stop=n, library="np")

    ehit = np.asarray(arrays[d.ehit_branch], dtype=np.float32).reshape(n, N_LAYER, N_CELL)
    expe = np.asarray(arrays[d.expehit_branch], dtype=np.float32).reshape(n, N_LAYER, N_CELL)
    energy = np.asarray(arrays[d.target_branch]).ravel()
    angle = np.stack([np.asarray(arrays[b]).ravel() for b in d.angle_branches], axis=1)
    fit = np.stack([np.asarray(arrays[b])[:, 0] for b in d.fit_angle_branches], axis=1)
    stat0 = np.asarray(arrays[d.stat_branch])[:, 0].astype(np.int64)
    nshower = np.asarray(arrays[d.nshwr_branch]).ravel().astype(np.int64)
    theta = np.asarray(arrays["mcTheta"]).ravel()
    phi = np.asarray(arrays["mcPhi"]).ravel()
    direction = np.stack([np.asarray(arrays[b]).ravel()
                          for b in ("mcDirX", "mcDirY", "mcDirZ")], axis=1)

    expected_direction = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)], axis=1)
    expected_slopes = expected_direction[:, :2] / expected_direction[:, 2:3]
    if not np.allclose(direction, expected_direction, rtol=0, atol=5e-6):
        raise RuntimeError("mcDirX/Y/Z do not match mcTheta/mcPhi")
    if not np.allclose(angle, expected_slopes, rtol=0, atol=5e-6):
        raise RuntimeError("mcKX/KY do not match mcTheta/mcPhi")
    norms = np.linalg.norm(direction, axis=1)

    profile = ehit[:1000].sum(axis=2).mean(axis=0)
    profile /= max(profile.sum(), 1e-12)
    nonzero = ehit[ehit > 0]
    selected = (energy > 0) & ((stat0 & 7) == 7) & (nshower == 1)
    fit_error = angular_error_np(fit[selected], angle[selected])
    finite_fit = fit_error[np.isfinite(fit_error)]
    print(f"first file: {path}")
    print(f"  entries={tree.num_entries} branches={len(tree.keys())}")
    print(f"  cell shape={ehit.shape[1:]} expected={(N_LAYER, N_CELL)}")
    print(f"  longitudinal peak layer={int(np.argmax(profile))}")
    print(f"  ehit nonzero min/median/max={nonzero.min():.5g}/"
          f"{np.median(nonzero):.5g}/{nonzero.max():.5g} stored MeV")
    print(f"  expe finite={np.isfinite(expe).all()} energy range={energy.min():.4g}..{energy.max():.4g} GeV")
    print(f"  direction norm max|norm-1|={np.max(np.abs(norms-1)):.3g}")
    print(f"  mcKX range={angle[:,0].min():.4f}..{angle[:,0].max():.4f} "
          f"mcKY range={angle[:,1].min():.4f}..{angle[:,1].max():.4f}")
    print(f"  selected sample={int(selected.sum())}/{n} "
          f"3D-fit p68={1e3*np.quantile(finite_fit, .68):.3f} mrad")
    print("ROOT inspection passed")


if __name__ == "__main__":
    main()
