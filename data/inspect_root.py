"""Run FIRST on the node. Verify branches, shapes, units before trusting anything.

    python -m data.inspect_root --config config/base.yaml
"""
import argparse
import sys

import numpy as np
import uproot

sys.path.insert(0, ".")
from utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    cfg, _ = load_config(args.config, args.overrides)

    import glob as globmod
    paths = sorted(globmod.glob(cfg.paths.root_file))
    if not paths:
        raise FileNotFoundError(f"no ROOT files match {cfg.paths.root_file!r}")
    print(f"{len(paths)} file(s) match; inspecting the first: {paths[0]}")
    f = uproot.open(paths[0])
    tree = f[cfg.paths.tree]
    print(f"tree={cfg.paths.tree}  n_entries={tree.num_entries} (first file only)")

    needed = [cfg.data.ehit_branch, cfg.data.expehit_branch, cfg.data.target_branch,
              cfg.data.stat_branch, cfg.data.nshwr_branch, cfg.data.ref_branch]
    needed += [c.branch for c in cfg.data.concepts if c.get("branch")]
    have = set(tree.keys())
    print("\n-- branch presence --")
    for b in needed:
        print(f"  {'OK ' if b in have else 'MISSING'}  {b}")

    n = min(5000, tree.num_entries)
    arrs = tree.arrays([cfg.data.ehit_branch, cfg.data.expehit_branch,
                        cfg.data.target_branch], entry_stop=n, library="np")
    ehit = arrs[cfg.data.ehit_branch]
    expe = arrs[cfg.data.expehit_branch]
    mc = arrs[cfg.data.target_branch]

    e0 = np.asarray(ehit[0])
    print(f"\n-- shapes (event 0) --\n  {cfg.data.ehit_branch}: {e0.shape} (expect (18,72) or flat 1296)")

    # Physics check of the axis order: averaged over events, the energy fraction per
    # *layer* (axis 0) must show a longitudinal EM shower profile — rising to a peak
    # at mid-depth then falling. If axis 0 were the cell axis, the profile would be
    # a broad lateral bump centred near index 36 instead, with no rise-fall over 18.
    grids = np.stack([np.asarray(a, dtype=np.float64).reshape(18, 72) for a in ehit[:1000]])
    prof = grids.sum(axis=2).mean(axis=0)
    prof = prof / max(prof.sum(), 1e-9)
    peak = int(np.argmax(prof))
    print("\n-- longitudinal profile check (axis 0 = layer?) --")
    print("  mean energy fraction per layer:", " ".join(f"{p:.3f}" for p in prof))
    print(f"  peak at layer {peak} (expect a smooth rise-then-fall; the peak deepens "
          f"with energy — ~4-8 at tens of GeV, ~9-13 for a TeV-weighted spectrum. "
          f"Flat or edge-peaked would mean the axis order is wrong)")

    ehit_flat = np.concatenate([np.asarray(a).ravel() for a in ehit])
    expe_flat = np.concatenate([np.asarray(a).ravel() for a in expe])
    nz = ehit_flat[ehit_flat > 0]
    print("\n-- kx_ehit (stored, ~MeV) --")
    print(f"  nonzero cells: min={nz.min():.4g} max={nz.max():.4g} median={np.median(nz):.4g}")
    print(f"  cells > 0.1 MeV per event (mean): {(ehit_flat > 0.1).sum() / n:.1f}")
    print(f"  cells > 5.0 MeV per event (mean): {(ehit_flat > 5.0).sum() / n:.1f}")
    print(f"  -> /{cfg.data.cell_scale:.0f} = GeV; sum/event mean = "
          f"{ehit_flat.sum() / n / cfg.data.cell_scale:.3f} GeV")
    print("\n-- kx_expehit (recon target) --")
    nze = expe_flat[expe_flat > 0]
    print(f"  nonzero: min={nze.min():.4g} max={nze.max():.4g}  frac>0 = {(expe_flat>0).mean():.3f}")
    print("\n-- mcEne (GeV) --")
    print(f"  min={mc.min():.3g} max={mc.max():.3g} mean={mc.mean():.3g}  frac<=0 = {(mc<=0).mean():.3f}")
    print("\nIf shapes/units look right, run: python -m data.preprocess --config", args.config)


if __name__ == "__main__":
    main()
