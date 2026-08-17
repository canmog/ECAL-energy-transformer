"""ROOT -> task-capable token cache with explicit schema metadata.

    python -m data.preprocess --config config/base.yaml

Output (in cfg.paths.cache_dir): train.npz / val.npz / test.npz + meta.json.
Each split stores a CSR-style ragged token layout:
    off            int64 [N+1]   event i tokens = [off[i] : off[i+1]]
    tok_layer      int16 [T]     ilayer 0..17
    tok_cell       int16 [T]     icell  0..71
    tok_ehit       f32   [T]     measured cell energy (stored ~MeV)  -> INPUT
    tok_expe       f32   [T]     3D-fit expected cell energy (~MeV)  -> recon TARGET
    energy         f32   [N]     mcEne (GeV)                          -> energy TARGET
    angle          f32   [N,2]   MC (dx/dz,dy/dz), when configured   -> angle TARGET
    fit_angle      f32   [N,2]   3D-fit slopes, when configured      -> reference only
    run,event      u32   [N]     stable event identity, when configured
    concepts       f32   [N, C]  raw 3D-fit concept values            -> bottleneck TARGET
Geometry is NOT baked in here — the dataset derives it from data.geometry so a
single edit there re-propagates everywhere.
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, ".")
from utils.config import load_config  # noqa: E402
from data.geometry import (COMPONENT_VIEWS, N_LAYER, N_CELL,
                           build_geometry_table)  # noqa: E402
from data.schema import CACHE_SCHEMA_VERSION, normalize_task_mode  # noqa: E402
from data.root_schema import (required_root_branches,
                              require_root_branches)  # noqa: E402


def _as_3d(arr):
    """Coerce a uproot [18][72] branch chunk to (n, 18, 72) float32.

    Axis order is verifiable, not assumed: the branch is declared [18][72] =
    [layer][cell] (var.md / geo.md: nLayer=18, nCell=72), and since 18 != 72 a
    swapped layout cannot pass the shape check below silently.
    """
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.stack([np.asarray(x, dtype=np.float32).reshape(N_LAYER, N_CELL) for x in a])
    a = a.astype(np.float32)
    if a.ndim == 2:                                     # flat [1296] per event
        a = a.reshape(a.shape[0], N_LAYER, N_CELL)
    if a.shape[1:] != (N_LAYER, N_CELL):
        raise ValueError(f"cell branch has per-event shape {a.shape[1:]}, expected "
                         f"({N_LAYER}, {N_CELL}) = (layer, cell); check the ROOT layout")
    return a


def _scalar(arr, idx=None):
    a = np.asarray(arr)
    if idx is not None:                      # branch is [3]; take primary shower
        a = np.stack([np.asarray(x).ravel()[idx] for x in a]) if a.dtype == object else a[:, idx]
    return a.astype(np.float32)


def _shape_concepts(src, table):
    """Energy-weighted shower-shape moments from a (n,18,72) cell-energy map.

    Returns {name: (n,)} for the derived bottleneck concepts:
        tmax       longitudinal centroid   <layer>_E               (~X0, layer idx)
        long_width longitudinal RMS         sqrt(<(layer-tmax)^2>_E)
        lat_width  transverse RMS           sqrt(<(t - t_view)^2>_E)  (cm)
    Weights are clipped >=0 (the fit map is non-negative; this also guards empty
    events). Absolute units are immaterial: concepts are standardised (mean/std)
    before the model, so layer-index vs X0 vs cm only rescales them. The transverse
    mean is taken PER VIEW (X-layers and Y-layers measure different projections),
    then both views feed one combined lateral RMS.
    """
    w = np.clip(src, 0.0, None).astype(np.float64)            # (n,18,72)
    inv = 1.0 / np.clip(w.sum(axis=(1, 2)), 1e-9, None)       # (n,)
    depth = table["depth"][None]                              # (1,18,72) layer idx
    t_cm = table["t_cm"][None]                                # (1,18,72) cm
    view = table["view"]                                      # (18,72) 0=X 1=Y

    tmax = (w * depth).sum(axis=(1, 2)) * inv                 # (n,)
    long_var = (w * (depth - tmax[:, None, None]) ** 2).sum(axis=(1, 2)) * inv
    long_width = np.sqrt(np.clip(long_var, 0.0, None))

    def _centroid(vw):                                        # per-view E-weighted <t>
        wv = w * (view == vw)[None]
        return (wv * t_cm).sum(axis=(1, 2)) / np.clip(wv.sum(axis=(1, 2)), 1e-9, None)
    tbar = np.where(view[None] == 0,                          # (n,18,72)
                    _centroid(0)[:, None, None], _centroid(1)[:, None, None])
    lat_var = (w * (t_cm - tbar) ** 2).sum(axis=(1, 2)) * inv
    lat_width = np.sqrt(np.clip(lat_var, 0.0, None))

    return {"tmax": tmax.astype(np.float32),
            "lat_width": lat_width.astype(np.float32),
            "long_width": long_width.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    d = cfg.data
    task_cfg = cfg.get("task", None)
    task_mode = normalize_task_mode(task_cfg.get("mode", "energy")
                                    if task_cfg is not None else "energy")
    angle_branches = list(d.get("angle_branches", []) or [])
    fit_angle_branches = list(d.get("fit_angle_branches", []) or [])
    has_angle = bool(angle_branches)
    has_fit_angle = bool(fit_angle_branches)
    has_identity = bool(d.get("run_branch", None) and d.get("event_branch", None))
    if task_mode in ("angle", "joint") and len(angle_branches) != 2:
        raise ValueError(
            f"task.mode={task_mode!r} requires data.angle_branches=[mcKX,mcKY]")
    if has_angle and len(angle_branches) != 2:
        raise ValueError("data.angle_branches must contain exactly two slope branches")
    if has_fit_angle and len(fit_angle_branches) != 2:
        raise ValueError("data.fit_angle_branches must contain exactly two slope branches")
    table = build_geometry_table(cfg.geometry.data_type)   # for derived shape concepts

    concept_names = [c.name for c in d.concepts]
    sel = d.selection
    rng = np.random.default_rng(cfg.seed)

    read = [d.ehit_branch, d.expehit_branch, d.target_branch,
            d.stat_branch, d.nshwr_branch, d.ref_branch]
    read += angle_branches + fit_angle_branches
    if has_identity:
        read += [d.run_branch, d.event_branch]
    read += [c.branch for c in d.concepts if not c.get("derive")]
    read = sorted(set(read))

    # paths.root_file may be a single file or a glob over the merged dataset
    paths = sorted(globmod.glob(cfg.paths.root_file))
    if not paths:
        raise FileNotFoundError(f"no ROOT files match {cfg.paths.root_file!r}")
    require_root_branches(
        paths, cfg.paths.tree, required_root_branches(cfg),
        operation=f"{task_mode} cache preprocessing")
    os.makedirs(cfg.paths.cache_dir, exist_ok=True)
    print(f"reading {len(paths)} input file(s)")

    tok_layer, tok_cell, tok_ehit, tok_expe = [], [], [], []
    counts, energy, concepts = [], [], []
    angles, fit_angles, runs, events = [], [], [], []
    n_seen = 0
    stop = None if d.max_events in (None, "null") else int(d.max_events)

    for chunk in uproot.iterate([f"{p}:{cfg.paths.tree}" for p in paths], read,
                                step_size=20000, library="np"):
        if stop is not None and n_seen >= stop:
            break
        ehit = _as_3d(chunk[d.ehit_branch])
        expe = _as_3d(chunk[d.expehit_branch])
        mc = np.asarray(chunk[d.target_branch]).astype(np.float32)
        angle = (np.stack([np.asarray(chunk[name]).ravel()
                           for name in angle_branches], axis=1).astype(np.float32)
                 if has_angle else None)
        fit_angle = (np.stack([_scalar(chunk[name], 0).ravel()
                               for name in fit_angle_branches], axis=1).astype(np.float32)
                     if has_fit_angle else None)
        run = (np.asarray(chunk[d.run_branch]).ravel().astype(np.uint32)
               if has_identity else None)
        event = (np.asarray(chunk[d.event_branch]).ravel().astype(np.uint32)
                 if has_identity else None)
        n = ehit.shape[0]

        keep = np.ones(n, dtype=bool)
        if sel.require_truth:
            keep &= mc > 0
        if sel.get("require_angle_truth", False):
            if angle is None:
                raise ValueError(
                    "data.selection.require_angle_truth=true requires "
                    "data.angle_branches")
            keep &= np.isfinite(angle).all(axis=1)
        if sel.select_contained:
            stat0 = _scalar(chunk[d.stat_branch], 0).astype(np.int64)
            keep &= (stat0 & 7) == 7
        if sel.select_single_shower:
            keep &= np.asarray(chunk[d.nshwr_branch]).ravel().astype(np.int64) == 1
        if sel.require_positive_ref:
            keep &= np.asarray(chunk[d.ref_branch]).ravel().astype(np.float32) > 0
        # sel.pid_electron_only: RESERVED — no PID branch yet, no-op.

        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            n_seen += n
            continue
        ehit, expe, mc = ehit[idx], expe[idx], mc[idx]
        if angle is not None:
            angle = angle[idx]
        if fit_angle is not None:
            fit_angle = fit_angle[idx]
        if run is not None:
            run, event = run[idx], event[idx]

        # concept matrix for kept events. Branch concepts read straight from the
        # ROOT chunk; derived shape concepts are energy-weighted moments of the
        # already-idx-filtered cell map -> compute each needed source map once.
        shape = {s: _shape_concepts(expe if s == "expe" else ehit, table)
                 for s in {c.get("source", "expe") for c in d.concepts if c.get("derive")}}
        cvals = []
        for c in d.concepts:
            if c.get("derive"):
                cvals.append(shape[c.get("source", "expe")][c.derive])
            else:
                cvals.append(_scalar(chunk[c.branch], c.index)[idx])
        cmat = np.stack(cvals, axis=1).astype(np.float32)

        # tokenise: cells above threshold (stored ~MeV)
        mask = ehit > float(d.threshold_mev)
        ev, la, ce = np.nonzero(mask)
        tok_layer.append(la.astype(np.int16))
        tok_cell.append(ce.astype(np.int16))
        tok_ehit.append(ehit[ev, la, ce].astype(np.float32))
        tok_expe.append(expe[ev, la, ce].astype(np.float32))
        counts.append(np.bincount(ev, minlength=ehit.shape[0]).astype(np.int64))
        energy.append(mc)
        concepts.append(cmat)
        if angle is not None:
            angles.append(angle)
        if fit_angle is not None:
            fit_angles.append(fit_angle)
        if run is not None:
            runs.append(run)
            events.append(event)
        n_seen += n
        print(f"  processed {n_seen} events, kept {sum(len(e) for e in energy)}", end="\r")

    print()
    tok_layer = np.concatenate(tok_layer)
    tok_cell = np.concatenate(tok_cell)
    tok_ehit = np.concatenate(tok_ehit)
    tok_expe = np.concatenate(tok_expe)
    counts = np.concatenate(counts)
    energy = np.concatenate(energy)
    concepts = np.concatenate(concepts, axis=0)
    angles = np.concatenate(angles, axis=0) if angles else None
    fit_angles = np.concatenate(fit_angles, axis=0) if fit_angles else None
    runs = np.concatenate(runs) if runs else None
    events = np.concatenate(events) if events else None
    N = energy.shape[0]
    print(f"kept {N} events, {tok_layer.shape[0]} tokens "
          f"(mean {tok_layer.shape[0]/max(N,1):.1f}/event)")

    # Drop events with 0 tokens above threshold: energy is unreconstructable from
    # no hits, and an empty token set => fully-masked attention => NaN. The token
    # arrays already contain nothing for these events, so filtering the per-event
    # arrays + recomputing offsets keeps everything aligned.
    nonempty = counts > 0
    if not nonempty.all():
        dropped = int((~nonempty).sum())
        counts = counts[nonempty]
        energy = energy[nonempty]
        concepts = concepts[nonempty]
        if angles is not None:
            angles = angles[nonempty]
        if fit_angles is not None:
            fit_angles = fit_angles[nonempty]
        if runs is not None:
            runs, events = runs[nonempty], events[nonempty]
        N = energy.shape[0]
        print(f"dropped {dropped} zero-token events -> {N} remain")

    # split
    perm = rng.permutation(N)
    n_test = int(d.test_fraction * N)
    n_val = int(d.val_fraction * N)
    splits = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val],
              "train": perm[n_test + n_val:]}

    off_all = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=off_all[1:])

    def gather(event_ids):
        order = np.sort(event_ids)
        offs = np.zeros(len(order) + 1, dtype=np.int64)
        li, ci, eh, ex = [], [], [], []
        for j, e in enumerate(order):
            s, t = off_all[e], off_all[e + 1]
            li.append(tok_layer[s:t]); ci.append(tok_cell[s:t])
            eh.append(tok_ehit[s:t]); ex.append(tok_expe[s:t])
            offs[j + 1] = offs[j] + (t - s)
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        pack = dict(off=offs, tok_layer=cat(li).astype(np.int16),
                    tok_cell=cat(ci).astype(np.int16),
                    tok_ehit=cat(eh).astype(np.float32),
                    tok_expe=cat(ex).astype(np.float32),
                    energy=energy[order].astype(np.float32),
                    concepts=concepts[order].astype(np.float32))
        if angles is not None:
            pack["angle"] = angles[order].astype(np.float32)
        if fit_angles is not None:
            pack["fit_angle"] = fit_angles[order].astype(np.float32)
        if runs is not None:
            pack["run"] = runs[order].astype(np.uint32)
            pack["event"] = events[order].astype(np.uint32)
        return pack

    packs = {k: gather(v) for k, v in splits.items()}
    for k, p in packs.items():
        np.savez(os.path.join(cfg.paths.cache_dir, f"{k}.npz"), **p)
        print(f"  {k}: {len(p['energy'])} events -> {k}.npz")

    # meta: normalisation from TRAIN only
    tr = packs["train"]
    loge = np.log(np.clip(tr["energy"], 1e-3, None))
    cmean = tr["concepts"].mean(axis=0)
    cstd = tr["concepts"].std(axis=0) + 1e-6
    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "storage_layout": "packed_npz",
        "fields": sorted(tr),
        "task_mode_built_for": task_mode,
        "component_views": list(COMPONENT_VIEWS),
        "n_concepts": len(concept_names),
        "concept_names": concept_names,
        "concept_reflect_x": [int(c.reflect_x) for c in d.concepts],
        "concept_reflect_y": [int(c.reflect_y) for c in d.concepts],
        "concept_coord": [c.get("coord", None) for c in d.concepts],
        "concept_mean": cmean.tolist(),
        "concept_std": cstd.tolist(),
        "log_energy_mean": float(loge.mean()),
        "log_energy_std": float(loge.std() + 1e-6),
        "threshold_mev": float(d.threshold_mev),
        "cell_scale": float(d.cell_scale),
        "geometry_data_type": cfg.geometry.data_type,
        "selection": sel.to_dict(),
    }
    if "angle" in tr:
        amean = tr["angle"].mean(axis=0)
        astd = tr["angle"].std(axis=0) + 1e-6
        meta.update({
            "angle_names": list(d.get("angle_names", ["mc_kx", "mc_ky"])),
            "angle_mean": amean.tolist(),
            "angle_std": astd.tolist(),
            "angle_reflect_x": [-1, 1],
            "angle_reflect_y": [1, -1],
        })
    with open(os.path.join(cfg.paths.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote meta.json:", {k: meta[k] for k in
          ["n_concepts", "log_energy_mean", "log_energy_std", "threshold_mev"]})


if __name__ == "__main__":
    main()
