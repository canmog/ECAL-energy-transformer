"""Build a low-memory ROOT -> token cache for the full, uncut event sample.

The legacy preprocessor accumulates all token arrays in RAM before splitting
and writing NPZ files.  That is convenient for the selected 3.38M-event sample
but cannot safely scale to all 17.7M events.  This builder makes two streaming
passes instead:

1. count above-threshold cells and assign retained events to train/val/test;
2. write each split directly into memory-mapped NPY arrays.

The stored physics fields are identical to ``data.preprocess``.  Two event-level
quality fields are additionally retained so uncut evaluation can report the old
selection domain separately:

    fit_status  int32 [N]  primary ``kx_ShwrStat[0]``
    n_shower    int16 [N]  ``kx_N_Shwr``
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, ".")
from data.geometry import (COMPONENT_VIEWS, N_CELL, N_LAYER,
                           build_geometry_table)  # noqa: E402
from data.preprocess import _as_3d, _scalar, _shape_concepts  # noqa: E402
from data.root_schema import (required_root_branches,
                              require_root_branches)  # noqa: E402
from data.schema import CACHE_SCHEMA_VERSION, normalize_task_mode  # noqa: E402
from utils.config import load_config  # noqa: E402


SPLITS = ("train", "val", "test")
DROP_ID = np.uint8(255)


def _entry_count(paths, tree_name, limit):
    total = 0
    for path in paths:
        with uproot.open(f"{path}:{tree_name}") as tree:
            total += int(tree.num_entries)
        if limit is not None and total >= limit:
            return int(limit)
    return total


def _chunks(paths, tree_name, branches, limit, step_size=20000):
    """Yield chunks, truncating the last one when data.max_events is active."""
    seen = 0
    specs = [f"{path}:{tree_name}" for path in paths]
    for chunk in uproot.iterate(specs, branches, step_size=step_size, library="np"):
        if limit is not None:
            remaining = limit - seen
            if remaining <= 0:
                break
            first = next(iter(chunk.values()))
            if len(first) > remaining:
                chunk = {name: values[:remaining] for name, values in chunk.items()}
        n = len(next(iter(chunk.values())))
        seen += n
        yield chunk


def _selection_mask(chunk, d, sel):
    n = len(next(iter(chunk.values())))
    keep = np.ones(n, dtype=bool)
    stages = {"input": n}
    if sel.require_truth:
        keep &= np.asarray(chunk[d.target_branch]).ravel().astype(np.float32) > 0
        stages["require_truth"] = int(keep.sum())
    if sel.get("require_angle_truth", True):
        angle = np.stack(
            [np.asarray(chunk[name]).ravel() for name in d.angle_branches], axis=1)
        keep &= np.isfinite(angle).all(axis=1)
        stages["require_angle_truth"] = int(keep.sum())
    if sel.select_contained:
        status = _scalar(chunk[d.stat_branch], 0).astype(np.int64)
        keep &= (status & 7) == 7
        stages["select_contained"] = int(keep.sum())
    if sel.select_single_shower:
        nshower = np.asarray(chunk[d.nshwr_branch]).ravel().astype(np.int64)
        keep &= nshower == 1
        stages["select_single_shower"] = int(keep.sum())
    if sel.require_positive_ref:
        ref = np.asarray(chunk[d.ref_branch]).ravel().astype(np.float32)
        keep &= ref > 0
        stages["require_positive_ref"] = int(keep.sum())
    return keep, stages


def _open_array(directory, name, dtype, shape):
    return np.lib.format.open_memmap(
        os.path.join(directory, f"{name}.npy"), mode="w+", dtype=dtype, shape=shape)


def _finite_moments(array, transform=None, chunk_size=500000):
    """Column-wise finite mean/std without loading a multi-GB array."""
    width = 1 if array.ndim == 1 else array.shape[1]
    total = np.zeros(width, dtype=np.float64)
    total2 = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.int64)
    for start in range(0, len(array), chunk_size):
        block = np.asarray(array[start:start + chunk_size], dtype=np.float64)
        if transform is not None:
            block = transform(block)
        if block.ndim == 1:
            block = block[:, None]
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0.0)
        total += safe.sum(axis=0)
        total2 += (safe * safe).sum(axis=0)
        count += finite.sum(axis=0)
    if np.any(count == 0):
        raise RuntimeError("normalization field has a column with no finite values")
    mean = total / count
    variance = np.maximum(total2 / count - mean * mean, 0.0)
    return mean, np.sqrt(variance) + 1e-6, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    d, sel = cfg.data, cfg.data.selection
    task_cfg = cfg.get("task", None)
    task_mode = normalize_task_mode(
        task_cfg.get("mode", "energy") if task_cfg else "energy")
    if task_mode not in ("angle", "joint"):
        raise ValueError(
            "data.preprocess_memmap stores the full direction-analysis cache and "
            "requires task.mode angle or joint")
    if len(d.get("angle_branches", []) or []) != 2:
        raise ValueError("memmap preprocessing requires two data.angle_branches")
    if len(d.get("fit_angle_branches", []) or []) != 2:
        raise ValueError("memmap preprocessing requires two data.fit_angle_branches")
    if not d.get("run_branch", None) or not d.get("event_branch", None):
        raise ValueError("memmap preprocessing requires data.run_branch and event_branch")
    cache_dir = cfg.paths.cache_dir
    if os.path.exists(os.path.join(cache_dir, "meta.json")):
        raise FileExistsError(
            f"refusing to overwrite completed or partial cache {cache_dir}")

    paths = sorted(globmod.glob(cfg.paths.root_file))
    if not paths:
        raise FileNotFoundError(f"no ROOT files match {cfg.paths.root_file!r}")
    require_root_branches(
        paths, cfg.paths.tree, required_root_branches(cfg),
        operation=f"{task_mode} memmap cache preprocessing")
    os.makedirs(cache_dir, exist_ok=False)
    limit = None if d.max_events in (None, "null") else int(d.max_events)
    n_raw = _entry_count(paths, cfg.paths.tree, limit)
    threshold = float(d.threshold_mev)
    print(f"MEMMAP_CACHE_START files={len(paths)} raw_events={n_raw} "
          f"threshold_mev={threshold} output={cache_dir}", flush=True)

    # Pass 1: only fields that can affect event retention plus measured cells.
    first_read = {d.ehit_branch}
    if sel.require_truth:
        first_read.add(d.target_branch)
    if sel.get("require_angle_truth", True):
        first_read.update(d.angle_branches)
    if sel.select_contained:
        first_read.add(d.stat_branch)
    if sel.select_single_shower:
        first_read.add(d.nshwr_branch)
    if sel.require_positive_ref:
        first_read.add(d.ref_branch)

    count_path = os.path.join(cache_dir, "_build_counts.npy")
    counts = np.lib.format.open_memmap(
        count_path, mode="w+", dtype=np.uint16, shape=(n_raw,))
    cursor = 0
    cutflow = {"input": 0}
    for chunk in _chunks(paths, cfg.paths.tree, sorted(first_read), limit):
        ehit = _as_3d(chunk[d.ehit_branch])
        keep, stages = _selection_mask(chunk, d, sel)
        n = len(keep)
        token_count = np.count_nonzero(ehit > threshold, axis=(1, 2)).astype(np.uint16)
        token_count[~keep] = 0
        counts[cursor:cursor + n] = token_count
        cursor += n
        for name, value in stages.items():
            cutflow[name] = cutflow.get(name, 0) + int(value)
        if cursor % 1000000 < n:
            print(f"PASS1 events={cursor}/{n_raw}", flush=True)
    if cursor != n_raw:
        raise RuntimeError(f"pass 1 saw {cursor} events, expected {n_raw}")
    counts.flush()

    n_kept = int(np.count_nonzero(counts))
    n_zero_or_cut = n_raw - n_kept
    kept_raw = np.flatnonzero(counts).astype(np.int64, copy=False)
    membership = np.full(n_kept, DROP_ID, dtype=np.uint8)
    rng = np.random.default_rng(cfg.seed)
    permutation = rng.permutation(n_kept)
    n_test = int(float(d.test_fraction) * n_kept)
    n_val = int(float(d.val_fraction) * n_kept)
    membership[permutation[:n_test]] = 2
    membership[permutation[n_test:n_test + n_val]] = 1
    membership[permutation[n_test + n_val:]] = 0
    del permutation

    split_path = os.path.join(cache_dir, "_build_split.npy")
    raw_split = np.lib.format.open_memmap(
        split_path, mode="w+", dtype=np.uint8, shape=(n_raw,))
    raw_split[:] = DROP_ID
    raw_split[kept_raw] = membership
    raw_split.flush()
    del kept_raw, membership

    n_concepts = len(d.concepts)
    writers = {}
    split_sizes = {}
    for split_id, split in enumerate(SPLITS):
        mask = raw_split == split_id
        n_event = int(np.count_nonzero(mask))
        n_token = int(np.asarray(counts[mask], dtype=np.int64).sum())
        split_sizes[split] = {"events": n_event, "tokens": n_token}
        directory = os.path.join(cache_dir, split)
        os.makedirs(directory)
        arrays = {
            "off": _open_array(directory, "off", np.int64, (n_event + 1,)),
            "tok_layer": _open_array(directory, "tok_layer", np.int16, (n_token,)),
            "tok_cell": _open_array(directory, "tok_cell", np.int16, (n_token,)),
            "tok_ehit": _open_array(directory, "tok_ehit", np.float32, (n_token,)),
            "tok_expe": _open_array(directory, "tok_expe", np.float32, (n_token,)),
            "energy": _open_array(directory, "energy", np.float32, (n_event,)),
            "angle": _open_array(directory, "angle", np.float32, (n_event, 2)),
            "fit_angle": _open_array(directory, "fit_angle", np.float32, (n_event, 2)),
            "concepts": _open_array(
                directory, "concepts", np.float32, (n_event, n_concepts)),
            "run": _open_array(directory, "run", np.uint32, (n_event,)),
            "event": _open_array(directory, "event", np.uint32, (n_event,)),
            "fit_status": _open_array(
                directory, "fit_status", np.int32, (n_event,)),
            "n_shower": _open_array(directory, "n_shower", np.int16, (n_event,)),
        }
        arrays["off"][0] = 0
        np.cumsum(np.asarray(counts[mask], dtype=np.int64), out=arrays["off"][1:])
        writers[split] = arrays
        print(f"ALLOCATE {split} events={n_event} tokens={n_token}", flush=True)

    # Pass 2: materialize all model targets and token fields directly to disk.
    table = build_geometry_table(cfg.geometry.data_type)
    second_read = {
        d.ehit_branch, d.expehit_branch, d.target_branch, d.stat_branch,
        d.nshwr_branch, d.ref_branch, d.run_branch, d.event_branch,
        *list(d.angle_branches), *list(d.fit_angle_branches),
    }
    second_read.update(c.branch for c in d.concepts if not c.get("derive"))
    event_cursor = {split: 0 for split in SPLITS}
    token_cursor = {split: 0 for split in SPLITS}
    cursor = 0
    for chunk in _chunks(paths, cfg.paths.tree, sorted(second_read), limit):
        ehit = _as_3d(chunk[d.ehit_branch])
        expe = _as_3d(chunk[d.expehit_branch])
        n = len(ehit)
        chunk_split = np.asarray(raw_split[cursor:cursor + n])

        energy = np.asarray(chunk[d.target_branch]).ravel().astype(np.float32)
        angle = np.stack(
            [np.asarray(chunk[name]).ravel() for name in d.angle_branches], axis=1
        ).astype(np.float32)
        fit_angle = np.stack(
            [_scalar(chunk[name], 0).ravel() for name in d.fit_angle_branches], axis=1
        ).astype(np.float32)
        run = np.asarray(chunk[d.run_branch]).ravel().astype(np.uint32)
        event = np.asarray(chunk[d.event_branch]).ravel().astype(np.uint32)
        fit_status = _scalar(chunk[d.stat_branch], 0).ravel().astype(np.int32)
        n_shower = np.asarray(chunk[d.nshwr_branch]).ravel().astype(np.int16)

        sources = {c.get("source", "expe") for c in d.concepts if c.get("derive")}
        shape = {source: _shape_concepts(expe if source == "expe" else ehit, table)
                 for source in sources}
        concept_columns = []
        for concept in d.concepts:
            if concept.get("derive"):
                concept_columns.append(shape[concept.get("source", "expe")][concept.derive])
            else:
                concept_columns.append(_scalar(chunk[concept.branch], concept.index))
        concepts = np.stack(concept_columns, axis=1).astype(np.float32)

        for split_id, split in enumerate(SPLITS):
            index = np.flatnonzero(chunk_split == split_id)
            if len(index) == 0:
                continue
            e0, e1 = event_cursor[split], event_cursor[split] + len(index)
            arrays = writers[split]
            arrays["energy"][e0:e1] = energy[index]
            arrays["angle"][e0:e1] = angle[index]
            arrays["fit_angle"][e0:e1] = fit_angle[index]
            arrays["concepts"][e0:e1] = concepts[index]
            arrays["run"][e0:e1] = run[index]
            arrays["event"][e0:e1] = event[index]
            arrays["fit_status"][e0:e1] = fit_status[index]
            arrays["n_shower"][e0:e1] = n_shower[index]

            selected_ehit = ehit[index]
            selected_expe = expe[index]
            token_event, layer, cell = np.nonzero(selected_ehit > threshold)
            expected = np.asarray(counts[cursor + index], dtype=np.int64)
            actual = np.bincount(token_event, minlength=len(index))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"pass-2 token counts disagree in {split}")
            t0, t1 = token_cursor[split], token_cursor[split] + len(layer)
            arrays["tok_layer"][t0:t1] = layer.astype(np.int16)
            arrays["tok_cell"][t0:t1] = cell.astype(np.int16)
            arrays["tok_ehit"][t0:t1] = selected_ehit[token_event, layer, cell]
            arrays["tok_expe"][t0:t1] = selected_expe[token_event, layer, cell]
            event_cursor[split], token_cursor[split] = e1, t1

        cursor += n
        if cursor % 1000000 < n:
            print(f"PASS2 events={cursor}/{n_raw}", flush=True)

    if cursor != n_raw:
        raise RuntimeError(f"pass 2 saw {cursor} events, expected {n_raw}")
    for split in SPLITS:
        expected = split_sizes[split]
        if event_cursor[split] != expected["events"]:
            raise RuntimeError(f"{split} event write count mismatch")
        if token_cursor[split] != expected["tokens"]:
            raise RuntimeError(f"{split} token write count mismatch")
        for array in writers[split].values():
            array.flush()

    train = writers["train"]
    log_mean, log_std, log_count = _finite_moments(
        train["energy"], transform=lambda x: np.log(np.clip(x, 1e-3, None)))
    concept_mean, concept_std, concept_count = _finite_moments(train["concepts"])
    angle_mean, angle_std, angle_count = _finite_moments(train["angle"])
    concept_names = [concept.name for concept in d.concepts]
    meta = {
        "complete": True,
        "schema_version": CACHE_SCHEMA_VERSION,
        "storage_layout": "split_npy_memmap_v1",
        "cache_layout": "split_npy_memmap_v1",
        "fields": sorted(writers["train"]),
        "task_mode_built_for": task_mode,
        "component_views": list(COMPONENT_VIEWS),
        "n_concepts": n_concepts,
        "concept_names": concept_names,
        "concept_reflect_x": [int(c.reflect_x) for c in d.concepts],
        "concept_reflect_y": [int(c.reflect_y) for c in d.concepts],
        "concept_coord": [c.get("coord", None) for c in d.concepts],
        "concept_mean": concept_mean.tolist(),
        "concept_std": concept_std.tolist(),
        "concept_finite_train_count": concept_count.tolist(),
        "log_energy_mean": float(log_mean[0]),
        "log_energy_std": float(log_std[0]),
        "log_energy_finite_train_count": int(log_count[0]),
        "angle_names": list(d.angle_names),
        "angle_mean": angle_mean.tolist(),
        "angle_std": angle_std.tolist(),
        "angle_finite_train_count": angle_count.tolist(),
        "angle_reflect_x": [-1, 1],
        "angle_reflect_y": [1, -1],
        "threshold_mev": threshold,
        "cell_scale": float(d.cell_scale),
        "geometry_data_type": cfg.geometry.data_type,
        "selection": sel.to_dict(),
        "raw_events_seen": n_raw,
        "retained_events": n_kept,
        "dropped_by_selection_or_zero_tokens": n_zero_or_cut,
        "split_seed": int(cfg.seed),
        "split_sizes": split_sizes,
        "quality_fields": {
            "fit_status": "kx_ShwrStat[0]",
            "n_shower": "kx_N_Shwr",
            "old_selection": "(fit_status & 7) == 7 and n_shower == 1",
        },
        "cutflow_cumulative": cutflow,
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    # Build-only bookkeeping is not part of the cache interface.
    del writers, raw_split, counts
    os.remove(split_path)
    os.remove(count_path)
    print(f"MEMMAP_CACHE_OK retained={n_kept}/{n_raw} splits={split_sizes}", flush=True)


if __name__ == "__main__":
    main()
