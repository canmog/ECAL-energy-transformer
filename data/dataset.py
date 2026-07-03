"""Torch dataset over the tokenised cache + physics-correct reflection augmentation.

Token feature vector (per hit cell):
    [ log1p(ehit_MeV), t_norm, z_norm, depth_norm, view_x, view_y ]   (dim = 6)
plus an integer (layer,cell) id for the learned positional embedding.

Augmentation = x / y mirror only (NO 90deg: 5 X vs 4 Y superlayers). A mirror
re-indexes the affected view's cells (icell -> 71-icell) and sign-flips the
concept targets via their reflect_x / reflect_y signs (from meta.json).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.geometry import (build_geometry_table, mirror_centers,
                           token_geometry_features, GEOM_FEATURE_DIM, N_CELL)

TOKEN_FEATURE_DIM = 1 + GEOM_FEATURE_DIM   # log1p(E) + geometry block


class EcalTokens(Dataset):
    def __init__(self, cache_dir, split, meta, train=False, augment=None, concept_use=None):
        self.train = train
        self.aug = augment or {}
        z = np.load(os.path.join(cache_dir, f"{split}.npz"))
        self.off = z["off"]
        self.layer = z["tok_layer"].astype(np.int64)
        self.cell = z["tok_cell"].astype(np.int64)
        self.ehit = z["tok_ehit"].astype(np.float32)
        self.expe = z["tok_expe"].astype(np.float32)
        self.energy = z["energy"].astype(np.float32)
        concepts = z["concepts"].astype(np.float32)

        # Concept-set trim (load-time column subset; NO cache rebuild). concept_use is
        # an optional list of concept NAMES to keep (energy-irrelevant x0/y0/kx/ky are
        # typically dropped); None = all stored concepts, in cache order.
        names = list(meta["concept_names"])
        idx = [names.index(n) for n in concept_use] if concept_use else list(range(len(names)))
        self.cidx = np.asarray(idx, dtype=np.int64)
        self.concept_names = [names[j] for j in idx]
        self.n_concepts = len(idx)
        self.concepts = concepts[:, self.cidx]

        self.table = build_geometry_table(meta["geometry_data_type"])
        self.view_of_layer = self.table["view"][:, 0]          # (18,) 0=X 1=Y
        self.log_mean = meta["log_energy_mean"]
        self.log_std = meta["log_energy_std"]
        self.cmean = np.asarray(meta["concept_mean"], np.float32)[self.cidx]
        self.cstd = np.asarray(meta["concept_std"], np.float32)[self.cidx]
        self.rx = np.asarray(meta["concept_reflect_x"], np.float32)[self.cidx]
        self.ry = np.asarray(meta["concept_reflect_y"], np.float32)[self.cidx]
        # Coordinate concepts flip about the per-view array centre, not 0: the
        # affine is  v -> rx*v + off  with off = 2*c̄_v for that axis's coords.
        # (Old caches without concept_coord fall back to the pure-sign mirror.)
        cents = mirror_centers(self.table)
        coord = meta.get("concept_coord") or [None] * len(names)
        coord = [coord[j] for j in idx]
        self.offx = np.array([2 * cents[0] if c == "x" else 0.0 for c in coord], np.float32)
        self.offy = np.array([2 * cents[1] if c == "y" else 0.0 for c in coord], np.float32)

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, i):
        s, t = self.off[i], self.off[i + 1]
        layer = self.layer[s:t].copy()
        cell = self.cell[s:t].copy()
        ehit = self.ehit[s:t]
        expe = self.expe[s:t]
        concept = self.concepts[i].copy()

        if self.train:
            view = self.view_of_layer[layer]                   # per-token view
            if self.aug.get("reflect_x") and np.random.rand() < 0.5:
                m = view == 0
                cell[m] = (N_CELL - 1) - cell[m]
                concept = concept * self.rx + self.offx
            if self.aug.get("reflect_y") and np.random.rand() < 0.5:
                m = view == 1
                cell[m] = (N_CELL - 1) - cell[m]
                concept = concept * self.ry + self.offy

        geo = token_geometry_features(layer, cell, self.table)       # (n,5)
        e_feat = np.log1p(np.clip(ehit, 0, None))[:, None].astype(np.float32)
        feats = np.concatenate([e_feat, geo], axis=1)                 # (n,6)
        pos_id = (layer * N_CELL + cell).astype(np.int64)            # (n,)
        recon = np.log1p(np.clip(expe, 0, None)).astype(np.float32)   # (n,) target

        log_e = (np.log(max(self.energy[i], 1e-3)) - self.log_mean) / self.log_std
        concept_std = (concept - self.cmean) / self.cstd

        return {
            "feats": torch.from_numpy(feats),
            "pos_id": torch.from_numpy(pos_id),
            "recon": torch.from_numpy(recon),
            "log_e": torch.tensor(log_e, dtype=torch.float32),
            "energy": torch.tensor(self.energy[i], dtype=torch.float32),
            "concepts": torch.from_numpy(concept_std.astype(np.float32)),
        }


def collate(batch):
    L = max(b["feats"].shape[0] for b in batch)
    B = len(batch)
    F = batch[0]["feats"].shape[1]
    feats = torch.zeros(B, L, F)
    pos_id = torch.zeros(B, L, dtype=torch.long)
    recon = torch.zeros(B, L)
    valid = torch.zeros(B, L, dtype=torch.bool)         # True = real token
    for j, b in enumerate(batch):
        n = b["feats"].shape[0]
        feats[j, :n] = b["feats"]
        pos_id[j, :n] = b["pos_id"]
        recon[j, :n] = b["recon"]
        valid[j, :n] = True
    return {
        "feats": feats,
        "pos_id": pos_id,
        "recon": recon,
        "valid": valid,                                  # True = real token
        "log_e": torch.stack([b["log_e"] for b in batch]),
        "energy": torch.stack([b["energy"] for b in batch]),
        "concepts": torch.stack([b["concepts"] for b in batch]),
    }


def load_meta(cache_dir):
    with open(os.path.join(cache_dir, "meta.json")) as f:
        return json.load(f)


class TokenBudgetBatchSampler(Sampler):
    """Batch by PADDED-token budget, not event count.

    Selected (contained, high-E) showers average ~390 tokens and reach ~1160, so a
    fixed event batch padded to the batch max blows past GPU memory (B x L_max
    activations). Instead: shuffle -> sort within windows by length -> greedily pack
    while (n_events x L_max) <= max_tokens -> shuffle batch order. Batches are
    near-homogeneous in length, bounding memory AND wasting little padding compute.
    """

    def __init__(self, lengths, max_tokens=65536, max_events=512,
                 shuffle=True, seed=0, window=8192):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.max_tokens = int(max_tokens)
        self.max_events = int(max_events)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.window = int(window)
        self._epoch = 0
        self._n_batches = len(self._make_batches(np.random.default_rng(self.seed)))

    def _pack(self, order):
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = int(self.lengths[i])
            new_max = max(cur_max, L)
            if cur and ((len(cur) + 1) * new_max > self.max_tokens
                        or len(cur) + 1 > self.max_events):
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def _make_batches(self, rng):
        n = len(self.lengths)
        if self.shuffle:
            idx = rng.permutation(n)
            order = np.concatenate([
                w[np.argsort(self.lengths[w], kind="stable")]
                for w in np.array_split(idx, max(1, n // self.window))])
        else:
            order = np.argsort(self.lengths, kind="stable")
        batches = self._pack(order)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        yield from self._make_batches(rng)

    def __len__(self):
        return self._n_batches


def _seed_worker(worker_id):
    """Re-seed numpy per DataLoader worker. Under fork, every worker inherits the
    SAME global np.random state, so the augmentation coin flips would be identical
    across all workers; torch gives each worker a distinct initial_seed()."""
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(ds, cfg, shuffle, seed=0):
    """Token-budget DataLoader shared by train / evaluate / probe."""
    lengths = ds.off[1:] - ds.off[:-1]
    sampler = TokenBudgetBatchSampler(
        lengths,
        max_tokens=cfg.train.get("max_tokens_per_batch", 65536),
        max_events=cfg.train.batch_size,
        shuffle=shuffle, seed=seed)
    nw = cfg.train.num_workers
    return DataLoader(ds, batch_sampler=sampler, num_workers=nw, collate_fn=collate,
                      pin_memory=True, persistent_workers=nw > 0,
                      worker_init_fn=_seed_worker)
