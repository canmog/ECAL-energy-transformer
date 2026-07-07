"""Dump fp32 predictions of a v3-lineage checkpoint for any cache split.

    python dump_preds_v3.py --run <run_dir> --ckpt best_robust.pt --split test val
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from utils.config import NS
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer

CACHE = "/aifs/user/data/lishanglin/chenhao/transformer/cache_full10"
OUT = "/aifs/user/data/lishanglin/chenhao/ecalTransformer/runs/methodfig2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best_robust.pt")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--tag", default=None, help="output basename (default: run dir name)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = NS(json.load(f))
    cfg.paths.cache_dir = args.cache
    meta = load_meta(args.cache)
    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, args.ckpt),
                                     map_location=device)["model"])
    model.eval()

    tag = args.tag or os.path.basename(args.run.rstrip("/"))
    for split in args.split:
        ds = EcalTokens(args.cache, split, meta, concept_use=concept_use)
        loader = make_loader(ds, cfg, shuffle=False)
        ep, et = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)                       # fp32, no autocast
                ep.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
                et.append(batch["energy"].cpu().numpy())
        path = f"{OUT}/{tag}_{split}.npz"
        np.savez(path, e_pred=np.concatenate(ep), e_true=np.concatenate(et))
        print(f"wrote {path} ({len(np.concatenate(et))} events)")


if __name__ == "__main__":
    main()
