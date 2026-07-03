"""Profile catastrophic events: WHERE do the |dE/E| > 20% outliers live?

    python outlier_study.py --run /abs/champion_run_dir --ckpt best.pt --out runs/outliers

Runs the model on the test split, splits events into
    core         |dE/E| <= 0.20
    outlier      0.20 < |dE/E| <= 1.0
    catastrophic |dE/E| > 1.0
and profiles both classes against per-event physics variables built from the cache
(no ROOT access needed):
    E_true, n_tokens, deposited sum(ehit), max-cell fraction (saturation proxy),
    fit_resid = sum|ehit-expe|/sum(ehit)  (3D-fit goodness, the v2m1 weight input),
    first/last/N layers hit, and the 11 stored 3D-fit concepts (raw units), plus
    |slope| = sqrt(kx^2+ky^2) and the transverse distance of (x0,y0) to the array edge.
Outputs: outliers.md (median profile + outlier-rate-by-decile tables),
         outlier_rates.png, worst100.csv (eyeball list).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from utils.config import NS
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer

CACHE = "/aifs/user/data/lishanglin/chenhao/transformer/cache_full10"


@torch.no_grad()
def predict(run_dir, ckpt_name, cache, device):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = NS(json.load(f))
    cfg.paths.cache_dir = cache
    meta = load_meta(cache)
    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, ckpt_name),
                                     map_location=device)["model"])
    model.eval()
    amp = str(cfg.train.get("amp_dtype", "bf16")).lower() == "bf16"
    ds = EcalTokens(cache, "test", meta, concept_use=concept_use)
    # shuffle=False -> batches enumerate sorted event order; recover the permutation
    # so per-event cache variables can be aligned with the prediction stream.
    loader = make_loader(ds, cfg, shuffle=False)
    idx_order = [i for b in loader.batch_sampler._make_batches(np.random.default_rng(0))
                 for i in b]
    ep, et = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        ep.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
        et.append(batch["energy"].cpu().numpy())
    return ds, meta, np.asarray(idx_order), np.concatenate(ep), np.concatenate(et)


def event_variables(ds, meta):
    """Per-event physics variables from the CSR token cache (event order)."""
    off, layer, ehit = ds.off, ds.layer, ds.ehit
    n = len(ds.energy)
    seg = np.repeat(np.arange(n), np.diff(off))
    esum = np.bincount(seg, weights=ehit, minlength=n)
    emax = np.zeros(n)
    np.maximum.at(emax, seg, ehit)
    first = np.full(n, 99); last = np.zeros(n)
    np.minimum.at(first, seg, layer)
    np.maximum.at(last, seg, layer)
    nlay = np.zeros(n, dtype=np.int64)
    for il in range(18):
        hasl = np.zeros(n, dtype=bool)
        hasl[seg[layer == il]] = True
        nlay += hasl
    v = {
        "E_true_GeV": ds.energy.astype(np.float64),
        "n_tokens": np.diff(off).astype(np.float64),
        "sum_ehit_GeV": esum / 1000.0,
        # deposited fraction: sampling calorimeter sees ~a constant fraction of E for
        # well-behaved showers -> a LOW dep_frac flags energy the ECAL never sampled
        "dep_frac": (esum / 1000.0) / np.clip(ds.energy.astype(np.float64), 1e-3, None),
        "maxcell_frac": emax / np.clip(esum, 1e-9, None),
        "fit_resid": ds.fit_resid.astype(np.float64),
        "first_layer": first.astype(np.float64),
        "last_layer": last.astype(np.float64),
        "n_layers": nlay.astype(np.float64),
    }
    # stored concepts, de-standardised to raw 3D-fit units
    craw = ds.concepts * ds.cstd + ds.cmean
    for j, name in enumerate(ds.concept_names):
        v[f"c_{name}"] = craw[:, j].astype(np.float64)
    if "c_shwr_kx" in v:
        v["slope_mag"] = np.hypot(v["c_shwr_kx"], v["c_shwr_ky"])
    if "c_shwr_x0" in v:
        # unit-agnostic edge proxy: distance of the shower centre from the array
        # centre, worst axis (the raw fit frame/units are not the token t_cm frame,
        # so an absolute edge distance would mix units; the DECILE TREND of this
        # variable is what flags edge events)
        v["abs_pos_max"] = np.maximum(np.abs(v["c_shwr_x0"] - np.median(v["c_shwr_x0"])),
                                      np.abs(v["c_shwr_y0"] - np.median(v["c_shwr_y0"])))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default="runs/outliers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, meta, idx, e_pred, e_true = predict(args.run, args.ckpt, args.cache, device)
    r = (e_pred - e_true) / np.clip(e_true, 1e-3, None)
    vs = event_variables(ds, meta)
    vs = {k: a[idx] for k, a in vs.items()}          # align cache order -> batch order
    assert np.allclose(vs["E_true_GeV"], e_true, rtol=1e-4), "event alignment broken"

    a = np.abs(r)
    cls = {"core": a <= 0.20, "outlier": (a > 0.20) & (a <= 1.0), "catastrophic": a > 1.0}
    n = len(r)
    print({k: int(m.sum()) for k, m in cls.items()}, "of", n)

    # ---- median profile per class + robust separation (median shift / core MAD)
    lines = ["## Median profile per class",
             "",
             f"events: core {cls['core'].sum()}, outlier {cls['outlier'].sum()}, "
             f"catastrophic {cls['catastrophic'].sum()} (of {n}; "
             f"outlier = 0.2<|dE/E|<=1, catastrophic = |dE/E|>1)",
             "",
             "| variable | core | outlier | catastrophic | sep(out) | sep(cat) |",
             "|---|---|---|---|---|---|"]
    seps = {}
    for k, x in vs.items():
        med = {c: float(np.median(x[m])) if m.any() else float("nan")
               for c, m in cls.items()}
        mad = float(np.median(np.abs(x[cls["core"]] - med["core"]))) + 1e-12
        s_out = (med["outlier"] - med["core"]) / mad
        s_cat = (med["catastrophic"] - med["core"]) / mad
        seps[k] = max(abs(s_out), abs(s_cat))
        lines.append(f"| {k} | {med['core']:.4g} | {med['outlier']:.4g} "
                     f"| {med['catastrophic']:.4g} | {s_out:+.2f} | {s_cat:+.2f} |")

    # ---- outlier RATE by decile for the most separating variables
    top = sorted(seps, key=seps.get, reverse=True)[:8]
    lines += ["", "## Outlier rate (|dE/E|>0.2) by variable decile", "",
              "| variable | " + " | ".join(f"d{i}" for i in range(10)) + " |",
              "|---|" + "---|" * 10]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    is_out = a > 0.20
    for ax, k in zip(axes.ravel(), top):
        x = vs[k]
        qs = np.quantile(x, np.linspace(0, 1, 11))
        qs[-1] += 1e-9
        rates, cts = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (x >= lo) & (x < hi)
            rates.append(float(np.mean(is_out[m])) if m.any() else 0.0)
            cts.append(0.5 * (lo + hi))
        lines.append(f"| {k} | " + " | ".join(f"{p*100:.1f}" for p in rates) + " |")
        ax.plot(cts, np.array(rates) * 100, "o-")
        ax.set_xlabel(k); ax.set_ylabel("outlier rate [%]"); ax.grid(alpha=.3)
    fig.suptitle("Outlier rate (|dE/E|>20%) vs the 8 most separating variables")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "outlier_rates.png"), dpi=130)

    # ---- worst-100 event dump for eyeballing
    worst = np.argsort(-a)[:100]
    cols = ["r", "E_true_GeV", "E_pred_GeV"] + list(vs.keys())
    with open(os.path.join(args.out, "worst100.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for i in worst:
            row = [f"{r[i]:.4f}", f"{e_true[i]:.1f}", f"{e_pred[i]:.1f}"]
            row += [f"{vs[k][i]:.5g}" for k in vs]
            f.write(",".join(row) + "\n")

    with open(os.path.join(args.out, "outliers.md"), "w") as f:
        f.write(f"# Catastrophic-event profile ({os.path.basename(args.run)}/"
                f"{args.ckpt}, test split)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}/outliers.md, outlier_rates.png, worst100.csv")


if __name__ == "__main__":
    main()
