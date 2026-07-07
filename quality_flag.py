"""Post-hoc per-event quality flag: predict P(|dE/E| > 0.2) from RECO-LEVEL
variables only (no truth), so downstream analysis can cut or deweight unreliable
events. Trained on the VAL split, evaluated on the TEST split — the energy model
itself is untouched (frozen); this is a separate small classifier.

Features (all reconstruction-level): 3D-fit residual, max-cell fraction, token and
layer counts, deposited energy relative to the PREDICTED energy, log E_pred, and the
3D-fit shape/leakage outputs (frac_rear, frac_lat, tmax, widths, z0, a0).

    python quality_flag.py    # needs dump_preds_v3 val+test npz first
Outputs: runs/quality/quality_flag.md + transformerReport/figs/f18_quality_flag.pdf
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from data.dataset import EcalTokens, load_meta

BASE = "/aifs/user/data/lishanglin/chenhao"
CACHE = f"{BASE}/transformer/cache_full10"
MF2 = f"{BASE}/ecalTransformer/runs/methodfig2"
FIGS = f"{BASE}/transformerReport/figs"
RECO_CONCEPTS = ["shwr_z0", "shwr_a0", "frac_lat", "frac_rear",
                 "tmax", "lat_width", "long_width"]


def features(split, e_pred):
    """Reco-level feature matrix for one split, aligned to the PREDICTION order
    (the shuffle=False loader enumerates events in length-sorted batch order)."""
    meta = load_meta(CACHE)
    ds = EcalTokens(CACHE, split, meta)
    from data.dataset import make_loader  # noqa: E402
    from utils.config import NS           # noqa: E402
    import json                            # noqa: E402
    with open(f"{BASE}/transformer_v3/runs/v3_d192_ep100/config.json") as f:
        cfg = NS(json.load(f))
    loader = make_loader(ds, cfg, shuffle=False)
    order = np.asarray([i for b in loader.batch_sampler._make_batches(
        np.random.default_rng(0)) for i in b])

    off, layer, ehit = ds.off, ds.layer, ds.ehit
    n = len(ds.energy)
    seg = np.repeat(np.arange(n), np.diff(off))
    esum = np.bincount(seg, weights=ehit, minlength=n)
    emax = np.zeros(n); np.maximum.at(emax, seg, ehit)
    num = np.add.reduceat(np.abs(ds.ehit - ds.expe), off[:-1])
    fit_resid = num / np.clip(esum, 1e-9, None)
    craw = ds.concepts * ds.cstd + ds.cmean
    cidx = {nm: j for j, nm in enumerate(ds.concept_names)}

    # align: features are computed in EVENT order; predictions arrive in BATCH order
    # (order[k] = event index of the k-th prediction) -> reindex features by `order`.
    X = np.column_stack([fit_resid, emax / np.clip(esum, 1e-9, None),
                         np.diff(off).astype(np.float64), np.log1p(esum)] +
                        [craw[:, cidx[nm]] for nm in RECO_CONCEPTS])[order]
    dep_over_pred = (esum[order] / 1000.0) / np.clip(e_pred, 1e-3, None)
    X = np.column_stack([X, dep_over_pred, np.log(np.clip(e_pred, 1e-3, None))])
    names = (["fit_resid", "maxcell_frac", "n_tokens", "log_sum_ehit"] + RECO_CONCEPTS
             + ["dep_over_pred", "log_e_pred"])
    e_true = ds.energy[order].astype(np.float64)
    return X, names, e_true


def main():
    os.makedirs(f"{BASE}/ecalTransformer/runs/quality", exist_ok=True)
    data = {}
    for split in ("val", "test"):
        z = np.load(f"{MF2}/v3_d192_ep100_{split}.npz")
        e_pred, e_true = z["e_pred"].astype(np.float64), z["e_true"].astype(np.float64)
        X, names, et_chk = features(split, e_pred)
        assert np.allclose(et_chk, e_true, rtol=1e-4), f"{split}: alignment broken"
        y = (np.abs((e_pred - e_true) / e_true) > 0.20).astype(int)
        data[split] = (X, y, e_pred, e_true)
        print(f"{split}: {X.shape[0]} events, {y.mean()*100:.2f}% outliers")

    Xv, yv, _, _ = data["val"]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         max_depth=6, random_state=0)
    clf.fit(Xv, yv)
    Xt, yt, ep_t, et_t = data["test"]
    score = clf.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, score)
    print(f"test AUC = {auc:.4f}")

    # rejection curve: cut the worst q% by score, measure remaining outlier fraction
    qs = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10])
    rows = []
    for q in qs:
        thr = np.quantile(score, 1 - q) if q > 0 else np.inf
        keep = score < thr
        rows.append((q * 100, float(np.mean(keep) * 100),
                     float(yt[keep].mean() * 100),
                     float(yt[~keep].mean() * 100) if (~keep).any() else 0.0))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", color="#2ca02c")
    ax.set_xlabel("events removed by quality flag [%]")
    ax.set_ylabel("outlier fraction of KEPT events [%]")
    ax.set_title(f"Post-hoc quality flag (reco-level only) — test AUC {auc:.3f}")
    ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIGS}/f18_quality_flag.pdf"); plt.close()

    with open(f"{BASE}/ecalTransformer/runs/quality/quality_flag.md", "w") as f:
        f.write(f"# Post-hoc quality flag (test AUC {auc:.4f})\n\n"
                "| cut worst % | kept % | outlier frac kept | outlier frac removed |\n"
                "|---|---|---|---|\n")
        for q, kept, ok, orem in rows:
            f.write(f"| {q:.1f} | {kept:.1f} | {ok:.3f}% | {orem:.1f}% |\n")
    for q, kept, ok, orem in rows:
        print(f"cut {q:4.1f}%  kept {kept:5.1f}%  outliers kept {ok:.3f}%  removed-purity {orem:.1f}%")
    print(f"wrote runs/quality/quality_flag.md + {FIGS}/f18_quality_flag.pdf")


if __name__ == "__main__":
    main()
