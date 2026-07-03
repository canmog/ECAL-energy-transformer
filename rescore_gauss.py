"""Re-score saved checkpoints with the GAUSSIAN CORE estimator (no retraining).

    python rescore_gauss.py --out runs/rescore              # default checkpoint list
    python rescore_gauss.py --out runs/rescore --runs lab=/abs/run_dir:best.pt ...

For every run: rebuild the model from the run's own config.json, run the shared
test split (cache_full10 by default -> the SAME 507k events for every row), and
report sigma/E under three estimators side by side:
    gauss  = iterative +/-2 sigma binned Gaussian fit  (utils/stats.gauss_core)
    robust = IQR/1.349                                 (the v2 primary)
    raw    = np.std                                    (tail-sensitive legacy)
overall, restricted to E<=2 TeV, and per energy bin (3 TeV edge included). The
3D-fit reference (kx_EneL2Cor stored as `anchor` in cache_m2) is scored the same
way without a model. Output: <out>/rescore_gauss.json + rescore_gauss.md.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from utils.config import NS
from utils.stats import gauss_core, robust_sigma
from data.dataset import EcalTokens, load_meta, make_loader
from models.model import EcalTransformer

BASE = "/aifs/user/data/lishanglin/chenhao"
CACHE = f"{BASE}/transformer/cache_full10"
ANCHOR_CACHE = f"{BASE}/transformer_m2/cache_m2"
BINS = [0, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 3000, 4000]
E_CUT = 2000.0

# label = run_dir : checkpoint  (v2-lineage architecture only; m3/dual-head builds
# and the eneRec MLP use different model code / samples and are noted, not scored)
DEFAULT_RUNS = [
    ("sw_d192",           f"{BASE}/transformer/runs/sw_d192",            "best.pt"),
    ("m1_d256",           f"{BASE}/transformer/runs/m1_d256",            "best.pt"),
    ("m1_d320",           f"{BASE}/transformer/runs/m1_d320",            "best.pt"),
    ("v2_d192",           f"{BASE}/transformer_v2/runs/v2_d192",         "best.pt"),
    ("v2_wdx",            f"{BASE}/transformer_v2/runs/v2_d192_wdx",     "best.pt"),
    ("v2_ep100",          f"{BASE}/transformer_v2/runs/v2_d192_ep100",   "best.pt"),
    ("v2m1",              f"{BASE}/transformer_v2m1/runs/v2m1_d192",     "best.pt"),
    ("v2m1_wdx",          f"{BASE}/transformer_v2m1/runs/v2m1_wdx_d192", "best.pt"),
    ("v2test_rawsel",     f"{BASE}/transformer_v2test/runs/v2test_d192", "best.pt"),
    ("v2b_best",          f"{BASE}/transformer_v2b/runs/v2b_d192",       "best.pt"),
    ("v2b_bestrobust",    f"{BASE}/transformer_v2b/runs/v2b_d192",       "best_robust.pt"),
    ("v2b_bestraw",       f"{BASE}/transformer_v2b/runs/v2b_d192",       "best_raw.pt"),
    ("v2cham_best",       f"{BASE}/transformer_v2cham/runs/v2cham_d192", "best.pt"),
    ("v2cham_bestrobust", f"{BASE}/transformer_v2cham/runs/v2cham_d192", "best_robust.pt"),
    ("v2cham_bestraw",    f"{BASE}/transformer_v2cham/runs/v2cham_d192", "best_raw.pt"),
    ("v2cham_ex03",       f"{BASE}/transformer_v2cham/runs/v2cham_ex03_d192", "best.pt"),
    ("v2cham_ex05",       f"{BASE}/transformer_v2cham/runs/v2cham_ex05_d192", "best.pt"),
    ("v2cham_ex10",       f"{BASE}/transformer_v2cham/runs/v2cham_ex10_d192", "best.pt"),
    # v3 runs (gauss-core selection); skipped gracefully if not yet trained
    ("v3_d192",           f"{BASE}/transformer_v3/runs/v3_d192",              "best.pt"),
    ("v3_d192_gauss",     f"{BASE}/transformer_v3/runs/v3_d192",              "best_gauss.pt"),
    ("v3_d256",           f"{BASE}/transformer_v3/runs/v3_d256",              "best.pt"),
    ("v3_d320",           f"{BASE}/transformer_v3/runs/v3_d320",              "best.pt"),
]


def score(r, e_true):
    """Full three-estimator summary + per-bin table for one residual set."""
    fin = np.isfinite(r)
    r, e_true = r[fin], e_true[fin]
    m2 = e_true <= E_CUT
    g_all, g_cut = gauss_core(r), gauss_core(r[m2])
    per_bin = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < 100:
            continue
        g = gauss_core(r[m])
        per_bin.append({"center": 0.5 * (lo + hi), "n": int(m.sum()),
                        "gauss": g["sigma"], "gauss_mu": g["mu"], "gauss_ok": int(g["ok"]),
                        "robust": robust_sigma(r[m]), "raw": float(np.std(r[m])),
                        "median_bias": float(np.median(r[m]))})
    return {"n": int(r.size),
            "gauss": g_all["sigma"], "gauss_mu": g_all["mu"], "gauss_ok": int(g_all["ok"]),
            "gauss_le_cut": g_cut["sigma"], "gauss_mu_le_cut": g_cut["mu"],
            "binavg_gauss": float(np.mean([b["gauss"] for b in per_bin])) if per_bin else None,
            "robust": robust_sigma(r), "robust_le_cut": robust_sigma(r[m2]),
            "raw": float(np.std(r)), "raw_le_cut": float(np.std(r[m2])),
            "outlier": float(np.mean(np.abs(r) > 0.20)),
            "median_bias": float(np.median(r)),
            "bins": per_bin}


@torch.no_grad()
def predict(run_dir, ckpt_name, cache, device, force_fp32=False):
    """Rebuild the model from the run's own config.json and predict the test split."""
    with open(os.path.join(run_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = NS(raw)
    cfg.paths.cache_dir = cache
    meta = load_meta(cache)
    concept_use = cfg.data.get("concept_use", None)
    n_concepts = len(concept_use) if concept_use else meta["n_concepts"]
    model = EcalTransformer(cfg, n_concepts).to(device)
    ck = torch.load(os.path.join(run_dir, ckpt_name), map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    # bf16 inference adds ~0.4% per-event quantisation noise -> inflates the core by
    # 5-13% relative (measured 2026-07-02). --fp32 removes it; bf16 kept as default
    # only to reproduce the historical convention.
    amp = (str(cfg.train.get("amp_dtype", "bf16")).lower() == "bf16") and not force_fp32

    ds = EcalTokens(cache, "test", meta, concept_use=concept_use)
    loader = make_loader(ds, cfg, shuffle=False)
    ep, et = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        ep.append(model.predict_energy_gev(out["energy"].float()).cpu().numpy())
        et.append(batch["energy"].cpu().numpy())
    e_pred, e_true = np.concatenate(ep), np.concatenate(et)
    return e_pred, e_true, int(ck.get("epoch", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/rescore")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--anchor-cache", default=ANCHOR_CACHE)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="label=/abs/run_dir[:ckpt.pt] entries; default = built-in list")
    ap.add_argument("--fp32", action="store_true",
                    help="force fp32 inference (removes the bf16 core inflation)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = DEFAULT_RUNS
    if args.runs:
        runs = []
        for spec in args.runs:
            lab, rest = spec.split("=", 1)
            d, _, ck = rest.partition(":")
            runs.append((lab, d, ck or "best.pt"))

    results, order = {}, []

    # 3D-fit reference: kx_EneL2Cor stored as `anchor` in the m2 cache (same events).
    apath = os.path.join(args.anchor_cache, "test.npz")
    if os.path.exists(apath):
        z = np.load(apath)
        et, anc = z["energy"].astype(np.float64), z["anchor"].astype(np.float64)
        r3 = (anc - et) / np.clip(et, 1e-3, None)
        results["3dfit_EneL2Cor"] = {"ckpt": "-", "epoch": -1, **score(r3, et)}
        order.append("3dfit_EneL2Cor")
        print(f"[3dfit_EneL2Cor] gauss<=2TeV {results['3dfit_EneL2Cor']['gauss_le_cut']*100:.2f}%")

    for lab, run_dir, ck in runs:
        try:
            e_pred, e_true, epoch = predict(run_dir, ck, args.cache, device,
                                            force_fp32=args.fp32)
            r = (e_pred - e_true) / np.clip(e_true, 1e-3, None)
            results[lab] = {"ckpt": os.path.join(run_dir, ck), "epoch": epoch,
                            **score(r, e_true)}
            order.append(lab)
            s = results[lab]
            print(f"[{lab}] gauss<=2TeV {s['gauss_le_cut']*100:.2f}%  "
                  f"gauss {s['gauss']*100:.2f}%  robust<=2TeV {s['robust_le_cut']*100:.2f}%  "
                  f"raw {s['raw']*100:.2f}%  outlier {s['outlier']*100:.2f}%  (ep {epoch})")
        except Exception as e:
            print(f"[{lab}] FAILED: {type(e).__name__}: {e}")
            results[lab] = {"error": f"{type(e).__name__}: {e}"}

    with open(os.path.join(args.out, "rescore_gauss.json"), "w") as f:
        json.dump({"bins": BINS, "e_cut": E_CUT, "cache": args.cache,
                   "results": results}, f, indent=2)

    # markdown summary table (values in %)
    lines = ["| run | gauss <=2TeV | gauss full | gauss mu | robust <=2TeV | raw std "
             "| binavg gauss | outlier | med bias | epoch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in order:
        s = results[lab]
        if "error" in s:
            continue
        ba = f"{s['binavg_gauss']*100:.2f}" if s["binavg_gauss"] is not None else "-"
        lines.append(
            f"| {lab} | {s['gauss_le_cut']*100:.2f} | {s['gauss']*100:.2f} "
            f"| {s['gauss_mu_le_cut']*100:+.2f} | {s['robust_le_cut']*100:.2f} "
            f"| {s['raw']*100:.2f} | {ba} | {s['outlier']*100:.2f} "
            f"| {s['median_bias']*100:+.2f} | {s['epoch']} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, "rescore_gauss.md"), "w") as f:
        f.write("# Gaussian-core re-scoring (iterative +/-2 sigma fit; all values %)\n\n"
                + md + "\n")
    print("\n" + md)
    print(f"\nwrote {args.out}/rescore_gauss.json + rescore_gauss.md")


if __name__ == "__main__":
    main()
