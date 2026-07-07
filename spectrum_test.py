"""Spectrum-dependence test: does the learned estimator's calibration move when the
test spectrum changes? A regressor trained on one MC spectrum can exploit it as a
prior (within-bin regression to the mean); reweighting the SAME test events to
different spectral shapes and re-fitting the per-bin Gaussian core measures that
dependence directly. Method: importance-resample the test set to the target
spectrum (dN/dE ~ E^-gamma over [E_LO, E_HI]) against the empirical density, then
reuse the standard iterated +/-2 sigma fit unchanged.

    python spectrum_test.py [--preds runs/methodfig2/v3_d192_ep100_test.npz]
Outputs: runs/spectrum/spectrum_test.md + transformerReport/figs/f17_spectrum_bias.pdf
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from utils.stats import gauss_core

BASE = "/aifs/user/data/lishanglin/chenhao"
FIGS = f"{BASE}/transformerReport/figs"
BINS = [0, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 3000, 4000]
E_LO, E_HI = 15.0, 4000.0
N_RESAMPLE = 3_000_000
GAMMAS = {"hard (E^-0.3)": 0.3, "nominal MC": None, "soft (E^-2.0)": 2.0}


def per_bin(e, r, nmin=400):
    c, sig, mu = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (e >= lo) & (e < hi)
        if m.sum() < nmin:
            continue
        g = gauss_core(r[m])
        c.append(0.5 * (lo + hi)); sig.append(g["sigma"] * 100); mu.append(g["mu"] * 100)
    return np.array(c), np.array(sig), np.array(mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=f"{BASE}/ecalTransformer/runs/methodfig2/"
                                       "v3_d192_ep100_test.npz")
    args = ap.parse_args()
    z = np.load(args.preds)
    e, p = z["e_true"].astype(np.float64), z["e_pred"].astype(np.float64)
    sel = (e >= E_LO) & (e <= E_HI)
    e, p = e[sel], p[sel]
    r = (p - e) / e
    rng = np.random.default_rng(0)

    # empirical density in log E (the MC spectrum), for importance weights
    loge = np.log(e)
    hist, edges = np.histogram(loge, bins=60, density=True)
    dens = np.clip(hist[np.clip(np.digitize(loge, edges) - 1, 0, 59)], 1e-12, None)

    out_dir = f"{BASE}/ecalTransformer/runs/spectrum"
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, gamma in GAMMAS.items():
        if gamma is None:
            ee, rr = e, r
        else:
            # target pdf in logE: E * E^-gamma = E^(1-gamma)
            w = np.exp((1.0 - gamma) * loge) / dens
            w /= w.sum()
            idx = rng.choice(len(e), size=N_RESAMPLE, replace=True, p=w)
            ee, rr = e[idx], r[idx]
        results[name] = per_bin(ee, rr)
        g2 = gauss_core(rr[ee <= 2000.0])
        results[name] += (g2["sigma"] * 100, g2["mu"] * 100)
        print(f"{name:16s} gauss<=2TeV {g2['sigma']*100:.3f}%  mu {g2['mu']*100:+.3f}%")

    fig, (axS, axB) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    marks = {"hard (E^-0.3)": ("#d62728", "^"), "nominal MC": ("#2ca02c", "D"),
             "soft (E^-2.0)": ("#1f77b4", "v")}
    axB.axhline(0, color="k", lw=.8)
    for name, (c, sig, mu, le2, mule2) in results.items():
        col, mk = marks[name]
        axS.plot(c, sig, marker=mk, color=col, ms=5, lw=1.7, label=name)
        axB.plot(c, mu, marker=mk, color=col, ms=5, lw=1.7, label=name)
    for ax, ylab, ttl in [(axS, r"$\sigma_G/E$ [%]", "resolution"),
                          (axB, r"$\mu_G$ [%]", "core calibration")]:
        ax.set_xscale("log"); ax.grid(True, alpha=.3, which="both")
        ax.set_xlabel(r"$E_{\rm true}$ [GeV]"); ax.set_ylabel(ylab)
        ax.set_title(f"Spectrum dependence — {ttl} (champion, resampled test set)",
                     fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIGS}/f17_spectrum_bias.pdf"); plt.close()

    with open(f"{out_dir}/spectrum_test.md", "w") as f:
        f.write("# Spectrum-dependence test (champion v3_d192_ep100, test set "
                f"resampled over [{E_LO:.0f},{E_HI:.0f}] GeV)\n\n"
                "| spectrum | gauss<=2TeV | mu<=2TeV | per-bin mu (%) |\n|---|---|---|---|\n")
        for name, (c, sig, mu, le2, mule2) in results.items():
            pb = " ".join(f"{ci:.0f}:{m:+.2f}" for ci, m in zip(c, mu))
            f.write(f"| {name} | {le2:.3f}% | {mule2:+.3f}% | {pb} |\n")
    print(f"wrote {out_dir}/spectrum_test.md + {FIGS}/f17_spectrum_bias.pdf")


if __name__ == "__main__":
    main()
