"""Resolution estimators for the relative residual r = (E_pred - E_true)/E_true.

gauss_core(): the calorimetry-standard GAUSSIAN CORE sigma — a binned Gaussian
fit restricted to mu +/- n_sigma*sigma, iterated until the fitted sigma is
stable. This estimates the width of the Gaussian core directly (what a ROOT
`Gaus` fit in a +/-2 sigma range gives), unlike IQR/1.349 which is exactly
Gaussian-consistent only for a pure Gaussian and biases HIGH when the +/-25%
quantiles already touch non-Gaussian shoulders. IQR/1.349 is kept everywhere
as the never-fails seed and cross-check.

Fallback chain (fit can fail on tiny/pathological samples, e.g. epoch 0):
    scipy curve_fit on the histogram  ->  corrected truncated moments  ->  IQR/1.349
The truncated-moments correction: for a true Gaussian, the sample std inside
mu +/- 2 sigma underestimates sigma by sqrt(1 - 4*phi(2)/(2*Phi(2)-1)) = 0.8796,
so we divide it back out each iteration (otherwise the window shrinks to zero).
"""
import numpy as np

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False

# std of a unit Gaussian truncated at +/- 2 sigma (see module docstring)
_TRUNC2 = 0.879626


def robust_sigma(x):
    """IQR/1.349 core width (tail-insensitive, never fails)."""
    q75, q25 = np.percentile(x, [75, 25])
    return float((q75 - q25) / 1.349)


def _fit_hist_gauss(w, mu0, sig0, nbins):
    """One binned LSQ Gaussian fit on the window sample w. Returns (mu, sigma)."""
    hist, edges = np.histogram(w, bins=nbins)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m = hist > 0
    if m.sum() < 5:
        raise RuntimeError("too few populated bins")
    gauss = lambda x, a, mu, sig: a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    p, _ = curve_fit(gauss, ctr[m], hist[m], p0=[hist.max(), mu0, sig0],
                     sigma=np.sqrt(np.clip(hist[m], 1, None)), maxfev=2000)
    return float(p[1]), float(abs(p[2]))


def gauss_core(x, n_sigma=2.0, tol=1e-3, max_iter=20, nbins=80, min_events=200):
    """Iterative Gaussian-core fit in mu +/- n_sigma*sigma.

    Returns dict(sigma, mu, ok, n_iter, method). Seeds from (median, IQR/1.349);
    iterates fit -> re-window -> fit until |d sigma| < tol*sigma or max_iter.
    ok=False means every fit attempt fell through and sigma is the robust seed.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    mu, sig = (float(np.median(x)), robust_sigma(x)) if x.size else (0.0, 0.0)
    if x.size < min_events or sig <= 0:
        return {"sigma": sig, "mu": mu, "ok": False, "n_iter": 0, "method": "robust"}

    method = "fit" if _HAVE_SCIPY else "trunc_moments"
    for it in range(1, max_iter + 1):
        w = x[(x >= mu - n_sigma * sig) & (x <= mu + n_sigma * sig)]
        if w.size < min_events:
            return {"sigma": robust_sigma(x), "mu": float(np.median(x)),
                    "ok": False, "n_iter": it, "method": "robust"}
        if method == "fit":
            try:
                mu_new, sig_new = _fit_hist_gauss(w, mu, sig, nbins)
                # reject a diverged fit (window escape); fall back to moments
                if not (0.2 * sig <= sig_new <= 5.0 * sig):
                    raise RuntimeError(f"fit sigma {sig_new:.3g} escaped window")
            except Exception:
                method = "trunc_moments"
                continue
        else:
            mu_new = float(np.mean(w))
            sig_new = float(np.std(w)) / _TRUNC2
        done = abs(sig_new - sig) < tol * max(sig, 1e-12)
        mu, sig = mu_new, sig_new
        if done:
            return {"sigma": sig, "mu": mu, "ok": True, "n_iter": it, "method": method}
    return {"sigma": sig, "mu": mu, "ok": True, "n_iter": max_iter, "method": method}
