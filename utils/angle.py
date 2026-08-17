"""Direction-coordinate conversions and angular-resolution metrics.

The regression representation is (kx, ky) = (dx/dz, dy/dz). All MC particles in
this sample enter the ECAL toward decreasing z, so the incoming unit vector is
(-kx, -ky, -1) / sqrt(1 + kx**2 + ky**2). This avoids the phi wrap at +/-pi.
"""
import numpy as np
import torch


def slopes_to_unit_np(slopes):
    slopes = np.asarray(slopes, dtype=np.float64)
    v = np.concatenate([-slopes, -np.ones((*slopes.shape[:-1], 1))], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def slopes_to_unit_torch(slopes):
    v = torch.cat([-slopes, -torch.ones_like(slopes[..., :1])], dim=-1)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)


def angular_error_np(pred_slopes, true_slopes):
    pred = slopes_to_unit_np(pred_slopes)
    truth = slopes_to_unit_np(true_slopes)
    dot = np.sum(pred * truth, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def slope_to_theta_phi_np(slopes):
    unit = slopes_to_unit_np(slopes)
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    return theta, phi


def containment_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        raise ValueError("no finite angular errors")
    q = np.quantile(errors, [0.50, 0.68, 0.90, 0.95])
    return {
        "median": float(q[0]),
        "p68": float(q[1]),
        "p90": float(q[2]),
        "p95": float(q[3]),
        "mean": float(np.mean(errors)),
        "rms": float(np.sqrt(np.mean(errors ** 2))),
        "count": int(errors.size),
    }
