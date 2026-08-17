"""Per-task losses. All accept an optional per-sample weight `w` (B,) so the
EM-hypothesis heads (energy / concept / recon) can be gated to electrons only
once protons + PID arrive — built in now, no refactor later."""
import torch
import torch.nn.functional as F


def _weighted_mean(per_sample, w):
    if w is None:
        return per_sample.mean()
    w = w.to(per_sample.dtype)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-6)


def energy_loss(pred_std_log, target_std_log, delta=0.1, w=None):
    """Huber on the standardised log-energy residual."""
    l = F.huber_loss(pred_std_log, target_std_log, delta=delta, reduction="none")
    return _weighted_mean(l, w)


def angle_loss(pred_std_slopes, target_std_slopes, delta=0.1, w=None):
    """Huber on standardised MC (dx/dz,dy/dz), averaged per event.

    The components are standardised using training-only statistics, so X and Y
    receive equal scale. MC truth is never fit-quality down-weighted.
    """
    l = F.huber_loss(pred_std_slopes, target_std_slopes,
                     delta=delta, reduction="none").mean(dim=1)
    return _weighted_mean(l, w)


def direction_chord_loss(pred_slopes, target_slopes, epsilon=1e-3, w=None):
    """Robust opening-angle proxy on incoming unit direction vectors.

    Chord distance equals the opening angle to leading order at the milliradian
    resolutions of interest.  The Charbonnier form has finite gradients at zero
    and behaves like L1 outside ``epsilon``, making it less sensitive to the long
    shower-tail distribution than an angular MSE.
    """
    def unit(slopes):
        kx, ky = slopes.unbind(dim=-1)
        inv_norm = torch.rsqrt(1.0 + kx.square() + ky.square())
        return torch.stack((-kx * inv_norm, -ky * inv_norm, -inv_norm), dim=-1)

    diff2 = (unit(pred_slopes) - unit(target_slopes)).square().sum(dim=-1)
    eps = float(epsilon)
    per_sample = torch.sqrt(diff2 + eps * eps) - eps
    return _weighted_mean(per_sample, w)


def concept_loss(pred, target, delta=0.1, w=None):
    """Huber on standardised concepts, averaged over the concept vector."""
    l = F.huber_loss(pred, target, delta=delta, reduction="none").mean(dim=1)
    return _weighted_mean(l, w)


def recon_loss(pred, target, valid, delta=0.1, per_event_norm=True, w=None):
    """Huber on per-token log1p(expehit), summed over valid tokens then
    per-event normalised so the 1296-scale term can't swamp the scalar heads."""
    l = F.huber_loss(pred, target, delta=delta, reduction="none") * valid
    denom = valid.sum(dim=1).clamp_min(1) if per_event_norm else torch.ones_like(l[:, 0])
    per_event = l.sum(dim=1) / denom
    return _weighted_mean(per_event, w)
