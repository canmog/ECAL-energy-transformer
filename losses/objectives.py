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
