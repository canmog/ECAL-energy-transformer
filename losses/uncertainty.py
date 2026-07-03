"""Homoscedastic uncertainty weighting (Kendall & Gal, CVPR 2018).

Each task gets a learnable log-variance s = log(sigma^2); the network balances the
heads automatically instead of hand-tuned lambdas:

    regression:      L += 0.5 * exp(-s) * L_task + 0.5 * s
    classification:  L +=       exp(-s) * L_task + 0.5 * s   (RESERVED, for PID)

The + s term penalises a task for hiding behind large uncertainty, preventing the
trivial all-weights -> 0 collapse. New heads (angle/position/pid) drop in by name.
"""
import torch
import torch.nn as nn


class UncertaintyWeighter(nn.Module):
    """Kendall-Gal homoscedastic weighting, stabilised against the weight runaway.

    Vanilla Kendall-Gal converges to weight = exp(-s) = 1/loss, which is UNBOUNDED:
    on a large dataset a task whose average loss gets tiny (energy, or the
    near-deterministic recon target) earns a huge weight (observed w_energy -> 421,
    w_recon -> 156), and that factor under a fixed grad-clip makes training bounce.
    Two stabilisers, instead of a hard clamp:

      * loss normalisation — divide each task loss by a running EMA of its own
        magnitude before weighting, so the weights track RELATIVE difficulty, not
        absolute scale; a tiny-loss task can no longer earn an unbounded weight.
      * mild weight decay on the log-variances (applied in train.py via
        loss.logvar_weight_decay) — softly pulls s toward 0 so any residual drift
        self-limits.
    """

    def __init__(self, task_names, kinds=None, norm_momentum=0.99):
        super().__init__()
        self.kinds = kinds or {t: "regression" for t in task_names}
        self.log_var = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in task_names})
        self.norm_momentum = float(norm_momentum)
        # running EMA of each task's raw loss magnitude; 0 = not yet initialised
        for t in task_names:
            self.register_buffer(f"ema_{t}", torch.zeros(()))

    def forward(self, losses):
        """losses: dict[name] -> scalar tensor. Returns (total, log_dict).
        forward() is only called in the training loop, so the EMA updates here."""
        total = torch.zeros((), device=next(iter(losses.values())).device)
        logs = {}
        m = self.norm_momentum
        for name, l in losses.items():
            ema = getattr(self, f"ema_{name}")
            if float(ema) == 0.0:                    # first step: seed at true scale
                ema.copy_(l.detach())
            else:
                ema.mul_(m).add_((1.0 - m) * l.detach())
            l_norm = l / (ema + 1e-8)                # ~O(1); gradient still flows via l
            s = self.log_var[name]
            factor = 0.5 if self.kinds.get(name, "regression") == "regression" else 1.0
            total = total + factor * torch.exp(-s) * l_norm + 0.5 * s
            logs[f"w_{name}"] = float(torch.exp(-s).detach())
            logs[f"loss_{name}"] = float(l.detach())
        return total, logs

    def task_weights(self):
        return {t: float(torch.exp(-s).detach()) for t, s in self.log_var.items()}


class NormalizedWeighter(nn.Module):
    """EMA-normalize each task loss to O(1), then sum with FIXED manual weights.

    Keeps the half of Kendall-Gal that works (the running loss-normalisation that puts
    every task on a common scale) and drops the learnable log-variance `s` — the part
    that ran away (w_energy -> 450 on large data). With no `s` there is nothing to
    diverge: the normalised gradient is just the log-loss gradient, O(1) as a task is
    learned. Weights express a DELIBERATE priority (e.g. energy 1.0, aux 0.3) on the
    normalised losses, so the numbers are scale-free, not magnitude hacks. Missing
    tasks default to weight 1.0. Same (total, log_dict) interface as the others."""

    def __init__(self, task_names, weights=None, norm_momentum=0.99):
        super().__init__()
        self.tasks = list(task_names)
        self.norm_momentum = float(norm_momentum)
        weights = weights or {}
        self.weights = {t: float(weights.get(t, 1.0)) for t in task_names}
        for t in task_names:
            self.register_buffer(f"ema_{t}", torch.zeros(()))

    def forward(self, losses):
        total = torch.zeros((), device=next(iter(losses.values())).device)
        logs = {}
        m = self.norm_momentum
        for name, l in losses.items():
            ema = getattr(self, f"ema_{name}")
            if float(ema) == 0.0:                    # first step: seed at true scale
                ema.copy_(l.detach())
            else:
                ema.mul_(m).add_((1.0 - m) * l.detach())
            w = self.weights.get(name, 1.0)
            total = total + w * l / (ema + 1e-8)     # gradient still flows via l
            logs[f"w_{name}"] = w
            logs[f"loss_{name}"] = float(l.detach())
        return total, logs

    def task_weights(self):
        return dict(self.weights)


class FixedWeighter(nn.Module):
    """Unweighted sum of task losses (loss.uncertainty_weighting: false ablation).
    Same interface as UncertaintyWeighter, no learnable parameters."""

    def __init__(self, task_names, kinds=None):
        super().__init__()
        self.tasks = list(task_names)

    def forward(self, losses):
        total = sum(losses.values())
        logs = {}
        for name, l in losses.items():
            logs[f"w_{name}"] = 1.0
            logs[f"loss_{name}"] = float(l.detach())
        return total, logs

    def task_weights(self):
        return {t: 1.0 for t in self.tasks}
