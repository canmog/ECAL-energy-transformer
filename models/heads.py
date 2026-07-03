"""Task heads + shared pooling. energy + recon are built; angle/position/pid are
RESERVED registry slots (config-gated, raise if enabled this round)."""
import math

import torch
import torch.nn as nn


def make_mlp(d_in, hidden, d_out, dropout=0.0):
    dims = [d_in] + list(hidden)
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-1], d_out)]
    return nn.Sequential(*layers)


class AttnPool(nn.Module):
    """Single learned-query attention pooling over valid tokens."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, x, valid):                      # x (B,L,d), valid (B,L) bool
        score = (self.k(x) @ self.q) * self.scale     # (B,L)
        score = score.masked_fill(~valid, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)  # (B,L,1)
        return (w * self.v(x)).sum(dim=1)             # (B,d)


class EnergyHead(nn.Module):
    """Global energy convergence head -> standardised log-energy (scalar)."""

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.mlp = make_mlp(d_model, hidden, 1, dropout)

    def forward(self, tokens, valid):
        return self.mlp(self.pool(tokens, valid)).squeeze(-1)   # (B,)


class ReconHead(nn.Module):
    """Micro-unit reconstruction head -> per-token log1p(expehit)."""

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.mlp = make_mlp(d_model, hidden, 1, dropout)

    def forward(self, tokens, valid):
        return self.mlp(tokens).squeeze(-1)                     # (B,L)


# ---- RESERVED heads (not implemented this round) -------------------------
class _ReservedHead(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, *a, **k):
        raise NotImplementedError(f"head '{self.name}' is a reserved interface — not built")


def build_heads(cfg_heads, d_model):
    heads = nn.ModuleDict()
    if cfg_heads.energy.enabled:
        heads["energy"] = EnergyHead(d_model, cfg_heads.energy.hidden)
    if cfg_heads.recon.enabled:
        heads["recon"] = ReconHead(d_model, cfg_heads.recon.hidden)
    for name in ("angle", "position", "pid"):
        if getattr(cfg_heads, name).enabled:
            heads[name] = _ReservedHead(name)
    return heads
