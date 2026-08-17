"""Task heads and shared learned-query pooling."""
import math

import torch
import torch.nn as nn

from data.geometry import COMPONENT_VIEWS


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


class AngleHead(nn.Module):
    """Global incidence direction -> standardised (dx/dz, dy/dz)."""

    view_aware = False

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.mlp = make_mlp(d_model, hidden, 2, dropout)

    def forward(self, tokens, valid):
        return self.mlp(self.pool(tokens, valid))                # (B,2)


class ViewAngleHead(nn.Module):
    """Direction head respecting ECAL's two orthogonal strip projections.

    Stored view 1 measures dx/dz and stored view 0 measures dy/dz. Independent
    learned-query pools prevent the two
    projections from competing for one attention distribution while the shared
    encoder can still exchange information between all layers.
    """

    view_aware = True

    def __init__(self, d_model, hidden, dropout=0.0, component_views=COMPONENT_VIEWS):
        super().__init__()
        self.component_views = tuple(int(v) for v in component_views)
        if sorted(self.component_views) != [0, 1]:
            raise ValueError("angle component_views must be a permutation of [0,1]")
        self.pool_x = AttnPool(d_model)
        self.pool_y = AttnPool(d_model)
        self.mlp_x = make_mlp(d_model, hidden, 1, dropout)
        self.mlp_y = make_mlp(d_model, hidden, 1, dropout)

    def forward(self, tokens, valid_view0, valid_view1):
        masks = (valid_view0, valid_view1)
        kx = self.mlp_x(self.pool_x(tokens, masks[self.component_views[0]]))
        ky = self.mlp_y(self.pool_y(tokens, masks[self.component_views[1]]))
        return torch.cat([kx, ky], dim=-1)                       # (B,2)


class ResidualViewAngleHead(ViewAngleHead):
    """Standardized correction to an analytic layer-centroid shower axis."""

    residual = True

    def __init__(self, d_model, hidden, dropout=0.0, component_views=COMPONENT_VIEWS):
        super().__init__(d_model, hidden, dropout, component_views)
        # Start exactly at the analytic direction.  After the first update the
        # correction network learns detector/shower-development biases.
        for mlp in (self.mlp_x, self.mlp_y):
            final = mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


# ---- Remaining reserved heads -------------------------------------------
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
    if cfg_heads.angle.enabled:
        angle_type = str(cfg_heads.angle.get("type", "global")).lower()
        component_views = cfg_heads.angle.get("component_views", list(COMPONENT_VIEWS))
        if angle_type == "global":
            heads["angle"] = AngleHead(d_model, cfg_heads.angle.hidden)
        elif angle_type == "view":
            heads["angle"] = ViewAngleHead(
                d_model, cfg_heads.angle.hidden, component_views=component_views)
        elif angle_type == "view_residual":
            heads["angle"] = ResidualViewAngleHead(
                d_model, cfg_heads.angle.hidden, component_views=component_views)
        else:
            raise ValueError(
                f"unknown heads.angle.type={angle_type!r}; "
                "use global, view, or view_residual")
    for name in ("position", "pid"):
        if getattr(cfg_heads, name).enabled:
            heads[name] = _ReservedHead(name)
    return heads
