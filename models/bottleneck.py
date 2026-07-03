"""Soft concept bottleneck (the physics tap).

At an intermediate layer the token representation is split into a supervised
physics sub-space h_phys (probe this) and a free residual h_free, recombined for
the upper blocks. A concept head reads the pooled h_phys and is trained against
the 3D-fit parameters. Soft = h_free keeps the network's freedom to beat the fit;
the supervision only grounds part of the representation in physics.
"""
import torch
import torch.nn as nn

from models.heads import AttnPool, make_mlp


class SoftBottleneck(nn.Module):
    def __init__(self, d_model, d_phys, d_free, n_concepts, dropout=0.0,
                 residual_bypass=True, energy_phys_head=False):
        super().__init__()
        self.to_phys = nn.Linear(d_model, d_phys)
        self.to_free = nn.Linear(d_model, d_free)
        self.recombine = nn.Linear(d_phys + d_free, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.concept_pool = AttnPool(d_phys)
        self.concept_head = make_mlp(d_phys, [d_phys], n_concepts, dropout)
        # residual_bypass=False removes the `tokens +` skip so the upper blocks see
        # ONLY recombine(h_phys, h_free) — the concept-shaped physics sub-space can no
        # longer be routed around (the energy/recon stream is forced through the split).
        self.residual_bypass = residual_bypass
        # Physics energy baseline read off the SAME pooled vector that predicts the
        # concepts -> the physics energy is forced to share the 3D-fit-grounded code.
        # The full energy is e_phys + e_free (e_free = the deep, full-depth head in
        # model.py); see the dual-head design in models/model.py.
        self.e_phys_head = (make_mlp(d_phys, [max(d_phys // 2, 8)], 1, dropout)
                            if energy_phys_head else None)

    def forward(self, tokens, valid, phys_scale=1.0, free_scale=1.0):
        h_phys = self.to_phys(tokens)                      # (B,L,d_phys)
        h_free = self.to_free(tokens)                      # (B,L,d_free)
        # phys_scale / free_scale: 1.0 in normal use; the probe sets one to 0 to
        # causally ablate a sub-space and measure its effect on the energy head.
        x = self.recombine(torch.cat([phys_scale * h_phys, free_scale * h_free], dim=-1))
        x = self.norm(tokens + x) if self.residual_bypass else self.norm(x)
        pooled_phys = self.concept_pool(h_phys, valid)     # (B,d_phys) — always unscaled
        concepts = self.concept_head(pooled_phys)          # (B,n_concepts)
        e_phys = (self.e_phys_head(pooled_phys).squeeze(-1)
                  if self.e_phys_head is not None else None)   # (B,) physics energy or None
        return x, h_phys, h_free, pooled_phys, concepts, e_phys
