"""Cell-token embedding: physical features -> d_model, plus a learned (layer,cell)
positional embedding. Geometry is already in the features (t, z, view, depth);
the positional embedding adds a learnable per-readout-channel offset."""
import torch
import torch.nn as nn

from data.geometry import N_LAYER, N_CELL


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.0, pos_embed=True):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed = pos_embed
        if pos_embed:
            self.pos = nn.Embedding(N_LAYER * N_CELL, d_model)
            nn.init.normal_(self.pos.weight, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, feats, pos_id):
        x = self.feat(feats)
        if self.pos_embed:
            x = x + self.pos(pos_id)
        return self.drop(self.norm(x))
