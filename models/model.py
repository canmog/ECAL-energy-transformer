"""Full model: embed -> lower blocks -> soft bottleneck -> upper blocks -> heads.

forward() returns a dict:
    energy      (B,)      standardised log-energy prediction
    recon       (B,L)     per-token log1p(expehit) prediction
    concepts    (B,C)     bottleneck concept prediction
    pooled_phys (B,d_phys) pooled physics sub-space  (for the linear probe)
    h_phys      (B,L,d_phys) per-token physics sub-space
"""
import torch
import torch.nn as nn

from data.dataset import TOKEN_FEATURE_DIM
from models.embedding import TokenEmbedding
from models.encoder import BlockStack
from models.bottleneck import SoftBottleneck
from models.heads import build_heads


class EcalTransformer(nn.Module):
    def __init__(self, cfg, n_concepts):
        super().__init__()
        m = cfg.model
        self.embed = TokenEmbedding(TOKEN_FEATURE_DIM, m.d_model, m.dropout, m.pos_embed)
        self.lower = BlockStack(m.tap_block, m.d_model, m.n_heads, m.ffn_mult, m.dropout)
        self.bottleneck = SoftBottleneck(
            m.d_model, m.d_phys, m.d_free, n_concepts, m.dropout,
            residual_bypass=m.get("bottleneck_residual", True),
            energy_phys_head=m.get("dual_energy_head", False))
        self.upper = BlockStack(m.n_blocks - m.tap_block, m.d_model, m.n_heads, m.ffn_mult, m.dropout)
        self.heads = build_heads(cfg.heads, m.d_model)

        self.register_buffer("log_mean", torch.zeros(1))
        self.register_buffer("log_std", torch.ones(1))

    def set_energy_norm(self, mean, std):
        self.log_mean.fill_(float(mean))
        self.log_std.fill_(float(std))

    def predict_energy_gev(self, std_log_e):
        return torch.exp(self.log_mean + self.log_std * std_log_e)

    def forward(self, batch, phys_scale=1.0, free_scale=1.0):
        valid = batch["valid"]
        x = self.embed(batch["feats"], batch["pos_id"])
        x = self.lower(x, valid)
        x, h_phys, h_free, pooled_phys, concepts, e_phys = self.bottleneck(
            x, valid, phys_scale, free_scale)
        x = self.upper(x, valid)

        out = {"concepts": concepts, "pooled_phys": pooled_phys,
               "h_phys": h_phys, "h_free": h_free, "valid": valid}
        if "energy" in self.heads:
            e_free = self.heads["energy"](x, valid)        # deep correction (full depth)
            if e_phys is not None:                          # dual head: energy = phys + free
                out["e_phys"], out["e_free"] = e_phys, e_free
                out["energy"] = e_phys + e_free
            else:
                out["energy"] = e_free
        if "recon" in self.heads:
            out["recon"] = self.heads["recon"](x, valid)
        return out
