"""Full model: embed -> lower blocks -> soft bottleneck -> upper blocks -> heads.

forward() returns a dict:
    energy      (B,)      standardised log-energy prediction
    recon       (B,L)     per-token log1p(expehit) prediction
    concepts    (B,C)     bottleneck concept prediction
    angle       (B,2)     standardised MC (dx/dz,dy/dz) prediction
    pooled_phys (B,d_phys) pooled physics sub-space  (for the linear probe)
    h_phys      (B,L,d_phys) per-token physics sub-space
"""
import torch
import torch.nn as nn

from data.dataset import TOKEN_FEATURE_DIM
from data.geometry import COMPONENT_VIEWS
from models.embedding import TokenEmbedding
from models.encoder import BlockStack
from models.bottleneck import SoftBottleneck
from models.heads import build_heads
from models.physics import LayerCentroidSlope, LearnedRobustLayerSlope


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
        angle_head = self.heads["angle"] if "angle" in self.heads else None
        if angle_head is not None and getattr(angle_head, "residual", False):
            angle_cfg = cfg.heads.angle
            slope_type = (LearnedRobustLayerSlope
                          if angle_cfg.get("learned_robust_fit", False)
                          else LayerCentroidSlope)
            slope_kwargs = {}
            if slope_type is LearnedRobustLayerSlope:
                slope_kwargs = {
                    "hidden": angle_cfg.get("robust_fit_hidden", 16),
                    "max_weight_multiplier": angle_cfg.get(
                        "robust_fit_max_multiplier", 4.0),
                }
            self.centroid_slope = slope_type(
                cfg.geometry.data_type,
                angle_cfg.get("centroid_weight_power", 0.5),
                angle_cfg.get("component_views", list(COMPONENT_VIEWS)),
                **slope_kwargs)
            self.angle_residual_scale = float(
                cfg.heads.angle.get("residual_scale", 1.0))
            self.residual_normalization = bool(
                cfg.heads.angle.get("residual_normalization", False))
            if self.residual_normalization:
                if self.angle_residual_scale != 1.0:
                    raise ValueError(
                        "residual_scale must be 1.0 with residual_normalization")
                self.register_buffer("angle_residual_mean", torch.zeros(2))
                self.register_buffer("angle_residual_std", torch.ones(2))
        else:
            self.centroid_slope = None
            self.residual_normalization = False

        self.register_buffer("log_mean", torch.zeros(1))
        self.register_buffer("log_std", torch.ones(1))
        # Do not add angle keys to energy-only state_dicts. This preserves strict
        # loading of every checkpoint created before the angle head existed.
        if angle_head is not None:
            self.register_buffer("angle_mean", torch.zeros(2))
            self.register_buffer("angle_std", torch.ones(2))
        else:
            self.angle_mean = None
            self.angle_std = None

    def set_energy_norm(self, mean, std):
        self.log_mean.fill_(float(mean))
        self.log_std.fill_(float(std))

    def predict_energy_gev(self, std_log_e):
        return torch.exp(self.log_mean + self.log_std * std_log_e)

    def set_angle_norm(self, mean, std):
        if self.angle_mean is None:
            raise RuntimeError("cannot set angle normalization: angle head is disabled")
        self.angle_mean.copy_(torch.as_tensor(mean, dtype=self.angle_mean.dtype))
        self.angle_std.copy_(torch.as_tensor(std, dtype=self.angle_std.dtype))

    def set_angle_residual_norm(self, mean, std):
        if not self.residual_normalization:
            raise RuntimeError("angle residual normalization is not enabled")
        self.angle_residual_mean.copy_(
            torch.as_tensor(mean, dtype=self.angle_residual_mean.dtype))
        self.angle_residual_std.copy_(
            torch.as_tensor(std, dtype=self.angle_residual_std.dtype))

    def predict_angle_slopes(self, std_angle):
        if self.angle_mean is None:
            raise RuntimeError("cannot predict angle slopes: angle head is disabled")
        return self.angle_mean + self.angle_std * std_angle

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
        if "angle" in self.heads:
            angle_head = self.heads["angle"]
            if getattr(angle_head, "view_aware", False):
                # feats = [logE, t, z, depth, stored_view_0, stored_view_1]
                valid_view0 = valid & (batch["feats"][..., 4] > 0.5)
                valid_view1 = valid & (batch["feats"][..., 5] > 0.5)
                angle_prediction = angle_head(x, valid_view0, valid_view1)
            else:
                angle_prediction = angle_head(x, valid)
            if self.centroid_slope is not None:
                baseline = self.centroid_slope(batch)
                out["angle_baseline"] = baseline
                if self.residual_normalization:
                    residual = (self.angle_residual_mean
                                + self.angle_residual_std * angle_prediction)
                    slopes = baseline + self.angle_residual_scale * residual
                    out["angle_residual_std"] = angle_prediction
                    out["angle_residual"] = residual
                    out["angle_residual_mean"] = self.angle_residual_mean
                    out["angle_residual_scale"] = self.angle_residual_std
                    out["angle_slopes"] = slopes
                    out["angle"] = (slopes - self.angle_mean) / self.angle_std
                else:
                    baseline_std = (baseline - self.angle_mean) / self.angle_std
                    out["angle"] = (baseline_std
                                    + self.angle_residual_scale * angle_prediction)
            else:
                out["angle"] = angle_prediction
            # Physical slopes are exposed for losses that optimize the actual
            # 3D direction rather than standardized slope components.
            if "angle_slopes" not in out:
                out["angle_slopes"] = self.predict_angle_slopes(out["angle"])
        return out
