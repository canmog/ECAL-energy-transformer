"""Differentiable ECAL shower-axis estimates used as neural starting points."""
import math

import torch
import torch.nn as nn

from data.geometry import (COMPONENT_VIEWS, N_CELL, N_LAYER, PITCH_CM,
                           build_geometry_table)


def weighted_line_slope(z, transverse, weight, epsilon=1e-8):
    """Weighted least-squares d(transverse)/dz for batched layer summaries."""
    norm = weight.sum(dim=1, keepdim=True).clamp_min(epsilon)
    z_mean = (weight * z).sum(dim=1, keepdim=True) / norm
    t_mean = (weight * transverse).sum(dim=1, keepdim=True) / norm
    dz = z - z_mean
    numerator = (weight * dz * (transverse - t_mean)).sum(dim=1)
    denominator = (weight * dz.square()).sum(dim=1).clamp_min(epsilon)
    return numerator / denominator


def weighted_line(z, transverse, weight, epsilon=1e-8):
    """Weighted slope and intercept for batched layer summaries."""
    norm = weight.sum(dim=1, keepdim=True).clamp_min(epsilon)
    z_mean = (weight * z).sum(dim=1, keepdim=True) / norm
    t_mean = (weight * transverse).sum(dim=1, keepdim=True) / norm
    dz = z - z_mean
    slope = ((weight * dz * (transverse - t_mean)).sum(dim=1)
             / (weight * dz.square()).sum(dim=1).clamp_min(epsilon))
    intercept = t_mean.squeeze(1) - slope * z_mean.squeeze(1)
    return slope, intercept


class LayerCentroidSlope(nn.Module):
    """Energy-centroid line fit in the independent X and Y layer projections.

    Cell energies are recovered from the input log1p(E/MeV) feature.  Cells are
    first reduced to one physical centroid per readout layer; a weighted line is
    then fit versus the exact layer z coordinate.  The calculation is forced to
    fp32 even during bf16 model training because it is a geometric baseline, not
    a learned activation.
    """

    def __init__(self, data_type="MC", layer_weight_power=0.5,
                 component_views=COMPONENT_VIEWS):
        super().__init__()
        table = build_geometry_table(data_type)
        self.register_buffer(
            "cell_t_cm", torch.as_tensor(table["t_cm"].reshape(-1), dtype=torch.float32))
        self.register_buffer(
            "layer_z_cm", torch.as_tensor(table["z_cm"][:, 0], dtype=torch.float32))
        self.register_buffer(
            "layer_view", torch.as_tensor(table["view"][:, 0], dtype=torch.long))
        self.layer_weight_power = float(layer_weight_power)
        self.component_views = tuple(int(v) for v in component_views)
        if sorted(self.component_views) != [0, 1]:
            raise ValueError("component_views must be a permutation of [0,1]")

    def forward(self, batch):
        device_type = batch["feats"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            feats = batch["feats"].float()
            valid = batch["valid"]
            pos_id = batch["pos_id"]
            layer = torch.div(pos_id, N_CELL, rounding_mode="floor")
            energy = torch.expm1(feats[..., 0]).clamp_min(0.0) * valid
            transverse = self.cell_t_cm[pos_id]

            shape = (feats.shape[0], N_LAYER)
            layer_energy = torch.zeros(shape, device=feats.device, dtype=torch.float32)
            layer_et = torch.zeros_like(layer_energy)
            layer_energy.scatter_add_(1, layer, energy)
            layer_et.scatter_add_(1, layer, energy * transverse)
            present = layer_energy > 0.0
            centroid = layer_et / layer_energy.clamp_min(1e-8)

            # Sub-linear energy weights keep shower-maximum layers reliable while
            # preserving the long z lever arm of lower-energy entrance/tail layers.
            regression_weight = layer_energy.clamp_min(0.0).pow(
                self.layer_weight_power) * present
            z = self.layer_z_cm.unsqueeze(0).expand_as(centroid)
            slopes = []
            for view in (0, 1):
                mask = (self.layer_view == view).unsqueeze(0)
                slopes.append(weighted_line_slope(
                    z, centroid, regression_weight * mask))
            by_view = torch.stack(slopes, dim=-1)
            return by_view[:, list(self.component_views)]


class LearnedRobustLayerSlope(LayerCentroidSlope):
    """Two-pass layer-centroid fit with learned ECAL-only reliability weights.

    The first pass is exactly :class:`LayerCentroidSlope`.  A small MLP shared by
    all 18 readout layers then sees only layer summaries and the first-fit
    residual, and returns a bounded multiplier on the original E**power weight.
    Its last layer is zero-initialized, so the initial multiplier is exactly one
    and the module starts numerically at the fixed J1 baseline.
    """

    def __init__(self, data_type="MC", layer_weight_power=0.5,
                 component_views=COMPONENT_VIEWS, hidden=16,
                 max_weight_multiplier=4.0):
        super().__init__(data_type, layer_weight_power, component_views)
        hidden = int(hidden)
        self.max_log_multiplier = math.log(float(max_weight_multiplier))
        if hidden <= 0 or max_weight_multiplier <= 1.0:
            raise ValueError("robust-fit hidden must be >0 and max multiplier >1")
        # [relative logE, width, occupancy, max fraction, depth, first-fit pull]
        self.reliability = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    @staticmethod
    def _scatter_sum(target, layer, source):
        target.scatter_add_(1, layer, source)

    def forward(self, batch):
        device_type = batch["feats"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            feats = batch["feats"].float()
            valid = batch["valid"]
            pos_id = batch["pos_id"]
            layer = torch.div(pos_id, N_CELL, rounding_mode="floor")
            energy = torch.expm1(feats[..., 0]).clamp_min(0.0) * valid
            transverse = self.cell_t_cm[pos_id]

            shape = (feats.shape[0], N_LAYER)
            layer_energy = torch.zeros(shape, device=feats.device, dtype=torch.float32)
            layer_et = torch.zeros_like(layer_energy)
            layer_et2 = torch.zeros_like(layer_energy)
            layer_count = torch.zeros_like(layer_energy)
            layer_max = torch.zeros_like(layer_energy)
            self._scatter_sum(layer_energy, layer, energy)
            self._scatter_sum(layer_et, layer, energy * transverse)
            self._scatter_sum(layer_et2, layer, energy * transverse.square())
            self._scatter_sum(layer_count, layer, valid.float())
            layer_max.scatter_reduce_(
                1, layer, energy, reduce="amax", include_self=True)

            present = layer_energy > 0.0
            centroid = layer_et / layer_energy.clamp_min(1e-8)
            variance = (layer_et2 / layer_energy.clamp_min(1e-8)
                        - centroid.square()).clamp_min(0.0)
            width = torch.sqrt(variance)
            base_weight = layer_energy.pow(self.layer_weight_power) * present
            z = self.layer_z_cm.unsqueeze(0).expand_as(centroid)

            # First-pass line and residual, separately in the two measured views.
            first_prediction = torch.zeros_like(centroid)
            for view in (0, 1):
                mask = (self.layer_view == view).unsqueeze(0)
                slope, intercept = weighted_line(
                    z, centroid, base_weight * mask)
                first_prediction = torch.where(
                    mask, intercept.unsqueeze(1) + slope.unsqueeze(1) * z,
                    first_prediction)
            first_pull = (centroid - first_prediction).abs() / (width + PITCH_CM)

            total_energy = layer_energy.sum(dim=1, keepdim=True).clamp_min(1e-8)
            relative_energy = layer_energy * float(N_LAYER) / total_energy
            depth = torch.linspace(
                -1.0, 1.0, N_LAYER, device=feats.device,
                dtype=torch.float32).unsqueeze(0).expand_as(centroid)
            summary = torch.stack([
                torch.log1p(relative_energy),
                width / (4.0 * PITCH_CM),
                layer_count / float(N_CELL),
                layer_max / layer_energy.clamp_min(1e-8),
                depth,
                first_pull,
            ], dim=-1)
            score = self.reliability(summary).squeeze(-1)
            multiplier = torch.exp(
                torch.tanh(score) * self.max_log_multiplier) * present
            robust_weight = base_weight * multiplier

            slopes = []
            for view in (0, 1):
                mask = (self.layer_view == view).unsqueeze(0)
                slopes.append(weighted_line_slope(
                    z, centroid, robust_weight * mask))
            by_view = torch.stack(slopes, dim=-1)
            return by_view[:, list(self.component_views)]
