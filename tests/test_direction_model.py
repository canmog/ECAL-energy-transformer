import unittest

import numpy as np
import torch

from data.geometry import COMPONENT_OF_VIEW, COMPONENT_VIEWS
from losses.objectives import direction_chord_loss
from models.model import EcalTransformer
from models.physics import LayerCentroidSlope, weighted_line_slope
from utils.angle import angular_error_np, slope_to_theta_phi_np, slopes_to_unit_np
from utils.config import load_config
from utils.tasks import active_tasks, cache_fields_for_tasks, validate_task_heads


class DirectionModelTest(unittest.TestCase):
    def small_config(self, extra=()):
        overrides = [
            "model.d_model=16", "model.n_blocks=2", "model.n_heads=4",
            "model.tap_block=1", "model.d_phys=8", "model.d_free=8",
            "model.pos_embed=false", "heads.energy.enabled=false",
            "heads.recon.enabled=false", "heads.angle.enabled=true",
            "heads.angle.type=view_residual", "heads.angle.hidden=[8]",
            "heads.angle.component_views=[1,0]",
            "heads.angle.residual_normalization=true",
            "heads.angle.residual_scale=1.0",
            "task.mode=angle",
        ]
        return load_config("config/base.yaml", overrides + list(extra))[0]

    def test_corrected_projection_convention(self):
        self.assertEqual(COMPONENT_VIEWS, (1, 0))
        self.assertEqual(COMPONENT_OF_VIEW, (1, 0))

    def test_energy_only_state_dict_remains_angle_free(self):
        cfg, _ = load_config("config/base.yaml")
        model = EcalTransformer(cfg, n_concepts=11)
        keys = set(model.state_dict())
        self.assertFalse(any("angle" in key for key in keys))
        self.assertFalse(any("centroid_slope" in key for key in keys))

    def test_normalized_residual_starts_at_calibrated_centroid(self):
        model = EcalTransformer(self.small_config(), n_concepts=2).eval()
        model.set_angle_norm([0.0, 0.0], [1.0, 1.0])
        center = torch.tensor([0.012, -0.023])
        model.set_angle_residual_norm(center, [0.04, 0.05])
        feats = torch.zeros(1, 2, 6)
        feats[0, 0, 4] = 1.0
        feats[0, 1, 5] = 1.0
        batch = {
            "feats": feats,
            "pos_id": torch.tensor([[0, 2 * 72]], dtype=torch.long),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        }
        with torch.no_grad():
            output = model(batch)
        torch.testing.assert_close(output["angle_baseline"], torch.zeros(1, 2))
        torch.testing.assert_close(output["angle_slopes"], center.unsqueeze(0))

    def test_weighted_line_slope(self):
        z = torch.tensor([[-3.0, -1.0, 2.0, 5.0]])
        expected = torch.tensor([0.25])
        transverse = 1.2 + expected[:, None] * z
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(
            weighted_line_slope(z, transverse, weight), expected)

    def test_direction_math_and_chord_loss(self):
        theta = np.array([2.7, 3.0])
        phi = np.array([-2.0, 0.7])
        slopes = np.stack([np.tan(theta) * np.cos(phi),
                           np.tan(theta) * np.sin(phi)], axis=1)
        expected = np.stack([np.sin(theta) * np.cos(phi),
                             np.sin(theta) * np.sin(phi),
                             np.cos(theta)], axis=1)
        np.testing.assert_allclose(slopes_to_unit_np(slopes), expected, atol=1e-12)
        got_theta, got_phi = slope_to_theta_phi_np(slopes)
        np.testing.assert_allclose(got_theta, theta, atol=1e-12)
        np.testing.assert_allclose(np.cos(got_phi), np.cos(phi), atol=1e-12)
        np.testing.assert_allclose(angular_error_np(slopes, slopes), 0.0, atol=1e-7)

        truth = torch.tensor([[0.2, -0.3], [0.0, 0.0]])
        pred = truth.clone().requires_grad_(True)
        wrong = direction_chord_loss(pred + 0.1, truth)
        self.assertGreater(float(wrong.detach()), 0.0)
        wrong.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_task_contracts(self):
        cfg = self.small_config(("task.auxiliary=[]",))
        tasks = validate_task_heads(cfg)
        self.assertEqual(tasks, ("angle",))
        self.assertEqual(cache_fields_for_tasks(tasks), frozenset({"angle"}))
        self.assertEqual(active_tasks(cfg), ("angle",))


if __name__ == "__main__":
    unittest.main()
