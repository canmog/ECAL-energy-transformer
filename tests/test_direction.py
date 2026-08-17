import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from data.dataset import EcalTokens
from data.geometry import COMPONENT_OF_VIEW, COMPONENT_VIEWS
from losses.objectives import direction_chord_loss
from models.model import EcalTransformer
from models.physics import (LayerCentroidSlope, LearnedRobustLayerSlope,
                            weighted_line_slope)
from train import angle_energy_weight, build_weighter
from utils.angle import angular_error_np, slope_to_theta_phi_np, slopes_to_unit_np
from utils.config import load_config


class DirectionMathTest(unittest.TestCase):
    def test_low_energy_angle_weights_are_bounded_and_normalized(self):
        cfg, _ = load_config("config/base.yaml", [
            "loss.angle_energy_weight.enabled=true",
            "loss.angle_energy_weight.pivot_gev=300.0",
            "loss.angle_energy_weight.power=0.5",
            "loss.angle_energy_weight.w_min=0.5",
            "loss.angle_energy_weight.w_max=3.0",
            "loss.angle_energy_weight.normalize_batch=true",
        ])
        energy = torch.tensor([30.0, 300.0, 3000.0])
        weight = angle_energy_weight(energy, cfg)
        self.assertGreater(float(weight[0]), float(weight[1]))
        self.assertGreater(float(weight[1]), float(weight[2]))
        self.assertAlmostEqual(float(weight.mean()), 1.0, places=6)

    def test_spherical_convention(self):
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
        np.testing.assert_allclose(np.sin(got_phi), np.sin(phi), atol=1e-12)
        # arccos(dot) loses a few ulps at dot~=1 even for identical vectors.
        np.testing.assert_allclose(angular_error_np(slopes, slopes), 0.0, atol=1e-7)

    def test_chord_loss_is_finite_and_direction_sensitive(self):
        truth = torch.tensor([[0.2, -0.3], [0.0, 0.0]], dtype=torch.float32)
        pred = truth.clone().requires_grad_(True)
        exact = direction_chord_loss(pred, truth)
        wrong = direction_chord_loss(pred + 0.1, truth)
        self.assertAlmostEqual(float(exact.detach()), 0.0, places=7)
        self.assertGreater(float(wrong.detach()), float(exact.detach()))
        wrong.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_weighted_line_slope(self):
        z = torch.tensor([[-3.0, -1.0, 2.0, 5.0], [-3.0, -1.0, 2.0, 5.0]])
        expected = torch.tensor([0.25, -0.4])
        intercept = torch.tensor([[1.2], [-0.7]])
        transverse = intercept + expected[:, None] * z
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]])
        torch.testing.assert_close(weighted_line_slope(z, transverse, weight), expected)

    def test_learned_robust_fit_starts_at_fixed_j1_fit_and_has_gradient(self):
        fixed = LayerCentroidSlope("MC", 1.5, [1, 0])
        learned = LearnedRobustLayerSlope(
            "MC", 1.5, [1, 0], hidden=8, max_weight_multiplier=4.0)
        layers = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        cells = torch.tensor([[24, 27, 31, 34, 29, 33, 38, 35, 42, 39]], dtype=torch.long)
        energy = torch.tensor(
            [[25.0, 80.0, 45.0, 120.0, 240.0, 170.0, 95.0, 60.0, 35.0, 20.0]])
        feats = torch.zeros(1, layers.shape[1], 6)
        feats[..., 0] = torch.log1p(energy)
        batch = {
            "feats": feats,
            "pos_id": layers * 72 + cells,
            "valid": torch.ones_like(layers, dtype=torch.bool),
        }
        expected = fixed(batch)
        got = learned(batch)
        torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-7)
        got.square().sum().backward()
        final_grad = learned.reliability[-1].weight.grad
        self.assertIsNotNone(final_grad)
        self.assertTrue(torch.isfinite(final_grad).all())
        self.assertGreater(float(final_grad.abs().sum()), 0.0)

    def test_joint_weighter_contains_all_four_tasks(self):
        cfg, _ = load_config("config/base.yaml", ["loss.weighting=uncertainty"])
        tasks = ["energy", "angle", "recon", "concept"]
        weighter = build_weighter(cfg, tasks, "cpu")
        self.assertEqual(set(weighter.task_weights()), set(tasks))

    def test_normalized_residual_head_starts_at_calibrated_centroid(self):
        cfg, _ = load_config("config/base.yaml", [
            "model.d_model=16", "model.n_blocks=2", "model.n_heads=4",
            "model.tap_block=1", "model.d_phys=8", "model.d_free=8",
            "model.pos_embed=false", "heads.energy.enabled=false",
            "heads.recon.enabled=false", "heads.angle.type=view_residual",
            "heads.angle.hidden=[8]", "heads.angle.component_views=[1,0]",
            "heads.angle.residual_normalization=true",
            "heads.angle.residual_scale=1.0",
        ])
        model = EcalTransformer(cfg, n_concepts=2).eval()
        model.set_angle_norm([0.0, 0.0], [1.0, 1.0])
        center = torch.tensor([0.012, -0.023])
        model.set_angle_residual_norm(center, [0.04, 0.05])
        # One token in each stored view. With only one layer per projection the
        # analytic fitted slopes are zero; the zero-initialized residual head
        # must therefore produce exactly the robust residual center.
        feats = torch.zeros(1, 2, 6)
        feats[0, 0, 4] = 1.0  # layer 0, stored view 0 -> ky
        feats[0, 1, 5] = 1.0  # layer 2, stored view 1 -> kx
        batch = {
            "feats": feats,
            "pos_id": torch.tensor([[0, 2 * 72]], dtype=torch.long),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        }
        with torch.no_grad():
            out = model(batch)
        torch.testing.assert_close(out["angle_baseline"], torch.zeros(1, 2))
        torch.testing.assert_close(out["angle_slopes"], center.unsqueeze(0))
        torch.testing.assert_close(out["angle_residual_std"], torch.zeros(1, 2))


class ReflectionTest(unittest.TestCase):
    def test_memory_mapped_split_matches_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            split_dir = os.path.join(directory, "test")
            os.makedirs(split_dir)
            arrays = {
                "off": np.array([0, 2], np.int64),
                "tok_layer": np.array([0, 2], np.int16),
                "tok_cell": np.array([10, 20], np.int16),
                "tok_ehit": np.array([100.0, 200.0], np.float32),
                "tok_expe": np.array([90.0, 180.0], np.float32),
                "energy": np.array([100.0], np.float32),
                "angle": np.array([[0.2, -0.3]], np.float32),
                "fit_angle": np.array([[0.1, -0.4]], np.float32),
                "concepts": np.array([[0.2, -0.3]], np.float32),
                "run": np.array([7], np.uint32),
                "event": np.array([11], np.uint32),
            }
            for name, values in arrays.items():
                np.save(os.path.join(split_dir, f"{name}.npy"), values)
            meta = {
                "concept_names": ["shwr_kx", "shwr_ky"],
                "concept_mean": [0.0, 0.0], "concept_std": [1.0, 1.0],
                "concept_reflect_x": [-1, 1], "concept_reflect_y": [1, -1],
                "concept_coord": [None, None],
                "angle_mean": [0.0, 0.0], "angle_std": [1.0, 1.0],
                "angle_reflect_x": [-1, 1], "angle_reflect_y": [1, -1],
                "log_energy_mean": 0.0, "log_energy_std": 1.0,
                "geometry_data_type": "MC",
            }
            dataset = EcalTokens(
                directory, "test", meta, compute_fit_resid=False)
            self.assertIsInstance(dataset.ehit.base, np.memmap)
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            np.testing.assert_array_equal(item["pos_id"].numpy(), [10, 164])
            self.assertEqual(float(item["fit_resid"]), 0.0)

    def test_physical_reflections_use_orthogonal_fibre_views(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                os.path.join(directory, "train.npz"),
                off=np.array([0, 2], np.int64),
                # layer 0 is stored view 0 -> measured Y; layer 2 is view 1 -> X
                tok_layer=np.array([0, 2], np.int16),
                tok_cell=np.array([10, 20], np.int16),
                tok_ehit=np.array([100.0, 200.0], np.float32),
                tok_expe=np.array([90.0, 180.0], np.float32),
                energy=np.array([100.0], np.float32),
                angle=np.array([[0.2, -0.3]], np.float32),
                fit_angle=np.array([[0.1, -0.4]], np.float32),
                concepts=np.array([[0.2, -0.3]], np.float32),
                run=np.array([7], np.uint32),
                event=np.array([11], np.uint32),
            )
            meta = {
                "concept_names": ["shwr_kx", "shwr_ky"],
                "concept_mean": [0.0, 0.0], "concept_std": [1.0, 1.0],
                "concept_reflect_x": [-1, 1], "concept_reflect_y": [1, -1],
                "concept_coord": [None, None],
                "angle_mean": [0.0, 0.0], "angle_std": [1.0, 1.0],
                "angle_reflect_x": [-1, 1], "angle_reflect_y": [1, -1],
                "log_energy_mean": 0.0, "log_energy_std": 1.0,
                "geometry_data_type": "MC",
            }
            dataset = EcalTokens(directory, "train", meta, train=True,
                                 augment={"reflect_x": True, "reflect_y": True})
            self.assertEqual(COMPONENT_VIEWS, (1, 0))
            self.assertEqual(COMPONENT_OF_VIEW, (1, 0))

            # X reflection mirrors stored view 1 (layer 2), then flips kx only.
            with mock.patch("numpy.random.rand", side_effect=[0.0, 1.0]):
                item_x = dataset[0]
            np.testing.assert_array_equal(
                item_x["pos_id"].numpy(), [0 * 72 + 10, 2 * 72 + 51])
            np.testing.assert_allclose(item_x["angle"].numpy(), [-0.2, -0.3])
            np.testing.assert_allclose(item_x["fit_angle"].numpy(), [-0.1, -0.4])
            np.testing.assert_allclose(item_x["concepts"].numpy(), [-0.2, -0.3])

            # Y reflection mirrors stored view 0 (layer 0), then flips ky only.
            with mock.patch("numpy.random.rand", side_effect=[1.0, 0.0]):
                item_y = dataset[0]
            np.testing.assert_array_equal(
                item_y["pos_id"].numpy(), [0 * 72 + 61, 2 * 72 + 20])
            np.testing.assert_allclose(item_y["angle"].numpy(), [0.2, 0.3])
            np.testing.assert_allclose(item_y["fit_angle"].numpy(), [0.1, 0.4])
            np.testing.assert_allclose(item_y["concepts"].numpy(), [0.2, 0.3])

            thresholded = EcalTokens(
                directory, "train", meta, train=False, threshold_mev=150.0)
            item_thresholded = thresholded[0]
            self.assertEqual(item_thresholded["feats"].shape[0], 1)
            np.testing.assert_array_equal(
                item_thresholded["pos_id"].numpy(), [2 * 72 + 20])


if __name__ == "__main__":
    unittest.main()
