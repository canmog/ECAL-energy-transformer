import json
import os
import tempfile
import unittest

import numpy as np
import torch

from data.schema import MissingCacheFieldsError
from train import build_loaders, checkpoint_specs, compute_losses
from utils.config import load_config
from utils.tasks import validate_training_contract


def core_arrays():
    return {
        "off": np.array([0, 2], np.int64),
        "tok_layer": np.array([0, 2], np.int16),
        "tok_cell": np.array([10, 20], np.int16),
        "tok_ehit": np.array([100.0, 200.0], np.float32),
    }


def base_meta():
    return {
        "geometry_data_type": "MC",
        "n_concepts": 0,
        "log_energy_mean": 0.0,
        "log_energy_std": 1.0,
        "angle_mean": [0.0, 0.0],
        "angle_std": [1.0, 1.0],
        "angle_reflect_x": [-1.0, 1.0],
        "angle_reflect_y": [1.0, -1.0],
    }


class TrainingModesTest(unittest.TestCase):
    def write_cache(self, directory, arrays, meta=None):
        for split in ("train", "val"):
            np.savez(os.path.join(directory, f"{split}.npz"), **arrays)
        with open(os.path.join(directory, "meta.json"), "w") as handle:
            json.dump(meta or base_meta(), handle)

    def config(self, directory, mode, extra=()):
        overrides = [
            f"paths.cache_dir={directory}",
            "train.num_workers=0",
            "train.augment={'reflect_x': False, 'reflect_y': False}",
            f"task.mode={mode}",
            "task.auxiliary=[]",
            "loss.fit_quality_weight.enabled=false",
            "loss.angle={'delta': 0.1, 'mode': 'huber'}",
            f"heads.energy.enabled={'true' if mode in ('energy', 'joint') else 'false'}",
            f"heads.angle.enabled={'true' if mode in ('angle', 'joint') else 'false'}",
            "heads.recon.enabled=false",
        ]
        return load_config("config/base.yaml", overrides + list(extra))[0]

    def test_energy_only_does_not_require_auxiliary_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = core_arrays()
            arrays["energy"] = np.array([100.0], np.float32)
            self.write_cache(directory, arrays)
            cfg = self.config(directory, "energy")
            train_loader, _ = build_loaders(cfg, base_meta())
            self.assertIsNotNone(train_loader.dataset.energy)
            self.assertIsNone(train_loader.dataset.expe)
            self.assertIsNone(train_loader.dataset.concepts)

    def test_angle_only_does_not_touch_energy_when_weighting_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = core_arrays()
            arrays["angle"] = np.array([[0.2, -0.3]], np.float32)
            self.write_cache(directory, arrays)
            cfg = self.config(directory, "angle")
            train_loader, _ = build_loaders(cfg, base_meta())
            batch = next(iter(train_loader))
            self.assertNotIn("energy", batch)
            output = {
                "angle": batch["angle_std"].clone().requires_grad_(True),
                "angle_slopes": batch["angle"].clone().requires_grad_(True),
            }
            losses = compute_losses(output, batch, cfg, tasks=("angle",))
            self.assertEqual(set(losses), {"angle"})

    def test_angle_energy_weight_declares_energy_as_required(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = core_arrays()
            arrays["angle"] = np.array([[0.2, -0.3]], np.float32)
            self.write_cache(directory, arrays)
            cfg = self.config(directory, "angle", (
                "loss.angle_energy_weight={'enabled': True}",))
            with self.assertRaisesRegex(
                    MissingCacheFieldsError, r"angle training.*missing.*energy"):
                build_loaders(cfg, base_meta())

    def test_joint_mode_fails_before_first_batch_on_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = core_arrays()
            arrays["angle"] = np.array([[0.2, -0.3]], np.float32)
            self.write_cache(directory, arrays)
            cfg = self.config(directory, "joint")
            with self.assertRaisesRegex(
                    MissingCacheFieldsError, r"joint training.*missing.*energy"):
                build_loaders(cfg, base_meta())

    def test_missing_loss_section_is_a_preflight_error(self):
        cfg, _ = load_config("config/base.yaml", [
            "task.mode=angle", "task.auxiliary=[]",
            "heads.energy.enabled=false", "heads.recon.enabled=false",
            "heads.angle.enabled=true",
        ])
        with self.assertRaisesRegex(ValueError, r"loss\.angle is missing"):
            validate_training_contract(cfg)

    def test_checkpoint_registry_covers_each_active_primary(self):
        specs = checkpoint_specs(("energy", "angle"))
        self.assertEqual(specs["primary"], ("val_metric", "best.pt"))
        self.assertEqual(specs["energy"][1], "best_energy.pt")
        self.assertEqual(specs["angle"][1], "best_angle.pt")


if __name__ == "__main__":
    unittest.main()
