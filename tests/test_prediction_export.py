import json
import os
import tempfile
import unittest

import numpy as np
import torch

from export_predictions import main as export_predictions
from models.model import EcalTransformer
from utils.config import load_config


class PredictionExportTest(unittest.TestCase):
    def test_target_free_cache_exports_without_mc_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = os.path.join(directory, "cache")
            os.makedirs(cache)
            arrays = {
                "off": np.array([0, 2], np.int64),
                "tok_layer": np.array([0, 2], np.int16),
                "tok_cell": np.array([10, 20], np.int16),
                "tok_ehit": np.array([100.0, 200.0], np.float32),
                "run": np.array([17], np.uint32),
                "event": np.array([29], np.uint32),
            }
            np.savez(os.path.join(cache, "test.npz"), **arrays)
            with open(os.path.join(cache, "meta.json"), "w") as handle:
                json.dump({"geometry_data_type": "MC"}, handle)

            output = os.path.join(directory, "predictions.npz")
            checkpoint_path = os.path.join(directory, "model.pt")
            overrides = [
                f"paths.cache_dir={cache}", f"paths.out_dir={directory}",
                "device=cpu", "train.num_workers=0",
                "model.d_model=16", "model.n_blocks=2", "model.n_heads=4",
                "model.tap_block=1", "model.d_phys=8", "model.d_free=8",
                "model.pos_embed=false", "heads.energy.enabled=false",
                "heads.recon.enabled=false", "heads.angle.enabled=true",
                "heads.angle.type=global", "heads.angle.hidden=[8]",
                "task.mode=angle", "task.selection=angle", "task.auxiliary=[]",
                "loss.angle={'delta': 0.1, 'mode': 'huber'}",
                "loss.fit_quality_weight.enabled=false",
            ]
            cfg, raw = load_config("config/base.yaml", overrides)
            model = EcalTransformer(cfg, n_concepts=1)
            model.set_angle_norm([0.0, 0.0], [1.0, 1.0])
            training_meta = {
                "n_concepts": 1,
                "geometry_data_type": "MC",
                "angle_mean": [0.0, 0.0],
                "angle_std": [1.0, 1.0],
            }
            torch.save({"model": model.state_dict(), "meta": training_meta,
                        "config": raw}, checkpoint_path)

            export_predictions([
                "--config", "config/base.yaml", "--ckpt", checkpoint_path,
                "--output", output, "--set", *overrides])
            with np.load(output) as result:
                self.assertEqual(set(result.files), {
                    "run", "event", "pred_kx", "pred_ky", "pred_dir",
                    "pred_theta", "pred_phi",
                })
                self.assertEqual(int(result["run"][0]), 17)
                self.assertEqual(int(result["event"][0]), 29)


if __name__ == "__main__":
    unittest.main()
