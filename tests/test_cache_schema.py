import json
import os
import tempfile
import unittest

import numpy as np

from data.dataset import EcalTokens, collate, load_meta
from data.schema import MissingCacheFieldsError


def core_arrays():
    return {
        "off": np.array([0, 2], np.int64),
        "tok_layer": np.array([0, 2], np.int16),
        "tok_cell": np.array([10, 20], np.int16),
        "tok_ehit": np.array([100.0, 200.0], np.float32),
    }


def energy_arrays():
    arrays = core_arrays()
    arrays.update({
        "tok_expe": np.array([90.0, 180.0], np.float32),
        "energy": np.array([100.0], np.float32),
        "concepts": np.array([[0.2, -0.3]], np.float32),
    })
    return arrays


def energy_meta():
    return {
        "concept_names": ["shwr_kx", "shwr_ky"],
        "concept_mean": [0.0, 0.0],
        "concept_std": [1.0, 1.0],
        "concept_reflect_x": [-1, 1],
        "concept_reflect_y": [1, -1],
        "concept_coord": [None, None],
        "log_energy_mean": 0.0,
        "log_energy_std": 1.0,
        "geometry_data_type": "MC",
    }


class CacheSchemaTest(unittest.TestCase):
    def write_npz(self, directory, arrays, meta):
        np.savez(os.path.join(directory, "train.npz"), **arrays)
        with open(os.path.join(directory, "meta.json"), "w") as handle:
            json.dump(meta, handle)

    def test_legacy_energy_cache_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_npz(directory, energy_arrays(), energy_meta())
            dataset = EcalTokens(directory, "train", load_meta(directory))
            item = dataset[0]
            self.assertEqual(set(item), {
                "feats", "pos_id", "recon", "log_e", "energy", "concepts",
                "fit_resid",
            })

    def test_angle_mode_fails_before_training_when_angle_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_npz(directory, energy_arrays(), energy_meta())
            with self.assertRaisesRegex(
                    MissingCacheFieldsError, r"missing required field.*angle"):
                EcalTokens(directory, "train", load_meta(directory),
                           task_mode="angle", compute_fit_resid=False,
                           operation="angle training")

    def test_required_angle_metadata_is_never_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = core_arrays()
            arrays["angle"] = np.array([[0.2, -0.3]], np.float32)
            meta = {"geometry_data_type": "MC", "angle_mean": [0.0, 0.0]}
            self.write_npz(directory, arrays, meta)
            with self.assertRaisesRegex(
                    MissingCacheFieldsError, r"angle_reflect_x.*angle_std"):
                EcalTokens(directory, "train", load_meta(directory),
                           task_mode="angle", compute_fit_resid=False,
                           operation="angle training")

    def test_prediction_cache_needs_inputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_npz(
                directory, core_arrays(), {"geometry_data_type": "MC"})
            dataset = EcalTokens(
                directory, "train", load_meta(directory), task_mode="predict",
                include_targets=False, compute_fit_resid=False,
                operation="prediction")
            batch = collate([dataset[0]])
            self.assertEqual(set(batch), {"feats", "pos_id", "valid"})

    def test_memory_mapped_layout_has_the_same_logical_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            split_dir = os.path.join(directory, "train")
            os.makedirs(split_dir)
            for name, values in energy_arrays().items():
                np.save(os.path.join(split_dir, f"{name}.npy"), values)
            with open(os.path.join(directory, "meta.json"), "w") as handle:
                json.dump(energy_meta(), handle)
            dataset = EcalTokens(directory, "train", load_meta(directory))
            self.assertIsInstance(dataset.ehit.base, np.memmap)
            self.assertIn("energy", dataset.available_fields)
            self.assertIn("concepts", dataset.available_fields)


if __name__ == "__main__":
    unittest.main()
