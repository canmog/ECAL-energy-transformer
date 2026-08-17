import unittest

from data.root_schema import required_root_branches
from utils.config import load_config


class RootSchemaTest(unittest.TestCase):
    def test_angle_profile_declares_truth_fit_and_identity(self):
        cfg, _ = load_config("config/angle.yaml")
        required = required_root_branches(cfg)
        self.assertTrue({
            "mcKX", "mcKY", "kx_ShwrKX", "kx_ShwrKY", "_run", "_event",
            "kx_ehit", "kx_expehit", "mcEne",
        }.issubset(required))

    def test_partial_identity_configuration_is_rejected(self):
        cfg, _ = load_config("config/angle.yaml", ["data.event_branch=null"])
        with self.assertRaisesRegex(ValueError, r"must either both be set"):
            required_root_branches(cfg)


if __name__ == "__main__":
    unittest.main()
