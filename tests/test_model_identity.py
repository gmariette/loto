import copy
import unittest

from loto_lab.model_identity import (
    build_model_specification,
    validate_model_specification,
)


class ModelIdentityTests(unittest.TestCase):
    def test_identical_configuration_has_stable_identity(self) -> None:
        first = build_model_specification(
            game="loto", bootstrap_simulations=2_000, seed=42
        )
        second = build_model_specification(
            game="loto", bootstrap_simulations=2_000, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(validate_model_specification(first), first["sha256"])
        self.assertNotIn("software_version", first)

    def test_scientific_configuration_changes_identity(self) -> None:
        baseline = build_model_specification(
            game="loto", bootstrap_simulations=2_000, seed=42
        )
        changed_seed = build_model_specification(
            game="loto", bootstrap_simulations=2_000, seed=43
        )
        changed_game = build_model_specification(
            game="super_loto", bootstrap_simulations=2_000, seed=42
        )
        self.assertNotEqual(baseline["sha256"], changed_seed["sha256"])
        self.assertNotEqual(baseline["sha256"], changed_game["sha256"])

    def test_tampered_specification_is_rejected(self) -> None:
        specification = build_model_specification(
            game="loto", bootstrap_simulations=2_000, seed=42
        )
        tampered = copy.deepcopy(specification)
        tampered["parameters"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "modifiee"):
            validate_model_specification(tampered)


if __name__ == "__main__":
    unittest.main()
