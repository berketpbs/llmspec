import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("add_models.py")
SPEC = importlib.util.spec_from_file_location("add_models", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CatalogValidationTests(unittest.TestCase):
    def test_valid_record_is_accepted(self):
        MODULE.validate_records(
            [MODULE.m("test/model", "Test", "Test", 1.0, 4096, "general")]
        )

    def test_non_finite_parameter_count_is_rejected(self):
        record = MODULE.m("test/model", "Test", "Test", 1.0, 4096, "general")
        record["params_b"] = math.inf
        with self.assertRaises(SystemExit):
            MODULE.validate_records([record])


if __name__ == "__main__":
    unittest.main()
