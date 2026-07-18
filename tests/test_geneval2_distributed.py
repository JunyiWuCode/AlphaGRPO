import importlib.util
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/eval_geneval2_distributed.py"
SPEC = importlib.util.spec_from_file_location("eval_geneval2_distributed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GenEval2DistributedTest(unittest.TestCase):
    def test_answer_candidates_match_bundled_scorer(self):
        self.assertEqual(MODULE.answer_candidates("Is it green?", "Yes"), MODULE.YES_ANSWERS)
        self.assertIn("7", MODULE.answer_candidates("How many?", "seven"))
        self.assertIn(" seven", MODULE.answer_candidates("How many?", "seven"))

    def test_geometric_mean(self):
        self.assertAlmostEqual(MODULE.geometric_mean([0.25, 1.0]), 0.5)
        self.assertEqual(MODULE.geometric_mean([0.0, 1.0]), 0.0)
        with self.assertRaises(ValueError):
            MODULE.geometric_mean([])


if __name__ == "__main__":
    unittest.main()
