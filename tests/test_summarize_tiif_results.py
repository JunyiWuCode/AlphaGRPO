import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_tiif_results import GROUPS, summarize_tiif


class TiifSummaryTest(unittest.TestCase):
    def test_nine_group_macro_average(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results" / "sd35"
            all_attributes = [attribute for attributes in GROUPS.values() for attribute in attributes]
            for index, attribute in enumerate(all_attributes):
                for length in ("short", "long"):
                    result_dir = root / attribute / length
                    result_dir.mkdir(parents=True)
                    is_correct = index % 2 == 0 if length == "short" else True
                    row = {
                        "gt_answers": ["yes"],
                        "model_pred": ["yes" if is_correct else "no"],
                    }
                    (result_dir / "0.json").write_text(json.dumps(row), encoding="utf-8")

            summary = summarize_tiif(root.parent)

        self.assertEqual(summary["file_counts"]["short"], len(all_attributes))
        self.assertEqual(summary["question_counts"]["long"], len(all_attributes))
        self.assertEqual(summary["score_long"], 100.0)
        expected_short = sum(
            sum(100.0 if all_attributes.index(attribute) % 2 == 0 else 0.0 for attribute in attributes)
            / len(attributes)
            for attributes in GROUPS.values()
        ) / len(GROUPS)
        self.assertAlmostEqual(summary["score_short"], expected_short)


if __name__ == "__main__":
    unittest.main()
