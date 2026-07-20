import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_geneval_result import summarize_geneval


class GenEvalSummaryTest(unittest.TestCase):
    def test_macro_average_and_counts(self):
        rows = [
            {"tag": "single_object", "correct": True, "metadata": "prompt-a"},
            {"tag": "single_object", "correct": False, "metadata": "prompt-a"},
            {"tag": "counting", "correct": True, "metadata": "prompt-b"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = summarize_geneval(path)

        self.assertEqual(summary["image_count"], 3)
        self.assertEqual(summary["prompt_count"], 2)
        self.assertAlmostEqual(summary["score"], 75.0)
        self.assertAlmostEqual(summary["image_accuracy"], 200 / 3)
        self.assertAlmostEqual(summary["prompt_accuracy_at_4"], 100.0)


if __name__ == "__main__":
    unittest.main()
