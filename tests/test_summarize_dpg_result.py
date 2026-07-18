import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_dpg_result import parse_dpg_result


SAMPLE_RESULT = """/tmp/images/a.jpg, 0.5, 0.5
/tmp/images/b.jpg, 1.0, 1.0

Model: images
L1 category scores:
\tEntity: 75.00 (n=4)
L2 category scores:
\tEntity - whole: 80.00
Image path: /tmp/images
Save results to: /tmp/results.txt
DPG-Bench score: 75.0000
"""


class DpgSummaryTest(unittest.TestCase):
    def test_parse_result(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.txt"
            result_path.write_text(SAMPLE_RESULT, encoding="utf-8")
            summary = parse_dpg_result(result_path)
        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(summary["score"], 75.0)
        self.assertEqual(summary["l1_category_scores"]["Entity"]["count"], 4)
        self.assertEqual(summary["l2_category_scores"]["Entity - whole"]["score"], 80.0)

    def test_cli_checks_image_count(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.txt"
            output_path = Path(directory) / "summary.json"
            result_path.write_text(SAMPLE_RESULT, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/summarize_dpg_result.py"),
                    str(result_path),
                    str(output_path),
                    "--expected-images",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(output_path.read_text())["score"], 75.0)


if __name__ == "__main__":
    unittest.main()
