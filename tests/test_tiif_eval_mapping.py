import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Bagel" / "eval" / "gen" / "tiif"))

from eval_with_vlm import OutputFormatError, collect_tasks, extract_yes_no


class TiifEvalMappingTest(unittest.TestCase):
    def test_manifest_mapping_does_not_depend_on_glob_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation_dir = root / "prompts"
            eval_dir = root / "eval_prompts"
            output_dir = root / "results"
            generation_dir.mkdir()
            eval_dir.mkdir()

            generation_rows = [
                {
                    "type": "shape",
                    "short_description": "short zero",
                    "long_description": "long zero",
                },
                {
                    "type": "shape",
                    "short_description": "short one",
                    "long_description": "long one",
                },
            ]
            eval_rows = [
                {"type": "shape", "yn_question_list": ["q0"], "yn_answer_list": ["yes"]},
                {"type": "shape", "yn_question_list": ["q1"], "yn_answer_list": ["yes"]},
            ]
            (generation_dir / "shape_prompts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in generation_rows),
                encoding="utf-8",
            )
            (eval_dir / "shape_eval_prompts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in eval_rows),
                encoding="utf-8",
            )

            manifest_rows = []
            for index, row in enumerate(generation_rows):
                for description in ("short_description", "long_description"):
                    image_path = root / "variant" / "tiif" / "images" / "shape" / "sd35" / description / f"{100 + index}.png"
                    manifest_rows.append({
                        "benchmark": "tiif",
                        "prompt": row[description],
                        "output_path": str(image_path),
                    })
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(manifest_rows)),
                encoding="utf-8",
            )

            tasks = collect_tasks(
                str(eval_dir),
                str(root / "unused_images"),
                "sd35",
                str(output_dir),
                generation_jsonl_dir=str(generation_dir),
                manifest_file=str(manifest),
            )

        self.assertEqual(len(tasks), 4)
        self.assertEqual({task["line_idx"] for task in tasks}, {100, 101})
        self.assertTrue(all("shape/sd35" in task["img_path"] for task in tasks))

    def test_extra_answers_can_be_truncated_explicitly(self):
        questions = ["q0", "q1"]
        with self.assertRaises(OutputFormatError):
            extract_yes_no("yes\nno\nno", questions)
        self.assertEqual(
            extract_yes_no("yes\nno\nno", questions, allow_extra_answers=True),
            ["yes", "no"],
        )


if __name__ == "__main__":
    unittest.main()
