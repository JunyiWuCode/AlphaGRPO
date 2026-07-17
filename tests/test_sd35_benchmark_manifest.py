import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/eval_sd35_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("eval_sd35_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SD35BenchmarkManifestTest(unittest.TestCase):
    def test_builds_paper_benchmark_layouts(self):
        data_root = Path(__file__).parents[1] / "Bagel/eval/gen"
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            tasks = MODULE.build_tasks(
                data_root,
                output_root,
                MODULE.SUPPORTED_BENCHMARKS,
                seed=0,
            )
            MODULE.write_benchmark_sidecars(data_root, output_root, tasks)

            counts = {}
            for task in tasks:
                counts[task.benchmark] = counts.get(task.benchmark, 0) + 1

            self.assertEqual(counts["geneval"], 553 * 4)
            self.assertEqual(counts["tiif"], 277 * 2)
            self.assertEqual(counts["dpg"], 1065)
            self.assertEqual(counts["geneval2"], 800)
            self.assertEqual(counts["wise"], 1000)
            self.assertEqual(len({task.output_path for task in tasks}), len(tasks))

            mapping = output_root / "geneval2/images/geneval_image_map.json"
            self.assertTrue(mapping.is_file())
            metadata = output_root / "geneval/images/00000/metadata.jsonl"
            self.assertTrue(metadata.is_file())

    def test_seed_assignment_is_reproducible(self):
        data_root = Path(__file__).parents[1] / "Bagel/eval/gen"
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_tasks = MODULE.build_tasks(data_root, Path(first), ("geneval2",), 123)
            second_tasks = MODULE.build_tasks(data_root, Path(second), ("geneval2",), 123)
            self.assertEqual(
                [(task.prompt, task.seed) for task in first_tasks],
                [(task.prompt, task.seed) for task in second_tasks],
            )


if __name__ == "__main__":
    unittest.main()
