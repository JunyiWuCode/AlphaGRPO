import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Bagel.eval.gen.dpg.compute_dpg_bench import prepare_dpg_data


class DpgDataTest(unittest.TestCase):
    def test_first_proposition_is_preserved_for_dependencies(self):
        csv_text = """item_id,text,keywords,proposition_id,dependency,category_broad,category_detailed,tuple,question_natural_language
sample,prompt,,1,0,entity,whole,entity - whole (parent),Is there a parent?
sample,prompt,,2,1,entity,part,entity - part (child),Is there a child?
"""
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dpg.csv"
            csv_path.write_text(csv_text, encoding="utf-8")
            question_dict = prepare_dpg_data(SimpleNamespace(csv=csv_path))

        self.assertEqual(question_dict["sample"]["qid2dependency"][1], [0])
        self.assertEqual(question_dict["sample"]["qid2dependency"][2], [1])


if __name__ == "__main__":
    unittest.main()
