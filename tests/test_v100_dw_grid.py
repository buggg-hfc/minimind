"""Check the public dW-ablation command without training data or GPUs."""
import csv
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/minimind_multiseed_grid.py"


class WeightGradientGridTests(unittest.TestCase):
    def test_plan_uses_historical_seeds_and_only_own_format_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            result = subprocess.run([sys.executable, str(SCRIPT), "--dw-mxfp8", "--plan",
                                     "--output", str(output)], text=True, capture_output=True, check=True)
            self.assertFalse(output.exists())
        rows = list(csv.DictReader(result.stdout.splitlines()[:-1]))
        expected = set(product((42, 43, 100, 2026), ("dense", "moe"),
                               ("hif4_thr_3mm_dw_mxfp8", "nvfp4_3mm_dw_mxfp8", "nvint4_3mm_dw_mxfp8")))
        self.assertEqual(len(rows), 24)
        self.assertEqual({(int(r["seed"]), r["architecture"], r["train_scheme"]) for r in rows}, expected)
        self.assertTrue(all(r["train_scheme"] == r["eval_scheme"] for r in rows))


if __name__ == "__main__":
    unittest.main()
