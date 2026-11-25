"""Regression checks for same-format warmup references and CE gap signs."""
import csv
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]

import minimind_moe_fp4_warmup as warmup


def reference_row(scheme="mxfp8", variant="warmup26", ce=4.205414651512526):
    return {
        "seed": "42", "architecture": "moe", "train_scheme": scheme, "eval_scheme": scheme,
        "variant": variant, "optimizer_updates": "512", "valid_tokens": "236442",
        "mean_ce": ce, "ppl": math.exp(ce),
    }


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class FourBitWarmupTests(unittest.TestCase):
    def test_default_schemes_reuse_only_the_matched_reference(self):
        fp4 = ("hif4_thr_3mm", "nvint4_3mm", "nvfp4_3mm")
        self.assertEqual(tuple(warmup.default_schemes(.0005)), fp4)
        self.assertEqual(tuple(warmup.default_schemes(.0001)), ("mxfp8", *fp4))

    def test_published_no_warmup_controls(self):
        historical = warmup.historical_ce(
            REPO / "experiments/five-format-multiseed/same_format_validation.csv")
        expected = {
            "mxfp8": 4.29545404797379, "hif4_thr_3mm": 4.330954670907075,
            "nvint4_3mm": 4.325956163574688, "nvfp4_3mm": 4.316044891605898,
        }
        for scheme, value in expected.items():
            with self.subTest(scheme=scheme):
                self.assertAlmostEqual(historical[scheme], value)

    def test_pick_warmup_reference_by_variant_and_format(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "noise_runs.csv"
            non_controls = [reference_row("bf16", ce=4.23140505918397),
                            reference_row(variant="repeat", ce=4.29545404797379),
                            reference_row(variant="warmup26_lr_x1p0001", ce=4.3)]
            write_rows(path, non_controls)
            self.assertIsNone(warmup.mxfp8_warmup_ce(path))
            write_rows(path, [*non_controls, reference_row()])
            self.assertAlmostEqual(warmup.mxfp8_warmup_ce(path), 4.205414651512526)

    def test_invalid_same_format_or_endpoint_reference(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "noise_runs.csv"
            for name, bad in (("eval_scheme", "bf16"), ("optimizer_updates", "256"),
                              ("valid_tokens", "3929"), ("ppl", 1.0)):
                with self.subTest(name=name):
                    write_rows(path, [dict(reference_row(), **{name: bad})])
                    with self.assertRaises(ValueError):
                        warmup.mxfp8_warmup_ce(path)
            write_rows(path, [reference_row(), reference_row(ce=4.26)])
            with self.assertRaises(ValueError):
                warmup.mxfp8_warmup_ce(path)

    def test_paired_gap_uses_own_schedule_and_format(self):
        schemes = ("hif4_thr_3mm", "nvint4_3mm", "nvfp4_3mm")
        self.assertEqual(set(warmup.FP4_SCHEMES), set(schemes))
        historical = dict(zip(schemes, (4.33, 4.32, 4.31)), mxfp8=4.30)
        scores = {s: (ce, math.exp(ce)) for s, ce in zip(schemes, (4.23, 4.19, 4.20))}
        rows = {r["train_scheme"]: r for r in warmup.comparison_rows(scores, historical, 4.20)}
        self.assertEqual(set(rows), set(schemes))
        for scheme, row in rows.items():
            with self.subTest(scheme=scheme):
                ce, ppl = scores[scheme]
                self.assertEqual(row["eval_scheme"], scheme)
                self.assertAlmostEqual(row["mean_ce"], ce)
                self.assertAlmostEqual(row["ppl"], ppl)
                self.assertAlmostEqual(row["no_warmup_ce"], historical[scheme])
                self.assertAlmostEqual(row["warmup_minus_no_warmup_ce"], ce - historical[scheme])
                self.assertAlmostEqual(row["gap_vs_mxfp8_no_warmup"], historical[scheme] - 4.30)
                self.assertAlmostEqual(row["gap_vs_mxfp8_warmup"], ce - 4.20)
                self.assertAlmostEqual(row["change_in_gap"], (ce - 4.20) - (historical[scheme] - 4.30))

        partial = list(warmup.comparison_rows({schemes[1]: scores[schemes[1]]}, historical, None))
        self.assertEqual(len(partial), 1)
        self.assertAlmostEqual(partial[0]["gap_vs_mxfp8_no_warmup"], .02)
        self.assertIsNone(partial[0]["gap_vs_mxfp8_warmup"])
        self.assertIsNone(partial[0]["change_in_gap"])


    def test_lower_lr_uses_only_same_lr_warmup_reference(self):
        hif4, nvfp4 = "hif4_thr_3mm", "nvfp4_3mm"
        scores = {hif4: (4.22, math.exp(4.22)), nvfp4: (4.30, math.exp(4.30))}
        historical = {"mxfp8": 4.29, hif4: 4.33, "nvint4_3mm": 4.32, nvfp4: 4.31}
        lower_lr = .0001
        def by_scheme():
            return {r["train_scheme"]: r for r in
                    warmup.comparison_rows(scores, historical, 4.2054, lr=lower_lr)}

        partial = by_scheme()
        for scheme in (hif4, nvfp4):
            self.assertIsNone(partial[scheme]["gap_vs_mxfp8_warmup"])

        scores["mxfp8"] = (4.25, math.exp(4.25))
        rows = by_scheme()
        self.assertAlmostEqual(rows[hif4]["gap_vs_mxfp8_warmup"], -.03)
        self.assertAlmostEqual(rows[nvfp4]["gap_vs_mxfp8_warmup"], .05)
        for row in rows.values():
            for key in ("no_warmup_ce", "warmup_minus_no_warmup_ce",
                        "gap_vs_mxfp8_no_warmup", "change_in_gap"):
                self.assertIsNone(row[key], key)


    def test_two_lr_run_and_subsequent_single_format_reuse(self):
        import minimind_fp4_warmup_arm as arm
        gpu = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 8, get_device_name=lambda _: "V100"))
        scores = {}
        calls = []

        def fit(work, scheme, lr):
            self.assertEqual(work.parent.name, warmup.LR_DIRS[lr])
            calls.append((lr, scheme))
            ce = 4.21 if lr == warmup.LOW_LR else 4.24
            scores[work, scheme, lr] = (ce, math.exp(ce))
            return scores[work, scheme, lr]

        with TemporaryDirectory() as folder:
            output = Path(folder)
            reference = output / "noise_runs.csv"
            write_rows(reference, [reference_row()])
            args = SimpleNamespace(output=output, data=output / "pilot", noise_runs=reference,
                                   lr=[warmup.BASE_LR, warmup.LOW_LR], schemes=None)
            paths = {"trainer/mock.py": Path(__file__)}
            hashes = {k: warmup.digest(v) for k, v in paths.items()}
            with (patch.dict(sys.modules, {"torch": gpu}), patch.dict(warmup.os.environ),
                  patch.object(warmup, "inputs", return_value=(paths, hashes)),
                  patch.object(warmup, "version", return_value="test"),
                  patch.object(arm, "cached_score", side_effect=lambda *key: scores.get(key)),
                  patch.object(arm, "run_arm", side_effect=fit)):
                warmup.run(args)
                self.assertEqual(calls, [(lr, s) for lr in args.lr for s in warmup.default_schemes(lr)])
                args.schemes, args.lr = ["nvfp4_3mm"], [warmup.LOW_LR]
                warmup.run(args)
                self.assertEqual(len(calls), 7)
            with (output / "warmup_comparison.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            low = next(r for r in rows if r["train_scheme"] == "nvfp4_3mm" and float(r["base_lr"]) == 1e-4)
            self.assertAlmostEqual(float(low["low_lr_minus_base_lr_warmup_ce"]), -.03)
            self.assertEqual(float(low["mxfp8_warmup_ce"]), 4.21)


if __name__ == "__main__":
    unittest.main()
