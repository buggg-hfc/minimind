import csv
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]

import minimind_moe_noise_floor as noise
from minimind_grid_train import arm_paths


def scores(ce):
    return ce, math.exp(ce)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class NoiseSummaryTests(unittest.TestCase):
    def test_controls_match_historical_same_format_moe_seed42(self):
        controls = noise.historical_controls(REPO / noise.ENDPOINTS)
        self.assertEqual(set(controls), {"bf16", "mxfp8"})
        self.assertAlmostEqual(controls["bf16"][0], 4.263754998716669)
        self.assertAlmostEqual(controls["mxfp8"][0], 4.29545404797379)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid.csv"
            rows = csv_rows(REPO / noise.ENDPOINTS)
            match = next(r for r in rows if (r["architecture"], r["seed"], r["train_scheme"]) == ("moe", "42", "bf16"))
            match["valid_tokens"] = "3929"
            noise.record_pairs(output, rows)
            with self.assertRaisesRegex(ValueError, "Invalid historical"):
                noise.historical_controls(output)

    def test_partial_and_complete_reports_match_controls(self):
        self.assertEqual(noise.DEFAULT_SCENARIOS, ("repeat", "lr_x1p0001", "lr_x0p9999"))
        controls = {"bf16": scores(4.25), "mxfp8": scores(4.30)}
        measured = {("bf16", "lr_x0p9999"): scores(4.20)}
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            noise.report(output, controls, measured)
            self.assertEqual(csv_rows(output / "noise_runs.csv")[0]["delta_ce_vs_repeat"], "")
            self.assertEqual(csv_rows(output / "noise_summary.csv")[0]["max_abs_perturbation_delta_ce_vs_repeat"], "")
            measured.update({
                ("bf16", "repeat"): scores(4.26), ("bf16", "lr_x1p0001"): scores(4.28),
                ("bf16", "warmup26"): scores(4.00), ("bf16", "warmup26_lr_x1p0001"): scores(4.06),
                ("mxfp8", "repeat"): scores(4.35)})
            noise.report(output, controls, measured)
            rows = {(r["train_scheme"], r["variant"]): r for r in csv_rows(output / "noise_runs.csv")}
            summary = {r["scheme"]: r for r in csv_rows(output / "noise_summary.csv")}
            self.assertEqual(len(rows), 6)
            self.assertAlmostEqual(float(summary["bf16"]["repeat_minus_historical_ce"]), .01)
            self.assertAlmostEqual(float(summary["bf16"]["lr_up_minus_down_ce"]), .08)
            self.assertAlmostEqual(float(rows["bf16", "lr_x0p9999"]["delta_ce_vs_repeat"]), -.06)
            self.assertAlmostEqual(float(summary["bf16"]["max_abs_perturbation_delta_ce_vs_repeat"]), .06)
            self.assertEqual(summary["bf16"]["perturbations_measured"], "2")
            self.assertEqual(summary["mxfp8"]["perturbations_measured"], "0")
            self.assertAlmostEqual(float(rows["bf16", "warmup26_lr_x1p0001"]["warmup_minus_no_warmup_same_lr_ce"]), -.22)
            self.assertAlmostEqual(float(rows["bf16", "warmup26_lr_x1p0001"]["delta_ce_vs_base_lr_same_schedule"]), .06)

    def test_saved_panels_are_merged_when_current_and_intact(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = output / "src"
            source.write_bytes(b"valid")
            paths = {"src": source}
            plan = dict(files={"src": noise.digest(source)}, seed=42, architecture="moe")
            work = output / "arms/bf16/repeat"
            arm_plan = dict(plan, scheme="bf16", variant="repeat")
            noise.prepare_arm(work, paths, arm_plan)
            checkpoint, export, result = arm_paths(work, "moe", 42, "bf16")
            for artifact in (checkpoint, export, result):
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.touch()
            for artifact in (checkpoint, export):
                os.utime(artifact, ns=(1, 1))
            ce, ppl = scores(4.28)
            result.write_text(json.dumps(dict(architecture="moe", seed=42, train_scheme="bf16", eval_scheme="bf16",
                                              documents=1024, valid_tokens=236442, seq_len=512, batch_size=2,
                                              valid_sha256=noise.SPLITS["valid"], checkpoint=export.name,
                                              mean_ce=ce, ppl=ppl)))
            self.assertEqual(noise.existing_scores(output, paths, plan), {("bf16", "repeat"): (ce, ppl)})
            result.write_text("{}")
            self.assertEqual(noise.existing_scores(output, paths, plan), {})
            (work / "src").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Staged source"):
                noise.prepare_arm(work, paths, arm_plan)


if __name__ == "__main__":
    unittest.main()
