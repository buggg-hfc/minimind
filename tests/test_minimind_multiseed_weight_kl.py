import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import minimind_multiseed_weight_kl as cli


class EndpointDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.grid = self.base / "grid"
        self.legacy = self.base / "legacy"

    def make_export(self, work, seed, scheme, ce=4.2):
        _, export, score = cli.arm_paths(work, "moe", seed, scheme)
        export.parent.mkdir(parents=True, exist_ok=True)
        export.touch()
        score.parent.mkdir(parents=True, exist_ok=True)
        row = dict(architecture="moe", train_scheme=scheme, eval_scheme=scheme,
                   documents=1024, valid_tokens=236442, seq_len=512, batch_size=2,
                   checkpoint=export.name, valid_sha256=cli.SPLITS["valid"], mean_ce=ce, ppl=math.exp(ce))
        if seed != 42:
            row["seed"] = seed
        if scheme in cli.RESIDUAL_SCHEMES:
            row.update(residual_policy=cli.RESIDUAL_POLICY, eval_residual_policy=cli.RESIDUAL_POLICY)
        score.write_text(json.dumps(row))
        return export, score, row

    def test_legacy_with_missing_seed_and_r2_policy(self):
        export, score, row = self.make_export(self.legacy, 42, "nvfp4_3mm")
        end = cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "nvfp4_3mm",
                                     historical=(4.2, math.exp(4.2)))
        self.assertEqual(end.path, export)
        del row["residual_policy"]
        score.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "residual_policy"):
            cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "nvfp4_3mm")

    def test_grid_index_selects_original_source(self):
        work = self.base / "other-paired" / "work"
        export, score, row = self.make_export(work, 43, "hif4_thr_3mm")
        self.make_export(self.grid / "work", 43, "hif4_thr_3mm", ce=4.3)
        self.grid.mkdir(exist_ok=True)
        path = self.grid / "same_format.csv"
        row.update(optimizer_updates="512", source_work=str(work))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        key = 43, "moe", "hif4_thr_3mm"
        row = cli.grid_index(path)[key]
        self.assertEqual(cli.available_endpoint(self.grid, self.legacy, None, row, *key).path, export)
        score.unlink()  # complete grid CSV records this export
        self.assertEqual(cli.available_endpoint(self.grid, self.legacy, None, row, *key).path, export)
        export.unlink()  # another source has an export of the same name, but a different score
        with self.assertRaisesRegex(ValueError, "disagree"):
            cli.available_endpoint(self.grid, self.legacy, None, row, *key)

    def test_missing_or_rewritten_endpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "pilot32k_s42_nvint4_3mm_r2_768_moe.pth"):
            cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "nvint4_3mm")
        export, score, _ = self.make_export(self.legacy, 42, "mxfp8")
        os.utime(export, ns=(score.stat().st_mtime_ns + 10000,) * 2)
        with self.assertRaisesRegex(FileNotFoundError, "Cannot verify"):
            cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "mxfp8",
                                   historical=(4.2, math.exp(4.2)))

        resume, _, _ = cli.arm_paths(self.legacy, "moe", 42, "mxfp8")
        resume.parent.mkdir(parents=True)
        resume.touch()
        for archive in (False, True):
            if archive:
                export.unlink()
            end = cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "mxfp8",
                                         historical=(4.2, math.exp(4.2)))
            self.assertEqual(end.path, resume)

    def test_loads_verified_resume_model_instead_of_replaced_export(self):
        import torch

        export, score, _ = self.make_export(self.legacy, 42, "mxfp8")
        resume, _, _ = cli.arm_paths(self.legacy, "moe", 42, "mxfp8")
        name = "model.layers.0.q_proj.weight"
        state = dict(world_size=8, epoch=0, step=2048, quant_sim="mxfp8",
                     model={name: torch.full((3,), 7.)}, optimizer={
                         "state": {0: {"exp_avg": torch.zeros(3), "exp_avg_sq": torch.zeros(3), "step": 512}},
                         "param_groups": [{"lr": 5e-5}]})
        resume.parent.mkdir(exist_ok=True)
        torch.save(state, resume)
        torch.save({name: torch.zeros(3)}, export)
        os.utime(export, ns=(score.stat().st_mtime_ns + 10000,) * 2)
        end = cli.available_endpoint(self.grid, self.legacy, None, None, 42, "moe", "mxfp8",
                                     historical=(4.2, math.exp(4.2)))
        self.assertEqual(end.path, resume)
        self.assertEqual(cli.load_endpoint(end, 42, "mxfp8")[name].tolist(), [7., 7., 7.])
        state["step"] = 1024
        torch.save(state, resume)
        with self.assertRaisesRegex(ValueError, "512"):
            cli.load_endpoint(end, 42, "mxfp8")


if __name__ == "__main__":
    unittest.main()
