"""Mixed-precision pilot controls and multi-case resume boundaries."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model.quant_policy import FOUR_BIT_SCHEMES, MIXED_SCHEMES, RESIDUAL_POLICY, projection_policy, scheme_artifact_name
from scripts import minimind_v100_pilot as pilot


class MixedPilotTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pilot_dir = self.root / "runs/v100-pilot"
        data = self.pilot_dir / "data"
        data.mkdir(parents=True)
        self.row = b'{"text":"sample"}\n'
        (data / "train.jsonl").write_bytes(self.row * 32)  # Two eight-rank microbatches.
        (data / "valid.jsonl").write_bytes(self.row)
        (self.root / "out").mkdir()
        (self.root / "checkpoints").mkdir()

    def paths(self, scheme, seed):
        name = f"pilot32k_s{seed}_{scheme_artifact_name(scheme)}_768_moe"
        return self.root / "out" / f"{name}.pth", self.root / "checkpoints" / f"{name}_resume.pth"

    def save_progress(self, scheme, seed, step):
        weights, state = self.paths(scheme, seed)
        weights.touch()
        torch.save({"quant_sim": scheme, "projection_policy": projection_policy(scheme),
                    "residual_policy": RESIDUAL_POLICY, "seed": seed, "epoch": 0,
                    "world_size": 8, "step": step}, state)

    def test_suite_resumes_partial_and_starts_missing_after_skipping_complete(self):
        legacy_cases = [base + "_ffn_mxfp8" for base in FOUR_BIT_SCHEMES]
        done, partial, fresh = legacy_cases
        seed = 43
        self.save_progress(done, seed, 2)
        self.save_progress(partial, seed, 1)
        with patch.object(pilot.subprocess, "Popen") as launcher, patch.object(pilot.subprocess, "run") as evaluate:
            def finish_training(cmd, **kwargs):
                scheme = cmd[cmd.index("--quant_sim") + 1]
                self.save_progress(scheme, seed, 2)
                return process

            process = launcher.return_value
            process.__enter__.return_value.wait.return_value = 0
            process.__enter__.return_value.stdout = ["loss: 1.0, logits_loss: 1.0\n"]
            launcher.side_effect = finish_training
            pilot.moe_ffn_mxfp8(self.root, seed, resume=True)
            commands = [call.args[0] for call in launcher.call_args_list]
            self.assertEqual([(c[c.index("--quant_sim") + 1], c[c.index("--from_resume") + 1]) for c in commands],
                             [(partial, "1"), (fresh, "0")])
            for cmd in commands:
                self.assertEqual(cmd[cmd.index("--seed") + 1], "43")
                self.assertEqual(cmd[cmd.index("--from_weight") + 1], "none")
                self.assertEqual(cmd[cmd.index("--save_weight") + 1], "pilot32k_s43")
                self.assertIn("--nproc-per-node=8", cmd)
            eval_commands = [call.args[0] for call in evaluate.call_args_list]
            self.assertEqual([(c[c.index("eval") + 1], c[c.index("--eval-scheme") + 1]) for c in eval_commands],
                             [(scheme, mode) for scheme in legacy_cases for mode in ("bf16", "same")])
            for cmd in eval_commands:
                self.assertEqual(cmd[cmd.index("--seed") + 1], "43")
                self.assertIn("--moe", cmd)
                self.assertIn("--reuse", cmd)

    def test_grid_schedules_each_case_once_and_each_case_gets_both_evals(self):
        groups = ("up", "gate", "down", "up_gate", "up_down", "gate_down", "all")
        cases = [f"{base}_ffn_mxfp8" if group == "all" else f"{base}_ffn_{group}_mxfp8"
                 for group in groups for base in FOUR_BIT_SCHEMES]
        with patch.object(pilot, "train") as train, patch.object(pilot.subprocess, "run") as evaluate:
            pilot.moe_ffn_mxfp8(self.root, 42, resume=True, groups=["grid"])
        self.assertEqual([call.args[2] for call in train.call_args_list], cases)
        self.assertEqual(len(set(cases)), 21)
        self.assertTrue(all(c.kwargs == {"seed": 42, "resume_if_present": True} for c in train.call_args_list))
        commands = [c.args[0] for c in evaluate.call_args_list]
        self.assertEqual([(cmd[cmd.index("eval") + 1], cmd[cmd.index("--eval-scheme") + 1]) for cmd in commands],
                         [(scheme, mode) for scheme in cases for mode in ("bf16", "same")])

    def test_cli_can_run_selected_groups_for_one_format(self):
        for name in ("model/quant_policy.py", "tests/test_v100_batch_inference.py"):
            marker = self.root / name
            marker.parent.mkdir()
            marker.touch()
        argv = ["pilot", "--root", str(self.root), "moe-ffn-mxfp8", "--groups", "up,down", "up_down", "up",
                "--formats", "nvint4_3mm", "--resume"]
        with patch.object(sys, "argv", argv), patch.object(sys, "path", sys.path.copy()), \
                patch.object(pilot, "train") as train, patch.object(pilot.subprocess, "run"):
            pilot.main()
        self.assertEqual([c.args[2] for c in train.call_args_list],
                         [f"nvint4_3mm_ffn_{group}_mxfp8" for group in ("up", "down", "up_down")])
        with patch.object(pilot, "train") as train, self.assertRaisesRegex(SystemExit, "bad_group"):
            pilot.moe_ffn_mxfp8(self.root, 42, resume=True, groups=["up", "bad_group"])
        train.assert_not_called()

    def test_rejects_dense_or_orphaned_mixed_run_and_wrong_seed(self):
        mixed = next(iter(MIXED_SCHEMES))
        with self.assertRaisesRegex(ValueError, "requires --moe"):
            pilot.train(self.root, False, mixed)
        with self.assertRaisesRegex(ValueError, "requires --moe"):
            pilot.evaluate(self.root, False, mixed, "same")
        weights, state = self.paths(mixed, 42)
        weights.touch()
        with self.assertRaisesRegex(SystemExit, "orphaned"):
            pilot.train(self.root, True, mixed, resume_if_present=True)
        self.save_progress(mixed, 43, 2)
        self.paths(mixed, 43)[1].rename(state)
        with self.assertRaisesRegex(ValueError, "Seed mismatch"):
            pilot.train(self.root, True, mixed, resume_if_present=True)

    def test_reused_eval_artifacts_are_scoped_to_seed_and_forward_policy(self):
        mixed = next(iter(MIXED_SCHEMES))
        for seed, forward in ((42, "bf16"), (43, "same")):
            weights, _ = self.paths(mixed, seed)
            weights.touch()
            os.utime(weights, (1, 1))
            scheme = mixed if forward == "same" else "bf16"
            directory = self.pilot_dir / "results"
            if seed != 42:
                directory /= f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            result = {"architecture": "moe", "seed": seed, "train_scheme": mixed, "eval_scheme": scheme,
                      "checkpoint": weights.name, "valid_sha256": hashlib.sha256(self.row).hexdigest(),
                      "seq_len": 512, "batch_size": 2, "documents": 1, "valid_tokens": 7,
                      "mean_ce": 4.0, "ppl": 54.6, "residual_policy": RESIDUAL_POLICY,
                      "train_projection_policy": projection_policy(mixed),
                      "eval_projection_policy": projection_policy(scheme)}
            if forward == "same":
                result["eval_residual_policy"] = RESIDUAL_POLICY
            dest = directory / f"moe__{scheme_artifact_name(mixed)}__eval_{scheme}.json"
            dest.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(pilot.evaluate(self.root, True, mixed, forward, seed=seed, reuse=True), result)
