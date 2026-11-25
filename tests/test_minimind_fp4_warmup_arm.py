"""Exercise both warmup base rates against the exact native launcher and cursor checks."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import minimind_noise_arm as legacy
from minimind_fp4_warmup_arm import _worker_for
from model.quant_policy import RESIDUAL_POLICY

WORK = Path(legacy.__file__).resolve().parents[1]


def cosine(step, total, lr):
    return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * step / total)))


class FP4WarmupArmTests(unittest.TestCase):
    def test_launch_uses_each_lr_without_altering_legacy_worker(self):
        native_args = ["--quant_sim", "nvfp4_3mm", "--learning_rate", "0.0005", "--from_resume", "1"]
        for lr in (5e-4, 1e-4):
            with self.subTest(lr=lr):
                worker = _worker_for(WORK, lr)
                cmd = worker.noise_launcher(lambda *args: list(args), "worker.py", "warmup26",
                                            "trainer.py", *native_args)
                self.assertEqual(cmd[:4], ["worker.py", "--rank-worker", "trainer.py", "warmup26"])
                self.assertEqual(cmd[4:], [*native_args[:3], repr(lr), *native_args[4:]])
                self.assertEqual(legacy.LR_MULTIPLIERS["warmup26"], 1.0)
                self.assertEqual(native_args[3], "0.0005")

    def test_partial_and_final_checkpoints_match_own_lr_and_r2(self):
        scheme = "nvint4_3mm"
        for lr in (5e-4, 1e-4):
            worker = _worker_for(WORK, lr)
            for step in (4, 104, 256, 2048):
                with self.subTest(lr=lr, step=step):
                    target = cosine(max(104, step), 2048, lr) * min(step / 104, 1)
                    param_group = {"lr": target}
                    state = {"quant_sim": scheme, "residual_policy": RESIDUAL_POLICY, "model": {"w": 1},
                             "seed": 42, "step": step, "epoch": 0, "world_size": 8,
                             "optimizer": {"state": {0: {"exp_avg": 1, "exp_avg_sq": 1, "step": step // 4}},
                                           "param_groups": [param_group]}}
                    dependencies = {"torch": SimpleNamespace(load=lambda *args, **kwargs: state),
                                    "trainer.trainer_utils": SimpleNamespace(get_lr=cosine)}
                    with patch.dict(sys.modules, dependencies):
                        self.assertIs(worker.checkpoint_state("test.pth", scheme, "warmup26"), state)
                        param_group["lr"] = target * 2
                        with self.assertRaises(ValueError):
                            worker.checkpoint_state("test.pth", scheme, "warmup26")
                        param_group["lr"] = target
                        state["residual_policy"] = "other"
                        with self.assertRaises(ValueError):
                            worker.checkpoint_state("test.pth", scheme, "warmup26")


if __name__ == "__main__":
    unittest.main()
