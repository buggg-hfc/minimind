from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import minimind_noise_arm as arm
from trainer.trainer_utils import get_lr


class NoiseArmTests(unittest.TestCase):
    def test_one_coordinate_one_ulp_without_advancing_rng(self):
        for initial in (0.01999, -0.01999, 0.03125, -0.03125):
            w = torch.full((3, 2), initial)
            before = w.clone()
            model = SimpleNamespace(get_parameter=lambda name: w if name == arm.COORDINATE else None)
            rng = torch.get_rng_state().clone()
            audit = arm.shift_initial_coordinate(model)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual((w != before).nonzero().tolist(), [[0, 0]])
            self.assertEqual(audit["fp32_after"] - audit["fp32_before"], audit["spacing"])
            self.assertEqual(audit["bf16_after"] - audit["bf16_before"], audit["spacing"])
            self.assertEqual(w[0, 0].bfloat16().item(), audit["bf16_after"])

    def test_launch_only_changes_registered_learning_rate(self):
        original = ["--seed", "42", "--learning_rate", "0.0005", "--accumulation_steps", "4"]
        for key, multiplier in arm.LR_MULTIPLIERS.items():
            args = arm.noise_launcher(lambda *args: list(args), "worker.py", key, "trainer.py", *original)
            self.assertEqual(args[:4], ["worker.py", "--rank-worker", "trainer.py", key])
            native = args[4:]
            self.assertEqual(float(native[3]), .0005*multiplier)
            self.assertEqual(native[:3] + native[4:], original[:3] + original[4:])

    def test_checkpoint_verification_uses_variant_schedule_and_adam_cursor(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "resume.pth"
            for step in (4, 100, 104, 256, 2048):
                for variant in arm.LR_MULTIPLIERS:
                    base = arm.BASE_LR * arm.LR_MULTIPLIERS[variant]
                    lr = (get_lr(104, 2048, base) * (step / 104)
                          if step < 104 and variant.startswith("warmup26") else get_lr(step, 2048, base))
                    state = dict(epoch=0, step=step, world_size=8, quant_sim="bf16",
                                 model={"weight": torch.tensor(1.)},
                                 optimizer={"param_groups": [{"lr": lr}], "state": {
                                     0: {"exp_avg": torch.tensor(0.), "exp_avg_sq": torch.tensor(0.),
                                         "step": step//4}}})
                    torch.save(state, file)
                    self.assertEqual(arm.checkpoint_state(file, "bf16", variant)["step"], step)
                    if arm.LR_MULTIPLIERS[variant] != 1 or (variant == "warmup26" and step < 104):
                        with self.assertRaisesRegex(ValueError, "learning rate"):
                            arm.checkpoint_state(file, "bf16", "repeat")
                    state["optimizer"]["state"][0]["step"] -= 1
                    torch.save(state, file)
                    with self.assertRaisesRegex(ValueError, "AdamW"):
                        arm.checkpoint_state(file, "bf16", variant)


if __name__ == "__main__":
    unittest.main()
