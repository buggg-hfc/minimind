import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from minimind_noise_lr import with_linear_warmup


def cosine_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


class NoiseWarmupTests(unittest.TestCase):
    def test_full_pilot_ramp_join_and_final_for_all_peaks(self):
        schedule = with_linear_warmup(cosine_lr)
        for factor in (1., .9999, 1.0001):
            peak = 5e-4 * factor
            with self.subTest(peak=peak):
                rates = [schedule(step, 2048, peak) for step in range(1, 2049)]
                endpoint = cosine_lr(104, 2048, peak)
                self.assertEqual(schedule(0, 2048, peak), 0.)
                self.assertTrue(all(rate > 0. for rate in rates))
                self.assertAlmostEqual(rates[0], endpoint / 104, places=18)
                self.assertAlmostEqual(rates[3], endpoint / 26, places=18)
                for update in range(1, 26):
                    self.assertAlmostEqual(rates[4 * update - 1], endpoint * update / 26, places=18)
                self.assertEqual(rates[103:], [cosine_lr(step, 2048, peak) for step in range(104, 2049)])
                self.assertEqual(rates[-1], peak * .1)

    def test_resuming_uses_global_microbatch_cursor(self):
        schedule = with_linear_warmup(cosine_lr)
        uninterrupted = [schedule(step, 2048, 5e-4) for step in range(1, 2049)]
        for cursor in (4, 100, 104, 256, 2044):
            resumed = with_linear_warmup(cosine_lr)
            self.assertEqual(uninterrupted[cursor:],
                             [resumed(step, 2048, 5e-4) for step in range(cursor + 1, 2049)])

    def test_join_calls_native_without_shift(self):
        seen = []

        def native(step, total, base):
            seen.append((step, total, base))
            return base

        schedule = with_linear_warmup(native)
        self.assertEqual(schedule(52, 2048, .125), .0625)
        self.assertEqual(schedule(104, 2048, .125), .125)
        self.assertEqual(schedule(105, 2048, .125), .125)
        self.assertEqual(seen, [(104, 2048, .125), (104, 2048, .125), (105, 2048, .125)])


if __name__ == "__main__":
    unittest.main()
