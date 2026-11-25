import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.minimind_weight_kl import compare_weight_states, load_weight_state


class EndpointWeightKLTest(unittest.TestCase):
    def test_native_export_per_tensor_direction_weighting_and_tied_head(self):
        # Constant router vs. changed expert and physically separate copies of
        # the tied embedding/head recreate the native CPU export layout.
        expert = "model.layers.6.mlp.experts.2.down_proj.weight"
        router = "model.layers.6.mlp.gate.weight"
        embed, head = "model.embed_tokens.weight", "lm_head.weight"
        reference = {expert: torch.tensor([[-1., -1., -1., 1.]]), router: torch.zeros((1, 3)),
                     embed: torch.ones((2, 4)), head: torch.ones((2, 4))}
        candidate = {**reference, expert: torch.tensor([[-1., 0., 1., 1.]])}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.pth"
            torch.save(reference, p)
            summary, rows = compare_weight_states(load_weight_state(p), candidate)
        self.assertEqual((summary["weight_elements"], summary["weight_tensors"]), (15, 3))
        row = next(item for item in rows if item["component"] == "down_proj")
        self.assertEqual((row["module"], row["layer"], row["expert"], row["elements"]),
                         (expert[:-7], 6, 2, 4))
        self.assertNotIn(head[:-7], {item["module"] for item in rows})
        p, q = np.full(128, .5), np.full(128, .5)
        p[[0, 127]] += [3, 1]
        q[[0, 64, 127]] += [1, 1, 2]
        p, q = p / 68., q / 68.
        fwd, rev = p @ np.log(p / q), q @ np.log(q / p)
        self.assertNotAlmostEqual(fwd, rev)
        for key, expected in (("kl_forward", fwd), ("kl_reverse", rev), ("kl_jeffreys", fwd + rev)):
            self.assertAlmostEqual(row[key], expected, places=12)
            self.assertAlmostEqual(summary[f"weight_{key}"], 4 * expected / 15, places=12)

    def test_self_and_permuted_weights_have_identical_distributions(self):
        values = torch.tensor([[-8., 2.5, 0., 3.]])
        for candidate in (values, values.flip([1])):
            name = "model.layers.0.self_attn.q_proj.weight"
            result, _ = compare_weight_states({name: values}, {name: candidate})
            self.assertEqual(result["weight_kl_forward"], 0.)
            self.assertEqual(result["weight_kl_reverse"], 0.)

    def test_invalid_pairs_fail(self):
        ref = {"a.weight": torch.zeros(4)}
        for other in ({"a.weight": torch.zeros(2)}, {"b.weight": torch.zeros(4)},
                      {"a.weight": torch.tensor([1., math.nan, 2., 3.])}):
            with self.assertRaises(ValueError):
                compare_weight_states(ref, other)

    def test_resume_or_broken_tied_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "invalid.pth"
            for obj in ({"model": {"projection.weight": torch.ones(1)}, "optimizer": {}},
                        {"model.embed_tokens.weight": torch.ones(2), "lm_head.weight": torch.zeros(2)},
                        {"projection.weight": torch.zeros(4, dtype=torch.long)}):
                torch.save(obj, path)
                with self.assertRaisesRegex(ValueError, str(path)):
                    load_weight_state(path)


if __name__ == "__main__":
    unittest.main()
