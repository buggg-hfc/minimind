import unittest

import numpy as np
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import _bf16_value, gemm, install_quantization
from scripts.moe_diagnostic_ops import OperatorProbe, SIGN_CLASSES, _PairedStats


class ProbeStatsTest(unittest.TestCase):
    def test_signed_errors_and_exclusive_signs_pool_finite_pairs(self):
        ref = np.array([1., -1., 1., -1., 1., 0., np.nan])
        est = np.array([.5, -1., 2., 0., -2., 2., 4.])
        stats = _PairedStats()
        for ix in (slice(3), slice(3, None)):
            stats.add(ref[ix], est[ix], 2, ref[ix].size)
        s = stats.result()
        error = est[:-1] - ref[:-1]
        self.assertEqual(s["finite_count"], 6)
        self.assertEqual(s["nonfinite_count"], 1)
        self.assertEqual(s["ref_zero_est_nonzero_count"], 1)
        self.assertEqual([s[k + "_count"] for k in SIGN_CLASSES], [1] * 5)
        self.assertAlmostEqual(s["signed_mean"], error.mean())
        self.assertAlmostEqual(s["mae"], np.abs(error).mean())
        self.assertAlmostEqual(s["relative_mae"], np.abs(error[:5]).mean())
        self.assertAlmostEqual(s["variance_retention"], est[:-1].var() / ref[:-1].var())
        self.assertEqual(s["ulp_mean"], (np.abs(error[:5]).sum() * 128 + 2**134) / 6)

    def test_zero_undefined_metrics(self):
        s = _PairedStats()
        self.assertIsNone(s.result()["signed_mean"])
        s.add(np.zeros(3), np.zeros(3), 3, 3)
        out = s.result()
        self.assertEqual(out["ulp_mean"], 0.)
        self.assertEqual(out["operator_hist_kl"], 0.)
        self.assertIsNone(out["relative_mae"])
        self.assertIsNone(out["variance_retention"])
        self.assertIsNone(out["zeroed_rate"])

    def test_histogram_kl_direction_and_weighted_event_averaging(self):
        stats = _PairedStats()
        self.assertIsNone(stats.result()["operator_hist_kl"])
        stats.add(np.array([0., 0., 1.]), np.array([1., 1., 1.]), 3, 3)
        # Counts differ only in bins 0 and 127, and total mass after smoothing is 67.
        expected = (2.5 * np.log(2.5 / .5) + 1.5 * np.log(1.5 / 3.5)) / 67
        self.assertAlmostEqual(stats.result()["operator_hist_kl"], expected)
        stats.add(np.array([0., 1.]), np.array([0., 1.]), 2, 2)
        self.assertAlmostEqual(stats.result()["operator_hist_kl"], expected * 3 / 5)
        self.assertEqual(stats.result()["min_input_rows"], 2)
        self.assertEqual(stats.result()["max_input_rows"], 3)

    def test_pool_matches_direct_across_experts_with_different_mean_gradients(self):
        events = [(np.array([-1., 3.]), np.array([0., -4.])),
                  (np.array([100., 102., 0.]), np.array([80., 104., 2.]))]
        pooled, direct = _PairedStats(), _PairedStats()
        for ref, est in events:
            local = _PairedStats()
            local.add(ref, est, ref.size, ref.size)
            pooled.merge(local)
            direct.add(ref, est, ref.size, ref.size)
        pooled.merge(_PairedStats())
        for field, want in direct.result().items():
            self.assertAlmostEqual(pooled.result()[field], want)
        r, e = (np.concatenate(tensors) for tensors in zip(*events))
        self.assertAlmostEqual(pooled.result()["variance_retention"], e.var() / r.var())


class OperatorProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        m = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=32, num_hidden_layers=1, vocab_size=48, use_moe=True,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            max_position_embeddings=32, flash_attn=False,
        ))
        self.model = install_quantization(m, "bf16").eval()

    def test_same_operands_for_all_gemms_and_probe_does_not_change_gradients(self):
        ids = torch.randint(0, 48, (2, 8))
        proj = self.model.model.layers[0].self_attn.q_proj
        loss = self.model(ids, labels=ids).loss
        loss.backward()
        grad = proj.weight.grad.clone()
        self.model.zero_grad(set_to_none=True)
        observed = {}

        def capture(module, inputs, output):
            observed["x"] = inputs[0].detach()
            def save_dy(dy):
                observed["dy"] = dy.detach()
            output.register_hook(save_dy)

        handle = proj.register_forward_hook(capture)
        p = OperatorProbe(self.model, "nvint4_3mm_ffn_up_mxfp8", [0], max_elements=100_000)
        with p:
            self.model(ids, labels=ids).loss.backward()
        handle.remove()
        torch.testing.assert_close(proj.weight.grad, grad, rtol=0, atol=0)
        x = _bf16_value(observed["x"].reshape(-1, proj.in_features))
        dy = _bf16_value(observed["dy"].reshape(-1, proj.out_features))
        w = _bf16_value(proj.weight.detach())
        rows = {(r["gemm"], r["stage"]): r for r in p.rows() if r["projection"] == "q_proj"}
        for region, lhs, rhs in (("FWD", x, w.T), ("dX", dy, w), ("dW", dy.T, x)):
            ref = gemm(lhs, rhs, "bf16")
            est = gemm(lhs, rhs, "nvint4_3mm", lhs_is_dy=region != "FWD")
            for stage in ["raw", "applied"]:
                if stage == "applied":
                    ref, est = _bf16_value(ref), _bf16_value(est)
                r = rows[region, stage]
                self.assertEqual(r["finite_count"], ref.numel())
                self.assertEqual(r["input_rows"], 16)
                self.assertEqual(r["calls"], 1)
                self.assertAlmostEqual(r["mae"], (est.double() - ref.double()).abs().mean().item())
        for r in p.rows():
            if r["expert"] is not None:
                expected = "mxfp8" if r["projection"] == "up_proj" else "nvint4_3mm"
                self.assertEqual(r["primitive"], expected)
        combined = next(r for r in p.component_rows() if
                        (r["projection"], r["gemm"], r["stage"]) == ("up_proj", "dW", "raw"))
        local = [r for r in p.rows() if
                 (r["projection"], r["gemm"], r["stage"]) == ("up_proj", "dW", "raw")]
        self.assertEqual(combined["finite_count"], sum(r["finite_count"] for r in local))
        self.assertEqual(combined["zeroed_count"], sum(r["zeroed_count"] for r in local))

    def test_sampling_and_clean_hook_removal(self):
        ids = torch.randint(0, 48, (2, 8))
        probes = [OperatorProbe(self.model, "hif4_thr_3mm", [0], max_elements=5) for _ in range(2)]
        for p in probes:
            with p, torch.no_grad():
                self.model(ids)
            self.assertTrue(all(r["sampled_count"] <= 5 and r["calls"] <= 1 for r in p.rows()))
        self.assertEqual(probes[0].rows(), probes[1].rows())

    def test_weight_gradient_replay_follows_the_mxfp8_override(self):
        # Observe identical BF16 operands for each policy, including the same sampled coordinates.
        ids = torch.randint(0, 48, (2, 8))
        results = {}
        for scheme in ("hif4_thr_3mm", "mxfp8", "hif4_thr_3mm_dw_mxfp8"):
            self.model.zero_grad(set_to_none=True)
            with OperatorProbe(self.model, scheme, [0], max_elements=16) as probe:
                self.model(ids, labels=ids).loss.backward()
            results[scheme] = (probe.rows(), probe.component_rows())
        for hybrid, base, mxfp8 in zip(results["hif4_thr_3mm_dw_mxfp8"],
                                       results["hif4_thr_3mm"], results["mxfp8"]):
            self.assertEqual(len(hybrid), len(base))
            self.assertEqual(len(hybrid), len(mxfp8))
            for row, base_row, mx_row in zip(hybrid, base, mxfp8):
                want = mx_row if row["gemm"] == "dW" else base_row
                self.assertEqual(row, {**want, "scheme": "hif4_thr_3mm_dw_mxfp8"})


if __name__ == "__main__":
    unittest.main()
