import copy
import math
import unittest

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import install_quantization
from scripts.moe_diagnostic_kl import _backbone, compare_forwards, weight_kl_rows


class PairedModelDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MiniMindForCausalLM(MiniMindConfig(
            use_moe=True, num_hidden_layers=2, hidden_size=32, vocab_size=64,
            intermediate_size=64, max_position_embeddings=64)).eval()
        self.tokens = torch.tensor([[1, 3, 4, 5, 2, 0], [1, 8, 2, 0, 0, 0]])
        self.labels = self.tokens.masked_fill(self.tokens == 0, -100)
        self.batches = [(self.tokens, self.labels)]

    def test_self_identity_shifted_tokens_and_ties_match_native_pilot(self):
        with torch.no_grad():
            for block in self.model.model.layers:
                block.mlp.gate.weight.zero_()  # argmax and native topk may disagree on ties.
            install_quantization(self.model, "bf16")
            out = self.model(self.tokens, labels=self.labels)
        report = compare_forwards(self.model, self.batches, "bf16", token_chunk=2)
        s = report["summary"]
        self.assertEqual(s["valid_tokens"], 6)
        self.assertEqual([r["valid_tokens"] for r in report["documents"]], [4, 2])
        self.assertAlmostEqual(s["mean_ce_bf16"], out.loss.item(), places=5)
        self.assertEqual(s["mean_ce_bf16"], s["mean_ce_candidate"])
        self.assertEqual(s["next_token_kl"], 0)
        chosen = int(torch.full((4,), .25).topk(1, sorted=False).indices)
        for r in report["routing"]:
            self.assertEqual(r["valid_tokens"], 6)
            self.assertEqual(r["top1_flips"], 0)
            self.assertEqual(r["router_kl"], 0)
            self.assertEqual(r["mean_bf16_margin"], 0)
        for r in report["expert_loads"]:
            self.assertEqual(r["bf16_tokens"], 6 if r["expert"] == chosen else 0)
        self.assertEqual(sum(r["tokens"] for r in report["margin_bins"]), 12)

    def test_cross_weight_full_vocab_and_per_document_pair(self):
        reference = copy.deepcopy(self.model)
        with torch.no_grad():
            self.model.lm_head.weight.mul_(1.4)
            install_quantization(reference, "bf16")
            install_quantization(self.model, "mxfp8")
            a, b = (m(self.tokens).logits[:, :-1] for m in (reference, self.model))
            keep = self.labels[:, 1:] != -100
            p, q = (x[keep].log_softmax(-1, dtype=torch.float64) for x in (a, b))
            expected = (p.exp() * (p-q)).sum(1).mean().item()
        r = compare_forwards(self.model, self.batches, "mxfp8", reference_model=reference, token_chunk=3)
        self.assertGreater(expected, 0.)
        self.assertAlmostEqual(r["summary"]["next_token_kl"], expected, places=8)
        self.assertAlmostEqual(sum(d["token_kl_sum"] for d in r["documents"])/6, expected, places=8)
        self.assertAlmostEqual(r["summary"]["ppl_candidate"], math.exp(r["summary"]["mean_ce_candidate"]))
        with self.assertRaisesRegex(ValueError, "identical weights"):
            compare_forwards(self.model, self.batches, "mxfp8", reference_model=reference, lock_routes=True)

    def test_forced_assignments_actually_dispatch_once_per_input_token(self):
        install_quantization(self.model, "bf16")
        with torch.no_grad():
            for block in self.model.model.layers:
                block.mlp.gate.weight.zero_()
            keep = self.labels[:, 1:] != -100
            normal, _, assignments = _backbone(self.model, self.tokens, keep, save_dispatch=True)
            forced = {l: (v + 1) % 4 for l, v in assignments.items()}
            calls = []
            first = self.model.model.layers[0].mlp
            handles = [expert.register_forward_pre_hook(
                lambda _m, inputs, i=i: calls.append((i, len(inputs[0])))) for i, expert in enumerate(first.experts)]
            try:
                changed, info, _ = _backbone(self.model, self.tokens, keep, force_dispatch=forced)
            finally:
                for h in handles:
                    h.remove()
        self.assertEqual(calls, [(int(forced[0][0]), self.tokens.numel())])
        self.assertFalse(torch.equal(normal, changed))
        actual, proposal, _, _ = info[0]
        self.assertTrue((actual != proposal).all())
        r = compare_forwards(self.model, self.batches, "hif4_thr_3mm", lock_routes=True)
        self.assertTrue(r["summary"]["lock_routes"])
        self.assertTrue(all(r["top1_flips"] == 0 for r in r["routing"]))

    def test_w_histograms_identical_shifted_and_layer_scoped(self):
        name = "model.layers.1.mlp.experts.2.down_proj.weight"
        a = {name: torch.ones(16, 3)}
        self.assertEqual(weight_kl_rows(a, a)[0]["kl_forward"], 0)
        b = {name: torch.zeros(16, 3)}
        row, = weight_kl_rows(a, b, layers=[1])
        self.assertEqual((row["component"], row["expert"], row["layer"]), ("down_proj", 2, 1))
        expected = 48 / (48+64) * math.log((48+.5)/.5)
        self.assertAlmostEqual(row["kl_forward"], expected)
        self.assertAlmostEqual(row["kl_reverse"], expected)
        self.assertEqual(weight_kl_rows(a, b, layers=[0]), [])
