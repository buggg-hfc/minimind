"""Verify the expert-only MXFP8 ablation against the existing projection kernels."""
import unittest

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import gemm, install_quantization


BASE_SCHEMES = ("hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm")


class MixedMoEQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=48, use_moe=True,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            max_position_embeddings=32, flash_attn=False,
        ))

    def test_all_experts_and_attention_projections_use_the_intended_format(self):
        keys = set(self.model.state_dict())
        for base in BASE_SCHEMES:
            with self.subTest(base=base):
                install_quantization(self.model, f"{base}_ffn_mxfp8")
                for block in self.model.model.layers:
                    self.assertTrue(block.self_attn.simulate_bf16_attention)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        self.assertEqual(getattr(block.self_attn, name)._simulation_scheme, base)
                    self.assertEqual(block.mlp.gate._simulation_scheme, "bf16")
                    for expert in block.mlp.experts:
                        for name in ("up_proj", "gate_proj", "down_proj"):
                            self.assertEqual(getattr(expert, name)._simulation_scheme, "mxfp8")
                self.assertEqual(self.model.lm_head._simulation_scheme, "bf16")
                self.assertIs(self.model.lm_head.weight, self.model.model.embed_tokens.weight)
                self.assertEqual(set(self.model.state_dict()), keys)

        install_quantization(self.model, "bf16")
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertEqual(module._simulation_scheme, "bf16")

    def test_expert_ablation_rejects_dense_configuration(self):
        dense = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=32, num_hidden_layers=1, vocab_size=48, max_position_embeddings=32,
        ))
        for base in BASE_SCHEMES:
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, "MoE"):
                install_quantization(dense, f"{base}_ffn_mxfp8")

    def test_forward_gradients_and_cached_eval_match_the_component_gemms(self):
        def bf16(x):
            return x.bfloat16().float()

        block = self.model.model.layers[0]
        for base in BASE_SCHEMES:
            install_quantization(self.model, f"{base}_ffn_mxfp8", weight_cache_mib=1)
            for layer, scheme in ((block.self_attn.q_proj, base),
                                  (block.mlp.experts[-1].gate_proj, "mxfp8")):
                with self.subTest(attention=base, component=scheme):
                    self.model.train()
                    x = torch.randn(9, layer.in_features, requires_grad=True)
                    dy = torch.randn(9, layer.out_features) * 2.0 ** -24
                    y = layer(x)
                    dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
                    with torch.no_grad():
                        a, w, g = bf16(x), bf16(layer.weight), bf16(dy)
                        want_y = bf16(gemm(a, w.T, scheme))
                        want_dx = bf16(gemm(g, w, scheme, lhs_is_dy=True))
                        want_dw = bf16(gemm(g.T, a, scheme, lhs_is_dy=True))
                        torch.testing.assert_close(y, want_y, rtol=0, atol=0)
                        torch.testing.assert_close(dx, want_dx, rtol=0, atol=0)
                        torch.testing.assert_close(dw, want_dw, rtol=0, atol=0)

                        self.model.eval()
                        torch.testing.assert_close(layer(x), want_y, rtol=0, atol=0)
                        self.assertIsNotNone(layer._weight_cache)
                        # Only the attention projection retains a residual weight in this pair.
                        residual = layer._weight_cache[2][1]
                        self.assertEqual(residual is None, scheme == "mxfp8")


if __name__ == "__main__":
    unittest.main()
