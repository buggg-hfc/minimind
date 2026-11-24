"""Paired 4-bit/MXFP8 expert ablations for each nonempty projection subset."""
import itertools
import unittest

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import gemm, install_quantization
from model.quant_policy import (FFN_MXFP8_GROUPS, FOUR_BIT_SCHEMES, MIXED_SCHEMES, RESIDUAL_POLICY,
                                ffn_mxfp8_scheme, projection_policy, scheme_artifact_name, validate_resume_policy)


PROJECTIONS = ("up", "gate", "down")
SUBSETS = [names for count in (1, 2, 3) for names in itertools.combinations(PROJECTIONS, count)]


class ExpertProjectionSubsetsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=48, use_moe=True,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            max_position_embeddings=32, flash_attn=False,
        ))

    def test_all_policies_have_unique_artifacts_and_install_on_every_expert(self):
        groups = ["all" if len(names) == 3 else "_".join(names) for names in SUBSETS]
        self.assertEqual(tuple(groups), FFN_MXFP8_GROUPS)
        artifacts = set()
        for base, (group, names) in itertools.product(FOUR_BIT_SCHEMES, zip(groups, SUBSETS)):
            scheme = ffn_mxfp8_scheme(base, group)
            with self.subTest(scheme=scheme):
                expected = {name + "_proj": "mxfp8" if name in names else base for name in PROJECTIONS}
                if group == "all":
                    self.assertEqual(scheme, f"{base}_ffn_mxfp8")
                    metadata = {"attention": base, "ffn": "mxfp8", "router": "bf16", "lm_head": "bf16"}
                else:
                    self.assertEqual(scheme, f"{base}_ffn_{group}_mxfp8")
                    metadata = {"attention": base, "ffn": expected, "router": "bf16", "lm_head": "bf16"}
                self.assertEqual(MIXED_SCHEMES[scheme], base)
                self.assertEqual(projection_policy(scheme), metadata)
                validate_resume_policy({"quant_sim": scheme, "projection_policy": metadata, "seed": 42,
                                        "residual_policy": RESIDUAL_POLICY}, scheme, seed=42)
                self.assertEqual(scheme_artifact_name(scheme), scheme + "_r2")
                artifacts.add(scheme_artifact_name(scheme))

                install_quantization(self.model, scheme)
                for block in self.model.model.layers:
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        self.assertEqual(getattr(block.self_attn, name)._simulation_scheme, base)
                    self.assertTrue(block.self_attn.simulate_bf16_attention)
                    self.assertEqual(block.mlp.gate._simulation_scheme, "bf16")
                    for expert in block.mlp.experts:
                        for leaf, precision in expected.items():
                            self.assertEqual(getattr(expert, leaf)._simulation_scheme, precision)
                self.assertEqual(self.model.lm_head._simulation_scheme, "bf16")
        self.assertEqual(len(artifacts), 21)

    def test_partial_expert_forward_dx_dw_and_cache_match_primitive_gemms(self):
        def bf16(values):
            return values.bfloat16().float()

        expert = self.model.model.layers[0].mlp.experts[0]
        for base in FOUR_BIT_SCHEMES:
            scheme = ffn_mxfp8_scheme(base, "gate_down")
            install_quantization(self.model, scheme, weight_cache_mib=2)
            for name, fmt in (("up_proj", base), ("gate_proj", "mxfp8"), ("down_proj", "mxfp8")):
                with self.subTest(scheme=scheme, projection=name):
                    layer = getattr(expert, name)
                    layer.train()
                    x = torch.randn(8, layer.in_features, requires_grad=True)
                    dy = torch.randn(8, layer.out_features) * 0.001
                    y = layer(x)
                    dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
                    with torch.no_grad():
                        a, w, g = map(bf16, (x, layer.weight, dy))
                        expected = bf16(gemm(a, w.T, fmt))
                        torch.testing.assert_close(y, expected, rtol=0, atol=0)
                        torch.testing.assert_close(dx, bf16(gemm(g, w, fmt, lhs_is_dy=True)), rtol=0, atol=0)
                        torch.testing.assert_close(dw, bf16(gemm(g.T, a, fmt, lhs_is_dy=True)), rtol=0, atol=0)
                        layer.eval()
                        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
                        self.assertIsNotNone(layer._weight_cache)
                        self.assertEqual(layer._weight_cache[2][1] is None, fmt == "mxfp8")
