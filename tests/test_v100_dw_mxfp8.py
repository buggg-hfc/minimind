"""Verify dW-only MXFP8 on real Dense and MoE Linear modules."""
import unittest

import torch
from torch import nn

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_formats import quantize
from model.quant_linear import gemm, install_quantization
from model.quant_policy import FOUR_BIT_SCHEMES


def round_bf16(tensor):
    return tensor.bfloat16().float()


def expected_dw(x, dy):
    # [out, tokens] @ [tokens, in]. Both operands group along the token axis.
    a = round_bf16(x.flatten(0, -2))
    g = round_bf16(dy.flatten(0, -2)).T
    return round_bf16(quantize(g, "mxfp8", axis=1) @ quantize(a, "mxfp8", axis=0))


class WeightGradientPrecisionTests(unittest.TestCase):
    def test_each_linear_uses_mxfp8_only_for_the_weight_gradient(self):
        torch.manual_seed(17)
        for moe in (False, True):
            cfg = MiniMindConfig(hidden_size=32, num_hidden_layers=1, use_moe=moe,
                                 num_attention_heads=4, num_key_value_heads=2,
                                 intermediate_size=64, vocab_size=48, max_position_embeddings=48)
            model = MiniMindForCausalLM(cfg)
            linears = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
            # q/k/v/o, 3 projections per FFN, the output head and (for MoE) the router.
            self.assertEqual(len(linears), 4 + 3 * (cfg.num_experts if moe else 1) + 1 + int(moe))
            for base in FOUR_BIT_SCHEMES:
                install_quantization(model, base + "_dw_mxfp8", weight_cache_mib=1)
                for name, layer in linears.items():
                    with self.subTest(moe=moe, base=base, projection=name):
                        # Rectangular shapes and 34 noncontiguous token rows span two MX groups.
                        x = torch.randn(17, 2, layer.in_features).transpose(0, 1).requires_grad_()
                        dy = torch.randn(17, 2, layer.out_features).transpose(0, 1) * 2.0**-24
                        y = layer(x)
                        dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
                        forward_scheme = "bf16" if name.rsplit(".", 1)[-1] in ("lm_head", "gate") else base
                        with torch.no_grad():
                            a, w = round_bf16(x.flatten(0, 1)), round_bf16(layer.weight)
                            g = round_bf16(dy.flatten(0, 1))
                            want_y = round_bf16(gemm(a, w.T, forward_scheme)).reshape_as(y)
                            want_dx = round_bf16(gemm(g, w, forward_scheme, lhs_is_dy=True)).reshape_as(x)
                            torch.testing.assert_close(y, want_y, atol=0, rtol=0)
                            torch.testing.assert_close(dx, want_dx, atol=0, rtol=0)
                            torch.testing.assert_close(dw, expected_dw(x, dy), atol=0, rtol=0)
                            old_dw = round_bf16(gemm(g.T, a, forward_scheme, lhs_is_dy=True))
                            self.assertFalse(torch.equal(dw, old_dw))
                            # Evaluation touches the cached-forward entry point too.
                            layer.eval()
                            torch.testing.assert_close(layer(x), want_y, atol=0, rtol=0)
                            layer.train()
                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_tied_embedding_keeps_native_lookup_gradient(self):
        torch.manual_seed(19)
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1,
                                                  intermediate_size=64, vocab_size=48))
        install_quantization(model, "hif4_thr_3mm_dw_mxfp8")
        embed = model.model.embed_tokens
        ids = torch.tensor([0, 5, 0, 7])
        x = torch.randn(9, 32)
        dy = torch.randn(9, 48)
        lookup_grad = torch.randn(len(ids), 32)
        torch.autograd.backward((model.lm_head(x), embed(ids)), (dy, lookup_grad))
        native_contribution = torch.zeros_like(embed.weight).index_add_(0, ids, lookup_grad)
        torch.testing.assert_close(embed.weight.grad, expected_dw(x, dy) + native_contribution,
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
