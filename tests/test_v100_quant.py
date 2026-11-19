"""Numeric and end-to-end tests. Run with python -m unittest discover -s tests."""
import unittest

import torch
from torch import nn

from model.quant_formats import bf16_residual, quantize
from model.quant_linear import gemm, install_quantization


SCHEMES = ("bf16", "mxfp8", "hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm")
NV_SCHEMES = ("nvfp4_3mm", "nvint4_3mm")


def bf16(x):
    return x.to(torch.bfloat16).float()


def reference_mm(a, b, scheme, backward_dy=False):
    """Explicit three independent mm calls, grouped on the logical K dimension."""
    a = bf16(a)
    b = bf16(b)
    ts = False
    if backward_dy and scheme in NV_SCHEMES:
        maximum = a.float().abs().max()
        qmax = 6 if scheme == "nvfp4_3mm" else 7
        ts = torch.where(maximum == 0, 1.0, (maximum / (qmax * 114688.0)).clamp_min(2.0 ** -149))
    a0 = quantize(a, scheme, axis=1, tensor_scale=ts)
    b0 = quantize(b, scheme, axis=0)
    out = a0 @ b0
    if scheme.endswith("_3mm"):
        a1 = quantize((a.bfloat16() - a0.bfloat16()).float(), scheme, axis=1, tensor_scale=ts)
        b1 = quantize((b.bfloat16() - b0.bfloat16()).float(), scheme, axis=0)
        out = (out + a1 @ b0) + a0 @ b1
    return bf16(out)


class QuantizerTests(unittest.TestCase):
    def test_mxfp8_next_scale_and_element_ties(self):
        x = torch.zeros(2, 32)
        x[:, 0] = torch.tensor([448.0, 449.0])
        x[:, 1] = 2.0 ** -9
        x[0, 2:5] = torch.tensor([1.0625, 1.1875, -1.0625])
        out = quantize(x, "mxfp8")
        self.assertEqual(out[:, 1].tolist(), [2.0 ** -9, 0.0])
        self.assertEqual(out[0, 2:5].tolist(), [1.0, 1.25, -1.0])

    def test_hif4_thresholds_and_half_up(self):
        x = torch.zeros(2, 64)
        x[:, 0] = 7.0  # sets the primary scale to one in both groups
        x[:, 8] = torch.tensor([3.75, 3.74])
        x[:, 9] = 0.25
        x[:, 12] = 0.25
        out = quantize(x, "hif4_thr_3mm")
        self.assertEqual(out[0, [8, 9, 12]].tolist(), [4.0, 0.0, 0.5])
        self.assertEqual(out[1, [8, 9, 12]].tolist(), [3.5, 0.5, 0.25])

    def test_nv_codebooks_and_unsigned_group_scale(self):
        for scheme, qmax in zip(NV_SCHEMES, (6, 7)):
            with self.subTest(scheme=scheme):
                x = torch.zeros(4, 16)
                x[0, :5] = torch.tensor([qmax, 0.25, 0.75, 1.25, -5.0])
                x[1, 0] = qmax * 1.0625  # UE5M3 tie -> even 1.0
                x[2, 0] = qmax * 114688.0 * 2  # saturate at maximum finite scale
                x[3, 0] = qmax * 2.0 ** -18  # half the first subnormal -> zero
                out = quantize(x, scheme)
                codes = [0, 1, 1, -4] if scheme == "nvfp4_3mm" else [0, 1, 1, -5]
                self.assertEqual(out[0, 1:5].tolist(), codes)
                self.assertEqual(out[1:, 0].tolist(), [qmax, qmax * 114688.0, 0])
                with_ts = quantize(x[3:], scheme, tensor_scale=True)
                torch.testing.assert_close(with_ts, x[3:], atol=0, rtol=2e-6)

    def test_both_inputs_and_result_of_subtraction_are_bf16(self):
        a = torch.tensor([1.003, 3.0])
        main = torch.tensor([0.999, -2.0 ** -7])
        result = bf16_residual(a, main)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.tolist(), [0.0, 3.0])
        self.assertNotEqual(result[0].item(), bf16(a - main)[0].item())
        self.assertNotEqual(result[1].item(), (bf16(a) - bf16(main))[1].item())

    def test_nv_residual_reuses_tensor_scale_and_refits_group_scale(self):
        gen = torch.Generator().manual_seed(0)
        a = bf16(torch.randn(5, 33, generator=gen) * 1.287)
        b = bf16(torch.randn(33, 11, generator=gen) * 1.91)
        for scheme, qmax in zip(NV_SCHEMES, (6, 7)):
            with self.subTest(scheme=scheme):
                # Separate reference: normalize by the main FP32 scale manually.
                # Both calls choose group scales using their own 16-element blocks.
                ts = a.abs().amax() / (qmax * 114688.0)
                a0 = quantize(a / ts, scheme, axis=1) * ts
                a_diff = (a.bfloat16() - a0.bfloat16()).float()
                a1 = quantize(a_diff / ts, scheme, axis=1) * ts
                b0 = quantize(b, scheme, axis=0)
                b_diff = (b.bfloat16() - b0.bfloat16()).float()
                b1 = quantize(b_diff, scheme, axis=0)
                expected = (a0 @ b0 + a1 @ b0) + a0 @ b1
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    result = gemm(a, b, scheme, lhs_is_dy=True)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                self.assertEqual(result.dtype, torch.float32)
                self.assertFalse(torch.equal(result, bf16(expected)))
                wrong_scale = quantize(a_diff, scheme, axis=1, tensor_scale=True)
                wrong_domain = quantize((a - a0) / ts, scheme, axis=1) * ts
                for wrong in (wrong_scale, wrong_domain):
                    self.assertFalse(torch.equal(result, (a0 @ b0 + wrong @ b0) + a0 @ b1))

    def test_tail_groups_strides_and_empty_inputs(self):
        torch.manual_seed(13)
        x = torch.randn(5, 67).t()  # noncontiguous; pad each logical K row independently
        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                batch = quantize(x, scheme, axis=0)
                rows = torch.stack([quantize(col, scheme) for col in x.unbind(1)], dim=1)
                self.assertTrue(torch.equal(batch, rows))
                self.assertEqual(quantize(x[:0], scheme).shape, (0, 5))
                self.assertFalse(quantize(torch.zeros(2, 19), scheme).any())


class LinearTests(unittest.TestCase):
    def test_all_three_linear_paths(self):
        torch.manual_seed(37)
        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                layer = nn.Module()
                layer.q_proj = nn.Linear(17, 19, bias=False)
                weight = layer.q_proj.weight
                x = torch.randn(2, 5, 17, requires_grad=True)
                grad = torch.randn(2, 5, 19) * 2.0 ** -25  # dY underflows no-TS UE5M3
                install_quantization(layer, scheme)
                self.assertIs(layer.q_proj.weight, weight)
                out = layer.q_proj(x)
                out.backward(grad)
                a, g = x.detach().reshape(-1, 17), grad.reshape(-1, 19)
                w = weight.detach()
                want_y = reference_mm(a, w.t(), scheme)
                want_dx = reference_mm(g, w, scheme, backward_dy=True)
                want_dw = reference_mm(g.t(), a, scheme, backward_dy=True)
                self.assertTrue(torch.equal(out.detach().reshape_as(want_y), want_y))
                self.assertTrue(torch.equal(x.grad.reshape_as(want_dx), want_dx))
                self.assertTrue(torch.equal(weight.grad, want_dw))
                if scheme in NV_SCHEMES:
                    self.assertGreater(want_dw.abs().max().item(), 0)
                    self.assertFalse(reference_mm(g.t(), a, scheme).any())


class NativeModelTests(unittest.TestCase):
    def test_dense_and_routed_training(self):
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        for use_moe in (False, True):
            for scheme in SCHEMES:
                with self.subTest(moe=use_moe, scheme=scheme):
                    torch.manual_seed(23)
                    config = MiniMindConfig(
                        hidden_size=32, num_hidden_layers=1, vocab_size=48,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=64, max_position_embeddings=32,
                        use_moe=use_moe, flash_attn=False,
                    )
                    model = MiniMindForCausalLM(config).train()
                    keys = set(model.state_dict())
                    install_quantization(model, scheme)
                    self.assertEqual(set(model.state_dict()), keys)
                    self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
                    ids = torch.randint(config.vocab_size, (2, 13))
                    result = model(ids, labels=ids)
                    loss = result.loss + result.aux_loss
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, param in model.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                    block = model.model.layers[0]
                    self.assertGreater(block.self_attn.q_proj.weight.grad.abs().max().item(), 0)
                    if use_moe:
                        self.assertGreater(block.mlp.gate.weight.grad.abs().max().item(), 0)
                        self.assertTrue(any(e.up_proj.weight.grad.any() for e in block.mlp.experts))
                    else:
                        self.assertTrue(block.mlp.up_proj.weight.grad.any())


    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU (V100 supported)")
    def test_cuda_optimizer_step(self):
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        for scheme in SCHEMES:
            for use_moe in (False, True):
                with self.subTest(scheme=scheme, moe=use_moe):
                    torch.manual_seed(42)
                    config = MiniMindConfig(
                        hidden_size=64, num_hidden_layers=1, vocab_size=80,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_position_embeddings=32,
                        use_moe=use_moe,
                    )
                    model = MiniMindForCausalLM(config).train().cuda()
                    install_quantization(model, scheme)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    ids = torch.randint(config.vocab_size, (2, 17), device="cuda")
                    w = model.model.layers[0].self_attn.o_proj.weight
                    initial = w.detach().clone()
                    result = model(ids, labels=ids)
                    loss = result.loss + result.aux_loss
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, param in model.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertTrue(w.grad.any())
                    optimizer.step()
                    self.assertFalse(torch.equal(initial, w))


if __name__ == "__main__":
    unittest.main()
