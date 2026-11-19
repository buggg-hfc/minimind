import unittest
from unittest.mock import patch

import torch

from model.model_minimind import Attention, MiniMindConfig, _bf16_attention_matmul, precompute_freqs_cis
from model.quant_linear import install_quantization


class AttentionSimulationTest(unittest.TestCase):
    def test_forward_and_both_input_gradients_round_to_bf16(self):
        rng = torch.Generator().manual_seed(5)
        a = torch.randn(2, 3, 11, 7, generator=rng, requires_grad=True)
        b = torch.randn(2, 3, 7, 13, generator=rng, requires_grad=True)
        grad = torch.randn(2, 3, 11, 13, generator=rng)
        result = _bf16_attention_matmul(a, b)
        da, db = torch.autograd.grad(result, (a, b), grad)

        def rounded(value):
            return value.detach().bfloat16().float()

        self.assertTrue(torch.equal(result, rounded(rounded(a) @ rounded(b))))
        self.assertTrue(torch.equal(da, rounded(rounded(grad) @ rounded(b).mT)))
        self.assertTrue(torch.equal(db, rounded(rounded(a).mT @ rounded(grad))))

    def test_mask_cache_and_install_toggle(self):
        torch.manual_seed(16)
        config = MiniMindConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
        model = torch.nn.Module()
        model.self_attn = Attention(config).eval()
        install_quantization(model, 'bf16')
        x = torch.randn(1, 5, 32)
        mask = torch.tensor([[1, 0, 1, 1, 1]])
        pos = precompute_freqs_cis(config.head_dim, end=5)
        sim_matmul = 'model.model_minimind._bf16_attention_matmul'
        with torch.no_grad(), patch(sim_matmul, wraps=_bf16_attention_matmul) as mm:
            full, _ = model.self_attn(x, pos, attention_mask=mask)
            changed = x.clone()
            changed[:, 1] += 100
            masked, _ = model.self_attn(changed, pos, attention_mask=mask)
            _, cache = model.self_attn(x[:, :3], tuple(p[:3] for p in pos), use_cache=True, attention_mask=mask[:, :3])
            suffix, _ = model.self_attn(x[:, 3:], tuple(p[3:] for p in pos), past_key_value=cache, attention_mask=mask)
            torch.testing.assert_close(masked[:, 2:], full[:, 2:])
            torch.testing.assert_close(suffix, full[:, 3:])
            self.assertEqual(mm.call_count, 8)

            # Even with no padding and SDPA available, both BMMs must use the simulator.
            model.self_attn(x, pos)
            self.assertEqual(mm.call_count, 10)
            install_quantization(model, 'none')
            model.self_attn(x, pos, attention_mask=mask)
            self.assertEqual(mm.call_count, 10)


if __name__ == '__main__':
    unittest.main()
