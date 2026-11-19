import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.quant_formats import quantize
from model.quant_linear import SIMULATION_SCHEMES, install_quantization


def projection(width=32):
    model = nn.Module()
    model.q_proj = nn.Linear(width, width, bias=True)
    return model.eval()


class InferenceWeightCacheTests(unittest.TestCase):
    def test_exact_output_and_reuse(self):
        torch.manual_seed(14)
        model = projection()
        x = torch.randn(1, 5, 32)
        for scheme in SIMULATION_SCHEMES:
            with self.subTest(scheme=scheme), torch.inference_mode():
                install_quantization(model, scheme)
                expected = model.q_proj(x)
                install_quantization(model, scheme, weight_cache_mib=1)
                with patch('model.quant_linear.quantize', wraps=quantize) as spy:
                    torch.testing.assert_close(model.q_proj(x), expected, rtol=0, atol=0)
                    entry = model.q_proj._weight_cache
                    self.assertIsNotNone(entry)
                    spy.reset_mock()
                    torch.testing.assert_close(model.q_proj(x), expected, rtol=0, atol=0)
                    self.assertIs(model.q_proj._weight_cache, entry)
                    self.assertFalse(any(call.kwargs['axis'] == 0 for call in spy.call_args_list))

    def test_reload_replace_and_train(self):
        torch.manual_seed(8)
        model = projection()
        install_quantization(model, 'nvfp4_3mm', weight_cache_mib=1)
        x = torch.randn(3, 32)
        with torch.no_grad():
            old = model.q_proj(x)
            weights = model.state_dict()
            weights['q_proj.weight'] = torch.randn_like(weights['q_proj.weight'])
            model.load_state_dict(weights)
            new = model.q_proj(x)
            self.assertFalse(torch.equal(old, new))
            # Module.to may preserve the Parameter identity while replacing storage.
            entry = model.q_proj._weight_cache
            model.to(dtype=torch.float64)
            torch.testing.assert_close(model.q_proj(x), new, atol=0, rtol=0)
            self.assertIsNot(model.q_proj._weight_cache, entry)
            model.q_proj.weight = nn.Parameter(model.q_proj.weight.clone().zero_())
            after_replace = model.q_proj(x)
            install_quantization(model, 'nvfp4_3mm')
            torch.testing.assert_close(after_replace, model.q_proj(x), atol=0, rtol=0)

        install_quantization(model, 'hif4_thr_3mm', weight_cache_mib=1)
        with torch.inference_mode():
            model.q_proj(x)
        model.train()
        model.q_proj(x).sum().backward()
        self.assertIsNone(model.q_proj._weight_cache)
        self.assertIsNotNone(model.q_proj.weight.grad)
        install_quantization(model, 'none')
        self.assertFalse(hasattr(model.q_proj, '_weight_cache'))

    def test_memory_budget_and_inference_tensor(self):
        model = projection(256)
        model.up_proj = nn.Linear(256, 512)
        model.eval()
        install_quantization(model, 'nvint4_3mm', weight_cache_mib=1)
        with torch.inference_mode():
            x = torch.randn(1, 256)
            model.q_proj(x)
            model.up_proj(x)
            self.assertIsNotNone(model.q_proj._weight_cache)
            self.assertIsNone(model.up_proj._weight_cache)
            model.q_proj.weight = nn.Parameter(torch.zeros_like(model.q_proj.weight))
            zeros = model.q_proj(x)
            self.assertIsNone(model.q_proj._weight_cache)
            model.q_proj.weight.fill_(1)
            self.assertFalse(torch.equal(model.q_proj(x), zeros))
