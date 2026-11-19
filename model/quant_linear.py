"""Software training precision for MiniMind projections, using FP32 carrier tensors."""

from types import MethodType

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .quant_formats import quantize

SIMULATION_SCHEMES = (
    "bf16", "mxfp8", "hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm",
)
_NV_SCHEMES = {"nvfp4_3mm", "nvint4_3mm"}
_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _bf16_value(x):
    # An actual dtype conversion rounds to nearest-even. GEMM uses FP32 on sm_70.
    return x.to(torch.bfloat16).float()


def gemm(lhs, rhs, scheme, *, lhs_is_dy=False):
    """Multiply two already BF16-rounded, FP32-carried matrices; return an FP32 result.

    Group along the contracting dimension of each logical operand. In both backward
    products dY (or its transpose) is the left operand. Only NV variants scale dY
    globally, and each invocation of quantize computes its scales from its input.
    """
    with torch.autocast(device_type=lhs.device.type, enabled=False):
        if scheme == "bf16" or lhs.numel() == 0 or rhs.numel() == 0:
            return lhs @ rhs
        dy_scale = lhs_is_dy and scheme in _NV_SCHEMES
        lhs0 = quantize(lhs, scheme, axis=1, tensor_scale=dy_scale)
        rhs0 = quantize(rhs, scheme, axis=0)
        result = lhs0 @ rhs0
        if scheme != "mxfp8":
            lhs1 = quantize(lhs - lhs0, scheme, axis=1, tensor_scale=dy_scale)
            result.add_(lhs1 @ rhs0)
            rhs1 = quantize(rhs - rhs0, scheme, axis=0)
            result.add_(lhs0 @ rhs1)
        return result


class _SimulatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, scheme):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.scheme = scheme
        inputs = _bf16_value(x.reshape(-1, weight.shape[1]))
        output = gemm(inputs, _bf16_value(weight).T, scheme)
        if bias is not None:
            output.add_(_bf16_value(bias))
        # MiniMind adds the attention residual in place, so expose an owning
        # tensor (a custom Function's returned view cannot be modified in place).
        return _bf16_value(output.view(*x.shape[:-1], weight.shape[0]))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dy = _bf16_value(grad_output.reshape(-1, weight.shape[0]))
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = gemm(dy, _bf16_value(weight), ctx.scheme, lhs_is_dy=True)
            dx = _bf16_value(dx).reshape_as(x).to(x.dtype)
        if ctx.needs_input_grad[1]:
            inputs = _bf16_value(x.reshape(-1, weight.shape[1]))
            dw = gemm(dy.T, inputs, ctx.scheme, lhs_is_dy=True)
            dw = _bf16_value(dw).to(weight.dtype)
        if ctx.needs_input_grad[2]:
            db = _bf16_value(dy.sum(dim=0)).to(ctx.bias_dtype)
        return dx, dw, db, None


def _simulated_forward(self, x):
    return _SimulatedLinear.apply(x, self.weight, self.bias, self._simulation_scheme)


def install_quantization(model, scheme):
    """Select simulator calls without replacing Parameters or changing checkpoint keys.

    Native MiniMind-3 stores each MoE expert as a separate MLP. Its unselected
    experts, router, and shared embedding/head retain their original identities.
    The router and output head use the same BF16 linear policy in all five modes.
    Passing 'none' restores the original forward methods.
    """
    if scheme != "none" and scheme not in SIMULATION_SCHEMES:
        raise ValueError(f"Unknown simulation scheme: {scheme!r}")
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "self_attn":
            module.simulate_bf16_attention = scheme != "none"
            continue
        if not isinstance(module, nn.Linear):
            continue
        if leaf not in _PROJECTIONS and leaf not in {"gate", "lm_head"}:
            continue
        if scheme == "none":
            if hasattr(module, "_simulation_scheme"):
                original = module._original_instance_forward
                del module._simulation_scheme, module._original_instance_forward
                if original is None:
                    del module.forward
                else:
                    module.forward = original
            continue
        if not hasattr(module, "_simulation_scheme"):
            module._original_instance_forward = module.__dict__.get("forward")
            module.forward = MethodType(_simulated_forward, module)
        module._simulation_scheme = scheme if leaf in _PROJECTIONS else "bf16"
    return model
