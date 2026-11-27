"""Software training precision for MiniMind projections, using FP32 carrier tensors."""

from types import MethodType
from weakref import ref

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .quant_formats import bf16_residual, compute_nv_tensor_scale, quantize
from .quant_policy import MIXED_SCHEMES, SIMULATION_SCHEMES, projection_policy

_NV_SCHEMES = {"nvfp4_3mm", "nvint4_3mm"}
_PROJECTION_ROLES = {
    "q_proj": "attention", "k_proj": "attention", "v_proj": "attention", "o_proj": "attention",
    "gate_proj": "ffn", "up_proj": "ffn", "down_proj": "ffn",
    "gate": "router", "lm_head": "lm_head",
}


def _bf16_value(x):
    # An actual dtype conversion rounds to nearest-even. GEMM uses FP32 on sm_70.
    return x.to(torch.bfloat16).float()


def gemm(lhs, rhs, scheme, *, lhs_is_dy=False, rhs_components=None):
    """Multiply two already BF16-rounded, FP32-carried matrices; return an FP32 result.

    Group along the contracting dimension of each logical operand. In both backward
    products dY (or its transpose) is the left operand. Only NV variants scale dY
    globally. Its residual reuses that FP32 scale, with fresh group scales.
    In inference, rhs_components can supply the same precomputed right operands.
    """
    with torch.autocast(device_type=lhs.device.type, enabled=False):
        if scheme == "bf16" or lhs.numel() == 0 or rhs.numel() == 0:
            return lhs @ rhs
        dy_scale = compute_nv_tensor_scale(lhs, scheme) if lhs_is_dy and scheme in _NV_SCHEMES else False
        lhs0 = quantize(lhs, scheme, axis=1, tensor_scale=dy_scale)
        rhs0 = quantize(rhs, scheme, axis=0) if rhs_components is None else rhs_components[0]
        result = lhs0 @ rhs0
        if scheme != "mxfp8":
            lhs1 = quantize(bf16_residual(lhs, lhs0), scheme, axis=1, tensor_scale=dy_scale)
            result.add_(lhs1 @ rhs0)
            rhs1 = quantize(bf16_residual(rhs, rhs0), scheme, axis=0) if rhs_components is None else rhs_components[1]
            result.add_(lhs0 @ rhs1)
        return result


def _linear_output(x, weight, bias, scheme, rhs_components=None):
    inputs = _bf16_value(x.reshape(-1, weight.shape[1]))
    rhs = _bf16_value(weight).T if rhs_components is None else rhs_components[0]
    output = gemm(inputs, rhs, scheme, rhs_components=rhs_components)
    if bias is not None:
        output.add_(_bf16_value(bias))
    # An owning result permits the decoder's in-place residual addition.
    return _bf16_value(output.view(*x.shape[:-1], weight.shape[0]))


class _SimulatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, scheme, dw_scheme):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.scheme = scheme
        ctx.dw_scheme = dw_scheme
        return _linear_output(x, weight, bias, scheme)

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
            dw = gemm(dy.T, inputs, ctx.dw_scheme, lhs_is_dy=True)
            dw = _bf16_value(dw).to(weight.dtype)
        if ctx.needs_input_grad[2]:
            db = _bf16_value(dy.sum(dim=0)).to(ctx.bias_dtype)
        return dx, dw, db, None, None


def _weight_cache_bytes(weight, scheme):
    components = 1 if scheme in ("bf16", "mxfp8") else 2
    return weight.numel() * 4 * components


def _inference_weight_components(layer):
    weight = layer.weight
    scheme = layer._simulation_scheme
    if torch.is_inference(weight) or _weight_cache_bytes(weight, scheme) > layer._cache_capacity:
        # Inference tensors lack a mutation counter; values may have changed since the last call.
        layer._weight_cache = None
        return None
    signature = (weight._version, weight.data_ptr(), weight.device, weight.dtype, weight.shape, weight.stride())
    cached = layer._weight_cache
    if cached is not None and cached[0]() is weight and cached[1] == signature:
        return cached[2]

    rhs = _bf16_value(weight).T
    main = rhs if scheme == "bf16" else quantize(rhs, scheme, axis=0)
    residual = None
    if scheme not in ("bf16", "mxfp8"):
        residual = quantize(bf16_residual(rhs, main), scheme, axis=0)
    parts = (main, residual)
    layer._weight_cache = (ref(weight), signature, parts)
    return parts


def _simulated_forward(self, x):
    if self._cache_capacity:
        if not self.training and not torch.is_grad_enabled():
            parts = _inference_weight_components(self)
            return _linear_output(x, self.weight, self.bias, self._simulation_scheme, parts)
        self._weight_cache = None
    scheme = self._simulation_scheme
    return _SimulatedLinear.apply(x, self.weight, self.bias, scheme, self._dw_scheme_override or scheme)


def install_quantization(model, scheme, *, weight_cache_mib=0):
    """Select simulator calls without replacing Parameters or changing checkpoint keys.

    Native MiniMind-3 stores each MoE expert as a separate MLP. Its unselected
    experts, router, and shared embedding/head retain their original identities.
    In mixed MoE schemes the selected gate/up/down projections in every expert use
    MXFP8 for forward, dX, and dW. The other expert projections and Q/K/V/O use
    the base 4-bit format. Router and head FWD/dX use BF16. In a *_dw_mxfp8 scheme
    every linear weight GEMM uses MXFP8, including the router and tied output head.
    The embedding lookup still contributes its own gradient to the shared weight.
    A positive inference cache budget reserves slots in module order, filled on first
    use under eval/no-grad. The budget covers persistent reconstructed weights, not
    transient GEMM/quantization workspaces. Reinstalling clears all old entries.
    Passing 'none' restores the original forward methods.
    """
    policy = projection_policy(scheme)
    if scheme in MIXED_SCHEMES and not getattr(getattr(model, "config", None), "use_moe", False):
        raise ValueError(f"{scheme} requires a MiniMind MoE model (use_moe=True)")
    if not isinstance(weight_cache_mib, int) or weight_cache_mib < 0:
        raise ValueError("weight_cache_mib must be a non-negative integer")
    if scheme == "none" and weight_cache_mib:
        raise ValueError("A simulated precision scheme is required for the weight cache")
    remaining = weight_cache_mib * (1024 ** 2)
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "self_attn":
            module.simulate_bf16_attention = scheme != "none"
            continue
        if not isinstance(module, nn.Linear):
            continue
        if leaf not in _PROJECTION_ROLES:
            continue
        if scheme == "none":
            if hasattr(module, "_simulation_scheme"):
                original = module._original_instance_forward
                del module._simulation_scheme, module._dw_scheme_override, module._original_instance_forward
                del module._cache_capacity, module._weight_cache
                if original is None:
                    del module.forward
                else:
                    module.forward = original
            continue
        if not hasattr(module, "_simulation_scheme"):
            module._original_instance_forward = module.__dict__.get("forward")
            module.forward = MethodType(_simulated_forward, module)
        layer_scheme = policy[_PROJECTION_ROLES[leaf]]
        module._simulation_scheme = layer_scheme[leaf] if isinstance(layer_scheme, dict) else layer_scheme
        module._dw_scheme_override = policy.get("dW")
        size = _weight_cache_bytes(module.weight, module._simulation_scheme)
        module._cache_capacity = size if size <= remaining else 0
        remaining -= module._cache_capacity
        module._weight_cache = None
    return model
