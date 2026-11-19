"""Block quantizers for training emulation, returning reconstructed FP32 operands."""

import torch
from torch.nn import functional as F


GROUP_SIZE = {"mxfp8": 32, "hif4_thr_3mm": 64, "nvfp4_3mm": 16, "nvint4_3mm": 16}
UE5M3_MAX = 114688.0


def _round_magnitude(x, fraction_bits, min_step_exponent, maximum):
    """Saturating nearest-even rounding on a binary floating-point magnitude grid."""
    x = x.clamp(max=maximum)
    _, exponent = torch.frexp(x)
    step_exponent = (exponent - 1 - fraction_bits).clamp(min=min_step_exponent)
    step = torch.exp2(step_exponent.float())
    return torch.round(x / step) * step


def _mxfp8(blocks):
    # E8M0 scale = next power of two >= amax / 448, including tiny E8M0 scales.
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    _, exponent = torch.frexp(maximum)
    base_exponent = (exponent - 1 - 8).clamp(-127, 127)
    base = torch.exp2(base_exponent.float())
    scale = torch.where(maximum > 448.0 * base, base * 2.0, base)
    magnitude = _round_magnitude(blocks.abs() / scale, 3, -9, 448.0)
    return torch.copysign(magnitude * scale, blocks)


def _hif4_thr(blocks):
    # One base scale per 64 elements. Pairs of 4-element groups share an
    # 8-element micro-scale; each 4-element group has an additional micro-scale.
    groups = blocks.unflatten(-1, (8, 2, 4))
    max4 = groups.abs().amax(dim=-1, keepdim=True)
    max8 = max4.amax(dim=-2, keepdim=True)
    max64 = max8.amax(dim=-3, keepdim=True)
    # BF16 round-to-nearest-even of 1/7, exact in the FP32 carrier.
    base = (max64 * 0.142578125).clamp(2.0 ** -48, 49152.0).to(torch.bfloat16).float()
    _, exponent = torch.frexp(base)
    unit = torch.exp2((exponent - 1).float())
    base = torch.round(base / unit * 4.0) * 0.25 * unit
    inverse = base.reciprocal().to(torch.bfloat16).float()
    scale8 = torch.where(max8 * inverse >= 3.75, 2.0, 1.0)
    scale4 = torch.where(max4 / scale8 * inverse >= 1.875, 2.0, 1.0)
    micro_scale = scale8 * scale4
    magnitude = groups.abs() / micro_scale * inverse
    codes = torch.floor(magnitude * 4.0 + 0.5).clamp(max=7.0)
    return torch.copysign((codes * 0.25) * micro_scale * base, groups).reshape_as(blocks)


def _nv4(blocks, scheme, use_tensor_scale):
    fp4 = scheme == "nvfp4_3mm"
    qmax = 6.0 if fp4 else 7.0
    # 'blocks' spans this entire logical operand (one expert for routed MLPs).
    # Keep scale positive even when a nonzero subnormal amax would underflow.
    if use_tensor_scale:
        maximum = blocks.abs().amax()
        tensor_scale = (maximum / (qmax * UE5M3_MAX)).clamp(min=2.0 ** -149)
        tensor_scale = torch.where(maximum == 0, 1.0, tensor_scale)
        blocks = blocks / tensor_scale
    ideal = blocks.abs().amax(dim=-1, keepdim=True) / qmax
    group_scale = _round_magnitude(ideal, 3, -17, UE5M3_MAX)
    safe_scale = torch.where(group_scale == 0, 1.0, group_scale)
    normalized = blocks / safe_scale
    if fp4:
        magnitude = _round_magnitude(normalized.abs(), 1, -1, qmax)
        codes = torch.copysign(magnitude, normalized)
    else:
        codes = torch.round(normalized).clamp(-7, 7)
    result = codes * group_scale
    return result * tensor_scale if use_tensor_scale else result


@torch.no_grad()
def quantize(x, scheme, axis=-1, *, tensor_scale=False):
    """Quantize and reconstruct one operand, grouping along its GEMM reduction axis.

    NV tensor scales apply only when requested by the caller for a backward dY.
    A separate call for a residual recalculates every scale on that residual.
    """
    if x.ndim < 1 or not -x.ndim <= axis < x.ndim:
        raise ValueError("axis must index a dimension of the operand")
    if scheme != "bf16" and scheme not in GROUP_SIZE:
        raise ValueError(f"Unknown quantization scheme: {scheme}")
    if tensor_scale and scheme not in ("nvfp4_3mm", "nvint4_3mm"):
        raise ValueError("FP32 tensor scale is only defined for the NV schemes")
    data = x.float()
    if scheme == "bf16":
        return data.to(torch.bfloat16).float()
    if data.numel() == 0:
        return data

    moved = data.movedim(axis, -1)
    width = moved.shape[-1]
    group = GROUP_SIZE[scheme]
    padding = (-width) % group
    if padding:
        moved = F.pad(moved, (0, padding))
    blocks = moved.unflatten(-1, (-1, group))
    if scheme == "mxfp8":
        values = _mxfp8(blocks)
    elif scheme == "hif4_thr_3mm":
        values = _hif4_thr(blocks)
    else:
        values = _nv4(blocks, scheme, tensor_scale)
    reconstructed = values.flatten(-2)[..., :width].movedim(-1, axis)
    # Saturating payload grids must not turn a training Inf/NaN into a finite value.
    return torch.where(torch.isfinite(data), reconstructed, data)
