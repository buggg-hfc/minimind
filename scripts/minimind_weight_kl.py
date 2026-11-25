"""Compare learned MiniMind weight distributions in terminal CPU exports."""

import math
from collections.abc import Mapping

import torch

from scripts.moe_diagnostic_kl import weight_kl_rows


def load_weight_state(path):
    """Memory-map a native pilot export (torch.save of a flat model state_dict)."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return validate_weight_state(state, path)


def validate_weight_state(state, path):
    """Validate the model payload from an export or a verified final resume file."""
    if (not isinstance(state, Mapping) or not state
            or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
            or not any(k.endswith(".weight") for k in state)):
        raise ValueError(f"Expected a flat MiniMind model-weight export at {path}")
    for name, tensor in state.items():
        if name.endswith(".weight") and (not tensor.is_floating_point() or tensor.layout != torch.strided):
            raise ValueError(f"Expected floating-point dense weights at {path}: {name}")
    # The fixed pilot ties both names. GPU-to-CPU export can serialize two
    # independent copies; value equality, rather than storage identity, matters.
    embed, head = "model.embed_tokens.weight", "lm_head.weight"
    if embed in state and head in state and not torch.equal(state[embed], state[head]):
        raise ValueError(f"Tied embedding and output-head weights differ at {path}")
    return state


def compare_weight_states(reference, candidate):
    """Return (element-weighted averages, per-tensor KL rows) in natural-log units.

    Forward is BF16-reference || candidate. Each tensor uses its own signed
    joint min/max, 128 equal-width bins, and a pseudocount of 0.5 per bin.
    Jeffreys is the sum of forward and reverse KL.
    """
    rows = weight_kl_rows(reference, candidate)
    elements = sum(row["elements"] for row in rows)
    for row in rows:
        row["kl_jeffreys"] = row["kl_forward"] + row["kl_reverse"]
    summary = dict(weight_elements=elements, weight_tensors=len(rows))
    for field in ("kl_forward", "kl_reverse", "kl_jeffreys"):
        summary[f"weight_{field}"] = math.fsum(row["elements"] * row[field] for row in rows) / elements
    return summary, rows
