"""Projection precision and checkpoint identifiers for MiniMind simulation."""

FOUR_BIT_SCHEMES = ("hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm")
SIMULATION_SCHEMES = ("bf16", "mxfp8", *FOUR_BIT_SCHEMES)
FFN_MXFP8_GROUPS = ("up", "gate", "down", "up_gate", "up_down", "gate_down", "all")


def ffn_mxfp8_scheme(base, group):
    """Canonical training and checkpoint identifier for an expert projection subset."""
    if base not in FOUR_BIT_SCHEMES or group not in FFN_MXFP8_GROUPS:
        raise ValueError(f"Unknown 4-bit format / expert projection group: {base!r}, {group!r}")
    if group == "all":
        return f"{base}_ffn_mxfp8"
    return f"{base}_ffn_{group}_mxfp8"


_MIXED_POLICIES = {ffn_mxfp8_scheme(base, group): (base, group)
                   for base in FOUR_BIT_SCHEMES for group in FFN_MXFP8_GROUPS}
MIXED_SCHEMES = {scheme: base for scheme, (base, _) in _MIXED_POLICIES.items()}
MODEL_SCHEMES = (*SIMULATION_SCHEMES, *MIXED_SCHEMES)
RESIDUAL_SCHEMES = {*FOUR_BIT_SCHEMES, *MIXED_SCHEMES}
RESIDUAL_POLICY = "bf16_diff_shared_tensor_scale_v2"


def projection_policy(scheme):
    """Return the primitive linear formats for a model-level precision scheme."""
    if scheme == "none":
        return {"attention": "none", "ffn": "none", "router": "none", "lm_head": "none"}
    if scheme not in MODEL_SCHEMES:
        raise ValueError(f"Unknown simulation scheme: {scheme!r}")
    base, group = _MIXED_POLICIES.get(scheme, (scheme, None))
    ffn = base
    if group == "all":
        # Existing full-FFN checkpoints and validation reports store a scalar here.
        ffn = "mxfp8"
    elif group is not None:
        selected = group.split("_")
        ffn = {f"{name}_proj": "mxfp8" if name in selected else base
               for name in ("up", "gate", "down")}
    return {
        "attention": base,
        "ffn": ffn,
        "router": "bf16",
        "lm_head": "bf16",
    }


def scheme_artifact_name(scheme):
    return f"{scheme}_r2" if scheme in RESIDUAL_SCHEMES else scheme


def validate_resume_policy(checkpoint, scheme, *, seed=None):
    if checkpoint.get("quant_sim", "none") != scheme:
        raise ValueError("Checkpoint quant_sim does not match the requested training format")
    if scheme in RESIDUAL_SCHEMES and checkpoint.get("residual_policy") != RESIDUAL_POLICY:
        raise ValueError(f"{scheme} resume requires residual_policy={RESIDUAL_POLICY}; "
                         "start a fresh run with --from_resume 0 for the revised residual experiment.")
    if scheme in MIXED_SCHEMES:
        if checkpoint.get("projection_policy") != projection_policy(scheme):
            raise ValueError(f"Checkpoint projection_policy does not match {scheme}")
        if seed is not None and checkpoint.get("seed") != seed:
            raise ValueError(f"Seed mismatch: expected {seed}, found {checkpoint.get('seed')}")
