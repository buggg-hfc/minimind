"""Projection precision and checkpoint identifiers for MiniMind simulation."""

_FOUR_BIT_SCHEMES = ("hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm")
SIMULATION_SCHEMES = ("bf16", "mxfp8", *_FOUR_BIT_SCHEMES)
MIXED_SCHEMES = {f"{scheme}_ffn_mxfp8": scheme for scheme in _FOUR_BIT_SCHEMES}
MODEL_SCHEMES = (*SIMULATION_SCHEMES, *MIXED_SCHEMES)
RESIDUAL_SCHEMES = {*_FOUR_BIT_SCHEMES, *MIXED_SCHEMES}
RESIDUAL_POLICY = "bf16_diff_shared_tensor_scale_v2"


def projection_policy(scheme):
    """Return the primitive linear formats for a model-level precision scheme."""
    if scheme == "none":
        return {"attention": "none", "ffn": "none", "router": "none", "lm_head": "none"}
    if scheme not in MODEL_SCHEMES:
        raise ValueError(f"Unknown simulation scheme: {scheme!r}")
    return {
        "attention": MIXED_SCHEMES.get(scheme, scheme),
        "ffn": "mxfp8" if scheme in MIXED_SCHEMES else scheme,
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
