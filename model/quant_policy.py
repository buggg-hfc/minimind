"""Identify weights and results trained with the current 3MM residual arithmetic."""

RESIDUAL_SCHEMES = {"hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm"}
RESIDUAL_POLICY = "bf16_diff_shared_tensor_scale_v2"


def scheme_artifact_name(scheme):
    return f"{scheme}_r2" if scheme in RESIDUAL_SCHEMES else scheme


def validate_resume_policy(checkpoint, scheme):
    if checkpoint.get("quant_sim", "none") != scheme:
        raise ValueError("Checkpoint quant_sim does not match the requested training format")
    if scheme in RESIDUAL_SCHEMES and checkpoint.get("residual_policy") != RESIDUAL_POLICY:
        raise ValueError(f"{scheme} resume requires residual_policy={RESIDUAL_POLICY}; "
                         "start a fresh run with --from_resume 0 for the revised residual experiment.")
