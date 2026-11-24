"""Reproduce the first 256 AdamW updates with one expert's down_proj using MXFP8."""
from early_prefix import BASE_SCHEME, STATE_TEMPLATE, make_anchor, policy_code


def expert_scheme(expert):
    if type(expert) is not int or expert not in (1, 2):
        raise ValueError("The seed-42 expert comparison uses layer 6, expert 1 or 2 (zero-based)")
    return f"{BASE_SCHEME}_l6e{expert}_down_mxfp8"


def expert_policy(expert):
    expert_scheme(expert)
    return {"attention": BASE_SCHEME, "router": "bf16", "lm_head": "bf16",
            "ffn": {name: BASE_SCHEME for name in ("gate_proj", "up_proj", "down_proj")},
            "module_overrides": {f"model.layers.6.mlp.experts.{expert}.down_proj": "mxfp8"}}


def expert_state(expert):
    return STATE_TEMPLATE.format(scheme=expert_scheme(expert))


# Inject this function into the frozen training snapshot, next to its native dispatch.
# Its module path and applicable scheme are embedded in the checkpoint's POLICY.
_SELECT_ONE = '''
_configure_base = configure

def configure(model, branch):
    if branch != "hif4_down_mxfp8":
        raise ValueError("Single-expert early policy requires its mixed dispatch")
    counts = _configure_base(model, "hif4")
    if (model.config.num_hidden_layers, model.config.num_experts) != (8, 4):
        raise ValueError("Single-expert comparison requires 8 layers with 4 experts per layer")
    ((path, fmt),) = POLICY["module_overrides"].items()
    target = model.get_submodule(path)
    if not isinstance(target, torch.nn.Linear) or target._simulation_scheme != HIF4 or fmt != "mxfp8":
        raise ValueError(f"Unexpected simulation policy at {path}")
    target._simulation_scheme = fmt
    # Verify the actual dispatch on each module, including its fully qualified name.
    changed = [name for name, layer in model.named_modules()
               if getattr(layer, "_simulation_scheme", None) == fmt]
    if changed != [path] or counts["down_proj/" + HIF4] != 32:
        raise ValueError(f"Expected a single MXFP8 expert down_proj at {path}, got {changed}")
    counts["down_proj/" + HIF4] -= 1
    counts["down_proj/" + fmt] = 1
    return counts
'''


def expert_policy_code(frozen_worker, scheme, policy):
    source = policy_code(frozen_worker, scheme, policy) + _SELECT_ONE
    compile(source, "anchor_policy.py", "exec")
    return source


def make_expert_anchor(original_prefix, anchor, tools_dir, expert):
    return make_anchor(original_prefix, anchor, tools_dir, scheme=expert_scheme(expert),
                       policy=expert_policy(expert), policy_writer=expert_policy_code)
