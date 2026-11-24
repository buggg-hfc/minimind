"""Reproduce the first 256 AdamW updates with an expert-specific down_proj policy."""
from early_prefix import BASE_SCHEME, STATE_TEMPLATE, make_anchor, policy_code


ALL_EXCEPT_L6E1 = "all_except_l6e1"
_L6E1_DOWN = "model.layers.6.mlp.experts.1.down_proj"


def expert_scheme(expert):
    if expert == ALL_EXCEPT_L6E1:
        return f"{BASE_SCHEME}_down_mxfp8_except_l6e1"
    if type(expert) is not int or expert not in (1, 2):
        raise ValueError("Expected expert 1, expert 2 or all_except_l6e1")
    return f"{BASE_SCHEME}_l6e{expert}_down_mxfp8"


def expert_policy(expert):
    expert_scheme(expert)
    if expert == ALL_EXCEPT_L6E1:
        return {"attention": BASE_SCHEME, "router": "bf16", "lm_head": "bf16",
                "ffn": {"gate_proj": BASE_SCHEME, "up_proj": BASE_SCHEME, "down_proj": "mxfp8"},
                "module_overrides": {_L6E1_DOWN: BASE_SCHEME}}
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


# Included verbatim in both the prefix trainer and the late worker. Collect and
# validate all 32 modules before the caller changes any forward dispatch.
COMPLEMENT_MODULES = f'''
def _other_expert_down_projections(model):
    if (model.config.num_hidden_layers, model.config.num_experts) != (8, 4):
        raise ValueError("Expected 8 MoE layers and 4 experts per layer")
    selected = []
    for layer in range(8):
        for expert in range(4):
            name = f"model.layers.{{layer}}.mlp.experts.{{expert}}.down_proj"
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear) or module._simulation_scheme != HIF4:
                raise ValueError(f"Expected HiF4 down projection at {{name}}")
            if name != {_L6E1_DOWN!r}:
                selected.append(module)
    if len(selected) != 31 or len(set(selected)) != 31:
        raise ValueError("Expected 31 distinct down projections outside L6E1")
    return selected
'''

_SELECT_COMPLEMENT = '''
_configure_base = configure

def configure(model, branch):
    if branch != "hif4_down_mxfp8":
        raise ValueError("Unexpected early branch")
    counts = _configure_base(model, "hif4")
    others = _other_expert_down_projections(model)
    for module in others:
        module._simulation_scheme = "mxfp8"
    counts["down_proj/" + HIF4] -= len(others)
    counts["down_proj/mxfp8"] = len(others)
    return counts
'''


def expert_policy_code(frozen_worker, scheme, policy):
    dispatch = COMPLEMENT_MODULES + _SELECT_COMPLEMENT if scheme == expert_scheme(ALL_EXCEPT_L6E1) else _SELECT_ONE
    source = policy_code(frozen_worker, scheme, policy) + dispatch
    compile(source, "anchor_policy.py", "exec")
    return source


def make_expert_anchor(original_prefix, anchor, tools_dir, expert):
    return make_anchor(original_prefix, anchor, tools_dir, scheme=expert_scheme(expert),
                       policy=expert_policy(expert), policy_writer=expert_policy_code)
