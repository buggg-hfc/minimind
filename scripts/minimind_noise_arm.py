"""Run a seed-42 MoE noise arm using the frozen, unmodified eight-V100 pilot."""
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

BASE_LR = 5e-4
LR_MULTIPLIERS = {"repeat": 1., "swap12": 1., "init_1ulp": 1.,
                  "lr_x1p0001": 1.0001, "lr_x0p9999": .9999, "warmup26": 1.,
                  "warmup26_lr_x0p9999": .9999, "warmup26_lr_x1p0001": 1.0001}
COORDINATE = "model.layers.0.self_attn.q_proj.weight"


def shift_initial_coordinate(model):
    """Shift one FP32 scalar up by its rounded BF16 value's spacing toward +infinity."""
    import torch

    weight = model.get_parameter(COORDINATE)
    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise ValueError(f"Expected FP32 projection matrix {COORDINATE}")
    # Compute the adjacent BF16 value on the CPU, including for a Volta worker.
    with torch.no_grad():
        original = weight[0, 0].detach().cpu().clone()
        rounded = original.to(torch.bfloat16)
        next_value = torch.nextafter(rounded, torch.full_like(rounded, math.inf))
        spacing = next_value.float() - rounded.float()
        shifted = original + spacing
        if (not torch.isfinite(shifted).item() or shifted.item() - original.item() != spacing.item()
                or shifted.to(torch.bfloat16).item() != next_value.item() or spacing.item() <= 0):
            raise RuntimeError("Cannot apply an exact positive BF16-ULP at the fixed coordinate")
        weight[0, 0].copy_(shifted)
    return dict(parameter=COORDINATE, index=[0, 0], fp32_before=original.item(), fp32_after=shifted.item(),
                bf16_before=rounded.item(), bf16_after=next_value.item(), spacing=spacing.item())


def variant_schedule(native_get_lr, variant):
    if variant.startswith("warmup26"):
        from minimind_noise_lr import with_linear_warmup
        return with_linear_warmup(native_get_lr)
    return native_get_lr


def rank_worker(trainer, variant, native_args):
    root = trainer.resolve().parents[1]
    sys.path.insert(0, str(root))
    scheme = native_args[native_args.index("--quant_sim") + 1]
    fresh = native_args[native_args.index("--from_resume") + 1] == "0"

    if variant.startswith("warmup26"):
        from trainer import trainer_utils
        trainer_utils.get_lr = variant_schedule(trainer_utils.get_lr, variant)
    if variant == "swap12":
        from trainer import trainer_utils
        from minimind_noise_swap import swapped_first_pair_sampler
        trainer_utils.SkipBatchSampler = swapped_first_pair_sampler(trainer_utils.SkipBatchSampler)
    if variant == "init_1ulp" and fresh and int(os.environ["RANK"]) == 0:
        from trainer import trainer_utils
        original_init = trainer_utils.init_model

        @functools.wraps(original_init)
        def shifted_init(*args, **kwargs):
            model, tokenizer = original_init(*args, **kwargs)
            # Native DDP initialization broadcasts rank 0's model after this hook.
            audit = dict(seed=42, scheme=scheme, **shift_initial_coordinate(model))
            text = json.dumps(audit, allow_nan=False)
            path = root / f"init_1ulp_{scheme}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp, path)
            print("INIT_1ULP " + text, flush=True)
            return model, tokenizer

        trainer_utils.init_model = shifted_init
    sys.argv = [str(trainer), *native_args]
    runpy.run_path(str(trainer), run_name="__main__")


def noise_launcher(native_launcher, worker, variant, trainer, *native_args):
    # Keep the pilot's arguments, GPU launcher, logging and resume behavior.
    args = list(map(str, native_args))
    idx = args.index("--learning_rate") + 1
    if float(args[idx]) != BASE_LR:
        raise ValueError("Pilot base LR no longer matches the registered protocol")
    args[idx] = repr(BASE_LR * LR_MULTIPLIERS[variant])
    return native_launcher(worker, "--rank-worker", trainer, variant, *args)


def checkpoint_state(path, scheme, variant):
    import torch
    from model.quant_policy import validate_resume_policy
    from trainer.trainer_utils import get_lr

    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    validate_resume_policy(state, scheme, seed=42)
    step = int(state["step"])
    if ((state["world_size"], state["epoch"]) != (8, 0) or not 0 < step <= 2048 or step % 4
            or state.get("seed", 42) != 42 or not state["model"]):
        raise ValueError(f"Invalid seed-42 noise checkpoint cursor: {path}")
    expected_lr = variant_schedule(get_lr, variant)(step, 2048, BASE_LR * LR_MULTIPLIERS[variant])
    moments = state["optimizer"]["state"]
    groups = state["optimizer"]["param_groups"]
    if (not groups or not all(math.isclose(g["lr"], expected_lr, rel_tol=1e-10) for g in groups)
            or not moments or not all({"exp_avg", "exp_avg_sq", "step"} <= m.keys() for m in moments.values())
            or max(int(m["step"]) for m in moments.values()) != step // 4):
        raise ValueError(f"AdamW update or learning rate differs from {variant} at microbatch {step}: {path}")
    return state


def run_arm(work, scheme, variant):
    """Return the terminal same-format validation (CE, PPL) for one isolated arm."""
    from minimind_grid_train import arm_paths, read_score, recover_export

    if scheme not in ("bf16", "mxfp8") or variant not in LR_MULTIPLIERS:
        raise ValueError(f"Unsupported MoE sensitivity arm {scheme}/{variant}")
    work = Path(work).resolve()
    spec = importlib.util.spec_from_file_location("frozen_noise_pilot", work / "scripts/minimind_v100_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    # The local module object is private to this call; the staged pilot files stay byte-identical.
    pilot.local_torchrun = functools.partial(noise_launcher, pilot.local_torchrun,
                                              work / "scripts/minimind_noise_arm.py", variant)
    checkpoint, export, score = arm_paths(work, "moe", 42, scheme)
    if checkpoint.is_file():
        state = checkpoint_state(checkpoint, scheme, variant)
        if variant == "init_1ulp" and not (work / f"init_1ulp_{scheme}.json").is_file():
            raise ValueError(f"Missing rank-0 initialization audit for {checkpoint}")
        if state["step"] == 2048 and not export.is_file():
            import torch
            export.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=export.parent, delete=False) as tmp:
                restored = Path(tmp.name)
            try:
                torch.save(state["model"], restored)
                os.replace(restored, export)
            finally:
                restored.unlink(missing_ok=True)
        del state
    elif export.is_file():
        # Native first save may publish the export and then be interrupted before saving AdamW.
        recover_export(work, checkpoint, export, 42, scheme)

    pilot.train(work, True, scheme, seed=42, resume_if_present=True)
    if checkpoint_state(checkpoint, scheme, variant)["step"] != 2048:
        raise ValueError(f"MoE training stopped before 512 updates: {checkpoint}")
    if variant == "init_1ulp" and not (work / f"init_1ulp_{scheme}.json").is_file():
        raise ValueError(f"Missing rank-0 initialization audit for {checkpoint}")
    if score.is_file() and score.stat().st_mtime_ns > max(checkpoint.stat().st_mtime_ns, export.stat().st_mtime_ns):
        try:
            return read_score(score, "moe", 42, scheme)
        except (ValueError, KeyError, TypeError):
            print(f"Re-evaluating invalid result: {score}", flush=True)
    subprocess.run([sys.executable, str(work / "scripts/minimind_v100_pilot.py"), "--root", str(work),
                    "eval", scheme, "--seed", "42", "--moe", "--eval-scheme", "same"], cwd=work, check=True)
    return read_score(score, "moe", 42, scheme)


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] != "--rank-worker" or sys.argv[3] not in LR_MULTIPLIERS:
        raise SystemExit("Internal rank worker. Use scripts/minimind_moe_noise_floor.py")
    rank_worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
