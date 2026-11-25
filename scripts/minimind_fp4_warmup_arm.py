"""Run one staged seed-42 MoE warmup experiment and score its native export."""
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import subprocess
import sys
import tempfile

VARIANT = "warmup26"
TOTAL_MICROBATCHES = 2048


def _staged(work, name):
    path = work / "scripts" / (name + ".py")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_for(work, lr):
    """Bind the launch LR and the checkpoint LR check to one private worker module."""
    if lr not in (5e-4, 1e-4):
        raise ValueError(f"Unsupported warmup base learning rate: {lr}")
    worker = _staged(work, "minimind_noise_arm")
    # The legacy launcher and checkpoint verifier share this multiplier. The
    # rank process gets the scaled --learning_rate and the original warmup26 hook.
    worker.LR_MULTIPLIERS = {**worker.LR_MULTIPLIERS, VARIANT: lr / worker.BASE_LR}
    return worker


def _paths(work, scheme):
    from minimind_grid_train import arm_paths
    from model.quant_policy import FOUR_BIT_SCHEMES

    if scheme not in ("mxfp8", *FOUR_BIT_SCHEMES):
        raise ValueError(f"Unsupported warmup precision: {scheme}")
    return arm_paths(work, "moe", 42, scheme)


def _fresh_score(checkpoint, export, result, scheme):
    from minimind_grid_train import read_score

    if (not result.is_file() or not export.is_file() or
            result.stat().st_mtime_ns <= max(checkpoint.stat().st_mtime_ns, export.stat().st_mtime_ns)):
        return None
    try:
        return read_score(result, "moe", 42, scheme)
    except (ValueError, KeyError, TypeError):
        print(f"Re-evaluating invalid result: {result}", flush=True)
        return None


def cached_score(work, scheme, lr):
    """Return same-format CE/PPL only for a complete run of the requested LR."""
    work = Path(work).resolve()
    checkpoint, export, result = _paths(work, scheme)
    if not checkpoint.is_file() or not result.is_file() or not export.is_file():
        return None
    worker = _worker_for(work, lr)
    if worker.checkpoint_state(checkpoint, scheme, VARIANT)["step"] != TOTAL_MICROBATCHES:
        return None
    return _fresh_score(checkpoint, export, result, scheme)


def _restore_final_export(export, weights):
    import torch

    export.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=export.parent, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        torch.save(weights, path)
        os.replace(path, export)
    finally:
        path.unlink(missing_ok=True)


def run_arm(work, scheme, lr):
    """Finish one warmup26 arm, resuming its native AdamW checkpoint when present."""
    from minimind_grid_train import read_score, recover_export

    work = Path(work).resolve()
    checkpoint, export, result = _paths(work, scheme)
    worker = _worker_for(work, lr)
    complete = False
    if checkpoint.is_file():
        state = worker.checkpoint_state(checkpoint, scheme, VARIANT)
        complete = state["step"] == TOTAL_MICROBATCHES
        if complete and not export.exists():
            _restore_final_export(export, state["model"])
        del state
    elif export.is_file():
        recover_export(work, checkpoint, export, 42, scheme)

    if not complete:
        pilot = _staged(work, "minimind_v100_pilot")
        pilot.local_torchrun = partial(worker.noise_launcher, pilot.local_torchrun, worker.__file__, VARIANT)
        pilot.train(work, True, scheme, seed=42, resume_if_present=True)
        if worker.checkpoint_state(checkpoint, scheme, VARIANT)["step"] != TOTAL_MICROBATCHES:
            raise ValueError(f"Training stopped before 512 AdamW updates: {checkpoint}")
    ce_ppl = _fresh_score(checkpoint, export, result, scheme)
    if ce_ppl is None:
        subprocess.run([sys.executable, str(work / "scripts/minimind_v100_pilot.py"), "--root", str(work),
                        "eval", scheme, "--seed", "42", "--moe", "--eval-scheme", "same"], cwd=work, check=True)
        ce_ppl = read_score(result, "moe", 42, scheme)
    return ce_ppl
