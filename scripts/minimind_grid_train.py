"""Stage and run the native eight-V100 pilot for a dense or MoE grid arm."""
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from minimind_moe_paired_seeds import DEFAULT_DATA, SPLITS, digest


REPO = Path(__file__).resolve().parents[1]
FOUR_BIT = ("hif4_thr_3mm", "nvint4_3mm", "nvfp4_3mm")
SCHEMES = (*FOUR_BIT, "mxfp8", "bf16")
DW_SCHEMES = tuple(f"{base}_dw_mxfp8" for base in FOUR_BIT)
ARCHITECTURES = ("dense", "moe")


def prepare(output, data, seeds, schemes=SCHEMES):
    """Freeze training sources, split bytes and protocol independently of older pilots."""
    output, data = Path(output).resolve(), Path(data).resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = [*sorted((REPO / "model").glob("*.py")), *sorted((REPO / "model").glob("*.json")),
            REPO / "dataset/__init__.py", REPO / "dataset/lm_dataset.py", REPO / "trainer/train_pretrain.py",
            REPO / "trainer/trainer_utils.py", REPO / "scripts/minimind_v100_pilot.py",
            REPO / "tests/test_v100_batch_inference.py"]
    files = {str(src.relative_to(REPO)): src for src in code}
    files.update({f"runs/v100-pilot/data/{name}.jsonl": data / f"{name}.jsonl" for name in SPLITS})
    hashes = {rel: digest(src) for rel, src in files.items()}
    for name, expected in SPLITS.items():
        if hashes[f"runs/v100-pilot/data/{name}.jsonl"] != expected:
            raise ValueError(f"{name}.jsonl SHA256 与原始 32k pilot 不符")
    plan = dict(seeds=list(seeds), architectures=list(ARCHITECTURES), schemes=list(schemes),
                effective_updates=512, microbatches=2048, world_size=8, seq_len=512, batch_per_gpu=2,
                accumulation=4, native_sampler_seed=0, train_documents=32768, validation_documents=1024,
                validation_targets=236442, files=hashes, runner_sha256=digest(Path(__file__)),
                versions={name: version(name) for name in ("torch", "transformers", "datasets")})
    work = output / "work"
    if work.exists():
        if json.loads((work / "protocol.json").read_text(encoding="utf-8")) != plan:
            raise ValueError(f"现有输出的训练配置、数据或代码与本次不同：{work}")
        for rel, expected in hashes.items():
            if digest(work / rel) != expected:
                raise ValueError(f"训练快照内容已改变：{work / rel}")
    else:
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            stage = Path(tmp) / "work"
            for rel, src in files.items():
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                if digest(dest) != hashes[rel]:
                    raise ValueError(f"复制过程中输入内容发生变化：{src}")
            (stage / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            stage.rename(work)
    sys.path[:0] = [str(work), str(work / "scripts")]
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(work), os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("HF_DATASETS_CACHE", str(output / "hf-datasets-cache"))
    return work


def arm_paths(work, architecture, seed, scheme):
    from model.quant_policy import scheme_artifact_name

    if architecture not in ARCHITECTURES or scheme not in (*SCHEMES, *DW_SCHEMES):
        raise ValueError(f"Unsupported pilot case: {architecture}, {scheme}")
    name = scheme_artifact_name(scheme)
    suffix = "_moe" if architecture == "moe" else ""
    stem = f"pilot32k_s{seed}_{name}_768{suffix}"
    results = work / "runs/v100-pilot/results"
    if seed != 42:
        results = results / f"seed{seed}"
    return (work / "checkpoints" / f"{stem}_resume.pth", work / "out" / f"{stem}.pth",
            results / f"{architecture}__{name}__eval_{scheme}.json")


def _completed_state(path, seed, scheme):
    import torch
    from model.quant_policy import validate_resume_policy

    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    validate_resume_policy(state, scheme, seed=seed)
    if (state["world_size"], state["epoch"], state["step"]) != (8, 0, 2048):
        raise ValueError(f"终点断点游标与第 512 次更新不符：{path}")
    if state.get("seed", seed) != seed or not state["model"]:
        raise ValueError(f"终点权重或种子不符：{path}")
    moments = state["optimizer"]["state"]
    groups = state["optimizer"]["param_groups"]
    if (not moments or not all({"exp_avg", "exp_avg_sq", "step"} <= s.keys() for s in moments.values())
            or max(int(s["step"]) for s in moments.values()) != 512 or not groups
            or not all(math.isclose(g["lr"], 5e-5, rel_tol=1e-9) for g in groups)):
        raise ValueError(f"AdamW 终点状态不符：{path}")
    return state


def check_finished(path, seed, scheme):
    _completed_state(path, seed, scheme)


def read_score(path, architecture, seed, scheme):
    from model.quant_policy import RESIDUAL_SCHEMES, RESIDUAL_POLICY, projection_policy, scheme_artifact_name

    score = json.loads(path.read_text(encoding="utf-8"))
    suffix = "_moe" if architecture == "moe" else ""
    filename = f"pilot32k_s{seed}_{scheme_artifact_name(scheme)}_768{suffix}.pth"
    expected = dict(architecture=architecture, seed=seed, train_scheme=scheme, eval_scheme=scheme,
                    documents=1024, valid_tokens=236442, seq_len=512, batch_size=2,
                    valid_sha256=SPLITS["valid"], checkpoint=filename)
    if scheme in RESIDUAL_SCHEMES:
        expected.update(residual_policy=RESIDUAL_POLICY, eval_residual_policy=RESIDUAL_POLICY)
    if scheme in DW_SCHEMES:
        policy = projection_policy(scheme)
        expected.update(train_projection_policy=policy, eval_projection_policy=policy)
    ce, ppl = score["mean_ce"], score["ppl"]
    if (any(score.get(k) != v for k, v in expected.items()) or not math.isfinite(ce) or ce < 0
            or not math.isfinite(ppl) or ppl < 1 or not math.isclose(math.log(ppl), ce, rel_tol=1e-8)):
        raise ValueError(f"同格式评估的配置、验证集或 CE/PPL 不符：{path}")
    return ce, ppl


def recover_export(work, checkpoint, export, seed, scheme):
    import torch

    if export.is_file() and not checkpoint.exists():
        # An interrupted first save can publish weights before the resumable AdamW state.
        archive = work / "orphaned-exports"
        archive.mkdir(exist_ok=True)
        fd, backup = tempfile.mkstemp(dir=archive, prefix=export.stem+"-", suffix=".pth")
        os.close(fd)
        os.replace(export, backup)
        print(f"首个断点未写完；导出已保留在 {backup}，将重跑该组。", flush=True)
    elif checkpoint.is_file() and not export.exists():
        state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        if state["step"] == 2048:
            del state
            model = _completed_state(checkpoint, seed, scheme)["model"]
            export.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=export.parent, prefix="recover-", suffix=".pth")
            os.close(fd)
            try:
                torch.save(model, tmp)
                os.replace(tmp, export)
            finally:
                Path(tmp).unlink(missing_ok=True)


def score_arm(work, architecture, seed, scheme):
    import minimind_v100_pilot as pilot

    checkpoint, export, result = arm_paths(work, architecture, seed, scheme)
    recover_export(work, checkpoint, export, seed, scheme)
    moe = architecture == "moe"
    pilot.train(work, moe, scheme, seed=seed, resume_if_present=True)
    check_finished(checkpoint, seed, scheme)
    if result.is_file() and result.stat().st_mtime_ns > export.stat().st_mtime_ns:
        try:
            return read_score(result, architecture, seed, scheme)
        except (ValueError, KeyError, TypeError):
            print(f"重新评估未通过核验的结果：{result}", flush=True)
    cmd = [sys.executable, str(work / "scripts/minimind_v100_pilot.py"), "--root", str(work),
           "eval", scheme, "--seed", str(seed), "--eval-scheme", "same"]
    if moe:
        cmd.append("--moe")
    subprocess.run(cmd, cwd=work, check=True)
    return read_score(result, architecture, seed, scheme)
