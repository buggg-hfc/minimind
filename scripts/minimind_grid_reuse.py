"""Read completed, compatible runs from the earlier two-format MoE experiment."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import pickle

from minimind_grid_train import arm_paths, check_finished, digest, read_score


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _collect(source, work, seeds):
    prior = _read_json(source / "protocol.json")
    current = _read_json(work / "protocol.json")
    expected = dict(architecture="moe", effective_updates=512, microbatches=2048, world_size=8,
                    seq_len=512, batch_per_gpu=2, accumulation=4, native_sampler_seed=0,
                    train_documents=32768, validation_documents=1024, validation_targets=236442)
    if any(prior.get(key) != value for key, value in expected.items()) or prior["versions"] != current["versions"]:
        raise ValueError("历史训练设置或软件版本与本次实验不同")
    # Compare every source and input used by native training and same-format validation.
    # The study drivers and statistical postprocessing differ between the two experiments.
    for name, sha in current["files"].items():
        if name.startswith(("model/", "trainer/", "dataset/", "runs/v100-pilot/data/")) or name == "scripts/minimind_v100_pilot.py":
            if prior["files"].get(name) != sha or digest(source / name) != sha:
                raise ValueError(f"历史训练代码或输入指纹不符：{name}")

    reused = {}
    for seed in seeds:
        if seed not in prior["seeds"]:
            continue
        for scheme in ("mxfp8", "hif4_thr_3mm"):
            if scheme not in prior["schemes"]:
                continue
            checkpoint, export, score = arm_paths(source, "moe", seed, scheme)
            if not all(path.is_file() for path in (checkpoint, export, score)):
                continue
            if score.stat().st_mtime_ns < max(export.stat().st_mtime_ns, checkpoint.stat().st_mtime_ns):
                continue
            try:
                check_finished(checkpoint, seed, scheme)
                reused[seed, scheme] = read_score(score, "moe", seed, scheme)
            except (KeyError, OSError, TypeError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                print(f"历史 MoE seed={seed} {scheme} 校验未通过，将在新目录训练：{exc}", flush=True)
    return reused


@contextmanager
def previous_moe_scores(previous_output, work, seeds):
    """Hold the old driver's advisory lock while reading and using its finished runs."""
    work = Path(work).resolve()
    if previous_output is None or not Path(previous_output).expanduser().is_dir():
        yield {}
        return
    previous = Path(previous_output).expanduser().resolve()
    source = previous / "work"
    if source == work:
        raise ValueError("五格式输出目录与两格式实验目录必须独立")
    # The legacy driver takes LOCK_EX before creating its protocol or training on the GPUs.
    # Only the coordination file is opened for creation; historical results stay read-only.
    with (previous / ".paired-seeds.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            print("旧版 MoE 两格式任务仍在运行，等待其释放 GPU 后继续。", flush=True)
            fcntl.flock(lock, fcntl.LOCK_SH)
        scores = {}
        if (source / "protocol.json").exists():
            try:
                scores = _collect(source, work, seeds)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                print(f"历史实验的协议、代码或数据校验未通过，按新实验运行：{exc}", flush=True)
        print(f"从 {source} 核实可复用 {len(scores)} 组已完成的 MoE 训推结果。", flush=True)
        yield scores
