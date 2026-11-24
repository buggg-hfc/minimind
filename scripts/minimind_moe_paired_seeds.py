#!/usr/bin/env python3
"""Compare full HiF4 R2 and full MXFP8 on the same new MiniMind-MoE seeds."""
import argparse
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from moe_paired_stats import write_paired_summary


REPO = Path(__file__).resolve().parent.parent
SCHEMES = ("mxfp8", "hif4_thr_3mm")
SPLITS = {
    "train": "4d691136c467563ebea032e972747e957bdc2d06919b9f1a8baea05ff466ffb6",
    "valid": "72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04",
}
DEFAULT_DATA = Path("/home/lz/minimind/runs/v100-pilot/data")
DEFAULT_OUTPUT = Path("/data/lz/minimind-moe-paired-seeds")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(output, data, seeds, margin):
    """Pin the complete plan, executable sources and actual input bytes before training."""
    code = [*sorted((REPO / "model").glob("*.py")), *sorted((REPO / "model").glob("*.json")),
            REPO / "dataset/__init__.py", REPO / "dataset/lm_dataset.py",
            REPO / "trainer/train_pretrain.py", REPO / "trainer/trainer_utils.py",
            REPO / "scripts/minimind_v100_pilot.py", REPO / "tests/test_v100_batch_inference.py"]
    files = {str(path.relative_to(REPO)): path for path in code}
    for name in SPLITS:
        files[f"runs/v100-pilot/data/{name}.jsonl"] = data / f"{name}.jsonl"
    hashes = {name: digest(path) for name, path in files.items()}
    for name, expected in SPLITS.items():
        if hashes[f"runs/v100-pilot/data/{name}.jsonl"] != expected:
            raise ValueError(f"{name}.jsonl 的 SHA256 与 seed 42 的实验数据不符")
    plan = dict(seeds=seeds, schemes=list(SCHEMES), margin_ce=margin, confidence_one_sided=0.95,
                architecture="moe", effective_updates=512, microbatches=2048, world_size=8,
                seq_len=512, batch_per_gpu=2, accumulation=4, native_sampler_seed=0,
                train_documents=32768, validation_documents=1024, validation_targets=236442,
                files=hashes, runner_sha256=digest(Path(__file__)),
                analysis_sha256=digest(Path(__file__).with_name("moe_paired_stats.py")),
                versions={name: version(name) for name in ("torch", "transformers", "datasets")})
    work = output / "work"
    if work.exists():
        if json.loads((work / "protocol.json").read_text(encoding="utf-8")) != plan:
            raise ValueError(f"已冻结的种子、阈值、依赖、数据或代码与本次设置不符：{work}/protocol.json")
        for name, expected in hashes.items():
            if digest(work / name) != expected:
                raise ValueError(f"实验快照文件发生改变：{work / name}")
    else:
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            stage = Path(tmp) / "work"
            for name, src in files.items():
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                if digest(dest) != hashes[name]:
                    raise ValueError(f"创建快照时输入发生改变：{src}")
            (stage / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            stage.rename(work)
    return work


def run_arm(work, seed, scheme):
    import torch
    import minimind_v100_pilot as pilot
    from model.quant_policy import scheme_artifact_name, validate_resume_policy

    name = scheme_artifact_name(scheme)
    stem = f"pilot32k_s{seed}_{name}_768_moe"
    checkpoint = work / "checkpoints" / f"{stem}_resume.pth"
    export = work / "out" / f"{stem}.pth"
    if export.is_file() and not checkpoint.exists():
        # The native trainer atomically publishes the export before the AdamW state.
        # A crash during the first save can leave a model file with no resumable state.
        archive = work / "orphaned-exports"
        archive.mkdir(exist_ok=True)
        fd, backup = tempfile.mkstemp(prefix=stem + "-", suffix=".pth", dir=archive)
        os.close(fd)
        os.replace(export, backup)
        print(f"已保留首次保存中断时的单独导出 {backup}，从头重跑该格式", flush=True)
    pilot.train(work, True, scheme, seed=seed, resume_if_present=True)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    validate_resume_policy(state, scheme, seed=seed)
    if (state["world_size"], state["epoch"], state["step"]) != (8, 0, 2048):
        raise ValueError(f"模型没有完成一个 epoch：{checkpoint}")
    opt = state["optimizer"]["state"]
    if not opt or max(int(v["step"]) for v in opt.values()) != 512:
        raise ValueError(f"AdamW 计数不符：{checkpoint}")
    del state, opt
    subprocess.run([sys.executable, str(work / "scripts/minimind_v100_pilot.py"),
                    "--root", str(work), "eval", scheme, "--moe", "--seed", str(seed),
                    "--eval-scheme", "same", "--reuse"], cwd=work, check=True)
    path = work / "runs/v100-pilot/results" / f"seed{seed}" / f"moe__{name}__eval_{scheme}.json"
    score = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(seed=seed, train_scheme=scheme, eval_scheme=scheme, documents=1024,
                    valid_tokens=236442, batch_size=2, seq_len=512, architecture="moe",
                    valid_sha256=SPLITS["valid"], checkpoint=export.name)
    ce, ppl = score["mean_ce"], score["ppl"]
    if (any(score.get(k) != v for k, v in expected.items()) or not math.isfinite(ce) or ce < 0
            or not math.isfinite(ppl) or ppl < 1 or not math.isclose(math.log(ppl), ce, rel_tol=1e-8)):
        raise ValueError(f"验证结果格式或数据不符：{path}")
    return ce, ppl


def record_pairs(path, rows):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pairs-")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def run(args):
    if (len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or 42 in args.seeds
            or any(not 0 <= s <= 2**32-8 for s in args.seeds)):
        raise ValueError("至少预设两个互异的新 seed，排除已用于探索的 seed 42，并为 seed+rank 保留 uint32 范围")
    if not math.isfinite(args.margin_ce) or args.margin_ce <= 0:
        raise ValueError("margin-ce 必须是预先选定的正有限数值（nat/token）")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".paired-seeds.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有任务在使用 {output}") from exc
        work = prepare(output, args.data.expanduser().resolve(), args.seeds, args.margin_ce)
        sys.path[:0] = [str(work), str(work / "scripts")]
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(work), os.environ.get("PYTHONPATH")]))
        os.environ.setdefault("HF_DATASETS_CACHE", str(output / "hf-datasets-cache"))
        import torch
        if torch.cuda.device_count() != 8 or any("V100" not in torch.cuda.get_device_name(i) for i in range(8)):
            raise RuntimeError("本配对实验需要 8 张可见 V100")
        print(f"方案已冻结 {work / 'protocol.json'}，历史 seed 42 只用于探索", flush=True)
        rows = []
        for seed in args.seeds:
            print(f"开始配对 seed={seed}", flush=True)
            (ce_m, ppl_m), (ce_h, ppl_h) = [run_arm(work, seed, scheme) for scheme in SCHEMES]
            rows.append(dict(seed=seed, optimizer_updates=512, valid_tokens=236442, ce_mxfp8=ce_m,
                             ce_hif4=ce_h, delta_ce=ce_h-ce_m, ppl_mxfp8=ppl_m, ppl_hif4=ppl_h))
            record_pairs(output / "paired_seeds.csv", rows)
            print(f"seed={seed}  ΔCE(HiF4−MXFP8)={ce_h-ce_m:+.6f}", flush=True)
        report = write_paired_summary(output, [row["delta_ce"] for row in rows], args.margin_ce)
        print(f"PAIRED_SEEDS_READY {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46])
    parser.add_argument("--margin-ce", required=True, type=float, help="预设允许的最大验证 CE 劣化，单位 nat/token")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="现有 train.jsonl、valid.jsonl 所在目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())
