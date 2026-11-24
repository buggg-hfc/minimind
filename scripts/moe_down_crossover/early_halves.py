"""Run both late policies from the same early-down-MXFP8 prefix on the eight V100s."""
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

HIF4 = "hif4_thr_3mm"
MIXED = "hif4_thr_3mm_ffn_down_mxfp8"
BRANCHES = ("hif4", "hif4_down_mxfp8")
MODULES = ("fork_worker.py", "fork_eval.py", "fork_report.py", "fork_metrics.py")
VALID_SHA256 = "72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04"
CKPT = f"checkpoints/reciprocal_s42_{MIXED}_r2_768_moe_resume.pth"
MIXED_POLICY = {"attention": HIF4, "router": "bf16", "lm_head": "bf16",
                "ffn": {"down_proj": "mxfp8", "gate_proj": HIF4, "up_proj": HIF4}}


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("训练器版本不匹配，不能定位 " + old.splitlines()[0])
    return text.replace(old, new)


def adapt_worker(text):
    """Change only the accepted fresh-fork checkpoint; retain native resume and metrics."""
    # The follow-on package adds exactly these two guards for a 100 -> 256 handoff.
    # Both belong only to the old continuation. Its otherwise identical worker is
    # usable when the user retained only pair-1024-to-2048 on this machine.
    continue_only = ('    if not restart:\n'
                     '        raise FileNotFoundError(f"续训需要分支自己的 checkpoint.pt：{saved}")\n')
    continuation_guard = ('        handoff = (args.updates == 256 and ck.get("relative_update") == 100\n'
                          '                   and ck.get("step") == base_step + ACCUM * 100\n'
                          '                   and ck["settings"] == dict(settings, updates=100))\n'
                          '        if ck["settings"] != settings and not handoff:\n')
    if "if ck[\"settings\"] != settings and not handoff:" in text or continue_only in text:
        text = _replace_once(text, continue_only, "")
        text = _replace_once(text, continuation_guard, '        if ck["settings"] != settings:\n')
    if text.count('        if ck["settings"] != settings:\n') != 1:
        raise ValueError("缺少原分支严格的续训身份校验")
    text = _replace_once(text, 'f"fork1024_s42_{HIF4}_r2_{hidden}_moe_resume.pth"',
                         f'f"reciprocal_s42_{MIXED}_r2_{{hidden}}_moe_resume.pth"')
    text = _replace_once(
        text, '        if ck.get("quant_sim") != HIF4 or ck.get("residual_policy") != RESIDUAL:\n',
        f'        if (ck.get("quant_sim") != {MIXED!r} or ck.get("residual_policy") != RESIDUAL\n'
        f'                or ck.get("seed") != 42 or ck.get("projection_policy") != {MIXED_POLICY!r}):\n')
    compile(text, "fork_worker.py", "exec")
    return text


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path, expected):
    if _hash(path) != expected:
        raise ValueError(f"SHA256 不符：{path}")
    return expected


def _snapshot(path, scheme):
    spec = _read(path / "inputs.json")
    if (spec["scheme"], spec["seed"], spec["target_microbatch"], spec["schedule_microbatches"],
            spec["world_size"]) != (scheme, 42, 1024, 2048, 8):
        raise ValueError(f"快照设置不匹配：{path}")
    for name, digest in spec["files"].items():
        _verify(path / name, digest)
    return spec["files"]


def _prepare(original, anchor, old, out, valid):
    import torch

    reference, candidate = _snapshot(original, HIF4), _snapshot(anchor, MIXED)
    shared = {rel for rel in reference if rel.startswith(("model/", "data/", "dataset/"))}
    shared.add("trainer/trainer_utils.py")
    for name in shared:
        if candidate.get(name) != reference[name]:
            raise ValueError(f"两种前缀的模型、分词、训练数据或公共工具不一致：{name}")
    old_manifest = old / "experiment.json"
    if old_manifest.is_file():
        old_spec = _read(old_manifest)
        if (old_spec["prefix"]["input_manifest_sha256"] != _hash(original / "inputs.json")
                or old_spec["valid_sha256"] != VALID_SHA256):
            raise ValueError(f"历史分支与给定的原始前缀和验证集不匹配：{old}")
        hashes = old_spec["workers"]
    else:
        old_manifest = old / "continuation.json"
        old_spec = _read(old_manifest)
        if old_spec["valid_sha256"] != VALID_SHA256 or old_spec["target_update"] != 256:
            raise ValueError(f"历史续训记录与训练任务不匹配：{old}")
        # If the old optimizer file was retained, read just its mmap metadata.
        # Only the verified frozen scripts are required from this prior run.
        prior_checkpoint = old / BRANCHES[0] / "checkpoint.pt"
        if prior_checkpoint.is_file():
            saved = torch.load(prior_checkpoint, weights_only=True, map_location="cpu", mmap=True)
            settings = saved["settings"]
            if (settings["inputs_sha256"], settings["seed"], settings["base_microbatch"],
                    settings["world_size"], settings["horizon"], settings["valid_sha256"]) != (
                    _hash(original / "inputs.json"), 42, 1024, 8, 2048, VALID_SHA256):
                raise ValueError("历史续训检查点与给定的原始前缀或数据不匹配")
            del saved
        hashes = old_spec["worker_hashes"]
    code = {}
    for name in MODULES:
        src = old / "scripts" / name
        _verify(src, hashes[name])
        code[name] = src.read_bytes()
    code["fork_worker.py"] = adapt_worker(code["fork_worker.py"].decode("utf-8")).encode("utf-8")
    _verify(valid, VALID_SHA256)
    env = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
           "datasets": version("datasets"), "transformers": version("transformers")}
    if "environment" in old_spec and env != old_spec["environment"]:
        raise ValueError(f"当前 Python 包与历史实验不匹配：{old_manifest}")
    manifest = {
        "early_policy": MIXED, "late_a": HIF4, "late_b": MIXED,
        "early_prefix": str(anchor), "early_inputs_sha256": _hash(anchor / "inputs.json"),
        "early_checkpoint_sha256": _hash(anchor / CKPT),
        "rng_sha256": [_hash(anchor / "rng" / f"micro-1024-rank-{r}.pt") for r in range(8)],
        "original_prefix": str(original), "original_inputs_sha256": _hash(original / "inputs.json"),
        "historic_scripts_sha256": hashes, "historic_manifest_sha256": _hash(old_manifest),
        "valid_sha256": VALID_SHA256, "train_sha256": reference["data/train.jsonl"],
        "world_size": 8, "microbatch_start": 1024, "microbatch_end": 2048,
        "relative_updates": 256, "validation_interval": 20, "seed": 42, "environment": env,
        "installed_scripts_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in code.items()},
    }
    if out.exists():
        if _read(out / "second_half.json") != manifest:
            raise ValueError(f"已有输出属于另一次实验：{out}")
        for name in MODULES:
            _verify(out / "scripts" / name, manifest["installed_scripts_sha256"][name])
        _verify(out / "valid.jsonl", VALID_SHA256)
        return
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".prepare-early-pair-") as tmp:
        stage = Path(tmp) / "run"
        (stage / "scripts").mkdir(parents=True)
        for name, blob in code.items():
            (stage / "scripts" / name).write_bytes(blob)
        shutil.copyfile(valid, stage / "valid.jsonl")
        _verify(stage / "valid.jsonl", VALID_SHA256)
        (stage / "second_half.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(out)


def _projection_counts(branch):
    expected = {r + "/" + HIF4: 8 for r in ("q_proj", "k_proj", "v_proj", "o_proj")}
    expected.update({r + "/" + HIF4: 32 for r in ("up_proj", "gate_proj")})
    late_down = "mxfp8" if branch == BRANCHES[1] else HIF4
    expected.update({"down_proj/" + late_down: 32, "gate/bf16": 8, "lm_head/bf16": 1})
    return expected


def _branch(anchor, out, branch):
    run = out / branch
    run.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc-per-node=8", str(out / "scripts/fork_worker.py"), "--prefix", str(anchor),
               "--branch-dir", str(run), "--branch", branch, "--valid", str(out / "valid.jsonl"),
               "--updates", "256", "--eval-every", "20"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    log = run / "console.log"
    print(f"早期 down=MXFP8，后半程分支 {branch}，日志 {log}", flush=True)
    with log.open("a", encoding="utf-8") as dst:
        with subprocess.Popen(command, cwd=out, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                dst.write(line)
                dst.flush()
            if process.wait():
                raise RuntimeError(f"训练中断，可重新运行同一命令：{log}")
    if _read(run / "policy.json") != _projection_counts(branch):
        raise ValueError(f"后半程配置与实际应用的量化格式不匹配：{run}")


def _check_report(path):
    with path.open(encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
    if [int(r["relative_update"]) for r in rows] != list(range(257)):
        raise ValueError(f"更新编号不完整：{path}")
    for u, row in enumerate(rows):
        if u > 0 and (int(row["world_size"]), int(row["route_paired_ranks"]),
                      int(row["optimizer_update"]), int(row["microbatch"])) != (8, 8, 256+u, 1024+4*u):
            raise ValueError(f"训练游标或 rank 不符：{path}，更新 {u}")
        if u % 20 == 0 or u == 256:
            if int(row["eval_valid_tokens"]) != 236442 or not all(
                    math.isfinite(float(row[col])) for col in ("eval_ce_a", "eval_ce_b", "eval_ppl_a", "eval_ppl_b")):
                raise ValueError(f"验证数据不完整：{path}，更新 {u}")
    final = rows[-1]
    print(f"早期 down=MXFP8，最终晚期 MXFP8 减晚期 HiF4："
          f"ΔCE={float(final['eval_ce_b_minus_a']):+.7f}，"
          f"ΔPPL={float(final['eval_ppl_b']) - float(final['eval_ppl_a']):+.6f}", flush=True)


def finish_pair(original_prefix, early_anchor, old_pair, output, valid=None):
    """Run M→H then M→M, each evaluated with its actual late policy. Return paired CSV."""
    original, anchor, old, out = (Path(p).resolve() for p in (original_prefix, early_anchor, old_pair, output))
    valid = Path(valid).resolve() if valid is not None else old / "valid.jsonl"
    if any(out == p or out in p.parents or p in out.parents for p in (original, anchor, old)):
        raise ValueError("后半程输出必须与公共前缀及历史实验目录隔离")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"这组分支已有任务运行：{out}") from exc
        _prepare(original, anchor, old, out, valid)
        for branch in BRANCHES:
            _branch(anchor, out, branch)
        report = out / "summary.csv"
        subprocess.run([sys.executable, str(out / "scripts/fork_report.py"), str(out / BRANCHES[0]),
                        str(out / BRANCHES[1]), str(report), "--strict"], check=True)
        _check_report(report)
        return report
