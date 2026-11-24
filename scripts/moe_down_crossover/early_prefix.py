"""Build the first 1024 microbatches with MXFP8 only on expert down projections."""
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SCHEME = "hif4_thr_3mm_ffn_down_mxfp8"
BASE_SCHEME = "hif4_thr_3mm"
RESIDUAL = "bf16_diff_shared_tensor_scale_v2"
POLICY = {"attention": BASE_SCHEME, "ffn": {"gate_proj": BASE_SCHEME, "up_proj": BASE_SCHEME,
           "down_proj": "mxfp8"}, "router": "bf16", "lm_head": "bf16"}
STATE = f"checkpoints/reciprocal_s42_{SCHEME}_r2_768_moe_resume.pth"
VALID_SHA = "72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04"


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_hash(path, expected):
    if _sha(path) != expected:
        raise ValueError(f"校验失败：{path}")


def policy_code(frozen_worker):
    """Copy only the dispatch function actually used by the completed branch."""
    defs = [n for n in ast.parse(frozen_worker).body if isinstance(n, ast.FunctionDef) and n.name == "configure"]
    if len(defs) != 1:
        raise ValueError("归档分支代码中找不到唯一的 configure")
    source = (
        "from collections import Counter\nimport torch\n"
        f"HIF4 = {BASE_SCHEME!r}\nSCHEME = {SCHEME!r}\nRESIDUAL = {RESIDUAL!r}\nPOLICY = {POLICY!r}\n"
        + ast.get_source_segment(frozen_worker, defs[0]) + "\n\n"
        "def anchored_checkpoint(native):\n"
        "    def checkpoint(*args, **kwargs):\n"
        "        if kwargs.get('model') is not None:\n"
        "            kwargs.update(quant_sim=SCHEME, residual_policy=RESIDUAL, projection_policy=POLICY, seed=42)\n"
        "        return native(*args, **kwargs)\n"
        "    return checkpoint\n\n"
        "def validate_state(state):\n"
        "    if (state.get('quant_sim'), state.get('residual_policy'), state.get('projection_policy'),\n"
        "            state.get('seed')) != (SCHEME, RESIDUAL, POLICY, 42):\n"
        "        raise ValueError('The early checkpoint does not match the expert-down MXFP8 policy')\n"
    )
    compile(source, "anchor_policy.py", "exec")
    return source


def patch_early_trainer(source):
    """The frozen trainer already contains the rank-RNG and 1024/2048 cutoff hooks."""
    for marker in ("restore_rng(start_step)", "capture_rng(step)", "if step == 1024:", "if iters != 2048"):
        if source.count(marker) != 1:
            raise ValueError(f"原公共前缀训练器缺少预期的保存、恢复或日程逻辑：{marker}")

    edits = {
        "def train_epoch(epoch, loader, iters, start_step=0, wandb=None):":
            "from anchor_policy import SCHEME as EARLY_SCHEME, configure as configure_early, anchored_checkpoint, validate_state\n"
            "lm_checkpoint = anchored_checkpoint(lm_checkpoint)\n\n\n"
            "def train_epoch(epoch, loader, iters, start_step=0, wandb=None):",
        "f'{args.save_weight}_{scheme_artifact_name(args.quant_sim)}'":
            "f'{args.save_weight}_{EARLY_SCHEME}_r2'",
        "        install_quantization(model.float(), args.quant_sim)":
            '        if args.quant_sim != "hif4_thr_3mm" or not args.use_moe or args.seed != 42:\n'
            '            raise ValueError("Expected the seed-42 HiF4 MoE base configuration")\n'
            '        counts = configure_early(model.float(), "hif4_down_mxfp8")\n'
            '        Logger(f"early_projection_counts={counts}")',
        'Logger(f\'quant_sim={args.quant_sim} | FP32 carrier | AMP off | save_weight={args.save_weight}\')':
            'Logger(f\'quant_sim={EARLY_SCHEME} | FP32 carrier | AMP off | save_weight={args.save_weight}\')',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f"原训练器与已知 MiniMind pilot 不一致：{old}")
        source = source.replace(old, new)
    source, count = re.subn(r"(?m)^        validate_resume_policy\(ckp_data, args\.quant_sim(?:, seed=args\.seed)?\)$",
                            "        validate_state(ckp_data)", source)
    if count != 1:
        raise ValueError("无法识别原训练器的续训校验")
    compile(source, "train_pretrain.py", "exec")
    return source


def _inputs(original, tools):
    manifest = original / "inputs.json"
    old = json.loads(manifest.read_text(encoding="utf-8"))
    pair_dir = tools.parent
    pair = json.loads((pair_dir / "experiment.json").read_text(encoding="utf-8"))
    if (old.get("scheme"), old.get("seed"), old.get("world_size"), old.get("target_microbatch"),
            old.get("schedule_microbatches")) != (BASE_SCHEME, 42, 8, 1024, 2048):
        raise ValueError("源公共前缀的模型、种子或训练日程不匹配")
    if pair["prefix"]["path"] != str(original) or pair["valid_sha256"] != VALID_SHA:
        raise ValueError("已完成分支引用了不同的前缀或验证集")
    _require_hash(manifest, pair["prefix"]["input_manifest_sha256"])
    _require_hash(pair_dir / "valid.jsonl", VALID_SHA)
    worker = tools / "fork_worker.py"
    _require_hash(worker, pair["workers"]["fork_worker.py"])
    for rel, expected in old["files"].items():
        file = Path(rel)
        if file.is_absolute() or ".." in file.parts or file.parts[0] not in ("model", "trainer", "data", "dataset"):
            raise ValueError(f"无效的公共前缀输入路径：{rel}")
        _require_hash(original / file, expected)
    if old["files"]["data/train.jsonl"] != pair["prefix"]["train_sha256"]:
        raise ValueError("种子 42 的训练集指纹发生变化")
    trainer = "trainer/train_pretrain.py"
    changed = {trainer: patch_early_trainer((original / trainer).read_text(encoding="utf-8")).encode("utf-8"),
               "trainer/anchor_policy.py": policy_code(worker.read_text(encoding="utf-8")).encode("utf-8")}
    for package in ("model", "trainer", "dataset"):
        marker = package + "/__init__.py"
        if marker not in old["files"]:
            changed[marker] = b""
    hashes = dict(old["files"])
    hashes.update({rel: hashlib.sha256(data).hexdigest() for rel, data in changed.items()})
    info = dict(scheme=SCHEME, projection_policy=POLICY, residual_policy=RESIDUAL, seed=42, world_size=8,
                target_microbatch=1024, schedule_microbatches=2048, files=hashes,
                source=str(original), source_inputs_sha256=_sha(manifest),
                source_worker_sha256=pair["workers"]["fork_worker.py"], valid_sha256=VALID_SHA,
                train_sha256=pair["prefix"]["train_sha256"])
    return info, changed


def _prepare(original, anchor, tools):
    info, changed = _inputs(original, tools)
    if anchor.exists():
        if json.loads((anchor / "inputs.json").read_text(encoding="utf-8")) != info:
            raise ValueError(f"已有前缀与本次实验配置不匹配：{anchor}")
    else:
        with tempfile.TemporaryDirectory(dir=anchor.parent, prefix=".early-") as tmp:
            stage = Path(tmp) / "anchor"
            stage.mkdir()
            for rel in info["files"]:
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if rel in changed:
                    dest.write_bytes(changed[rel])
                else:
                    shutil.copy2(original / rel, dest)
            (stage / "rng").mkdir()
            (stage / "inputs.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            for rel, expected in info["files"].items():
                _require_hash(stage / rel, expected)
            stage.rename(anchor)
    for rel, expected in info["files"].items():
        _require_hash(anchor / rel, expected)


def check_anchor(anchor, torch):
    path = anchor / STATE
    if not path.is_file():
        return 0
    c = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = c.get("step")
    if (not isinstance(step, int) or not 0 < step <= 1024 or step % 256 or
            (c.get("quant_sim"), c.get("residual_policy"), c.get("projection_policy"), c.get("seed"),
             c.get("world_size"), c.get("epoch")) != (SCHEME, RESIDUAL, POLICY, 42, 8, 0)):
        raise ValueError(f"前缀断点的数值格式、步数或种子错误：{path}")
    moment_states = c["optimizer"]["state"]
    lr = 5e-4 * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 2048)))
    if (not c.get("model") or "scaler" not in c or not moment_states or
            not all({"exp_avg", "exp_avg_sq", "step"} <= v.keys() for v in moment_states.values()) or
            max(int(v["step"]) for v in moment_states.values()) != step // 4 or
            not all(math.isclose(g["lr"], lr, rel_tol=1e-9) for g in c["optimizer"]["param_groups"])):
        raise ValueError(f"AdamW 状态或学习率日程不符：{path}")
    for rank in range(8):
        state = torch.load(anchor / "rng" / f"micro-{step:04d}-rank-{rank}.pt", weights_only=True, map_location="cpu")
        if not {"cpu", "cuda", "python", "numpy"} <= state.keys():
            raise ValueError(f"rank{rank} RNG 不完整")
    return step


def make_anchor(original_prefix, anchor, tools_dir):
    """Prepare/run the early mixed half. tools_dir is the old pair's frozen scripts directory."""
    import torch
    original, anchor, tools = (Path(p).resolve() for p in (original_prefix, anchor, tools_dir))
    if original == anchor or original in anchor.parents or anchor in original.parents:
        raise ValueError("新前缀目录与原实验目录必须独立")
    anchor.parent.mkdir(parents=True, exist_ok=True)
    with (anchor.parent / f".{anchor.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"正在生成同一前缀：{anchor}")
        _prepare(original, anchor, tools)
        step = check_anchor(anchor, torch)
        if step < 1024:
            if torch.cuda.device_count() != 8:
                raise RuntimeError("早期分支训练需要 8 张可见 GPU")
            cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   "--nproc-per-node=8", str(anchor / "trainer/train_pretrain.py"), "--from_weight", "none",
                   "--from_resume", str(int(step > 0)), "--save_weight", "reciprocal_s42", "--use_moe", "1",
                   "--quant_sim", BASE_SCHEME, "--hidden_size", "768", "--num_hidden_layers", "8",
                   "--max_seq_len", "512", "--epochs", "1", "--batch_size", "2", "--accumulation_steps", "4",
                   "--learning_rate", "0.0005", "--num_workers", "2", "--seed", "42", "--log_interval", "20",
                   "--save_interval", "256", "--data_path", str(anchor / "data/train.jsonl")]
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            env.setdefault("OMP_NUM_THREADS", "1")
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(anchor), env.get("PYTHONPATH","")]))
            print(f"前半程仅专家 down 使用 MXFP8。microbatch {step} → 1024，日志 {anchor / 'prefix.log'}", flush=True)
            with (anchor / "prefix.log").open("a", encoding="utf-8") as f:
                print(" ".join(cmd), file=f, flush=True)
                with subprocess.Popen(cmd, cwd=anchor / "trainer", env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace") as p:
                    for line in p.stdout:
                        print(line, end="", flush=True)
                        f.write(line)
                        f.flush()
                    if p.wait():
                        raise RuntimeError(f"训练中断，可用同一命令续训。日志：{anchor / 'prefix.log'}")
            if check_anchor(anchor, torch) != 1024:
                raise ValueError("未生成符合预期的专家 down MXFP8 前半程断点")
        print(f"M_ANCHOR_READY {anchor / STATE}", flush=True)
    return anchor / STATE
