#!/usr/bin/env python3
"""Run the eight-V100 MiniMind pilot from the repository root.

Typical commands: check, smoke, prepare, train mxfp8, eval mxfp8.
Append --moe to train/eval for MoE. Use eval --eval-scheme same to also
measure the training format in the evaluation forward pass. moe-ffn-mxfp8
uses MXFP8 on all three expert projections by default. Add --groups grid
to compare the seven nonempty subsets across the three 4-bit attention formats.
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import timedelta
from pathlib import Path


def prepare(root: Path):
    """Prepare a reproducible, deduplicated pilot split from the pinned corpus."""
    count_train, count_valid = 32768, 1024
    capacity = count_train + count_valid
    revision = "f66c92ca277fe7a71a217a1d1e8d43a84684f934"
    corpus_hash = "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"
    filename = "pretrain_t2t_mini.jsonl"
    source = Path(root) / "dataset" / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        from huggingface_hub import hf_hub_download

        print(f"Downloading {filename} to {source.parent}", flush=True)
        hf_hub_download(repo_type="dataset", repo_id="jingyaogong/minimind_dataset",
                        filename=filename, revision=revision, local_dir=str(source.parent))

    # Use the SHA256 of NFC text with whitespace runs collapsed only for sampling
    # and deduplication. Keep the original whitespace in the emitted training text.
    kept = {}
    heap = []
    digest = hashlib.sha256()
    with source.open("rb") as corpus:
        for lineno, raw in enumerate(corpus, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            text = json.loads(raw).get("text")
            if not isinstance(text, str):
                raise ValueError(f"{source}, line {lineno}: expected a string field 'text'")
            normalized = " ".join(unicodedata.normalize("NFC", text).split())
            if not normalized:
                continue
            key = int.from_bytes(hashlib.sha256(normalized.encode("utf-8")).digest(), "big")
            if key in kept:
                kept[key] = min(kept[key], text)
            elif len(kept) < capacity:
                heapq.heappush(heap, -key)
                kept[key] = text
            elif key < -heap[0]:
                removed = -heapq.heapreplace(heap, -key)
                del kept[removed]
                kept[key] = text
    if digest.hexdigest() != corpus_hash:
        raise ValueError(f"{source} SHA256 is {digest.hexdigest()}, expected {corpus_hash}")
    if len(kept) < capacity:
        raise ValueError(f"Only {len(kept)} distinct nonempty texts, expected at least {capacity}")
    print(f"{source} sha256={corpus_hash}", flush=True)

    # Independent hash order assigns a uniform held-out subset of the selected rows.
    def split_order(key):
        return hashlib.sha256(b"minimind-v100-split-v1\x00" + key.to_bytes(32, "big")).digest(), key

    keys = sorted(kept, key=split_order)
    output = Path(root) / "runs" / "v100-pilot" / "data"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        for name, selected in (("valid", keys[:count_valid]), ("train", keys[count_valid:])):
            target = output / f"{name}.jsonl"
            partial = Path(folder) / target.name
            split_hash = hashlib.sha256()
            with partial.open("wb") as dest:
                for key in selected:
                    line = (json.dumps({"text": kept[key]}, ensure_ascii=False) + "\n").encode("utf-8")
                    dest.write(line)
                    split_hash.update(line)
            os.replace(partial, target)
            print(f"{target} rows={len(selected)} sha256={split_hash.hexdigest()}", flush=True)


def validate_case(moe, scheme, seed):
    from model.quant_policy import MIXED_SCHEMES, MODEL_SCHEMES

    if scheme not in MODEL_SCHEMES:
        raise ValueError(f"Unknown format {scheme!r}. Choose one of: {', '.join(MODEL_SCHEMES)}")
    if scheme in MIXED_SCHEMES and not moe:
        raise ValueError(f"{scheme} requires --moe")
    if not 0 <= seed <= 2**32 - 8:
        raise ValueError("--seed must leave room for all eight seed+rank values in uint32")


def evaluate(root, moe, scheme, eval_scheme, device="cuda:0", *, seed=42, reuse=False):
    """Score a pilot checkpoint on the reserved documents with token-weighted NLL."""
    validate_case(moe, scheme, seed)
    from model.quant_policy import (
        POLICY_SCHEMES, RESIDUAL_POLICY, RESIDUAL_SCHEMES, projection_policy, scheme_artifact_name,
    )

    if eval_scheme not in ("same", "bf16"):
        raise ValueError("Expected an evaluation format of bf16 or same.")
    forward_scheme = scheme if eval_scheme == "same" else "bf16"
    root = Path(root)
    pilot = root / "runs" / "v100-pilot"
    valid = pilot / "data" / "valid.jsonl"
    architecture = "moe" if moe else "dense"
    suffix = "_moe" if moe else ""
    scheme_name = scheme_artifact_name(scheme)
    weights = root / "out" / f"pilot32k_s{seed}_{scheme_name}_768{suffix}.pth"
    if not valid.is_file() or not weights.is_file():
        raise FileNotFoundError(valid if not valid.is_file() else weights)
    directory = pilot / "results"
    if seed != 42:
        directory = directory / f"seed{seed}"
    dest = directory / f"{architecture}__{scheme_name}__eval_{forward_scheme}.json"
    identity = {
        "architecture": architecture, "seed": seed, "train_scheme": scheme, "eval_scheme": forward_scheme,
        "checkpoint": weights.name, "valid_sha256": hashlib.sha256(valid.read_bytes()).hexdigest(),
        "seq_len": 512, "batch_size": 2,
    }
    if scheme in POLICY_SCHEMES:
        identity["train_projection_policy"] = projection_policy(scheme)
        identity["eval_projection_policy"] = projection_policy(forward_scheme)
    if scheme in RESIDUAL_SCHEMES:
        identity["residual_policy"] = RESIDUAL_POLICY
    if forward_scheme in RESIDUAL_SCHEMES:
        identity["eval_residual_policy"] = RESIDUAL_POLICY
    if reuse and dest.is_file() and dest.stat().st_mtime_ns > weights.stat().st_mtime_ns:
        try:
            previous = json.loads(dest.read_text(encoding="utf-8"))
            if (isinstance(previous, dict) and all(previous.get(k) == v for k, v in identity.items())
                    and previous["valid_tokens"] > 0 and math.isfinite(previous["mean_ce"])
                    and math.isfinite(previous["ppl"])):
                print(f"Reusing completed evaluation: {dest}", flush=True)
                return previous
        except (ValueError, KeyError, TypeError):
            pass

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from model.quant_linear import install_quantization

    torch.set_float32_matmul_precision("highest")
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(root / "model"), local_files_only=True)
    documents = PretrainDataset(str(valid), tokenizer, max_length=512)
    loader = DataLoader(documents, batch_size=2, shuffle=False, pin_memory=device.type == "cuda")
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=bool(moe)))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = install_quantization(model.float(), forward_scheme).eval().to(device)

    nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = 0
    with torch.inference_mode():
        for tokens, labels in loader:
            count = labels[:, 1:].ne(-100).sum().item()
            if count:
                tokens = tokens.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                out = model(tokens, labels=labels)
                # Native loss is shifted-token cross entropy. The router aux loss is separate.
                nll_sum.add_(out.loss.double() * count)
                token_count += count
                del out
    if not token_count:
        raise ValueError("The validation split contains no unmasked prediction targets.")
    mean_ce = nll_sum.item() / token_count
    if not math.isfinite(mean_ce):
        raise ValueError(f"Nonfinite validation CE: {mean_ce}")
    result = dict(identity, documents=len(documents), valid_tokens=token_count, mean_ce=mean_ce, ppl=math.exp(mean_ce))
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = tempfile.NamedTemporaryFile("w", dir=directory, encoding="utf-8", delete=False)
    try:
        with tmp:
            tmp.write(payload + "\n")
        os.replace(tmp.name, dest)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    print(payload)
    return result


def local_torchrun(script, *args):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=8", str(script), *map(str, args)]


def rank_check():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_nccl_available() or torch.cuda.get_device_capability(local_rank) != (7, 0):
        raise RuntimeError(f"Expected a V100 and NCCL on local rank {local_rank}")
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"Wheel lacks sm_70: {torch.cuda.get_arch_list()}")
    x = torch.ones((64, 64), device=f"cuda:{local_rank}", dtype=torch.float32)
    torch.testing.assert_close(x @ x, torch.full_like(x, 64), rtol=0, atol=0)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        value = torch.tensor([local_rank + 1.0], device=x.device)
        dist.all_reduce(value)
        if dist.get_world_size() != 8 or value.item() != 36:
            raise RuntimeError(f"Unexpected eight-rank NCCL result {value.item()}")
        if local_rank == 0:
            print(f"PASS: 8 CUDA FP32 GEMMs; NCCL sum(1..8)={value.item():.0f}", flush=True)
    finally:
        dist.destroy_process_group()


def check(root):
    import torch
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda} | archs {torch.cuda.get_arch_list()}", flush=True)
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected 8 visible V100 GPUs, found {torch.cuda.device_count()}")
    for i in range(8):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), flush=True)
    subprocess.run(local_torchrun(Path(__file__).resolve(), "--root", root, "_rank_check"), cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_v100*.py", "-v"],
                   cwd=root, check=True)


def train(root, moe, scheme, smoke=False, resume=False, *, seed=42, resume_if_present=False):
    from model.quant_policy import scheme_artifact_name, validate_resume_policy

    validate_case(moe, scheme, seed)
    label = "smoke_v100" if smoke else f"pilot32k_s{seed}"
    arch = "moe" if moe else "dense"
    run_dir = root / "runs" / "v100-pilot"
    data_path = run_dir / "data" / ("smoke.jsonl" if smoke else "train.jsonl")
    hidden = 128 if smoke else 768
    suffix = "_moe" if moe else ""
    scheme_name = scheme_artifact_name(scheme)
    basename = f"{label}_{scheme_name}_{hidden}{suffix}"
    weights = root / "out" / (basename + ".pth")
    state = root / "checkpoints" / (basename + "_resume.pth")
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path}. Run prepare first.")
    if resume_if_present:
        resume = state.is_file()
        if not resume and weights.exists():
            raise SystemExit(f"Found {weights} without {state}. Restore the matching resume checkpoint, or move "
                             "the orphaned export to a backup path before restarting this scheme from scratch.")
        if resume:
            import torch

            # Read the committed cursor without allocating all the optimizer tensors.
            checkpoint = torch.load(state, map_location="cpu", weights_only=True, mmap=True)
            validate_resume_policy(checkpoint, scheme, seed=seed)
            with data_path.open("rb") as rows:
                final_step = math.ceil(sum(bool(line.strip()) for line in rows) / (8 * 2))
            if checkpoint["world_size"] != 8 or checkpoint["epoch"] != 0 or not 0 < checkpoint["step"] <= final_step:
                raise ValueError(f"Unexpected 32k pilot progress in {state}")
            if checkpoint["step"] == final_step:
                if not weights.is_file():
                    raise FileNotFoundError(weights)
                print(f"Training already complete: {weights}", flush=True)
                return
            del checkpoint
    if resume and not state.is_file():
        raise SystemExit(f"Missing resume checkpoint {state}")
    if not smoke and not resume and not resume_if_present and (state.exists() or weights.exists()):
        raise SystemExit(f"{basename} already exists. Use eval for completed runs, or train --resume for an interrupted run.")
    cmd = local_torchrun(root / "trainer" / "train_pretrain.py",
                         "--from_weight", "none", "--from_resume", int(resume),
                         "--save_weight", label, "--use_moe", int(moe), "--quant_sim", scheme,
                         "--hidden_size", hidden, "--num_hidden_layers", 2 if smoke else 8,
                         "--max_seq_len", 64 if smoke else 512, "--epochs", 1, "--batch_size", 2,
                         "--accumulation_steps", 2 if smoke else 4, "--learning_rate", 5e-4,
                         "--num_workers", 0 if smoke else 2, "--seed", seed,
                         "--log_interval", 1 if smoke else 20, "--save_interval", 4 if smoke else 256,
                         "--data_path", data_path)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{label}_{arch}_{scheme_name}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.setdefault("OMP_NUM_THREADS", "1")
    with log_path.open("a", encoding="utf-8") as log:
        print(" ".join(cmd), file=log, flush=True)
        last_loss = None
        with subprocess.Popen(cmd, cwd=root / "trainer", env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                match = re.search(r"\bloss:\s*([^,\s]+)", line)
                if match:
                    last_loss = float(match.group(1))
            rc = proc.wait()
        if rc:
            raise SystemExit(f"Training failed with exit code {rc}. Log: {log_path}")
    if not state.is_file() or not weights.is_file():
        raise RuntimeError(f"Expected training outputs {state} and {weights}")
    if (last_loss is None and not resume) or (last_loss is not None and not math.isfinite(last_loss)):
        raise RuntimeError(f"Missing or nonfinite final training loss: {last_loss}. Log: {log_path}")
    print(f"PASS {arch} {scheme} | {weights} | {log_path}", flush=True)


def moe_ffn_mxfp8(root, seed, resume, *, groups=("all",), formats=None):
    from model.quant_policy import FFN_MXFP8_GROUPS, FOUR_BIT_SCHEMES, ffn_mxfp8_scheme

    requested = {group.strip() for arg in groups for group in arg.split(",")}
    unknown = requested.difference(FFN_MXFP8_GROUPS, ("grid",))
    if unknown:
        raise SystemExit(f"Unknown --groups: {', '.join(sorted(unknown))}. Choose from {', '.join(FFN_MXFP8_GROUPS)} or grid.")
    groups = [group for group in FFN_MXFP8_GROUPS if "grid" in requested or group in requested]
    formats = FOUR_BIT_SCHEMES if formats is None else tuple(dict.fromkeys(formats))
    unknown = set(formats).difference(FOUR_BIT_SCHEMES)
    if unknown:
        raise SystemExit(f"Unknown --formats: {', '.join(sorted(unknown))}. Choose from {', '.join(FOUR_BIT_SCHEMES)}.")
    schemes = [ffn_mxfp8_scheme(base, group) for group in groups for base in formats]
    if not (root / "runs/v100-pilot/data/valid.jsonl").is_file():
        raise SystemExit("Missing valid.jsonl. Prepare the fixed pilot split before training.")
    for index, scheme in enumerate(schemes, 1):
        print(f"[{index}/{len(schemes)}] {scheme} | seed={seed}", flush=True)
        train(root, True, scheme, seed=seed, resume_if_present=resume)
        for forward in ("bf16", "same"):
            # Let the evaluator release its CUDA context before the next eight-rank job.
            cmd = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "eval", scheme,
                   "--moe", "--seed", str(seed), "--eval-scheme", forward]
            if resume:
                cmd.append("--reuse")
            subprocess.run(cmd, cwd=root, check=True)


def smoke_test(root):
    data_dir = root / "runs" / "v100-pilot" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "smoke.jsonl").open("w", encoding="utf-8") as out:
        for i in range(64):
            print(json.dumps({"text": f"第 {i} 个训练样本用于验证八卡梯度同步。" * 4}, ensure_ascii=False), file=out)
    train(root, False, "bf16", smoke=True)
    train(root, True, "hif4_thr_3mm", smoke=True)


def default_root():
    script_dir = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), script_dir, script_dir.parent):
        if (candidate / "model" / "quant_linear.py").is_file():
            return candidate
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser(description="MiniMind-3 eight-V100 pilot with held-out perplexity evaluation.")
    parser.add_argument("--root", "--repo", type=Path, default=default_root(),
                        help="MiniMind repository root (default current or script's repository)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check eight GPUs, NCCL and repository tests")
    commands.add_parser("smoke", help="Run small eight-GPU Dense BF16 and MoE HiF4 training jobs")
    commands.add_parser("prepare", help="Fetch or verify the public corpus; build fixed train/valid splits")
    train_args = commands.add_parser("train", help="Pretrain from scratch on the fixed 32k subset, one eight-GPU job")
    eval_args = commands.add_parser("eval", help="Evaluate one pilot checkpoint on the held-out dataset")
    for p in (train_args, eval_args):
        p.add_argument("scheme", help="Training scheme (bf16, mxfp8, a 3MM format, a *_dw_mxfp8 variant or a MoE FFN scheme)")
        p.add_argument("--moe", action="store_true", help="Use MiniMind-3-MoE; default Dense")
    ablation = commands.add_parser("moe-ffn-mxfp8", help="Compare MXFP8 subsets of the three expert projections")
    ablation.add_argument("--groups", nargs="+", default=("all",), metavar="GROUP",
                          help="up, gate, down, up_gate, up_down, gate_down, all (default: all three projections). "
                               "Use grid for all seven subsets. Accepts spaces or commas.")
    ablation.add_argument("--formats", nargs="+", metavar="FORMAT",
                          help="4-bit base formats to train (default: hif4_thr_3mm nvfp4_3mm nvint4_3mm)")
    ablation.add_argument("--resume", action="store_true", help="Continue the suite, including previously unstarted cases")
    for p in (train_args, eval_args, ablation):
        p.add_argument("--seed", type=int, default=42, help="Training seed (default 42 for paired historical comparisons)")
    train_args.add_argument("--resume", action="store_true", help="Continue from an existing pilot optimizer checkpoint")
    eval_args.add_argument("--eval-scheme", choices=("bf16", "same"), default="bf16")
    eval_args.add_argument("--device", default="cuda:0")
    eval_args.add_argument("--reuse", action="store_true", help="Reuse a matching evaluation newer than the weights")
    commands.add_parser("_rank_check", help="Internal NCCL worker launched by check")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "model" / "quant_policy.py").is_file() or not (root / "tests" / "test_v100_batch_inference.py").is_file():
        raise SystemExit(f"Missing revised 3MM or eight-GPU changes in {root}. Set --root to the updated MiniMind repository.")
    sys.path.insert(0, str(root))
    if args.command == "_rank_check":
        rank_check()
    elif args.command == "check":
        check(root)
    elif args.command == "smoke":
        smoke_test(root)
    elif args.command == "prepare":
        prepare(root)
    elif args.command == "moe-ffn-mxfp8":
        moe_ffn_mxfp8(root, args.seed, args.resume, groups=args.groups, formats=args.formats)
    elif args.command == "train":
        train(root, args.moe, args.scheme, resume=args.resume, seed=args.seed)
    else:
        evaluate(root, args.moe, args.scheme, args.eval_scheme, args.device, seed=args.seed, reuse=args.reuse)


if __name__ == "__main__":
    main()
